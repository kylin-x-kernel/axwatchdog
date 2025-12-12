//! Per-CPU watchdog state management.
//!
//! This module provides lock-free, per-CPU data structures for watchdog state.
//! Each CPU has its own isolated state to avoid cross-core contention in NMI handlers.
//!
//! # Detection Mechanisms
//!
//! ## Softlockup Detection
//! A watchdog kernel thread runs on each CPU and updates `soft_timestamp` periodically.
//! If the timer interrupt detects that `soft_timestamp` hasn't been updated for too long,
//! it indicates the scheduler is stuck (tasks can't run, but interrupts work).
//!
//! ## Hardlockup Detection
//! Timer interrupts increment `hrtimer_interrupts`. The NMI handler checks if this
//! counter has increased since the last check. If not, the CPU is completely stuck
//! (even interrupts can't run).
//!
//! ## Deadlock Detection
//! Track lock acquisition times. If a lock is held for too long, it may indicate
//! a deadlock or resource starvation.
//!
//! # Memory Layout
//!
//! Each `PerCpuState` is aligned to 64 bytes (cache line) to prevent false sharing.

use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

use log::warn;

use crate::snapshot::CpuSnapshot;

/// Maximum number of CPUs supported.
pub const MAX_CPUS: usize = 64;

/// Default softlockup threshold in nanoseconds (20 seconds).
pub const DEFAULT_SOFTLOCKUP_THRESH_NS: u64 = 20_000_000_000;

/// Default hardlockup threshold in nanoseconds (10 seconds).
pub const DEFAULT_HARDLOCKUP_THRESH_NS: u64 = 10_000_000_000;

/// Default lock timeout in nanoseconds (60 seconds).
pub const DEFAULT_LOCK_TIMEOUT_NS: u64 = 60_000_000_000;

/// CPU health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CpuHealth {
    /// CPU is operating normally.
    Healthy          = 0,
    /// Softlockup: tasks not being scheduled, but interrupts work.
    SoftLockup       = 1,
    /// Hardlockup: CPU completely stuck, even interrupts don't run.
    HardLockup       = 2,
    /// Possible deadlock: lock held for too long.
    PossibleDeadlock = 3,
}

impl CpuHealth {
    /// Convert from raw u32 value.
    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => CpuHealth::Healthy,
            1 => CpuHealth::SoftLockup,
            2 => CpuHealth::HardLockup,
            3 => CpuHealth::PossibleDeadlock,
            _ => CpuHealth::Healthy,
        }
    }

    /// Check if this is a lockup condition.
    pub fn is_lockup(&self) -> bool {
        matches!(self, CpuHealth::SoftLockup | CpuHealth::HardLockup)
    }
}

/// Watchdog state for a single CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WatchdogState {
    /// Watchdog is idle, not monitoring.
    Idle      = 0,
    /// Watchdog is active and monitoring.
    Active    = 1,
    /// Health check detected a problem.
    Triggered = 2,
    /// Currently dumping CPU state.
    Dumping   = 3,
    /// Watchdog completed (terminal state until reset).
    Completed = 4,
}

impl WatchdogState {
    /// Convert from raw u32 value.
    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => WatchdogState::Idle,
            1 => WatchdogState::Active,
            2 => WatchdogState::Triggered,
            3 => WatchdogState::Dumping,
            4 => WatchdogState::Completed,
            _ => WatchdogState::Idle,
        }
    }
}

/// Per-CPU watchdog state.
///
/// This structure is cache-line aligned to prevent false sharing between CPUs.
/// All operations are lock-free using atomic primitives.
///
/// # Fields for Detection
///
/// - `soft_timestamp`: Updated by watchdog thread, checked by timer interrupt (softlockup)
/// - `hrtimer_interrupts`: Incremented by timer interrupt, checked by NMI (hardlockup)
/// - `locks_held` / `earliest_lock_time`: Track lock holding for deadlock detection
#[repr(C, align(64))]
pub struct PerCpuState {
    // === Softlockup Detection ===
    /// Timestamp when watchdog thread last ran (nanoseconds).
    /// Updated by watchdog thread, checked by timer interrupt.
    soft_timestamp: AtomicU64,

    // === Hardlockup Detection ===
    /// Timer interrupt counter (incremented in timer interrupt).
    hrtimer_interrupts: AtomicU32,
    /// Saved hrtimer_interrupts value from last NMI check.
    hrtimer_interrupts_saved: AtomicU32,
    /// Whether hardlockup warning has been issued.
    hardlockup_warned: AtomicU32,

    // === Deadlock Detection ===
    /// Number of locks currently held by this CPU.
    locks_held: AtomicU32,
    /// Timestamp when the earliest lock was acquired.
    earliest_lock_time: AtomicU64,

    // === State ===
    /// Number of consecutive failed checks.
    fail_count: AtomicU32,
    /// Current watchdog state.
    state: AtomicU32,
    /// Last detected health status.
    last_health: AtomicU32,
    /// Pointer to latest CPU snapshot (if any).
    snapshot: AtomicPtr<CpuSnapshot>,
    /// Timestamp of last heartbeat update (nanoseconds).
    last_update_ns: AtomicU64,
}

impl PerCpuState {
    /// Create a new per-CPU state.
    pub const fn new() -> Self {
        Self {
            soft_timestamp: AtomicU64::new(0),
            hrtimer_interrupts: AtomicU32::new(0),
            hrtimer_interrupts_saved: AtomicU32::new(0),
            hardlockup_warned: AtomicU32::new(0),
            locks_held: AtomicU32::new(0),
            earliest_lock_time: AtomicU64::new(0),
            fail_count: AtomicU32::new(0),
            state: AtomicU32::new(WatchdogState::Idle as u32),
            last_health: AtomicU32::new(CpuHealth::Healthy as u32),
            snapshot: AtomicPtr::new(core::ptr::null_mut()),
            last_update_ns: AtomicU64::new(0),
        }
    }

    // =========================================================================
    // Softlockup detection
    // =========================================================================

    /// Update the soft timestamp (called by watchdog thread).
    ///
    /// The watchdog thread should call this every time it gets scheduled.
    #[inline]
    pub fn touch_softlockup(&self, timestamp_ns: u64) {
        self.soft_timestamp.store(timestamp_ns, Ordering::Release);
    }

    /// Get the soft timestamp.
    #[inline]
    pub fn soft_timestamp(&self) -> u64 {
        self.soft_timestamp.load(Ordering::Acquire)
    }

    /// Check for softlockup condition.
    ///
    /// Call this from timer interrupt context.
    /// Returns true if softlockup is detected.
    #[inline]
    pub fn check_softlockup(&self, now_ns: u64, threshold_ns: u64) -> bool {
        let last = self.soft_timestamp.load(Ordering::Acquire);
        if last == 0 {
            // Not yet initialized
            return false;
        }
        // warn!("last: {}, now_ns: {}",last,now_ns);
        now_ns.saturating_sub(last) > threshold_ns
    }

    // =========================================================================
    // Hardlockup detection
    // =========================================================================

    /// Increment hrtimer interrupt counter (called from timer interrupt).
    #[inline]
    pub fn timer_tick(&self) {
        self.hrtimer_interrupts.fetch_add(1, Ordering::Release);
    }

    /// Get current hrtimer interrupt count.
    #[inline]
    pub fn hrtimer_interrupts(&self) -> u32 {
        self.hrtimer_interrupts.load(Ordering::Acquire)
    }

    /// Check for hardlockup condition (called from NMI).
    ///
    /// Returns true if hardlockup is detected (timer interrupts stopped).
    #[inline]
    pub fn check_hardlockup(&self) -> bool {
        let current = self.hrtimer_interrupts.load(Ordering::Acquire);
        let saved = self.hrtimer_interrupts_saved.load(Ordering::Acquire);

        // Update saved value for next check
        self.hrtimer_interrupts_saved
            .store(current, Ordering::Release);

        // If counts are equal, no timer interrupts occurred
        current == saved && current != 0
    }

    /// Check if hardlockup warning has been issued.
    #[inline]
    pub fn hardlockup_warned(&self) -> bool {
        self.hardlockup_warned.load(Ordering::Acquire) != 0
    }

    /// Set hardlockup warned flag.
    #[inline]
    pub fn set_hardlockup_warned(&self, warned: bool) {
        self.hardlockup_warned
            .store(warned as u32, Ordering::Release);
    }

    // =========================================================================
    // Deadlock detection (lock tracking)
    // =========================================================================

    /// Record a lock acquisition.
    ///
    /// Call this when acquiring a lock.
    #[inline]
    pub fn lock_acquired(&self, timestamp_ns: u64) {
        let count = self.locks_held.fetch_add(1, Ordering::AcqRel);
        if count == 0 {
            // First lock, record time
            self.earliest_lock_time
                .store(timestamp_ns, Ordering::Release);
        }
    }

    /// Record a lock release.
    ///
    /// Call this when releasing a lock.
    #[inline]
    pub fn lock_released(&self) {
        let count = self.locks_held.fetch_sub(1, Ordering::AcqRel);
        if count == 1 {
            // Last lock released, clear time
            self.earliest_lock_time.store(0, Ordering::Release);
        }
    }

    /// Get number of locks currently held.
    #[inline]
    pub fn locks_held(&self) -> u32 {
        self.locks_held.load(Ordering::Acquire)
    }

    /// Check for possible deadlock (lock held too long).
    ///
    /// Returns true if a lock has been held longer than threshold.
    #[inline]
    pub fn check_deadlock(&self, now_ns: u64, threshold_ns: u64) -> bool {
        let locks = self.locks_held.load(Ordering::Acquire);
        if locks == 0 {
            return false;
        }
        let lock_time = self.earliest_lock_time.load(Ordering::Acquire);
        if lock_time == 0 {
            return false;
        }
        now_ns.saturating_sub(lock_time) > threshold_ns
    }

    // =========================================================================
    // Comprehensive health check
    // =========================================================================

    /// Perform comprehensive health check.
    ///
    /// Call this from NMI handler.
    /// Returns the current health status of this CPU.
    pub fn check_health(&self, now_ns: u64, soft_thresh_ns: u64, lock_thresh_ns: u64) -> CpuHealth {
        // Priority: Hardlockup > Softlockup > Deadlock > Healthy

        // 1. Check hardlockup (most severe)
        if self.check_hardlockup() {
            self.last_health
                .store(CpuHealth::HardLockup as u32, Ordering::Release);
            return CpuHealth::HardLockup;
        }

        // 2. Check softlockup
        if self.check_softlockup(now_ns, soft_thresh_ns) {
            self.last_health
                .store(CpuHealth::SoftLockup as u32, Ordering::Release);
            return CpuHealth::SoftLockup;
        }

        // 3. Check deadlock
        if self.check_deadlock(now_ns, lock_thresh_ns) {
            self.last_health
                .store(CpuHealth::PossibleDeadlock as u32, Ordering::Release);
            return CpuHealth::PossibleDeadlock;
        }

        self.last_health
            .store(CpuHealth::Healthy as u32, Ordering::Release);
        CpuHealth::Healthy
    }

    /// Get last recorded health status.
    #[inline]
    pub fn last_health(&self) -> CpuHealth {
        CpuHealth::from_u32(self.last_health.load(Ordering::Acquire))
    }

    // =========================================================================
    // State management
    // =========================================================================

    /// Increment fail count and return new value.
    #[inline]
    pub fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Reset fail count to zero.
    #[inline]
    pub fn reset_fail_count(&self) {
        self.fail_count.store(0, Ordering::Release);
    }

    /// Get current fail count.
    #[inline]
    pub fn fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Acquire)
    }

    /// Get current state.
    #[inline]
    pub fn state(&self) -> WatchdogState {
        WatchdogState::from_u32(self.state.load(Ordering::Acquire))
    }

    /// Set state unconditionally.
    #[inline]
    pub fn set_state(&self, state: WatchdogState) {
        self.state.store(state as u32, Ordering::Release);
    }

    /// Try to transition from one state to another.
    ///
    /// Returns `true` if transition succeeded, `false` if current state didn't match.
    #[inline]
    pub fn try_transition(&self, from: WatchdogState, to: WatchdogState) -> bool {
        self.state
            .compare_exchange(from as u32, to as u32, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Store a snapshot pointer.
    ///
    /// # Safety
    /// The snapshot must remain valid for as long as it's stored here.
    #[inline]
    pub fn set_snapshot(&self, snapshot: *mut CpuSnapshot) {
        self.snapshot.store(snapshot, Ordering::Release);
    }

    /// Get the stored snapshot pointer.
    #[inline]
    pub fn snapshot(&self) -> *mut CpuSnapshot {
        self.snapshot.load(Ordering::Acquire)
    }

    /// Clear the snapshot pointer.
    #[inline]
    pub fn clear_snapshot(&self) {
        self.snapshot
            .store(core::ptr::null_mut(), Ordering::Release);
    }

    /// Get last update timestamp.
    #[inline]
    pub fn last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Acquire)
    }

    /// Reset all state to initial values.
    pub fn reset(&self) {
        self.soft_timestamp.store(0, Ordering::Release);
        self.hrtimer_interrupts.store(0, Ordering::Release);
        self.hrtimer_interrupts_saved.store(0, Ordering::Release);
        self.hardlockup_warned.store(0, Ordering::Release);
        self.locks_held.store(0, Ordering::Release);
        self.earliest_lock_time.store(0, Ordering::Release);
        self.fail_count.store(0, Ordering::Release);
        self.state
            .store(WatchdogState::Idle as u32, Ordering::Release);
        self.last_health
            .store(CpuHealth::Healthy as u32, Ordering::Release);
        self.snapshot
            .store(core::ptr::null_mut(), Ordering::Release);
        self.last_update_ns.store(0, Ordering::Release);
    }
}

impl Default for PerCpuState {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: PerCpuState uses only atomic operations
unsafe impl Send for PerCpuState {}
unsafe impl Sync for PerCpuState {}

/// Array of per-CPU states.
///
/// Access is indexed by CPU ID. Each CPU should only write to its own state,
/// though reading other CPUs' state is allowed (and lock-free).
pub struct PerCpuArray {
    states: [PerCpuState; MAX_CPUS],
}

impl PerCpuArray {
    /// Create a new per-CPU array.
    pub const fn new() -> Self {
        const INIT: PerCpuState = PerCpuState::new();
        Self {
            states: [INIT; MAX_CPUS],
        }
    }

    /// Get the state for a specific CPU.
    ///
    /// # Panics
    /// Panics if `cpu_id >= MAX_CPUS`.
    #[inline]
    pub fn get(&self, cpu_id: usize) -> &PerCpuState {
        &self.states[cpu_id]
    }

    /// Get the state for a specific CPU (checked).
    #[inline]
    pub fn try_get(&self, cpu_id: usize) -> Option<&PerCpuState> {
        self.states.get(cpu_id)
    }

    /// Touch softlockup timestamp for a CPU (watchdog thread).
    #[inline]
    pub fn touch_softlockup(&self, cpu_id: usize, timestamp_ns: u64) {
        if let Some(state) = self.try_get(cpu_id) {
            state.touch_softlockup(timestamp_ns);
        }
    }

    /// Timer tick for a CPU (called from timer interrupt).
    #[inline]
    pub fn timer_tick(&self, cpu_id: usize) {
        if let Some(state) = self.try_get(cpu_id) {
            state.timer_tick();
        }
    }

    /// Check softlockup of a specific CPU.
    pub fn check_softlockup(&self, cpu_id: usize, now_ns: u64, soft_thresh_ns: u64) -> CpuHealth {
        if let Some(state) = self.try_get(cpu_id) {
            if state.check_softlockup(now_ns, soft_thresh_ns) {
                return CpuHealth::SoftLockup;
            }
        }
        CpuHealth::Healthy
    }

    /// Check health of a specific CPU.
    pub fn check_cpu_health(
        &self,
        cpu_id: usize,
        now_ns: u64,
        soft_thresh_ns: u64,
        lock_thresh_ns: u64,
    ) -> CpuHealth {
        self.try_get(cpu_id)
            .map(|s| s.check_health(now_ns, soft_thresh_ns, lock_thresh_ns))
            .unwrap_or(CpuHealth::Healthy)
    }

    /// Reset all CPU states.
    pub fn reset_all(&self) {
        for state in &self.states {
            state.reset();
        }
    }

    /// Get iterator over all states.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &PerCpuState)> {
        self.states.iter().enumerate()
    }
}

impl Default for PerCpuArray {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: PerCpuArray uses only atomic operations internally
unsafe impl Send for PerCpuArray {}
unsafe impl Sync for PerCpuArray {}

/// Global per-CPU state array.
pub static PERCPU_STATE: PerCpuArray = PerCpuArray::new();

/// Touch softlockup timestamp (called from watchdog thread).
#[inline]
pub fn touch_softlockup(cpu_id: usize, timestamp_ns: u64) {
    // warn!("cpu_id: {}, timestamp: {}", cpu_id, timestamp_ns);
    PERCPU_STATE.touch_softlockup(cpu_id, timestamp_ns);
}

/// Timer tick (called from timer interrupt).
#[inline]
pub fn timer_tick(cpu_id: usize) {
    PERCPU_STATE.timer_tick(cpu_id);
}

/// Record lock acquisition.
#[inline]
pub fn lock_acquired(cpu_id: usize, timestamp_ns: u64) {
    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        state.lock_acquired(timestamp_ns);
    }
}

/// Record lock release.
#[inline]
pub fn lock_released(cpu_id: usize) {
    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        state.lock_released();
    }
}

/// Check comprehensive health of a CPU.
#[inline]
pub fn check_softlockup(cpu_id: usize, now_ns: u64) -> CpuHealth {
    PERCPU_STATE.check_softlockup(cpu_id, now_ns, DEFAULT_SOFTLOCKUP_THRESH_NS)
}

/// Check comprehensive health of a CPU.
#[inline]
pub fn check_cpu_health(cpu_id: usize, now_ns: u64) -> CpuHealth {
    PERCPU_STATE.check_cpu_health(
        cpu_id,
        now_ns,
        DEFAULT_SOFTLOCKUP_THRESH_NS,
        DEFAULT_LOCK_TIMEOUT_NS,
    )
}
