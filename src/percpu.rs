//! Per-CPU watchdog state management.
//!
//! This module provides lock-free, per-CPU data structures for watchdog state.
//! Each CPU has its own isolated state to avoid cross-core contention in NMI handlers.
//!
//! # Memory Layout
//!
//! Each `PerCpuState` is aligned to 64 bytes (cache line) to prevent false sharing.

use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

use crate::snapshot::CpuSnapshot;

/// Maximum number of CPUs supported.
pub const MAX_CPUS: usize = 64;

/// Watchdog state for a single CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WatchdogState {
    /// Watchdog is idle, not monitoring.
    Idle = 0,
    /// Watchdog is active and monitoring.
    Active = 1,
    /// Health check detected a problem.
    Triggered = 2,
    /// Currently dumping CPU state.
    Dumping = 3,
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
#[repr(C, align(64))]
pub struct PerCpuState {
    /// Heartbeat counter - tasks increment this to show they're alive.
    heartbeat: AtomicU64,
    /// Last heartbeat value seen during NMI check.
    last_checked_heartbeat: AtomicU64,
    /// Number of consecutive failed checks.
    fail_count: AtomicU32,
    /// Current watchdog state.
    state: AtomicU32,
    /// Pointer to latest CPU snapshot (if any).
    snapshot: AtomicPtr<CpuSnapshot>,
    /// Timestamp of last heartbeat update (nanoseconds).
    last_update_ns: AtomicU64,
    /// Padding to ensure 64-byte alignment.
    _padding: [u8; 16],
}

impl PerCpuState {
    /// Create a new per-CPU state.
    pub const fn new() -> Self {
        Self {
            heartbeat: AtomicU64::new(0),
            last_checked_heartbeat: AtomicU64::new(0),
            fail_count: AtomicU32::new(0),
            state: AtomicU32::new(WatchdogState::Idle as u32),
            snapshot: AtomicPtr::new(core::ptr::null_mut()),
            last_update_ns: AtomicU64::new(0),
            _padding: [0; 16],
        }
    }

    /// Increment the heartbeat counter (called by monitored tasks).
    ///
    /// This is the "pet the watchdog" operation.
    #[inline]
    pub fn pet(&self) {
        self.heartbeat.fetch_add(1, Ordering::Release);
    }

    /// Increment heartbeat with timestamp.
    #[inline]
    pub fn pet_with_timestamp(&self, timestamp_ns: u64) {
        self.heartbeat.fetch_add(1, Ordering::Release);
        self.last_update_ns.store(timestamp_ns, Ordering::Release);
    }

    /// Get the current heartbeat value.
    #[inline]
    pub fn heartbeat(&self) -> u64 {
        self.heartbeat.load(Ordering::Acquire)
    }

    /// Check if heartbeat has increased since last check.
    ///
    /// Returns `true` if healthy (heartbeat increased), `false` if stuck.
    /// This also updates the last_checked_heartbeat.
    #[inline]
    pub fn check_heartbeat(&self) -> bool {
        let current = self.heartbeat.load(Ordering::Acquire);
        let last = self.last_checked_heartbeat.swap(current, Ordering::AcqRel);
        current != last
    }

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
        self.heartbeat.store(0, Ordering::Release);
        self.last_checked_heartbeat.store(0, Ordering::Release);
        self.fail_count.store(0, Ordering::Release);
        self.state
            .store(WatchdogState::Idle as u32, Ordering::Release);
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

    /// Pet the watchdog for a specific CPU.
    #[inline]
    pub fn pet(&self, cpu_id: usize) {
        if let Some(state) = self.try_get(cpu_id) {
            state.pet();
        }
    }

    /// Pet the watchdog with timestamp.
    #[inline]
    pub fn pet_with_timestamp(&self, cpu_id: usize, timestamp_ns: u64) {
        if let Some(state) = self.try_get(cpu_id) {
            state.pet_with_timestamp(timestamp_ns);
        }
    }

    /// Check heartbeat for a specific CPU.
    #[inline]
    pub fn check_heartbeat(&self, cpu_id: usize) -> bool {
        self.try_get(cpu_id)
            .map(|s| s.check_heartbeat())
            .unwrap_or(true) // Non-existent CPUs are "healthy"
    }

    /// Check all CPUs and return a bitmap of unhealthy ones.
    ///
    /// Bit N is set if CPU N failed the heartbeat check.
    pub fn check_all(&self, num_cpus: usize) -> u64 {
        let mut unhealthy: u64 = 0;
        for i in 0..num_cpus.min(64) {
            if !self.check_heartbeat(i) {
                unhealthy |= 1 << i;
            }
        }
        unhealthy
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

/// Convenience function to pet the watchdog for the current CPU.
///
/// # Arguments
/// * `cpu_id` - Current CPU ID (caller must provide this)
#[inline]
pub fn pet(cpu_id: usize) {
    PERCPU_STATE.pet(cpu_id);
}

/// Convenience function to pet with timestamp.
#[inline]
pub fn pet_with_timestamp(cpu_id: usize, timestamp_ns: u64) {
    PERCPU_STATE.pet_with_timestamp(cpu_id, timestamp_ns);
}

/// Check if a CPU's heartbeat is healthy.
#[inline]
pub fn check_cpu(cpu_id: usize) -> bool {
    PERCPU_STATE.check_heartbeat(cpu_id)
}
