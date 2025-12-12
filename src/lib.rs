//! AArch64 Watchdog Timer with NMI Support.
//!
//! This crate provides a watchdog implementation that uses NMI (Non-Maskable Interrupt)
//! sources like SDEI or PMU overflow to ensure the watchdog can trigger even when
//! normal interrupts are disabled.
//!
//! # Detection Mechanisms
//!
//! ## Softlockup Detection
//! Detects when a CPU can still handle interrupts but tasks cannot be scheduled.
//! A high-priority watchdog thread updates a timestamp; if this timestamp is stale,
//! the scheduler is stuck.
//!
//! ## Hardlockup Detection  
//! Detects when a CPU is completely stuck (even interrupts don't run).
//! Timer interrupts increment a counter; NMI checks if this counter has increased.
//! If not, the CPU is completely unresponsive.
//!
//! ## Deadlock Detection
//! Tracks lock acquisition times. If a lock is held longer than a threshold,
//! it may indicate a deadlock.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Detection Layers                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │   Watchdog Thread (each CPU)                                    │
//! │        │                                                        │
//! │        └─── Updates soft_timestamp periodically                 │
//! │                                                                 │
//! │   Timer Interrupt                                               │
//! │        │                                                        │
//! │        ├─── Increments hrtimer_interrupts                       │
//! │        └─── Checks soft_timestamp (Softlockup)                  │
//! │                                                                 │
//! │   NMI (PMU/SDEI)                                               │
//! │        │                                                        │
//! │        └─── Checks hrtimer_interrupts (Hardlockup)              │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Lock-free Design
//!
//! All NMI handlers are completely lock-free. They only use:
//! - Atomic operations
//! - Per-CPU data structures
//! - Direct memory access
//!
//! # Example
//!
//! ```no_run
//! use axwatchdog::{Watchdog, WatchdogConfig, WatchdogTask};
//!
//! // Define a task to monitor
//! struct MyTask;
//!
//! impl WatchdogTask for MyTask {
//!     fn name(&self) -> &'static str {
//!         "my_task"
//!     }
//!
//!     fn health_check(&self) -> bool {
//!         true
//!     }
//! }
//!
//! // Initialize watchdog
//! let config = WatchdogConfig::default();
//! Watchdog::init(config).expect("watchdog init failed");
//!
//! // Register task
//! static MY_TASK: MyTask = MyTask;
//! let handle = axwatchdog::register_task(&MY_TASK).unwrap();
//!
//! // Start watchdog
//! Watchdog::start().expect("watchdog start failed");
//!
//! // In your task loop, pet the watchdog
//! axwatchdog::pet(0); // 0 = current CPU ID
//! ```

#![no_std]

pub mod lockfree;
pub mod nmi;
pub mod percpu;
pub mod snapshot;
pub mod task;

// Re-exports
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

pub use lockfree::{AtomicBitmap, AtomicSequence, SpscRingBuffer};
#[cfg(feature = "pmu")]
pub use nmi::PmuNmiSource;
#[cfg(feature = "sdei")]
pub use nmi::SdeiNmi;
pub use nmi::{NmiContext, NmiError, NmiHandler, NmiResult, NmiSource};
pub use percpu::{
    CpuHealth, DEFAULT_HARDLOCKUP_THRESH_NS, DEFAULT_LOCK_TIMEOUT_NS, DEFAULT_SOFTLOCKUP_THRESH_NS,
    PERCPU_STATE, PerCpuArray, PerCpuState, WatchdogState, check_cpu_health, check_softlockup,
    lock_acquired, lock_released, timer_tick, touch_softlockup,
};
pub use snapshot::{CpuSnapshot, LockupEvent, LockupEventBuffer, LockupType, SnapshotCollector};
pub use task::{
    HeartbeatTask, RegistryError, TASK_REGISTRY, TaskHandle, TaskRegistry, WatchdogTask,
    check_all_tasks, register_task, unregister_task,
};

/// Watchdog configuration.
#[derive(Debug, Clone, Copy)]
pub struct WatchdogConfig {
    /// Hardlockup timeout in nanoseconds (NMI checks timer interrupts).
    /// Default: 10 seconds.
    pub hardlockup_thresh_ns: u64,
    /// Softlockup timeout in nanoseconds (timer checks watchdog thread).
    /// Default: 20 seconds.
    pub softlockup_thresh_ns: u64,
    /// Lock timeout for deadlock detection in nanoseconds.
    /// Default: 60 seconds.
    pub lock_timeout_ns: u64,
    /// Number of consecutive failures before triggering.
    pub fail_threshold: u32,
    /// Number of CPUs to monitor.
    pub num_cpus: usize,
    /// Whether to automatically restart after handling.
    pub auto_restart: bool,
    /// Whether to panic on hardlockup.
    pub panic_on_hardlockup: bool,
    /// Whether to panic on softlockup.
    pub panic_on_softlockup: bool,
}

impl WatchdogConfig {
    /// Create a new configuration with default values.
    pub const fn new() -> Self {
        Self {
            hardlockup_thresh_ns: DEFAULT_HARDLOCKUP_THRESH_NS,
            softlockup_thresh_ns: DEFAULT_SOFTLOCKUP_THRESH_NS,
            lock_timeout_ns: DEFAULT_LOCK_TIMEOUT_NS,
            fail_threshold: 3,
            num_cpus: 1,
            auto_restart: true,
            panic_on_hardlockup: true,
            panic_on_softlockup: false,
        }
    }

    /// Set the hardlockup timeout in nanoseconds.
    pub const fn with_hardlockup_thresh_ns(mut self, ns: u64) -> Self {
        self.hardlockup_thresh_ns = ns;
        self
    }

    /// Set the hardlockup timeout in seconds.
    pub const fn with_hardlockup_thresh_secs(mut self, secs: u64) -> Self {
        self.hardlockup_thresh_ns = secs * 1_000_000_000;
        self
    }

    /// Set the softlockup timeout in nanoseconds.
    pub const fn with_softlockup_thresh_ns(mut self, ns: u64) -> Self {
        self.softlockup_thresh_ns = ns;
        self
    }

    /// Set the softlockup timeout in seconds.
    pub const fn with_softlockup_thresh_secs(mut self, secs: u64) -> Self {
        self.softlockup_thresh_ns = secs * 1_000_000_000;
        self
    }

    /// Set the lock timeout in nanoseconds.
    pub const fn with_lock_timeout_ns(mut self, ns: u64) -> Self {
        self.lock_timeout_ns = ns;
        self
    }

    /// Set the failure threshold.
    pub const fn with_fail_threshold(mut self, threshold: u32) -> Self {
        self.fail_threshold = threshold;
        self
    }

    /// Set the number of CPUs.
    pub const fn with_num_cpus(mut self, num_cpus: usize) -> Self {
        self.num_cpus = num_cpus;
        self
    }

    /// Set auto-restart behavior.
    pub const fn with_auto_restart(mut self, auto_restart: bool) -> Self {
        self.auto_restart = auto_restart;
        self
    }

    /// Set panic on hardlockup.
    pub const fn with_panic_on_hardlockup(mut self, panic: bool) -> Self {
        self.panic_on_hardlockup = panic;
        self
    }

    /// Set panic on softlockup.
    pub const fn with_panic_on_softlockup(mut self, panic: bool) -> Self {
        self.panic_on_softlockup = panic;
        self
    }
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Watchdog error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchdogError {
    /// Watchdog not initialized.
    NotInitialized,
    /// Watchdog already initialized.
    AlreadyInitialized,
    /// Watchdog already running.
    AlreadyRunning,
    /// Watchdog not running.
    NotRunning,
    /// NMI source error.
    NmiError(NmiError),
    /// Task registry error.
    TaskError(RegistryError),
    /// Invalid configuration.
    InvalidConfig,
}

impl From<NmiError> for WatchdogError {
    fn from(e: NmiError) -> Self {
        WatchdogError::NmiError(e)
    }
}

impl From<RegistryError> for WatchdogError {
    fn from(e: RegistryError) -> Self {
        WatchdogError::TaskError(e)
    }
}

impl core::fmt::Display for WatchdogError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WatchdogError::NotInitialized => write!(f, "Watchdog not initialized"),
            WatchdogError::AlreadyInitialized => write!(f, "Watchdog already initialized"),
            WatchdogError::AlreadyRunning => write!(f, "Watchdog already running"),
            WatchdogError::NotRunning => write!(f, "Watchdog not running"),
            WatchdogError::NmiError(e) => write!(f, "NMI error: {}", e),
            WatchdogError::TaskError(e) => write!(f, "Task error: {}", e),
            WatchdogError::InvalidConfig => write!(f, "Invalid configuration"),
        }
    }
}

/// Result type for watchdog operations.
pub type WatchdogResult<T> = Result<T, WatchdogError>;

/// Global watchdog state.
pub struct Watchdog {
    /// Whether initialized.
    initialized: AtomicBool,
    /// Whether running.
    running: AtomicBool,
    /// Configuration - hardlockup threshold.
    hardlockup_thresh_ns: AtomicU64,
    /// Configuration - softlockup threshold.
    softlockup_thresh_ns: AtomicU64,
    /// Configuration - lock timeout.
    lock_timeout_ns: AtomicU64,
    /// Configuration - fail threshold.
    fail_threshold: AtomicU32,
    /// Configuration - number of CPUs.
    num_cpus: AtomicU32,
    /// Configuration - auto restart.
    auto_restart: AtomicBool,
    /// Configuration - panic on hardlockup.
    panic_on_hardlockup: AtomicBool,
    /// Configuration - panic on softlockup.
    panic_on_softlockup: AtomicBool,
    /// Statistics - total checks.
    total_checks: AtomicU64,
    /// Statistics - total failures.
    total_failures: AtomicU64,
    /// Statistics - hardlockup count.
    hardlockup_count: AtomicU64,
    /// Statistics - softlockup count.
    softlockup_count: AtomicU64,
}

impl Watchdog {
    /// Create a new watchdog instance.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            running: AtomicBool::new(false),
            hardlockup_thresh_ns: AtomicU64::new(DEFAULT_HARDLOCKUP_THRESH_NS),
            softlockup_thresh_ns: AtomicU64::new(DEFAULT_SOFTLOCKUP_THRESH_NS),
            lock_timeout_ns: AtomicU64::new(DEFAULT_LOCK_TIMEOUT_NS),
            fail_threshold: AtomicU32::new(3),
            num_cpus: AtomicU32::new(4),
            auto_restart: AtomicBool::new(true),
            panic_on_hardlockup: AtomicBool::new(true),
            panic_on_softlockup: AtomicBool::new(false),
            total_checks: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            hardlockup_count: AtomicU64::new(0),
            softlockup_count: AtomicU64::new(0),
        }
    }

    /// Initialize the watchdog with the given configuration.
    pub fn init(config: WatchdogConfig) -> WatchdogResult<()> {
        if WATCHDOG.initialized.swap(true, Ordering::AcqRel) {
            return Err(WatchdogError::AlreadyInitialized);
        }

        WATCHDOG
            .hardlockup_thresh_ns
            .store(config.hardlockup_thresh_ns, Ordering::Release);
        WATCHDOG
            .softlockup_thresh_ns
            .store(config.softlockup_thresh_ns, Ordering::Release);
        WATCHDOG
            .lock_timeout_ns
            .store(config.lock_timeout_ns, Ordering::Release);
        WATCHDOG
            .fail_threshold
            .store(config.fail_threshold, Ordering::Release);
        WATCHDOG
            .num_cpus
            .store(config.num_cpus as u32, Ordering::Release);
        WATCHDOG
            .auto_restart
            .store(config.auto_restart, Ordering::Release);
        WATCHDOG
            .panic_on_hardlockup
            .store(config.panic_on_hardlockup, Ordering::Release);
        WATCHDOG
            .panic_on_softlockup
            .store(config.panic_on_softlockup, Ordering::Release);

        #[cfg(feature = "log")]
        log::info!(
            "Watchdog initialized: hardlockup={}s, softlockup={}s, lock_timeout={}s, cpus={}",
            config.hardlockup_thresh_ns / 1_000_000_000,
            config.softlockup_thresh_ns / 1_000_000_000,
            config.lock_timeout_ns / 1_000_000_000,
            config.num_cpus
        );

        Ok(())
    }

    /// Start the watchdog.
    pub fn start() -> WatchdogResult<()> {
        if !WATCHDOG.initialized.load(Ordering::Acquire) {
            return Err(WatchdogError::NotInitialized);
        }

        if WATCHDOG.running.swap(true, Ordering::AcqRel) {
            return Err(WatchdogError::AlreadyRunning);
        }

        // Activate all per-CPU states
        let num_cpus = WATCHDOG.num_cpus.load(Ordering::Acquire) as usize;
        for i in 0..num_cpus {
            if let Some(state) = PERCPU_STATE.try_get(i) {
                state.set_state(WatchdogState::Active);
            }
        }

        #[cfg(feature = "log")]
        log::info!("Watchdog started");

        Ok(())
    }

    /// Stop the watchdog.
    pub fn stop() -> WatchdogResult<()> {
        if !WATCHDOG.running.swap(false, Ordering::AcqRel) {
            return Err(WatchdogError::NotRunning);
        }

        // Set all per-CPU states to idle
        let num_cpus = WATCHDOG.num_cpus.load(Ordering::Acquire) as usize;
        for i in 0..num_cpus {
            if let Some(state) = PERCPU_STATE.try_get(i) {
                state.set_state(WatchdogState::Idle);
            }
        }

        #[cfg(feature = "log")]
        log::info!("Watchdog stopped");

        Ok(())
    }

    /// Check if watchdog is initialized.
    pub fn is_initialized() -> bool {
        WATCHDOG.initialized.load(Ordering::Acquire)
    }

    /// Check if watchdog is running.
    pub fn is_running() -> bool {
        WATCHDOG.running.load(Ordering::Acquire)
    }

    /// Get hardlockup threshold in nanoseconds.
    pub fn hardlockup_thresh_ns() -> u64 {
        WATCHDOG.hardlockup_thresh_ns.load(Ordering::Acquire)
    }

    /// Get softlockup threshold in nanoseconds.
    pub fn softlockup_thresh_ns() -> u64 {
        WATCHDOG.softlockup_thresh_ns.load(Ordering::Acquire)
    }

    /// Get lock timeout in nanoseconds.
    pub fn lock_timeout_ns() -> u64 {
        WATCHDOG.lock_timeout_ns.load(Ordering::Acquire)
    }

    /// Get total check count.
    pub fn total_checks() -> u64 {
        WATCHDOG.total_checks.load(Ordering::Acquire)
    }

    /// Get total failure count.
    pub fn total_failures() -> u64 {
        WATCHDOG.total_failures.load(Ordering::Acquire)
    }

    /// Get hardlockup count.
    pub fn hardlockup_count() -> u64 {
        WATCHDOG.hardlockup_count.load(Ordering::Acquire)
    }

    /// Get softlockup count.
    pub fn softlockup_count() -> u64 {
        WATCHDOG.softlockup_count.load(Ordering::Acquire)
    }
}

impl Default for Watchdog {
    fn default() -> Self {
        Self::new()
    }
}

/// Global watchdog instance.
pub static WATCHDOG: Watchdog = Watchdog::new();

// =============================================================================
// Detection Entry Points
// =============================================================================

/// Timer interrupt handler for watchdog.
///
/// Call this from your timer interrupt handler on each CPU.
/// This handles:
/// 1. Incrementing the hrtimer_interrupts counter (for hardlockup detection)
/// 2. Checking for softlockup (watchdog thread not running)
///
/// # Arguments
/// * `cpu_id` - Current CPU ID
/// * `now_ns` - Current timestamp in nanoseconds
pub fn watchdog_timer_handler(cpu_id: usize, now_ns: u64) {
    if !WATCHDOG.running.load(Ordering::Acquire) {
        return;
    }

    // 1. Increment timer interrupt counter (for hardlockup detection by NMI)
    timer_tick(cpu_id);

    // 2. Check for softlockup
    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        let soft_thresh = WATCHDOG.softlockup_thresh_ns.load(Ordering::Acquire);

        if state.check_softlockup(now_ns, soft_thresh) {
            handle_softlockup(cpu_id, now_ns);
        }
    }
}

/// NMI handler for watchdog (hardlockup detection).
///
/// This is called from the NMI source (PMU overflow or SDEI) when triggered.
/// It performs the critical hardlockup check: if timer interrupts have stopped,
/// the CPU is completely stuck.
///
/// # Arguments
/// * `cpu_id` - CPU ID that received the NMI
/// * `ctx` - NMI context with interrupted state
pub fn watchdog_nmi_handler(cpu_id: usize, ctx: &NmiContext) {
    if !WATCHDOG.running.load(Ordering::Acquire) {
        return;
    }

    // Increment check counter
    WATCHDOG.total_checks.fetch_add(1, Ordering::Relaxed);

    let now_ns = ctx.timestamp_ns;
    let soft_thresh = WATCHDOG.softlockup_thresh_ns.load(Ordering::Acquire);
    let lock_thresh = WATCHDOG.lock_timeout_ns.load(Ordering::Acquire);

    // Perform comprehensive health check
    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        let health = state.check_health(now_ns, soft_thresh, lock_thresh);

        match health {
            CpuHealth::HardLockup => {
                handle_hardlockup(cpu_id, ctx);
            }
            CpuHealth::SoftLockup => {
                // Softlockup in NMI - this is informational, timer handler is primary
                handle_softlockup(cpu_id, now_ns);
            }
            CpuHealth::PossibleDeadlock => {
                handle_possible_deadlock(cpu_id, now_ns);
            }
            CpuHealth::Healthy => {
                // Reset warnings
                state.set_hardlockup_warned(false);
                state.reset_fail_count();
            }
        }
    }

    // Also check registered tasks
    if !TASK_REGISTRY.check_all() {
        WATCHDOG.total_failures.fetch_add(1, Ordering::Relaxed);
        handle_task_timeout(cpu_id, ctx);
    }
}

/// Watchdog thread entry point.
///
/// This should be called periodically from a high-priority kernel thread.
/// It updates the soft_timestamp to indicate the thread is running.
///
/// # Arguments
/// * `cpu_id` - Current CPU ID
/// * `now_ns` - Current timestamp in nanoseconds
#[inline]
pub fn watchdog_thread_tick(cpu_id: usize, now_ns: u64) {
    touch_softlockup(cpu_id, now_ns);
}

// =============================================================================
// Internal Handlers
// =============================================================================

/// Handle hardlockup detection.
fn handle_hardlockup(cpu_id: usize, ctx: &NmiContext) {
    WATCHDOG.hardlockup_count.fetch_add(1, Ordering::Relaxed);
    WATCHDOG.total_failures.fetch_add(1, Ordering::Relaxed);

    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        // Only warn once per lockup
        if state.hardlockup_warned() {
            return;
        }
        state.set_hardlockup_warned(true);

        // Transition state
        let _ = state.try_transition(WatchdogState::Active, WatchdogState::Triggered);
    }

    #[cfg(feature = "log")]
    log::error!(
        "HARDLOCKUP detected on CPU {}: PC={:#x}, SP={:#x}, LR={:#x}",
        cpu_id,
        ctx.pc,
        ctx.sp,
        ctx.lr
    );

    #[cfg(feature = "log")]
    log::error!(
        "CPU {} stuck: timer interrupts stopped, CPU is completely unresponsive",
        cpu_id
    );

    // Check if we should panic
    if WATCHDOG.panic_on_hardlockup.load(Ordering::Acquire) {
        // In a real implementation, this would call panic!
        // For now, just log
        #[cfg(feature = "log")]
        log::error!("PANIC: Hard LOCKUP on CPU {}", cpu_id);
    }
}

/// Handle softlockup detection.
fn handle_softlockup(cpu_id: usize, now_ns: u64) {
    WATCHDOG.softlockup_count.fetch_add(1, Ordering::Relaxed);
    WATCHDOG.total_failures.fetch_add(1, Ordering::Relaxed);

    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        let fail_count = state.increment_fail_count();
        let threshold = WATCHDOG.fail_threshold.load(Ordering::Acquire);

        if fail_count < threshold {
            return; // Not enough consecutive failures
        }

        // Transition state
        let _ = state.try_transition(WatchdogState::Active, WatchdogState::Triggered);

        let soft_ts = state.soft_timestamp();
        let elapsed_ms = now_ns.saturating_sub(soft_ts) / 1_000_000;

        #[cfg(feature = "log")]
        log::error!(
            "SOFTLOCKUP detected on CPU {}: watchdog thread not scheduled for {}ms",
            cpu_id,
            elapsed_ms
        );

        #[cfg(feature = "log")]
        log::error!(
            "CPU {} stuck: interrupts work but tasks cannot be scheduled",
            cpu_id
        );

        // Check if we should panic
        if WATCHDOG.panic_on_softlockup.load(Ordering::Acquire) {
            #[cfg(feature = "log")]
            log::error!("PANIC: Soft LOCKUP on CPU {}", cpu_id);
        }
    }
}

/// Handle possible deadlock detection.
fn handle_possible_deadlock(cpu_id: usize, now_ns: u64) {
    WATCHDOG.total_failures.fetch_add(1, Ordering::Relaxed);

    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        let lock_time = state.soft_timestamp(); // Using soft_timestamp as proxy
        let elapsed_ms = now_ns.saturating_sub(lock_time) / 1_000_000;
        let locks_held = state.locks_held();

        #[cfg(feature = "log")]
        log::warn!(
            "Possible DEADLOCK on CPU {}: {} locks held for {}ms",
            cpu_id,
            locks_held,
            elapsed_ms
        );
    }
}

/// Handle task timeout.
fn handle_task_timeout(cpu_id: usize, ctx: &NmiContext) {
    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        let fail_count = state.increment_fail_count();
        let threshold = WATCHDOG.fail_threshold.load(Ordering::Acquire);

        if fail_count >= threshold {
            if state.try_transition(WatchdogState::Active, WatchdogState::Triggered) {
                #[cfg(feature = "log")]
                log::error!(
                    "Task watchdog timeout on CPU {}: PC={:#x}, SP={:#x}",
                    cpu_id,
                    ctx.pc,
                    ctx.sp
                );

                // Notify tasks
                TASK_REGISTRY.notify_timeout(cpu_id);

                // Auto-restart if configured
                if WATCHDOG.auto_restart.load(Ordering::Acquire) {
                    state.reset();
                    state.set_state(WatchdogState::Active);
                }
            }
        }
    }
}
