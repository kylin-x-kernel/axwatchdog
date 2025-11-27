//! AArch64 Watchdog Timer with NMI Support.
//!
//! This crate provides a watchdog implementation that uses NMI (Non-Maskable Interrupt)
//! sources like SDEI or PMU overflow to ensure the watchdog can trigger even when
//! normal interrupts are disabled.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                   Application Layer                  │
//! │  - Register tasks via WatchdogTask trait            │
//! │  - Call pet() to indicate liveness                  │
//! └───────────────────────────┬─────────────────────────┘
//!                             │
//! ┌───────────────────────────▼─────────────────────────┐
//! │                   Watchdog Core                      │
//! │  - Per-CPU state tracking (lock-free)               │
//! │  - Task registry                                    │
//! │  - Snapshot generation                              │
//! └───────────────────────────┬─────────────────────────┘
//!                             │
//! ┌───────────────────────────▼─────────────────────────┐
//! │                   NMI Source Layer                   │
//! │  - SDEI (Software Delegated Exception Interface)    │
//! │  - PMU Overflow                                     │
//! └─────────────────────────────────────────────────────┘
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
//!     fn name(&self) -> &'static str { "my_task" }
//!     fn health_check(&self) -> bool { true }
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
pub use lockfree::{AtomicBitmap, AtomicSequence, SpscRingBuffer};
pub use nmi::{NmiContext, NmiError, NmiHandler, NmiResult, NmiSource};
pub use percpu::{pet, pet_with_timestamp, PerCpuArray, PerCpuState, WatchdogState, PERCPU_STATE};
pub use snapshot::{CpuSnapshot, SnapshotCollector};
pub use task::{
    register_task, unregister_task, HeartbeatTask, RegistryError, TaskHandle, TaskRegistry,
    WatchdogTask, TASK_REGISTRY,
};

#[cfg(feature = "sdei")]
pub use nmi::SdeiNmi;

#[cfg(feature = "pmu")]
pub use nmi::PmuNmiSource;

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Watchdog configuration.
#[derive(Debug, Clone, Copy)]
pub struct WatchdogConfig {
    /// Timeout period in nanoseconds.
    pub timeout_ns: u64,
    /// Number of consecutive failures before triggering.
    pub fail_threshold: u32,
    /// Number of CPUs to monitor.
    pub num_cpus: usize,
    /// Whether to automatically restart after handling.
    pub auto_restart: bool,
}

impl WatchdogConfig {
    /// Create a new configuration with default values.
    pub const fn new() -> Self {
        Self {
            timeout_ns: 10_000_000_000, // 10 seconds
            fail_threshold: 3,
            num_cpus: 1,
            auto_restart: true,
        }
    }

    /// Set the timeout period.
    pub const fn with_timeout_ns(mut self, timeout_ns: u64) -> Self {
        self.timeout_ns = timeout_ns;
        self
    }

    /// Set the timeout period in milliseconds.
    pub const fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ns = timeout_ms * 1_000_000;
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
    /// Configuration.
    timeout_ns: AtomicU64,
    fail_threshold: AtomicU32,
    num_cpus: AtomicU32,
    auto_restart: AtomicBool,
    /// Statistics.
    total_checks: AtomicU64,
    total_failures: AtomicU64,
}

impl Watchdog {
    /// Create a new watchdog instance.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            running: AtomicBool::new(false),
            timeout_ns: AtomicU64::new(10_000_000_000),
            fail_threshold: AtomicU32::new(3),
            num_cpus: AtomicU32::new(1),
            auto_restart: AtomicBool::new(true),
            total_checks: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
        }
    }

    /// Initialize the watchdog with the given configuration.
    pub fn init(config: WatchdogConfig) -> WatchdogResult<()> {
        if WATCHDOG.initialized.swap(true, Ordering::AcqRel) {
            return Err(WatchdogError::AlreadyInitialized);
        }

        WATCHDOG.timeout_ns.store(config.timeout_ns, Ordering::Release);
        WATCHDOG.fail_threshold.store(config.fail_threshold, Ordering::Release);
        WATCHDOG.num_cpus.store(config.num_cpus as u32, Ordering::Release);
        WATCHDOG.auto_restart.store(config.auto_restart, Ordering::Release);

        #[cfg(feature = "log")]
        log::info!(
            "Watchdog initialized: timeout={}ns, threshold={}, cpus={}",
            config.timeout_ns,
            config.fail_threshold,
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

    /// Get current timeout in nanoseconds.
    pub fn timeout_ns() -> u64 {
        WATCHDOG.timeout_ns.load(Ordering::Acquire)
    }

    /// Get total check count.
    pub fn total_checks() -> u64 {
        WATCHDOG.total_checks.load(Ordering::Acquire)
    }

    /// Get total failure count.
    pub fn total_failures() -> u64 {
        WATCHDOG.total_failures.load(Ordering::Acquire)
    }
}

impl Default for Watchdog {
    fn default() -> Self {
        Self::new()
    }
}

/// Global watchdog instance.
pub static WATCHDOG: Watchdog = Watchdog::new();

/// NMI handler for watchdog.
///
/// This is called from the NMI source when triggered.
/// It performs lock-free health checks and captures state if needed.
pub fn watchdog_nmi_handler(cpu_id: usize, ctx: &NmiContext) {
    if !WATCHDOG.running.load(Ordering::Acquire) {
        return;
    }

    // Increment check counter
    WATCHDOG.total_checks.fetch_add(1, Ordering::Relaxed);

    // Check per-CPU heartbeat
    let cpu_healthy = PERCPU_STATE.check_heartbeat(cpu_id);

    // Check registered tasks
    let tasks_healthy = TASK_REGISTRY.check_all();

    if !cpu_healthy || !tasks_healthy {
        // Increment failure counter
        WATCHDOG.total_failures.fetch_add(1, Ordering::Relaxed);

        if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
            let fail_count = state.increment_fail_count();
            let threshold = WATCHDOG.fail_threshold.load(Ordering::Acquire);

            if fail_count >= threshold {
                // Transition to triggered state
                if state.try_transition(WatchdogState::Active, WatchdogState::Triggered) {
                    // Capture snapshot
                    handle_watchdog_timeout(cpu_id, ctx);
                }
            }
        }
    } else {
        // Reset fail count on success
        if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
            state.reset_fail_count();
        }
    }
}

/// Handle watchdog timeout.
///
/// This captures the CPU state and optionally triggers recovery.
fn handle_watchdog_timeout(cpu_id: usize, ctx: &NmiContext) {
    #[cfg(feature = "log")]
    log::error!(
        "Watchdog timeout on CPU {}: PC={:#x}, SP={:#x}",
        cpu_id,
        ctx.pc,
        ctx.sp
    );

    // The actual handling (logging, panic, etc.) would be done here
    // For now, we just record the state

    if let Some(state) = PERCPU_STATE.try_get(cpu_id) {
        // Transition to completed
        state.set_state(WatchdogState::Completed);

        // Auto-restart if configured
        if WATCHDOG.auto_restart.load(Ordering::Acquire) {
            state.reset();
            state.set_state(WatchdogState::Active);
        }
    }
}
