//! NMI (Non-Maskable Interrupt) source abstraction.
//!
//! This module defines the trait for NMI sources that can be used to trigger
//! watchdog checks. Different hardware mechanisms (SDEI, PMU overflow, etc.)
//! can implement this trait.
//!
//! # Lock-free Requirement
//!
//! All NMI handlers MUST be lock-free. The handler runs in a special context
//! where acquiring any lock could cause deadlock. Only use:
//! - Atomic operations
//! - Per-CPU data structures
//! - Direct memory/register access

use core::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// NMI context containing interrupted CPU state.
///
/// This is populated when an NMI fires and passed to the handler.
/// All fields are read-only snapshots of the interrupted state.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NmiContext {
    /// CPU ID that received the NMI.
    pub cpu_id: usize,
    /// Program counter at interrupt.
    pub pc: usize,
    /// Stack pointer at interrupt.
    pub sp: usize,
    /// Link register at interrupt.
    pub lr: usize,
    /// Frame pointer at interrupt.
    pub fp: usize,
    /// Processor state at interrupt.
    pub pstate: u64,
    /// Timestamp (nanoseconds) when NMI occurred.
    pub timestamp_ns: u64,
}

impl NmiContext {
    /// Create a new empty NMI context.
    pub const fn empty() -> Self {
        Self {
            cpu_id: 0,
            pc: 0,
            sp: 0,
            lr: 0,
            fp: 0,
            pstate: 0,
            timestamp_ns: 0,
        }
    }

    /// Create context with basic information.
    pub const fn new(cpu_id: usize, pc: usize, sp: usize, lr: usize, fp: usize) -> Self {
        Self {
            cpu_id,
            pc,
            sp,
            lr,
            fp,
            pstate: 0,
            timestamp_ns: 0,
        }
    }
}

impl Default for NmiContext {
    fn default() -> Self {
        Self::empty()
    }
}

/// NMI handler function type.
///
/// # Safety Contract
///
/// The handler function MUST:
/// - Be completely lock-free (no mutexes, spinlocks, etc.)
/// - Use only atomic operations for shared data
/// - Complete quickly (avoid long computations)
/// - Not allocate memory
/// - Not call any function that may block
///
/// The handler receives:
/// - `cpu_id`: The CPU that received the NMI
/// - `ctx`: Context information about the interrupted state
pub type NmiHandler = fn(cpu_id: usize, ctx: &NmiContext);

/// Error type for NMI operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmiError {
    /// NMI source not available on this platform.
    NotAvailable,
    /// NMI source not initialized.
    NotInitialized,
    /// NMI source already initialized.
    AlreadyInitialized,
    /// Invalid configuration parameter.
    InvalidConfig,
    /// Handler already registered.
    HandlerExists,
    /// No handler registered.
    NoHandler,
    /// Operation not supported.
    NotSupported,
    /// Hardware error.
    HardwareError,
}

impl core::fmt::Display for NmiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NmiError::NotAvailable => write!(f, "NMI source not available"),
            NmiError::NotInitialized => write!(f, "NMI source not initialized"),
            NmiError::AlreadyInitialized => write!(f, "NMI source already initialized"),
            NmiError::InvalidConfig => write!(f, "Invalid configuration"),
            NmiError::HandlerExists => write!(f, "Handler already registered"),
            NmiError::NoHandler => write!(f, "No handler registered"),
            NmiError::NotSupported => write!(f, "Operation not supported"),
            NmiError::HardwareError => write!(f, "Hardware error"),
        }
    }
}

/// Result type for NMI operations.
pub type NmiResult<T> = Result<T, NmiError>;

/// Trait for NMI sources.
///
/// Implementors provide a mechanism to trigger NMI-like interrupts
/// at regular intervals for watchdog purposes.
pub trait NmiSource: Send + Sync {
    /// Initialize the NMI source.
    ///
    /// This should configure the hardware but not start triggering.
    fn init(&self) -> NmiResult<()>;

    /// Set the NMI handler callback.
    ///
    /// Only one handler can be registered at a time.
    fn set_handler(&self, handler: NmiHandler) -> NmiResult<()>;

    /// Clear the registered handler.
    fn clear_handler(&self) -> NmiResult<()>;

    /// Enable NMI generation.
    ///
    /// After this call, NMIs will be triggered at the configured period.
    fn enable(&self) -> NmiResult<()>;

    /// Disable NMI generation.
    fn disable(&self) -> NmiResult<()>;

    /// Check if NMI generation is currently enabled.
    fn is_enabled(&self) -> bool;

    /// Set the NMI trigger period in nanoseconds.
    ///
    /// The actual period may be approximate depending on hardware.
    fn set_period_ns(&self, period: u64) -> NmiResult<()>;

    /// Get the current configured period in nanoseconds.
    fn period_ns(&self) -> u64;

    /// Acknowledge/complete the NMI.
    ///
    /// Must be called at the end of NMI handling.
    fn acknowledge(&self);

    /// Get the name of this NMI source (for debugging).
    fn name(&self) -> &'static str;
}

// =============================================================================
// Generic NMI Handler Storage (Lock-free)
// =============================================================================

/// Maximum number of CPUs supported.
pub const MAX_CPUS: usize = 64;

/// Lock-free storage for NMI handler.
pub struct NmiHandlerStorage {
    /// The registered handler function.
    handler: AtomicPtr<()>,
    /// Period in nanoseconds.
    period_ns: AtomicU64,
    /// Whether enabled.
    enabled: AtomicUsize,
}

impl NmiHandlerStorage {
    /// Create new empty storage.
    pub const fn new() -> Self {
        Self {
            handler: AtomicPtr::new(core::ptr::null_mut()),
            period_ns: AtomicU64::new(0),
            enabled: AtomicUsize::new(0),
        }
    }

    /// Set the handler.
    pub fn set_handler(&self, handler: NmiHandler) -> NmiResult<()> {
        let ptr = handler as *mut ();
        let old = self.handler.swap(ptr, Ordering::AcqRel);
        if !old.is_null() && old != ptr {
            // There was a different handler - that's okay, we replaced it
        }
        Ok(())
    }

    /// Clear the handler.
    pub fn clear_handler(&self) -> NmiResult<()> {
        self.handler.store(core::ptr::null_mut(), Ordering::Release);
        Ok(())
    }

    /// Get the handler if registered.
    pub fn get_handler(&self) -> Option<NmiHandler> {
        let ptr = self.handler.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            // SAFETY: We only store valid function pointers
            Some(unsafe { core::mem::transmute::<*mut (), NmiHandler>(ptr) })
        }
    }

    /// Invoke the handler if registered.
    pub fn invoke(&self, cpu_id: usize, ctx: &NmiContext) {
        if let Some(handler) = self.get_handler() {
            handler(cpu_id, ctx);
        }
    }

    /// Set enabled state.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled as usize, Ordering::Release);
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire) != 0
    }

    /// Set period.
    pub fn set_period_ns(&self, period: u64) {
        self.period_ns.store(period, Ordering::Release);
    }

    /// Get period.
    pub fn period_ns(&self) -> u64 {
        self.period_ns.load(Ordering::Acquire)
    }
}

impl Default for NmiHandlerStorage {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: NmiHandlerStorage uses only atomic operations
unsafe impl Send for NmiHandlerStorage {}
unsafe impl Sync for NmiHandlerStorage {}

// =============================================================================
// SDEI-based NMI Source
// =============================================================================

#[cfg(feature = "sdei")]
pub mod sdei_nmi {
    //! SDEI-based NMI implementation.

    use aarch64_sdei::{SDEI_EVENT_SOFTWARE_NMI, Sdei};

    use super::*;

    /// SDEI-based NMI source.
    pub struct SdeiNmi {
        sdei: Sdei,
        storage: NmiHandlerStorage,
    }

    impl SdeiNmi {
        /// Create a new SDEI NMI source.
        pub const fn new() -> Self {
            Self {
                sdei: Sdei::new(),
                storage: NmiHandlerStorage::new(),
            }
        }

        /// Get the underlying SDEI interface.
        pub fn sdei(&self) -> &Sdei {
            &self.sdei
        }
    }

    impl NmiSource for SdeiNmi {
        fn init(&self) -> NmiResult<()> {
            self.sdei.init().map_err(|_| NmiError::NotAvailable)?;
            Ok(())
        }

        fn set_handler(&self, handler: NmiHandler) -> NmiResult<()> {
            self.storage.set_handler(handler)
        }

        fn clear_handler(&self) -> NmiResult<()> {
            self.storage.clear_handler()
        }

        fn enable(&self) -> NmiResult<()> {
            // For SDEI, enable happens when event is registered and enabled
            // The actual triggering is done via sdei_event_signal
            self.storage.set_enabled(true);
            Ok(())
        }

        fn disable(&self) -> NmiResult<()> {
            self.storage.set_enabled(false);
            self.sdei
                .event_disable(SDEI_EVENT_SOFTWARE_NMI)
                .map_err(|_| NmiError::HardwareError)
        }

        fn is_enabled(&self) -> bool {
            self.storage.is_enabled()
        }

        fn set_period_ns(&self, period: u64) -> NmiResult<()> {
            // SDEI doesn't have built-in periodic triggering
            // Store for informational purposes; actual triggering needs timer
            self.storage.set_period_ns(period);
            Ok(())
        }

        fn period_ns(&self) -> u64 {
            self.storage.period_ns()
        }

        fn acknowledge(&self) {
            // SDEI completion is handled by the low-level handler
        }

        fn name(&self) -> &'static str {
            "SDEI"
        }
    }

    impl Default for SdeiNmi {
        fn default() -> Self {
            Self::new()
        }
    }
}

// =============================================================================
// PMU-based NMI Source
// =============================================================================

#[cfg(feature = "pmu")]
pub mod pmu_nmi {
    //! PMU overflow-based NMI implementation.

    use aarch64_pmuv3::PmuNmi as PmuNmiHw;

    use super::*;

    /// PMU overflow-based NMI source.
    pub struct PmuNmiSource {
        pmu: PmuNmiHw,
        storage: NmiHandlerStorage,
    }

    impl PmuNmiSource {
        /// Create a new PMU NMI source with the given cycle threshold.
        pub const fn new(threshold: u64) -> Self {
            Self {
                pmu: PmuNmiHw::new_cycle_counter(threshold),
                storage: NmiHandlerStorage::new(),
            }
        }

        /// Get the underlying PMU interface.
        pub fn pmu(&self) -> &PmuNmiHw {
            &self.pmu
        }

        /// Handle PMU overflow.
        ///
        /// Call this from your interrupt handler.
        /// Returns true if overflow was handled.
        pub fn handle_overflow(&self, cpu_id: usize) -> bool {
            if self.pmu.handle_overflow() {
                // Create context and invoke handler
                let ctx = NmiContext::new(cpu_id, 0, 0, 0, 0);
                self.storage.invoke(cpu_id, &ctx);
                true
            } else {
                false
            }
        }
    }

    impl NmiSource for PmuNmiSource {
        fn init(&self) -> NmiResult<()> {
            self.pmu.init().map_err(|_| NmiError::NotAvailable)
        }

        fn set_handler(&self, handler: NmiHandler) -> NmiResult<()> {
            self.storage.set_handler(handler)
        }

        fn clear_handler(&self) -> NmiResult<()> {
            self.storage.clear_handler()
        }

        fn enable(&self) -> NmiResult<()> {
            self.pmu.enable().map_err(|_| NmiError::HardwareError)?;
            self.storage.set_enabled(true);
            Ok(())
        }

        fn disable(&self) -> NmiResult<()> {
            self.storage.set_enabled(false);
            self.pmu.disable().map_err(|_| NmiError::HardwareError)
        }

        fn is_enabled(&self) -> bool {
            self.storage.is_enabled()
        }

        fn set_period_ns(&self, period: u64) -> NmiResult<()> {
            // For PMU, we'd need to convert ns to cycles based on CPU frequency
            // For now, store the value
            self.storage.set_period_ns(period);
            Ok(())
        }

        fn period_ns(&self) -> u64 {
            self.storage.period_ns()
        }

        fn acknowledge(&self) {
            // PMU acknowledgment is handled in handle_overflow
        }

        fn name(&self) -> &'static str {
            "PMU"
        }
    }
}

// =============================================================================
// Re-exports
// =============================================================================

#[cfg(feature = "pmu")]
pub use pmu_nmi::PmuNmiSource;
#[cfg(feature = "sdei")]
pub use sdei_nmi::SdeiNmi;
