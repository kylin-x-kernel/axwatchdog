//! CPU snapshot generation for watchdog dumps.
//!
//! This module provides lock-free mechanisms to capture CPU state
//! when a watchdog timeout is detected.

use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of snapshots to keep per CPU.
pub const MAX_SNAPSHOTS_PER_CPU: usize = 4;

/// CPU register snapshot.
///
/// Captures the essential CPU state at the time of watchdog trigger.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpuSnapshot {
    /// CPU ID that this snapshot belongs to.
    pub cpu_id: usize,
    /// Timestamp when snapshot was taken (nanoseconds since boot).
    pub timestamp_ns: u64,
    /// Sequence number for this snapshot.
    pub sequence: u64,
    /// Program counter at interrupt.
    pub pc: usize,
    /// Stack pointer.
    pub sp: usize,
    /// Link register (return address).
    pub lr: usize,
    /// Frame pointer.
    pub fp: usize,
    /// Processor state (PSTATE/CPSR).
    pub pstate: u64,
    /// Exception Syndrome Register (ESR_EL1).
    pub esr: u64,
    /// Fault Address Register (FAR_EL1).
    pub far: u64,
    /// Exception Link Register (ELR_EL1).
    pub elr: u64,
    /// General purpose registers (x0-x30).
    pub gpr: [u64; 31],
}

impl CpuSnapshot {
    /// Create an empty snapshot.
    pub const fn empty() -> Self {
        Self {
            cpu_id: 0,
            timestamp_ns: 0,
            sequence: 0,
            pc: 0,
            sp: 0,
            lr: 0,
            fp: 0,
            pstate: 0,
            esr: 0,
            far: 0,
            elr: 0,
            gpr: [0; 31],
        }
    }

    /// Create a snapshot with basic info.
    pub const fn new(cpu_id: usize, pc: usize, sp: usize, lr: usize, fp: usize) -> Self {
        Self {
            cpu_id,
            timestamp_ns: 0,
            sequence: 0,
            pc,
            sp,
            lr,
            fp,
            pstate: 0,
            esr: 0,
            far: 0,
            elr: 0,
            gpr: [0; 31],
        }
    }

    /// Set timestamp.
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    /// Set sequence number.
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = sequence;
        self
    }

    /// Check if this snapshot is valid (has been populated).
    pub fn is_valid(&self) -> bool {
        self.sequence > 0 || self.pc != 0
    }
}

impl Default for CpuSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

/// Snapshot callback type.
///
/// Called when a new snapshot is captured.
pub type SnapshotCallback = fn(snapshot: &CpuSnapshot);

/// Snapshot collector for a single CPU.
///
/// Uses a circular buffer to store multiple snapshots without allocation.
pub struct SnapshotCollector {
    /// Circular buffer of snapshots.
    snapshots: [CpuSnapshot; MAX_SNAPSHOTS_PER_CPU],
    /// Current write index.
    write_index: AtomicU64,
    /// Global sequence counter.
    sequence: AtomicU64,
}

impl SnapshotCollector {
    /// Create a new snapshot collector.
    pub const fn new() -> Self {
        const EMPTY: CpuSnapshot = CpuSnapshot::empty();
        Self {
            snapshots: [EMPTY; MAX_SNAPSHOTS_PER_CPU],
            write_index: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
        }
    }

    /// Capture a snapshot from the current context.
    ///
    /// This is safe to call from NMI context as it uses no locks.
    ///
    /// # Arguments
    /// * `cpu_id` - Current CPU ID
    /// * `pc` - Program counter
    /// * `sp` - Stack pointer
    /// * `lr` - Link register
    /// * `fp` - Frame pointer
    /// * `timestamp_ns` - Current timestamp
    ///
    /// # Returns
    /// Reference to the stored snapshot.
    pub fn capture(
        &mut self,
        cpu_id: usize,
        pc: usize,
        sp: usize,
        lr: usize,
        fp: usize,
        timestamp_ns: u64,
    ) -> &CpuSnapshot {
        // Get next sequence number
        let seq = self.sequence.fetch_add(1, Ordering::AcqRel) + 1;

        // Get write index (circular)
        let idx = self.write_index.fetch_add(1, Ordering::AcqRel) as usize % MAX_SNAPSHOTS_PER_CPU;

        // Fill snapshot
        self.snapshots[idx] = CpuSnapshot {
            cpu_id,
            timestamp_ns,
            sequence: seq,
            pc,
            sp,
            lr,
            fp,
            pstate: 0,
            esr: 0,
            far: 0,
            elr: 0,
            gpr: [0; 31],
        };

        &self.snapshots[idx]
    }

    /// Capture with full register state.
    pub fn capture_full(
        &mut self,
        cpu_id: usize,
        pc: usize,
        sp: usize,
        lr: usize,
        fp: usize,
        pstate: u64,
        esr: u64,
        far: u64,
        elr: u64,
        gpr: &[u64; 31],
        timestamp_ns: u64,
    ) -> &CpuSnapshot {
        let seq = self.sequence.fetch_add(1, Ordering::AcqRel) + 1;
        let idx = self.write_index.fetch_add(1, Ordering::AcqRel) as usize % MAX_SNAPSHOTS_PER_CPU;

        self.snapshots[idx] = CpuSnapshot {
            cpu_id,
            timestamp_ns,
            sequence: seq,
            pc,
            sp,
            lr,
            fp,
            pstate,
            esr,
            far,
            elr,
            gpr: *gpr,
        };

        &self.snapshots[idx]
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&CpuSnapshot> {
        let idx = self.write_index.load(Ordering::Acquire);
        if idx == 0 {
            return None;
        }
        let actual_idx = ((idx - 1) as usize) % MAX_SNAPSHOTS_PER_CPU;
        let snapshot = &self.snapshots[actual_idx];
        if snapshot.is_valid() {
            Some(snapshot)
        } else {
            None
        }
    }

    /// Get snapshot by index (0 = oldest, N-1 = newest).
    pub fn get(&self, index: usize) -> Option<&CpuSnapshot> {
        if index >= MAX_SNAPSHOTS_PER_CPU {
            return None;
        }
        let snapshot = &self.snapshots[index];
        if snapshot.is_valid() {
            Some(snapshot)
        } else {
            None
        }
    }

    /// Get all valid snapshots, oldest first.
    pub fn iter_valid(&self) -> impl Iterator<Item = &CpuSnapshot> {
        self.snapshots.iter().filter(|s| s.is_valid())
    }

    /// Get the total number of snapshots taken.
    pub fn total_count(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        for snapshot in &mut self.snapshots {
            *snapshot = CpuSnapshot::empty();
        }
        self.write_index.store(0, Ordering::Release);
        // Note: we don't reset sequence to preserve history
    }
}

impl Default for SnapshotCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Capture current CPU state (AArch64 specific).
///
/// # Safety
/// Must be called from privileged mode (EL1 or higher).
#[cfg(target_arch = "aarch64")]
pub unsafe fn capture_current_state() -> CpuSnapshot {
    use core::arch::asm;

    let pc: usize;
    let sp: usize;
    let lr: usize;
    let fp: usize;
    let pstate: u64;
    let esr: u64;
    let far: u64;
    let elr: u64;

    unsafe {
        asm!(
            "adr {pc}, .",
            "mov {sp}, sp",
            "mov {lr}, x30",
            "mov {fp}, x29",
            "mrs {pstate}, DAIF",
            "mrs {esr}, ESR_EL1",
            "mrs {far}, FAR_EL1",
            "mrs {elr}, ELR_EL1",
            pc = out(reg) pc,
            sp = out(reg) sp,
            lr = out(reg) lr,
            fp = out(reg) fp,
            pstate = out(reg) pstate,
            esr = out(reg) esr,
            far = out(reg) far,
            elr = out(reg) elr,
            options(nomem, nostack)
        );
    }

    CpuSnapshot {
        cpu_id: 0,       // Caller should set this
        timestamp_ns: 0, // Caller should set this
        sequence: 0,
        pc,
        sp,
        lr,
        fp,
        pstate,
        esr,
        far,
        elr,
        gpr: [0; 31], // GPRs need separate capture if needed
    }
}

/// Non-AArch64 fallback.
#[cfg(not(target_arch = "aarch64"))]
pub fn capture_current_state() -> CpuSnapshot {
    CpuSnapshot::empty()
}

/// Format a snapshot for display.
pub fn format_snapshot(
    snapshot: &CpuSnapshot,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    writeln!(
        f,
        "=== CPU {} Snapshot (seq: {}) ===",
        snapshot.cpu_id, snapshot.sequence
    )?;
    writeln!(f, "Timestamp: {} ns", snapshot.timestamp_ns)?;
    writeln!(f, "PC:  {:#018x}", snapshot.pc)?;
    writeln!(f, "SP:  {:#018x}", snapshot.sp)?;
    writeln!(f, "LR:  {:#018x}", snapshot.lr)?;
    writeln!(f, "FP:  {:#018x}", snapshot.fp)?;
    writeln!(f, "ELR: {:#018x}", snapshot.elr)?;
    writeln!(f, "ESR: {:#018x}", snapshot.esr)?;
    writeln!(f, "FAR: {:#018x}", snapshot.far)?;
    writeln!(f, "PSTATE: {:#018x}", snapshot.pstate)?;

    // Print GPRs in groups of 4
    for i in (0..31).step_by(4) {
        write!(f, "X{:02}: {:#018x}", i, snapshot.gpr[i])?;
        if i + 1 < 31 {
            write!(f, "  X{:02}: {:#018x}", i + 1, snapshot.gpr[i + 1])?;
        }
        if i + 2 < 31 {
            write!(f, "  X{:02}: {:#018x}", i + 2, snapshot.gpr[i + 2])?;
        }
        if i + 3 < 31 {
            write!(f, "  X{:02}: {:#018x}", i + 3, snapshot.gpr[i + 3])?;
        }
        writeln!(f)?;
    }

    Ok(())
}

impl core::fmt::Display for CpuSnapshot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        format_snapshot(self, f)
    }
}
