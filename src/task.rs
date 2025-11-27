//! Watchdog task registration and management.
//!
//! This module provides a lock-free task registry for registering
//! tasks that need to be monitored by the watchdog.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Maximum number of tasks that can be registered.
pub const MAX_WATCHDOG_TASKS: usize = 32;

/// Trait for tasks that can be monitored by the watchdog.
///
/// # Safety Contract
///
/// The `health_check` method may be called from NMI context and MUST:
/// - Be completely lock-free
/// - Complete quickly
/// - Not allocate memory
/// - Not panic
pub trait WatchdogTask: Send + Sync {
    /// Get the task name (for diagnostics).
    fn name(&self) -> &'static str;

    /// Check if the task is healthy.
    ///
    /// Returns `true` if the task is operating normally,
    /// `false` if it appears to be stuck or unhealthy.
    ///
    /// This method may be called from NMI context and must be lock-free.
    fn health_check(&self) -> bool;

    /// Called when watchdog detects this task is unhealthy.
    ///
    /// This is called from normal context (not NMI) after detection.
    /// Can perform logging, recovery attempts, etc.
    fn on_timeout(&self, cpu_id: usize) {
        let _ = cpu_id; // Default: do nothing
    }

    /// Get the timeout threshold for this task (in nanoseconds).
    ///
    /// If not overridden, uses the global watchdog timeout.
    fn timeout_ns(&self) -> Option<u64> {
        None // Use global default
    }
}

/// Handle returned when registering a task.
///
/// Used to unregister the task later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskHandle {
    /// Slot index in the registry.
    index: usize,
    /// Generation counter to detect stale handles.
    generation: u32,
}

impl TaskHandle {
    /// Create a new task handle.
    const fn new(index: usize, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Get the slot index.
    pub fn index(&self) -> usize {
        self.index
    }
}

/// Error type for task registry operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryError {
    /// Registry is full, cannot register more tasks.
    Full,
    /// Invalid handle (wrong index or generation).
    InvalidHandle,
    /// Slot is already occupied.
    SlotOccupied,
    /// Task not found.
    NotFound,
}

impl core::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RegistryError::Full => write!(f, "Task registry is full"),
            RegistryError::InvalidHandle => write!(f, "Invalid task handle"),
            RegistryError::SlotOccupied => write!(f, "Slot already occupied"),
            RegistryError::NotFound => write!(f, "Task not found"),
        }
    }
}

/// Internal slot in the task registry.
struct TaskSlot {
    /// Pointer to the registered task (null if empty).
    /// Stored as raw pointer to trait object.
    task_ptr: AtomicUsize,
    task_vtable: AtomicUsize,
    /// Generation counter for this slot.
    generation: AtomicU32,
}

impl TaskSlot {
    const fn new() -> Self {
        Self {
            task_ptr: AtomicUsize::new(0),
            task_vtable: AtomicUsize::new(0),
            generation: AtomicU32::new(0),
        }
    }

    fn is_empty(&self) -> bool {
        self.task_ptr.load(Ordering::Acquire) == 0
    }

    fn get(&self) -> Option<&'static dyn WatchdogTask> {
        let ptr = self.task_ptr.load(Ordering::Acquire);
        let vtable = self.task_vtable.load(Ordering::Acquire);
        if ptr == 0 {
            None
        } else {
            // SAFETY: We only store valid task trait object pointers
            // Reconstruct the fat pointer from data + vtable
            let fat_ptr: *const dyn WatchdogTask = unsafe {
                core::mem::transmute::<(usize, usize), *const dyn WatchdogTask>((ptr, vtable))
            };
            Some(unsafe { &*fat_ptr })
        }
    }

    fn try_set(&self, task: &'static dyn WatchdogTask) -> Option<u32> {
        // Convert trait object to raw parts
        let fat_ptr: *const dyn WatchdogTask = task;
        let (ptr, vtable): (usize, usize) = unsafe { core::mem::transmute(fat_ptr) };

        // Try to set the pointer (use ptr as the "occupied" indicator)
        let old = self
            .task_ptr
            .compare_exchange(0, ptr, Ordering::AcqRel, Ordering::Acquire);

        if old.is_ok() {
            self.task_vtable.store(vtable, Ordering::Release);
            let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
            Some(generation)
        } else {
            None
        }
    }

    fn try_clear(&self, expected_generation: u32) -> bool {
        let current_generation = self.generation.load(Ordering::Acquire);
        if current_generation != expected_generation {
            return false;
        }

        // Clear the pointer
        let old = self.task_ptr.swap(0, Ordering::AcqRel);
        if old != 0 {
            self.task_vtable.store(0, Ordering::Release);
            true
        } else {
            false
        }
    }
}

// SAFETY: TaskSlot uses only atomic operations
unsafe impl Send for TaskSlot {}
unsafe impl Sync for TaskSlot {}

/// Lock-free task registry.
///
/// Tasks can be registered and unregistered without locks.
/// Health checks iterate over all registered tasks.
pub struct TaskRegistry {
    /// Task slots.
    slots: [TaskSlot; MAX_WATCHDOG_TASKS],
    /// Number of registered tasks (approximate, for optimization).
    count: AtomicUsize,
}

impl TaskRegistry {
    /// Create a new empty registry.
    pub const fn new() -> Self {
        const EMPTY_SLOT: TaskSlot = TaskSlot::new();
        Self {
            slots: [EMPTY_SLOT; MAX_WATCHDOG_TASKS],
            count: AtomicUsize::new(0),
        }
    }

    /// Register a task.
    ///
    /// Returns a handle that can be used to unregister the task.
    pub fn register(&self, task: &'static dyn WatchdogTask) -> Result<TaskHandle, RegistryError> {
        // Find an empty slot
        for (index, slot) in self.slots.iter().enumerate() {
            if let Some(generation) = slot.try_set(task) {
                self.count.fetch_add(1, Ordering::AcqRel);

                #[cfg(feature = "log")]
                log::debug!(
                    "Registered watchdog task '{}' at slot {}",
                    task.name(),
                    index
                );

                return Ok(TaskHandle::new(index, generation));
            }
        }

        Err(RegistryError::Full)
    }

    /// Unregister a task using its handle.
    pub fn unregister(&self, handle: TaskHandle) -> Result<(), RegistryError> {
        if handle.index >= MAX_WATCHDOG_TASKS {
            return Err(RegistryError::InvalidHandle);
        }

        let slot = &self.slots[handle.index];
        if slot.try_clear(handle.generation) {
            self.count.fetch_sub(1, Ordering::AcqRel);

            #[cfg(feature = "log")]
            log::debug!("Unregistered watchdog task at slot {}", handle.index);

            Ok(())
        } else {
            Err(RegistryError::InvalidHandle)
        }
    }

    /// Get the number of registered tasks.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Check health of all registered tasks.
    ///
    /// Returns `true` if all tasks are healthy, `false` if any task failed.
    ///
    /// This is safe to call from NMI context.
    pub fn check_all(&self) -> bool {
        for slot in &self.slots {
            if let Some(task) = slot.get() {
                if !task.health_check() {
                    return false;
                }
            }
        }
        true
    }

    /// Check health and return bitmap of unhealthy tasks.
    ///
    /// Bit N is set if task at slot N is unhealthy.
    pub fn check_all_bitmap(&self) -> u32 {
        let mut unhealthy: u32 = 0;
        for (index, slot) in self.slots.iter().enumerate() {
            if let Some(task) = slot.get() {
                if !task.health_check() {
                    unhealthy |= 1 << index;
                }
            }
        }
        unhealthy
    }

    /// Get a task by its slot index.
    pub fn get(&self, index: usize) -> Option<&'static dyn WatchdogTask> {
        if index >= MAX_WATCHDOG_TASKS {
            return None;
        }
        self.slots[index].get()
    }

    /// Iterate over all registered tasks.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &'static dyn WatchdogTask)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.get().map(|t| (i, t)))
    }

    /// Call on_timeout for all unhealthy tasks.
    ///
    /// This should be called from normal context (not NMI).
    pub fn notify_timeout(&self, cpu_id: usize) {
        for slot in &self.slots {
            if let Some(task) = slot.get() {
                if !task.health_check() {
                    task.on_timeout(cpu_id);
                }
            }
        }
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: TaskRegistry uses only atomic operations
unsafe impl Send for TaskRegistry {}
unsafe impl Sync for TaskRegistry {}

/// Global task registry.
pub static TASK_REGISTRY: TaskRegistry = TaskRegistry::new();

/// Register a task with the global registry.
pub fn register_task(task: &'static dyn WatchdogTask) -> Result<TaskHandle, RegistryError> {
    TASK_REGISTRY.register(task)
}

/// Unregister a task from the global registry.
pub fn unregister_task(handle: TaskHandle) -> Result<(), RegistryError> {
    TASK_REGISTRY.unregister(handle)
}

/// Check health of all registered tasks.
pub fn check_all_tasks() -> bool {
    TASK_REGISTRY.check_all()
}

// =============================================================================
// Simple task implementations
// =============================================================================

/// A simple heartbeat-based watchdog task.
///
/// The task is considered healthy if `pet()` has been called
/// since the last health check.
pub struct HeartbeatTask {
    name: &'static str,
    counter: core::sync::atomic::AtomicU64,
    last_checked: core::sync::atomic::AtomicU64,
}

impl HeartbeatTask {
    /// Create a new heartbeat task.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            counter: core::sync::atomic::AtomicU64::new(0),
            last_checked: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Pet the watchdog (indicate the task is alive).
    #[inline]
    pub fn pet(&self) {
        self.counter.fetch_add(1, Ordering::Release);
    }

    /// Get current counter value.
    pub fn counter(&self) -> u64 {
        self.counter.load(Ordering::Acquire)
    }
}

impl WatchdogTask for HeartbeatTask {
    fn name(&self) -> &'static str {
        self.name
    }

    fn health_check(&self) -> bool {
        let current = self.counter.load(Ordering::Acquire);
        let last = self.last_checked.swap(current, Ordering::AcqRel);
        current != last
    }
}

// SAFETY: HeartbeatTask uses only atomic operations
unsafe impl Send for HeartbeatTask {}
unsafe impl Sync for HeartbeatTask {}
