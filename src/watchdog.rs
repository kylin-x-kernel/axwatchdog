extern crate alloc;
use alloc::{boxed::Box, vec::Vec};
use axplat::cpu::this_cpu_id;
use core::sync::atomic::{AtomicBool, Ordering};

pub static mut WATCHDOG: Watchdog = Watchdog { tasks: Vec::new() };
/// Watchdog task trait. Modules that should be monitored implement this trait.
pub trait WatchdogTask {
    /// Unique identifier for the task (e.g. name or ID).
    /// Keep it simple in no_std environments.
    fn id(&self) -> &str;

    /// Check whether the task is healthy.
    /// Return `true` if healthy, `false` to trigger recovery actions.
    fn is_healthy(&self) -> bool;
}

pub struct Watchdog {
    tasks: Vec<Box<dyn WatchdogTask + Send>>,
}

impl Watchdog {
    /// Register a task. Returns error on duplicate ID.
    pub fn register_task(
        &mut self,
        task: Box<dyn WatchdogTask + Send>,
    ) -> Result<(), WatchdogError> {
        // 防止重复注册相同ID的任务
        if self.tasks.iter().any(|t| t.id() == task.id()) {
            return Err(WatchdogError::DuplicateTaskId);
        }
        self.tasks.push(task);
        Ok(())
    }

    /// Poll all registered tasks and handle any unhealthy ones.
    pub fn poll(&mut self) {
        for task in &mut self.tasks {
            if !task.is_healthy() {
                // trigger freeze and dump across CPUs
                axipi::run_on_each_cpu(freeze_cpu_and_dump);
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum WatchdogError {
    DuplicateTaskId, // duplicate task ID
}

/// Freeze CPU and wait
fn freeze_cpu_and_dump() {
    let cpu_id = this_cpu_id();

    CPU_FROZEN_FLAGS[cpu_id].store(true, Ordering::Release);

    wait_all_cpus_frozen();

    // Master CPU is responsible for generating snapshot
    if cpu_id == 0{
        generate_system_snapshot(cpu_id);
    }
    // Enter infinite loop, waiting for debugging or restart
    loop{
        axcpu::asm::halt();
    }
}

fn generate_system_snapshot(_cpu_id: usize){
    axtask::show_global_task_queue();
    panic!("watchdog panic");
}

// Per-CPU frozen state flags
static CPU_FROZEN_FLAGS: [AtomicBool; axconfig::plat::CPU_NUM] = {
    let flags = [const { AtomicBool::new(false) }; axconfig::plat::CPU_NUM];
    flags
};

fn wait_all_cpus_frozen() {
    loop {
        let mut all_frozen = true;
        for i in 0..axconfig::plat::CPU_NUM {
                    // Use Acquire ordering to read latest value
            if !CPU_FROZEN_FLAGS[i].load(Ordering::Acquire) {
                all_frozen = false;
                break;
            }
        }
        if all_frozen {
            break;
        }
        // Hint to CPU to reduce spin power
        core::hint::spin_loop();
    }
}