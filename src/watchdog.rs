extern crate alloc;
use alloc::{boxed::Box, vec::Vec};
use log::warn;
use lazy_static::lazy_static;
use spin::Mutex;
use axplat::cpu::this_cpu_id;
use core::sync::atomic::{AtomicBool, Ordering};
lazy_static! {
    pub static ref WATCHDOG: Mutex<Watchdog> = Mutex::new(Watchdog::new());
}
/// 看门狗任务 trait。所有需要被看门狗监控的模块都必须实现这个 trait。
pub trait WatchdogTask {
    /// 任务的唯一标识符（例如，任务ID或名称）。
    /// 在 no_std 环境下，通常使用固定大小的数组或数字ID以避免动态分配。
    fn id(&self) -> &str;

    /// 检查任务是否健康。
    /// 看门狗定期调用此方法。如果返回 `true`，表示任务运行正常。
    /// 如果返回 `false`，看门狗可能根据策略触发系统恢复。
    fn is_healthy(&self) -> bool;
}

pub struct Watchdog {
    tasks: Vec<Box<dyn WatchdogTask + Send>>,
}

impl Watchdog {
    fn new() -> Self {
        Watchdog { tasks: Vec::new() }
    }

    /// 注册一个任务。如果容量已满，返回错误。
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

    /// 它遍历所有任务，检查健康状态，并处理不健康的任务。
    pub fn poll(&mut self) {
        for task in &mut self.tasks {
            if !task.is_healthy() {
                //warn!("send ipi,current cpu: {},cpu_num: {}",this_cpu_id(),axconfig::plat::CPU_NUM);
                axipi::run_on_each_cpu(freeze_cpu_and_dump);
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum WatchdogError {
    DuplicateTaskId, // 重复的任务ID
}

/// Freeze CPU and wait
fn freeze_cpu_and_dump() {
    let cpu_id = this_cpu_id();

    // 1. 标记当前CPU为已冻结
    CPU_FROZEN_FLAGS[cpu_id].store(true, Ordering::Release);

    // 2. 等待所有CPU冻结
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

fn generate_system_snapshot(cpu_id: usize){
    warn!("cpu id: {},system log",cpu_id);
}
// 每个CPU的冻结状态标志数组
static CPU_FROZEN_FLAGS: [AtomicBool; axconfig::plat::CPU_NUM] = {
    let flags = [const { AtomicBool::new(false) }; axconfig::plat::CPU_NUM];
    flags
};

fn wait_all_cpus_frozen() {
    loop {
        let mut all_frozen = true;
        for i in 0..axconfig::plat::CPU_NUM {
            // 使用Acquire排序确保读取到最新值
            if !CPU_FROZEN_FLAGS[i].load(Ordering::Acquire) {
                all_frozen = false;
                break;
            }
        }
        if all_frozen {
            break;
        }
        // 提示CPU降低自旋功耗
        core::hint::spin_loop();
    }
}