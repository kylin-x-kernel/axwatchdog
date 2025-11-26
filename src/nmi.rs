extern crate alloc;
use alloc::boxed::Box;
use axlog::warn;

use crate::watchdog::{WATCHDOG, WatchdogTask};

pub fn init(irq_num: usize) {
    // Set IRQ priority and initialize PMU for watchdog
    axplat::irq::set_priority(irq_num, 0);
    axplat::irq::pmu_init(0xf0000000);
    // Register a simple test watchdog task
    let _ = WATCHDOG.lock().register_task(Box::new(Test {}));
}

pub fn handle() {
    // Poll registered watchdog tasks
    WATCHDOG.lock().poll();
    warn!("finish poll");
}

struct Test;

impl WatchdogTask for Test {
    fn id(&self) -> &str {
        "test"
    }

    fn is_healthy(&self) -> bool {
        false
    }
}
