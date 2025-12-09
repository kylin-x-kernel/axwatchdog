use alloc::boxed::Box;

use crate::watchdog::{WATCHDOG, WatchdogTask};

pub fn init(irq_num: usize) {
    // Register a simple test watchdog task
    let _ = unsafe { WATCHDOG.register_task(Box::new(Test {})) };
    // Set IRQ priority and initialize PMU for watchdog
    axplat::irq::set_priority(irq_num, 0x0);
    axplat::irq::pmu_init(0xf0000000);
}

pub fn handle() {
    // Poll registered watchdog tasks
    unsafe {  WATCHDOG.poll();};
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
