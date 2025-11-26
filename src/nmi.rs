extern crate alloc;
use alloc::boxed::Box;
use log::warn;
use axplat_aarch64_peripherals::{gic, pmu};

use crate::watchdog::{WATCHDOG, WatchdogTask};

pub fn init(irq_num: usize) {
    gic::set_priority(irq_num as u32, 0);
    pmu::init(0xf0000000);
    let _ = WATCHDOG.lock().register_task(Box::new(Test {}));
}

pub fn handle() {
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
