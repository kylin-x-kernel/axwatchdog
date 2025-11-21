use axplat_aarch64_peripherals::{pmu,gic};
use log::warn;

pub fn init(irq_num:usize) {
    warn!("nmi init");
    gic::set_priority(irq_num as u32,0);
    pmu::init();
}

pub fn handle(){
    warn!("nmi handle");
}