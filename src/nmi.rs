use axplat::irq;
use log::warn;

// pub fn init(irq_num:usize) {
//     warn!("nmi init");
//     gic::set_priority(irq_num as u32,0);
//     pmu::init(0xf0000000);
// }

pub fn handle(){
    warn!("nmi handle");
}