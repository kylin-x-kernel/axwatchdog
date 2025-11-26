use log::warn;

/// Non Maskable Interrupt
pub trait NonMaskableInterruptTrait {
    fn nmi_init(&self);
    fn nmi_handle(&self);
}

pub fn handle() {
    warn!("nmi handle");
}