#![no_std]

extern crate alloc;

use alloc::sync::Arc;

pub mod nmi;

pub trait WatchdogTask {
    fn health_check(&self) -> bool;
}

pub fn watchdog_init(nmi_irq: usize) {
    // axplat::irq::enable_nmi(nmi_irq);
}

pub fn watchdog_register(new_task: Arc<&dyn WatchdogTask>) -> Result<(), &'static str> {
    // axplat::irq::register_handler(29, nmi::handle);
    Ok(())
}

use core::marker::PhantomData;

// 标记：禁止锁的上下文
pub struct NoLockContext<'a> {
    _phantom: PhantomData<&'a ()>,
}

impl<'a> NoLockContext<'a> {
    // 私有构造函数，只能内部创建
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    // 执行无锁闭包
    pub fn execute<F, R>(f: F) -> R
    where
        F: FnOnce(&NoLockContext) -> R,
    {
        let ctx = NoLockContext::new();
        f(&ctx)
    }
}

// 只允许无锁操作的 API
impl NoLockContext<'_> {
    // 只提供无锁的原子操作
    pub fn atomic_load(&self, atomic: &core::sync::atomic::AtomicU32) -> u32 {
        atomic.load(core::sync::atomic::Ordering::Relaxed)
    }

    pub fn atomic_store(&self, atomic: &core::sync::atomic::AtomicU32, val: u32) {
        atomic.store(val, core::sync::atomic::Ordering::Relaxed)
    }

    // 不提供任何可能涉及锁的操作
}

pub fn watchdog_handle() {
    // nmi::handle();

    use core::sync::atomic::AtomicU32;

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    // 可以执行 - 只使用原子操作
    NoLockContext::execute(|ctx| {
        let val = ctx.atomic_load(&COUNTER);
        ctx.atomic_store(&COUNTER, val + 1);
    });
}
