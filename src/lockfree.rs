//! Lock-free data structures for NMI-safe operations.
//!
//! These data structures are designed to be used in NMI/interrupt context
//! where no locks can be held. They use only atomic operations.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

// =============================================================================
// Lock-free SPSC Ring Buffer
// =============================================================================

/// A lock-free Single-Producer Single-Consumer ring buffer.
///
/// This is safe for one producer (e.g., NMI handler) and one consumer
/// (e.g., normal context) to use concurrently without locks.
///
/// # Type Parameters
/// * `T` - Element type (must be Copy for simplicity)
/// * `N` - Buffer capacity (must be power of 2)
pub struct SpscRingBuffer<T: Copy, const N: usize> {
    /// Buffer storage.
    buffer: UnsafeCell<[MaybeUninit<T>; N]>,
    /// Write position (producer).
    head: AtomicUsize,
    /// Read position (consumer).
    tail: AtomicUsize,
}

impl<T: Copy, const N: usize> SpscRingBuffer<T, N> {
    /// Mask for wrapping indices (N must be power of 2).
    const MASK: usize = N - 1;

    /// Create a new empty ring buffer.
    ///
    /// # Panics
    /// Panics if N is not a power of 2.
    pub const fn new() -> Self {
        // Note: const assert that N is power of 2
        // This will be checked at compile time
        assert!(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

        Self {
            buffer: UnsafeCell::new(unsafe { MaybeUninit::uninit().assume_init() }),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Try to push an item (producer side).
    ///
    /// Returns `Ok(())` if successful, `Err(item)` if buffer is full.
    #[inline]
    pub fn push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // Check if buffer is full
        if head.wrapping_sub(tail) >= N {
            return Err(item);
        }

        // Write the item
        unsafe {
            let slot = (*self.buffer.get()).get_unchecked_mut(head & Self::MASK);
            slot.write(item);
        }

        // Publish the write
        self.head.store(head.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Try to pop an item (consumer side).
    ///
    /// Returns `Some(item)` if available, `None` if buffer is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // Check if buffer is empty
        if tail == head {
            return None;
        }

        // Read the item
        let item = unsafe {
            let slot = (*self.buffer.get()).get_unchecked(tail & Self::MASK);
            slot.assume_init_read()
        };

        // Publish the read
        self.tail.store(tail.wrapping_add(1), Ordering::Release);

        Some(item)
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail == head
    }

    /// Check if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail) >= N
    }

    /// Get the number of items in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Get the capacity of the buffer.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Clear all items from the buffer (consumer side only).
    pub fn clear(&self) {
        while self.pop().is_some() {}
    }
}

impl<T: Copy, const N: usize> Default for SpscRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: SpscRingBuffer is safe to share between threads when used correctly
// (one producer, one consumer)
unsafe impl<T: Copy + Send, const N: usize> Send for SpscRingBuffer<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for SpscRingBuffer<T, N> {}

// =============================================================================
// Atomic Bitmap
// =============================================================================

/// An atomic bitmap for tracking set bits.
///
/// Useful for tracking which CPUs have pending events, etc.
pub struct AtomicBitmap<const N: usize> {
    /// Bitmap storage (each usize holds 64 bits on 64-bit platforms).
    words: [AtomicUsize; N],
}

impl<const N: usize> AtomicBitmap<N> {
    /// Bits per word.
    const BITS_PER_WORD: usize = core::mem::size_of::<usize>() * 8;

    /// Total capacity in bits.
    pub const CAPACITY: usize = N * Self::BITS_PER_WORD;

    /// Create a new empty bitmap.
    pub const fn new() -> Self {
        const ZERO: AtomicUsize = AtomicUsize::new(0);
        Self { words: [ZERO; N] }
    }

    /// Set a bit.
    #[inline]
    pub fn set(&self, index: usize) {
        if index >= Self::CAPACITY {
            return;
        }
        let word_idx = index / Self::BITS_PER_WORD;
        let bit_idx = index % Self::BITS_PER_WORD;
        self.words[word_idx].fetch_or(1 << bit_idx, Ordering::AcqRel);
    }

    /// Clear a bit.
    #[inline]
    pub fn clear(&self, index: usize) {
        if index >= Self::CAPACITY {
            return;
        }
        let word_idx = index / Self::BITS_PER_WORD;
        let bit_idx = index % Self::BITS_PER_WORD;
        self.words[word_idx].fetch_and(!(1 << bit_idx), Ordering::AcqRel);
    }

    /// Test if a bit is set.
    #[inline]
    pub fn test(&self, index: usize) -> bool {
        if index >= Self::CAPACITY {
            return false;
        }
        let word_idx = index / Self::BITS_PER_WORD;
        let bit_idx = index % Self::BITS_PER_WORD;
        (self.words[word_idx].load(Ordering::Acquire) & (1 << bit_idx)) != 0
    }

    /// Test and set a bit atomically.
    ///
    /// Returns the previous value of the bit.
    #[inline]
    pub fn test_and_set(&self, index: usize) -> bool {
        if index >= Self::CAPACITY {
            return false;
        }
        let word_idx = index / Self::BITS_PER_WORD;
        let bit_idx = index % Self::BITS_PER_WORD;
        let mask = 1 << bit_idx;
        let old = self.words[word_idx].fetch_or(mask, Ordering::AcqRel);
        (old & mask) != 0
    }

    /// Test and clear a bit atomically.
    ///
    /// Returns the previous value of the bit.
    #[inline]
    pub fn test_and_clear(&self, index: usize) -> bool {
        if index >= Self::CAPACITY {
            return false;
        }
        let word_idx = index / Self::BITS_PER_WORD;
        let bit_idx = index % Self::BITS_PER_WORD;
        let mask = 1 << bit_idx;
        let old = self.words[word_idx].fetch_and(!mask, Ordering::AcqRel);
        (old & mask) != 0
    }

    /// Find the first set bit.
    ///
    /// Returns the index of the first set bit, or None if all bits are clear.
    pub fn find_first_set(&self) -> Option<usize> {
        for (word_idx, word) in self.words.iter().enumerate() {
            let val = word.load(Ordering::Acquire);
            if val != 0 {
                let bit_idx = val.trailing_zeros() as usize;
                return Some(word_idx * Self::BITS_PER_WORD + bit_idx);
            }
        }
        None
    }

    /// Count the number of set bits.
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }

    /// Check if all bits are clear.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| w.load(Ordering::Acquire) == 0)
    }

    /// Clear all bits.
    pub fn clear_all(&self) {
        for word in &self.words {
            word.store(0, Ordering::Release);
        }
    }

    /// Get raw word at index.
    #[inline]
    pub fn word(&self, word_idx: usize) -> usize {
        if word_idx < N {
            self.words[word_idx].load(Ordering::Acquire)
        } else {
            0
        }
    }
}

impl<const N: usize> Default for AtomicBitmap<N> {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: AtomicBitmap uses only atomic operations
unsafe impl<const N: usize> Send for AtomicBitmap<N> {}
unsafe impl<const N: usize> Sync for AtomicBitmap<N> {}

// =============================================================================
// Atomic Counter with Overflow Detection
// =============================================================================

/// An atomic counter that can detect overflow.
///
/// Useful for sequence numbers and generation counters.
pub struct AtomicSequence {
    value: AtomicU64,
}

use core::sync::atomic::AtomicU64;

impl AtomicSequence {
    /// Create a new sequence starting at 0.
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Create a new sequence with initial value.
    pub const fn with_initial(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
        }
    }

    /// Get the current value.
    #[inline]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Increment and return new value.
    #[inline]
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
    }

    /// Compare and swap.
    #[inline]
    pub fn compare_exchange(&self, current: u64, new: u64) -> Result<u64, u64> {
        self.value
            .compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire)
    }

    /// Reset to zero.
    #[inline]
    pub fn reset(&self) {
        self.value.store(0, Ordering::Release);
    }
}

impl Default for AtomicSequence {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spsc_basic() {
        let buf: SpscRingBuffer<u32, 4> = SpscRingBuffer::new();

        assert!(buf.is_empty());
        assert!(!buf.is_full());

        buf.push(1).unwrap();
        buf.push(2).unwrap();
        buf.push(3).unwrap();
        buf.push(4).unwrap();

        assert!(buf.is_full());
        assert!(buf.push(5).is_err());

        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
        assert_eq!(buf.pop(), None);

        assert!(buf.is_empty());
    }

    #[test]
    fn test_bitmap_basic() {
        let bitmap: AtomicBitmap<2> = AtomicBitmap::new();

        assert!(bitmap.is_empty());
        assert!(!bitmap.test(0));

        bitmap.set(0);
        assert!(bitmap.test(0));
        assert!(!bitmap.test(1));

        bitmap.set(63);
        bitmap.set(64);

        assert_eq!(bitmap.count_ones(), 3);
        assert_eq!(bitmap.find_first_set(), Some(0));

        bitmap.clear(0);
        assert_eq!(bitmap.find_first_set(), Some(63));
    }

    #[test]
    fn test_sequence() {
        let seq = AtomicSequence::new();

        assert_eq!(seq.get(), 0);
        assert_eq!(seq.increment(), 1);
        assert_eq!(seq.increment(), 2);
        assert_eq!(seq.get(), 2);

        seq.reset();
        assert_eq!(seq.get(), 0);
    }
}
