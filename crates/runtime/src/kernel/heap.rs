//! The guest heap: a first-fit allocator with a 16-byte granule, over the heap's RAM in the
//! kernel's view (`0x81000000..0x84000000`), and the same allocator over the low contiguous
//! memory. Spec 8.4.
//!
//! The bookkeeping lives on the host, not in guest memory: a title that writes past the end of a
//! block cannot corrupt the allocator, and freeing an address that is not a block start is
//! recognised as such instead of reading the title's data as a header.

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;
use std::ops::Range;

use crate::mem::heap_view;

/// The allocation granule.
const GRANULE: u32 = 16;

/// The heap allocator.
// A deserialized heap has a null `base` until `adopt` puts its blocks over live memory.
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Debug, Serialize, Deserialize)]
pub struct Heap {
    /// The heap's backing memory, at guest address `start`.
    #[serde(skip, default = "std::ptr::null_mut")]
    base: *mut u8,
    /// The guest address of the first byte.
    start: u32,
    /// Free ranges: offset to length, never adjacent (neighbours are merged).
    free: BTreeMap<u32, u32>,
    /// Live blocks: offset to length.
    used: BTreeMap<u32, u32>,
}

// The heap is owned by the runtime and used under the kernel lock.
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

impl Default for Heap {
    /// An empty allocator over nothing, as a placeholder while the heap is lent to a snapshot.
    fn default() -> Self {
        Self::new(std::ptr::null_mut(), 0, 0)
    }
}

impl Heap {
    /// Takes `saved`'s blocks over this heap's memory: the allocator a snapshot kept, over this
    /// boot's memory.
    pub fn adopt(&mut self, saved: Self) {
        self.start = saved.start;
        self.free = saved.free;
        self.used = saved.used;
    }

    /// Creates the allocator over a region of `len` bytes at `base`, which the guest sees at
    /// `start` (page aligned).
    #[must_use]
    pub fn new(base: *mut u8, start: u32, len: u32) -> Self {
        let mut free = BTreeMap::new();
        let len = len / GRANULE * GRANULE;
        if len > 0 {
            free.insert(0, len);
        }
        Self {
            base,
            start,
            free,
            used: BTreeMap::new(),
        }
    }

    /// Allocates a zeroed, 16-aligned block of `size` bytes rounded up to the granule. Returns its
    /// guest address, or 0 when nothing fits.
    pub fn alloc(&mut self, size: u32) -> u32 {
        self.alloc_aligned(size, GRANULE)
    }

    /// Allocates a zeroed block of `size` bytes rounded up to the granule, aligned to `align` (a
    /// power of two; at least the granule). Returns its guest address, or 0 when nothing fits.
    pub fn alloc_aligned(&mut self, size: u32, align: u32) -> u32 {
        self.alloc_in(size, align, &(0..u32::MAX))
    }

    /// Allocates as [`alloc_aligned`](Self::alloc_aligned) does, with the whole block inside
    /// `window` (guest addresses). Returns its guest address, or 0 when nothing fits there.
    pub fn alloc_in(&mut self, size: u32, align: u32, window: &Range<u32>) -> u32 {
        let Some(size) = size.max(1).checked_next_multiple_of(GRANULE) else {
            return 0;
        };
        let align = align.max(GRANULE).next_power_of_two();
        // The window in offsets; alignment is of the guest address, and the start is page
        // aligned, so offsets align the same way.
        let low = window.start.saturating_sub(self.start);
        let high = window.end.saturating_sub(self.start);
        let fit = self.free.iter().find_map(|(&offset, &len)| {
            let start = offset.max(low).checked_next_multiple_of(align)?;
            let end = start.checked_add(size)?;
            (end <= offset + len && end <= high).then_some((offset, len, start))
        });
        let Some((offset, len, start)) = fit else {
            return 0;
        };
        self.free.remove(&offset);
        if start > offset {
            self.free.insert(offset, start - offset);
        }
        if offset + len > start + size {
            self.free.insert(start + size, offset + len - start - size);
        }
        self.used.insert(start, size);
        unsafe {
            std::ptr::write_bytes(self.base.add(start as usize), 0, size as usize);
        }
        self.start + start
    }

    /// The offset of `address`, which may be in the user-space view of the heap's RAM.
    fn offset(&self, address: u32) -> Option<u32> {
        heap_view(address).checked_sub(self.start)
    }

    /// Frees the block starting at `address`. Returns its size, or 0 when `address` is not the
    /// start of a live block (then nothing happens).
    pub fn free(&mut self, address: u32) -> u32 {
        let Some(offset) = self.offset(address) else {
            return 0;
        };
        let Some(size) = self.used.remove(&offset) else {
            return 0;
        };
        let mut start = offset;
        let mut len = size;
        if let Some((&prev, &prev_len)) = self.free.range(..offset).next_back()
            && prev + prev_len == offset
        {
            self.free.remove(&prev);
            start = prev;
            len += prev_len;
        }
        if let Some(next_len) = self.free.remove(&(offset + size)) {
            len += next_len;
        }
        self.free.insert(start, len);
        size
    }

    /// The size of the live block starting at `address`, or 0.
    #[must_use]
    pub fn size_of(&self, address: u32) -> u32 {
        self.offset(address)
            .and_then(|offset| self.used.get(&offset).copied())
            .unwrap_or(0)
    }

    /// The live block containing `address`, as `(start, size)`, the start in the allocator's
    /// own view.
    #[must_use]
    pub fn block_containing(&self, address: u32) -> Option<(u32, u32)> {
        let offset = self.offset(address)?;
        let (&start, &size) = self.used.range(..=offset).next_back()?;
        (offset < start + size).then_some((self.start + start, size))
    }

    /// Whether `address` lies inside a live block.
    #[must_use]
    pub fn contains(&self, address: u32) -> bool {
        self.block_containing(address).is_some()
    }

    /// The bytes held by live blocks.
    #[must_use]
    pub fn used_bytes(&self) -> u32 {
        self.used.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::{HEAP_BASE, user_view};

    fn heap() -> Heap {
        let len = 0x10000u32;
        let backing = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(backing, libc::MAP_FAILED);
        Heap::new(backing.cast::<u8>(), HEAP_BASE, len)
    }

    #[test]
    fn allocation_is_aligned_and_zeroed() {
        let mut h = heap();
        let a = h.alloc(100);
        assert_ne!(a, 0);
        assert_eq!(a % 16, 0);
        assert_eq!(h.size_of(a), 112); // rounded to 16
        let b = h.alloc(16);
        assert_ne!(b, 0);
        assert_eq!(b % 16, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn free_merges_neighbours() {
        let mut h = heap();
        let a = h.alloc(100);
        let b = h.alloc(100);
        let c = h.alloc(100);
        h.free(b);
        h.free(a);
        let merged = h.alloc(200);
        assert_eq!(merged, a);
        h.free(c);
        h.free(merged);
        assert_eq!(
            h.alloc(0x10000),
            HEAP_BASE,
            "everything merged back into one range"
        );
    }

    #[test]
    fn freeing_unknown_or_twice_is_a_no_op() {
        let mut h = heap();
        let a = h.alloc(64);
        assert_eq!(h.free(a), 64);
        assert_eq!(h.free(a), 0);
        assert_eq!(h.free(HEAP_BASE + 0x1000), 0);
        assert_eq!(h.free(HEAP_BASE + 2), 0);
        assert_eq!(h.free(0), 0);
    }

    #[test]
    fn an_interior_pointer_is_not_a_block() {
        let mut h = heap();
        let a = h.alloc(64);
        assert_eq!(h.free(a + 16), 0);
        assert_eq!(h.size_of(a), 64);
    }

    #[test]
    fn size_of_and_contains() {
        let mut h = heap();
        let a = h.alloc(64);
        assert_eq!(h.size_of(a), 64);
        assert!(h.contains(a));
        assert!(h.contains(a + 60));
        assert!(!h.contains(a + 64));
        assert_eq!(h.block_containing(a + 10), Some((a, 64)));
        h.free(a);
        assert_eq!(h.size_of(a), 0);
        assert!(!h.contains(a));
    }

    #[test]
    fn aligned_allocation() {
        let mut h = heap();
        let _ = h.alloc(16);
        let a = h.alloc_aligned(64, 4096);
        assert_ne!(a, 0);
        assert_eq!(a % 4096, 0);
        assert_eq!(h.size_of(a), 64);
        // The gap before the aligned block is still usable.
        assert_eq!(h.alloc(16), HEAP_BASE + 16);
    }

    #[test]
    fn exhaustion_returns_zero() {
        let mut h = heap();
        let mut blocks = Vec::new();
        loop {
            let a = h.alloc(0x1000);
            if a == 0 {
                break;
            }
            blocks.push(a);
        }
        assert_eq!(blocks.len(), 16);
        h.free(blocks[0]);
        assert_ne!(h.alloc(16), 0);
    }

    #[test]
    fn a_free_block_at_the_heap_start_stays_reusable() {
        let mut h = heap();
        let a = h.alloc(0x100);
        let _b = h.alloc(0x100);
        h.free(a);
        let big = h.alloc(0x8000);
        assert_ne!(big, 0);
        assert_eq!(h.alloc(0x100), a, "the block at the heap start was lost");
    }

    #[test]
    fn blocks_are_zeroed_when_handed_out() {
        let mut h = heap();
        let a = h.alloc(32);
        unsafe { std::ptr::write_bytes(h.base.add((a - HEAP_BASE) as usize), 0xFF, 32) };
        h.free(a);
        let b = h.alloc(32);
        assert_eq!(a, b);
        let bytes = unsafe { std::slice::from_raw_parts(h.base.add((b - HEAP_BASE) as usize), 32) };
        assert!(bytes.iter().all(|&x| x == 0));
    }

    #[test]
    fn a_window_keeps_the_block_inside_it() {
        let mut h = heap();
        let window = HEAP_BASE + 0x3000..HEAP_BASE + 0x6000;
        assert_eq!(h.alloc_in(0x1000, 0x2000, &window), HEAP_BASE + 0x4000);
        assert_eq!(h.alloc_in(0x2000, 0x1000, &window), 0, "no room left");
        assert_eq!(h.alloc_in(0x1000, 0x1000, &window), HEAP_BASE + 0x3000);
        assert_eq!(h.alloc(16), HEAP_BASE, "the rest of the heap is untouched");
        assert_eq!(
            h.alloc_in(16, 16, &(0..HEAP_BASE)),
            0,
            "a window under the heap"
        );
    }

    #[test]
    fn a_block_is_found_by_its_user_space_address_too() {
        let mut h = heap();
        let a = h.alloc(64);
        let user = user_view(a);
        assert_eq!(h.size_of(user), 64);
        assert_eq!(h.block_containing(user + 8), Some((a, 64)));
        assert_eq!(h.free(user), 64);
        assert_eq!(h.size_of(a), 0);
    }
}
