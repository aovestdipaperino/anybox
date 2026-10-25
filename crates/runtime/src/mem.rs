//! The guest address space: a 4 GiB reservation, committed on demand, with the RAM the kernel
//! hands out mapped at several addresses. Spec 6.1, 6.2.
//!
//! The heap's RAM (physical `0x01000000..0x04000000`) and the low contiguous memory (physical
//! `0x4000..0xC000`) are one shared memory object mapped at several guest addresses, so a write
//! through one view is seen through the others. The heap lives in the kernel's cached view of
//! RAM (`0x80000000 + physical`), where the console's contiguous-memory mapping puts it, so the
//! low addresses stay free for the title's own image: some titles' images (`BLiNX`: sections up
//! to `0x031BE0E0` and 38 MB more loaded on demand) cover the whole of physical
//! `0x01000000..0x04000000` as virtual addresses. The heap's RAM is also seen in user space at
//! `0x10000000 + physical`, where `NtAllocateVirtualMemory` hands it out, and both pools at the
//! write-combined GPU view `0xF0000000 + physical`. The rest of the guest space is committed on
//! demand: a bitmap tracks which pages are committed.

use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The serial number in the next heap object's name.
static NEXT_HEAP_NAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The guest base, as `anyxbox_guest_base` returns it.
static GUEST_BASE: AtomicUsize = AtomicUsize::new(0);

/// The physical RAM the heap is made of.
pub const HEAP_PHYSICAL: Range<u32> = 0x0100_0000..0x0400_0000;
/// The physical RAM under 16 MiB the kernel hands out as contiguous memory to a caller whose
/// highest acceptable address is under the heap (Direct3D's report pages, whose address must fit
/// the report method's 24-bit offset). It lies over physical page 0, where Direct3D parks its
/// push-buffer jump, and under the title's image base (`0x10000`), in whole 16 KiB host pages.
pub const LOW_PHYSICAL: Range<u32> = 0x4000..0xC000;
/// The cached kernel view of RAM: virtual `0x80000000 + physical`.
pub const KERNEL_ALIAS: u32 = 0x8000_0000;
/// The user-space view of the heap's RAM: virtual `0x10000000 + physical`, below `0x80000000` as
/// a console's user-space allocations are. Direct3D tells a vertex-shader handle from an FVF code
/// by its value, so a title's virtual allocations in the kernel's view break every shader.
pub const USER_ALIAS: u32 = 0x1000_0000;
/// The write-combined GPU view of RAM: virtual `0xF0000000 + physical`.
pub const GPU_ALIAS: u32 = 0xF000_0000;
/// The heap's first address: the kernel's view of its RAM.
pub const HEAP_BASE: u32 = KERNEL_ALIAS + HEAP_PHYSICAL.start;
/// One past the heap's end.
pub const HEAP_END: u32 = KERNEL_ALIAS + HEAP_PHYSICAL.end;
/// The low contiguous memory's first address, in the kernel's view.
pub const LOW_BASE: u32 = KERNEL_ALIAS + LOW_PHYSICAL.start;
/// One past the low contiguous memory's end.
pub const LOW_END: u32 = KERNEL_ALIAS + LOW_PHYSICAL.end;
/// The GPU register window.
pub const MMIO_BASE: u32 = 0xFD00_0000;
/// One past the GPU register window.
pub const MMIO_END: u32 = 0xFE00_0000;

/// The guest ranges backed by the shared memory object, with the object offset of each: the
/// heap in its three views, then the low contiguous memory in two.
const SHARED: [(u32, u32, usize); 5] = {
    let heap = HEAP_PHYSICAL.end - HEAP_PHYSICAL.start;
    let low = LOW_PHYSICAL.end - LOW_PHYSICAL.start;
    [
        (HEAP_BASE, heap, 0),
        (USER_ALIAS + HEAP_PHYSICAL.start, heap, 0),
        (GPU_ALIAS + HEAP_PHYSICAL.start, heap, 0),
        (LOW_BASE, low, heap as usize),
        (GPU_ALIAS + LOW_PHYSICAL.start, low, heap as usize),
    ]
};

/// The user-space view of `heap`, an address in the heap.
#[must_use]
pub const fn user_view(heap: u32) -> u32 {
    heap - KERNEL_ALIAS + USER_ALIAS
}

/// The heap address behind `address` when it is in the user-space view of the heap's RAM, else
/// `address`.
#[must_use]
pub const fn heap_view(address: u32) -> u32 {
    if address >= USER_ALIAS + HEAP_PHYSICAL.start && address < USER_ALIAS + HEAP_PHYSICAL.end {
        address - USER_ALIAS + KERNEL_ALIAS
    } else {
        address
    }
}

/// The host page size (16 KiB on Apple silicon).
pub const HOST_PAGE: usize = 16 * 1024;

/// The number of pages in the 4 GiB guest space.
const PAGE_COUNT: usize = (1usize << 32) / HOST_PAGE;

/// The guest memory.
pub struct Memory {
    /// The reservation's address.
    base: *mut u8,
    /// The shared memory object backing the heap, the low contiguous memory and their views.
    heap_fd: i32,
    /// The size of the heap (each of its views is this large).
    heap_len: usize,
    /// Which non-heap pages are committed.
    committed: Mutex<Vec<bool>>,
}

impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory")
            .field("base", &self.base)
            .field("heap_fd", &self.heap_fd)
            .field("heap_len", &self.heap_len)
            .field("committed", &self.committed.lock().unwrap().len())
            .finish()
    }
}

// The guest memory is owned by the runtime and shared between host threads through raw pointers;
// the kernel lock serialises access, and the reservation lives for the process's life.
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

impl Memory {
    /// Reserves the guest space and the heap backing. Returns `None` when a reservation fails.
    ///
    /// # Panics
    ///
    /// Panics if the heap or an alias cannot be mapped.
    #[must_use]
    pub fn new() -> Option<Self> {
        let len = 1usize << 32;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return None;
        }
        let heap_len = (HEAP_END - HEAP_BASE) as usize;
        let object_len = heap_len + (LOW_END - LOW_BASE) as usize;
        // A name of our own, created exclusively: two guest spaces (two apps running, or a reset
        // overlapping the old memory) must never share one heap object.
        let serial = NEXT_HEAP_NAME.fetch_add(1, Ordering::Relaxed);
        let name = std::ffi::CString::new(format!("/anyxbox-{}-{serial}", std::process::id()))
            .expect("the name has no NUL");
        let heap_fd = unsafe {
            libc::shm_open(
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600,
            )
        };
        assert!(
            heap_fd >= 0,
            "shm_open failed: {}",
            std::io::Error::last_os_error()
        );
        // Unlink immediately so the object vanishes when the process exits.
        unsafe {
            libc::shm_unlink(name.as_ptr());
        }
        let result =
            unsafe { libc::ftruncate(heap_fd, libc::off_t::try_from(object_len).unwrap()) };
        assert_eq!(result, 0, "ftruncate failed");
        let base = base.cast::<u8>();
        // Map every view of the RAM the kernel hands out from the same object: guest
        // 0x81000000, 0x11000000 and 0xF1000000 are one byte, so the kernel's structures below
        // 0x81000000 stay its own.
        for (address, len, offset) in SHARED {
            let at = unsafe { base.add(address as usize) };
            let mapped = unsafe {
                libc::mmap(
                    at.cast(),
                    len as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    heap_fd,
                    libc::off_t::try_from(offset).unwrap(),
                )
            };
            assert_ne!(
                mapped,
                libc::MAP_FAILED,
                "a view of RAM could not be mapped"
            );
        }
        Some(Self {
            base,
            heap_fd,
            heap_len,
            committed: Mutex::new(vec![false; PAGE_COUNT]),
        })
    }

    /// The guest base.
    #[must_use]
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// The page index of a guest address.
    fn page(address: u32) -> usize {
        (address as usize) / HOST_PAGE
    }

    /// Commits the page containing `address` (rounded down to the host page) read-write,
    /// zero-filled. Committing an already committed page keeps its contents. Returns whether the
    /// address is inside the reservation.
    ///
    /// # Panics
    ///
    /// Panics if the committed-pages lock is poisoned.
    pub fn commit(&self, address: u32) -> bool {
        let at = unsafe { self.base.add(address as usize) };
        let page = (at as usize) & !(HOST_PAGE - 1);
        let result = unsafe {
            libc::mprotect(
                page as *mut libc::c_void,
                HOST_PAGE,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        if result == 0 {
            let mut committed = self.committed.lock().unwrap();
            committed[Self::page(address)] = true;
        }
        result == 0
    }

    /// Commits every page that `address..address + len` touches. Returns whether all of them are
    /// inside the reservation.
    ///
    /// # Panics
    ///
    /// Panics if the committed-pages lock is poisoned.
    pub fn commit_range(&self, address: u32, len: u32) -> bool {
        if len == 0 {
            return true;
        }
        let first = Self::page(address);
        let last = Self::page(address.saturating_add(len - 1));
        (first..=last).all(|page| self.commit(u32::try_from(page * HOST_PAGE).unwrap()))
    }

    /// The guest addresses of the committed pages outside the shared views of RAM, which a
    /// snapshot keeps page by page (the shared RAM it keeps once, whole).
    ///
    /// # Panics
    ///
    /// Panics if the committed-pages lock is poisoned.
    #[must_use]
    pub fn committed_pages(&self) -> Vec<u32> {
        let committed = self.committed.lock().unwrap();
        committed
            .iter()
            .enumerate()
            .filter(|&(_, &on)| on)
            .filter_map(|(page, _)| u32::try_from(page * HOST_PAGE).ok())
            .filter(|&address| {
                !SHARED
                    .iter()
                    .any(|&(start, len, _)| address >= start && address - start < len)
            })
            .collect()
    }

    /// Whether the page containing `address` is committed.
    ///
    /// # Panics
    ///
    /// Panics if the committed-pages lock is poisoned.
    #[must_use]
    pub fn is_committed(&self, address: u32) -> bool {
        let page = Self::page(address);
        if page >= PAGE_COUNT {
            return false;
        }
        if SHARED
            .iter()
            .any(|&(start, len, _)| address >= start && address - start < len)
        {
            return true;
        }
        self.committed.lock().unwrap()[page]
    }

    /// Reads `len` bytes at `address` into `out`. Returns false when any page is uncommitted.
    ///
    /// # Panics
    ///
    /// Panics if the committed-pages lock is poisoned.
    pub fn read(&self, address: u32, out: &mut [u8]) -> bool {
        let start = address as usize;
        let end = start.saturating_add(out.len());
        if end > 1usize << 32 {
            return false;
        }
        let first = start / HOST_PAGE;
        let last = (end - 1) / HOST_PAGE;
        for page in first..=last {
            if !self.is_committed(u32::try_from(page * HOST_PAGE).unwrap()) {
                return false;
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.base.add(start), out.as_mut_ptr(), out.len());
        }
        true
    }

    /// The [`fingerprint`] of `len` bytes at `address`, read where they are; `None` when any
    /// page is uncommitted.
    ///
    /// # Panics
    ///
    /// Panics if the committed-pages lock is poisoned.
    #[must_use]
    pub fn fingerprint(&self, address: u32, len: u32, seed: u64) -> Option<u64> {
        let start = address as usize;
        let end = start.saturating_add(len as usize);
        if end > 1usize << 32 {
            return None;
        }
        if len > 0 {
            for page in start / HOST_PAGE..=(end - 1) / HOST_PAGE {
                if !self.is_committed(u32::try_from(page * HOST_PAGE).unwrap()) {
                    return None;
                }
            }
        }
        // SAFETY: the range is inside the reserved guest space and committed; guest threads may
        // write it meanwhile, as they may during `read`, which only makes the fingerprint stale.
        let bytes = unsafe { std::slice::from_raw_parts(self.base.add(start), len as usize) };
        Some(fingerprint(bytes, seed))
    }

    /// Writes `data` at `address`. Returns false when any page is uncommitted.
    ///
    /// # Panics
    ///
    /// Panics if the committed-pages lock is poisoned.
    pub fn write(&self, address: u32, data: &[u8]) -> bool {
        let start = address as usize;
        let end = start.saturating_add(data.len());
        if end > 1usize << 32 {
            return false;
        }
        let first = start / HOST_PAGE;
        let last = (end - 1) / HOST_PAGE;
        for page in first..=last {
            if !self.is_committed(u32::try_from(page * HOST_PAGE).unwrap()) {
                return false;
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(start), data.len());
        }
        true
    }

    /// The heap's backing memory, for the allocator.
    #[must_use]
    pub fn heap(&self) -> *mut u8 {
        unsafe { self.base.add(HEAP_BASE as usize) }
    }

    /// The low contiguous memory's backing memory, for its allocator.
    #[must_use]
    pub fn low(&self) -> *mut u8 {
        unsafe { self.base.add(LOW_BASE as usize) }
    }

    /// The heap's length.
    #[must_use]
    pub fn heap_len(&self) -> usize {
        self.heap_len
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.cast(), 1usize << 32);
            libc::close(self.heap_fd);
        }
    }
}

/// Sets the guest base for `anyxbox_guest_base`.
/// A fast fingerprint of `bytes`, from `seed`: a word-at-a-time multiplicative hash, enough to
/// tell whether texture bytes changed between frames.
#[must_use]
pub fn fingerprint(bytes: &[u8], seed: u64) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let (words, rest) = bytes.as_chunks::<8>();
    let mut h = seed ^ (bytes.len() as u64).wrapping_mul(K);
    for word in words {
        h = (h.rotate_left(5) ^ u64::from_le_bytes(*word)).wrapping_mul(K);
    }
    for &byte in rest {
        h = (h.rotate_left(5) ^ u64::from(byte)).wrapping_mul(K);
    }
    h ^ (h >> 29)
}

pub fn set_guest_base(base: *mut u8) {
    GUEST_BASE.store(base as usize, Ordering::Relaxed);
}

/// The current guest base, as the C ABI reads it.
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_guest_base() -> *mut u8 {
    GUEST_BASE.load(Ordering::Relaxed) as *mut u8
}

/// The physical address of a guest address: the low 26 bits (64 MiB). Spec 6.2.
#[must_use]
pub const fn physical(address: u32) -> u32 {
    address & 0x03FF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reservation_commits_on_demand() {
        let memory = Memory::new().unwrap();
        assert!(!memory.is_committed(0x0001_0000));
        assert!(memory.commit(0x0001_0000));
        assert!(memory.is_committed(0x0001_0000));
        let mut buf = [0u8; 4];
        assert!(memory.read(0x0001_0000, &mut buf));
        assert_eq!(buf, [0; 4]);
        assert!(memory.write(0x0001_0000, &[1, 2, 3, 4]));
        assert!(memory.read(0x0001_0000, &mut buf));
        assert_eq!(buf, [1, 2, 3, 4]);
        // Committing again keeps the contents.
        assert!(memory.commit(0x0001_0000));
        assert!(memory.read(0x0001_0000, &mut buf));
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn a_fingerprint_in_place_matches_one_of_the_bytes_read() {
        let memory = Memory::new().unwrap();
        assert!(memory.commit(0x0001_0000));
        let bytes: Vec<u8> = (0..=255).collect();
        assert!(memory.write(0x0001_0003, &bytes));
        assert_eq!(
            memory.fingerprint(0x0001_0003, 256, 7),
            Some(fingerprint(&bytes, 7))
        );
        assert_ne!(
            fingerprint(&bytes, 7),
            fingerprint(&bytes, 8),
            "the seed counts"
        );
        assert_ne!(fingerprint(&bytes, 7), fingerprint(&bytes[1..], 7));
        assert_eq!(memory.fingerprint(0x0002_0000, 4, 7), None, "uncommitted");
    }

    #[test]
    fn the_views_of_the_heap_share_memory() {
        let memory = Memory::new().unwrap();
        assert!(memory.write(HEAP_BASE, &[0xAA, 0xBB]));
        let mut buf = [0u8; 2];
        assert!(memory.read(USER_ALIAS + HEAP_PHYSICAL.start, &mut buf));
        assert_eq!(buf, [0xAA, 0xBB]);
        assert!(memory.read(GPU_ALIAS + HEAP_PHYSICAL.start, &mut buf));
        assert_eq!(buf, [0xAA, 0xBB]);
        assert!(memory.write(GPU_ALIAS + HEAP_PHYSICAL.start, &[0xCC, 0xDD]));
        assert!(memory.read(HEAP_BASE, &mut buf));
        assert_eq!(buf, [0xCC, 0xDD]);
    }

    #[test]
    fn the_low_addresses_the_heap_used_to_take_are_left_to_the_title() {
        let memory = Memory::new().unwrap();
        // BLiNX's image has sections at these addresses.
        assert!(!memory.is_committed(HEAP_PHYSICAL.start));
        assert!(memory.commit_range(0x0200_0000, 0x10_0000));
        assert!(memory.write(0x0200_0000, &[1, 2, 3, 4]));
        let mut buf = [0u8; 4];
        assert!(memory.read(KERNEL_ALIAS + 0x0200_0000, &mut buf));
        assert_eq!(buf, [0; 4], "the image is not the heap");
    }

    #[test]
    fn the_low_contiguous_memory_has_its_cached_and_gpu_views() {
        let memory = Memory::new().unwrap();
        assert!(memory.is_committed(LOW_BASE));
        assert!(memory.is_committed(LOW_END - 1));
        assert!(!memory.is_committed(LOW_END));
        assert!(memory.write(LOW_BASE + 0x10, &[5, 6]));
        let mut buf = [0u8; 2];
        assert!(memory.read(GPU_ALIAS + LOW_PHYSICAL.start + 0x10, &mut buf));
        assert_eq!(buf, [5, 6]);
        // It is not the heap's RAM.
        assert!(memory.read(HEAP_BASE + 0x10, &mut buf));
        assert_eq!(buf, [0, 0]);
    }

    #[test]
    fn user_and_heap_views_convert() {
        assert_eq!(user_view(HEAP_BASE), 0x1100_0000);
        assert_eq!(user_view(HEAP_END - 1), 0x13FF_FFFF);
        assert_eq!(heap_view(0x1100_0000), HEAP_BASE);
        assert_eq!(heap_view(0x13FF_FFFF), HEAP_END - 1);
        assert_eq!(heap_view(0x1400_0000), 0x1400_0000);
        assert_eq!(heap_view(0x10FF_FFFF), 0x10FF_FFFF);
        assert_eq!(heap_view(HEAP_BASE + 4), HEAP_BASE + 4);
        // Both views are the same physical address.
        assert_eq!(
            physical(0x1123_4560),
            physical(user_view(HEAP_BASE + 0x23_4560))
        );
        assert_eq!(physical(HEAP_BASE + 0x23_4560), 0x0123_4560);
    }

    #[test]
    fn uncommitted_reads_and_writes_fail() {
        let memory = Memory::new().unwrap();
        let mut buf = [0u8; 4];
        assert!(!memory.read(0x0000_1000, &mut buf));
        assert!(!memory.write(0x0000_1000, &[1, 2, 3, 4]));
        // Reads and writes that cross an uncommitted page fail too.
        assert!(memory.commit(0x0000_0000));
        let mut big = [0u8; 8];
        assert!(!memory.read(0x0000_3FFC, &mut big));
        assert!(!memory.write(0x0000_3FFC, &[1, 2, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn the_views_sit_at_the_physical_addresses_plus_their_base() {
        let memory = Memory::new().unwrap();
        assert!(memory.write(0x8123_4560, &[1, 2, 3, 4]));
        let mut buf = [0u8; 4];
        assert!(memory.read(0x1123_4560, &mut buf));
        assert_eq!(buf, [1, 2, 3, 4]);
        assert!(memory.read(0xF123_4560, &mut buf));
        assert_eq!(buf, [1, 2, 3, 4]);
        // The kernel's own structures below 0x81000000 are not heap memory.
        assert!(memory.commit(0x8004_0000));
        assert!(memory.write(0x8004_0028, &[9, 9, 9, 9]));
        assert!(memory.read(0x8104_0028, &mut buf));
        assert_eq!(buf, [0, 0, 0, 0]);
    }

    #[test]
    fn a_range_commits_every_page_it_touches() {
        let memory = Memory::new().unwrap();
        // Unaligned start, and a length that fits in one page's worth of bytes but spans two.
        let start = 0x0001_3000;
        let len = 0x2000;
        assert!(memory.commit_range(start, len));
        assert!(memory.is_committed(start));
        assert!(memory.is_committed(start + len - 1));
        assert!(memory.write(start, &vec![0xAB; len as usize]));
    }

    #[test]
    fn physical_addresses_are_the_low_26_bits() {
        assert_eq!(physical(0x0001_0000), 0x0001_0000);
        assert_eq!(physical(0x8000_0000), 0);
        assert_eq!(physical(0xF000_1234), 0x1234);
    }

    #[test]
    fn the_heap_is_always_committed() {
        let memory = Memory::new().unwrap();
        assert!(memory.is_committed(HEAP_BASE));
        assert!(memory.is_committed(USER_ALIAS + HEAP_PHYSICAL.start));
        assert!(memory.is_committed(GPU_ALIAS + HEAP_PHYSICAL.end - 1));
        assert!(!memory.is_committed(KERNEL_ALIAS));
        assert!(memory.is_committed(HEAP_END - 1));
        assert!(!memory.is_committed(HEAP_END));
    }
}
