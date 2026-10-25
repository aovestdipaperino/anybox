//! Memory manager exports (spec 8.4).

use super::value;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};
use crate::mem::{HEAP_PHYSICAL, KERNEL_ALIAS, LOW_PHYSICAL, heap_view, user_view};

const PAGE: u32 = 0x1000;
const PAGE_NOACCESS: u32 = 1;
const PAGE_READWRITE: u32 = 4;
const MEM_COMMIT: u32 = 0x1000;
const MEM_DECOMMIT: u32 = 0x4000;
const MEM_FREE: u32 = 0x1_0000;
const MEM_PRIVATE: u32 = 0x2_0000;

/// A page-aligned block of `size` rounded up to a page.
fn alloc_pages(k: &mut Kernel, size: u32, align: u32) -> u32 {
    match size.checked_next_multiple_of(PAGE) {
        Some(size) => k.heap.alloc_aligned(size.max(PAGE), align),
        None => 0,
    }
}

pub fn allocate_contiguous(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(alloc_pages(k, c.arg(0), PAGE))
}

/// `MmAllocateContiguousMemoryEx(size, lowest, highest, align, protect)`: a block whose
/// physical pages all lie between the two acceptable addresses, from the heap or, for a caller
/// that wants memory under the heap (Direct3D's report pages, under 16 MiB), from the low
/// contiguous memory. When neither has room the block comes from anywhere in the heap.
pub fn allocate_contiguous_ex(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (lowest, highest) = (c.arg(1), c.arg(2));
    let align = c
        .arg(3)
        .max(PAGE)
        .checked_next_power_of_two()
        .unwrap_or(PAGE);
    let Some(size) = c.arg(0).max(1).checked_next_multiple_of(PAGE) else {
        return value(0);
    };
    for (physical, low) in [(HEAP_PHYSICAL, false), (LOW_PHYSICAL, true)] {
        let start = lowest.max(physical.start);
        let end = u32::try_from((u64::from(highest) + 1).min(u64::from(physical.end)))
            .unwrap_or(physical.end);
        if start >= end {
            continue;
        }
        let window = KERNEL_ALIAS + start..KERNEL_ALIAS + end;
        let pool = if low { &mut k.low_heap } else { &mut k.heap };
        let block = pool.alloc_in(size, align, &window);
        if block != 0 {
            return value(block);
        }
    }
    value(alloc_pages(k, size, align))
}

/// Frees a block of the heap or the low contiguous memory; returns its size, or 0.
fn free_block(k: &mut Kernel, address: u32) -> u32 {
    match k.heap.free(address) {
        0 => k.low_heap.free(address),
        size => size,
    }
}

/// The live block of the heap or the low contiguous memory containing `address`.
fn block_containing(k: &Kernel, address: u32) -> Option<(u32, u32)> {
    k.heap
        .block_containing(address)
        .or_else(|| k.low_heap.block_containing(address))
}

pub fn free_contiguous(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    free_block(k, c.arg(0));
    value(0)
}

/// Returns the freed size or 0.
pub fn free_system(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(free_block(k, c.arg(0)))
}

/// `MmClaimGpuInstanceMemory(bytes, *skip)`.
pub fn claim_gpu_instance_memory(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let block = alloc_pages(k, c.arg(0), PAGE);
    k.w32_opt(c.arg(1), 0)?;
    value(block)
}

pub fn get_physical_address(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(crate::mem::physical(c.arg(0)))
}

pub fn query_allocation_size(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(block_containing(k, c.arg(0)).map_or(0, |(_, size)| size))
}

pub fn query_address_protect(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(if k.memory.is_committed(c.arg(0)) {
        PAGE_READWRITE
    } else {
        PAGE_NOACCESS
    })
}

/// `MmQueryStatistics(*s)`: `s->Length` must be 36.
pub fn query_statistics(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(0);
    if k.r32(at)? != 36 {
        return value(status::INVALID_PARAMETER);
    }
    let used = k.heap.used_bytes() + k.low_heap.used_bytes();
    let mut fields = [0u32; 8];
    fields[0] = 0x4000;
    fields[1] = 0x4000u32.saturating_sub(used / PAGE);
    fields[2] = used;
    fields[3] = used;
    for (i, f) in fields.iter().enumerate() {
        k.w32(at + 4 + 4 * u32::try_from(i).unwrap_or(0), *f)?;
    }
    value(status::SUCCESS)
}

pub fn is_address_valid(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from(k.memory.is_committed(c.arg(0))))
}

pub fn map_io_space(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(c.arg(0))
}

/// `MmCreateKernelStack(n, dbg)`: returns the top of a page-rounded block, or 0.
pub fn create_kernel_stack(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let size = c.arg(0).next_multiple_of(PAGE);
    let block = alloc_pages(k, size, PAGE);
    value(if block == 0 { 0 } else { block + size })
}

/// `MmDeleteKernelStack(base, limit)`: frees the block at `limit`.
pub fn delete_kernel_stack(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    free_block(k, c.arg(1));
    value(0)
}

/// `NtAllocateVirtualMemory(*base, zerobits, *size, type, protect)`. New allocations are handed
/// out in the user-space view of the heap's RAM, below `0x80000000` as on a console: Direct3D
/// tells a vertex-shader handle (an address) from an FVF code by its value, and takes a handle
/// in the kernel's view for an FVF code.
pub fn allocate_virtual(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (base_ptr, size_ptr) = (c.arg(0), c.arg(2));
    let base = k.r32(base_ptr)?;
    let size = k.r32(size_ptr)?;
    if base == 0 {
        let rounded = size.next_multiple_of(PAGE);
        let block = alloc_pages(k, rounded, PAGE);
        if block == 0 {
            return value(status::NO_MEMORY);
        }
        k.w32(base_ptr, user_view(block))?;
        k.w32(size_ptr, rounded)?;
        return value(status::SUCCESS);
    }
    let Some((start, len)) = k.heap.block_containing(base) else {
        return Ok(CallResult::Unsupported(format!(
            "committing {base:#010x}, which is not inside an allocation"
        )));
    };
    // The block is in the kernel's view; `base` may be in either.
    if u64::from(heap_view(base)) + u64::from(size) > u64::from(start) + u64::from(len) {
        return value(status::CONFLICTING_ADDRESSES);
    }
    let low = base & !(PAGE - 1);
    k.w32(base_ptr, low)?;
    k.w32(size_ptr, (base + size - low).next_multiple_of(PAGE))?;
    value(status::SUCCESS)
}

/// `NtFreeVirtualMemory(*base, *size, type)`.
pub fn free_virtual(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if c.arg(2) & MEM_DECOMMIT != 0 {
        return value(status::SUCCESS);
    }
    let base = k.r32(c.arg(0))?;
    let size = free_block(k, base);
    if size == 0 {
        return value(status::INVALID_PARAMETER);
    }
    k.w32_opt(c.arg(1), size)?;
    value(status::SUCCESS)
}

/// `NtQueryVirtualMemory(a, *info)`: a `MEMORY_BASIC_INFORMATION`.
pub fn query_virtual(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let address = c.arg(0);
    let page = address & !(PAGE - 1);
    let (base, size, state, protect) = if let Some((start, len)) = block_containing(k, address) {
        // In the view the caller asked about.
        let start = if heap_view(address) == address {
            start
        } else {
            user_view(start)
        };
        (start, len, MEM_COMMIT, PAGE_READWRITE)
    } else if k.memory.is_committed(address) {
        (page, PAGE, MEM_COMMIT, PAGE_READWRITE)
    } else {
        (page, PAGE, MEM_FREE, PAGE_NOACCESS)
    };
    let info = [base, base, protect, size, state, protect, MEM_PRIVATE];
    for (i, v) in info.iter().enumerate() {
        k.w32(c.arg(1) + 4 * u32::try_from(i).unwrap_or(0), *v)?;
    }
    value(status::SUCCESS)
}

/// `NtProtectVirtualMemory(*base, *size, new, *old)`.
pub fn protect_virtual(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w32_opt(c.arg(3), PAGE_READWRITE)?;
    value(status::SUCCESS)
}

/// `XeLoadSection(*section)`: on the first reference, commit and copy the section and count the
/// shared pages.
pub fn load_section(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let header = c.arg(0);
    let address = k.r32(header + 4)?;
    let size = k.r32(header + 8)?;
    let Some(raw) = k
        .sections
        .iter()
        .find(|(a, _)| *a == address)
        .map(|(_, r)| r.clone())
    else {
        return value(status::INVALID_PARAMETER);
    };
    let count = k.r32(header + 24)?;
    if count == 0 {
        k.memory
            .commit_range(address, size.max(u32::try_from(raw.len()).unwrap_or(0)));
        let mut bytes = raw;
        bytes.resize(bytes.len().max(size as usize), 0);
        k.write(address, &bytes)?;
        for counter in [k.r32(header + 28)?, k.r32(header + 32)?] {
            let n = k.r16(counter)?;
            k.w16(counter, n.wrapping_add(1))?;
        }
    }
    k.w32(header + 24, count + 1)?;
    value(status::SUCCESS)
}

/// `XeUnloadSection(*section)`: pages stay mapped.
pub fn unload_section(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let header = c.arg(0);
    let count = k.r32(header + 24)?;
    if count == 0 {
        return value(status::INVALID_PARAMETER);
    }
    k.w32(header + 24, count - 1)?;
    if count == 1 {
        for counter in [k.r32(header + 28)?, k.r32(header + 32)?] {
            let n = k.r16(counter)?;
            k.w16(counter, n.wrapping_sub(1))?;
        }
    }
    value(status::SUCCESS)
}

/// `MmDbgWriteCheck(VirtualAddress, Opaque)`: the address to write through, when it is mapped.
/// Copied from the author's original anybox-tk code, with the author's permission.
pub fn dbg_write_check(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let address = c.arg(0);
    value(if k.memory.is_committed(address) {
        address
    } else {
        0
    })
}
