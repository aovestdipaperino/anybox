//! Executive exports: the pool, non-volatile settings, interlocked operations (spec 8.4, 8.6,
//! 8.12).

use super::value;
use crate::kernel::status;
use crate::kernel::{Call, CallResult, KResult, Kernel};

/// The default EEPROM: a North American console (spec 8.12).
#[must_use]
pub fn default_eeprom() -> [u8; 256] {
    let mut e = [0u8; 256];
    e[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
    e[0x34..0x40].copy_from_slice(b"000000000000");
    e[0x58..0x5C].copy_from_slice(&0x0040_0100u32.to_le_bytes());
    e[0x68..0x6C].copy_from_slice(b"GMT\0");
    e[0x6C..0x70].copy_from_slice(b"BST\0");
    e[0x90..0x94].copy_from_slice(&1u32.to_le_bytes());
    e[0xBC..0xC0].copy_from_slice(&1u32.to_le_bytes());
    e
}

/// A setting's EEPROM offset, size, and type (4 dword, 3 binary).
fn setting(index: u32) -> Option<(usize, usize, u32)> {
    const DWORD: u32 = 4;
    const BINARY: u32 = 3;
    Some(match index {
        0x00 => (0x64, 4, DWORD),
        0x01 => (0x68, 4, BINARY),
        0x02 => (0x6C, 4, BINARY),
        0x03 => (0x78, 4, DWORD),
        0x04 => (0x7C, 4, DWORD),
        0x05 => (0x88, 4, DWORD),
        0x06 => (0x8C, 4, DWORD),
        0x07..=0x12 => (0x90 + 4 * (index as usize - 7), 4, DWORD),
        0x100 => (0x34, 12, BINARY),
        0x101 => (0x40, 6, BINARY),
        0x102 => (0x48, 16, BINARY),
        0x103 => (0x58, 4, DWORD),
        0xFFFF => (0x00, 0x30, BINARY),
        _ => return None,
    })
}

pub fn allocate_pool(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(k.heap.alloc(c.arg(0)))
}

pub fn free_pool(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.heap.free(c.arg(0));
    value(0)
}

pub fn query_pool_block_size(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(k.heap.size_of(c.arg(0)))
}

/// `ExQueryNonVolatileSetting(index, *type, *value, len, *result_len)`.
pub fn query_non_volatile_setting(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some((offset, size, kind)) = setting(c.arg(0)) else {
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    k.w32_opt(c.arg(1), kind)?;
    let size32 = u32::try_from(size).unwrap_or(0);
    k.w32_opt(c.arg(4), size32)?;
    if c.arg(3) < size32 {
        return value(status::BUFFER_TOO_SMALL);
    }
    let bytes = k.eeprom[offset..offset + size].to_vec();
    k.write(c.arg(2), &bytes)?;
    value(status::SUCCESS)
}

/// `ExSaveNonVolatileSetting(index, type, *value, len)`: user settings only, for this run.
pub fn save_non_volatile_setting(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let index = c.arg(0);
    let Some((offset, size, _)) = setting(index) else {
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    let len = c.arg(3) as usize;
    if index >= 0x100 || len > size {
        return value(status::INVALID_PARAMETER);
    }
    let mut bytes = k.read(c.arg(2), c.arg(3))?;
    bytes.resize(size, 0);
    k.eeprom[offset..offset + size].copy_from_slice(&bytes);
    value(status::SUCCESS)
}

/// Fastcall `(destination, exchange, comparand)`; returns the value before.
pub fn interlocked_compare_exchange(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (at, exchange, comparand) = (c.arg(0), c.arg(1), c.arg(2));
    let old = k.r32(at)?;
    if old == comparand {
        k.w32(at, exchange)?;
    }
    value(old)
}

pub fn interlocked_increment(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let new = k.r32(c.arg(0))?.wrapping_add(1);
    k.w32(c.arg(0), new)?;
    value(new)
}

pub fn interlocked_decrement(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let new = k.r32(c.arg(0))?.wrapping_sub(1);
    k.w32(c.arg(0), new)?;
    value(new)
}

pub fn interlocked_exchange(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let old = k.r32(c.arg(0))?;
    k.w32(c.arg(0), c.arg(1))?;
    value(old)
}

pub fn interlocked_exchange_add(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let old = k.r32(c.arg(0))?;
    k.w32(c.arg(0), old.wrapping_add(c.arg(1)))?;
    value(old)
}

/// Bumps an `SLIST_HEADER`'s depth by `delta` and its sequence by one, with a new first entry.
fn slist_update(k: &Kernel, head: u32, first: u32, delta: i16) -> KResult<()> {
    let depth = k.r16(head + 4)?.wrapping_add_signed(delta);
    let sequence = k.r16(head + 6)?.wrapping_add(1);
    k.w32(head, first)?;
    k.w16(head + 4, depth)?;
    k.w16(head + 6, sequence)
}

/// Fastcall `(head, entry)`; returns the old first entry.
pub fn slist_push(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (head, entry) = (c.arg(0), c.arg(1));
    let old = k.r32(head)?;
    k.w32(entry, old)?;
    slist_update(k, head, entry, 1)?;
    value(old)
}

/// Fastcall `(head)`; returns the first entry or 0.
pub fn slist_pop(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let head = c.arg(0);
    let first = k.r32(head)?;
    if first != 0 {
        let next = k.r32(first)?;
        slist_update(k, head, next, -1)?;
    }
    value(first)
}

/// Fastcall `(head)`; empties the list and returns the first entry.
pub fn slist_flush(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let head = c.arg(0);
    let first = k.r32(head)?;
    k.w32(head, 0)?;
    k.w16(head + 4, 0)?;
    let sequence = k.r16(head + 6)?.wrapping_add(1);
    k.w16(head + 6, sequence)?;
    value(first)
}

/// `(p, lo, hi, lock)`: 64-bit add; returns the value before.
pub fn add_large_integer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let old = k.r64(c.arg(0))?;
    k.w64(c.arg(0), old.wrapping_add(c.arg64(1)))?;
    Ok(CallResult::Value64(old as u32, (old >> 32) as u32))
}

/// Fastcall `(p, increment)`: 64-bit add of a 32-bit increment.
pub fn add_large_statistic(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let old = k.r64(c.arg(0))?;
    k.w64(c.arg(0), old.wrapping_add(u64::from(c.arg(1))))?;
    value(0)
}

/// Fastcall `(p, *exchange, *comparand)`; returns the value before.
pub fn compare_exchange_64(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let old = k.r64(c.arg(0))?;
    if old == k.r64(c.arg(2))? {
        let exchange = k.r64(c.arg(1))?;
        k.w64(c.arg(0), exchange)?;
    }
    Ok(CallResult::Value64(old as u32, (old >> 32) as u32))
}

/// Fastcall `(head, entry)`: inserts at the head; returns the former first entry or 0.
pub fn insert_head_list(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (head, entry) = (c.arg(0), c.arg(1));
    let first = k.r32(head)?;
    k.w32(entry, first)?;
    k.w32(entry + 4, head)?;
    k.w32(first + 4, entry)?;
    k.w32(head, entry)?;
    value(if first == head { 0 } else { first })
}

/// Fastcall `(head, entry)`: inserts at the tail; returns the former last entry or 0.
pub fn insert_tail_list(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (head, entry) = (c.arg(0), c.arg(1));
    let last = k.r32(head + 4)?;
    k.w32(entry, head)?;
    k.w32(entry + 4, last)?;
    k.w32(last, entry)?;
    k.w32(head + 4, entry)?;
    value(if last == head { 0 } else { last })
}

/// Fastcall `(head)`: removes and returns the first entry, or 0.
pub fn remove_head_list(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let head = c.arg(0);
    let first = k.r32(head)?;
    if first == head {
        return value(0);
    }
    let next = k.r32(first)?;
    k.w32(head, next)?;
    k.w32(next + 4, head)?;
    value(first)
}
