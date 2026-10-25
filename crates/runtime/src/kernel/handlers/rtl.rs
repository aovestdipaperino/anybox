//! Run-time library exports: counted strings, critical sections, time fields, memory (spec 8.6,
//! 8.15).

use super::value;
use crate::kernel::object::WaitOn;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `RTL_CRITICAL_SECTION` fields (spec 8.6).
const LOCK_COUNT: u32 = 0x10;
const RECURSION_COUNT: u32 = 0x14;
const OWNING_THREAD: u32 = 0x18;

/// A counted string: length, capacity, buffer.
fn counted(k: &Kernel, at: u32) -> KResult<(u16, u16, u32)> {
    Ok((k.r16(at)?, k.r16(at + 2)?, k.r32(at + 4)?))
}

/// `RtlInitAnsiString(dst, src)`: length = strlen (at most 65534), capacity length + 1.
pub fn init_ansi_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (dst, src) = (c.arg(0), c.arg(1));
    let len = if src == 0 {
        0
    } else {
        k.c_string(src)?.len().min(65534)
    };
    let len = u16::try_from(len).unwrap_or(65534);
    k.w16(dst, len)?;
    k.w16(dst + 2, if src == 0 { 0 } else { len + 1 })?;
    k.w32(dst + 4, src)?;
    value(0)
}

/// `RtlInitUnicodeString(dst, src)`: lengths in bytes.
pub fn init_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (dst, src) = (c.arg(0), c.arg(1));
    let mut units = 0u32;
    if src != 0 {
        while units < 32766 && k.r16(src + 2 * units)? != 0 {
            units += 1;
        }
    }
    let len = u16::try_from(units * 2).unwrap_or(u16::MAX);
    k.w16(dst, len)?;
    k.w16(dst + 2, if src == 0 { 0 } else { len + 2 })?;
    k.w32(dst + 4, src)?;
    value(0)
}

/// The bytes of a counted 8-bit string.
fn string_bytes(k: &Kernel, at: u32) -> KResult<Vec<u8>> {
    let (len, _, buffer) = counted(k, at)?;
    k.read(buffer, u32::from(len))
}

fn fold(b: u8, ci: bool) -> u8 {
    if ci { b.to_ascii_uppercase() } else { b }
}

/// `RtlEqualString(a, b, ci)`.
pub fn equal_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let a = string_bytes(k, c.arg(0))?;
    let b = string_bytes(k, c.arg(1))?;
    let ci = c.arg(2) & 0xFF != 0;
    let equal = a.len() == b.len() && a.iter().zip(&b).all(|(&x, &y)| fold(x, ci) == fold(y, ci));
    value(u32::from(equal))
}

/// `RtlCompareString(a, b, ci)`: by unit, then by length.
pub fn compare_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let first = string_bytes(k, c.arg(0))?;
    let second = string_bytes(k, c.arg(1))?;
    let ci = c.arg(2) & 0xFF != 0;
    for (&x, &y) in first.iter().zip(&second) {
        let difference = i32::from(fold(x, ci)) - i32::from(fold(y, ci));
        if difference != 0 {
            return value(difference.cast_unsigned());
        }
    }
    let len = |s: &[u8]| i32::try_from(s.len()).unwrap_or(0);
    value((len(&first) - len(&second)).cast_unsigned())
}

/// Copies what fits of `bytes` into the counted string at `dst`, with a NUL after when there is
/// room.
fn store(k: &Kernel, dst: u32, bytes: &[u8]) -> KResult<()> {
    let (_, capacity, buffer) = counted(k, dst)?;
    let n = bytes.len().min(usize::from(capacity));
    k.write(buffer, &bytes[..n])?;
    if n < usize::from(capacity) {
        k.w8(buffer + u32::try_from(n).unwrap_or(0), 0)?;
    }
    k.w16(dst, u16::try_from(n).unwrap_or(0))
}

/// `RtlCopyString(dst, src)`: an absent source empties.
pub fn copy_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let bytes = if c.arg(1) == 0 {
        Vec::new()
    } else {
        string_bytes(k, c.arg(1))?
    };
    store(k, c.arg(0), &bytes)?;
    value(0)
}

/// `RtlUpperString(dst, src)`.
pub fn upper_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let bytes = string_bytes(k, c.arg(1))?.to_ascii_uppercase();
    store(k, c.arg(0), &bytes)?;
    value(0)
}

/// `RtlAppendStringToString(dst, src)`: `BUFFER_TOO_SMALL` when it does not fit.
pub fn append_string_to_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (len, capacity, buffer) = counted(k, c.arg(0))?;
    let more = string_bytes(k, c.arg(1))?;
    let total = usize::from(len) + more.len();
    if total > usize::from(capacity) {
        return value(status::BUFFER_TOO_SMALL);
    }
    k.write(buffer + u32::from(len), &more)?;
    k.w16(c.arg(0), u16::try_from(total).unwrap_or(0))?;
    value(status::SUCCESS)
}

/// `RtlFreeAnsiString` and `RtlFreeUnicodeString`: frees the buffer.
pub fn free_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let buffer = k.r32(c.arg(0) + 4)?;
    k.heap.free(buffer);
    k.w32(c.arg(0) + 4, 0)?;
    value(0)
}

/// Stores `bytes` in the string at `dst`, in a new heap buffer when `allocate`, else in its
/// buffer, which must hold them.
fn store_converted(
    k: &mut Kernel,
    dst: u32,
    bytes: &[u8],
    terminator: u32,
    allocate: bool,
) -> KResult<u32> {
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    let Ok(len16) = u16::try_from(len) else {
        return Ok(status::INVALID_PARAMETER);
    };
    let buffer = if allocate {
        let buffer = k.heap.alloc(len + terminator);
        if buffer == 0 {
            return Ok(status::NO_MEMORY);
        }
        k.w16(dst + 2, u16::try_from(len + terminator).unwrap_or(u16::MAX))?;
        k.w32(dst + 4, buffer)?;
        buffer
    } else {
        let (_, capacity, buffer) = counted(k, dst)?;
        if len > u32::from(capacity) {
            return Ok(status::BUFFER_OVERFLOW);
        }
        buffer
    };
    k.write(buffer, bytes)?;
    if allocate {
        k.write(buffer + len, &vec![0; terminator as usize])?;
    }
    k.w16(dst, len16)?;
    Ok(status::SUCCESS)
}

/// `RtlAnsiStringToUnicodeString(dst, src, allocate)`: Latin-1 to UTF-16.
pub fn ansi_string_to_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let bytes: Vec<u8> = string_bytes(k, c.arg(1))?
        .iter()
        .flat_map(|&b| u16::from(b).to_le_bytes())
        .collect();
    value(store_converted(
        k,
        c.arg(0),
        &bytes,
        2,
        c.arg(2) & 0xFF != 0,
    )?)
}

/// `RtlUnicodeStringToAnsiString(dst, src, allocate)`: characters above 255 become `?`.
pub fn unicode_string_to_ansi_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let units = string_bytes(k, c.arg(1))?;
    let bytes: Vec<u8> = units
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&u| u8::try_from(u16::from_le_bytes(u)).unwrap_or(b'?'))
        .collect();
    value(store_converted(
        k,
        c.arg(0),
        &bytes,
        1,
        c.arg(2) & 0xFF != 0,
    )?)
}

pub fn nt_status_to_dos_error(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(status::to_dos_error(c.arg(0)))
}

/// `RtlInitializeCriticalSection(cs)`: free.
pub fn initialize_critical_section(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let cs = c.arg(0);
    k.w32(cs + LOCK_COUNT, u32::MAX)?;
    k.w32(cs + RECURSION_COUNT, 0)?;
    k.w32(cs + OWNING_THREAD, 0)?;
    value(0)
}

/// Enters a critical section if it is free or the caller's; returns whether it did.
fn try_enter(k: &Kernel, cs: u32) -> KResult<bool> {
    let me = k.current_kthread();
    let owner = k.r32(cs + OWNING_THREAD)?;
    if owner == 0 {
        k.w32(cs + LOCK_COUNT, 0)?;
        k.w32(cs + RECURSION_COUNT, 1)?;
        k.w32(cs + OWNING_THREAD, me)?;
        return Ok(true);
    }
    if owner == me {
        let recursion = k.r32(cs + RECURSION_COUNT)?;
        let lock = k.r32(cs + LOCK_COUNT)?;
        k.w32(cs + RECURSION_COUNT, recursion + 1)?;
        k.w32(cs + LOCK_COUNT, lock.wrapping_add(1))?;
        return Ok(true);
    }
    Ok(false)
}

/// `RtlEnterCriticalSection(cs)`: another owner makes the caller count itself in `LockCount` and
/// wait on the section's queue.
pub fn enter_critical_section(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let cs = c.arg(0);
    if try_enter(k, cs)? {
        return value(0);
    }
    let lock = k.r32(cs + LOCK_COUNT)?;
    k.w32(cs + LOCK_COUNT, lock.wrapping_add(1))?;
    let me = k.current;
    k.section_waiters.entry(cs).or_default().push_back(me);
    k.block(WaitOn::Explicit, None);
    Ok(CallResult::Block)
}

/// `RtlLeaveCriticalSection(cs)`: hands the section to the oldest waiter, if any.
pub fn leave_critical_section(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let cs = c.arg(0);
    if k.r32(cs + OWNING_THREAD)? != k.current_kthread() {
        return value(0);
    }
    let recursion = k.r32(cs + RECURSION_COUNT)?;
    let lock = k.r32(cs + LOCK_COUNT)?;
    if recursion > 1 {
        k.w32(cs + RECURSION_COUNT, recursion - 1)?;
        k.w32(cs + LOCK_COUNT, lock.wrapping_sub(1))?;
        return value(0);
    }
    let next = k
        .section_waiters
        .get_mut(&cs)
        .and_then(std::collections::VecDeque::pop_front);
    if let Some(tid) = next {
        let kthread = k.threads.get(tid).map_or(0, |t| t.kthread);
        k.w32(cs + OWNING_THREAD, kthread)?;
        k.w32(cs + RECURSION_COUNT, 1)?;
        k.w32(cs + LOCK_COUNT, lock.wrapping_sub(1))?;
        k.wake(tid, 0);
    } else {
        k.w32(cs + LOCK_COUNT, u32::MAX)?;
        k.w32(cs + RECURSION_COUNT, 0)?;
        k.w32(cs + OWNING_THREAD, 0)?;
    }
    value(0)
}

pub fn try_enter_critical_section(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from(try_enter(k, c.arg(0))?))
}

/// Days from 1970-01-01 to a civil date (proleptic Gregorian).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date of a day count from 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Days from 1601-01-01 to 1970-01-01.
const DAYS_1601_TO_1970: i64 = 134_774;
const TICKS_PER_DAY: i64 = 864_000_000_000;

/// `RtlTimeToTimeFields(*time, *fields)`.
pub fn time_to_time_fields(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let time = k.r64(c.arg(0))?.cast_signed();
    let days = time.div_euclid(TICKS_PER_DAY);
    let rest = time.rem_euclid(TICKS_PER_DAY);
    let (year, month, day) = civil_from_days(days - DAYS_1601_TO_1970);
    let fields = [
        year,
        month,
        day,
        rest / 36_000_000_000,
        rest / 600_000_000 % 60,
        rest / 10_000_000 % 60,
        rest / 10_000 % 1000,
        // 1601-01-01 was a Monday; 0 is Sunday.
        (days + 1).rem_euclid(7),
    ];
    for (i, f) in fields.iter().enumerate() {
        k.w16(
            c.arg(1) + 2 * u32::try_from(i).unwrap_or(0),
            u16::try_from(*f).unwrap_or(0),
        )?;
    }
    value(0)
}

/// `RtlTimeFieldsToTime(*fields, *time)`: false for out-of-range fields.
pub fn time_fields_to_time(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mut f = [0i64; 7];
    for (i, field) in f.iter_mut().enumerate() {
        *field = i64::from(k.r16(c.arg(0) + 2 * u32::try_from(i).unwrap_or(0))?);
    }
    let [year, month, day, hour, minute, second, ms] = f;
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let valid = year >= 1601
        && (1..=12).contains(&month)
        && day >= 1
        && day <= month_days[usize::try_from(month.clamp(1, 12) - 1).unwrap_or(0)]
        && hour < 24
        && minute < 60
        && second < 60
        && ms < 1000;
    if !valid {
        return value(0);
    }
    let days = days_from_civil(year, month, day) + DAYS_1601_TO_1970;
    let time =
        days * TICKS_PER_DAY + ((hour * 60 + minute) * 60 + second) * 10_000_000 + ms * 10_000;
    k.w64(c.arg(1), time.cast_unsigned())?;
    value(1)
}

/// `RtlCompareMemory(a, b, n)`: the leading equal bytes.
pub fn compare_memory(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let a = k.read(c.arg(0), c.arg(2))?;
    let b = k.read(c.arg(1), c.arg(2))?;
    value(u32::try_from(a.iter().zip(&b).take_while(|(x, y)| x == y).count()).unwrap_or(0))
}

/// `RtlCompareMemoryUlong(p, n, pattern)`: leading bytes, in whole dwords, equal to the pattern.
pub fn compare_memory_ulong(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mut n = 0;
    while n + 4 <= c.arg(1) && k.r32(c.arg(0) + n)? == c.arg(2) {
        n += 4;
    }
    value(n)
}

/// `RtlFillMemory(dst, len, byte)`.
pub fn fill_memory(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.write(c.arg(0), &vec![c.arg(2) as u8; c.arg(1) as usize])?;
    value(0)
}

/// `RtlFillMemoryUlong(dst, len, pattern)`.
pub fn fill_memory_ulong(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let bytes: Vec<u8> = (0..c.arg(1) / 4)
        .flat_map(|_| c.arg(2).to_le_bytes())
        .collect();
    k.write(c.arg(0), &bytes)?;
    value(0)
}

/// `RtlZeroMemory(dst, len)`.
pub fn zero_memory(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.write(c.arg(0), &vec![0; c.arg(1) as usize])?;
    value(0)
}

/// `RtlMoveMemory(dst, src, len)`: overlap is handled.
pub fn move_memory(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let bytes = k.read(c.arg(1), c.arg(2))?;
    k.write(c.arg(0), &bytes)?;
    value(0)
}

pub fn ulong_byte_swap(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(c.arg(0).swap_bytes())
}

pub fn ushort_byte_swap(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from((c.arg(0) as u16).swap_bytes()))
}

/// `RtlExtendedIntegerMultiply(lo, hi, m)`: the wrapping 64-bit product.
pub fn extended_integer_multiply(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let product = c
        .arg64(0)
        .cast_signed()
        .wrapping_mul(i64::from(c.arg(2).cast_signed()));
    let product = product.cast_unsigned();
    Ok(CallResult::Value64(product as u32, (product >> 32) as u32))
}

/// `RtlExtendedLargeIntegerDivide(lo, hi, d, *rem)`: unsigned.
pub fn extended_large_integer_divide(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (dividend, divisor) = (c.arg64(0), u64::from(c.arg(2)));
    if divisor == 0 {
        return Ok(CallResult::Unsupported("a division by zero".to_owned()));
    }
    k.w32_opt(c.arg(3), (dividend % divisor) as u32)?;
    let quotient = dividend / divisor;
    Ok(CallResult::Value64(
        quotient as u32,
        (quotient >> 32) as u32,
    ))
}

pub fn upper_char(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from((c.arg(0) as u8).to_ascii_uppercase()))
}

pub fn lower_char(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from((c.arg(0) as u8).to_ascii_lowercase()))
}

/// `RtlGetCallersAddress(*caller, *callers_caller)`: the return address, and 0.
pub fn get_callers_address(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let ret = k.r32(c.esp)?;
    k.w32_opt(c.arg(0), ret)?;
    k.w32_opt(c.arg(1), 0)?;
    value(0)
}

/// `RtlAssert(expr, file, line, msg)`: reported; the title continues.
pub fn assert(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let text = |at: u32| -> String {
        if at == 0 {
            String::new()
        } else {
            k.c_string(at)
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .unwrap_or_default()
        }
    };
    eprintln!(
        "RtlAssert: {} at {}:{} {}",
        text(c.arg(0)),
        text(c.arg(1)),
        c.arg(2),
        text(c.arg(3))
    );
    value(0)
}

/// Exceptions: lifted code has no structured exception handling.
pub fn raise(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let ret = k.r32(c.esp).unwrap_or(0);
    Ok(CallResult::Unsupported(format!(
        "an exception ({:#x}) raised from {ret:#010x}",
        c.arg(0)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1601, 1, 1), -DAYS_1601_TO_1970);
        assert_eq!(civil_from_days(days_from_civil(2004, 2, 29)), (2004, 2, 29));
        // The boot clock is 2004-01-01 00:00:00.
        let boot = crate::kernel::time::BOOT_CLOCK.cast_signed();
        assert_eq!(boot % TICKS_PER_DAY, 0);
        assert_eq!(
            civil_from_days(boot / TICKS_PER_DAY - DAYS_1601_TO_1970),
            (2004, 1, 1)
        );
        // 2004-01-01 was a Thursday (4).
        assert_eq!((boot / TICKS_PER_DAY + 1).rem_euclid(7), 4);
    }
}
