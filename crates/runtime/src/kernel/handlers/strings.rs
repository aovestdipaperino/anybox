//! Run-time library exports on UTF-16 counted strings, characters, and integers in text.
//!
//! A counted string (`STRING` for 8-bit text, `UNICODE_STRING` for UTF-16) is a 16-bit length
//! and a 16-bit capacity, both in bytes, then a buffer pointer. Conversions between the two are
//! Latin-1: the console's code page for the first 256 characters; others become `?`.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission, and adapted
//! to this crate's handler interface.

use std::cmp::Ordering;

use super::value;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// A counted string's header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counted {
    /// Bytes in use.
    length: u16,
    /// Bytes the buffer holds.
    maximum: u16,
    buffer: u32,
}

impl Counted {
    fn read(k: &Kernel, address: u32) -> KResult<Self> {
        Ok(Self {
            length: k.r16(address)?,
            maximum: k.r16(address + 2)?,
            buffer: k.r32(address + 4)?,
        })
    }

    fn write(self, k: &Kernel, address: u32) -> KResult<()> {
        k.w16(address, self.length)?;
        k.w16(address + 2, self.maximum)?;
        k.w32(address + 4, self.buffer)
    }

    /// The text of a UTF-16 string.
    fn wide(self, k: &Kernel) -> KResult<Vec<u16>> {
        if self.buffer == 0 {
            return Ok(Vec::new());
        }
        read_wide(k, self.buffer, u32::from(self.length / 2))
    }
}

/// Reads `count` UTF-16 code units at `address`.
fn read_wide(k: &Kernel, address: u32, count: u32) -> KResult<Vec<u16>> {
    let bytes = k.read(address, count * 2)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|unit| u16::from_le_bytes(*unit))
        .collect())
}

/// Reads a NUL-terminated UTF-16 string at `address`, without the NUL.
fn read_wide_c(k: &Kernel, address: u32) -> KResult<Vec<u16>> {
    let mut text = Vec::new();
    // A counted string holds at most 32767 characters.
    for i in 0..0x7FFFu32 {
        let unit = k.r16(address.wrapping_add(2 * i))?;
        if unit == 0 {
            break;
        }
        text.push(unit);
    }
    Ok(text)
}

/// Reads a NUL-terminated 8-bit string of at most `max` bytes.
fn read_c(k: &Kernel, address: u32, max: u32) -> KResult<Vec<u8>> {
    let mut text = Vec::new();
    for i in 0..max {
        let byte = k.r8(address.wrapping_add(i))?;
        if byte == 0 {
            break;
        }
        text.push(byte);
    }
    Ok(text)
}

/// Latin-1 to UTF-16.
fn widen(text: &[u8]) -> Vec<u16> {
    text.iter().map(|&byte| u16::from(byte)).collect()
}

/// UTF-16 to Latin-1.
fn narrow(text: &[u16]) -> Vec<u8> {
    text.iter()
        .map(|&unit| u8::try_from(unit).unwrap_or(b'?'))
        .collect()
}

/// A UTF-16 unit in upper case, when its upper case is one unit.
fn upcase(unit: u16) -> u16 {
    char::from_u32(u32::from(unit))
        .and_then(|c| {
            let mut upper = c.to_uppercase();
            let single = upper.next()?;
            upper.next().is_none().then_some(single)
        })
        .and_then(|c| u16::try_from(u32::from(c)).ok())
        .unwrap_or(unit)
}

/// A UTF-16 unit in lower case, when its lower case is one unit.
fn downcase(unit: u16) -> u16 {
    char::from_u32(u32::from(unit))
        .and_then(|c| {
            let mut lower = c.to_lowercase();
            let single = lower.next()?;
            lower.next().is_none().then_some(single)
        })
        .and_then(|c| u16::try_from(u32::from(c)).ok())
        .unwrap_or(unit)
}

fn wide_bytes(text: &[u16]) -> Vec<u8> {
    text.iter().flat_map(|unit| unit.to_le_bytes()).collect()
}

/// Stores `bytes` as the string at `dest`: into a new buffer with `allocate`, else into the one
/// it has, which must hold it. A NUL follows when there is room. `unit` is the bytes per unit.
fn store(k: &mut Kernel, dest: u32, bytes: &[u8], unit: u16, allocate: bool) -> KResult<u32> {
    let Ok(length) = u16::try_from(bytes.len()) else {
        return Ok(status::INVALID_PARAMETER);
    };
    let mut counted = Counted::read(k, dest)?;
    if allocate {
        let maximum = length.saturating_add(unit);
        let buffer = k.heap.alloc(u32::from(maximum));
        if buffer == 0 {
            return Ok(status::NO_MEMORY);
        }
        counted = Counted {
            length,
            maximum,
            buffer,
        };
    } else if length > counted.maximum {
        return Ok(status::BUFFER_OVERFLOW);
    } else {
        counted.length = length;
    }
    k.write(counted.buffer, bytes)?;
    if u32::from(counted.maximum) >= u32::from(length) + u32::from(unit) {
        k.write(
            counted.buffer + u32::from(length),
            &vec![0; usize::from(unit)],
        )?;
    }
    counted.write(k, dest)?;
    Ok(status::SUCCESS)
}

/// The UTF-16 text of the string at argument `index`, mapped through `map`, stored at argument 0
/// as `RtlUpcaseUnicodeString` and its kin do.
fn convert_case(k: &mut Kernel, c: &Call, map: fn(u16) -> u16) -> KResult<CallResult> {
    let text: Vec<u16> = Counted::read(k, c.arg(1))?
        .wide(k)?
        .into_iter()
        .map(map)
        .collect();
    value(store(
        k,
        c.arg(0),
        &wide_bytes(&text),
        2,
        c.arg(2) & 0xFF != 0,
    )?)
}

/// `RtlUpcaseUnicodeString(Destination, Source, AllocateDestinationString)`.
pub fn upcase_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    convert_case(k, c, upcase)
}

/// `RtlDowncaseUnicodeString(Destination, Source, AllocateDestinationString)`.
pub fn downcase_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    convert_case(k, c, downcase)
}

/// Appends `bytes` to the string at `dest`, which must have room; a NUL follows when there is
/// room for `unit` more bytes.
fn append(k: &Kernel, dest: u32, bytes: &[u8], unit: u16) -> KResult<u32> {
    let mut counted = Counted::read(k, dest)?;
    let Some(length) = u16::try_from(bytes.len())
        .ok()
        .and_then(|len| counted.length.checked_add(len))
        .filter(|&length| length <= counted.maximum)
    else {
        return Ok(status::BUFFER_TOO_SMALL);
    };
    k.write(counted.buffer + u32::from(counted.length), bytes)?;
    if u32::from(counted.maximum) >= u32::from(length) + u32::from(unit) {
        k.write(
            counted.buffer + u32::from(length),
            &vec![0; usize::from(unit)],
        )?;
    }
    counted.length = length;
    counted.write(k, dest)?;
    Ok(status::SUCCESS)
}

/// `RtlAppendUnicodeStringToString(Destination, Source)`.
pub fn append_unicode_string_to_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let text = Counted::read(k, c.arg(1))?.wide(k)?;
    value(append(k, c.arg(0), &wide_bytes(&text), 2)?)
}

/// `RtlAppendUnicodeToString(Destination, Source)`: a NUL-terminated source.
pub fn append_unicode_to_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if c.arg(1) == 0 {
        return value(status::SUCCESS);
    }
    let text = read_wide_c(k, c.arg(1))?;
    value(append(k, c.arg(0), &wide_bytes(&text), 2)?)
}

/// Copies as much of `bytes` as fits into the string at `dest`, with a NUL after it when
/// there is room for `unit` more bytes.
fn copy_into(k: &Kernel, dest: u32, bytes: &[u8], unit: u16) -> KResult<()> {
    let mut counted = Counted::read(k, dest)?;
    let length = u16::try_from(bytes.len())
        .unwrap_or(u16::MAX)
        .min(counted.maximum);
    let length = length - length % unit.max(1);
    k.write(counted.buffer, &bytes[..usize::from(length)])?;
    if u32::from(counted.maximum) >= u32::from(length) + u32::from(unit) && unit > 0 {
        k.write(
            counted.buffer + u32::from(length),
            &vec![0; usize::from(unit)],
        )?;
    }
    counted.length = length;
    counted.write(k, dest)
}

/// `RtlCopyUnicodeString(Destination, Source)`: an absent source empties the destination.
pub fn copy_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let text = if c.arg(1) == 0 {
        Vec::new()
    } else {
        Counted::read(k, c.arg(1))?.wide(k)?
    };
    copy_into(k, c.arg(0), &wide_bytes(&text), 2)?;
    value(0)
}

/// `RtlCreateUnicodeString(Destination, Source)`: a copy of a NUL-terminated string in a new
/// buffer. Returns whether it could be allocated.
pub fn create_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let text = read_wide_c(k, c.arg(1))?;
    let done = store(k, c.arg(0), &wide_bytes(&text), 2, true)? == status::SUCCESS;
    value(u32::from(done))
}

/// Compares two texts the way the run-time library does: by unit, then by length.
fn compare<T: Ord + Copy>(a: &[T], b: &[T], fold: impl Fn(T) -> T) -> i32 {
    for (&x, &y) in a.iter().zip(b) {
        match fold(x).cmp(&fold(y)) {
            Ordering::Less => return -1,
            Ordering::Greater => return 1,
            Ordering::Equal => {}
        }
    }
    match a.len().cmp(&b.len()) {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

/// The case fold of a case-insensitive comparison, or none.
fn fold(insensitive: bool) -> fn(u16) -> u16 {
    if insensitive {
        upcase
    } else {
        std::convert::identity
    }
}

/// `RtlCompareUnicodeString(String1, String2, CaseInSensitive)`.
pub fn compare_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let a = Counted::read(k, c.arg(0))?.wide(k)?;
    let b = Counted::read(k, c.arg(1))?.wide(k)?;
    let order = compare(&a, &b, fold(c.arg(2) & 0xFF != 0));
    value(order.cast_unsigned())
}

/// `RtlEqualUnicodeString(String1, String2, CaseInSensitive)`.
pub fn equal_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let a = Counted::read(k, c.arg(0))?.wide(k)?;
    let b = Counted::read(k, c.arg(1))?.wide(k)?;
    let equal = a.len() == b.len() && compare(&a, &b, fold(c.arg(2) & 0xFF != 0)) == 0;
    value(u32::from(equal))
}

/// `RtlUpcaseUnicodeChar(SourceCharacter)`.
pub fn upcase_unicode_char(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from(upcase(c.arg(0) as u16)))
}

/// `RtlDowncaseUnicodeChar(SourceCharacter)`.
pub fn downcase_unicode_char(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from(downcase(c.arg(0) as u16)))
}

/// Parses an integer the way `RtlCharToInteger` does: blanks, a sign, then with base 0 a
/// `0x`, `0o` or `0b` prefix or else decimal; digits up to the first that is not one.
/// Returns `None` for a base other than 0, 2, 8, 10 and 16.
fn parse_integer(text: &[u8], base: u32) -> Option<u32> {
    let mut rest = text;
    while let [b' ' | b'\t', tail @ ..] = rest {
        rest = tail;
    }
    let negative = match rest {
        [b'-', tail @ ..] => {
            rest = tail;
            true
        }
        [b'+', tail @ ..] => {
            rest = tail;
            false
        }
        _ => false,
    };
    let base = match base {
        0 => match rest {
            [b'0', b'x' | b'X', tail @ ..] => {
                rest = tail;
                16
            }
            [b'0', b'o' | b'O', tail @ ..] => {
                rest = tail;
                8
            }
            [b'0', b'b' | b'B', tail @ ..] => {
                rest = tail;
                2
            }
            _ => 10,
        },
        2 | 8 | 10 | 16 => base,
        _ => return None,
    };
    let mut value = 0u32;
    for &byte in rest {
        let Some(digit) = char::from(byte).to_digit(base) else {
            break;
        };
        value = value.wrapping_mul(base).wrapping_add(digit);
    }
    Some(if negative {
        value.wrapping_neg()
    } else {
        value
    })
}

/// Parses `text` in the base at argument 1 into the dword at argument 2.
fn store_integer(k: &Kernel, c: &Call, text: &[u8]) -> KResult<CallResult> {
    let Some(parsed) = parse_integer(text, c.arg(1)) else {
        return value(status::INVALID_PARAMETER);
    };
    k.w32(c.arg(2), parsed)?;
    value(status::SUCCESS)
}

/// `RtlCharToInteger(String, Base, Value)`: a NUL-terminated string.
pub fn char_to_integer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let text = read_c(k, c.arg(0), 64)?;
    store_integer(k, c, &text)
}

/// `RtlUnicodeStringToInteger(String, Base, Value)`.
pub fn unicode_string_to_integer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let text = narrow(&Counted::read(k, c.arg(0))?.wide(k)?);
    store_integer(k, c, &text)
}

/// The digits of `value` in `base` (0 meaning 10), or `None` for an unsupported base.
fn digits(value: u32, base: u32) -> Option<Vec<u8>> {
    let base = match base {
        0 => 10,
        2 | 8 | 10 | 16 => base,
        _ => return None,
    };
    let mut out = Vec::new();
    let mut rest = value;
    loop {
        let digit = rest % base;
        out.push(char::from_digit(digit, base).map_or(b'0', |c| c.to_ascii_uppercase() as u8));
        rest /= base;
        if rest == 0 {
            break;
        }
    }
    out.reverse();
    Some(out)
}

/// `RtlIntegerToChar(Value, Base, OutputLength, String)`.
pub fn integer_to_char(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(text) = digits(c.arg(0), c.arg(1)) else {
        return value(status::INVALID_PARAMETER);
    };
    let room = c.arg(2).cast_signed();
    let len = i32::try_from(text.len()).unwrap_or(i32::MAX);
    if len > room {
        return value(status::BUFFER_OVERFLOW);
    }
    k.write(c.arg(3), &text)?;
    if len < room {
        k.w8(c.arg(3) + len.cast_unsigned(), 0)?;
    }
    value(status::SUCCESS)
}

/// `RtlIntegerToUnicodeString(Value, Base, String)`.
pub fn integer_to_unicode_string(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(text) = digits(c.arg(0), c.arg(1)) else {
        return value(status::INVALID_PARAMETER);
    };
    value(store(k, c.arg(2), &wide_bytes(&widen(&text)), 2, false)?)
}

/// `RtlMultiByteToUnicodeN(UnicodeString, MaxBytesInUnicodeString, BytesInUnicodeString,
/// MultiByteString, BytesInMultiByteString)`.
pub fn multi_byte_to_unicode_n(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let units = (c.arg(1) / 2).min(c.arg(4));
    let text = k.read(c.arg(3), units)?;
    k.write(c.arg(0), &wide_bytes(&widen(&text)))?;
    k.w32_opt(c.arg(2), units * 2)?;
    value(status::SUCCESS)
}

/// `RtlMultiByteToUnicodeSize(BytesInUnicodeString, MultiByteString, BytesInMultiByteString)`.
pub fn multi_byte_to_unicode_size(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w32(c.arg(0), c.arg(2).wrapping_mul(2))?;
    value(status::SUCCESS)
}

/// Converts UTF-16 to 8-bit text, as `RtlUnicodeToMultiByteN` does, upper-casing with `upper`.
fn unicode_to_multi_byte(k: &Kernel, c: &Call, upper: bool) -> KResult<CallResult> {
    let units = c.arg(1).min(c.arg(4) / 2);
    let mut text = read_wide(k, c.arg(3), units)?;
    if upper {
        text = text.into_iter().map(upcase).collect();
    }
    k.write(c.arg(0), &narrow(&text))?;
    k.w32_opt(c.arg(2), units)?;
    value(status::SUCCESS)
}

/// `RtlUnicodeToMultiByteN(MultiByteString, MaxBytesInMultiByteString,
/// BytesInMultiByteString, UnicodeString, BytesInUnicodeString)`.
pub fn unicode_to_multi_byte_n(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    unicode_to_multi_byte(k, c, false)
}

/// `RtlUpcaseUnicodeToMultiByteN`: as `RtlUnicodeToMultiByteN`, upper-cased.
pub fn upcase_unicode_to_multi_byte_n(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    unicode_to_multi_byte(k, c, true)
}

/// `RtlUnicodeToMultiByteSize(BytesInMultiByteString, UnicodeString, BytesInUnicodeString)`.
pub fn unicode_to_multi_byte_size(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w32(c.arg(0), c.arg(2) / 2)?;
    value(status::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::handlers::lookup;
    use crate::kernel::testing::kernel;

    /// Guest scratch memory inside the committed heap.
    const SCRATCH: u32 = crate::mem::HEAP_BASE + 0x2F_0000;

    fn call(k: &mut Kernel, name: &str, args: &[u32]) -> u32 {
        let handler = lookup(name).unwrap_or_else(|| panic!("no handler for {name}"));
        let c = Call {
            args: args.to_vec(),
            esp: 0,
        };
        match handler(k, &c) {
            Ok(CallResult::Value(v)) => v,
            other => panic!("{name} returned {other:?}"),
        }
    }

    /// A `UNICODE_STRING` at `at` holding `text`, with room for `capacity` bytes after the header.
    fn unicode(k: &Kernel, at: u32, text: &str, capacity: u16) -> u32 {
        let bytes: Vec<u8> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
        Counted {
            length: u16::try_from(bytes.len()).unwrap(),
            maximum: capacity,
            buffer: at + 8,
        }
        .write(k, at)
        .unwrap();
        k.write(at + 8, &bytes).unwrap();
        at
    }

    fn text_of(k: &Kernel, at: u32) -> String {
        String::from_utf16(&Counted::read(k, at).unwrap().wide(k).unwrap()).unwrap()
    }

    #[test]
    fn integers_parse_with_signs_prefixes_and_bases() {
        assert_eq!(parse_integer(b"  42xyz", 0), Some(42));
        assert_eq!(parse_integer(b"-0x1F", 0), Some(0x1Fu32.wrapping_neg()));
        assert_eq!(parse_integer(b"0b101", 0), Some(5));
        assert_eq!(parse_integer(b"0o17", 0), Some(15));
        assert_eq!(parse_integer(b"ff", 16), Some(255));
        assert_eq!(parse_integer(b"12", 3), None);
        assert_eq!(digits(255, 16), Some(b"FF".to_vec()));
        assert_eq!(digits(0, 0), Some(b"0".to_vec()));
        assert_eq!(digits(5, 2), Some(b"101".to_vec()));
        assert_eq!(digits(1, 7), None);
    }

    #[test]
    fn comparisons_go_by_unit_then_length_and_case_folds() {
        assert_eq!(compare(b"abc", b"abd", |b| b), -1);
        assert_eq!(compare(b"ab", b"abc", |b| b), -1);
        assert_eq!(compare(b"ABC", b"abc", |b: u8| b.to_ascii_uppercase()), 0);
        assert_eq!(upcase(u16::from(b'a')), u16::from(b'A'));
        assert_eq!(downcase(0xC9), 0xE9, "É to é");
        assert_eq!(narrow(&[0x41, 0x263A]), b"A?");
    }

    #[test]
    fn unicode_strings_compare_append_copy_and_change_case() {
        let mut k = kernel();
        let a = unicode(&k, SCRATCH, "Hello", 32);
        let b = unicode(&k, SCRATCH + 0x100, "hello", 32);
        assert_eq!(call(&mut k, "RtlEqualUnicodeString", &[a, b, 0]), 0);
        assert_eq!(call(&mut k, "RtlEqualUnicodeString", &[a, b, 1]), 1);
        assert_eq!(
            call(&mut k, "RtlCompareUnicodeString", &[a, b, 0]),
            u32::MAX
        );
        assert_eq!(call(&mut k, "RtlCompareUnicodeString", &[a, b, 1]), 0);
        let world = unicode(&k, SCRATCH + 0x200, ", world", 32);
        assert_eq!(
            call(&mut k, "RtlAppendUnicodeStringToString", &[a, world]),
            status::SUCCESS
        );
        assert_eq!(text_of(&k, a), "Hello, world");
        assert_eq!(k.r16(SCRATCH + 8 + 24), Ok(0), "a NUL after it");
        let bang = SCRATCH + 0x300;
        k.write(bang, &[b'!', 0, 0, 0]).unwrap();
        assert_eq!(
            call(&mut k, "RtlAppendUnicodeToString", &[a, bang]),
            status::SUCCESS
        );
        assert_eq!(text_of(&k, a), "Hello, world!");
        let small = unicode(&k, SCRATCH + 0x400, "", 4);
        assert_eq!(
            call(&mut k, "RtlAppendUnicodeStringToString", &[small, a]),
            status::BUFFER_TOO_SMALL
        );
        assert_eq!(call(&mut k, "RtlCopyUnicodeString", &[small, a]), 0);
        assert_eq!(text_of(&k, small), "He", "as much as fits");
        assert_eq!(call(&mut k, "RtlCopyUnicodeString", &[small, 0]), 0);
        assert_eq!(text_of(&k, small), "");
        assert_eq!(
            call(&mut k, "RtlUpcaseUnicodeString", &[b, a, 0]),
            status::SUCCESS
        );
        assert_eq!(text_of(&k, b), "HELLO, WORLD!");
        assert_eq!(
            call(&mut k, "RtlDowncaseUnicodeString", &[b, a, 1]),
            status::SUCCESS
        );
        assert_eq!(text_of(&k, b), "hello, world!");
        assert_ne!(
            Counted::read(&k, b).unwrap().buffer,
            SCRATCH + 0x108,
            "a new buffer"
        );
        assert_eq!(
            call(&mut k, "RtlUpcaseUnicodeChar", &[u32::from(b'q')]),
            u32::from(b'Q')
        );
        assert_eq!(call(&mut k, "RtlDowncaseUnicodeChar", &[0xC9]), 0xE9);
    }

    #[test]
    fn create_unicode_string_allocates_a_terminated_copy() {
        let mut k = kernel();
        let (dest, source) = (SCRATCH, SCRATCH + 0x100);
        k.write(source, &[b'a', 0, b'b', 0, 0, 0]).unwrap();
        assert_eq!(call(&mut k, "RtlCreateUnicodeString", &[dest, source]), 1);
        let counted = Counted::read(&k, dest).unwrap();
        assert_eq!((counted.length, counted.maximum), (4, 6));
        assert_eq!(text_of(&k, dest), "ab");
        assert_eq!(k.r16(counted.buffer + 4), Ok(0));
    }

    #[test]
    fn integers_to_and_from_text() {
        let mut k = kernel();
        let (text, out) = (SCRATCH, SCRATCH + 0x100);
        k.write(text, b" -0x10\0").unwrap();
        assert_eq!(
            call(&mut k, "RtlCharToInteger", &[text, 0, out]),
            status::SUCCESS
        );
        assert_eq!(k.r32(out), Ok(16u32.wrapping_neg()));
        assert_eq!(
            call(&mut k, "RtlCharToInteger", &[text, 7, out]),
            status::INVALID_PARAMETER
        );
        let wide = unicode(&k, SCRATCH + 0x200, "123", 16);
        assert_eq!(
            call(&mut k, "RtlUnicodeStringToInteger", &[wide, 8, out]),
            status::SUCCESS
        );
        assert_eq!(k.r32(out), Ok(0o123));
        let buffer = SCRATCH + 0x300;
        k.write(buffer, &[0xEE; 8]).unwrap();
        assert_eq!(
            call(&mut k, "RtlIntegerToChar", &[255, 16, 8, buffer]),
            status::SUCCESS
        );
        assert_eq!(k.read(buffer, 3).unwrap(), b"FF\0");
        assert_eq!(
            call(&mut k, "RtlIntegerToChar", &[255, 2, 4, buffer]),
            status::BUFFER_OVERFLOW
        );
        let string = unicode(&k, SCRATCH + 0x400, "", 16);
        assert_eq!(
            call(&mut k, "RtlIntegerToUnicodeString", &[1234, 10, string]),
            status::SUCCESS
        );
        assert_eq!(text_of(&k, string), "1234");
    }

    #[test]
    fn multi_byte_conversions_are_latin_1() {
        let mut k = kernel();
        let (narrow_at, wide_at, count) = (SCRATCH, SCRATCH + 0x100, SCRATCH + 0x200);
        k.write(narrow_at, b"abc\xE9").unwrap();
        assert_eq!(
            call(
                &mut k,
                "RtlMultiByteToUnicodeN",
                &[wide_at, 6, count, narrow_at, 4]
            ),
            status::SUCCESS
        );
        assert_eq!(k.r32(count), Ok(6), "three units fit");
        assert_eq!(k.read(wide_at, 6).unwrap(), [b'a', 0, b'b', 0, b'c', 0]);
        assert_eq!(
            call(&mut k, "RtlMultiByteToUnicodeSize", &[count, narrow_at, 4]),
            status::SUCCESS
        );
        assert_eq!(k.r32(count), Ok(8));
        k.write(wide_at, &[b'x', 0, 0xE9, 0, 0x3A, 0x26]).unwrap();
        assert_eq!(
            call(
                &mut k,
                "RtlUnicodeToMultiByteN",
                &[narrow_at, 8, count, wide_at, 6]
            ),
            status::SUCCESS
        );
        assert_eq!(k.r32(count), Ok(3));
        assert_eq!(k.read(narrow_at, 3).unwrap(), b"x\xE9?");
        assert_eq!(
            call(
                &mut k,
                "RtlUpcaseUnicodeToMultiByteN",
                &[narrow_at, 2, 0, wide_at, 6]
            ),
            status::SUCCESS
        );
        assert_eq!(k.read(narrow_at, 3).unwrap(), b"X\xC9?");
        assert_eq!(
            call(&mut k, "RtlUnicodeToMultiByteSize", &[count, wide_at, 6]),
            status::SUCCESS
        );
        assert_eq!(k.r32(count), Ok(3));
    }
}
