//! The run-time library's formatted printing: `RtlSprintf`, `RtlSnprintf` and their `va_list`
//! forms. The format is the C library's, with the kernel's additions: `%S` and `%ws` for UTF-16
//! strings, `%Z` and `%wZ` for counted strings, and `I64` for 64-bit integers.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission, and adapted
//! to this crate's handler interface.

use crate::kernel::{Call, CallResult, KResult, Kernel};

/// Where the values a format consumes come from: consecutive dwords at `next`.
struct Values<'a> {
    kernel: &'a Kernel,
    next: u32,
}

impl Values<'_> {
    fn u32(&mut self) -> KResult<u32> {
        let value = self.kernel.r32(self.next)?;
        self.next = self.next.wrapping_add(4);
        Ok(value)
    }

    fn u64(&mut self) -> KResult<u64> {
        let low = u64::from(self.u32()?);
        let high = u64::from(self.u32()?);
        Ok(high << 32 | low)
    }
}

/// What a conversion's flags, width, precision and length ask for: the C library's, so many
/// switches.
#[derive(Debug, Default, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
struct Spec {
    left: bool,
    plus: bool,
    space: bool,
    alternate: bool,
    zero: bool,
    width: usize,
    precision: Option<usize>,
    /// `I64` or `ll`: a 64-bit integer.
    long64: bool,
    /// `l` or `w`: wide characters and strings.
    wide: bool,
    /// `h`: a 16-bit integer.
    short: bool,
}

impl Spec {
    /// Pads `body` (after `prefix`, such as a sign) to the width.
    fn pad(self, prefix: &str, body: &[u8], numeric: bool) -> Vec<u8> {
        let len = prefix.len() + body.len();
        let fill = self.width.saturating_sub(len);
        let mut out = Vec::with_capacity(len + fill);
        if self.left {
            out.extend_from_slice(prefix.as_bytes());
            out.extend_from_slice(body);
            out.resize(out.len() + fill, b' ');
        } else if self.zero && numeric && self.precision.is_none() {
            out.extend_from_slice(prefix.as_bytes());
            out.resize(out.len() + fill, b'0');
            out.extend_from_slice(body);
        } else {
            out.resize(fill, b' ');
            out.extend_from_slice(prefix.as_bytes());
            out.extend_from_slice(body);
        }
        out
    }

    /// Formats an integer's digits with the precision as a minimum digit count.
    fn integer(self, digits: String, prefix: &str) -> Vec<u8> {
        let digits = match self.precision {
            Some(0) if digits == "0" => String::new(),
            Some(min) if digits.len() < min => {
                format!("{}{digits}", "0".repeat(min - digits.len()))
            }
            _ => digits,
        };
        self.pad(prefix, digits.as_bytes(), true)
    }

    /// The sign or blank a signed number starts with.
    fn sign(self, negative: bool) -> &'static str {
        if negative {
            "-"
        } else if self.plus {
            "+"
        } else if self.space {
            " "
        } else {
            ""
        }
    }

    /// Takes an integer argument of the spec's length.
    fn integer_value(self, values: &mut Values<'_>) -> KResult<u64> {
        Ok(if self.long64 {
            values.u64()?
        } else {
            u64::from(values.u32()?)
        })
    }
}

/// Reads a decimal number at `at`.
fn number(format: &[u8], at: &mut usize) -> usize {
    let mut value = 0usize;
    while let Some(digit) = format.get(*at).filter(|b| b.is_ascii_digit()) {
        value = value
            .saturating_mul(10)
            .saturating_add(usize::from(digit - b'0'));
        *at += 1;
    }
    value
}

/// Reads a conversion's flags, width, precision and length, which start at `at`; a `*` takes
/// its value from the arguments.
fn parse_spec(format: &[u8], at: &mut usize, values: &mut Values<'_>) -> KResult<Spec> {
    let mut spec = Spec::default();
    while let Some(&flag) = format.get(*at) {
        match flag {
            b'-' => spec.left = true,
            b'+' => spec.plus = true,
            b' ' => spec.space = true,
            b'#' => spec.alternate = true,
            b'0' => spec.zero = true,
            _ => break,
        }
        *at += 1;
    }
    if format.get(*at) == Some(&b'*') {
        let width = values.u32()?.cast_signed();
        spec.left |= width < 0;
        spec.width = width.unsigned_abs() as usize;
        *at += 1;
    } else {
        spec.width = number(format, at);
    }
    if format.get(*at) == Some(&b'.') {
        *at += 1;
        spec.precision = Some(if format.get(*at) == Some(&b'*') {
            *at += 1;
            values.u32()?.cast_signed().max(0).unsigned_abs() as usize
        } else {
            number(format, at)
        });
    }
    loop {
        match format.get(*at..).unwrap_or_default() {
            [b'I', b'6', b'4', ..] => {
                spec.long64 = true;
                *at += 3;
            }
            [b'I', b'3', b'2', ..] => *at += 3,
            [b'l', b'l', ..] => {
                spec.long64 = true;
                *at += 2;
            }
            [b'l' | b'w', ..] => {
                spec.wide = true;
                *at += 1;
            }
            [b'h', ..] => {
                spec.short = true;
                *at += 1;
            }
            _ => break,
        }
    }
    Ok(spec)
}

/// Formats `format` with the values at `values`, as the C library's `sprintf` does.
fn format(k: &Kernel, format: &[u8], values: &mut Values<'_>) -> KResult<Vec<u8>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < format.len() {
        let byte = format[at];
        at += 1;
        if byte != b'%' {
            out.push(byte);
            continue;
        }
        let spec = parse_spec(format, &mut at, values)?;
        let Some(&conversion) = format.get(at) else {
            break;
        };
        at += 1;
        out.extend(convert(k, conversion, spec, values)?);
    }
    Ok(out)
}

/// The text of one conversion.
fn convert(k: &Kernel, conversion: u8, spec: Spec, values: &mut Values<'_>) -> KResult<Vec<u8>> {
    Ok(match conversion {
        b'%' => vec![b'%'],
        b'd' | b'i' => {
            let raw = spec.integer_value(values)?;
            let value: i64 = if spec.long64 {
                raw.cast_signed()
            } else if spec.short {
                i64::from((raw as u16).cast_signed())
            } else {
                i64::from((raw as u32).cast_signed())
            };
            spec.integer(value.unsigned_abs().to_string(), spec.sign(value < 0))
        }
        b'u' | b'x' | b'X' | b'o' => {
            let mut value = spec.integer_value(values)?;
            if spec.short {
                value &= 0xFFFF;
            }
            let marked = spec.alternate && value != 0;
            let (digits, prefix) = match conversion {
                b'u' => (value.to_string(), ""),
                b'x' => (format!("{value:x}"), if marked { "0x" } else { "" }),
                b'X' => (format!("{value:X}"), if marked { "0X" } else { "" }),
                _ => (format!("{value:o}"), if marked { "0" } else { "" }),
            };
            spec.integer(digits, prefix)
        }
        b'p' => {
            let value = values.u32()?;
            spec.pad("", format!("{value:08X}").as_bytes(), false)
        }
        b'c' | b'C' => {
            let value = values.u32()?;
            // %c is a byte, %C (or %lc) a UTF-16 unit, narrowed.
            let byte = if conversion == b'C' || spec.wide {
                u8::try_from(value & 0xFFFF).unwrap_or(b'?')
            } else {
                value as u8
            };
            spec.pad("", &[byte], false)
        }
        b's' | b'S' => {
            let address = values.u32()?;
            let mut text = if address == 0 {
                b"(null)".to_vec()
            } else if (conversion == b'S') != spec.wide {
                narrow_text(&read_wide_c(k, address)?)
            } else {
                k.c_string(address)?
            };
            if let Some(precision) = spec.precision {
                text.truncate(precision);
            }
            spec.pad("", &text, false)
        }
        b'Z' => {
            // A counted string: `%Z` 8-bit, `%wZ` UTF-16.
            let address = values.u32()?;
            let text = if address == 0 {
                b"(null)".to_vec()
            } else {
                counted_text(k, address, spec.wide)?
            };
            spec.pad("", &text, false)
        }
        b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
            let value = f64::from_bits(values.u64()?);
            let precision = spec.precision.unwrap_or(6);
            let body = match conversion {
                b'f' | b'F' => format!("{:.*}", precision, value.abs()),
                b'e' => format!("{:.*e}", precision, value.abs()),
                b'E' => format!("{:.*E}", precision, value.abs()),
                b'G' => value.abs().to_string().to_uppercase(),
                _ => value.abs().to_string(),
            };
            let sign = if value.is_sign_negative() {
                "-"
            } else {
                spec.sign(false)
            };
            spec.pad(sign, body.as_bytes(), true)
        }
        b'n' => {
            // Would store how much has been produced; the count is not kept.
            let _ = values.u32()?;
            Vec::new()
        }
        other => vec![other],
    })
}

/// Reads a NUL-terminated UTF-16 string (at most 32 Ki units), without the NUL.
fn read_wide_c(k: &Kernel, address: u32) -> KResult<Vec<u16>> {
    let mut text = Vec::new();
    for i in 0..0x8000u32 {
        let unit = k.r16(address.wrapping_add(2 * i))?;
        if unit == 0 {
            break;
        }
        text.push(unit);
    }
    Ok(text)
}

fn narrow_text(text: &[u16]) -> Vec<u8> {
    text.iter()
        .map(|&unit| u8::try_from(unit).unwrap_or(b'?'))
        .collect()
}

fn counted_text(k: &Kernel, address: u32, wide: bool) -> KResult<Vec<u8>> {
    let length = k.r16(address)?;
    let buffer = k.r32(address + 4)?;
    if buffer == 0 {
        return Ok(Vec::new());
    }
    if wide {
        let bytes = k.read(buffer, u32::from(length & !1))?;
        let units: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|unit| u16::from_le_bytes(*unit))
            .collect();
        Ok(narrow_text(&units))
    } else {
        k.read(buffer, u32::from(length))
    }
}

/// Formats into `buffer`. With a `size`, at most that many bytes are written: when the text does
/// not fit, no NUL is added and the result is -1, as the C library's `_snprintf` does.
fn print(
    k: &Kernel,
    buffer: u32,
    size: Option<u32>,
    format_at: u32,
    values_at: u32,
) -> KResult<CallResult> {
    let pattern = k.c_string(format_at)?;
    let text = format(
        k,
        &pattern,
        &mut Values {
            kernel: k,
            next: values_at,
        },
    )?;
    let len = u32::try_from(text.len()).unwrap_or(u32::MAX);
    match size {
        Some(size) if len >= size => {
            k.write(buffer, &text[..size as usize])?;
            Ok(CallResult::Value(if len == size { len } else { u32::MAX }))
        }
        _ => {
            k.write(buffer, &text)?;
            k.w8(buffer.wrapping_add(len), 0)?;
            Ok(CallResult::Value(len))
        }
    }
}

/// The address of the cdecl argument after the first `fixed` ones.
fn variadic(c: &Call, fixed: u32) -> u32 {
    c.esp.wrapping_add(4 + 4 * fixed)
}

/// `RtlSprintf(Buffer, Format, ...)`.
pub fn sprintf(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    print(k, c.arg(0), None, c.arg(1), variadic(c, 2))
}

/// `RtlSnprintf(Buffer, Size, Format, ...)`.
pub fn snprintf(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    print(k, c.arg(0), Some(c.arg(1)), c.arg(2), variadic(c, 3))
}

/// `RtlVsprintf(Buffer, Format, Arguments)`.
pub fn vsprintf(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    print(k, c.arg(0), None, c.arg(1), c.arg(2))
}

/// `RtlVsnprintf(Buffer, Size, Format, Arguments)`.
pub fn vsnprintf(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    print(k, c.arg(0), Some(c.arg(1)), c.arg(2), c.arg(3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::handlers::lookup;
    use crate::kernel::testing::kernel;

    /// Guest scratch memory inside the committed heap.
    const SCRATCH: u32 = crate::mem::HEAP_BASE + 0x2F_0000;
    const WORDS: u32 = SCRATCH + 0x100;
    const HELLO: u32 = SCRATCH + 0x200;
    const HI: u32 = SCRATCH + 0x300;
    const COUNTED: u32 = SCRATCH + 0x400;

    /// A kernel with "hello" at `HELLO`, UTF-16 "hi" at `HI`, and a counted "hello" at `COUNTED`.
    fn setup() -> Kernel {
        let k = kernel();
        k.write(HELLO, b"hello\0").unwrap();
        k.write(HI, &[b'h', 0, b'i', 0, 0, 0]).unwrap();
        k.w16(COUNTED, 5).unwrap();
        k.w16(COUNTED + 2, 6).unwrap();
        k.w32(COUNTED + 4, HELLO).unwrap();
        k
    }

    fn printed(pattern: &str, words: &[u32]) -> String {
        let k = setup();
        for (index, word) in (0u32..).zip(words) {
            k.w32(WORDS + 4 * index, *word).unwrap();
        }
        let mut values = Values {
            kernel: &k,
            next: WORDS,
        };
        String::from_utf8(format(&k, pattern.as_bytes(), &mut values).unwrap()).unwrap()
    }

    #[test]
    fn integers_strings_and_padding_print_as_in_c() {
        assert_eq!(
            printed("%d|%5d|%-5d|%05d", &[42, 7, 7, (-7i32).cast_unsigned()]),
            "42|    7|7    |-0007"
        );
        assert_eq!(
            printed("%x %X %#x %08X", &[255, 255, 255, 0xBEEF]),
            "ff FF 0xff 0000BEEF"
        );
        assert_eq!(
            printed("%u %c%%", &[u32::MAX, u32::from(b'A')]),
            "4294967295 A%"
        );
        assert_eq!(
            printed("[%s] [%.3s] [%7s]", &[HELLO, HELLO, HELLO]),
            "[hello] [hel] [  hello]"
        );
        assert_eq!(printed("%S %ws", &[HI, HI]), "hi hi");
        assert_eq!(printed("%I64u", &[0, 1]), "4294967296");
        assert_eq!(printed("%.2f", &[0, 0x4009_21FB]), "3.14");
        assert_eq!(printed("%*d", &[4, 9]), "   9");
        assert_eq!(printed("%Z|%s", &[COUNTED, 0]), "hello|(null)");
        assert_eq!(printed("%hd %p", &[0xFFFF, 0x1234]), "-1 00001234");
    }

    /// Calls `name` with `args` and the stack holding the return address then `args`, as a cdecl
    /// call sees it.
    fn call(k: &mut Kernel, name: &str, args: &[u32]) -> u32 {
        let esp = SCRATCH + 0x800;
        k.w32(esp, 0).unwrap();
        for (i, arg) in (0u32..).zip(args) {
            k.w32(esp + 4 + 4 * i, *arg).unwrap();
        }
        let handler = lookup(name).unwrap();
        let c = Call {
            args: args.to_vec(),
            esp,
        };
        match handler(k, &c) {
            Ok(CallResult::Value(v)) => v,
            other => panic!("{name} returned {other:?}"),
        }
    }

    #[test]
    fn the_exports_take_variadic_or_va_list_values_and_truncate_with_a_size() {
        let mut k = setup();
        let (buffer, format) = (SCRATCH + 0xA00, SCRATCH + 0xB00);
        k.write(format, b"%s=%d\0").unwrap();
        assert_eq!(call(&mut k, "RtlSprintf", &[buffer, format, HELLO, 42]), 8);
        assert_eq!(k.read(buffer, 9).unwrap(), b"hello=42\0");
        k.write(buffer, &[0xEE; 16]).unwrap();
        assert_eq!(
            call(&mut k, "RtlSnprintf", &[buffer, 4, format, HELLO, 42]),
            u32::MAX,
            "too long"
        );
        assert_eq!(k.read(buffer, 5).unwrap(), b"hell\xEE", "no NUL added");
        assert_eq!(
            call(&mut k, "RtlSnprintf", &[buffer, 8, format, HELLO, 42]),
            8,
            "an exact fit"
        );
        let list = SCRATCH + 0xC00;
        k.w32(list, HELLO).unwrap();
        k.w32(list + 4, 7).unwrap();
        assert_eq!(call(&mut k, "RtlVsprintf", &[buffer, format, list]), 7);
        assert_eq!(k.read(buffer, 8).unwrap(), b"hello=7\0");
        assert_eq!(call(&mut k, "RtlVsnprintf", &[buffer, 16, format, list]), 7);
    }
}
