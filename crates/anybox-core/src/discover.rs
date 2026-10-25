//! Finds the functions of a title: the sections of its executable, and the addresses that start
//! code.

use std::path::Path;

use lift::Image;

/// Returns the title's image with the address ranges of its executable sections and its entry
/// point.
pub(crate) fn load_image(xbe: &xbe::Xbe) -> (Image, Vec<std::ops::Range<u32>>, u32) {
    let mut image = Image::default();
    let mut code = Vec::new();
    for section in xbe.sections() {
        let data = xbe.section_data(section);
        image.push(section.virtual_address(), data.to_vec());
        if is_code_section(section.name(), section.flags().executable(), data) {
            let len = u32::try_from(data.len()).unwrap_or(u32::MAX);
            code.push(section.virtual_address()..section.virtual_address() + len);
        }
    }
    (image, code, xbe.entry_point())
}

/// The bytes of `default.xbe` on the disc at `iso`.
pub(crate) fn read_default_xbe(iso: &Path) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(iso).map_err(|error| format!("{}: {error}", iso.display()))?;
    let mut disc = xiso::XIso::open(file).map_err(|error| format!("{error:?}"))?;
    let entry = disc
        .lookup("default.xbe")
        .map_err(|error| format!("{error:?}"))?
        .ok_or_else(|| "the disc has no default.xbe".to_owned())?;
    disc.read_file(&entry).map_err(|error| format!("{error:?}"))
}

/// The bytes of one code range.
pub(crate) fn range_bytes<'a>(image: &'a Image, range: &std::ops::Range<u32>) -> &'a [u8] {
    let bytes = image.at(range.start).unwrap_or(&[]);
    &bytes[..bytes.len().min((range.end - range.start) as usize)]
}

/// The start addresses of the functions that begin with a frame prologue (`push ebp; mov ebp,
/// esp`, with the hot-patch `mov edi, edi` form) in the code ranges.
pub(crate) fn prologue_starts(image: &Image, code: &[std::ops::Range<u32>]) -> Vec<u32> {
    let mut starts = Vec::new();
    for range in code {
        let bytes = range_bytes(image, range);
        for offset in 0..bytes.len().saturating_sub(4) {
            let here = &bytes[offset..];
            if here.starts_with(&[0x55, 0x8B, 0xEC])
                || here.starts_with(&[0x8B, 0xFF, 0x55, 0x8B, 0xEC])
            {
                starts.push(range.start + u32::try_from(offset).unwrap_or(u32::MAX));
            }
        }
    }
    starts
}

/// Addresses that follow a `ret`, a `ret imm16`, or a run of `nop` or `int3` padding in the code
/// ranges: where the next function begins, including leaf functions with no frame prologue. An address inside an instruction only adds a function that may fail to lift.
pub(crate) fn boundary_starts(image: &Image, code: &[std::ops::Range<u32>]) -> Vec<u32> {
    let mut starts = Vec::new();
    for range in code {
        let bytes = range_bytes(image, range);
        let mut padding = 0usize;
        for at in 0..bytes.len() {
            let byte = bytes[at];
            if matches!(byte, 0xCC | 0x90) {
                padding += 1;
                continue;
            }
            // The return may be followed by padding of any length, even a single byte.
            let end = at - padding;
            let after_ret =
                end >= 1 && bytes[end - 1] == 0xC3 || end >= 3 && bytes[end - 3] == 0xC2;
            if (after_ret || padding >= 2) && byte != 0x00 {
                starts.push(range.start + u32::try_from(at).unwrap_or(u32::MAX));
            }
            padding = 0;
        }
    }
    starts
}

/// Addresses of code that the code names as 32-bit immediates, found by decoding the code ranges
/// linearly: `push`, `mov` to a register or to memory, `cmp`, and so on, whose value is inside a
/// code range. A thread routine, a callback or a vtable entry is usually built this way.
pub(crate) fn code_addresses(image: &Image, code: &[std::ops::Range<u32>]) -> Vec<u32> {
    use iced_x86::{Decoder, DecoderOptions, OpKind};
    let mut found = Vec::new();
    for range in code {
        let bytes = range_bytes(image, range);
        let mut decoder = Decoder::with_ip(32, bytes, u64::from(range.start), DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                continue;
            }
            for operand in 0..instruction.op_count() {
                if matches!(
                    instruction.op_kind(operand),
                    OpKind::Immediate32 | OpKind::Immediate32to64
                ) {
                    let value =
                        u32::try_from(instruction.immediate(operand) & 0xFFFF_FFFF).unwrap_or(0);
                    if code.iter().any(|r| r.contains(&value)) {
                        found.push(value);
                    }
                }
            }
        }
    }
    found
}

/// Addresses of code that data names: any dword, at any offset, outside the code ranges that
/// points into one,
/// such as a function table or a callback stored in a structure.
///
/// Only addresses where a linear decode of the code puts an instruction count, or that start a
/// function as compilers lay one out: on a 16-byte boundary, or with an exception frame. Data is
/// full of numbers that happen to fall in the code's range, and most land inside an instruction:
/// in Blinx two in three of the 200,000 found did, each lifted as a function of its own. But a
/// linear decode loses step over a `switch`'s jump table inside the code, and a function right
/// after one is missed by it: Dragon's Lair 3D's at 0x172670, named only by a vtable, which opens
/// with an exception frame rather than `push ebp`.
pub(crate) fn data_addresses(image: &Image, code: &[std::ops::Range<u32>]) -> Vec<u32> {
    let starts: Vec<Vec<bool>> = code
        .iter()
        .map(|range| instruction_starts(range.start, range_bytes(image, range)))
        .collect();
    let starts_code = |value: u32| {
        code.iter().zip(&starts).any(|(range, starts)| {
            range.contains(&value)
                && (starts.get((value - range.start) as usize) == Some(&true)
                    || value.is_multiple_of(16)
                    || starts_exception_frame(image, value))
        })
    };
    let mut found = Vec::new();
    for base in image.bases() {
        if code.iter().any(|r| r.start == base) {
            continue;
        }
        let Some(bytes) = image.at(base) else {
            continue;
        };
        // Every byte offset: structures in the title's data are not always dword-aligned.
        for word in bytes.windows(4) {
            let value = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            if starts_code(value) {
                found.push(value);
            }
        }
    }
    found
}

/// Whether the code at `address` sets up an MSVC exception frame, as a function compiled with
/// exception handling starts: `push -1; push handler; mov eax, fs:[0]`.
fn starts_exception_frame(image: &Image, address: u32) -> bool {
    image.at(address).is_some_and(|bytes| {
        bytes.len() >= 13
            && bytes[..3] == [0x6A, 0xFF, 0x68]
            && bytes[7..13] == [0x64, 0xA1, 0x00, 0x00, 0x00, 0x00]
    })
}

/// For each byte of `bytes`, decoded linearly from `base`, whether an instruction starts there.
fn instruction_starts(base: u32, bytes: &[u8]) -> Vec<bool> {
    use iced_x86::{Decoder, DecoderOptions};
    let mut starts = vec![false; bytes.len()];
    let mut decoder = Decoder::with_ip(32, bytes, u64::from(base), DecoderOptions::NONE);
    while decoder.can_decode() {
        let at = decoder.position();
        starts[at] = true;
        let _ = decoder.decode();
    }
    starts
}

/// The sections the XDK's own libraries are linked into: code, whatever their bytes look like.
/// Built without frame pointers and with tables inside the code, WMADEC and WMVDEC read as data
/// to [`looks_like_code`], which cost Dragon's Lair 3D 700 functions.
const LIBRARY_SECTIONS: [&str; 15] = [
    "D3D", "D3DX", "DSOUND", "WMADEC", "WMVDEC", "XMV", "XGRPH", "XPP", "XACTENG", "XNET", "XNETS",
    "XNETN", "XONLINE", "XONLINES", "XVOICE",
];

/// Whether a section holds code. The XBE executable flag is also set on `.rdata` and `.data`, so
/// the name decides first: `.text` and the XDK libraries' sections are code; the other
/// dot-sections, the `$$` image sections and the audio DSP image are data. A title's own named
/// sections may be either (Blinx keeps tens of megabytes of models and maps in executable
/// sections), so the rest must also look like code.
pub(crate) fn is_code_section(name: &str, executable: bool, bytes: &[u8]) -> bool {
    name == ".text"
        || (executable && LIBRARY_SECTIONS.contains(&name))
        || (executable
            && !name.starts_with('.')
            && !name.starts_with("$$")
            && name != "DOLBY"
            && looks_like_code(bytes))
}

/// Whether `bytes` read as compiled x86: frame prologues are common, or decoding them linearly
/// rarely meets what compilers do not emit (invalid encodings, segment registers other than FS,
/// far branches, port I/O, BCD arithmetic). Data decodes to about one such instruction in ten,
/// code to a few in a hundred at most; a library built without frame pointers still passes on
/// the second test, and one that mixes in tables on the first.
fn looks_like_code(bytes: &[u8]) -> bool {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind, Register};
    if bytes.is_empty() {
        return false;
    }
    let prologues = bytes
        .windows(3)
        .filter(|w| *w == [0x55, 0x8B, 0xEC])
        .count();
    // At least 64 a megabyte: real code has hundreds, data almost none.
    if prologues * (1 << 20) >= 64 * bytes.len() {
        return true;
    }
    let mut decoder = Decoder::new(32, bytes, DecoderOptions::NONE);
    let (mut total, mut unusual) = (0usize, 0usize);
    while decoder.can_decode() {
        let instruction = decoder.decode();
        total += 1;
        let segment =
            |register: Register| register.is_segment_register() && register != Register::FS;
        let odd = instruction.is_invalid()
            || segment(instruction.segment_prefix())
            || (0..instruction.op_count()).any(|operand| {
                instruction.op_kind(operand) == OpKind::Register
                    && segment(instruction.op_register(operand))
            })
            || instruction.is_call_far()
            || instruction.is_jmp_far()
            || matches!(
                instruction.mnemonic(),
                Mnemonic::Int
                    | Mnemonic::Into
                    | Mnemonic::Int1
                    | Mnemonic::Hlt
                    | Mnemonic::Les
                    | Mnemonic::Lds
                    | Mnemonic::In
                    | Mnemonic::Out
                    | Mnemonic::Insb
                    | Mnemonic::Insw
                    | Mnemonic::Insd
                    | Mnemonic::Outsb
                    | Mnemonic::Outsw
                    | Mnemonic::Outsd
                    | Mnemonic::Arpl
                    | Mnemonic::Bound
                    | Mnemonic::Aaa
                    | Mnemonic::Aas
                    | Mnemonic::Daa
                    | Mnemonic::Das
                    | Mnemonic::Aam
                    | Mnemonic::Aad
                    | Mnemonic::Salc
                    | Mnemonic::Iretd
                    | Mnemonic::Retf
            );
        unusual += usize::from(odd);
    }
    // Under one in twenty.
    unusual * 20 < total
}

/// Reports what the lifter could not translate, grouped by reason, most frequent first.
pub(crate) fn report_traps(
    traps: &[lift::LiftError],
    failures: &[lift::LiftError],
    reporter: &dyn crate::progress::Reporter,
) {
    let mut reasons: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for trap in traps.iter().chain(failures) {
        let reason: String = trap
            .message
            .split(':')
            .next()
            .unwrap_or("")
            .chars()
            .take(60)
            .collect();
        *reasons.entry(reason).or_default() += 1;
    }
    let mut sorted: Vec<_> = reasons.into_iter().collect();
    sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (reason, count) in sorted.iter().take(40) {
        crate::progress::note(reporter, format!("  {count:6}  {reason}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_names_code_only_where_an_instruction_starts() {
        let mut image = Image::default();
        // `mov eax, 1 ; ret` at 0x1000.
        image.push(0x1000, vec![0xB8, 1, 0, 0, 0, 0xC3]);
        // Data naming the `mov`, the middle of its immediate, and the `ret`.
        let data: Vec<u8> = [0x1000u32, 0x1002, 0x1005]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        image.push(0x2000, data);
        #[allow(clippy::single_range_in_vec_init)] // one code range, not the addresses in it
        let found = data_addresses(&image, &[0x1000..0x1006]);
        assert_eq!(found, [0x1000, 0x1005]);
    }

    #[test]
    fn data_names_a_function_the_linear_decode_misses_on_a_boundary_or_with_a_frame() {
        let mut image = Image::default();
        // Padding, then `mov eax, imm32` over 0x100E..0x1013: 0x1010 and 0x1011 are inside it.
        let mut code = vec![0x90; 14];
        code.extend([0xB8, 0, 0, 0, 0]);
        // `mov eax, imm32` at 0x1013 whose immediate hides an exception frame at 0x1015, as a
        // decode that lost step over a jump table sees a function after it.
        code.extend([0xB8, 0x00]);
        code.extend([
            0x6A, 0xFF, 0x68, 0x00, 0x20, 0x00, 0x00, 0x64, 0xA1, 0, 0, 0, 0,
        ]);
        code.push(0xC3);
        let end = 0x1000 + u32::try_from(code.len()).unwrap();
        image.push(0x1000, code);
        let data: Vec<u8> = [0x1010u32, 0x1011, 0x1015, 0x1016]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        image.push(0x3000, data);
        #[allow(clippy::single_range_in_vec_init)] // one code range, not the addresses in it
        let found = data_addresses(&image, &[0x1000..end]);
        assert_eq!(
            found,
            [0x1010, 0x1015],
            "a 16-byte boundary and an exception frame"
        );
    }

    #[test]
    fn an_xdk_library_section_is_code_whatever_its_bytes_look_like() {
        let mut state = 0x1234_5678u32;
        let noise: Vec<u8> = (0..1 << 16)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state.to_le_bytes()[3]
            })
            .collect();
        assert!(is_code_section("WMVDEC", true, &noise));
        assert!(is_code_section("WMADEC", true, &noise));
        assert!(
            !is_code_section("WMVDEC", false, &noise),
            "not when it is not executable"
        );
        assert!(!is_code_section("MAP13", true, &noise));
    }

    #[test]
    fn a_named_section_is_code_only_when_its_bytes_read_as_code() {
        // `push ebp; mov ebp, esp; mov eax, [ebp+8]; pop ebp; ret`, over and over.
        let framed = [0x55, 0x8B, 0xEC, 0x8B, 0x45, 0x08, 0x5D, 0xC3].repeat(512);
        assert!(is_code_section("D3D", true, &framed));
        // No frames: `mov eax, [esp+4]; add eax, ecx; ret 4`.
        let leaf = [0x8B, 0x44, 0x24, 0x04, 0x03, 0xC1, 0xC2, 0x04, 0x00].repeat(512);
        assert!(is_code_section("WMADEC", true, &leaf));
        // Bytes with no structure, as packed models and maps look to a decoder.
        let mut state = 0x1234_5678u32;
        let noise: Vec<u8> = (0..1 << 16)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state.to_le_bytes()[3]
            })
            .collect();
        assert!(!is_code_section("MAP13", true, &noise));
        // The name still decides for `.text` and the data sections.
        assert!(is_code_section(".text", true, &noise));
        assert!(!is_code_section(".data", true, &framed));
        assert!(!is_code_section("D3D", false, &framed));
    }

    /// `bytes` as one code range at 0x1000, and the boundaries found in it.
    #[allow(clippy::single_range_in_vec_init)] // one code range, not the addresses in it
    fn boundaries(bytes: &[u8]) -> Vec<u32> {
        let mut image = Image::default();
        image.push(0x1000, bytes.to_vec());
        let end = 0x1000 + u32::try_from(bytes.len()).unwrap();
        boundary_starts(&image, &[0x1000..end])
    }

    #[test]
    fn a_function_starts_after_a_return_and_its_padding() {
        // `ret`, then `push esi`.
        assert_eq!(boundaries(&[0xC3, 0x56]), [0x1001]);
        // `ret 4`, then two `nop`s.
        assert_eq!(boundaries(&[0xC2, 0x04, 0x00, 0x90, 0x90, 0x56]), [0x1005]);
        // `ret 4`, one `nop`, then `push esi`: Marvel vs. Capcom 2's render-state handler at
        // 0x284CE0, named only by a table inside the D3D section.
        assert_eq!(boundaries(&[0xC2, 0x04, 0x00, 0x90, 0x56]), [0x1004]);
        // `ret`, one `int3`, then code.
        assert_eq!(boundaries(&[0xC3, 0xCC, 0x56]), [0x1002]);
        // One `nop` inside code is not a boundary.
        assert_eq!(boundaries(&[0x56, 0x90, 0x57]), []);
    }
}
