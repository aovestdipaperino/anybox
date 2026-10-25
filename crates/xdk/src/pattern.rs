//! Byte patterns with wildcards, and a generator that builds them from code.

use std::fmt;
use std::ops::Range;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind};

/// A byte pattern in which some bytes match anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    bytes: Vec<Option<u8>>,
}

/// Why a pattern string could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternError {
    token: String,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid pattern byte `{}`; expected two hex digits or `??`",
            self.token
        )
    }
}

impl std::error::Error for PatternError {}

impl Pattern {
    /// Parses space-separated hex bytes, where `??` is a wildcard.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first token that is neither a byte nor `??`.
    pub fn parse(text: &str) -> Result<Self, PatternError> {
        let bytes = text
            .split_whitespace()
            .map(|token| {
                if token == "??" {
                    Ok(None)
                } else if token.len() == 2 {
                    u8::from_str_radix(token, 16)
                        .map(Some)
                        .map_err(|_| PatternError {
                            token: token.to_owned(),
                        })
                } else {
                    Err(PatternError {
                        token: token.to_owned(),
                    })
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { bytes })
    }

    /// Returns the pattern's length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns `true` for the pattern with no bytes, which matches everywhere.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns `true` when `data` starts with this pattern.
    #[must_use]
    pub fn matches(&self, data: &[u8]) -> bool {
        data.len() >= self.bytes.len()
            && self
                .bytes
                .iter()
                .zip(data)
                .all(|(wanted, &actual)| wanted.is_none_or(|byte| byte == actual))
    }

    /// Returns the offset of every match in `data`.
    #[must_use]
    pub fn find_all(&self, data: &[u8]) -> Vec<usize> {
        if self.bytes.is_empty() || data.len() < self.bytes.len() {
            return Vec::new();
        }
        (0..=data.len() - self.bytes.len())
            .filter(|&offset| self.matches(&data[offset..]))
            .collect()
    }

    /// Builds a pattern from known code, masking what differs between links.
    ///
    /// `code` starts at guest address `base`. Relative `call` and `jmp`
    /// targets are masked, because they depend on where the function sits,
    /// and so is any immediate or displacement that looks like an address
    /// inside `image`. Decoding stops at the first invalid instruction or
    /// after `max_len` bytes, and the pattern covers whole instructions only.
    #[must_use]
    pub fn from_code(code: &[u8], base: u32, image: &Range<u32>, max_len: usize) -> Self {
        let mut decoder = Decoder::with_ip(32, code, u64::from(base), DecoderOptions::NONE);
        let mut bytes: Vec<Option<u8>> = Vec::new();
        while decoder.can_decode() && bytes.len() < max_len {
            let start = decoder.position();
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            let offsets = decoder.get_constant_offsets(&instruction);
            let end = decoder.position();
            let mut masked: Vec<Option<u8>> = code[start..end].iter().copied().map(Some).collect();
            if offsets.has_displacement() && looks_like_address(&instruction, true, image) {
                mask(
                    &mut masked,
                    offsets.displacement_offset(),
                    offsets.displacement_size(),
                );
            }
            if offsets.has_immediate() && is_relocated_immediate(&instruction, image) {
                mask(
                    &mut masked,
                    offsets.immediate_offset(),
                    offsets.immediate_size(),
                );
            }
            bytes.extend(masked);
        }
        Self { bytes }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .bytes
            .iter()
            .map(|byte| byte.map_or_else(|| "??".to_owned(), |b| format!("{b:02X}")))
            .collect();
        f.write_str(&parts.join(" "))
    }
}

fn mask(bytes: &mut [Option<u8>], offset: usize, size: usize) {
    for byte in bytes.iter_mut().skip(offset).take(size) {
        *byte = None;
    }
}

fn looks_like_address(instruction: &Instruction, displacement: bool, image: &Range<u32>) -> bool {
    displacement && image.contains(&instruction.memory_displacement32())
}

/// Whether the immediate depends on where code and data are linked.
fn is_relocated_immediate(instruction: &Instruction, image: &Range<u32>) -> bool {
    match instruction.flow_control() {
        // Near branches and calls hold an offset from the next instruction.
        FlowControl::Call | FlowControl::UnconditionalBranch
            if instruction.op0_kind() == OpKind::NearBranch32 =>
        {
            true
        }
        FlowControl::Next => (0..instruction.op_count()).any(|op| {
            matches!(instruction.op_kind(op), OpKind::Immediate32)
                && image.contains(&instruction.immediate32())
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::{CodeAssembler, dword_ptr, eax, ebx, ecx, esp};

    use super::*;

    const IMAGE: Range<u32> = 0x1_0000..0x40_0000;

    fn assemble(
        build: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>,
        at: u64,
    ) -> Vec<u8> {
        let mut a = CodeAssembler::new(32).unwrap();
        build(&mut a).unwrap();
        a.assemble(at).unwrap()
    }

    #[test]
    fn parses_and_prints_bytes_and_wildcards() {
        let pattern = Pattern::parse("53 ?? 8b").unwrap();
        assert_eq!(pattern.len(), 3);
        assert_eq!(pattern.to_string(), "53 ?? 8B");
        assert_eq!(Pattern::parse("  ").unwrap().len(), 0);
        for bad in ["5", "zz", "123", "?"] {
            let error = Pattern::parse(&format!("53 {bad}")).unwrap_err();
            assert!(error.to_string().contains(bad), "{error}");
        }
    }

    #[test]
    fn wildcards_match_anything_and_matching_needs_enough_data() {
        let pattern = Pattern::parse("AA ?? CC").unwrap();
        assert!(pattern.matches(&[0xAA, 0x00, 0xCC, 0xFF]));
        assert!(pattern.matches(&[0xAA, 0x99, 0xCC]));
        assert!(!pattern.matches(&[0xAA, 0x99, 0xCD]));
        assert!(!pattern.matches(&[0xAA, 0x99]), "too short");
        assert_eq!(pattern.find_all(&[0, 0xAA, 1, 0xCC, 0xAA, 2, 0xCC]), [1, 4]);
        assert_eq!(Pattern::parse("AA").unwrap().find_all(&[]).len(), 0);
        assert_eq!(Pattern::parse("").unwrap().find_all(&[1, 2]).len(), 0);
    }

    #[test]
    fn generated_patterns_mask_calls_and_addresses_but_keep_instructions() {
        // push ebx; call [0x235BA8]; mov eax, 0x22E034; call rel32; mov ecx, 5; ret 8
        let code = assemble(
            |a| {
                a.push(ebx)?;
                a.call(dword_ptr(0x0023_5BA8))?;
                a.mov(eax, 0x0022_E034)?;
                a.mov(ecx, 5)?;
                a.ret_1(8)
            },
            0x22_EA03,
        );
        let pattern = Pattern::from_code(&code, 0x22_EA03, &IMAGE, 64);
        assert_eq!(
            pattern.to_string(),
            "53 FF 15 ?? ?? ?? ?? B8 ?? ?? ?? ?? B9 05 00 00 00 C2 08 00"
        );
        assert!(pattern.matches(&code));
    }

    #[test]
    fn the_same_function_linked_elsewhere_still_matches() {
        let build = |table: u32, helper: u32| {
            assemble(
                |a| {
                    a.mov(eax, table)?;
                    a.push(eax)?;
                    a.call(u64::from(helper))?;
                    a.add(esp, 4)?;
                    a.ret()
                },
                0x1000,
            )
        };
        let one = build(0x0022_E034, 0x0023_184C);
        let two = build(0x0024_0500, 0x0025_0000);
        let pattern = Pattern::from_code(&one, 0x1000, &IMAGE, 64);
        assert!(pattern.matches(&two), "{pattern}");
        // A different instruction is not hidden by the masking.
        let other = assemble(
            |a| {
                a.mov(eax, 0x0022_E034u32)?;
                a.pop(eax)?;
                a.ret()
            },
            0x1000,
        );
        assert!(!pattern.matches(&other));
    }

    #[test]
    fn small_constants_are_not_masked_and_length_is_capped_on_instruction_boundaries() {
        let code = assemble(
            |a| {
                a.mov(eax, 0x40)?;
                a.mov(ecx, 0x1234_5678)?; // outside the image, so a real constant
                a.mov(ebx, 0x0030_0000)?;
                a.ret()
            },
            0,
        );
        let pattern = Pattern::from_code(&code, 0, &IMAGE, 7);
        assert_eq!(
            pattern.to_string(),
            "B8 40 00 00 00 B9 78 56 34 12",
            "cap of 7 keeps whole instructions only"
        );
        let all = Pattern::from_code(&code, 0, &IMAGE, 64);
        assert!(all.to_string().ends_with("BB ?? ?? ?? ?? C3"), "{all}");
    }

    #[test]
    fn decoding_stops_at_invalid_code() {
        let pattern = Pattern::from_code(&[0x53, 0x56, 0x0F, 0xFF], 0, &IMAGE, 64);
        assert_eq!(pattern.to_string(), "53 56");
    }
}
