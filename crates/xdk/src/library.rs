//! Finds the entry points a title uses in a statically linked library.
//!
//! A library such as `DirectSound` is a block of code in its own XBE section.
//! The rest of the image reaches into it through direct calls, so every call or
//! jump that lands inside the section marks a function the title uses.

use std::collections::BTreeSet;
use std::ops::Range;

use iced_x86::{Decoder, DecoderOptions, FlowControl};

/// Instructions to decode from a function's entry while looking for its return.
const MAX_INSTRUCTIONS: usize = 256;
/// How many jumps to follow from a function that only forwards to another.
const MAX_FORWARDS: usize = 4;
/// Bytes a patched-in near jump occupies.
const JUMP_LEN: u32 = 5;

/// A function in a library that code outside the library calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EntryPoint {
    /// Guest address of the function's first byte.
    pub address: u32,
    /// Bytes of stack arguments the function pops on return.
    pub arg_bytes: u16,
}

/// Returns the functions of `library` that the code in `callers` calls or jumps to.
///
/// `callers` and `library_code` are `(guest address, bytes)`. Functions whose
/// return cannot be found, or that are too short to hold a jump, are left out.
///
/// # Panics
///
/// Panics if a section is larger than the 32-bit address space, which cannot happen
/// for guest memory.
#[must_use]
pub fn entry_points(
    callers: &[(u32, &[u8])],
    library: Range<u32>,
    library_code: (u32, &[u8]),
) -> Vec<EntryPoint> {
    let mut targets = BTreeSet::new();
    for &(base, data) in callers {
        for (offset, window) in data.windows(5).enumerate() {
            if window[0] != 0xE8 && window[0] != 0xE9 {
                continue;
            }
            let rel = u32::from_le_bytes([window[1], window[2], window[3], window[4]]);
            let site = base + u32::try_from(offset).expect("section fits the address space");
            let target = site.wrapping_add(5).wrapping_add(rel);
            if library.contains(&target) {
                targets.insert(target);
            }
        }
    }
    targets
        .into_iter()
        .filter_map(|address| {
            let arg_bytes = return_pops(library_code, address)?;
            Some(EntryPoint { address, arg_bytes })
        })
        .collect()
}

/// Decodes from `entry` to the first return and reports the argument bytes it pops.
fn return_pops(code: (u32, &[u8]), entry: u32) -> Option<u16> {
    let mut address = entry;
    let mut forwards = 0;
    loop {
        let offset = usize::try_from(address.checked_sub(code.0)?).ok()?;
        let mut decoder = Decoder::with_ip(
            32,
            code.1.get(offset..)?,
            u64::from(address),
            DecoderOptions::NONE,
        );
        for _ in 0..MAX_INSTRUCTIONS {
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                return None;
            }
            match instruction.flow_control() {
                FlowControl::Return => {
                    let end = u32::try_from(instruction.next_ip()).ok()?;
                    if forwards == 0 && end - entry < JUMP_LEN {
                        return None;
                    }
                    let bytes = if instruction.op_count() == 1 {
                        instruction.immediate16()
                    } else {
                        0
                    };
                    return (bytes % 4 == 0).then_some(bytes);
                }
                FlowControl::UnconditionalBranch if instruction.near_branch_target() != 0 => {
                    if forwards == 0
                        && instruction.next_ip() - u64::from(entry) < u64::from(JUMP_LEN)
                    {
                        return None;
                    }
                    forwards += 1;
                    if forwards > MAX_FORWARDS {
                        return None;
                    }
                    address = u32::try_from(instruction.near_branch_target()).ok()?;
                    break;
                }
                FlowControl::UnconditionalBranch
                | FlowControl::Interrupt
                | FlowControl::Exception => {
                    return None;
                }
                _ => {}
            }
        }
        if forwards == 0 || address == entry {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIBRARY: u32 = 0x2000;

    /// `mov eax, 1; ret 8` then `xor eax, eax; ret` (too short to hold a jump).
    fn library() -> Vec<u8> {
        vec![
            0xB8, 1, 0, 0, 0, 0xC2, 8, 0, // 0x2000: pops 8
            0x33, 0xC0, 0xC3, // 0x2008: three bytes
            0xE9, 0xF0, 0xFF, 0xFF, 0xFF, // 0x200B: jmp 0x2000
        ]
    }

    fn call(from: u32, to: u32) -> Vec<u8> {
        let mut bytes = vec![0xE8];
        bytes.extend_from_slice(&to.wrapping_sub(from + 5).to_le_bytes());
        bytes
    }

    #[test]
    fn calls_into_the_library_name_the_function_and_its_stack_bytes() {
        let lib = library();
        let caller = [call(0x1000, 0x2000), call(0x1005, 0x2000), vec![0x90]].concat();
        let found = entry_points(
            &[(0x1000, &caller)],
            LIBRARY..LIBRARY + 0x100,
            (LIBRARY, &lib),
        );
        assert_eq!(
            found,
            [EntryPoint {
                address: 0x2000,
                arg_bytes: 8
            }]
        );
    }

    #[test]
    fn a_function_that_forwards_takes_the_stack_bytes_of_its_target() {
        let lib = library();
        let caller = call(0x1000, 0x200B);
        let found = entry_points(
            &[(0x1000, &caller)],
            LIBRARY..LIBRARY + 0x100,
            (LIBRARY, &lib),
        );
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].address, found[0].arg_bytes), (0x200B, 8));
    }

    #[test]
    fn functions_too_short_for_a_jump_are_left_alone() {
        let lib = library();
        let caller = call(0x1000, 0x2008);
        let found = entry_points(
            &[(0x1000, &caller)],
            LIBRARY..LIBRARY + 0x100,
            (LIBRARY, &lib),
        );
        assert_eq!(found.len(), 0);
    }

    #[test]
    fn calls_that_stay_outside_the_library_are_ignored() {
        let lib = library();
        let caller = call(0x1000, 0x3000);
        let found = entry_points(
            &[(0x1000, &caller)],
            LIBRARY..LIBRARY + 0x100,
            (LIBRARY, &lib),
        );
        assert_eq!(found.len(), 0);
    }
}
