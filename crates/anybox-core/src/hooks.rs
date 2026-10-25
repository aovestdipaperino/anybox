//! Which functions of a title the runtime replaces.
//!
//! A title carries its own copy of the XDK libraries, so its library functions are found by byte
//! signature, and a library section the runtime does not support is stubbed function by function.
//! Each replaced function is compiled as a call to a trap address the runtime recognises. The
//! runtime plans the same hooks when it boots, from the same signatures, in the same order, so the
//! indices (and with them the trap addresses) agree.

use crate::layout::{MAX_HOOKS, hook_trap_address};

pub use xdk::Section;

/// The sections of `xbe`, as the planner reads them.
#[must_use]
pub fn sections(xbe: &xbe::Xbe) -> Vec<Section<'_>> {
    xbe.sections()
        .iter()
        .map(|section| Section {
            name: section.name(),
            address: section.virtual_address(),
            virtual_size: section.virtual_size(),
            preload: section.flags().preload(),
            data: xbe.section_data(section),
        })
        .collect()
}

/// The plan: every replaced function with its trap address, and the signatures that matched
/// several functions and so were not used.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// `(function address, trap address)`, in hook order.
    pub hooks: Vec<(u32, u32)>,
    /// Each hook's signature name, or `None` for a stubbed entry point; parallel to `hooks`.
    pub names: Vec<Option<&'static str>>,
    /// Signature names with the number of matches.
    pub ambiguous: Vec<(&'static str, usize)>,
}

/// Plans the hooks of a title with `sections` (`xdk::plan`, which the runtime calls too) and
/// gives each its trap address.
///
/// # Errors
///
/// Returns an error when there are more hooks than trap addresses.
pub fn plan(
    sections: &[Section<'_>],
    entries: &'static [xdk::Entry],
    stubbed: &[&str],
) -> Result<Plan, String> {
    let planned = xdk::plan(sections, entries, stubbed, MAX_HOOKS as usize)?;
    Ok(Plan {
        hooks: (0..)
            .zip(&planned.hooks)
            .map(|(index, hook)| (hook.address, hook_trap_address(index)))
            .collect(),
        names: planned.hooks.iter().map(|hook| hook.name).collect(),
        ambiguous: planned.ambiguous,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mov eax, 1 ; ret 8`, which a library section's function can be.
    const FUNCTION: [u8; 8] = [0xB8, 1, 0, 0, 0, 0xC2, 8, 0];

    fn call(from: u32, to: u32) -> [u8; 5] {
        let rel = to.wrapping_sub(from + 5);
        let [a, b, c, d] = rel.to_le_bytes();
        [0xE8, a, b, c, d]
    }

    fn entries() -> &'static [xdk::Entry] {
        xdk::parse_owned("Known | 8 | 53 56 57 58 59 C3\n".to_owned())
            .map(|entries| &*Box::leak(entries.into_boxed_slice()))
            .unwrap()
    }

    #[test]
    fn an_entry_located_only_for_reference_is_not_a_hook() {
        let entries: &'static [xdk::Entry] =
            xdk::parse_owned("Helper | - | 53 56 57 58 59 C3\n".to_owned())
                .map(|entries| &*Box::leak(entries.into_boxed_slice()))
                .unwrap();
        let text = [0x90, 0x53, 0x56, 0x57, 0x58, 0x59, 0xC3];
        let sections = [Section {
            name: ".text",
            address: 0x1_1000,
            virtual_size: 7,
            preload: true,
            data: &text,
        }];
        let plan = plan(&sections, entries, &[]).unwrap();
        assert_eq!(plan.hooks, []);
    }

    #[test]
    fn a_signature_hit_is_a_hook_with_the_first_trap() {
        let text = [0x90, 0x53, 0x56, 0x57, 0x58, 0x59, 0xC3];
        let sections = [Section {
            name: ".text",
            address: 0x1_1000,
            virtual_size: 7,
            preload: true,
            data: &text,
        }];
        let plan = plan(&sections, entries(), &[]).unwrap();
        assert_eq!(plan.hooks, [(0x1_1001, 0x8001_0800)]);
    }

    #[test]
    fn a_section_that_is_not_preloaded_is_not_searched() {
        let text = [0x53, 0x56, 0x57, 0x58, 0x59, 0xC3];
        let sections = [Section {
            name: ".text",
            address: 0x1_1000,
            virtual_size: 6,
            preload: false,
            data: &text,
        }];
        assert_eq!(plan(&sections, entries(), &[]).unwrap().hooks, []);
    }

    #[test]
    fn a_function_of_a_stubbed_library_that_the_title_calls_is_hooked_after_the_hits() {
        let mut text = vec![0x53, 0x56, 0x57, 0x58, 0x59, 0xC3];
        text.extend_from_slice(&call(0x1_1006, 0x2_0000));
        text.push(0xC3);
        let sections = [
            Section {
                name: ".text",
                address: 0x1_1000,
                virtual_size: u32::try_from(text.len()).unwrap(),
                preload: true,
                data: &text,
            },
            Section {
                name: "DSOUND",
                address: 0x2_0000,
                virtual_size: 0x100,
                preload: true,
                data: &FUNCTION,
            },
        ];
        let plan = plan(&sections, entries(), &["DSOUND"]).unwrap();
        assert_eq!(
            plan.hooks,
            [(0x1_1000, 0x8001_0800), (0x2_0000, 0x8001_0804)]
        );
    }

    #[test]
    fn a_function_found_twice_is_hooked_once() {
        let mut text = vec![0x53, 0x56, 0x57, 0x58, 0x59, 0xC3];
        text.extend_from_slice(&call(0x1_1006, 0x1_1000));
        let sections = [Section {
            name: "XAPI",
            address: 0x1_1000,
            virtual_size: u32::try_from(text.len()).unwrap(),
            preload: true,
            data: &text,
        }];
        // The signature hooks 0x11000; the stub pass sees the same address and skips it.
        let outside = [Section {
            name: ".text",
            address: 0x3_0000,
            virtual_size: 5,
            preload: true,
            data: &call(0x3_0000, 0x1_1000),
        }];
        let all = [sections[0], outside[0]];
        let plan = plan(&all, entries(), &["XAPI"]).unwrap();
        assert_eq!(plan.hooks.len(), 1);
    }
}
