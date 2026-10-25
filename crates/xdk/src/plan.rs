//! Which functions of a title the runtime replaces: the hook plan.
//!
//! The transpiler compiles each replaced function as a call to a trap address, and the runtime
//! recognises the trap by the hook's index, so both must plan the same hooks in the same order.
//! They do, by calling this one function with the same signatures.

use std::ops::Range;

use crate::database::{Entry, search};
use crate::library::entry_points;

/// One section of the title's executable, as the planner reads it.
#[derive(Debug, Clone, Copy)]
pub struct Section<'a> {
    /// The section's name, such as `.text` or `DSOUND`.
    pub name: &'a str,
    /// Its virtual address.
    pub address: u32,
    /// Its size in memory.
    pub virtual_size: u32,
    /// Whether the loader maps it before the title starts.
    pub preload: bool,
    /// Its bytes in the file.
    pub data: &'a [u8],
}

/// One replaced function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hook {
    /// The function's guest address.
    pub address: u32,
    /// The signature that found it, or `None` for an entry point of a stubbed library.
    pub name: Option<&'static str>,
    /// Bytes of stack arguments it pops: from the signature, or inferred from a stubbed
    /// function's return.
    pub arg_bytes: u16,
}

/// The plan: the hooks in index order, and the signatures that were not used because they matched
/// several functions.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub hooks: Vec<Hook>,
    /// Signature names with the number of matches.
    pub ambiguous: Vec<(&'static str, usize)>,
}

/// Plans the hooks of a title with `sections`: first, in database order, every function a hook
/// entry of `entries` finds in the preloaded sections; then every entry point into a section
/// named in `stubbed` that code outside it calls, unless already hooked.
///
/// # Errors
///
/// Returns an error when there are more than `max_hooks` hooks.
pub fn plan(
    sections: &[Section<'_>],
    entries: &'static [Entry],
    stubbed: &[&str],
    max_hooks: usize,
) -> Result<Plan, String> {
    let mut plan = Plan::default();
    let mapped: Vec<(u32, &[u8])> = sections
        .iter()
        .filter(|section| section.preload)
        .map(|section| (section.address, section.data))
        .collect();
    let found = search(entries, &mapped);
    // Entries located only so that others can refer to them are not replaced.
    for hit in found.hits.iter().filter(|hit| entries[hit.entry].hook) {
        let entry = &entries[hit.entry];
        plan.hooks.push(Hook {
            address: hit.address,
            name: Some(entry.name),
            arg_bytes: entry.arg_bytes,
        });
    }
    plan.ambiguous = found
        .ambiguous
        .iter()
        .map(|(entry, addresses)| (entries[*entry].name, addresses.len()))
        .collect();
    for name in stubbed {
        for library in sections.iter().filter(|section| section.name == *name) {
            let range: Range<u32> = library.address..library.address + library.virtual_size;
            let callers: Vec<(u32, &[u8])> = sections
                .iter()
                .filter(|section| !range.contains(&section.address))
                .map(|section| (section.address, section.data))
                .collect();
            for point in entry_points(&callers, range, (library.address, library.data)) {
                if plan.hooks.iter().any(|hook| hook.address == point.address) {
                    continue;
                }
                plan.hooks.push(Hook {
                    address: point.address,
                    name: None,
                    arg_bytes: point.arg_bytes,
                });
            }
        }
    }
    if plan.hooks.len() > max_hooks {
        return Err(format!(
            "{} library hooks, more than the {max_hooks} there are traps for",
            plan.hooks.len()
        ));
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::parse;

    /// `mov eax, 1 ; ret 8`, which a library section's function can be.
    const FUNCTION: [u8; 8] = [0xB8, 1, 0, 0, 0, 0xC2, 8, 0];

    fn call(from: u32, to: u32) -> [u8; 5] {
        let [a, b, c, d] = to.wrapping_sub(from + 5).to_le_bytes();
        [0xE8, a, b, c, d]
    }

    fn entries() -> &'static [Entry] {
        Box::leak(
            parse("Known | 8 | 53 56 57 58 59 C3\n")
                .unwrap()
                .into_boxed_slice(),
        )
    }

    #[test]
    fn signature_hits_come_first_then_stubbed_entry_points_with_their_inferred_pops() {
        let mut text = vec![0x53, 0x56, 0x57, 0x58, 0x59, 0xC3];
        text.extend_from_slice(&call(0x1_1006, 0x2_0000));
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
        let plan = plan(&sections, entries(), &["DSOUND"], 511).unwrap();
        assert_eq!(
            plan.hooks,
            [
                Hook {
                    address: 0x1_1000,
                    name: Some("Known"),
                    arg_bytes: 8
                },
                Hook {
                    address: 0x2_0000,
                    name: None,
                    arg_bytes: 8
                },
            ]
        );
    }

    #[test]
    fn too_many_hooks_is_an_error() {
        let text = [0x53, 0x56, 0x57, 0x58, 0x59, 0xC3];
        let sections = [Section {
            name: ".text",
            address: 0x1_1000,
            virtual_size: 6,
            preload: true,
            data: &text,
        }];
        assert!(plan(&sections, entries(), &[], 0).is_err());
    }
}
