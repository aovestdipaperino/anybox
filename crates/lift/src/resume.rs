//! The resume map: every place a hibernated guest thread can stand in the lifted code, so a
//! snapshot made by one build can be checked against another before it is resumed there.
//!
//! A frame record names a function, a site in it and a kind, all as guest addresses and the
//! `LIFT_K_*` values; a saved `setjmp` context names the call that made it. A build whose map
//! holds every frame and context of a snapshot can resume it, whatever else changed in its code.
//!
//! The map's file, which the runtime reads (`hibernate::resume_map`): the magic, the
//! [`RESUME_ABI`], the number of entries, then each entry as three words (function, site, kind),
//! sorted, all little-endian.

use std::collections::BTreeSet;

/// The layout of what a snapshot keeps of a guest CPU: the C prelude's `lift_cpu_t`,
/// `lift_frame` and `lift_jmp_saved`, and the Rust prelude's `Cpu`, `Frame` and `JmpSaved`.
/// A snapshot is resumed only by code of the same layout, so any change to those structures must
/// raise it (a test pins them).
pub const RESUME_ABI: u32 = 1;

/// The map file's magic.
pub const MAGIC: &[u8; 8] = b"ANYXRSM\0";

/// The kind of an entry that names a `setjmp` call, beside the `LIFT_K_*` kinds of frame sites.
pub const KIND_SETJMP: u32 = 3;

/// The places a resume can re-enter the lifted code at.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ResumeMap {
    entries: BTreeSet<(u32, u32, u32)>,
}

impl ResumeMap {
    /// Records that function `function` can be resumed at `site`, of `kind` (a `LIFT_K_*` value,
    /// or [`KIND_SETJMP`] for a `setjmp` call).
    pub fn insert(&mut self, function: u32, site: u32, kind: u32) {
        self.entries.insert((function, site, kind));
    }

    /// Whether function `function` can be resumed at `site`, of `kind`.
    #[must_use]
    pub fn contains(&self, function: u32, site: u32, kind: u32) -> bool {
        self.entries.contains(&(function, site, kind))
    }

    /// The number of places.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The map's file (see the module).
    ///
    /// # Panics
    ///
    /// If it holds more than `u32::MAX` places.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let count = u32::try_from(self.entries.len()).expect("fewer than 2^32 resume sites");
        let mut bytes = Vec::with_capacity(16 + self.entries.len() * 12);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&RESUME_ABI.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        for &(function, site, kind) in &self.entries {
            for word in [function, site, kind] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_holds_the_entries_sorted_after_its_header() {
        let mut map = ResumeMap::default();
        map.insert(0x2000, 0x2010, 0);
        map.insert(0x1000, 0x1004, KIND_SETJMP);
        map.insert(0x1000, 0x1004, KIND_SETJMP);
        assert_eq!(map.len(), 2);
        let bytes = map.encode();
        assert_eq!(&bytes[..8], MAGIC);
        assert_eq!(bytes[8..12], RESUME_ABI.to_le_bytes());
        assert_eq!(bytes[12..16], 2u32.to_le_bytes());
        let words: Vec<u32> = bytes[16..]
            .chunks(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        assert_eq!(words, [0x1000, 0x1004, KIND_SETJMP, 0x2000, 0x2010, 0]);
    }

    /// FNV-1a, stable across Rust versions, unlike the standard hasher.
    fn fnv(text: &str) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// The text from the line holding `start` to the line holding `end`, both included.
    fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
        let from = text[..text.find(start).expect(start)]
            .rfind('\n')
            .map_or(0, |at| at + 1);
        let to = from + text[from..].find(end).expect(end) + end.len();
        &text[from..to]
    }

    /// A change to what a snapshot keeps of a guest CPU must raise [`RESUME_ABI`]: then update
    /// the hash here.
    #[test]
    fn the_resume_layout_is_the_one_its_abi_names() {
        let c = crate::c::PRELUDE;
        let rust = crate::rust::PRELUDE;
        let layout = [
            between(
                c,
                "typedef struct { uint32_t fn, kind, site; } lift_frame;",
                "lift_frame;",
            ),
            between(
                c,
                "    uint32_t eax, ecx, edx, ebx, esp, ebp, esi, edi;",
                "} lift_cpu_t;",
            ),
            between(c, "typedef union { float f[4];", "} xmm_t;"),
            between(
                c,
                "typedef struct { uint32_t buf, resume, depth, site; } lift_jmp_saved;",
                "lift_jmp_saved;",
            ),
            between(rust, "pub struct Frame {", "}"),
            between(rust, "pub struct Cpu {", "\n}"),
            between(rust, "pub struct JmpSaved {", "}"),
        ]
        .concat();
        assert_eq!(
            (RESUME_ABI, fnv(&layout)),
            (1, 0xd03f_9c4a_9135_2629),
            "the resume layout changed: raise RESUME_ABI and update the hash"
        );
    }
}
