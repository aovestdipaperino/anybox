//! The lifted code's resume map: every place in it a hibernated thread can stand, which the glue
//! hands over before boot. A snapshot made by another build of the title is resumed only when this
//! build's map has every frame and `setjmp` context it holds.
//!
//! The map's file, as the lifter writes it (`lift::resume`): the magic, the layout of a saved
//! guest CPU, the number of entries, then each entry as three words (function, site, kind),
//! sorted, all little-endian. A kind is a frame record's, or [`KIND_SETJMP`] for a `setjmp` call.

use std::sync::OnceLock;

use super::cpu::GuestCpu;

/// The map file's magic.
const MAGIC: &[u8; 8] = b"ANYXRSM\0";
/// The header: magic, layout, count.
const HEADER: usize = 8 + 4 + 4;
/// An entry: function, site, kind.
const ENTRY: usize = 12;
/// The kind of an entry that names a `setjmp` call.
pub const KIND_SETJMP: u32 = 3;

/// A resume map, read in place.
#[derive(Debug, Clone, Copy)]
pub struct ResumeMap<'a> {
    /// The layout of a saved guest CPU (`lift::resume::RESUME_ABI`).
    pub abi: u32,
    entries: &'a [u8],
}

impl<'a> ResumeMap<'a> {
    /// The map in `bytes`.
    ///
    /// # Errors
    ///
    /// A message when `bytes` is not a whole map.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return Err("not a resume map".to_owned());
        }
        let abi = word(bytes, 8);
        let count = word(bytes, 12) as usize;
        let entries = &bytes[HEADER..];
        if count.checked_mul(ENTRY) != Some(entries.len()) {
            return Err(format!(
                "a resume map of {count} entries in {} bytes",
                entries.len()
            ));
        }
        Ok(Self { abi, entries })
    }

    /// The entry at `index`.
    fn entry(&self, index: usize) -> (u32, u32, u32) {
        let at = index * ENTRY;
        (
            word(self.entries, at),
            word(self.entries, at + 4),
            word(self.entries, at + 8),
        )
    }

    /// Whether function `function` can be resumed at `site`, of `kind`.
    #[must_use]
    pub fn contains(&self, function: u32, site: u32, kind: u32) -> bool {
        let wanted = (function, site, kind);
        let (mut low, mut high) = (0, self.entries.len() / ENTRY);
        while low < high {
            let mid = low + (high - low) / 2;
            match self.entry(mid).cmp(&wanted) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return true,
            }
        }
        false
    }

    /// Whether the code this map describes can resume `threads`: every frame is at a site its
    /// function has, and every `setjmp` context belongs to a `setjmp` call of its frame's function.
    ///
    /// # Errors
    ///
    /// A message naming the first frame or context it cannot resume.
    pub fn check(&self, threads: &[GuestCpu]) -> Result<(), String> {
        for thread in threads {
            for (depth, frame) in thread.frames.iter().enumerate() {
                if !self.contains(frame.function, frame.site, frame.kind) {
                    return Err(format!(
                        "thread {}'s frame {depth} stands in {:08x} at {:08x} (kind {}), which this \
                         build's code does not have",
                        thread.tid, frame.function, frame.site, frame.kind
                    ));
                }
            }
            for jump in &thread.jumps {
                let Some(frame) = thread.frames.get(jump.depth as usize) else {
                    return Err(format!(
                        "thread {} has a setjmp context in frame {}, below its last",
                        thread.tid, jump.depth
                    ));
                };
                if !self.contains(frame.function, jump.site, KIND_SETJMP) {
                    return Err(format!(
                        "thread {}'s setjmp context at {:08x} in {:08x} is one this build's code \
                         does not have",
                        thread.tid, jump.site, frame.function
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The little-endian word at `at` in `bytes`.
fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or_default())
}

/// The running code's map, which the glue hands over before boot.
static MAP: OnceLock<ResumeMap<'static>> = OnceLock::new();

/// Records the lifted code's resume map: `len` bytes at `map`, which last as long as the program.
/// A map that cannot be read is ignored, saying why: snapshots of this build alone resume then.
///
/// # Safety
///
/// `map` must point to `len` readable bytes that are never freed or changed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_set_resume_map(map: *const u8, len: usize) {
    if map.is_null() {
        return;
    }
    // SAFETY: the caller hands `len` bytes that live as long as the program.
    let bytes = unsafe { std::slice::from_raw_parts(map, len) };
    match ResumeMap::parse(bytes) {
        Ok(parsed) => {
            let _ = MAP.set(parsed);
        }
        Err(why) => eprintln!("anyxbox: the lifted code's resume map: {why}"),
    }
}

/// The running code's map, if the glue handed one over.
#[must_use]
pub fn current() -> Option<ResumeMap<'static>> {
    MAP.get().copied()
}

#[cfg(test)]
mod tests {
    use super::super::cpu::{Frame, Jump};
    use super::*;

    /// A map file of `entries`, which must be sorted.
    fn map_file(entries: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_le_bytes());
        for &(function, site, kind) in entries {
            for word in [function, site, kind] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }

    fn thread(frames: &[(u32, u32, u32)], jumps: &[(u32, u32)]) -> GuestCpu {
        GuestCpu {
            tid: 9,
            cpu: Vec::new(),
            frames: frames
                .iter()
                .map(|&(function, site, kind)| Frame {
                    function,
                    kind,
                    site,
                })
                .collect(),
            jumps: jumps
                .iter()
                .map(|&(depth, site)| Jump {
                    buf: 0,
                    resume: 0,
                    depth,
                    site,
                })
                .collect(),
        }
    }

    #[test]
    fn a_map_finds_its_entries_and_nothing_else() {
        let bytes = map_file(&[
            (0x1000, 0x1004, 0),
            (0x1000, 0x1010, 3),
            (0x2000, 0x2000, 1),
        ]);
        let map = ResumeMap::parse(&bytes).unwrap();
        assert_eq!(map.abi, 7);
        assert!(map.contains(0x1000, 0x1004, 0));
        assert!(map.contains(0x1000, 0x1010, 3));
        assert!(map.contains(0x2000, 0x2000, 1));
        assert!(!map.contains(0x1000, 0x1004, 2), "another kind");
        assert!(!map.contains(0x1004, 0x1004, 0), "another function");
        assert!(!map.contains(0x3000, 0x3000, 0));
        assert!(
            ResumeMap::parse(&bytes[..bytes.len() - 1]).is_err(),
            "cut short"
        );
        assert!(ResumeMap::parse(b"not a map at all").is_err());
    }

    #[test]
    fn a_snapshot_resumes_where_every_frame_and_context_is_in_the_map() {
        let bytes = map_file(&[
            (0x1000, 0x1004, 3),
            (0x1000, 0x1010, 0),
            (0x2000, 0x2008, 2),
        ]);
        let map = ResumeMap::parse(&bytes).unwrap();
        let good = thread(&[(0x1000, 0x1010, 0), (0x2000, 0x2008, 2)], &[(0, 0x1004)]);
        map.check(&[good]).unwrap();
        let moved_poll = thread(&[(0x1000, 0x1010, 0), (0x2000, 0x200C, 2)], &[]);
        assert!(map.check(&[moved_poll]).unwrap_err().contains("frame 1"));
        let lost_setjmp = thread(&[(0x1000, 0x1010, 0)], &[(0, 0x1008)]);
        assert!(map.check(&[lost_setjmp]).unwrap_err().contains("setjmp"));
        let context_too_deep = thread(&[(0x1000, 0x1010, 0)], &[(1, 0x1004)]);
        assert!(map.check(&[context_too_deep]).is_err());
    }
}
