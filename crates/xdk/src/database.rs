//! The signature database and the search that applies it to an image.

use std::collections::HashMap;
use std::fmt;

use crate::pattern::{Pattern, PatternError};

/// One known XDK function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The function's name in the XDK.
    pub name: &'static str,
    /// Bytes of stack arguments the function pops on return (stdcall).
    pub arg_bytes: u16,
    /// Whether the function is replaced. An entry that is not (written with `-` for its argument
    /// bytes) is only located, so that other entries can refer to it.
    pub hook: bool,
    /// What its machine code looks like; the bytes of references are wildcards in it.
    pub pattern: Pattern,
    /// References to other entries: at each offset, four bytes that hold the named function's
    /// address, relative to the end of those bytes (a call or jump) or absolute.
    pub references: Vec<(usize, &'static str)>,
    /// Which match to use, counting from 1 by address, when the pattern matches
    /// several functions that differ only in masked bytes.
    pub occurrence: Option<usize>,
    /// The functions it calls, followed one after the other, for functions whose own code is
    /// the same as others' (the `DirectSound` thunks): the offset in the previous function of a
    /// call's 32-bit relative target, and the pattern the code there must match.
    pub callees: Vec<(usize, Pattern)>,
}

/// Why a database line could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatabaseError {
    line: usize,
    message: String,
}

impl fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "signature database line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for DatabaseError {}

/// Splits the references out of a pattern: each `<Name>` token stands for four wildcard bytes
/// that refer to `Name`. Returns the plain pattern text and the references.
fn split_references(pattern: &'static str) -> (String, Vec<(usize, &'static str)>) {
    let mut plain = Vec::new();
    let mut references = Vec::new();
    let mut offset = 0;
    for token in pattern.split_whitespace() {
        if let Some(name) = token.strip_prefix('<').and_then(|t| t.strip_suffix('>')) {
            references.push((offset, name));
            plain.extend(["??"; 4]);
            offset += 4;
        } else {
            plain.push(token);
            offset += 1;
        }
    }
    (plain.join(" "), references)
}

/// Parses a database: one `name | argument bytes | pattern` entry per line.
///
/// A name may end in `@N` to use only the Nth match of a pattern that finds several
/// functions with the same code. Argument bytes of `-` mark an entry that is located but not
/// replaced. In a pattern, `<Name>` is four bytes that refer to the entry `Name`.
///
/// More `| offset: pattern` fields follow the function's calls: the 32-bit relative target at
/// `offset` (decimal) in the code the previous pattern matched must match `pattern`, and so on
/// down the chain. A function matches only when every callee does.
///
/// Blank lines and lines starting with `#` are ignored.
///
/// # Errors
///
/// Returns an error naming the first malformed line.
pub fn parse(text: &'static str) -> Result<Vec<Entry>, DatabaseError> {
    let mut entries = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fail = |message: String| DatabaseError {
            line: index + 1,
            message,
        };
        let mut fields = line.split('|').map(str::trim);
        let (Some(label), Some(args), Some(pattern)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return Err(fail(
                "expected `name | argument bytes | pattern`".to_owned(),
            ));
        };
        let (name, occurrence) = match label.split_once('@') {
            Some((name, n)) => {
                let n = n
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| fail(format!("`{n}` is not a match number from 1")))?;
                (name, Some(n))
            }
            None => (label, None),
        };
        let hook = args != "-";
        let arg_bytes: u16 = if hook {
            args.parse()
                .ok()
                .filter(|bytes| bytes % 4 == 0)
                .ok_or_else(|| fail(format!("`{args}` is not a multiple of four")))?
        } else {
            0
        };
        let (plain, references) = split_references(pattern);
        let pattern = Pattern::parse(&plain).map_err(|e: PatternError| fail(e.to_string()))?;
        if pattern.len() < 5 {
            return Err(fail(
                "a pattern needs at least five bytes to hold a jump".to_owned(),
            ));
        }
        let callees = parse_callees(fields, pattern.len()).map_err(fail)?;
        if entries
            .iter()
            .any(|entry: &Entry| entry.name == name && entry.occurrence == occurrence)
        {
            return Err(fail(format!("`{name}` appears twice")));
        }
        entries.push(Entry {
            name,
            arg_bytes,
            hook,
            pattern,
            references,
            occurrence,
            callees,
        });
    }
    for (line, entry) in entries.iter().enumerate() {
        if let Some((_, missing)) = entry
            .references
            .iter()
            .find(|(_, target)| !entries.iter().any(|e| e.name == *target))
        {
            return Err(DatabaseError {
                line: line + 1,
                message: format!(
                    "`{}` refers to `{missing}`, which is not in the database",
                    entry.name
                ),
            });
        }
    }
    Ok(entries)
}

/// The `offset: pattern` fields after an entry's pattern, `caller_len` bytes long.
fn parse_callees<'a>(
    fields: impl Iterator<Item = &'a str>,
    mut caller_len: usize,
) -> Result<Vec<(usize, Pattern)>, String> {
    let mut callees = Vec::new();
    for field in fields {
        let (offset, callee) = field
            .split_once(':')
            .ok_or_else(|| format!("`{field}` is not `offset: pattern`"))?;
        let offset: usize = offset
            .trim()
            .parse()
            .ok()
            .filter(|offset| offset + 4 <= caller_len)
            .ok_or_else(|| {
                format!(
                    "`{}` is not an offset inside the caller's pattern",
                    offset.trim()
                )
            })?;
        let callee = Pattern::parse(callee).map_err(|e: PatternError| e.to_string())?;
        if callee.is_empty() {
            return Err("a callee needs a pattern".to_owned());
        }
        caller_len = callee.len();
        callees.push((offset, callee));
    }
    Ok(callees)
}

/// Parses a database read at run time. The text is kept for the life of the process, because the
/// entries borrow their names from it.
///
/// # Errors
///
/// Returns an error naming the first malformed line.
pub fn parse_owned(text: String) -> Result<Vec<Entry>, DatabaseError> {
    parse(Box::leak(text.into_boxed_str()))
}

/// A function found in an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    /// Index of the entry in the database that was searched.
    pub entry: usize,
    /// Guest address of the function's first byte.
    pub address: u32,
}

/// The result of searching an image.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Search {
    /// Functions found in exactly one place, in database order, located-only entries included
    /// (a hook planner skips those whose entry is not a hook).
    pub hits: Vec<Hit>,
    /// Entries that matched in several places, with every address, and entries that matched the
    /// same function as another entry, with that address.
    ///
    /// Only entries that name an occurrence are used, through the one hit they pick.
    pub ambiguous: Vec<(usize, Vec<u32>)>,
}

/// Whether the four bytes at `address` in `sections` refer to `target`: as a displacement from
/// the end of the four bytes, or as the address itself.
fn refers_to(sections: &[(u32, &[u8])], address: u32, target: u32) -> bool {
    sections.iter().any(|&(base, data)| {
        let Some(at) = address.checked_sub(base).map(|a| a as usize) else {
            return false;
        };
        data.get(at..at + 4).is_some_and(|b| {
            let value = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            value == target || address.wrapping_add(4).wrapping_add(value) == target
        })
    })
}

/// The code of `sections` from `address` to the end of its section.
fn code_at<'a>(sections: &[(u32, &'a [u8])], address: u32) -> Option<&'a [u8]> {
    sections.iter().find_map(|&(base, data)| {
        let offset = usize::try_from(address.checked_sub(base)?).ok()?;
        data.get(offset..).filter(|code| !code.is_empty())
    })
}

/// Whether the function at `address` calls, down the chain, code that matches each callee.
fn calls_match(callees: &[(usize, Pattern)], sections: &[(u32, &[u8])], address: u32) -> bool {
    let mut caller = address;
    for (offset, callee) in callees {
        let Some(rel) = code_at(sections, caller)
            .and_then(|code| code.get(*offset..offset + 4))
            .map(|rel| u32::from_le_bytes([rel[0], rel[1], rel[2], rel[3]]))
        else {
            return false;
        };
        let offset = u32::try_from(*offset).expect("patterns are short");
        caller = caller.wrapping_add(offset + 4).wrapping_add(rel);
        if !code_at(sections, caller).is_some_and(|code| callee.matches(code)) {
            return false;
        }
    }
    true
}

/// Where an entry's search stands.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    /// Its candidates, not yet checked against references that are not located yet.
    Open(Vec<u32>),
    Found(u32),
    /// Matched several places (after checking its references), or none.
    Failed(Vec<u32>),
}

/// Every place each entry's pattern matches, sorted, as the starting status.
fn candidates(entries: &[Entry], sections: &[(u32, &[u8])]) -> Vec<Status> {
    entries
        .iter()
        .map(|entry| {
            let mut addresses: Vec<u32> = sections
                .iter()
                .flat_map(|&(base, data)| {
                    entry.pattern.find_all(data).into_iter().map(move |offset| {
                        base + u32::try_from(offset).expect("section fits the address space")
                    })
                })
                .filter(|&address| calls_match(&entry.callees, sections, address))
                .collect();
            addresses.sort_unstable();
            Status::Open(addresses)
        })
        .collect()
}

/// Settles the open entries whose references are settled, until nothing changes: a candidate
/// stays when every reference holds; an entry whose reference can never be found fails.
fn settle(entries: &[Entry], sections: &[(u32, &[u8])], status: &mut [Status]) {
    let by_name: HashMap<&str, usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.occurrence.is_none()
                || !entries
                    .iter()
                    .any(|o| o.name == e.name && o.occurrence.is_none())
        })
        .map(|(i, e)| (e.name, i))
        .collect();
    loop {
        let mut changed = false;
        for (index, entry) in entries.iter().enumerate() {
            let Status::Open(candidates) = status[index].clone() else {
                continue;
            };
            let mut targets = Vec::with_capacity(entry.references.len());
            let mut unsettled = false;
            let mut impossible = false;
            for (offset, name) in &entry.references {
                match by_name.get(name).map(|&t| &status[t]) {
                    Some(Status::Found(address)) => targets.push((*offset, *address)),
                    Some(Status::Open(_)) => unsettled = true,
                    _ => impossible = true,
                }
            }
            if impossible {
                status[index] = Status::Failed(candidates);
                changed = true;
                continue;
            }
            if unsettled {
                continue;
            }
            let kept: Vec<u32> = candidates
                .into_iter()
                .filter(|&start| {
                    targets.iter().all(|&(offset, target)| {
                        refers_to(
                            sections,
                            start + u32::try_from(offset).unwrap_or(u32::MAX),
                            target,
                        )
                    })
                })
                .collect();
            status[index] = match (entry.occurrence, kept.as_slice()) {
                (Some(n), _) if kept.len() >= n => Status::Found(kept[n - 1]),
                (None, [only]) => Status::Found(*only),
                _ => Status::Failed(kept),
            };
            changed = true;
        }
        if !changed {
            break;
        }
    }
}

/// Searches code sections, given as `(guest address, bytes)`, for every entry. Entries with
/// references are settled once the entries they refer to are: a candidate stays when every
/// reference holds.
///
/// # Panics
///
/// Panics if a section is larger than the 32-bit address space, which cannot happen
/// for guest memory.
#[must_use]
pub fn search(entries: &[Entry], sections: &[(u32, &[u8])]) -> Search {
    let mut status = candidates(entries, sections);
    settle(entries, sections, &mut status);
    let mut result = Search::default();
    for (index, status) in status.iter().enumerate() {
        match status {
            // An entry with an occurrence names one of several identical functions: not
            // ambiguous.
            Status::Found(address) => result.hits.push(Hit {
                entry: index,
                address: *address,
            }),
            // Left open: a cycle of references that nothing settles.
            Status::Open(addresses) | Status::Failed(addresses) if addresses.len() > 1 => {
                result.ambiguous.push((index, addresses.clone()));
            }
            _ => {}
        }
    }
    // A function two entries find is neither of them for certain: drop both hits.
    let shared: Vec<u32> = result
        .hits
        .iter()
        .filter(|hit| {
            result
                .hits
                .iter()
                .filter(|other| other.address == hit.address)
                .count()
                > 1
        })
        .map(|hit| hit.address)
        .collect();
    let (kept, dropped): (Vec<Hit>, Vec<Hit>) = result
        .hits
        .into_iter()
        .partition(|hit| !shared.contains(&hit.address));
    result.hits = kept;
    result.ambiguous.extend(
        dropped
            .into_iter()
            .map(|hit| (hit.entry, vec![hit.address])),
    );
    result.ambiguous.sort_by_key(|(entry, _)| *entry);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "\
# a comment

First  | 8 | 53 56 ?? C3 90
Second | 0 | 90 90 90 90 90 90
";

    #[test]
    fn callees_tell_apart_functions_with_the_same_code() {
        // Two thunks, `call rel32; ret`, the same but for where they call: one function that
        // returns 1 and one that returns 2.
        let entries = parse(
            "Two | 0 | E8 ?? ?? ?? ?? C3 | 1: B8 02 00 00 00 C3\n\
             One | 0 | E8 ?? ?? ?? ?? C3 | 1: B8 01 00 00 00 C3",
        )
        .unwrap();
        let mut code = vec![0x90; 0x40];
        let thunk = |code: &mut [u8], at: usize, target: usize| {
            code[at] = 0xE8;
            let rel = u32::try_from(target).unwrap() - u32::try_from(at + 5).unwrap();
            code[at + 1..at + 5].copy_from_slice(&rel.to_le_bytes());
            code[at + 5] = 0xC3;
        };
        thunk(&mut code, 0, 0x20);
        thunk(&mut code, 8, 0x30);
        code[0x20..0x26].copy_from_slice(&[0xB8, 1, 0, 0, 0, 0xC3]);
        code[0x30..0x36].copy_from_slice(&[0xB8, 2, 0, 0, 0, 0xC3]);
        let found = search(&entries, &[(0x1000, &code)]);
        assert_eq!(
            found.hits,
            [
                Hit {
                    entry: 0,
                    address: 0x1008
                },
                Hit {
                    entry: 1,
                    address: 0x1000
                }
            ]
        );
        assert_eq!(found.ambiguous, []);
        assert!(
            parse("a | 8 | E8 ?? ?? ?? ?? C3 | 3: C3")
                .unwrap_err()
                .to_string()
                .contains("offset inside")
        );
    }

    #[test]
    fn parses_entries_and_skips_comments() {
        let entries = parse(TEXT).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!((entries[0].name, entries[0].arg_bytes), ("First", 8));
        assert_eq!(entries[1].pattern.len(), 6);
    }

    #[test]
    fn malformed_lines_are_reported_with_their_number() {
        for (text, needle) in [
            ("a | 8", "expected"),
            ("a | x | 53 56 57 58 59", "multiple of four"),
            ("a | 6 | 53 56 57 58 59", "multiple of four"),
            ("a | 8 | 53 56", "five bytes"),
            ("a@0 | 8 | 53 56 57 58 59", "match number"),
            ("a | 8 | 53 zz 57 58 59", "zz"),
            ("a | 8 | 53 56 57 58 59\na | 4 | 53 56 57 58 59", "twice"),
        ] {
            let error = parse(text).unwrap_err().to_string();
            assert!(error.contains(needle), "{text:?}: {error}");
        }
        assert!(parse("\n\nbad").unwrap_err().to_string().contains("line 3"));
    }

    #[test]
    fn an_occurrence_picks_one_of_several_identical_functions() {
        let entries = parse("Second@2 | 0 | 01 02 03 04 05\n").unwrap();
        let code = [1u8, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5];
        let found = search(&entries, &[(0x1000, &code)]);
        assert_eq!(found.hits.len(), 1);
        assert_eq!(found.hits[0].address, 0x1006);
    }

    #[test]
    fn two_entries_that_find_the_same_function_are_both_ambiguous() {
        let entries =
            parse("A | 0 | 01 02 ?? 04 05\nB | 0 | 01 ?? 03 04 05\nC | 0 | 09 09 09 09 09\n")
                .unwrap();
        let code = [1u8, 2, 3, 4, 5, 9, 9, 9, 9, 9];
        let found = search(&entries, &[(0x1000, &code)]);
        assert_eq!(
            found.hits,
            [Hit {
                entry: 2,
                address: 0x1005
            }]
        );
        assert_eq!(found.ambiguous, [(0, vec![0x1000]), (1, vec![0x1000])]);
    }

    #[test]
    fn references_and_located_only_entries_parse() {
        let entries =
            parse("Inner | - | 55 8B EC 5D C3\nOuter | 4 | E8 <Inner> C2 04 00\n").unwrap();
        assert!(!entries[0].hook);
        assert!(entries[1].hook);
        assert_eq!(entries[1].references, [(1, "Inner")]);
        assert_eq!(entries[1].pattern.len(), 8);
        let error = parse("Outer | 4 | E8 <Missing> C2 04 00\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("Missing"), "{error}");
    }

    #[test]
    fn a_reference_tells_apart_functions_that_differ_only_in_what_they_call() {
        // Two thunks `call rel32 ; ret 4` at 0x1000 and 0x1008; only the second calls 0x1020.
        let mut code = vec![0u8; 0x30];
        let thunk = |code: &mut Vec<u8>, at: usize, target: u32| {
            let rel = target.wrapping_sub(0x1000 + u32::try_from(at).unwrap() + 5);
            code[at] = 0xE8;
            code[at + 1..at + 5].copy_from_slice(&rel.to_le_bytes());
            code[at + 5..at + 8].copy_from_slice(&[0xC2, 0x04, 0x00]);
        };
        thunk(&mut code, 0, 0x1018);
        thunk(&mut code, 8, 0x1020);
        code[0x18..0x1D].copy_from_slice(&[0x31, 0xC0, 0x40, 0x90, 0xC3]);
        code[0x20..0x25].copy_from_slice(&[0x55, 0x8B, 0xEC, 0x5D, 0xC3]);
        let entries =
            parse("Outer | 4 | E8 <Inner> C2 04 00\nInner | - | 55 8B EC 5D C3\n").unwrap();
        let found = search(&entries, &[(0x1000, &code)]);
        assert_eq!(
            found.hits,
            [
                Hit {
                    entry: 0,
                    address: 0x1008
                },
                Hit {
                    entry: 1,
                    address: 0x1020
                }
            ]
        );
        assert_eq!(found.ambiguous, []);
    }

    #[test]
    fn an_absolute_reference_matches_too() {
        // `call [0x00002000]`, where 0x2000 is the referenced function.
        let code = [0xFF, 0x15, 0x00, 0x20, 0x00, 0x00, 0xC3];
        let inner = [0x55, 0x8B, 0xEC, 0x5D, 0xC3];
        let entries = parse("Outer | 0 | FF 15 <Inner> C3\nInner | - | 55 8B EC 5D C3\n").unwrap();
        let found = search(&entries, &[(0x1000, &code), (0x2000, &inner)]);
        assert_eq!(
            found.hits[0],
            Hit {
                entry: 0,
                address: 0x1000
            }
        );
    }

    #[test]
    fn an_entry_whose_reference_is_never_found_is_not_found() {
        let code = [0xE8, 0, 0, 0, 0, 0xC3];
        let entries = parse("Outer | 0 | E8 <Inner> C3\nInner | - | 55 8B EC 5D C3\n").unwrap();
        let found = search(&entries, &[(0x1000, &code)]);
        assert_eq!(found.hits, []);
    }

    #[test]
    fn a_database_read_at_run_time_parses_like_a_static_one() {
        let entries = parse_owned(TEXT.to_owned()).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(parse_owned("bad".to_owned()).is_err());
    }

    #[test]
    fn search_finds_unique_matches_and_flags_ambiguous_ones() {
        let entries = parse(TEXT).unwrap();
        let code: Vec<u8> = [
            vec![0u8; 3],
            vec![0x53, 0x56, 0x77, 0xC3, 0x90],
            vec![0; 2],
            vec![0x90; 6],
        ]
        .concat();
        let found = search(&entries, &[(0x1000, &code)]);
        assert_eq!(
            found.hits,
            [
                Hit {
                    entry: 0,
                    address: 0x1003
                },
                Hit {
                    entry: 1,
                    address: 0x100A
                }
            ]
        );
        assert_eq!(found.ambiguous.len(), 0);

        let twice: Vec<u8> = [
            vec![0x53, 0x56, 0x01, 0xC3, 0x90],
            vec![0x53, 0x56, 0x02, 0xC3, 0x90],
        ]
        .concat();
        let found = search(
            &entries,
            &[(0x2000, &twice), (0x3000, &[0x53, 0x56, 0x03, 0xC3, 0x90])],
        );
        assert_eq!(found.hits.len(), 0);
        assert_eq!(found.ambiguous, [(0, vec![0x2000, 0x2005, 0x3000])]);
        assert_eq!(search(&entries, &[]), Search::default());
    }
}
