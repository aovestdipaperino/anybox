//! Converts `XbSymbolDatabase`'s OOVPA signatures into the `name | argument bytes | pattern`
//! format of `xdk::parse`, for the functions the runtime replaces.
//!
//! `XbSymbolDatabase` (<https://github.com/Cxbx-Reloaded/XbSymbolDatabase>) is MIT licensed. Its
//! signatures are sparse `{offset, byte}` lists per function and XDK build; the bytes between
//! them, and the call targets it checks by cross-reference, become `??` here.
//!
//! ```text
//! cargo run -p xdk --example oovpa -- XBSDB/src/OOVPADatabase 4831 > signatures.txt
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The functions the runtime replaces (spec 9.1, 9.2), by their names in `XbSymbolDatabase`.
const WANTED: &[&str] = &[
    // XInput (spec 9.1).
    "XInitDevices",
    "XGetDevices",
    "XGetDeviceChanges",
    "XInputOpen",
    "XInputClose",
    "XInputPoll",
    "XInputGetState",
    "XInputSetState",
    "XInputGetCapabilities",
    // DirectSound (spec 9.2).
    "DirectSoundCreate",
    "DirectSoundCreateBuffer",
    "DirectSoundCreateStream",
    "IDirectSound_DownloadEffectsImage",
    "IDirectSound_SetPosition",
    "IDirectSound_SetOrientation",
    "IDirectSound_SetRolloffFactor",
    "IDirectSound_SetVelocity",
    "IDirectSound_SetDistanceFactor",
    "IDirectSound_SetDopplerFactor",
    "IDirectSound_SetAllParameters",
    "IDirectSound_SetMixBinHeadroom",
    "IDirectSound_EnableHeadphones",
    "IDirectSound_SynchPlayback",
    "IDirectSound_CommitDeferredSettings",
    "IDirectSoundBuffer_SetBufferData",
    "IDirectSoundBuffer_SetPlayRegion",
    "IDirectSoundBuffer_SetLoopRegion",
    "IDirectSoundBuffer_PlayEx",
    "IDirectSoundBuffer_Play",
    "IDirectSoundBuffer_Stop",
    "IDirectSoundBuffer_GetStatus",
    "IDirectSoundBuffer_SetVolume",
    // `IDirectSoundBuffer_SetFrequency` matches several thunks in the titles tried; the runtime's
    // `signatures-titles.txt` has one that follows the thunk's calls instead.
    "IDirectSoundBuffer_SetFormat",
    "IDirectSoundBuffer_SetPosition",
    "IDirectSoundBuffer_SetMinDistance",
    "IDirectSoundBuffer_SetMaxDistance",
    "IDirectSoundBuffer_SetRolloffFactor",
    "IDirectSoundBuffer_SetMode",
    "IDirectSoundBuffer_SetVelocity",
    "IDirectSoundBuffer_SetConeAngles",
    "IDirectSoundBuffer_SetConeOrientation",
    "IDirectSoundBuffer_SetConeOutsideVolume",
    "IDirectSoundBuffer_SetDistanceFactor",
    "IDirectSoundBuffer_SetDopplerFactor",
    "IDirectSoundBuffer_SetAllParameters",
    "IDirectSoundStream_SetVolume",
];

/// The fewest known bytes a usable signature has.
const MIN_KNOWN_BYTES: usize = 5;

/// A function's registration: calling convention, argument bytes, and signature versions.
#[derive(Debug)]
struct Registration {
    convention: String,
    arg_bytes: u32,
    versions: Vec<u32>,
}

/// The text inside the parentheses that open at `open` (which holds `(`), and the index after
/// the closing one.
fn balanced(text: &str, open: usize) -> Option<(&str, usize)> {
    let mut depth = 0;
    for (i, c) in text[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&text[open + 1..open + i], open + i + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// Removes `//` and `/* */` comments.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("//") {
            rest = after.find('\n').map_or("", |i| &after[i..]);
        } else if let Some(after) = rest.strip_prefix("/*") {
            rest = after.find("*/").map_or("", |i| &after[i + 2..]);
        } else {
            let c = rest.chars().next().unwrap_or(' ');
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// The registrations in a library table (`<Library>_OOVPA.c`), by name.
fn registrations(table: &str) -> BTreeMap<String, Registration> {
    let table = strip_comments(table);
    let mut found = BTreeMap::new();
    let mut at = 0;
    while let Some(i) = table[at..].find("SYM_FUN(") {
        let open = at + i + "SYM_FUN".len();
        let Some((fun, after)) = balanced(&table, open) else {
            break;
        };
        at = after;
        let Some(sig_at) = table[after..].find("SYM_SIG(") else {
            break;
        };
        let Some((sig, _)) = balanced(&table, after + sig_at + "SYM_SIG".len()) else {
            break;
        };
        let name = fun.split(',').next().unwrap_or("").trim().to_owned();
        let field = |tag: &str| {
            fun.find(tag)
                .and_then(|p| balanced(fun, p + tag.len() - 1))
                .map(|(inner, _)| inner.trim().to_owned())
        };
        let convention = field("CALL(").unwrap_or_default();
        let pushes = fun.matches("PARAM(psh,").count() + fun.matches("PARAM(psh ,").count();
        let wide_pushes = fun.matches("PARAM(psh2").count();
        let default_bytes = u32::try_from(4 * pushes + 8 * wide_pushes).unwrap_or(0);
        let arg_bytes = field("STACK(")
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_bytes);
        let versions = sig
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        found.insert(
            name,
            Registration {
                convention,
                arg_bytes,
                versions,
            },
        );
    }
    found
}

/// The `DetectSelect` of a signature header, as an occurrence counting from 1.
fn occurrence(header: &str) -> Option<usize> {
    ["DetectFirst", "DetectSecond", "DetectThird", "DetectFourth"]
        .iter()
        .position(|d| header.contains(d))
        .map(|i| i + 1)
}

/// Parses a number written in C (`0x1F` or `31`).
fn number(text: &str) -> Option<u32> {
    let text = text.trim();
    match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => text.parse().ok(),
    }
}

/// A signature: its known bytes, its references (`XREF_` names without the prefix) by offset,
/// and its occurrence.
#[derive(Debug, Default)]
struct Signature {
    bytes: BTreeMap<u32, u8>,
    references: BTreeMap<u32, String>,
    occurrence: Option<usize>,
}

/// Reads a signature body into `signature`.
fn read_body(body: &str, signature: &mut Signature) -> Result<(), String> {
    let put = |bytes: &mut BTreeMap<u32, u8>, offset: u32, value: u32| -> Result<(), String> {
        let value = u8::try_from(value).map_err(|_| format!("{value:#x} is not a byte"))?;
        if bytes.insert(offset, value).is_some_and(|old| old != value) {
            return Err(format!("offset {offset:#x} has two values"));
        }
        Ok(())
    };
    let mut at = 0;
    while at < body.len() {
        let rest = &body[at..];
        if rest.starts_with("OV_MATCH(") {
            let (inner, after) = balanced(body, at + "OV_MATCH".len()).ok_or("an open OV_MATCH")?;
            let mut values = inner.split(',').map(str::trim);
            let offset = values
                .next()
                .and_then(number)
                .ok_or("an OV_MATCH without an offset")?;
            for (i, value) in values.enumerate() {
                let at = offset + u32::try_from(i).unwrap_or(0);
                if let Some(name) = value.strip_prefix("XREF_") {
                    // A reference written as a match: the first `XRefCount` entries are.
                    signature.references.insert(at, name.to_owned());
                    continue;
                }
                let value = number(value).ok_or("an OV_MATCH byte that is not a number")?;
                put(&mut signature.bytes, at, value)?;
            }
            at = after;
        } else if rest.starts_with("XREF_ENTRY(") {
            let (inner, after) =
                balanced(body, at + "XREF_ENTRY".len()).ok_or("an open XREF_ENTRY")?;
            let mut fields = inner.split(',').map(str::trim);
            let offset = fields
                .next()
                .and_then(number)
                .ok_or("an XREF_ENTRY without an offset")?;
            if let Some(name) = fields.next().and_then(|n| n.strip_prefix("XREF_")) {
                signature.references.insert(offset, name.to_owned());
            }
            at = after;
        } else if rest.starts_with('{') {
            let end = rest.find('}').ok_or("an open brace")?;
            let mut values = rest[1..end].split(',').map(str::trim);
            if let (Some(offset), Some(value)) = (values.next().and_then(number), values.next()) {
                if let Some(name) = value.strip_prefix("XREF_") {
                    signature.references.insert(offset, name.to_owned());
                } else if let Some(value) = number(value) {
                    put(&mut signature.bytes, offset, value)?;
                }
            }
            at += end + 1;
        } else {
            at += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    Ok(())
}

/// Finds the signature of `name` at `version` in a library's `<version>.inl`.
fn signature(directory: &Path, name: &str, version: u32) -> Result<Signature, String> {
    let path = directory.join(format!("{version}.inl"));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = strip_comments(&text);
    let mut at = 0;
    while let Some(i) = text[at..].find("OOVPA_") {
        let start = at + i;
        at = start + 1;
        let Some(open) = text[start..].find('(').map(|p| start + p) else {
            break;
        };
        let macro_name = &text[start..open];
        let is_identifier = macro_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !is_identifier
            || (!macro_name.starts_with("OOVPA_SIG_HEADER")
                && !macro_name.ends_with("XREF")
                && !macro_name.ends_with("DETECT"))
        {
            continue;
        }
        let Some((header, after)) = balanced(&text, open) else {
            break;
        };
        let mut fields = header.split(',').map(str::trim);
        if fields.next() != Some(name) || fields.next().and_then(number) != Some(version) {
            continue;
        }
        let body = if macro_name.starts_with("OOVPA_SIG_HEADER") {
            let rest = &text[after..];
            let open = rest
                .find("OOVPA_SIG_MATCH")
                .ok_or("a header without OOVPA_SIG_MATCH")?;
            if rest[open..].starts_with("OOVPA_SIG_MATCH_DUMMY") {
                return Err("a dummy signature".to_owned());
            }
            let paren = after + open + "OOVPA_SIG_MATCH".len();
            balanced(&text, paren).ok_or("an open OOVPA_SIG_MATCH")?.0
        } else {
            let end = text[after..]
                .find("OOVPA_END")
                .ok_or("a signature without OOVPA_END")?;
            &text[after..after + end]
        };
        let mut found = Signature {
            occurrence: occurrence(header),
            ..Signature::default()
        };
        read_body(body, &mut found)?;
        return Ok(found);
    }
    Err(format!(
        "no signature of {name} at {version} in {}",
        path.display()
    ))
}

/// The runtime's name for an `XbSymbolDatabase` name: `IDirectSoundBuffer_Play` is
/// `IDirectSoundBuffer::Play`.
fn runtime_name(name: &str) -> String {
    for class in ["IDirectSoundBuffer", "IDirectSoundStream", "IDirectSound"] {
        if let Some(method) = name.strip_prefix(class).and_then(|m| m.strip_prefix('_')) {
            return format!("{class}::{method}");
        }
    }
    name.to_owned()
}

/// Every registration in the database, by name, with its library's directory.
fn all_registrations(database: &Path) -> Result<BTreeMap<String, (PathBuf, Registration)>, String> {
    let mut all = BTreeMap::new();
    let listing =
        std::fs::read_dir(database).map_err(|e| format!("{}: {e}", database.display()))?;
    let mut tables: Vec<PathBuf> = listing
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with("_OOVPA.c"))
        .collect();
    tables.sort();
    for table in tables {
        let library = table
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix("_OOVPA.c"))
            .ok_or("a table without a library name")?;
        let directory = database.join(library);
        let text =
            std::fs::read_to_string(&table).map_err(|e| format!("{}: {e}", table.display()))?;
        for (name, registration) in registrations(&text) {
            all.entry(name).or_insert((directory.clone(), registration));
        }
    }
    Ok(all)
}

/// The converter's state: the registrations, the build, and the signatures read so far.
struct Converter {
    registrations: BTreeMap<String, (PathBuf, Registration)>,
    build: u32,
    /// Signatures by name; `None` for a name without a usable one.
    read: BTreeMap<String, Option<Signature>>,
}

impl Converter {
    /// Reads the signature of `name` and, recursively, of what it refers to.
    fn load(&mut self, name: &str, depth: usize) {
        if self.read.contains_key(name) || depth > 8 {
            return;
        }
        let loaded = self
            .registrations
            .get(name)
            .and_then(|(directory, registration)| {
                let version = registration
                    .versions
                    .iter()
                    .filter(|&&v| v <= self.build)
                    .max()?;
                match signature(directory, name, *version) {
                    Ok(signature) => Some(signature),
                    Err(error) => {
                        eprintln!("oovpa: {name} ({version}): {error}");
                        None
                    }
                }
            });
        let references: Vec<String> = loaded
            .as_ref()
            .map(|s| s.references.values().cloned().collect())
            .unwrap_or_default();
        self.read.insert(name.to_owned(), loaded);
        for reference in references {
            self.load(&reference, depth + 1);
        }
    }

    /// Whether `name` has a signature that can be emitted.
    fn usable(&self, name: &str) -> bool {
        self.pattern(name).is_some()
    }

    /// The pattern of `name`: known bytes, `<Reference>` where the reference is usable, else
    /// wildcards. `None` with too few known bytes.
    fn pattern(&self, name: &str) -> Option<String> {
        self.pattern_guarded(name, &mut Vec::new())
    }

    fn pattern_guarded(&self, name: &str, visiting: &mut Vec<String>) -> Option<String> {
        let signature = self.read.get(name)?.as_ref()?;
        if visiting.iter().any(|v| v == name) {
            // A cycle: count the entry usable while it is being checked.
            return Some(String::new());
        }
        visiting.push(name.to_owned());
        let references: BTreeMap<u32, &str> = signature
            .references
            .iter()
            .filter(|(_, target)| {
                target.as_str() != name && self.pattern_guarded(target, visiting).is_some()
            })
            .map(|(offset, target)| (*offset, target.as_str()))
            .collect();
        visiting.pop();
        let known = signature.bytes.len() + 4 * references.len();
        let last_byte = signature.bytes.keys().next_back().copied();
        let last_reference = references.keys().next_back().map(|o| o + 3);
        let last = last_byte.into_iter().chain(last_reference).max()?;
        if known < MIN_KNOWN_BYTES {
            return None;
        }
        let mut tokens = Vec::new();
        let mut offset = 0;
        while offset <= last {
            if let Some(target) = references.get(&offset) {
                tokens.push(format!("<{}>", runtime_name(target)));
                offset += 4;
            } else {
                tokens.push(
                    signature
                        .bytes
                        .get(&offset)
                        .map_or_else(|| "??".to_owned(), |b| format!("{b:02X}")),
                );
                offset += 1;
            }
        }
        Some(tokens.join(" "))
    }

    /// The referenced names `name` needs, recursively.
    fn needed(&self, name: &str, out: &mut Vec<String>) {
        let Some(Some(signature)) = self.read.get(name) else {
            return;
        };
        for target in signature.references.values() {
            if target != name && self.usable(target) && !out.contains(target) {
                out.push(target.clone());
                self.needed(target, out);
            }
        }
    }
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (Some(database), Some(build)) = (args.next(), args.next()) else {
        return Err("usage: oovpa OOVPA_DATABASE_DIR XDK_BUILD".to_owned());
    };
    let database = PathBuf::from(database);
    let build: u32 = build
        .parse()
        .map_err(|_| format!("`{build}` is not an XDK build"))?;
    let mut converter = Converter {
        registrations: all_registrations(&database)?,
        build,
        read: BTreeMap::new(),
    };
    let mut lines = Vec::new();
    let mut emitted = Vec::new();
    let mut helpers = Vec::new();
    for &name in WANTED {
        let Some((_, registration)) = converter.registrations.get(name) else {
            eprintln!("oovpa: {name} is not registered; skipped");
            continue;
        };
        if registration.convention != "std" {
            eprintln!(
                "oovpa: {name} is not stdcall ({}); skipped",
                registration.convention
            );
            continue;
        }
        let arg_bytes = registration.arg_bytes;
        converter.load(name, 0);
        let Some(pattern) = converter.pattern(name) else {
            eprintln!("oovpa: {name} has no usable signature for XDK {build}; skipped");
            continue;
        };
        let occurrence = converter.read[name].as_ref().and_then(|s| s.occurrence);
        let label = runtime_name(name) + &occurrence.map(|n| format!("@{n}")).unwrap_or_default();
        lines.push(format!("{label} | {arg_bytes} | {pattern}"));
        emitted.push(name.to_owned());
        converter.needed(name, &mut helpers);
    }
    for helper in helpers.iter().filter(|h| !emitted.contains(h)) {
        if let Some(pattern) = converter.pattern(helper) {
            let occurrence = converter.read[helper].as_ref().and_then(|s| s.occurrence);
            let label =
                runtime_name(helper) + &occurrence.map(|n| format!("@{n}")).unwrap_or_default();
            lines.push(format!("{label} | - | {pattern}"));
        }
    }
    println!("# Signatures of the XDK {build} functions the runtime replaces, converted from");
    println!("# XbSymbolDatabase (MIT, Copyright (c) 2023 The XbSymbolDatabase authors) by");
    println!(
        "# `cargo run -p xdk --example oovpa`. Unknown bytes are `??`; `<Name>` is a call target"
    );
    println!("# or address that must be the function `Name`; entries with `-` are only located.");
    for line in lines {
        println!("{line}");
    }
    Ok(())
}
