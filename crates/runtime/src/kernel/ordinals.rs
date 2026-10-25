//! The kernel's ordinal table, read from nxdk's CC0 `xboxkrnl.exe.def` (spec 8.1).
//!
//! A line of the `EXPORTS` section names one export: `Name@N` is stdcall with `N` argument
//! bytes, `@Name@N` is fastcall, a bare `Name` is cdecl, and `DATA` marks a data export.

use std::sync::LazyLock;

/// The def file, as nxdk ships it.
const DEF: &str = include_str!("../../data/xboxkrnl.exe.def");

/// The number of ordinals (0..384); data export storage is reserved for all of them.
pub const COUNT: u32 = 384;

/// How a function export takes its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convention {
    Stdcall,
    Fastcall,
    Cdecl,
}

/// One export.
#[derive(Debug, Clone)]
pub struct Export {
    pub name: &'static str,
    pub ordinal: u32,
    pub convention: Convention,
    /// The argument bytes (stdcall and fastcall); 0 for cdecl and data.
    pub arg_bytes: u32,
    /// Whether this is a data export.
    pub data: bool,
}

impl Export {
    /// The bytes the callee pops besides the return address (spec 5.2).
    #[must_use]
    pub fn callee_pops(&self) -> u32 {
        match self.convention {
            Convention::Stdcall => self.arg_bytes,
            Convention::Fastcall => self.arg_bytes.saturating_sub(8),
            Convention::Cdecl => 0,
        }
    }
}

/// Parses one line of the `EXPORTS` section.
fn parse_line(line: &'static str) -> Option<Export> {
    let line = line.split(';').next()?.trim();
    let mut fields = line.split_whitespace();
    let symbol = fields.next()?;
    if fields.next()? != "@" {
        return None;
    }
    let ordinal = fields.next()?.parse().ok()?;
    let data = fields.any(|f| f == "DATA");
    let (convention, rest) = match symbol.strip_prefix('@') {
        Some(rest) => (Convention::Fastcall, rest),
        None => (Convention::Stdcall, symbol),
    };
    let (name, arg_bytes, convention) = match rest.rsplit_once('@') {
        Some((name, bytes)) => (name, bytes.parse().ok()?, convention),
        None => (rest, 0, Convention::Cdecl),
    };
    Some(Export {
        name,
        ordinal,
        convention,
        arg_bytes,
        data,
    })
}

/// The exports by ordinal.
static TABLE: LazyLock<Vec<Option<Export>>> = LazyLock::new(|| {
    let mut table = vec![None; COUNT as usize];
    for export in DEF.lines().filter_map(parse_line) {
        if let Some(slot) = table.get_mut(export.ordinal as usize) {
            *slot = Some(export);
        }
    }
    table
});

/// The export with `ordinal`, if there is one.
#[must_use]
pub fn export(ordinal: u32) -> Option<&'static Export> {
    TABLE.get(ordinal as usize)?.as_ref()
}

/// The export named `name`, if there is one.
#[must_use]
pub fn by_name(name: &str) -> Option<&'static Export> {
    TABLE.iter().flatten().find(|e| e.name == name)
}

/// Every export.
pub fn all() -> impl Iterator<Item = &'static Export> {
    TABLE.iter().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_conventions_come_from_the_decoration() {
        let create = export(255).unwrap();
        assert_eq!(create.name, "PsCreateSystemThreadEx");
        assert_eq!(create.convention, Convention::Stdcall);
        assert_eq!(create.arg_bytes, 40);
        assert_eq!(create.callee_pops(), 40);

        let statistic = export(20).unwrap();
        assert_eq!(statistic.name, "ExInterlockedAddLargeStatistic");
        assert_eq!(statistic.convention, Convention::Fastcall);
        assert_eq!(statistic.callee_pops(), 0);

        let print = export(8).unwrap();
        assert_eq!(print.name, "DbgPrint");
        assert_eq!(print.convention, Convention::Cdecl);
        assert_eq!(print.callee_pops(), 0);
    }

    #[test]
    fn data_exports_are_marked() {
        assert!(export(16).unwrap().data); // ExEventObjectType
        assert!(export(156).unwrap().data); // KeTickCount
        assert!(!export(14).unwrap().data); // ExAllocatePool
        assert_eq!(by_name("XboxKrnlVersion").unwrap().ordinal, 324);
    }

    #[test]
    fn the_table_covers_the_kernel() {
        assert!(all().count() > 360);
        assert!(export(0).is_none());
    }
}
