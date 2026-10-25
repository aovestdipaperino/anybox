//! Finds statically linked Xbox XDK functions by byte signature.
//!
//! Every Xbox title carries its own copy of the XDK libraries (XAPI, D3D8,
//! `DirectSound` and the USB stack) inside its executable, so there is no
//! import table to bind. Instead, known functions are recognised by their
//! machine code, with the parts that depend on link addresses masked out.
//!
//! # Examples
//!
//! ```
//! use xdk::Pattern;
//!
//! let pattern = Pattern::parse("53 56 ?? C3")?;
//! assert_eq!(pattern.find_all(&[0, 0x53, 0x56, 0x99, 0xC3]), [1]);
//! # Ok::<(), xdk::PatternError>(())
//! ```

mod database;
mod library;
mod pattern;
mod plan;

pub use database::{DatabaseError, Entry, Hit, Search, parse, parse_owned, search};
pub use library::{EntryPoint, entry_points};
pub use pattern::{Pattern, PatternError};
pub use plan::{Hook, Plan, Section, plan};
