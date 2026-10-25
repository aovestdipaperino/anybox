//! Error type for XBE parsing.

use std::backtrace::Backtrace;
use std::fmt;

/// An error raised while parsing an XBE image.
#[derive(Debug)]
pub struct Error {
    kind: Kind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum Kind {
    BadMagic,
    Corrupt(&'static str),
    UnknownKeys,
}

impl Error {
    pub(crate) fn bad_magic() -> Self {
        Self::new(Kind::BadMagic)
    }

    pub(crate) fn corrupt(what: &'static str) -> Self {
        Self::new(Kind::Corrupt(what))
    }

    pub(crate) fn unknown_keys() -> Self {
        Self::new(Kind::UnknownKeys)
    }

    fn new(kind: Kind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns `true` when the data does not start with `XBEH`.
    #[must_use]
    pub fn is_bad_magic(&self) -> bool {
        matches!(self.kind, Kind::BadMagic)
    }

    /// Returns `true` when a header field points outside the image.
    #[must_use]
    pub fn is_corrupt(&self) -> bool {
        matches!(self.kind, Kind::Corrupt(_))
    }

    /// Returns `true` when neither retail nor debug keys decode the header.
    #[must_use]
    pub fn is_unknown_keys(&self) -> bool {
        matches!(self.kind, Kind::UnknownKeys)
    }

    /// Returns the backtrace captured when the error was created.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            Kind::BadMagic => f.write_str("not an XBE: missing XBEH magic"),
            Kind::Corrupt(what) => write!(f, "corrupt XBE: {what}"),
            Kind::UnknownKeys => {
                f.write_str("XBE entry point matches neither retail nor debug keys")
            }
        }
    }
}

impl std::error::Error for Error {}
