//! Error type for XDVDFS image access.

use std::backtrace::Backtrace;
use std::fmt;
use std::io;

/// An error raised while reading an XDVDFS image.
#[derive(Debug)]
pub struct Error {
    kind: Kind,
    backtrace: Backtrace,
}

#[derive(Debug)]
pub(crate) enum Kind {
    Io(io::Error),
    NotXdvdfs,
    Corrupt(&'static str),
    NotADirectory(String),
    IsADirectory(String),
}

impl Error {
    pub(crate) fn new(kind: Kind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    pub(crate) fn corrupt(what: &'static str) -> Self {
        Self::new(Kind::Corrupt(what))
    }

    /// Returns `true` when no XDVDFS volume descriptor was found.
    #[must_use]
    pub fn is_not_xdvdfs(&self) -> bool {
        matches!(self.kind, Kind::NotXdvdfs)
    }

    /// Returns `true` when on-disc structures are inconsistent or truncated.
    #[must_use]
    pub fn is_corrupt(&self) -> bool {
        matches!(self.kind, Kind::Corrupt(_))
    }

    /// Returns `true` when the underlying reader failed.
    #[must_use]
    pub fn is_io(&self) -> bool {
        matches!(self.kind, Kind::Io(_))
    }

    /// Returns `true` when a file was used where a directory was required.
    #[must_use]
    pub fn is_not_a_directory(&self) -> bool {
        matches!(self.kind, Kind::NotADirectory(_))
    }

    /// Returns `true` when a directory was used where a file was required.
    #[must_use]
    pub fn is_a_directory(&self) -> bool {
        matches!(self.kind, Kind::IsADirectory(_))
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
            Kind::Io(e) => write!(f, "I/O error reading XDVDFS image: {e}"),
            Kind::NotXdvdfs => f.write_str("no XDVDFS volume descriptor found"),
            Kind::Corrupt(what) => write!(f, "corrupt XDVDFS image: {what}"),
            Kind::NotADirectory(name) => write!(f, "'{name}' is not a directory"),
            Kind::IsADirectory(name) => write!(f, "'{name}' is a directory"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            Kind::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::new(Kind::Io(e))
    }
}
