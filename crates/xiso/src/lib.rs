//! Reader for original Xbox XDVDFS disc images.
//!
//! Supports plain XISO images and full redump images, whose game partition
//! starts at a fixed offset after the DVD video partition.
//!
//! # Examples
//!
//! ```no_run
//! use std::fs::File;
//!
//! let mut iso = xiso::XIso::open(File::open("game.iso")?)?;
//! let entry = iso.lookup("default.xbe")?.expect("not an Xbox game disc");
//! let bytes = iso.read_file(&entry)?;
//! println!("default.xbe is {} bytes", bytes.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
#[cfg(test)]
mod tests;

use std::io::{self, Read, Seek, SeekFrom};

pub use error::Error;
use error::Kind;

/// Size of one XDVDFS sector in bytes.
pub const SECTOR_SIZE: u64 = 2048;

const SECTOR_LEN: usize = 2048;
const MAGIC: &[u8; 20] = b"MICROSOFT*XBOX*MEDIA";
const VOLUME_DESCRIPTOR_SECTOR: u64 = 32;
const TRAILING_MAGIC_OFFSET: usize = 0x7EC;
const PARTITION_OFFSETS: [u64; 4] = [0, 0x1830_0000, 0x0FD9_0000, 0x0208_0000];
const ATTR_DIRECTORY: u8 = 0x10;
const DIR_ENTRY_HEADER: usize = 14;
const NO_SUBTREE: u16 = 0xFFFF;

/// One record from an XDVDFS directory table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    name: String,
    sector: u32,
    size: u32,
    attributes: u8,
}

impl Entry {
    /// Returns the record's name as stored on disc.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the size in bytes of the file or directory table.
    #[must_use]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Returns the first sector, relative to the game partition.
    #[must_use]
    pub fn sector(&self) -> u32 {
        self.sector
    }

    /// Returns `true` when the record describes a directory.
    #[must_use]
    pub fn is_dir(&self) -> bool {
        self.attributes & ATTR_DIRECTORY != 0
    }
}

/// An open XDVDFS image over any seekable reader.
#[derive(Debug)]
pub struct XIso<R> {
    reader: R,
    partition_offset: u64,
    root: Entry,
}

impl<R: Read + Seek> XIso<R> {
    /// Opens an image by locating its XDVDFS volume descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error when no volume descriptor exists at any known
    /// partition offset, or when the reader fails.
    pub fn open(mut reader: R) -> Result<Self, Error> {
        let mut descriptor = [0u8; SECTOR_LEN];
        for offset in PARTITION_OFFSETS {
            reader.seek(SeekFrom::Start(
                offset + VOLUME_DESCRIPTOR_SECTOR * SECTOR_SIZE,
            ))?;
            if read_up_to(&mut reader, &mut descriptor)? < SECTOR_LEN {
                continue;
            }
            let leading = &descriptor[..MAGIC.len()];
            let trailing = &descriptor[TRAILING_MAGIC_OFFSET..];
            if leading != MAGIC || trailing != MAGIC {
                continue;
            }
            let root = Entry {
                name: String::new(),
                sector: le_u32(&descriptor, 20),
                size: le_u32(&descriptor, 24),
                attributes: ATTR_DIRECTORY,
            };
            return Ok(Self {
                reader,
                partition_offset: offset,
                root,
            });
        }
        Err(Error::new(Kind::NotXdvdfs))
    }

    /// Returns the byte offset of the game partition inside the image.
    #[must_use]
    pub fn partition_offset(&self) -> u64 {
        self.partition_offset
    }

    /// Returns the root directory record.
    #[must_use]
    pub fn root(&self) -> &Entry {
        &self.root
    }

    /// Lists a directory, sorted case-insensitively by name.
    ///
    /// # Errors
    ///
    /// Returns an error when `dir` is not a directory, when its table is
    /// truncated or cyclic, or when the reader fails.
    pub fn read_dir(&mut self, dir: &Entry) -> Result<Vec<Entry>, Error> {
        if !dir.is_dir() {
            return Err(Error::new(Kind::NotADirectory(dir.name.clone())));
        }
        let table = self.read_extent(dir.sector, dir.size)?;
        let max_entries = table.len() / DIR_ENTRY_HEADER;
        let mut entries = Vec::new();
        let mut pending = if table.is_empty() {
            Vec::new()
        } else {
            vec![0]
        };
        while let Some(offset) = pending.pop() {
            let header = table
                .get(offset..offset + DIR_ENTRY_HEADER)
                .ok_or_else(|| Error::corrupt("directory record past end of table"))?;
            let left = le_u16(header, 0);
            let right = le_u16(header, 2);
            if left == NO_SUBTREE && right == NO_SUBTREE {
                continue;
            }
            let name_start = offset + DIR_ENTRY_HEADER;
            let name = table
                .get(name_start..name_start + usize::from(header[13]))
                .ok_or_else(|| Error::corrupt("directory record name past end of table"))?;
            entries.push(Entry {
                name: name.iter().copied().map(char::from).collect(),
                sector: le_u32(header, 4),
                size: le_u32(header, 8),
                attributes: header[12],
            });
            if entries.len() > max_entries {
                return Err(Error::corrupt("directory tree contains a cycle"));
            }
            for child in [left, right] {
                if child != 0 && child != NO_SUBTREE {
                    pending.push(usize::from(child) * 4);
                }
            }
        }
        entries.sort_by_key(|entry| entry.name.to_ascii_lowercase());
        Ok(entries)
    }

    /// Finds a file or directory by path, ignoring ASCII case.
    ///
    /// Components may be separated by `/` or `\`. An empty path names the
    /// root directory.
    ///
    /// # Errors
    ///
    /// Returns an error when a directory on the path cannot be read.
    pub fn lookup(&mut self, path: impl AsRef<str>) -> Result<Option<Entry>, Error> {
        let mut current = self.root.clone();
        for component in path.as_ref().split(['/', '\\']).filter(|c| !c.is_empty()) {
            if !current.is_dir() {
                return Ok(None);
            }
            let children = self.read_dir(&current)?;
            match children
                .into_iter()
                .find(|child| child.name.eq_ignore_ascii_case(component))
            {
                Some(child) => current = child,
                None => return Ok(None),
            }
        }
        Ok(Some(current))
    }

    /// Reads a whole file into memory.
    ///
    /// # Errors
    ///
    /// Returns an error when `entry` is a directory, when the file extends
    /// past the end of the image, or when the reader fails.
    pub fn read_file(&mut self, entry: &Entry) -> Result<Vec<u8>, Error> {
        if entry.is_dir() {
            return Err(Error::new(Kind::IsADirectory(entry.name.clone())));
        }
        self.read_extent(entry.sector, entry.size)
    }

    /// Reads up to `buf.len()` bytes of a file starting at `offset`.
    ///
    /// Returns the number of bytes read, which is zero at or past the end
    /// of the file.
    ///
    /// # Errors
    ///
    /// Returns an error when `entry` is a directory, when the file extends
    /// past the end of the image, or when the reader fails.
    pub fn read_at(&mut self, entry: &Entry, offset: u64, buf: &mut [u8]) -> Result<usize, Error> {
        if entry.is_dir() {
            return Err(Error::new(Kind::IsADirectory(entry.name.clone())));
        }
        let size = u64::from(entry.size);
        if offset >= size {
            return Ok(0);
        }
        let len = buf
            .len()
            .min(usize::try_from(size - offset).unwrap_or(usize::MAX));
        let start = self.partition_offset + u64::from(entry.sector) * SECTOR_SIZE + offset;
        self.reader.seek(SeekFrom::Start(start))?;
        self.reader.read_exact(&mut buf[..len]).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                Error::corrupt("extent extends past end of image")
            } else {
                e.into()
            }
        })?;
        Ok(len)
    }

    fn read_extent(&mut self, sector: u32, size: u32) -> Result<Vec<u8>, Error> {
        let start = self.partition_offset + u64::from(sector) * SECTOR_SIZE;
        self.reader.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; size as usize];
        self.reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                Error::corrupt("extent extends past end of image")
            } else {
                e.into()
            }
        })?;
        Ok(buf)
    }
}

/// Reads until `buf` is full or the reader is exhausted.
fn read_up_to(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

fn le_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
