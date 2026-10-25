//! Parser for original Xbox executables (XBE).
//!
//! An XBE is the Xbox counterpart of a PE image. This crate decodes its
//! header, certificate, section table, linked XDK library versions and
//! kernel import table, and reads bytes by virtual address.
//!
//! # Examples
//!
//! ```no_run
//! let xbe = xbe::Xbe::parse(std::fs::read("default.xbe")?)?;
//! println!(
//!     "{} imports {} kernel exports",
//!     xbe.certificate().title_name(),
//!     xbe.kernel_imports()?.len()
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
#[cfg(test)]
mod tests;

use std::fmt;

pub use error::Error;

const MAGIC: &[u8; 4] = b"XBEH";
const IMAGE_HEADER_LEN: usize = 0x178;
const RETAIL_ENTRY_KEY: u32 = 0xA8FC_57AB;
const DEBUG_ENTRY_KEY: u32 = 0x9485_9D4B;
const RETAIL_THUNK_KEY: u32 = 0x5B6D_40B6;
const DEBUG_THUNK_KEY: u32 = 0xEFB1_F152;
const SECTION_HEADER_LEN: u32 = 56;
const LIBRARY_VERSION_LEN: u32 = 16;
const CERTIFICATE_LEN: usize = 0xB0;
const TITLE_NAME_UNITS: usize = 40;
const IMPORT_BY_ORDINAL: u32 = 0x8000_0000;
const MAX_KERNEL_IMPORTS: u32 = 1024;

/// Which signing keys scramble the entry point and thunk address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildKind {
    /// A retail (production-signed) executable.
    Retail,
    /// A debug-kit executable.
    Debug,
}

/// Load and protection flags of one XBE section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectionFlags(u32);

impl SectionFlags {
    /// Returns the raw flag bits.
    #[must_use]
    pub fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` when the section may be written at run time.
    #[must_use]
    pub fn writable(self) -> bool {
        self.0 & 0x01 != 0
    }

    /// Returns `true` when the loader maps the section before entry.
    #[must_use]
    pub fn preload(self) -> bool {
        self.0 & 0x02 != 0
    }

    /// Returns `true` when the section contains code.
    #[must_use]
    pub fn executable(self) -> bool {
        self.0 & 0x04 != 0
    }

    /// Returns `true` when the section is an inserted data file.
    #[must_use]
    pub fn inserted_file(self) -> bool {
        self.0 & 0x08 != 0
    }

    /// Returns `true` when the section's first page is read-only.
    #[must_use]
    pub fn head_page_read_only(self) -> bool {
        self.0 & 0x10 != 0
    }

    /// Returns `true` when the section's last page is read-only.
    #[must_use]
    pub fn tail_page_read_only(self) -> bool {
        self.0 & 0x20 != 0
    }
}

/// One entry of the XBE section table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    name: String,
    flags: SectionFlags,
    virtual_address: u32,
    virtual_size: u32,
    raw_address: u32,
    raw_size: u32,
}

impl Section {
    /// Returns the section name, such as `.text` or `D3D`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the section's load and protection flags.
    #[must_use]
    pub fn flags(&self) -> SectionFlags {
        self.flags
    }

    /// Returns the address the section is loaded at.
    #[must_use]
    pub fn virtual_address(&self) -> u32 {
        self.virtual_address
    }

    /// Returns the in-memory size; bytes past the raw size are zero.
    #[must_use]
    pub fn virtual_size(&self) -> u32 {
        self.virtual_size
    }

    /// Returns the file offset of the section's bytes.
    #[must_use]
    pub fn raw_address(&self) -> u32 {
        self.raw_address
    }

    /// Returns the number of bytes stored in the file.
    #[must_use]
    pub fn raw_size(&self) -> u32 {
        self.raw_size
    }
}

/// An XDK library version stamp linked into the executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryVersion {
    name: String,
    major: u16,
    minor: u16,
    build: u16,
    flags: u16,
}

impl LibraryVersion {
    /// Returns the library name, such as `D3D8` or `XAPILIB`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the major version number.
    #[must_use]
    pub fn major(&self) -> u16 {
        self.major
    }

    /// Returns the minor version number.
    #[must_use]
    pub fn minor(&self) -> u16 {
        self.minor
    }

    /// Returns the XDK build number, such as 5849.
    #[must_use]
    pub fn build(&self) -> u16 {
        self.build
    }

    /// Returns the raw flags (QFE number, debug and approval bits).
    #[must_use]
    pub fn flags(&self) -> u16 {
        self.flags
    }
}

impl fmt::Display for LibraryVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}.{}.{}",
            self.name, self.major, self.minor, self.build
        )
    }
}

/// Title metadata from the XBE certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Certificate {
    title_id: u32,
    title_name: String,
    game_region: u32,
    version: u32,
}

impl Certificate {
    /// Returns the raw 32-bit title ID.
    #[must_use]
    pub fn title_id(&self) -> u32 {
        self.title_id
    }

    /// Formats the title ID as publisher code and number, like `UB-034`.
    #[must_use]
    pub fn title_id_display(&self) -> String {
        let bytes = self.title_id.to_be_bytes();
        format!(
            "{}{}-{:03}",
            char::from(bytes[0]),
            char::from(bytes[1]),
            self.title_id & 0xFFFF
        )
    }

    /// Returns the title's display name.
    #[must_use]
    pub fn title_name(&self) -> &str {
        &self.title_name
    }

    /// Returns the region bit mask (1 = North America, 2 = Japan, 4 = rest of world).
    #[must_use]
    pub fn game_region(&self) -> u32 {
        self.game_region
    }

    /// Returns the title's version number.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }
}

/// A parsed XBE image that owns its bytes.
#[derive(Debug, Clone)]
pub struct Xbe {
    data: Vec<u8>,
    base_address: u32,
    headers_size: u32,
    image_size: u32,
    build_kind: BuildKind,
    entry_point: u32,
    kernel_thunk_address: u32,
    tls_address: u32,
    certificate: Certificate,
    sections: Vec<Section>,
    libraries: Vec<LibraryVersion>,
}

impl Xbe {
    /// Parses an XBE image from its raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when the magic is missing, when a header field
    /// points outside the header block or file, or when neither the retail
    /// nor the debug keys yield an entry point inside the image.
    pub fn parse(data: impl Into<Vec<u8>>) -> Result<Self, Error> {
        let data = data.into();
        if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
            return Err(Error::bad_magic());
        }
        if data.len() < IMAGE_HEADER_LEN {
            return Err(Error::corrupt("image header truncated"));
        }
        let field = |offset| le_u32(&data, offset);
        let base_address = field(0x104);
        let headers_size = field(0x108);
        let image_size = field(0x10C);
        let headers_len = headers_size as usize;
        if headers_len < IMAGE_HEADER_LEN || headers_len > data.len() {
            return Err(Error::corrupt("header size out of range"));
        }
        let (build_kind, entry_point, kernel_thunk_address) =
            decode_keys(base_address, image_size, field(0x128), field(0x158))?;
        let headers = Headers {
            bytes: &data[..headers_len],
            base: base_address,
        };
        let certificate = parse_certificate(&headers, field(0x118))?;
        let sections = (0..field(0x11C))
            .map(|i| {
                let address = table_entry(field(0x120), i, SECTION_HEADER_LEN)?;
                parse_section(&headers, address, data.len())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let libraries = (0..field(0x160))
            .map(|i| {
                let address = table_entry(field(0x164), i, LIBRARY_VERSION_LEN)?;
                parse_library(&headers, address)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tls_address = field(0x12C);
        Ok(Self {
            data,
            base_address,
            headers_size,
            image_size,
            build_kind,
            entry_point,
            kernel_thunk_address,
            tls_address,
            certificate,
            sections,
            libraries,
        })
    }

    /// Returns the address the header block is loaded at, normally 0x10000.
    #[must_use]
    pub fn base_address(&self) -> u32 {
        self.base_address
    }

    /// Returns the size of the header block in bytes.
    #[must_use]
    pub fn headers_size(&self) -> u32 {
        self.headers_size
    }

    /// Returns the size of the loaded image in bytes.
    #[must_use]
    pub fn image_size(&self) -> u32 {
        self.image_size
    }

    /// Returns which keys decoded the entry point and thunk address.
    #[must_use]
    pub fn build_kind(&self) -> BuildKind {
        self.build_kind
    }

    /// Returns the decoded entry point address.
    #[must_use]
    pub fn entry_point(&self) -> u32 {
        self.entry_point
    }

    /// Returns the decoded address of the kernel thunk table.
    #[must_use]
    pub fn kernel_thunk_address(&self) -> u32 {
        self.kernel_thunk_address
    }

    /// Returns the address of the TLS directory, or zero when absent.
    #[must_use]
    pub fn tls_address(&self) -> u32 {
        self.tls_address
    }

    /// Returns the title certificate.
    #[must_use]
    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }

    /// Returns the section table in file order.
    #[must_use]
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// Returns the XDK library versions linked into the image.
    #[must_use]
    pub fn libraries(&self) -> &[LibraryVersion] {
        &self.libraries
    }

    /// Returns the bytes stored in the file for `section`.
    ///
    /// # Panics
    ///
    /// Panics when `section` did not come from this image and its raw range
    /// lies outside the file.
    #[must_use]
    pub fn section_data(&self, section: &Section) -> &[u8] {
        let start = section.raw_address as usize;
        &self.data[start..start + section.raw_size as usize]
    }

    /// Reads `len` file-backed bytes at a virtual address.
    ///
    /// Returns `None` when the range is not fully backed by the header
    /// block or by one section's raw data.
    #[must_use]
    pub fn read_virtual(&self, address: u32, len: usize) -> Option<&[u8]> {
        let offset_within = |start: u32, available: u32| {
            let delta = address.checked_sub(start)?;
            let end = u64::from(delta) + len as u64;
            (end <= u64::from(available)).then_some(delta as usize)
        };
        let file_offset = offset_within(self.base_address, self.headers_size).or_else(|| {
            self.sections.iter().find_map(|s| {
                offset_within(s.virtual_address, s.raw_size)
                    .map(|delta| s.raw_address as usize + delta)
            })
        })?;
        self.data.get(file_offset..file_offset + len)
    }

    /// Returns the kernel ordinals imported through the thunk table.
    ///
    /// # Errors
    ///
    /// Returns an error when the table lies outside the image, is not
    /// zero-terminated, or contains an import that is not by ordinal.
    pub fn kernel_imports(&self) -> Result<Vec<u32>, Error> {
        let mut ordinals = Vec::new();
        for index in 0..MAX_KERNEL_IMPORTS {
            let address = table_entry(self.kernel_thunk_address, index, 4)?;
            let slot = self
                .read_virtual(address, 4)
                .ok_or_else(|| Error::corrupt("kernel thunk table outside image"))?;
            let value = le_u32(slot, 0);
            if value == 0 {
                return Ok(ordinals);
            }
            if value & IMPORT_BY_ORDINAL == 0 {
                return Err(Error::corrupt("kernel import is not by ordinal"));
            }
            ordinals.push(value & !IMPORT_BY_ORDINAL);
        }
        Err(Error::corrupt("kernel thunk table is not terminated"))
    }
}

/// The header block, addressed by virtual address.
struct Headers<'a> {
    bytes: &'a [u8],
    base: u32,
}

impl<'a> Headers<'a> {
    fn slice(&self, address: u32, len: usize) -> Result<&'a [u8], Error> {
        let start = address
            .checked_sub(self.base)
            .ok_or_else(|| Error::corrupt("header pointer below base address"))?
            as usize;
        self.bytes
            .get(start..start.saturating_add(len))
            .ok_or_else(|| Error::corrupt("header pointer outside header block"))
    }

    fn c_string(&self, address: u32) -> Result<String, Error> {
        let start = address
            .checked_sub(self.base)
            .map(|delta| delta as usize)
            .filter(|&start| start < self.bytes.len())
            .ok_or_else(|| Error::corrupt("string pointer outside header block"))?;
        Ok(latin1_until_nul(&self.bytes[start..]))
    }
}

fn decode_keys(
    base: u32,
    image_size: u32,
    entry: u32,
    thunk: u32,
) -> Result<(BuildKind, u32, u32), Error> {
    let in_image = |address: u32| address.wrapping_sub(base) < image_size;
    [
        (BuildKind::Retail, RETAIL_ENTRY_KEY, RETAIL_THUNK_KEY),
        (BuildKind::Debug, DEBUG_ENTRY_KEY, DEBUG_THUNK_KEY),
    ]
    .into_iter()
    .map(|(kind, entry_key, thunk_key)| (kind, entry ^ entry_key, thunk ^ thunk_key))
    .find(|&(_, entry, thunk)| in_image(entry) && in_image(thunk))
    .ok_or_else(Error::unknown_keys)
}

fn table_entry(table: u32, index: u32, entry_len: u32) -> Result<u32, Error> {
    index
        .checked_mul(entry_len)
        .and_then(|offset| table.checked_add(offset))
        .ok_or_else(|| Error::corrupt("table address overflows"))
}

fn parse_certificate(headers: &Headers<'_>, address: u32) -> Result<Certificate, Error> {
    let raw = headers.slice(address, CERTIFICATE_LEN)?;
    let units: Vec<u16> = raw[0x0C..0x0C + TITLE_NAME_UNITS * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .take_while(|&unit| unit != 0)
        .collect();
    Ok(Certificate {
        title_id: le_u32(raw, 0x08),
        title_name: String::from_utf16_lossy(&units),
        game_region: le_u32(raw, 0xA0),
        version: le_u32(raw, 0xAC),
    })
}

fn parse_section(headers: &Headers<'_>, address: u32, file_len: usize) -> Result<Section, Error> {
    let raw = headers.slice(address, SECTION_HEADER_LEN as usize)?;
    let raw_address = le_u32(raw, 12);
    let raw_size = le_u32(raw, 16);
    if u64::from(raw_address) + u64::from(raw_size) > file_len as u64 {
        return Err(Error::corrupt("section raw data past end of file"));
    }
    Ok(Section {
        name: headers.c_string(le_u32(raw, 20))?,
        flags: SectionFlags(le_u32(raw, 0)),
        virtual_address: le_u32(raw, 4),
        virtual_size: le_u32(raw, 8),
        raw_address,
        raw_size,
    })
}

fn parse_library(headers: &Headers<'_>, address: u32) -> Result<LibraryVersion, Error> {
    let raw = headers.slice(address, LIBRARY_VERSION_LEN as usize)?;
    Ok(LibraryVersion {
        name: latin1_until_nul(&raw[..8]),
        major: le_u16(raw, 8),
        minor: le_u16(raw, 10),
        build: le_u16(raw, 12),
        flags: le_u16(raw, 14),
    })
}

fn latin1_until_nul(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take_while(|&&b| b != 0)
        .copied()
        .map(char::from)
        .collect()
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
