//! The file system (spec 8.9): paths, symbolic links, the disc, the hard disk's partitions as
//! in-memory file systems or raw block devices, and the save file that keeps `E:`.

use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use super::status;

/// The disc's geometry: units of one 2048-byte sector.
pub const DISC_UNITS: u64 = 2_295_104;
/// A partition file system's geometry: units of 32 sectors of 512 bytes, 1 GiB in all. Titles
/// work out the free space in 32-bit arithmetic, so its bytes must fit a signed 32-bit value:
/// with 16 GiB, the low half of the free bytes was 0 on an empty drive and negative once
/// anything was saved, and Dragon's Lair 3D saw a negative number of free blocks.
pub const FS_UNITS: u64 = 0x1_0000;
pub const FS_UNIT_BYTES: u64 = 32 * 512;
/// The raw disk's size in 512-byte sectors (a 10 GB drive).
pub const RAW_DISK_SECTORS: u64 = 0x0140_0000;

/// The retail partitions: start and size in sectors (index = partition number).
pub const PARTITIONS: [(u64, u64); 6] = [
    (0, RAW_DISK_SECTORS),
    (0x0055_F400, 0x0098_96B0),
    (0x0046_5400, 0x000F_A000),
    (0x400, 0x0017_7000),
    (0x400 + 0x0017_7000, 0x0017_7000),
    (0x400 + 2 * 0x0017_7000, 0x0017_7000),
];

/// The save file's magic.
const SAVE_MAGIC: &[u8; 12] = b"ANYXBOXSAVE1";

/// File attributes.
pub const ATTR_READONLY: u32 = 0x01;
pub const ATTR_DIRECTORY: u32 = 0x10;
pub const ATTR_NORMAL: u32 = 0x80;

/// `NtCreateFile` create options.
pub const FILE_DIRECTORY_FILE: u32 = 0x01;
pub const FILE_NON_DIRECTORY_FILE: u32 = 0x40;

/// The `information` an open reports.
pub const FILE_SUPERSEDED: u32 = 0;
pub const FILE_OPENED: u32 = 1;
pub const FILE_CREATED: u32 = 2;
pub const FILE_OVERWRITTEN: u32 = 3;

/// One node of an in-memory file system.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    name: String,
    parent: usize,
    is_dir: bool,
    data: Vec<u8>,
    /// The children by lower-case name (sorted listing).
    children: BTreeMap<String, usize>,
    alive: bool,
}

/// An in-memory, case-insensitive file system (a hard-disk partition).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemFs {
    /// Node 0 is the root; the order is creation order.
    nodes: Vec<Node>,
}

impl Default for MemFs {
    fn default() -> Self {
        Self::new()
    }
}

impl MemFs {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                name: String::new(),
                parent: 0,
                is_dir: true,
                data: Vec::new(),
                children: BTreeMap::new(),
                alive: true,
            }],
        }
    }

    /// The node at a path of components.
    fn lookup(&self, parts: &[&str]) -> Option<usize> {
        let mut at = 0;
        for part in parts {
            if !self.nodes[at].is_dir {
                return None;
            }
            at = *self.nodes[at].children.get(&part.to_lowercase())?;
        }
        Some(at)
    }

    /// Creates a node under `parent`.
    fn create(&mut self, parent: usize, name: &str, is_dir: bool) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            name: name.to_owned(),
            parent,
            is_dir,
            data: Vec::new(),
            children: BTreeMap::new(),
            alive: true,
        });
        self.nodes[parent].children.insert(name.to_lowercase(), id);
        id
    }

    /// Unlinks a node; open handles keep it.
    fn remove(&mut self, id: usize) {
        let node = &self.nodes[id];
        let (parent, key) = (node.parent, node.name.to_lowercase());
        self.nodes[parent].children.remove(&key);
        self.nodes[id].alive = false;
    }

    /// The units in use: one per directory and per empty file, else one per 16 KiB.
    fn used_units(&self) -> u64 {
        self.nodes
            .iter()
            .skip(1)
            .filter(|n| n.alive)
            .map(|n| (n.data.len() as u64).div_ceil(FS_UNIT_BYTES).max(1))
            .sum()
    }

    /// Writes the save format: the nodes after the root in creation order.
    #[must_use]
    pub fn to_save(&self) -> Vec<u8> {
        let live: Vec<usize> = (1..self.nodes.len())
            .filter(|&i| self.reachable(i))
            .collect();
        let index: HashMap<usize, u32> = live
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, u32::try_from(i + 1).unwrap_or(0)))
            .collect();
        let mut out = SAVE_MAGIC.to_vec();
        out.extend_from_slice(&u32::try_from(live.len()).unwrap_or(0).to_le_bytes());
        for &id in &live {
            let node = &self.nodes[id];
            let parent = if node.parent == 0 {
                0
            } else {
                index[&node.parent]
            };
            out.extend_from_slice(&parent.to_le_bytes());
            out.push(u8::from(node.is_dir));
            let name = node.name.as_bytes();
            out.extend_from_slice(&u16::try_from(name.len()).unwrap_or(0).to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(&(node.data.len() as u64).to_le_bytes());
            out.extend_from_slice(&node.data);
        }
        out
    }

    /// Whether a node is alive and every directory above it is.
    fn reachable(&self, mut id: usize) -> bool {
        while id != 0 {
            if !self.nodes[id].alive {
                return false;
            }
            id = self.nodes[id].parent;
        }
        true
    }

    /// Reads the save format.
    ///
    /// # Errors
    ///
    /// A message when the bytes do not start with the magic, are cut short, or name a parent
    /// that is not an earlier directory.
    pub fn from_save(bytes: &[u8]) -> Result<Self, String> {
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let slice = bytes.get(at..at + n).ok_or("the save file is cut short")?;
            at += n;
            Ok(slice)
        };
        if take(12)? != SAVE_MAGIC {
            return Err("the save file does not start with its magic".to_owned());
        }
        let count = u32::from_le_bytes(take(4)?.try_into().unwrap_or_default());
        let mut fs = Self::new();
        // Save index (1-based) to node id; 0 is the root.
        let mut ids = vec![0usize];
        for _ in 0..count {
            let parent = u32::from_le_bytes(take(4)?.try_into().unwrap_or_default()) as usize;
            let is_dir = take(1)?[0] != 0;
            let name_len = u16::from_le_bytes(take(2)?.try_into().unwrap_or_default());
            let name = String::from_utf8_lossy(take(usize::from(name_len))?).into_owned();
            let data_len = u64::from_le_bytes(take(8)?.try_into().unwrap_or_default());
            let data =
                take(usize::try_from(data_len).map_err(|_| "a file is too large")?)?.to_vec();
            let parent = *ids
                .get(parent)
                .filter(|&&p| fs.nodes[p].is_dir)
                .ok_or("the save file names a missing parent directory")?;
            let id = fs.create(parent, &name, is_dir);
            fs.nodes[id].data = data;
            ids.push(id);
        }
        Ok(fs)
    }
}

/// A sparse block device of 512-byte sectors, zero where never written.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RawDevice {
    #[serde(with = "crate::hibernate::serde_util::sectors")]
    sectors: HashMap<u64, Box<[u8; 512]>>,
}

/// Where an open file lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Location {
    /// The disc device itself.
    DiscDevice,
    /// A file or directory on the disc.
    Disc(pack::NodeId),
    /// A node of partition `part`'s file system.
    Fs { part: usize, node: usize },
    /// The raw disk (0) or a partition's raw block device.
    Raw(usize),
}

/// A directory listing fixed at the first query (spec 8.9).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Listing {
    pub entries: Vec<Entry>,
    pub next: usize,
}

/// One directory entry, or what a query about a file reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub attributes: u32,
}

impl Entry {
    /// The allocation size: the size rounded up to 2048 bytes (spec 8.9).
    #[must_use]
    pub fn allocation(&self) -> u64 {
        self.size.next_multiple_of(2048)
    }
}

/// An open file.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenFile {
    pub location: Location,
    pub position: u64,
    pub listing: Option<Listing>,
    /// The device path it was opened by, for reports.
    pub path: String,
}

/// A resolved path.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    DiscDevice,
    Disc(Vec<String>),
    Fs(usize, Vec<String>),
    Raw(usize),
}

/// The file system.
#[derive(Debug, Serialize, Deserialize)]
pub struct FileSystem {
    /// Symbolic links by lower-case name: (name as given, target).
    links: BTreeMap<String, (String, String)>,
    #[serde(skip)]
    disc: Option<pack::Pack>,
    /// Partitions 1 to 5 (index 0 unused).
    partitions: Vec<MemFs>,
    /// The raw disk and the partitions' raw devices.
    raw: Vec<RawDevice>,
    /// Where `E:` is kept, when saves are kept.
    #[serde(skip)]
    save_path: Option<PathBuf>,
    /// Whether partition 1 changed since it was last written.
    dirty: bool,
}

impl FileSystem {
    /// Takes `saved`'s files and devices over, keeping this boot's disc and save file.
    pub fn adopt(&mut self, saved: Self) {
        let disc = self.disc.take();
        let save_path = self.save_path.take();
        *self = saved;
        self.disc = disc;
        self.save_path = save_path;
    }
}

impl Default for FileSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalises a guest path: a leading `\`, single separators, `X:...` as `\??\X:...`. Returns
/// the path without a trailing separator, and whether it had one.
#[must_use]
pub fn normalize(path: &str) -> (String, bool) {
    let path = path.replace('/', "\\");
    let path = if path.as_bytes().get(1) == Some(&b':') {
        format!("\\??\\{path}")
    } else {
        path
    };
    let trailing = path.len() > 1 && path.ends_with('\\');
    let parts: Vec<&str> = path.split('\\').filter(|p| !p.is_empty()).collect();
    (format!("\\{}", parts.join("\\")), trailing)
}

/// The components of a path below a device.
fn components(rest: &str) -> Vec<String> {
    rest.split('\\')
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Whether `path` starts with `prefix` at a separator boundary (case-insensitive). Returns the
/// rest.
fn strip_prefix_ci<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    let head = path.get(..prefix.len())?;
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let rest = &path[prefix.len()..];
    (rest.is_empty() || rest.starts_with('\\')).then_some(rest)
}

/// DOS wildcard match: `*` and `?`, case-insensitive; `*.*` is `*`.
#[must_use]
pub fn wildcard(pattern: &str, name: &str) -> bool {
    fn matches(p: &[u8], n: &[u8]) -> bool {
        match (p.first(), n.first()) {
            (None, None) => true,
            (Some(b'*'), _) => matches(&p[1..], n) || (!n.is_empty() && matches(p, &n[1..])),
            (Some(b'?'), Some(_)) => matches(&p[1..], &n[1..]),
            (Some(a), Some(b)) => a.eq_ignore_ascii_case(b) && matches(&p[1..], &n[1..]),
            _ => false,
        }
    }
    let pattern = if pattern == "*.*" { "*" } else { pattern };
    matches(pattern.as_bytes(), name.as_bytes())
}

impl FileSystem {
    /// The initial file system: `D:` and `E:` linked, empty partitions, no disc.
    #[must_use]
    pub fn new() -> Self {
        let mut fs = Self {
            links: BTreeMap::new(),
            disc: None,
            partitions: (0..6).map(|_| MemFs::new()).collect(),
            raw: (0..6).map(|_| RawDevice::default()).collect(),
            save_path: None,
            dirty: false,
        };
        let _ = fs.add_link("\\??\\D:", "\\Device\\CdRom0");
        let _ = fs.add_link("\\??\\E:", "\\Device\\Harddisk0\\Partition1");
        fs
    }

    /// Mounts the disc.
    pub fn mount_disc(&mut self, pack: pack::Pack) {
        self.disc = Some(pack);
    }

    /// Keeps `E:` in the file at `path`, loading it when it exists.
    ///
    /// # Errors
    ///
    /// A message when the file exists but cannot be read or is not a valid save; then saves are
    /// not kept.
    pub fn keep_saves(&mut self, path: &Path) -> Result<(), String> {
        match std::fs::read(path) {
            Ok(bytes) => {
                self.partitions[1] = MemFs::from_save(&bytes)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        self.save_path = Some(path.to_owned());
        Ok(())
    }

    /// Writes `E:` to the save file if it changed, through a partial file renamed over it.
    pub fn persist(&mut self) {
        if !self.dirty {
            return;
        }
        let Some(path) = &self.save_path else {
            return;
        };
        let mut partial = path.clone().into_os_string();
        partial.push("-partial");
        let partial = PathBuf::from(partial);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let written = std::fs::write(&partial, self.partitions[1].to_save())
            .and_then(|()| std::fs::rename(&partial, path));
        match written {
            Ok(()) => self.dirty = false,
            Err(error) => eprintln!(
                "anyxbox: could not write the saves to {}: {error}",
                path.display()
            ),
        }
    }

    /// Adds a symbolic link; `OBJECT_NAME_COLLISION` when the name is taken.
    pub fn add_link(&mut self, name: &str, target: &str) -> u32 {
        let (name, _) = normalize(name);
        let key = name.to_lowercase();
        if self.links.contains_key(&key) {
            return status::OBJECT_NAME_COLLISION;
        }
        let (target, _) = normalize(target);
        self.links.insert(key, (name, target));
        status::SUCCESS
    }

    /// Removes a symbolic link; `OBJECT_NAME_NOT_FOUND` when absent.
    pub fn delete_link(&mut self, name: &str) -> u32 {
        let (name, _) = normalize(name);
        if self.links.remove(&name.to_lowercase()).is_some() {
            status::SUCCESS
        } else {
            status::OBJECT_NAME_NOT_FOUND
        }
    }

    /// The target of a symbolic link.
    #[must_use]
    pub fn link_target(&self, name: &str) -> Option<&str> {
        let (name, _) = normalize(name);
        self.links
            .get(&name.to_lowercase())
            .map(|(_, t)| t.as_str())
    }

    /// The link names, as given.
    pub fn link_names(&self) -> impl Iterator<Item = &str> {
        self.links.values().map(|(n, _)| n.as_str())
    }

    /// Follows symbolic links (longest prefix at a separator boundary, at most 16 times).
    #[must_use]
    pub fn follow_links(&self, path: &str) -> String {
        let mut path = path.to_owned();
        for _ in 0..16 {
            let best = self
                .links
                .values()
                .filter_map(|(name, target)| {
                    strip_prefix_ci(&path, name).map(|rest| (name.len(), target, rest))
                })
                .max_by_key(|(len, ..)| *len);
            let Some((_, target, rest)) = best else {
                break;
            };
            path = format!("{target}{rest}");
        }
        path
    }

    /// Resolves a guest path to a volume and a path in it.
    fn resolve(&self, path: &str) -> Result<Target, u32> {
        let (normal, trailing) = normalize(path);
        let device = self.follow_links(&normal);
        if let Some(rest) = strip_prefix_ci(&device, "\\Device\\CdRom0") {
            if rest.is_empty() && !trailing {
                return Ok(Target::DiscDevice);
            }
            return Ok(Target::Disc(components(rest)));
        }
        let Some(rest) = strip_prefix_ci(&device, "\\Device\\Harddisk0") else {
            return Err(status::OBJECT_NAME_NOT_FOUND);
        };
        let mut parts = rest.splitn(3, '\\').skip(1);
        let partition = parts.next().unwrap_or("");
        let number = partition
            .get(..9)
            .filter(|p| p.eq_ignore_ascii_case("partition"))
            .and_then(|_| partition[9..].parse::<usize>().ok())
            .filter(|&n| n < PARTITIONS.len())
            .ok_or(status::OBJECT_NAME_NOT_FOUND)?;
        let below = parts.next();
        match below {
            None if !trailing => Ok(Target::Raw(number)),
            _ if number == 0 => Err(status::OBJECT_NAME_NOT_FOUND),
            _ => Ok(Target::Fs(number, components(below.unwrap_or("")))),
        }
    }

    /// Opens or creates a file (`NtCreateFile`, spec 8.9). Returns the open file and the
    /// `information` value (`FILE_OPENED` and so on).
    ///
    /// # Errors
    ///
    /// The NTSTATUS the open fails with.
    pub fn open(
        &mut self,
        path: &str,
        disposition: u32,
        options: u32,
    ) -> Result<(OpenFile, u32), u32> {
        let target = self.resolve(path)?;
        let opened = |location| OpenFile {
            location,
            position: 0,
            listing: None,
            path: path.to_owned(),
        };
        let check_kind = |is_dir: bool| -> Result<(), u32> {
            if is_dir && options & FILE_NON_DIRECTORY_FILE != 0 {
                Err(status::FILE_IS_A_DIRECTORY)
            } else if !is_dir && options & FILE_DIRECTORY_FILE != 0 {
                Err(status::NOT_A_DIRECTORY)
            } else {
                Ok(())
            }
        };
        match target {
            Target::DiscDevice => Ok((opened(Location::DiscDevice), FILE_OPENED)),
            Target::Raw(n) => Ok((opened(Location::Raw(n)), FILE_OPENED)),
            Target::Disc(parts) => {
                let disc = self.disc.as_ref().ok_or(status::OBJECT_NAME_NOT_FOUND)?;
                let Some(node) = disc.lookup(&parts.join("\\")) else {
                    let parent = disc.lookup(&parts[..parts.len().saturating_sub(1)].join("\\"));
                    return Err(if parent.is_some() {
                        if matches!(disposition, 1 | 4) {
                            status::OBJECT_NAME_NOT_FOUND
                        } else {
                            status::MEDIA_WRITE_PROTECTED
                        }
                    } else {
                        status::OBJECT_PATH_NOT_FOUND
                    });
                };
                if !matches!(disposition, 1 | 3) {
                    return Err(if disposition == 2 {
                        status::OBJECT_NAME_COLLISION
                    } else {
                        status::MEDIA_WRITE_PROTECTED
                    });
                }
                check_kind(disc.info(node).is_dir)?;
                Ok((opened(Location::Disc(node)), FILE_OPENED))
            }
            Target::Fs(part, parts) => {
                let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
                let fs = &mut self.partitions[part];
                if let Some(node) = fs.lookup(&refs) {
                    let is_dir = fs.nodes[node].is_dir;
                    check_kind(is_dir)?;
                    let information = match disposition {
                        2 => return Err(status::OBJECT_NAME_COLLISION),
                        1 | 3 => FILE_OPENED,
                        _ if is_dir => return Err(status::FILE_IS_A_DIRECTORY),
                        0 => FILE_SUPERSEDED,
                        _ => FILE_OVERWRITTEN,
                    };
                    if information != FILE_OPENED {
                        fs.nodes[node].data.clear();
                        self.dirty |= part == 1;
                    }
                    return Ok((opened(Location::Fs { part, node }), information));
                }
                if matches!(disposition, 1 | 4) {
                    return Err(status::OBJECT_NAME_NOT_FOUND);
                }
                let (name, parent) = refs.split_last().ok_or(status::OBJECT_NAME_NOT_FOUND)?;
                let parent = fs
                    .lookup(parent)
                    .filter(|&p| fs.nodes[p].is_dir)
                    .ok_or(status::OBJECT_PATH_NOT_FOUND)?;
                let node = fs.create(parent, name, options & FILE_DIRECTORY_FILE != 0);
                self.dirty |= part == 1;
                Ok((opened(Location::Fs { part, node }), FILE_CREATED))
            }
        }
    }

    /// What a query about an open file reports.
    #[must_use]
    pub fn stat(&self, location: Location) -> Entry {
        match location {
            Location::DiscDevice => Entry {
                name: String::new(),
                is_dir: false,
                size: DISC_UNITS * 2048,
                attributes: ATTR_READONLY,
            },
            Location::Disc(node) => {
                let info = self.disc.as_ref().map(|d| d.info(node));
                let (name, is_dir, size) =
                    info.map_or((String::new(), false, 0), |i| (i.name, i.is_dir, i.size));
                Entry {
                    name,
                    is_dir,
                    size,
                    attributes: ATTR_READONLY | if is_dir { ATTR_DIRECTORY } else { 0 },
                }
            }
            Location::Fs { part, node } => {
                let node = &self.partitions[part].nodes[node];
                Entry {
                    name: node.name.clone(),
                    is_dir: node.is_dir,
                    size: node.data.len() as u64,
                    attributes: if node.is_dir {
                        ATTR_DIRECTORY
                    } else {
                        ATTR_NORMAL
                    },
                }
            }
            Location::Raw(n) => Entry {
                name: String::new(),
                is_dir: false,
                size: PARTITIONS[n].1 * 512,
                attributes: ATTR_NORMAL,
            },
        }
    }

    /// Reads up to `len` bytes at `offset`; reads stop at the end of the file.
    ///
    /// # Errors
    ///
    /// `INVALID_DEVICE_REQUEST` for a directory or the disc device, or a damaged disc.
    pub fn read(&self, location: Location, offset: u64, len: u32) -> Result<Vec<u8>, u32> {
        let size = self.stat(location).size;
        let len = usize::try_from(u64::from(len).min(size.saturating_sub(offset))).unwrap_or(0);
        let mut out = vec![0u8; len];
        match location {
            Location::Disc(node) => {
                let disc = self.disc.as_ref().ok_or(status::INVALID_DEVICE_REQUEST)?;
                let mut done = 0;
                while done < len {
                    let n = disc
                        .read_at(node, offset + done as u64, &mut out[done..])
                        .map_err(|_| status::INVALID_DEVICE_REQUEST)?;
                    if n == 0 {
                        break;
                    }
                    done += n;
                }
                out.truncate(done);
            }
            Location::Fs { part, node } => {
                let node = &self.partitions[part].nodes[node];
                if node.is_dir {
                    return Err(status::INVALID_DEVICE_REQUEST);
                }
                let start = usize::try_from(offset)
                    .unwrap_or(usize::MAX)
                    .min(node.data.len());
                out.copy_from_slice(&node.data[start..start + len]);
            }
            Location::Raw(n) => {
                for (i, byte) in out.iter_mut().enumerate() {
                    let at = offset + i as u64;
                    if let Some(sector) = self.raw[n].sectors.get(&(at / 512)) {
                        *byte = sector[(at % 512) as usize];
                    }
                }
            }
            Location::DiscDevice => return Err(status::INVALID_DEVICE_REQUEST),
        }
        Ok(out)
    }

    /// Writes `data` at `offset`. Returns the bytes written.
    ///
    /// # Errors
    ///
    /// `MEDIA_WRITE_PROTECTED` on the disc, `INVALID_DEVICE_REQUEST` for a directory, and
    /// `END_OF_FILE` for a raw write past the device's end.
    pub fn write(&mut self, location: Location, offset: u64, data: &[u8]) -> Result<usize, u32> {
        match location {
            Location::Disc(_) | Location::DiscDevice => Err(status::MEDIA_WRITE_PROTECTED),
            Location::Fs { part, node } => {
                let node = &mut self.partitions[part].nodes[node];
                if node.is_dir {
                    return Err(status::INVALID_DEVICE_REQUEST);
                }
                let start = usize::try_from(offset).map_err(|_| status::INVALID_PARAMETER)?;
                let end = start + data.len();
                if node.data.len() < end {
                    node.data.resize(end, 0);
                }
                node.data[start..end].copy_from_slice(data);
                self.dirty |= part == 1;
                Ok(data.len())
            }
            Location::Raw(n) => {
                let end = offset + data.len() as u64;
                if end > PARTITIONS[n].1 * 512 {
                    return Err(status::END_OF_FILE);
                }
                for (i, &byte) in data.iter().enumerate() {
                    let at = offset + i as u64;
                    let sector = self.raw[n]
                        .sectors
                        .entry(at / 512)
                        .or_insert_with(|| Box::new([0; 512]));
                    sector[(at % 512) as usize] = byte;
                }
                Ok(data.len())
            }
        }
    }

    /// Resizes a file (`FileEndOfFileInformation`).
    pub fn set_size(&mut self, location: Location, size: u64) -> u32 {
        let Location::Fs { part, node } = location else {
            return status::INVALID_PARAMETER;
        };
        let Ok(size) = usize::try_from(size) else {
            return status::INVALID_PARAMETER;
        };
        self.partitions[part].nodes[node].data.resize(size, 0);
        self.dirty |= part == 1;
        status::SUCCESS
    }

    /// Deletes an open file or an empty directory.
    pub fn delete(&mut self, location: Location) -> u32 {
        let Location::Fs { part, node } = location else {
            return status::MEDIA_WRITE_PROTECTED;
        };
        let fs = &mut self.partitions[part];
        if node == 0 {
            return status::ACCESS_DENIED;
        }
        if !fs.nodes[node].children.is_empty() {
            return status::DIRECTORY_NOT_EMPTY;
        }
        fs.remove(node);
        self.dirty |= part == 1;
        status::SUCCESS
    }

    /// Renames an open file to `path` (on the same volume).
    pub fn rename(&mut self, location: Location, path: &str, replace: bool) -> u32 {
        let Location::Fs { part, node } = location else {
            return status::MEDIA_WRITE_PROTECTED;
        };
        let Ok(Target::Fs(to_part, parts)) = self.resolve(path) else {
            return status::NOT_SAME_DEVICE;
        };
        if to_part != part {
            return status::NOT_SAME_DEVICE;
        }
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        let Some((name, parent)) = refs.split_last() else {
            return status::INVALID_PARAMETER;
        };
        let fs = &mut self.partitions[part];
        let Some(parent) = fs.lookup(parent).filter(|&p| fs.nodes[p].is_dir) else {
            return status::OBJECT_PATH_NOT_FOUND;
        };
        if let Some(existing) = fs.nodes[parent].children.get(&name.to_lowercase()).copied()
            && existing != node
        {
            if !replace || fs.nodes[existing].is_dir {
                return status::OBJECT_NAME_COLLISION;
            }
            fs.remove(existing);
        }
        fs.remove(node);
        fs.nodes[node].alive = true;
        (*name).clone_into(&mut fs.nodes[node].name);
        fs.nodes[node].parent = parent;
        fs.nodes[parent].children.insert(name.to_lowercase(), node);
        self.dirty |= part == 1;
        status::SUCCESS
    }

    /// The entries of a directory, sorted case-insensitively.
    ///
    /// # Errors
    ///
    /// `INVALID_PARAMETER` when it is not a directory.
    pub fn list(&self, location: Location) -> Result<Vec<Entry>, u32> {
        match location {
            Location::Disc(node) => {
                let disc = self.disc.as_ref().ok_or(status::INVALID_PARAMETER)?;
                if !disc.info(node).is_dir {
                    return Err(status::INVALID_PARAMETER);
                }
                Ok(disc
                    .children(node)
                    .into_iter()
                    .map(|c| self.stat(Location::Disc(c)))
                    .collect())
            }
            Location::Fs { part, node } => {
                let fs = &self.partitions[part];
                if !fs.nodes[node].is_dir {
                    return Err(status::INVALID_PARAMETER);
                }
                Ok(fs.nodes[node]
                    .children
                    .values()
                    .map(|&c| self.stat(Location::Fs { part, node: c }))
                    .collect())
            }
            _ => Err(status::INVALID_PARAMETER),
        }
    }

    /// The volume's size: total units, free units, sectors per unit, bytes per sector.
    #[must_use]
    pub fn volume_size(&self, location: Location) -> (u64, u64, u32, u32) {
        match location {
            Location::DiscDevice | Location::Disc(_) => (DISC_UNITS, 0, 1, 2048),
            Location::Fs { part, .. } => {
                let used = self.partitions[part].used_units();
                (FS_UNITS, FS_UNITS.saturating_sub(used), 32, 512)
            }
            Location::Raw(n) => {
                let fs = &self.partitions[n.max(1)];
                (FS_UNITS, FS_UNITS.saturating_sub(fs.used_units()), 32, 512)
            }
        }
    }

    /// The device type and characteristics: the disc is a CD-ROM (2), removable and read-only
    /// (3); the hard disk is a disk (7).
    #[must_use]
    pub fn device_info(location: Location) -> (u32, u32) {
        match location {
            Location::DiscDevice | Location::Disc(_) => (2, 3),
            Location::Fs { .. } | Location::Raw(_) => (7, 0),
        }
    }

    /// Whether `location` is on the disc.
    #[must_use]
    pub fn is_disc(location: Location) -> bool {
        matches!(location, Location::DiscDevice | Location::Disc(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partitions_size_and_free_space_fit_a_titles_32_bit_arithmetic() {
        let mut fs = FileSystem::new();
        let empty = fs.volume_size(Location::Fs { part: 1, node: 0 });
        fs.open("E:\\UDATA", 2, FILE_DIRECTORY_FILE).unwrap();
        let (f, _) = fs.open("E:\\UDATA\\save.bin", 2, 0).unwrap();
        fs.write(f.location, 0, &vec![7; 40_000]).unwrap();
        let used = fs.volume_size(Location::Fs { part: 1, node: 0 });
        for (total, free, sectors, bytes) in [empty, used] {
            let unit = u64::from(sectors) * u64::from(bytes);
            assert!(
                i32::try_from(total * unit).is_ok(),
                "total bytes {}",
                total * unit
            );
            assert!(free * unit <= total * unit);
        }
        assert!(used.1 < empty.1, "a save takes free space");
    }

    #[test]
    fn paths_are_normalised() {
        assert_eq!(
            normalize("D:\\a\\\\b\\"),
            ("\\??\\D:\\a\\b".to_owned(), true)
        );
        assert_eq!(
            normalize("\\Device\\CdRom0"),
            ("\\Device\\CdRom0".to_owned(), false)
        );
        assert_eq!(normalize("a/b"), ("\\a\\b".to_owned(), false));
    }

    #[test]
    fn links_follow_the_longest_prefix_at_a_separator() {
        let mut fs = FileSystem::new();
        assert_eq!(
            fs.add_link("\\??\\T:", "\\Device\\Harddisk0\\Partition1\\TDATA"),
            0
        );
        assert_eq!(fs.add_link("\\??\\t:", "x"), status::OBJECT_NAME_COLLISION);
        assert_eq!(
            fs.follow_links("\\??\\T:\\save"),
            "\\Device\\Harddisk0\\Partition1\\TDATA\\save"
        );
        assert_eq!(fs.follow_links("\\??\\TX:\\save"), "\\??\\TX:\\save");
        assert_eq!(fs.follow_links("\\??\\d:\\a"), "\\Device\\CdRom0\\a");
        assert_eq!(fs.delete_link("\\??\\T:"), 0);
        assert_eq!(fs.delete_link("\\??\\T:"), status::OBJECT_NAME_NOT_FOUND);
    }

    #[test]
    fn a_device_path_without_a_separator_is_the_device() {
        let fs = FileSystem::new();
        assert_eq!(
            fs.resolve("\\Device\\Harddisk0\\Partition5"),
            Ok(Target::Raw(5))
        );
        assert_eq!(
            fs.resolve("\\Device\\Harddisk0\\partition5\\"),
            Ok(Target::Fs(5, vec![]))
        );
        assert_eq!(
            fs.resolve("\\Device\\Harddisk0\\Partition0"),
            Ok(Target::Raw(0))
        );
        assert_eq!(fs.resolve("\\Device\\CdRom0"), Ok(Target::DiscDevice));
        assert_eq!(fs.resolve("\\Device\\CdRom0\\"), Ok(Target::Disc(vec![])));
        assert_eq!(
            fs.resolve("E:\\a\\b"),
            Ok(Target::Fs(1, vec!["a".into(), "b".into()]))
        );
        assert_eq!(
            fs.resolve("\\Device\\Harddisk0\\Partition9\\"),
            Err(status::OBJECT_NAME_NOT_FOUND)
        );
        assert_eq!(
            fs.resolve("\\Device\\Nothing"),
            Err(status::OBJECT_NAME_NOT_FOUND)
        );
    }

    #[test]
    fn every_disposition() {
        let mut fs = FileSystem::new();
        let open = |fs: &mut FileSystem, d| fs.open("E:\\f", d, 0).map(|(_, i)| i);
        assert_eq!(open(&mut fs, 1), Err(status::OBJECT_NAME_NOT_FOUND));
        assert_eq!(open(&mut fs, 4), Err(status::OBJECT_NAME_NOT_FOUND));
        assert_eq!(open(&mut fs, 2), Ok(FILE_CREATED));
        assert_eq!(open(&mut fs, 2), Err(status::OBJECT_NAME_COLLISION));
        assert_eq!(open(&mut fs, 1), Ok(FILE_OPENED));
        assert_eq!(open(&mut fs, 3), Ok(FILE_OPENED));
        assert_eq!(open(&mut fs, 0), Ok(FILE_SUPERSEDED));
        assert_eq!(open(&mut fs, 4), Ok(FILE_OVERWRITTEN));
        assert_eq!(open(&mut fs, 5), Ok(FILE_OVERWRITTEN));
        assert_eq!(fs.open("E:\\g", 5, 0).map(|(_, i)| i), Ok(FILE_CREATED));
        assert_eq!(
            fs.open("E:\\no\\g", 3, 0).map(|(_, i)| i),
            Err(status::OBJECT_PATH_NOT_FOUND)
        );
        assert_eq!(
            fs.open("E:\\f", 1, FILE_DIRECTORY_FILE).map(|(_, i)| i),
            Err(status::NOT_A_DIRECTORY)
        );
        fs.open("E:\\d", 3, FILE_DIRECTORY_FILE).unwrap();
        assert_eq!(
            fs.open("E:\\d", 1, FILE_NON_DIRECTORY_FILE).map(|(_, i)| i),
            Err(status::FILE_IS_A_DIRECTORY)
        );
    }

    #[test]
    fn files_read_and_write_and_overwrite_truncates() {
        let mut fs = FileSystem::new();
        let (f, _) = fs.open("E:\\f", 2, 0).unwrap();
        assert_eq!(fs.write(f.location, 2, b"abc"), Ok(3));
        assert_eq!(fs.read(f.location, 0, 100), Ok(b"\0\0abc".to_vec()));
        assert_eq!(fs.read(f.location, 5, 100), Ok(vec![]));
        let (f, _) = fs.open("E:\\F", 5, 0).unwrap();
        assert_eq!(fs.stat(f.location).size, 0);
    }

    #[test]
    fn raw_devices_are_sparse_and_bounded() {
        let mut fs = FileSystem::new();
        let (f, _) = fs.open("\\Device\\Harddisk0\\Partition5", 1, 0).unwrap();
        assert_eq!(fs.write(f.location, 510, &[1, 2, 3, 4]), Ok(4));
        assert_eq!(
            fs.read(f.location, 508, 8),
            Ok(vec![0, 0, 1, 2, 3, 4, 0, 0])
        );
        let end = PARTITIONS[5].1 * 512;
        assert_eq!(
            fs.write(f.location, end - 1, &[1, 2]),
            Err(status::END_OF_FILE)
        );
    }

    #[test]
    fn listings_are_sorted_case_insensitively_and_wildcards_match() {
        let mut fs = FileSystem::new();
        for name in ["b.txt", "A.dat", "c"] {
            fs.open(&format!("E:\\{name}"), 2, 0).unwrap();
        }
        let (root, _) = fs.open("E:\\", 1, 0).unwrap();
        let names: Vec<String> = fs
            .list(root.location)
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, ["A.dat", "b.txt", "c"]);
        assert!(wildcard("*.*", "c"));
        assert!(wildcard("*.TXT", "b.txt"));
        assert!(wildcard("?.dat", "A.dat"));
        assert!(!wildcard("*.dat", "b.txt"));
    }

    #[test]
    fn the_save_format_round_trips_and_rejects_bad_files() {
        let mut fs = FileSystem::new();
        fs.open("E:\\UDATA", 2, FILE_DIRECTORY_FILE).unwrap();
        let (f, _) = fs.open("E:\\UDATA\\save.bin", 2, 0).unwrap();
        fs.write(f.location, 0, b"hello").unwrap();
        let bytes = fs.partitions[1].to_save();
        let back = MemFs::from_save(&bytes).unwrap();
        assert_eq!(back.to_save(), bytes);
        let node = back.lookup(&["udata", "SAVE.BIN"]).unwrap();
        assert_eq!(back.nodes[node].data, b"hello");

        assert!(MemFs::from_save(b"NOTASAVEFILE\0\0\0\0").is_err());
        assert!(MemFs::from_save(&bytes[..bytes.len() - 1]).is_err());
        let mut orphan = SAVE_MAGIC.to_vec();
        orphan.extend_from_slice(&1u32.to_le_bytes());
        orphan.extend_from_slice(&5u32.to_le_bytes()); // a parent that does not exist
        orphan.extend_from_slice(&[0, 1, 0, b'x']);
        orphan.extend_from_slice(&0u64.to_le_bytes());
        assert!(MemFs::from_save(&orphan).is_err());
    }

    #[test]
    fn rename_and_delete() {
        let mut fs = FileSystem::new();
        let (f, _) = fs.open("E:\\a", 2, 0).unwrap();
        assert_eq!(fs.rename(f.location, "E:\\b", false), 0);
        assert!(fs.open("E:\\a", 1, 0).is_err());
        assert!(fs.open("E:\\b", 1, 0).is_ok());
        fs.open("E:\\d", 2, FILE_DIRECTORY_FILE).unwrap();
        let (d, _) = fs.open("E:\\d", 1, 0).unwrap();
        fs.open("E:\\d\\x", 2, 0).unwrap();
        assert_eq!(fs.delete(d.location), status::DIRECTORY_NOT_EMPTY);
        assert_eq!(
            fs.rename(f.location, "\\Device\\Harddisk0\\Partition2\\b", false),
            status::NOT_SAME_DEVICE
        );
    }
}
