//! A read-only archive of the files on a disc.
//!
//! The layout, all integers little endian:
//!
//! ```text
//! 0   8   the magic "ANYXPAK2"
//! 8   8   the offset of the index
//! 16  8   the length of the index
//! 24  ..  the chunks, back to back: a file is cut into CHUNK-byte pieces, and each is a zstd
//!         frame, or the bytes themselves when the frame is no shorter
//! index:  u64 chunk count, then per chunk: u64 offset, u32 stored length;
//!         u32 entry count, then per entry: u16 path length, the path (UTF-8, `\` between
//!         names), u8 1 for a directory else 0, u64 first chunk, u64 size
//! ```
//!
//! Compressing is slow (level 19, on every core) and decoding is fast: a read decodes only the
//! chunks it touches, and the most recent are kept.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt as _;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use xiso::XIso;

const MAGIC: &[u8; 8] = b"ANYXPAK2";
const HEADER_LEN: usize = 24;

/// The bytes of a file that are compressed together. A read decodes whole chunks, so this is the
/// most it decodes to return one byte; larger chunks would shrink the pack a little more.
pub const CHUNK: usize = 256 * 1024;

/// The zstd level `Writer::new` compresses at: slow to write, and no slower to read than level 1.
pub const DEFAULT_LEVEL: i32 = 19;

const CHUNK_BYTES: u64 = CHUNK as u64;

/// How many decoded chunks a `Pack` keeps.
const CACHED_CHUNKS: usize = 32;

fn invalid(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

/// The names in `path`, which separates them with `\` or `/`.
fn components(path: &str) -> impl Iterator<Item = &str> {
    path.split(['\\', '/']).filter(|part| !part.is_empty())
}

/// One file or directory to put in the index.
#[derive(Debug)]
struct Record {
    path: String,
    is_dir: bool,
    first_chunk: u64,
    size: u64,
}

/// Where a chunk is stored, and how long it is there. A chunk stored at its full length is raw.
#[derive(Debug, Clone, Copy)]
struct ChunkRef {
    offset: u64,
    stored: u32,
}

/// Reads until `buf` is full or `data` ends, and returns how many bytes came.
fn read_full(data: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match data.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

/// Writes a pack: the files in the order added, then the index.
#[derive(Debug)]
pub struct Writer<W: Write + Seek> {
    out: W,
    level: i32,
    records: Vec<Record>,
    chunks: Vec<ChunkRef>,
}

impl<W: Write + Seek> Writer<W> {
    /// Starts a pack in `out`, which must be empty.
    ///
    /// # Errors
    ///
    /// Returns an error when `out` cannot be written.
    pub fn new(out: W) -> io::Result<Self> {
        Self::with_level(out, DEFAULT_LEVEL)
    }

    /// Starts a pack in `out` that compresses at zstd `level`: higher is slower and smaller.
    ///
    /// # Errors
    ///
    /// Returns an error when `out` cannot be written.
    pub fn with_level(mut out: W, level: i32) -> io::Result<Self> {
        out.write_all(&[0; HEADER_LEN])?;
        Ok(Self {
            out,
            level,
            records: Vec::new(),
            chunks: Vec::new(),
        })
    }

    /// Adds a directory. Only an empty one needs this: a file's directories are implied.
    pub fn add_dir(&mut self, path: &str) {
        self.records.push(Record {
            path: path.to_owned(),
            is_dir: true,
            first_chunk: 0,
            size: 0,
        });
    }

    /// Adds a file, compressing `data` to the end of the pack. Returns the file's size.
    ///
    /// # Errors
    ///
    /// Returns an error when `data` cannot be read or the pack cannot be written.
    pub fn add_file(&mut self, path: &str, data: &mut impl Read) -> io::Result<u64> {
        let first_chunk = self.chunks.len() as u64;
        let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let level = self.level;
        let mut size = 0u64;
        let mut ended = false;
        loop {
            // One chunk per core, compressed side by side and written in order.
            let mut batch: Vec<Vec<u8>> = Vec::with_capacity(threads);
            while batch.len() < threads && !ended {
                let mut chunk = vec![0u8; CHUNK];
                let read = read_full(data, &mut chunk)?;
                chunk.truncate(read);
                ended = read < CHUNK;
                if read > 0 {
                    batch.push(chunk);
                }
            }
            if batch.is_empty() {
                break;
            }
            let packed: Vec<io::Result<Vec<u8>>> = std::thread::scope(|scope| {
                let handles: Vec<_> = batch
                    .iter()
                    .map(|chunk| scope.spawn(move || zstd::bulk::compress(chunk, level)))
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| {
                        handle.join().unwrap_or_else(|_| {
                            Err(io::Error::other("a compression thread panicked"))
                        })
                    })
                    .collect()
            });
            for (chunk, packed) in batch.iter().zip(packed) {
                let packed = packed?;
                let stored: &[u8] = if packed.len() < chunk.len() {
                    &packed
                } else {
                    chunk
                };
                let offset = self.out.stream_position()?;
                self.out.write_all(stored)?;
                self.chunks.push(ChunkRef {
                    offset,
                    stored: u32::try_from(stored.len()).map_err(|_| invalid("chunk too large"))?,
                });
                size += chunk.len() as u64;
            }
        }
        self.records.push(Record {
            path: path.to_owned(),
            is_dir: false,
            first_chunk,
            size,
        });
        Ok(size)
    }

    /// Writes the index and the header, and returns the output.
    ///
    /// # Errors
    ///
    /// Returns an error when the pack cannot be written, or a path is longer than 65535 bytes.
    pub fn finish(mut self) -> io::Result<W> {
        let index_offset = self.out.stream_position()?;
        let chunk_count =
            u64::try_from(self.chunks.len()).map_err(|_| invalid("too many chunks"))?;
        let mut index = chunk_count.to_le_bytes().to_vec();
        for chunk in &self.chunks {
            index.extend_from_slice(&chunk.offset.to_le_bytes());
            index.extend_from_slice(&chunk.stored.to_le_bytes());
        }
        let count = u32::try_from(self.records.len()).map_err(|_| invalid("too many entries"))?;
        index.extend_from_slice(&count.to_le_bytes());
        for record in &self.records {
            let path = record.path.as_bytes();
            let length = u16::try_from(path.len()).map_err(|_| invalid("path too long"))?;
            index.extend_from_slice(&length.to_le_bytes());
            index.extend_from_slice(path);
            index.push(u8::from(record.is_dir));
            index.extend_from_slice(&record.first_chunk.to_le_bytes());
            index.extend_from_slice(&record.size.to_le_bytes());
        }
        self.out.write_all(&index)?;
        self.out.seek(SeekFrom::Start(0))?;
        self.out.write_all(MAGIC)?;
        self.out.write_all(&index_offset.to_le_bytes())?;
        self.out.write_all(
            &u64::try_from(index.len())
                .map_err(|_| invalid("index too large"))?
                .to_le_bytes(),
        )?;
        self.out.flush()?;
        Ok(self.out)
    }
}

/// Reads little-endian fields off a byte slice.
struct Fields<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Fields<'a> {
    fn take(&mut self, length: usize) -> io::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(length)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| invalid("pack truncated"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
}

/// A file or directory in a pack: an index into the pack's nodes.
pub type NodeId = usize;

/// The root directory.
pub const ROOT: NodeId = 0;

#[derive(Debug)]
struct Node {
    name: String,
    is_dir: bool,
    first_chunk: u64,
    size: u64,
    /// The children by lower-case name, which is also the order they list in.
    children: BTreeMap<String, NodeId>,
}

impl Node {
    fn directory(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            is_dir: true,
            first_chunk: 0,
            size: 0,
            children: BTreeMap::new(),
        }
    }
}

/// What a node is, for a caller that lists or opens it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    /// The name as the disc spells it.
    pub name: String,
    /// Whether it is a directory.
    pub is_dir: bool,
    /// The size in bytes; zero for a directory.
    pub size: u64,
}

/// The decoded chunks used most recently, the newest last.
#[derive(Debug, Default)]
struct Cache {
    recent: Vec<(u64, Arc<Vec<u8>>)>,
}

impl Cache {
    fn get(&mut self, index: u64) -> Option<Arc<Vec<u8>>> {
        let at = self.recent.iter().position(|(held, _)| *held == index)?;
        let entry = self.recent.remove(at);
        let chunk = Arc::clone(&entry.1);
        self.recent.push(entry);
        Some(chunk)
    }

    fn put(&mut self, index: u64, chunk: Arc<Vec<u8>>) {
        // Another thread may have decoded the same chunk meanwhile: keep one copy, as the newest.
        if self.get(index).is_some() {
            return;
        }
        if self.recent.len() == CACHED_CHUNKS {
            self.recent.remove(0);
        }
        self.recent.push((index, chunk));
    }
}

/// An open pack.
#[derive(Debug)]
pub struct Pack {
    file: File,
    nodes: Vec<Node>,
    chunks: Vec<ChunkRef>,
    cache: Mutex<Cache>,
}

impl Pack {
    /// Opens the pack at `path` and reads its index.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, is not a pack, or has an index or a file
    /// record that lies outside the file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut header = [0u8; HEADER_LEN];
        file.read_exact_at(&mut header, 0)
            .map_err(|_| invalid("not a pack"))?;
        let mut fields = Fields {
            bytes: &header,
            at: 0,
        };
        if fields.take(MAGIC.len())? != MAGIC {
            return Err(invalid("not a pack"));
        }
        let index_offset = fields.u64()?;
        let index_length = fields.u64()?;
        let file_length = file.metadata()?.len();
        if index_offset
            .checked_add(index_length)
            .is_none_or(|end| end > file_length)
        {
            return Err(invalid("index out of range"));
        }
        let index_length = usize::try_from(index_length).map_err(|_| invalid("index too large"))?;
        let mut index = vec![0u8; index_length];
        file.read_exact_at(&mut index, index_offset)
            .map_err(|_| invalid("pack truncated"))?;
        let mut fields = Fields {
            bytes: &index,
            at: 0,
        };
        let chunk_count = fields.u64()?;
        let too_many = || invalid("chunk table out of range");
        if chunk_count
            .checked_mul(12)
            .is_none_or(|bytes| bytes > index_length as u64)
        {
            return Err(too_many());
        }
        let mut chunks = Vec::with_capacity(usize::try_from(chunk_count).map_err(|_| too_many())?);
        for _ in 0..chunk_count {
            let offset = fields.u64()?;
            let stored = fields.u32()?;
            if offset
                .checked_add(u64::from(stored))
                .is_none_or(|end| end > index_offset)
            {
                return Err(invalid("chunk out of range"));
            }
            chunks.push(ChunkRef { offset, stored });
        }
        let mut nodes = vec![Node::directory("")];
        for _ in 0..fields.u32()? {
            let length = usize::from(fields.u16()?);
            let path = std::str::from_utf8(fields.take(length)?)
                .map_err(|_| invalid("path is not UTF-8"))?
                .to_owned();
            let is_dir = fields.u8()? != 0;
            let first_chunk = fields.u64()?;
            let size = fields.u64()?;
            if !is_dir && !file_fits(&chunks, first_chunk, size) {
                return Err(invalid("file out of range"));
            }
            insert(&mut nodes, &path, is_dir, first_chunk, size);
        }
        Ok(Self {
            file,
            nodes,
            chunks,
            cache: Mutex::default(),
        })
    }

    /// Finds a file or directory by path, ignoring ASCII case. `\` or `/` separate the names, and
    /// the empty path is the root.
    #[must_use]
    pub fn lookup(&self, path: &str) -> Option<NodeId> {
        let mut at = ROOT;
        for part in components(path) {
            at = *self.nodes[at].children.get(&part.to_ascii_lowercase())?;
        }
        Some(at)
    }

    /// Describes a node.
    #[must_use]
    pub fn info(&self, id: NodeId) -> Info {
        let node = &self.nodes[id];
        Info {
            name: node.name.clone(),
            is_dir: node.is_dir,
            size: node.size,
        }
    }

    /// The nodes in a directory, sorted by lower-case name; none for a file.
    #[must_use]
    pub fn children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes[id].children.values().copied().collect()
    }

    /// Reads up to `buf.len()` bytes of a file from `offset`. Returns how many: zero at or past
    /// the end.
    ///
    /// # Errors
    ///
    /// Returns an error for a directory, or when a chunk is damaged or cannot be read.
    pub fn read_at(&self, id: NodeId, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let node = &self.nodes[id];
        if node.is_dir {
            return Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                "a directory has no bytes",
            ));
        }
        if offset >= node.size {
            return Ok(0);
        }
        let length = buf
            .len()
            .min(usize::try_from(node.size - offset).unwrap_or(usize::MAX));
        let mut done = 0;
        while done < length {
            let position = offset + done as u64;
            let within_file = position / CHUNK_BYTES;
            let chunk = self.chunk(node.first_chunk + within_file, node.size, within_file)?;
            let start = usize::try_from(position % CHUNK_BYTES)
                .map_err(|_| invalid("offset out of range"))?;
            let take = (chunk.len() - start).min(length - done);
            buf[done..done + take].copy_from_slice(&chunk[start..start + take]);
            done += take;
        }
        Ok(length)
    }

    /// The decoded chunk `index` of the pack, the `within_file`th of a file of `file_size` bytes.
    fn chunk(&self, index: u64, file_size: u64, within_file: u64) -> io::Result<Arc<Vec<u8>>> {
        let cached = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(index);
        if let Some(chunk) = cached {
            return Ok(chunk);
        }
        let length = usize::try_from((file_size - within_file * CHUNK_BYTES).min(CHUNK_BYTES))
            .expect("at most a chunk");
        let reference =
            self.chunks[usize::try_from(index).map_err(|_| invalid("chunk out of range"))?];
        let mut stored = vec![0u8; reference.stored as usize];
        self.file.read_exact_at(&mut stored, reference.offset)?;
        let bytes = if stored.len() == length {
            stored
        } else {
            let bytes = zstd::bulk::decompress(&stored, length)
                .map_err(|_| invalid("a chunk does not decompress"))?;
            if bytes.len() != length {
                return Err(invalid("a chunk decompresses to the wrong length"));
            }
            bytes
        };
        let chunk = Arc::new(bytes);
        self.cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .put(index, Arc::clone(&chunk));
        Ok(chunk)
    }
}

/// Whether a file of `size` bytes whose first chunk is `first_chunk` has all its chunks in the
/// table, none stored longer than it decodes.
fn file_fits(chunks: &[ChunkRef], first_chunk: u64, size: u64) -> bool {
    let wanted = size.div_ceil(CHUNK_BYTES);
    let Some(end) = first_chunk.checked_add(wanted) else {
        return false;
    };
    if end > chunks.len() as u64 {
        return false;
    }
    (0..wanted).all(|k| {
        let decoded = (size - k * CHUNK_BYTES).min(CHUNK_BYTES);
        let at = usize::try_from(first_chunk + k).expect("within the table");
        u64::from(chunks[at].stored) <= decoded
    })
}

/// Puts `path` in the tree, creating the directories on the way.
fn insert(nodes: &mut Vec<Node>, path: &str, is_dir: bool, first_chunk: u64, size: u64) {
    let mut at = ROOT;
    let mut named = false;
    for part in components(path) {
        named = true;
        let key = part.to_ascii_lowercase();
        let existing = nodes[at].children.get(&key).copied();
        at = if let Some(id) = existing {
            id
        } else {
            let id = nodes.len();
            nodes.push(Node::directory(part));
            nodes[at].children.insert(key, id);
            id
        };
    }
    if named {
        let node = &mut nodes[at];
        node.is_dir = is_dir;
        node.first_chunk = first_chunk;
        node.size = size;
    }
}

/// A file on a disc, read through the disc image. Each read reports the bytes of the disc read so
/// far, `read_before` of them in the files before this one, out of `total`.
struct DiscFile<'a, R> {
    iso: &'a mut XIso<R>,
    entry: &'a xiso::Entry,
    position: u64,
    read_before: u64,
    total: u64,
    progress: &'a mut dyn FnMut(u64, u64),
}

impl<R: Read + Seek> Read for DiscFile<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self
            .iso
            .read_at(self.entry, self.position, buf)
            .map_err(|error| io::Error::other(error.to_string()))?;
        self.position += read as u64;
        (self.progress)(self.read_before + self.position, self.total);
        Ok(read)
    }
}

/// Writes every file and directory of `iso` to a new pack in `out` at the default compression, and
/// returns `out`.
///
/// # Errors
///
/// Returns a message naming the path when the disc cannot be read or `out` cannot be written.
pub fn pack_xiso<R: Read + Seek, W: Write + Seek>(iso: &mut XIso<R>, out: W) -> Result<W, String> {
    pack_xiso_with_level(iso, out, DEFAULT_LEVEL)
}

/// Writes every file and directory of `iso` to a new pack in `out`, compressed at zstd `level`, and
/// returns `out`.
///
/// # Errors
///
/// Returns a message naming the path when the disc cannot be read or `out` cannot be written.
pub fn pack_xiso_with_level<R: Read + Seek, W: Write + Seek>(
    iso: &mut XIso<R>,
    out: W,
    level: i32,
) -> Result<W, String> {
    pack_xiso_with_progress(iso, out, level, &mut |_, _| {})
}

/// The bytes in the files of `iso`, all its directories down.
///
/// # Errors
///
/// Returns a message naming the directory when the disc cannot be read.
pub fn xiso_file_bytes<R: Read + Seek>(iso: &mut XIso<R>) -> Result<u64, String> {
    let mut total = 0;
    let mut pending = vec![iso.root().clone()];
    while let Some(dir) = pending.pop() {
        for child in iso
            .read_dir(&dir)
            .map_err(|error| format!("{}: {error}", dir.name()))?
        {
            if child.is_dir() {
                pending.push(child);
            } else {
                total += u64::from(child.size());
            }
        }
    }
    Ok(total)
}

/// [`pack_xiso_with_level`], calling `progress` as the files are read with the bytes read so far
/// and the bytes in all the files.
///
/// # Errors
///
/// Returns a message naming the path when the disc cannot be read or `out` cannot be written.
pub fn pack_xiso_with_progress<R: Read + Seek, W: Write + Seek>(
    iso: &mut XIso<R>,
    out: W,
    level: i32,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<W, String> {
    let total = xiso_file_bytes(iso)?;
    let mut done = 0;
    let mut writer = Writer::with_level(out, level).map_err(|error| error.to_string())?;
    let mut pending = vec![(iso.root().clone(), String::new())];
    while let Some((dir, path)) = pending.pop() {
        if !path.is_empty() {
            writer.add_dir(&path);
        }
        for child in iso
            .read_dir(&dir)
            .map_err(|error| format!("{path}: {error}"))?
        {
            let child_path = if path.is_empty() {
                child.name().to_owned()
            } else {
                format!("{path}\\{}", child.name())
            };
            if child.is_dir() {
                pending.push((child, child_path));
                continue;
            }
            let mut file = BufReader::with_capacity(
                1 << 20,
                DiscFile {
                    iso: &mut *iso,
                    entry: &child,
                    position: 0,
                    read_before: done,
                    total,
                    progress: &mut *progress,
                },
            );
            done += writer
                .add_file(&child_path, &mut file)
                .map_err(|error| format!("{child_path}: {error}"))?;
        }
    }
    writer.finish().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::os::unix::fs::FileExt as _;
    use std::path::PathBuf;

    use super::{CHUNK, DEFAULT_LEVEL, Pack, ROOT, Writer};

    /// Builds a pack at compression `level` in a temporary file with `build`, and returns its path.
    fn temp_pack_at(name: &str, level: i32, build: impl FnOnce(&mut Writer<File>)) -> PathBuf {
        let path = std::env::temp_dir().join(format!("anybox-pack-{}-{name}", std::process::id()));
        let mut writer = Writer::with_level(File::create(&path).unwrap(), level).unwrap();
        build(&mut writer);
        writer.finish().unwrap();
        path
    }

    fn temp_pack(name: &str, build: impl FnOnce(&mut Writer<File>)) -> PathBuf {
        temp_pack_at(name, DEFAULT_LEVEL, build)
    }

    /// `length` bytes of a repeatable pseudo-random stream: nothing compresses it.
    fn noise(length: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..length)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state.to_le_bytes()[3]
            })
            .collect()
    }

    #[test]
    fn files_read_back_by_path_ignoring_case() {
        let path = temp_pack("read", |writer| {
            writer
                .add_file("Movies\\Intro.bik", &mut &b"hello world"[..])
                .unwrap();
            writer.add_file("readme.txt", &mut &b"abc"[..]).unwrap();
        });
        let pack = Pack::open(&path).unwrap();
        let id = pack.lookup("movies/INTRO.BIK").unwrap();
        let info = pack.info(id);
        assert_eq!(info.name, "Intro.bik");
        assert_eq!(info.size, 11);
        assert!(!info.is_dir);
        let mut buf = [0u8; 5];
        assert_eq!(pack.read_at(id, 6, &mut buf).unwrap(), 5);
        assert_eq!(&buf, b"world");
        // A read that runs past the end is short, and one at the end is empty.
        let mut buf = [0u8; 10];
        assert_eq!(pack.read_at(id, 8, &mut buf).unwrap(), 3);
        assert_eq!(&buf[..3], b"rld");
        assert_eq!(pack.read_at(id, 11, &mut buf).unwrap(), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn directories_list_in_case_insensitive_name_order() {
        let path = temp_pack("list", |writer| {
            writer.add_file("b.txt", &mut &b"1"[..]).unwrap();
            writer.add_file("A.txt", &mut &b"2"[..]).unwrap();
            writer.add_file("sub\\c.txt", &mut &b"3"[..]).unwrap();
        });
        let pack = Pack::open(&path).unwrap();
        let names: Vec<String> = pack
            .children(ROOT)
            .into_iter()
            .map(|id| pack.info(id).name)
            .collect();
        assert_eq!(names, ["A.txt", "b.txt", "sub"]);
        let sub = pack.lookup("SUB").unwrap();
        assert!(pack.info(sub).is_dir);
        assert_eq!(pack.children(sub).len(), 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn an_empty_directory_is_kept() {
        let path = temp_pack("empty", |writer| writer.add_dir("saves"));
        let pack = Pack::open(&path).unwrap();
        let id = pack.lookup("saves").unwrap();
        assert!(pack.info(id).is_dir);
        assert_eq!(pack.children(id), Vec::<usize>::new());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_missing_path_is_none_and_a_directory_cannot_be_read() {
        let path = temp_pack("missing", |writer| {
            writer.add_file("d\\f", &mut &b"x"[..]).unwrap();
        });
        let pack = Pack::open(&path).unwrap();
        assert!(pack.lookup("nope").is_none());
        assert!(pack.lookup("d\\f\\deeper").is_none());
        let dir = pack.lookup("d").unwrap();
        assert!(pack.read_at(dir, 0, &mut [0u8; 1]).is_err());
        assert_eq!(pack.lookup("").unwrap(), ROOT);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_file_that_is_not_a_pack_is_rejected() {
        let path = std::env::temp_dir().join(format!("anybox-pack-{}-junk", std::process::id()));
        std::fs::write(&path, [7u8; 64]).unwrap();
        assert!(Pack::open(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    /// Writes a file holding the pack magic, then the given index offset and length, then `rest`.
    fn raw_pack(name: &str, index_offset: u64, index_length: u64, rest: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("anybox-pack-{}-{name}", std::process::id()));
        let mut bytes = super::MAGIC.to_vec();
        bytes.extend_from_slice(&index_offset.to_le_bytes());
        bytes.extend_from_slice(&index_length.to_le_bytes());
        bytes.extend_from_slice(rest);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn a_huge_index_length_is_rejected_without_allocating() {
        let path = raw_pack("hugelen", super::HEADER_LEN as u64, u64::MAX, &[]);
        let error = Pack::open(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn an_index_offset_past_the_end_is_rejected() {
        let path = raw_pack("badoff", u64::MAX - 4, 16, &[0u8; 16]);
        let error = Pack::open(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "index out of range");
        std::fs::remove_file(path).unwrap();
    }

    /// A v2 index: the chunk table (`offset`, `stored`), then one record per `(path, first chunk, size)`.
    fn index_of(chunks: &[(u64, u32)], records: &[(&str, u64, u64)]) -> Vec<u8> {
        let mut index = (chunks.len() as u64).to_le_bytes().to_vec();
        for (offset, stored) in chunks {
            index.extend_from_slice(&offset.to_le_bytes());
            index.extend_from_slice(&stored.to_le_bytes());
        }
        index.extend_from_slice(&u32::try_from(records.len()).unwrap().to_le_bytes());
        for (path, first_chunk, size) in records {
            index.extend_from_slice(&u16::try_from(path.len()).unwrap().to_le_bytes());
            index.extend_from_slice(path.as_bytes());
            index.push(0);
            index.extend_from_slice(&first_chunk.to_le_bytes());
            index.extend_from_slice(&size.to_le_bytes());
        }
        index
    }

    /// A pack of `data` bytes of filler followed by `index`.
    fn pack_with_index(name: &str, data: usize, index: &[u8]) -> PathBuf {
        let mut rest = vec![0u8; data];
        rest.extend_from_slice(index);
        raw_pack(
            name,
            (super::HEADER_LEN + data) as u64,
            index.len() as u64,
            &rest,
        )
    }

    fn assert_rejected(path: PathBuf, message: &str) {
        let error = Pack::open(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), message);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_file_record_past_the_end_is_rejected() {
        // No chunks at all, yet the file claims ten bytes.
        let index = index_of(&[], &[("f", 0, 10)]);
        assert_rejected(pack_with_index("badrec", 0, &index), "file out of range");
    }

    #[test]
    fn a_chunk_past_the_data_region_is_rejected() {
        // Ten bytes of data, and a chunk that runs one byte into the index.
        let index = index_of(&[(super::HEADER_LEN as u64, 11)], &[("f", 0, 11)]);
        assert_rejected(
            pack_with_index("badchunk", 10, &index),
            "chunk out of range",
        );
    }

    #[test]
    fn a_chunk_table_larger_than_the_index_is_rejected() {
        let index = u64::MAX.to_le_bytes().to_vec();
        assert_rejected(
            pack_with_index("hugetable", 0, &index),
            "chunk table out of range",
        );
    }

    #[test]
    fn a_file_whose_chunks_run_past_the_table_is_rejected() {
        // One chunk in the table, but 2 * CHUNK bytes need two.
        let index = index_of(
            &[(super::HEADER_LEN as u64, 10)],
            &[("f", 0, 2 * CHUNK as u64)],
        );
        assert_rejected(pack_with_index("badspan", 10, &index), "file out of range");
        // The same through a first chunk near the top of the range.
        let index = index_of(&[(super::HEADER_LEN as u64, 10)], &[("f", u64::MAX, 10)]);
        assert_rejected(pack_with_index("badfirst", 10, &index), "file out of range");
    }

    #[test]
    fn a_chunk_stored_longer_than_it_decodes_is_rejected() {
        // A 10-byte file whose only chunk claims 20 stored bytes.
        let index = index_of(&[(super::HEADER_LEN as u64, 20)], &[("f", 0, 10)]);
        assert_rejected(pack_with_index("badlen", 20, &index), "file out of range");
    }

    #[test]
    fn reads_across_chunk_boundaries_match_the_source() {
        let mut data = noise(CHUNK + 1000, 7);
        data.extend(std::iter::repeat_n(42u8, CHUNK));
        let path = temp_pack_at("chunks", 3, |writer| {
            writer.add_file("big.bin", &mut &data[..]).unwrap();
        });
        let pack = Pack::open(&path).unwrap();
        let id = pack.lookup("big.bin").unwrap();
        let mut all = vec![0u8; data.len() + 10];
        assert_eq!(pack.read_at(id, 0, &mut all).unwrap(), data.len());
        assert!(all[..data.len()] == data[..], "the whole file differs");
        for at in [CHUNK - 5, CHUNK + 995, 2 * CHUNK - 3] {
            let mut buf = [0u8; 10];
            assert_eq!(pack.read_at(id, at as u64, &mut buf).unwrap(), 10);
            assert_eq!(buf, data[at..at + 10], "at {at}");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn noise_is_stored_raw_and_repetition_is_squeezed() {
        let noisy = noise(CHUNK * 2, 1);
        let zeros = vec![0u8; CHUNK * 4];
        let path = temp_pack_at("sizes", 3, |writer| {
            writer.add_file("noise", &mut &noisy[..]).unwrap();
            writer.add_file("zeros", &mut &zeros[..]).unwrap();
        });
        // The noise takes its two chunks whole; the zeros almost nothing.
        let length = std::fs::metadata(&path).unwrap().len();
        let raw = 2 * CHUNK as u64;
        assert!(length > raw && length < raw + 16 * 1024, "{length}");
        let pack = Pack::open(&path).unwrap();
        for (name, want) in [("noise", &noisy), ("zeros", &zeros)] {
            let id = pack.lookup(name).unwrap();
            let mut got = vec![0u8; want.len()];
            assert_eq!(pack.read_at(id, 0, &mut got).unwrap(), want.len());
            assert!(&got == want, "{name} differs");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn caching_a_chunk_twice_keeps_one_copy() {
        use super::{CACHED_CHUNKS, Cache};
        use std::sync::Arc;
        let mut cache = Cache::default();
        cache.put(1, Arc::new(vec![1]));
        cache.put(1, Arc::new(vec![2]));
        assert_eq!(cache.recent.len(), 1);
        assert_eq!(*cache.get(1).unwrap(), vec![1]);
        for index in 10..10 + CACHED_CHUNKS as u64 {
            cache.put(index, Arc::new(vec![0]));
        }
        assert!(cache.get(1).is_none(), "the oldest was evicted");
        assert_eq!(cache.recent.len(), CACHED_CHUNKS);
    }

    #[test]
    fn an_empty_file_and_a_whole_chunk_round_trip() {
        let whole = noise(CHUNK, 3);
        let path = temp_pack_at("edges", 3, |writer| {
            assert_eq!(writer.add_file("empty", &mut &b""[..]).unwrap(), 0);
            assert_eq!(
                writer.add_file("whole", &mut &whole[..]).unwrap(),
                CHUNK as u64
            );
        });
        let pack = Pack::open(&path).unwrap();
        let empty = pack.lookup("empty").unwrap();
        assert_eq!(pack.info(empty).size, 0);
        assert_eq!(pack.read_at(empty, 0, &mut [0u8; 4]).unwrap(), 0);
        let id = pack.lookup("whole").unwrap();
        assert_eq!(pack.info(id).size, CHUNK as u64);
        let mut got = vec![0u8; CHUNK + 5];
        assert_eq!(pack.read_at(id, 0, &mut got).unwrap(), CHUNK);
        assert_eq!(got[..CHUNK], whole[..]);
        assert_eq!(pack.read_at(id, CHUNK as u64, &mut got).unwrap(), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_damaged_chunk_is_an_error_not_a_panic() {
        let path = temp_pack_at("damaged", 3, |writer| {
            writer
                .add_file("sevens", &mut &vec![7u8; CHUNK][..])
                .unwrap();
        });
        // The compressed frame starts right after the 24-byte header: spoil its magic number.
        let file = File::options().write(true).open(&path).unwrap();
        file.write_all_at(&[0xFF; 16], 24).unwrap();
        let pack = Pack::open(&path).unwrap();
        let id = pack.lookup("sevens").unwrap();
        let error = pack.read_at(id, 0, &mut [0u8; 8]).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_pack_with_no_entries_opens_empty() {
        let path = temp_pack("noentries", |_| {});
        let pack = Pack::open(&path).unwrap();
        assert_eq!(pack.children(ROOT), Vec::<usize>::new());
        std::fs::remove_file(path).unwrap();
    }
}
