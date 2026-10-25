//! The snapshot file: a header that names the title and the build, then the snapshot,
//! compressed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use super::cpu::GuestCpu;
use super::state::{KernelState, MemoryImage};

/// The file's magic.
const MAGIC: &[u8; 8] = b"ANYXHIB\0";
/// The format's version: a snapshot of another version is not read. As a runtime that changes
/// can still resume a title's snapshots (see [`executable_digest`]), any change to what a snapshot
/// holds, or how, must raise it.
const VERSION: u32 = 2;
/// The header: magic, version, title id, the build's digest.
const HEADER: usize = 8 + 4 + 4 + 20;

/// The last picture the window showed, kept so a resume can fade it in.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Picture {
    /// The picture's 0RGB pixels, row by row.
    pub pixels: Vec<u32>,
    /// Its width in pixels.
    pub width: u32,
    /// Its height in pixels.
    pub height: u32,
}

/// A frozen title.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub memory: MemoryImage,
    pub kernel: KernelState,
    /// The guest threads that had run, each parked where it can be resumed.
    pub threads: Vec<GuestCpu>,
    /// The guest's time-stamp counter when the title froze.
    pub tsc: u64,
    /// The last picture the window showed; none when the title hibernated without a window.
    pub picture: Option<Picture>,
}

/// The digest of the title's lifted code, which the glue hands over before boot.
static CODE_DIGEST: std::sync::OnceLock<[u8; 20]> = std::sync::OnceLock::new();

/// Records the digest of the title's lifted code: the SHA-1 the build took of the lifted sources,
/// the title's executable and how they were compiled. Called by the glue before
/// `anyxbox_boot`.
///
/// # Safety
///
/// `digest` must point to 20 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_set_code_digest(digest: *const u8) {
    if digest.is_null() {
        return;
    }
    let mut bytes = [0u8; 20];
    // SAFETY: the caller hands 20 readable bytes.
    unsafe { std::ptr::copy_nonoverlapping(digest, bytes.as_mut_ptr(), 20) };
    let _ = CODE_DIGEST.set(bytes);
}

/// The build a snapshot belongs to: the digest of the title's lifted code, so a build that only
/// changes the runtime still resumes the snapshots of the one before; else (a program that did
/// not hand one over) the SHA-1 of the running executable.
///
/// # Errors
///
/// A message when the executable cannot be read.
pub fn executable_digest() -> Result<[u8; 20], String> {
    if let Some(digest) = CODE_DIGEST.get() {
        return Ok(*digest);
    }
    let exe = std::env::current_exe().map_err(|error| error.to_string())?;
    let bytes = std::fs::read(&exe).map_err(|error| format!("{}: {error}", exe.display()))?;
    Ok(Sha1::digest(&bytes).into())
}

/// The file for `snapshot` of title `title` made by the executable whose SHA-1 is `digest`.
///
/// # Errors
///
/// A message when the snapshot cannot be encoded or compressed.
pub fn encode(snapshot: &Snapshot, title: u32, digest: &[u8; 20]) -> Result<Vec<u8>, String> {
    let body = postcard::to_stdvec(snapshot).map_err(|error| error.to_string())?;
    let packed = zstd::stream::encode_all(body.as_slice(), 3).map_err(|error| error.to_string())?;
    let mut file = Vec::with_capacity(HEADER + packed.len());
    file.extend_from_slice(MAGIC);
    file.extend_from_slice(&VERSION.to_le_bytes());
    file.extend_from_slice(&title.to_le_bytes());
    file.extend_from_slice(digest);
    file.extend_from_slice(&packed);
    Ok(file)
}

/// The snapshot in `file`, if title `title` built as `digest` made it.
///
/// # Errors
///
/// A message saying why the file does not hold such a snapshot.
pub fn decode(file: &[u8], title: u32, digest: &[u8; 20]) -> Result<Snapshot, String> {
    if file.len() < HEADER || &file[..8] != MAGIC {
        return Err("not a snapshot".to_owned());
    }
    let word = |at: usize| u32::from_le_bytes(file[at..at + 4].try_into().unwrap_or_default());
    if word(8) != VERSION {
        return Err(format!("a snapshot of format {}, not {VERSION}", word(8)));
    }
    if word(12) != title {
        return Err(format!("a snapshot of another title ({:08x})", word(12)));
    }
    if &file[16..HEADER] != digest && !any_build() {
        return Err(
            "a snapshot made by another build of the title (ANYXBOX_SNAPSHOT_ANY_BUILD=1 tries it \
             anyway)"
                .to_owned(),
        );
    }
    let body = zstd::stream::decode_all(&file[HEADER..]).map_err(|error| error.to_string())?;
    postcard::from_bytes(&body).map_err(|error| error.to_string())
}

/// Whether `ANYXBOX_SNAPSHOT_ANY_BUILD=1` asks to resume a snapshot whatever build made it, at
/// the risk of a title that misbehaves when its code has changed.
fn any_build() -> bool {
    std::env::var_os("ANYXBOX_SNAPSHOT_ANY_BUILD").is_some_and(|v| v == "1")
}

/// Writes `bytes` to `path` through a partial file renamed over it, making its directory.
///
/// # Errors
///
/// A message when the file cannot be written.
pub fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut partial = path.to_owned().into_os_string();
    partial.push("-partial");
    let partial = PathBuf::from(partial);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    }
    std::fs::write(&partial, bytes)
        .and_then(|()| std::fs::rename(&partial, path))
        .map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Snapshot {
        Snapshot {
            memory: MemoryImage {
                pages: vec![(0x0001_0000, vec![7; 16])],
                shared: vec![1, 2, 3],
            },
            kernel: KernelState {
                current: 4,
                ..KernelState::default()
            },
            threads: vec![GuestCpu {
                tid: 4,
                cpu: vec![9; 8],
                ..GuestCpu::default()
            }],
            tsc: 0x1234_5678_9ABC,
            picture: None,
        }
    }

    #[test]
    fn a_snapshot_round_trips_through_its_file() {
        let bytes = encode(&snapshot(), 0x4D53_0004, &[5; 20]).unwrap();
        assert_eq!(&bytes[..8], MAGIC);
        let back = decode(&bytes, 0x4D53_0004, &[5; 20]).unwrap();
        assert_eq!(back.memory.pages, snapshot().memory.pages);
        assert_eq!(back.kernel.current, 4);
        assert_eq!(back.threads, snapshot().threads);
        assert_eq!(back.tsc, 0x1234_5678_9ABC);
    }

    #[test]
    fn a_snapshot_keeps_its_picture() {
        let picture = Picture {
            pixels: vec![0x0012_3456, 0x00AB_CDEF, 0, 0x00FF_FFFF, 1, 2],
            width: 3,
            height: 2,
        };
        let with = Snapshot {
            picture: Some(picture.clone()),
            ..snapshot()
        };
        let bytes = encode(&with, 0x4D53_0004, &[5; 20]).unwrap();
        let back = decode(&bytes, 0x4D53_0004, &[5; 20]).unwrap();
        assert_eq!(back.picture, Some(picture));
        let bytes = encode(&snapshot(), 0x4D53_0004, &[5; 20]).unwrap();
        assert_eq!(decode(&bytes, 0x4D53_0004, &[5; 20]).unwrap().picture, None);
    }

    #[test]
    fn a_snapshot_of_another_title_or_build_or_no_snapshot_at_all_is_refused() {
        let bytes = encode(&snapshot(), 0x4D53_0004, &[5; 20]).unwrap();
        assert!(
            decode(&bytes, 0x4D53_0005, &[5; 20])
                .unwrap_err()
                .contains("another title")
        );
        assert!(
            decode(&bytes, 0x4D53_0004, &[6; 20])
                .unwrap_err()
                .contains("another build")
        );
        assert!(
            decode(
                b"not a snapshot at all, not even close",
                0x4D53_0004,
                &[5; 20]
            )
            .is_err()
        );
        assert!(
            decode(&bytes[..40], 0x4D53_0004, &[5; 20]).is_err(),
            "cut short"
        );
    }

    #[test]
    fn a_write_replaces_the_file_whole() {
        let dir = std::env::temp_dir().join(format!("anybox-hibernate-{}", std::process::id()));
        let path = dir.join("nested").join("title.hibernate");
        write(&path, b"first").unwrap();
        write(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
