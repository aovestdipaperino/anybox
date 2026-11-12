//! The snapshot file: a header that names the title and the build, then the snapshot,
//! compressed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use super::cpu::GuestCpu;
use super::resume_map::{self, ResumeMap};
use super::state::{KernelState, MemoryImage};

/// The file's magic.
const MAGIC: &[u8; 8] = b"ANYXHIB\0";
/// The format's version: a snapshot of another version is not read. As a runtime that changes
/// can still resume a title's snapshots (see [`executable_digest`]), any change to what a snapshot
/// holds, or how, must raise it.
const VERSION: u32 = 3;
/// The header: magic, version, title id, then the [`Build`]: the lifted code's digest, the
/// executable's, and the layout of a saved guest CPU.
const HEADER: usize = 8 + 4 + 4 + 20 + 20 + 4;

/// The build that makes a snapshot, or resumes one.
///
/// Guest memory holds the executable's image, and each thread's CPU is kept as the lifted code
/// lays it out, so a snapshot is resumed only by a build of the same executable and layout. The
/// lifted code itself may differ: a build that has every frame and `setjmp` context of the
/// snapshot in its resume map resumes it (see [`decode`]), so a fix to the lifter can be tried on
/// the snapshot that shows the bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Build {
    /// The lifted code's digest (see [`executable_digest`]).
    pub code: [u8; 20],
    /// The SHA-1 of the title's executable.
    pub xbe: [u8; 20],
    /// The layout of a saved guest CPU, from the lifted code's resume map; 0 for code without one.
    pub abi: u32,
}

impl Build {
    /// The running build, of the title whose executable is `xbe`.
    ///
    /// # Errors
    ///
    /// A message when the running executable cannot be read.
    pub fn current(xbe: &[u8]) -> Result<Self, String> {
        Ok(Self {
            code: executable_digest()?,
            xbe: Sha1::digest(xbe).into(),
            abi: resume_map::current().map_or(0, |map| map.abi),
        })
    }
}

/// A snapshot read back from its file.
#[derive(Debug)]
pub struct Decoded {
    pub snapshot: Snapshot,
    /// Whether another build of the title's lifted code made it.
    pub other_build: bool,
}

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

/// The file for `snapshot` of title `title` made by `build`.
///
/// # Errors
///
/// A message when the snapshot cannot be encoded or compressed.
pub fn encode(snapshot: &Snapshot, title: u32, build: &Build) -> Result<Vec<u8>, String> {
    let body = postcard::to_stdvec(snapshot).map_err(|error| error.to_string())?;
    let packed = zstd::stream::encode_all(body.as_slice(), 3).map_err(|error| error.to_string())?;
    let mut file = Vec::with_capacity(HEADER + packed.len());
    file.extend_from_slice(MAGIC);
    file.extend_from_slice(&VERSION.to_le_bytes());
    file.extend_from_slice(&title.to_le_bytes());
    file.extend_from_slice(&build.code);
    file.extend_from_slice(&build.xbe);
    file.extend_from_slice(&build.abi.to_le_bytes());
    file.extend_from_slice(&packed);
    Ok(file)
}

/// The snapshot in `file`, if `build` of title `title` can resume it: one of the same executable
/// and CPU layout, and either the same lifted code or code whose resume map `map` has every
/// frame and `setjmp` context of the snapshot.
///
/// # Errors
///
/// A message saying why the file does not hold such a snapshot.
pub fn decode(
    file: &[u8],
    title: u32,
    build: &Build,
    map: Option<ResumeMap>,
) -> Result<Decoded, String> {
    const ANYWAY: &str = "(ANYXBOX_SNAPSHOT_ANY_BUILD=1 tries it anyway)";
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
    let any = any_build();
    if file[36..56] != build.xbe && !any {
        return Err(format!(
            "a snapshot of another version of the title's executable {ANYWAY}"
        ));
    }
    if word(56) != build.abi && !any {
        return Err(format!(
            "a snapshot of code that lays a guest CPU out differently (layout {}, not {}) {ANYWAY}",
            word(56),
            build.abi
        ));
    }
    let other_build = file[16..36] != build.code;
    let body = zstd::stream::decode_all(&file[HEADER..]).map_err(|error| error.to_string())?;
    let snapshot: Snapshot = postcard::from_bytes(&body).map_err(|error| error.to_string())?;
    if other_build && !any {
        let map = map.ok_or_else(|| {
            format!(
                "a snapshot made by another build of the title, and this one has no resume map to \
                 check it against {ANYWAY}"
            )
        })?;
        map.check(&snapshot.threads).map_err(|why| {
            format!("a snapshot made by another build of the title, which cannot resume it: {why} {ANYWAY}")
        })?;
    }
    Ok(Decoded {
        snapshot,
        other_build,
    })
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

    const TITLE: u32 = 0x4D53_0004;
    const BUILD: Build = Build {
        code: [5; 20],
        xbe: [1; 20],
        abi: 2,
    };

    /// A snapshot whose one thread stands in function 0x1000 at the call at 0x1010.
    fn standing() -> Snapshot {
        let mut snapshot = snapshot();
        snapshot.threads[0].frames = vec![super::super::cpu::Frame {
            function: 0x1000,
            kind: 0,
            site: 0x1010,
        }];
        snapshot
    }

    /// A resume map file holding `entries`, sorted.
    fn map_file(entries: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut bytes = b"ANYXRSM\0".to_vec();
        bytes.extend_from_slice(&BUILD.abi.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_le_bytes());
        for &(function, site, kind) in entries {
            for word in [function, site, kind] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn a_snapshot_round_trips_through_its_file() {
        let bytes = encode(&snapshot(), TITLE, &BUILD).unwrap();
        assert_eq!(&bytes[..8], MAGIC);
        let back = decode(&bytes, TITLE, &BUILD, None).unwrap();
        assert!(!back.other_build);
        let back = back.snapshot;
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
        let bytes = encode(&with, TITLE, &BUILD).unwrap();
        let back = decode(&bytes, TITLE, &BUILD, None).unwrap();
        assert_eq!(back.snapshot.picture, Some(picture));
        let bytes = encode(&snapshot(), TITLE, &BUILD).unwrap();
        let back = decode(&bytes, TITLE, &BUILD, None).unwrap();
        assert_eq!(back.snapshot.picture, None);
    }

    #[test]
    fn a_snapshot_of_another_title_executable_or_layout_or_no_snapshot_at_all_is_refused() {
        let bytes = encode(&snapshot(), TITLE, &BUILD).unwrap();
        let refused = |title, build: Build| decode(&bytes, title, &build, None).unwrap_err();
        assert!(refused(0x4D53_0005, BUILD).contains("another title"));
        let other_xbe = Build {
            xbe: [2; 20],
            ..BUILD
        };
        assert!(refused(TITLE, other_xbe).contains("executable"));
        let other_abi = Build { abi: 3, ..BUILD };
        assert!(refused(TITLE, other_abi).contains("lays a guest CPU out"));
        assert!(
            decode(
                b"not a snapshot at all, not even close",
                TITLE,
                &BUILD,
                None
            )
            .is_err()
        );
        assert!(
            decode(&bytes[..60], TITLE, &BUILD, None).is_err(),
            "cut short"
        );
    }

    #[test]
    fn another_build_resumes_a_snapshot_only_when_its_map_has_every_frame() {
        let bytes = encode(&standing(), TITLE, &BUILD).unwrap();
        let relifted = Build {
            code: [6; 20],
            ..BUILD
        };
        assert!(
            decode(&bytes, TITLE, &relifted, None)
                .unwrap_err()
                .contains("no resume map")
        );
        let has = map_file(&[(0x1000, 0x1010, 0)]);
        let back = decode(
            &bytes,
            TITLE,
            &relifted,
            Some(ResumeMap::parse(&has).unwrap()),
        )
        .unwrap();
        assert!(back.other_build);
        assert_eq!(back.snapshot.threads, standing().threads);
        let lacks = map_file(&[(0x1000, 0x1014, 0)]);
        let lacks = ResumeMap::parse(&lacks).unwrap();
        assert!(
            decode(&bytes, TITLE, &relifted, Some(lacks))
                .unwrap_err()
                .contains("cannot resume it")
        );
        // The same build needs no map.
        assert!(
            !decode(&bytes, TITLE, &BUILD, Some(lacks))
                .unwrap()
                .other_build
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
