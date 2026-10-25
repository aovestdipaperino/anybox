//! Resuming a title from its snapshot at boot, or from its perma snapshot while it runs.

use super::file;
use crate::kernel::Kernel;

/// Boots the title in `xbe` from its snapshot, if it has one this build made, else from its perma
/// snapshot: new guest memory filled from it, a kernel booted and given its state, and a host
/// thread for each guest thread it had running. The snapshot is consumed; the perma one stays.
/// Returns the guest base; `None` (after saying why, and deleting a snapshot that cannot be used,
/// though never a perma one) to boot afresh.
pub(crate) fn try_restore(xbe: &[u8]) -> Option<*mut u8> {
    let parsed = xbe::Xbe::parse(xbe.to_vec()).ok()?;
    let title = parsed.certificate().title_id();
    super::set_title(title);
    let regular = super::path(title)?;
    let read = |path: std::path::PathBuf| match std::fs::read(&path) {
        Ok(bytes) => Some(Some((path, bytes))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(None),
        Err(error) => {
            eprintln!("anyxbox: {}: {error}; booting afresh", path.display());
            None
        }
    };
    let (path, bytes) = match read(regular.clone())? {
        Some(found) => found,
        None => read(super::perma_path(&regular))??,
    };
    let perma = path != regular;
    // A host that cannot read the executable says nothing of the snapshot: it stays.
    let digest = match file::executable_digest() {
        Ok(digest) => digest,
        Err(error) => {
            eprintln!(
                "anyxbox: cannot check the snapshot against this build: {error}; booting afresh"
            );
            return None;
        }
    };
    let snapshot = file::decode(&bytes, title, &digest)
        .map_err(|why| {
            // A perma snapshot is kept for the bug it reproduces, even when this build cannot
            // resume it.
            if perma {
                eprintln!(
                    "anyxbox: {}: {why}; booting afresh and keeping it",
                    path.display()
                );
            } else {
                eprintln!("anyxbox: {}: {why}; booting afresh", path.display());
                super::discard();
            }
        })
        .ok()?;
    let booted = boot_from(snapshot, &parsed)
        .map_err(|error| eprintln!("anyxbox: {error}; booting afresh"))
        .ok()?;
    eprintln!("anyxbox: resuming from {}", path.display());
    if !perma {
        super::consume(&path);
    }
    let base = booted.base;
    crate::host::set_restored(start(booted));
    Some(base)
}

/// Reads the running title's perma snapshot, which must be one this build made: the title can
/// go back to it with [`rewind`].
///
/// # Errors
///
/// A message when there is no perma snapshot, or it cannot be read or used.
pub(crate) fn read_perma() -> Result<file::Snapshot, String> {
    let title = super::title().ok_or("no title has booted")?;
    let path = super::path(title)
        .map(|path| super::perma_path(&path))
        .ok_or("no place to keep the snapshot (HOME is not set)")?;
    let bytes = std::fs::read(&path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => "there is no perma snapshot".to_owned(),
        _ => format!("{}: {error}", path.display()),
    })?;
    let digest = file::executable_digest()?;
    file::decode(&bytes, title, &digest).map_err(|why| format!("{}: {why}", path.display()))
}

/// Takes the title back to `snapshot` after the old one has stopped (see
/// `crate::abi::rewind_console`): new guest memory and kernel from it, and its threads resumed.
/// Returns the picture the window should fade in; the title is held until
/// [`super::release`] when there is one.
///
/// # Errors
///
/// A message when the title cannot be booted again; the caller cannot carry on then.
pub(crate) fn rewind(snapshot: file::Snapshot) -> Result<Option<file::Picture>, String> {
    let bytes = crate::abi::xbe().ok_or("the title's executable is not known")?;
    let parsed = xbe::Xbe::parse(bytes.to_vec()).map_err(|error| format!("{error:?}"))?;
    Ok(start(boot_from(snapshot, &parsed)?))
}

/// A title booted from a snapshot, before its threads start.
struct Booted {
    kernel: Kernel,
    /// The guest base.
    base: *mut u8,
    threads: Vec<super::cpu::GuestCpu>,
    picture: Option<file::Picture>,
}

/// Boots `parsed` over new guest memory and gives the kernel `snapshot`'s memory and state.
fn boot_from(snapshot: file::Snapshot, parsed: &xbe::Xbe) -> Result<Booted, String> {
    let memory = crate::mem::Memory::new().ok_or("could not reserve the guest space to resume")?;
    let base = memory.base();
    crate::mem::set_guest_base(base);
    crate::fault::set_guest_base(base);
    let mut kernel = crate::boot::boot(memory, parsed)
        .map_err(|error| format!("cannot boot the title to resume it: {error}"))?;
    snapshot.memory.load(&kernel.memory);
    snapshot.kernel.apply(&mut kernel);
    kernel.publish_time();
    super::resume_tsc_at_start(snapshot.tsc);
    Ok(Booted {
        kernel,
        base,
        threads: snapshot.threads,
        picture: snapshot.picture,
    })
}

/// Installs the booted kernel and starts its threads. Returns the picture to fade in, which
/// holds the title frozen until the window has done so; the hold is set before any thread starts.
fn start(booted: Booted) -> Option<file::Picture> {
    let picture = booted.picture.filter(|_| crate::host::window_enabled());
    if picture.is_some() {
        super::hold();
    }
    crate::abi::install_restored(booted.kernel, booted.threads);
    picture
}
