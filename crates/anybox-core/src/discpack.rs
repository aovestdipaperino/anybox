//! The pack of the disc's files the app reads from beside itself.

use std::path::Path;

use crate::progress::{Phase, Reporter, Tracker, clock, note};

/// What `out/disc.pack.source` holds for a pack of `iso` at `level`: the disc's absolute path, its
/// length in bytes and the level, a line each.
pub(crate) fn pack_source(iso: &Path, level: i32) -> Result<String, String> {
    let absolute = iso.canonicalize().map_err(|error| error.to_string())?;
    let length = std::fs::metadata(iso)
        .map_err(|error| error.to_string())?
        .len();
    Ok(format!("{}\n{length}\n{level}", absolute.display()))
}

/// Whether `out/disc.pack` exists, is at least as new as `iso`, and was made from this disc at this
/// level, so need not be written again: it is gigabytes.
pub(crate) fn pack_is_current(out: &Path, iso: &Path, level: i32) -> bool {
    let modified = |path: &Path| std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let newer = match (modified(&out.join("disc.pack")), modified(iso)) {
        (Some(pack), Some(iso)) => pack >= iso,
        _ => false,
    };
    newer
        && pack_source(iso, level).is_ok_and(|want| {
            std::fs::read_to_string(out.join("disc.pack.source")).is_ok_and(|have| have == want)
        })
}

/// Writes the disc's files to `out/disc.pack`, unless that is current. It is written under
/// another name first, so an interrupted write is never taken for a pack.
pub(crate) fn write_pack(
    iso: &Path,
    out: &Path,
    level: i32,
    reporter: &dyn Reporter,
) -> Result<(), String> {
    let pack = out.join("disc.pack");
    let packing = Tracker::start(Phase::Compressing, reporter);
    if pack_is_current(out, iso, level) {
        packing.finish();
        note(reporter, format!("{} is current", pack.display()));
        return Ok(());
    }
    let mut took = std::time::Duration::ZERO;
    replace_pack(out, iso, level, |partial| {
        let file = std::fs::File::create(partial).map_err(|error| error.to_string())?;
        let source =
            std::fs::File::open(iso).map_err(|error| format!("{}: {error}", iso.display()))?;
        let mut disc = xiso::XIso::open(source).map_err(|error| format!("{error:?}"))?;
        note(reporter, format!("compressing the pack at level {level}"));
        let mut tracker = packing;
        pack::pack_xiso_with_progress(
            &mut disc,
            std::io::BufWriter::new(file),
            level,
            &mut |done, total| tracker.update(done, total),
        )?;
        took = tracker.finish();
        Ok(())
    })?;
    note(
        reporter,
        format!("wrote {} in {}", pack.display(), clock(took)),
    );
    Ok(())
}

/// Replaces `out/disc.pack` with what `build` writes to the path it is given. The old source file
/// goes first and the new one is written last, so a failure or a crash at any point leaves no
/// source, and a pack with no source is never current.
pub(crate) fn replace_pack(
    out: &Path,
    iso: &Path,
    level: i32,
    build: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    let source_file = out.join("disc.pack.source");
    match std::fs::remove_file(&source_file) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let partial = out.join("disc.pack.partial");
    build(&partial)?;
    std::fs::rename(&partial, out.join("disc.pack")).map_err(|error| error.to_string())?;
    std::fs::write(&source_file, pack_source(iso, level)?).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stale_source_is_removed_before_the_pack_is_replaced() {
        let dir = std::env::temp_dir().join(format!("anybox-replace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let iso = dir.join("a.iso");
        std::fs::write(&iso, b"x").unwrap();
        let (pack, source) = (dir.join("disc.pack"), dir.join("disc.pack.source"));
        std::fs::write(&pack, b"old").unwrap();
        std::fs::write(&source, pack_source(&iso, 3).unwrap()).unwrap();
        // A build that fails leaves the old pack and no source.
        let failed = replace_pack(&dir, &iso, 3, |_| Err("boom".to_owned()));
        assert_eq!(failed, Err("boom".to_owned()));
        assert!(!source.exists(), "the stale source survived a failed build");
        assert!(!pack_is_current(&dir, &iso, 3));
        // The build runs with the source already gone.
        let checked = replace_pack(&dir, &iso, 3, |partial| {
            assert!(!source.exists());
            std::fs::write(partial, b"new").map_err(|e| e.to_string())
        });
        assert_eq!(checked, Ok(()));
        assert_eq!(std::fs::read(&pack).unwrap(), b"new");
        assert_eq!(
            std::fs::read_to_string(&source).unwrap(),
            pack_source(&iso, 3).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_pack_is_current_only_for_the_same_newer_disc_and_level() {
        use std::time::{Duration, SystemTime};
        let dir = std::env::temp_dir().join(format!("anybox-fresh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (iso, other) = (dir.join("a.iso"), dir.join("b.iso"));
        let (pack, source) = (dir.join("disc.pack"), dir.join("disc.pack.source"));
        std::fs::write(&iso, b"x").unwrap();
        std::fs::write(&other, b"x").unwrap();
        assert!(!pack_is_current(&dir, &iso, 3), "no pack yet");
        std::fs::write(&pack, b"x").unwrap();
        let set = |path: &Path, age: u64| {
            let time = SystemTime::now() - Duration::from_secs(age);
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(time)
                .unwrap();
        };
        set(&iso, 100);
        set(&pack, 10);
        assert!(!pack_is_current(&dir, &iso, 3), "no source file");
        std::fs::write(&source, pack_source(&iso, 3).unwrap()).unwrap();
        assert!(pack_is_current(&dir, &iso, 3));
        assert!(!pack_is_current(&dir, &iso, 19), "another level");
        std::fs::write(&source, pack_source(&other, 3).unwrap()).unwrap();
        assert!(!pack_is_current(&dir, &iso, 3), "another disc");
        std::fs::write(&source, pack_source(&iso, 3).unwrap()).unwrap();
        std::fs::write(&iso, b"xy").unwrap();
        set(&iso, 100);
        assert!(!pack_is_current(&dir, &iso, 3), "another size");
        std::fs::write(&source, pack_source(&iso, 3).unwrap()).unwrap();
        assert!(pack_is_current(&dir, &iso, 3));
        set(&pack, 1000);
        assert!(!pack_is_current(&dir, &iso, 3), "the disc is newer");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
