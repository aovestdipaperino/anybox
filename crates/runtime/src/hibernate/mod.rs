//! Hibernation (Cmd-X): every guest thread is frozen where it can be resumed, the title is
//! written to a snapshot, and the next boot of the same title resumes from it.
//!
//! A perma snapshot (Cmd-Shift-X) is written the same way, but the title carries on, and it is
//! never consumed: every boot resumes from it, which makes a bug easy to reproduce, until a new
//! one replaces it or a hibernation deletes it.

pub mod cpu;
pub mod file;
pub(crate) mod restore;
pub mod serde_util;
pub mod state;

use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use crate::kernel::Kernel;
use crate::kernel::thread::{Park, State};

/// No hibernation is under way.
const IDLE: u8 = 0;
/// Threads are being parked where they can be resumed.
const FREEZING: u8 = 1;
/// Every guest thread is parked where it can be resumed, and none runs.
const FROZEN: u8 = 2;

/// Where hibernation stands.
static PHASE: AtomicU8 = AtomicU8::new(IDLE);

/// Whether threads are being parked for a hibernation.
#[must_use]
pub fn freezing() -> bool {
    PHASE.load(Ordering::SeqCst) == FREEZING
}

/// Whether the title is frozen for a hibernation.
#[must_use]
pub fn frozen() -> bool {
    PHASE.load(Ordering::SeqCst) == FROZEN
}

/// Starts freezing; false if a hibernation is already under way.
pub(crate) fn request() -> bool {
    PHASE
        .compare_exchange(IDLE, FREEZING, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

/// The last thread has parked: the title is frozen. False if the hibernation was called off.
pub(crate) fn mark_frozen() -> bool {
    PHASE
        .compare_exchange(FREEZING, FROZEN, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

/// Why the last freeze was given up, for the driver waiting on it.
static GIVE_UP: Mutex<Option<String>> = Mutex::new(None);

/// Calls a freeze off because it cannot succeed, saying why; the caller keeps the baton and
/// carries on, so nothing is handed back. If the freeze was already called off (thawed), there
/// is nothing to give up and no reason is kept; carrying on is right in both cases.
pub(crate) fn give_up(reason: String) {
    if PHASE
        .compare_exchange(FREEZING, IDLE, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
        && let Ok(mut slot) = GIVE_UP.lock()
    {
        *slot = Some(reason);
    }
}

/// The reason the freeze was given up, if it was; taken.
fn take_give_up() -> Option<String> {
    GIVE_UP.lock().ok()?.take()
}

/// Calls the hibernation off: a frozen title gets its turn back.
pub(crate) fn thaw() {
    if PHASE.swap(IDLE, Ordering::SeqCst) == FROZEN {
        crate::abi::hand_back();
    }
}

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU32;
use std::sync::{Condvar, Mutex};

/// The title running, once booted; 0 before.
static TITLE: AtomicU32 = AtomicU32::new(0);

/// Records the title booted, for its snapshot's path.
pub(crate) fn set_title(title: u32) {
    TITLE.store(title, Ordering::SeqCst);
}

/// The title booted, if one has.
#[must_use]
pub fn title() -> Option<u32> {
    Some(TITLE.load(Ordering::SeqCst)).filter(|&title| title != 0)
}

/// Where title `title`'s snapshot is kept: `ANYXBOX_HIBERNATE`, else
/// `~/.anyxbox/TTTTTTTT.hibernate`, beside its saves.
#[must_use]
pub fn path(title: u32) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ANYXBOX_HIBERNATE") {
        return Some(path.into());
    }
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".anyxbox")
            .join(format!("{title:08x}.hibernate")),
    )
}

/// Where a snapshot is kept once a resume has consumed it: `path` with `-resumed` appended.
#[must_use]
pub fn resumed_path(path: &Path) -> PathBuf {
    let mut resumed = path.to_owned().into_os_string();
    resumed.push("-resumed");
    PathBuf::from(resumed)
}

/// Where the perma snapshot beside the snapshot at `path` is kept: `path` with `-perma` appended.
#[must_use]
pub fn perma_path(path: &Path) -> PathBuf {
    let mut perma = path.to_owned().into_os_string();
    perma.push("-perma");
    PathBuf::from(perma)
}

/// Whether the running title has a perma snapshot.
#[must_use]
pub fn perma_exists() -> bool {
    title()
        .and_then(path)
        .is_some_and(|path| perma_path(&path).exists())
}

/// Deletes the running title's snapshot and its consumed `-resumed` one, as a normal exit does;
/// a perma snapshot stays.
pub fn discard() {
    if let Some(path) = title().and_then(path) {
        discard_at(&path);
    }
}

/// Deletes the snapshot at `path` and the one a resume consumed from it, if either exists.
fn discard_at(path: &Path) {
    for file in [path.to_owned(), resumed_path(path)] {
        match std::fs::remove_file(&file) {
            Ok(()) => eprintln!("anyxbox: removed the snapshot {}", file.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => eprintln!("anyxbox: {}: {error}", file.display()),
        }
    }
}

/// Consumes the snapshot at `path` once it is applied, so no later launch resumes it again: it is
/// renamed to its `-resumed` path, or deleted if it cannot be.
pub(crate) fn consume(path: &Path) {
    let resumed = resumed_path(path);
    if let Err(error) = std::fs::rename(path, &resumed) {
        eprintln!(
            "anyxbox: cannot move {} to {}: {error}; deleting it",
            path.display(),
            resumed.display()
        );
        if let Err(error) = std::fs::remove_file(path) {
            eprintln!("anyxbox: {}: {error}", path.display());
        }
    }
}

/// Added to the host's time-stamp counter by the lifted code's `lift_rdtsc`: 0 in a fresh boot,
/// and set by a resume so the guest's counter goes on from where the hibernated title left it.
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static anyxbox_tsc_offset: AtomicI64 = AtomicI64::new(0);

/// The host's time-stamp counter as `lift_rdtsc` reads it: the monotonic clock at the Xbox CPU's
/// 733 MHz.
fn host_tsc() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid timespec to write.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
    #[allow(clippy::cast_sign_loss)]
    let nanos = (ts.tv_sec as u64)
        .wrapping_mul(1_000_000_000)
        .wrapping_add(ts.tv_nsec as u64);
    nanos.wrapping_mul(733) / 1000
}

/// The guest's time-stamp counter now, as the lifted code would read it.
fn guest_tsc() -> u64 {
    host_tsc().wrapping_add(anyxbox_tsc_offset.load(Ordering::Relaxed).cast_unsigned())
}

/// The offset that makes the guest's counter read `saved` when the host's reads `host_now`.
#[must_use]
pub fn tsc_offset(saved: u64, host_now: u64) -> i64 {
    saved.wrapping_sub(host_now).cast_signed()
}

/// Makes the guest's counter go on from `saved`, the value it had when the title hibernated.
pub(crate) fn continue_tsc(saved: u64) {
    anyxbox_tsc_offset.store(tsc_offset(saved, host_tsc()), Ordering::Relaxed);
}

/// The counter a restored title had when it hibernated, until it starts running again.
static RESUMED_TSC: Mutex<Option<u64>> = Mutex::new(None);

/// Keeps the counter a restored title had, for [`start_resumed`]; until then it reads that value.
pub(crate) fn resume_tsc_at_start(saved: u64) {
    continue_tsc(saved);
    *RESUMED_TSC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(saved);
}

/// The restored title starts running: its clock follows the host from now and its counter goes
/// on from the value it had, so neither the restore nor the fade-in shows as time passed. Without
/// that, the sound mixed the whole wait at once, over buffers the title had not refilled yet.
pub(crate) fn start_resumed(kernel: &mut Kernel) {
    kernel.clock.follow_host();
    if let Some(saved) = RESUMED_TSC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        continue_tsc(saved);
    }
}

/// Whether the resumed title is held frozen until the window has faded its picture in.
static HOLD: Mutex<bool> = Mutex::new(false);
/// Signalled when `HOLD` is released.
static RELEASED: Condvar = Condvar::new();

/// Holds the resumed title frozen: its current thread waits in `wait_until_released` before it
/// schedules. Called before any resumed thread starts.
pub(crate) fn hold() {
    *HOLD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
}

/// Lets the held title run; a no-op when it was never held.
pub fn release() {
    *HOLD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
    RELEASED.notify_all();
}

/// Waits until the title is released; returns at once when it is not held.
pub(crate) fn wait_until_released() {
    let mut held = HOLD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while *held {
        held = RELEASED
            .wait(held)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

/// How long the threads have to reach points they can be resumed from.
const FREEZE_TIMEOUT: Duration = Duration::from_secs(2);

/// What the driver does next while it waits for the title to freeze.
#[derive(Debug, PartialEq, Eq)]
enum Step {
    /// The title is frozen: write the snapshot.
    Proceed,
    /// Still freezing: look again shortly.
    Wait,
    /// Still freezing past the deadline: thaw and give up.
    TimedOut,
    /// The freeze was called off by a thread or a thaw.
    CalledOff,
}

/// Decides the next step from one reading of the phase.
fn wait_step(phase: u8, timed_out: bool) -> Step {
    match phase {
        FROZEN => Step::Proceed,
        FREEZING if timed_out => Step::TimedOut,
        FREEZING => Step::Wait,
        _ => Step::CalledOff,
    }
}

/// Hibernates the title: freezes it, writes its snapshot (with `picture`, the last image shown,
/// for the resume to fade in), and leaves it frozen for the caller to exit. Returns where the
/// snapshot went. A perma snapshot of the title is deleted, as this one is newer.
///
/// # Errors
///
/// A message when the title could not be frozen or written; then it carries on.
pub fn hibernate(picture: Option<file::Picture>) -> Result<PathBuf, String> {
    let path = freeze_and_write(picture, false)?;
    // A new snapshot supersedes the one a resume consumed, and the perma one.
    for old in [resumed_path(&path), perma_path(&path)] {
        match std::fs::remove_file(&old) {
            Ok(()) => eprintln!("anyxbox: removed the snapshot {}", old.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => eprintln!("anyxbox: {}: {error}", old.display()),
        }
    }
    Ok(path)
}

/// Writes a perma snapshot of the title, replacing any it had, and lets the title carry on.
/// Returns where the snapshot went.
///
/// # Errors
///
/// A message when the title could not be frozen or written; it carries on all the same.
pub fn perma_snapshot(picture: Option<file::Picture>) -> Result<PathBuf, String> {
    let path = freeze_and_write(picture, true)?;
    thaw();
    Ok(path)
}

/// Freezes the title and writes its snapshot, or its perma snapshot when `perma`, leaving it
/// frozen. Returns where the snapshot went.
fn freeze_and_write(picture: Option<file::Picture>, perma: bool) -> Result<PathBuf, String> {
    let title = title().ok_or("no title has booted")?;
    let path = path(title).ok_or("no place to keep the snapshot (HOME is not set)")?;
    let path = if perma { perma_path(&path) } else { path };
    let digest = file::executable_digest()?;
    // A reason left by an earlier freeze must go before this one starts, or a thread that gives
    // up at once would have its reason cleared.
    drop(take_give_up());
    if !request() {
        return Err("a hibernation is already under way".to_owned());
    }
    let deadline = Instant::now() + FREEZE_TIMEOUT;
    loop {
        // The phase is read once: reading it twice lets the last thread freeze in between.
        match wait_step(PHASE.load(Ordering::SeqCst), Instant::now() >= deadline) {
            Step::Proceed => break,
            Step::Wait => std::thread::sleep(Duration::from_millis(1)),
            Step::TimedOut => {
                thaw();
                return Err(
                    "the title did not reach a point it can be resumed from in 2 s".to_owned(),
                );
            }
            Step::CalledOff => {
                return Err(
                    take_give_up().unwrap_or_else(|| "the hibernation was called off".to_owned())
                );
            }
        }
    }
    let written = crate::abi::with_kernel(|kernel| encode_frozen(kernel, title, &digest, picture))
        .unwrap_or_else(|| Err("the title is not running".to_owned()))
        .and_then(|bytes| file::write(&path, &bytes));
    match written {
        Ok(()) => Ok(path),
        Err(error) => {
            thaw();
            Err(error)
        }
    }
}

/// Encodes the frozen title: each parked thread's CPU, guest memory, and the kernel, which lends
/// its state for the encoding and gets it back. Saves are written first, as a quit does.
fn encode_frozen(
    kernel: &mut Kernel,
    title: u32,
    digest: &[u8; 20],
    picture: Option<file::Picture>,
) -> Result<Vec<u8>, String> {
    kernel.fs.persist();
    let mut threads = Vec::new();
    for thread in kernel.threads.iter().filter(|t| t.state != State::Done) {
        if thread.park == Park::Start || kernel.unstarted.contains(&thread.id) {
            continue;
        }
        if !thread.park.resumable() {
            return Err(format!(
                "thread {} is not at a point it can be resumed from",
                thread.id
            ));
        }
        threads.push(cpu::read(thread.id, thread.park)?);
    }
    let memory = state::MemoryImage::read(&kernel.memory);
    let snapshot = file::Snapshot {
        memory,
        kernel: state::KernelState::take(kernel),
        threads,
        tsc: guest_tsc(),
        picture,
    };
    let encoded = file::encode(&snapshot, title, digest);
    snapshot.kernel.put_back(kernel);
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hibernation_goes_from_freezing_to_frozen_and_thaws() {
        assert!(!freezing() && !frozen());
        assert!(request());
        assert!(!request(), "one hibernation at a time");
        assert!(freezing());
        assert!(mark_frozen());
        assert!(frozen() && !freezing());
        assert!(!mark_frozen());
        PHASE.store(IDLE, Ordering::SeqCst);
        assert!(request());
        thaw();
        assert!(!freezing() && !frozen());
        assert!(
            !mark_frozen(),
            "a thawed title is not frozen by a late thread"
        );
        assert!(request());
        give_up("because".to_owned());
        assert!(!freezing() && !frozen());
        assert!(!mark_frozen(), "a given-up freeze is not frozen later");
        assert_eq!(take_give_up().as_deref(), Some("because"));
        assert_eq!(take_give_up(), None, "the reason is taken once");
    }

    #[test]
    fn a_held_title_waits_for_its_release() {
        wait_until_released(); // not held: returns at once
        hold();
        let waiter = std::thread::spawn(wait_until_released);
        std::thread::sleep(Duration::from_millis(50));
        assert!(!waiter.is_finished(), "held");
        release();
        waiter.join().unwrap();
        release(); // releasing twice is harmless
    }

    #[test]
    fn the_driver_proceeds_when_the_title_froze_even_past_the_deadline() {
        assert_eq!(wait_step(FROZEN, false), Step::Proceed);
        assert_eq!(wait_step(FROZEN, true), Step::Proceed);
    }

    #[test]
    fn the_driver_waits_until_the_deadline_then_times_out() {
        assert_eq!(wait_step(FREEZING, false), Step::Wait);
        assert_eq!(wait_step(FREEZING, true), Step::TimedOut);
    }

    #[test]
    fn the_driver_reports_a_freeze_called_off() {
        assert_eq!(wait_step(IDLE, false), Step::CalledOff);
        assert_eq!(wait_step(IDLE, true), Step::CalledOff);
    }

    #[test]
    fn the_resumed_path_appends_a_suffix_to_the_file_name() {
        assert_eq!(
            resumed_path(Path::new("/a/b/4d530004.hibernate")),
            Path::new("/a/b/4d530004.hibernate-resumed")
        );
    }

    #[test]
    fn discard_removes_the_snapshot_and_the_resumed_one() {
        let dir = std::env::temp_dir().join(format!("anyxbox-discard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.hibernate");
        discard_at(&path); // nothing there: silent
        std::fs::write(&path, b"x").unwrap();
        std::fs::write(resumed_path(&path), b"y").unwrap();
        discard_at(&path);
        assert!(!path.exists() && !resumed_path(&path).exists());
        std::fs::write(resumed_path(&path), b"y").unwrap();
        discard_at(&path);
        assert!(!resumed_path(&path).exists(), "either alone goes too");
        std::fs::write(&path, b"x").unwrap();
        consume(&path);
        assert!(!path.exists() && resumed_path(&path).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_perma_snapshot_sits_beside_the_snapshot_and_a_quit_keeps_it() {
        let dir = std::env::temp_dir().join(format!("anyxbox-perma-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.hibernate");
        assert_eq!(perma_path(&path), dir.join("t.hibernate-perma"));
        std::fs::write(perma_path(&path), b"p").unwrap();
        std::fs::write(&path, b"x").unwrap();
        discard_at(&path);
        assert!(!path.exists() && perma_path(&path).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_time_stamp_counter_goes_on_from_its_saved_value() {
        let at = |saved, host_now: u64| {
            host_now.wrapping_add(tsc_offset(saved, host_now).cast_unsigned())
        };
        // Resumed later on the same host: the counter would have jumped ahead.
        assert_eq!(tsc_offset(1_000, 5_000), -4_000);
        assert_eq!(at(1_000, 5_000), 1_000);
        // After a host reboot the host's counter is behind the saved one: it would have gone back.
        assert_eq!(tsc_offset(5_000, 1_000), 4_000);
        assert_eq!(at(5_000, 1_000), 5_000);
        assert_eq!(at(u64::MAX - 3, 2), u64::MAX - 3, "wraps as the C does");
    }

    #[test]
    fn the_snapshot_lives_beside_the_saves_named_by_the_title() {
        let path = path(0x4D53_0004).unwrap();
        if std::env::var_os("ANYXBOX_HIBERNATE").is_none() {
            assert!(
                path.ends_with(".anyxbox/4d530004.hibernate"),
                "{}",
                path.display()
            );
        }
    }
}
