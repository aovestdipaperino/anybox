//! Recording the controller to a file and playing it back (spec 12.2): a run can be repeated to
//! reach the same place in the title without a human at the keyboard. `ANYXBOX_RECORD_INPUT`
//! names the file to record to, `ANYXBOX_PLAY_INPUT` the one to play.
//!
//! A recording has one line per change of the controller, timed from boot:
//! `WHEN BUTTONS A,B,X,Y,BLACK,WHITE,LT,RT LX,LY,RX,RY`, the buttons in hexadecimal. `WHEN` is
//! either seconds (`12.500`, the older form) or a frame count (`f750`): the number of frames the
//! title has shown. Frames are paced at 60 a second and counted by the title's own flips, so a
//! playback by frame applies a change at the same point of the title's run each time, whatever
//! the host's speed; one by seconds follows the wall clock and reaches the same place only as
//! far as the title gets there at the same moments.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission.

use std::collections::VecDeque;
use std::io::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::input::Pad;

/// When in a run a change of the controller happens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum When {
    /// Seconds after boot.
    Seconds(f64),
    /// Once this many frames have been shown.
    Frame(u64),
}

/// One controller state as a recording line, at `when`.
#[must_use]
pub fn line(when: When, pad: &Pad) -> String {
    let stamp = match when {
        When::Seconds(seconds) => format!("{seconds:.3}"),
        When::Frame(frame) => format!("f{frame}"),
    };
    let list = |values: &mut dyn Iterator<Item = String>| values.collect::<Vec<_>>().join(",");
    format!(
        "{stamp} {:04x} {} {}",
        pad.buttons,
        list(&mut pad.analog.iter().map(u8::to_string)),
        list(&mut pad.sticks.iter().map(i16::to_string)),
    )
}

/// Reads one recording line back.
#[must_use]
pub fn parse(line: &str) -> Option<(When, Pad)> {
    let mut parts = line.split_whitespace();
    let stamp = parts.next()?;
    let when = match stamp.strip_prefix('f') {
        Some(frame) => When::Frame(frame.parse().ok()?),
        None => When::Seconds(stamp.parse().ok()?),
    };
    let buttons = u16::from_str_radix(parts.next()?, 16).ok()?;
    let analog: Vec<u8> = parts
        .next()?
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    let sticks: Vec<i16> = parts
        .next()?
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    Some((
        when,
        Pad {
            buttons,
            analog: analog.try_into().ok()?,
            sticks: sticks.try_into().ok()?,
        },
    ))
}

/// Writes each change of the controller to a recording.
#[derive(Debug)]
pub struct Recorder {
    out: std::io::BufWriter<std::fs::File>,
}

impl Recorder {
    /// Starts a recording at `path`, replacing any file there.
    ///
    /// # Errors
    ///
    /// When the file cannot be created.
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self {
            out: std::io::BufWriter::new(std::fs::File::create(path)?),
        })
    }

    /// With `ANYXBOX_RECORD_INPUT`, starts recording to that file.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let path = std::path::PathBuf::from(std::env::var_os("ANYXBOX_RECORD_INPUT")?);
        match Self::create(&path) {
            Ok(recorder) => {
                eprintln!("anyxbox: recording the controller to {}", path.display());
                Some(recorder)
            }
            Err(error) => {
                eprintln!("anyxbox: cannot record to {}: {error}", path.display());
                None
            }
        }
    }

    /// Records that the controller became `pad` at `when`. Each line is flushed, so a run that
    /// ends abruptly keeps its recording.
    pub fn record(&mut self, when: When, pad: &Pad) {
        let written = writeln!(self.out, "{}", line(when, pad)).and_then(|()| self.out.flush());
        if let Err(error) = written {
            eprintln!("anyxbox: cannot record the controller: {error}");
        }
    }
}

/// Reads a recording, in time order.
///
/// # Errors
///
/// When the file cannot be read.
pub fn load(path: &std::path::Path) -> std::io::Result<Vec<(When, Pad)>> {
    let text = std::fs::read_to_string(path)?;
    Ok(sorted(text.lines().filter_map(parse).collect()))
}

/// `changes` in time order; changes at the same time keep their order.
fn sorted(mut changes: Vec<(When, Pad)>) -> Vec<(When, Pad)> {
    let key = |when: &When| match *when {
        When::Seconds(seconds) => seconds,
        #[allow(clippy::cast_precision_loss)]
        When::Frame(frame) => frame as f64 / 60.0,
    };
    changes.sort_by(|a, b| key(&a.0).total_cmp(&key(&b.0)));
    changes
}

/// Controller changes of a recording still to apply, each once the title has shown its frame.
/// They are applied at the flip itself, so a replay does not depend on thread timing.
static PLAYBACK: Mutex<VecDeque<(u64, Pad)>> = Mutex::new(VecDeque::new());

/// Counts the playbacks cleared, so a playback thread stops once its own is cleared.
static PLAYBACK_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Forgets the changes still to play, as a reset does.
pub fn clear_playback() {
    PLAYBACK_GENERATION.fetch_add(1, Ordering::SeqCst);
    PLAYBACK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// Plays the recording `ANYXBOX_PLAY_INPUT` names, if any: each change is applied at its frame,
/// or at its time after boot on a thread of its own. The keyboard still works, and takes over
/// when a key changes.
pub fn start_playback() {
    let Some(path) = std::env::var_os("ANYXBOX_PLAY_INPUT").map(std::path::PathBuf::from) else {
        return;
    };
    let changes = match load(&path) {
        Ok(changes) => changes,
        Err(error) => {
            eprintln!(
                "anyxbox: cannot read the recording {}: {error}",
                path.display()
            );
            return;
        }
    };
    eprintln!(
        "anyxbox: playing {} controller changes from {}",
        changes.len(),
        path.display()
    );
    // Changes stamped with a frame wait in a queue the flips drain; the rest follow the clock.
    let mut timed = Vec::new();
    {
        let mut queue = PLAYBACK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (when, pad) in changes {
            match when {
                When::Frame(frame) => queue.push_back((frame, pad)),
                When::Seconds(at) => timed.push((at, pad)),
            }
        }
    }
    if timed.is_empty() {
        return;
    }
    let start = std::time::Instant::now();
    let generation = PLAYBACK_GENERATION.load(Ordering::SeqCst);
    let spawned = std::thread::Builder::new()
        .name("anyxbox-playback".to_owned())
        .spawn(move || {
            for (at, pad) in timed {
                let due = start + std::time::Duration::from_secs_f64(at.max(0.0));
                std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
                // A reset cleared this playback; the new boot has its own.
                if PLAYBACK_GENERATION.load(Ordering::SeqCst) != generation {
                    return;
                }
                crate::abi::with_kernel(|kernel| super::set_pad(kernel, pad));
            }
            eprintln!("anyxbox: the recording has ended");
        });
    if let Err(error) = spawned {
        eprintln!("anyxbox: cannot play the recording: {error}");
    }
}

/// Applies the recorded controller changes that are due once `frames` frames have been shown.
pub(crate) fn apply_due(kernel: &mut crate::kernel::Kernel, frames: u64) {
    let mut queue = PLAYBACK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while queue.front().is_some_and(|(frame, _)| *frame <= frames) {
        if let Some((frame, pad)) = queue.pop_front() {
            super::set_pad(kernel, pad);
            if queue.is_empty() {
                eprintln!("anyxbox: the recording has ended (frame {frames}, due {frame})");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_reads_back_as_the_state_it_records() {
        let pad = Pad {
            buttons: 0x0011,
            analog: [255, 0, 0, 0, 0, 0, 12, 0],
            sticks: [i16::MIN, 0, 0, i16::MAX],
        };
        let text = line(When::Seconds(12.5), &pad);
        assert_eq!(text, "12.500 0011 255,0,0,0,0,0,12,0 -32768,0,0,32767");
        assert_eq!(parse(&text), Some((When::Seconds(12.5), pad)));
    }

    #[test]
    fn a_frame_stamped_line_reads_back_too() {
        let pad = Pad::default();
        let text = line(When::Frame(750), &pad);
        assert!(text.starts_with("f750 0000 "));
        assert_eq!(parse(&text), Some((When::Frame(750), pad)));
    }

    #[test]
    fn changes_keep_their_order_when_the_stamps_tie() {
        let a = Pad {
            buttons: 1,
            ..Pad::default()
        };
        let b = Pad {
            buttons: 2,
            ..Pad::default()
        };
        let out = sorted(vec![
            (When::Frame(5), a),
            (When::Frame(2), b),
            (When::Frame(5), b),
        ]);
        assert_eq!(
            out,
            vec![
                (When::Frame(2), b),
                (When::Frame(5), a),
                (When::Frame(5), b)
            ]
        );
    }

    #[test]
    fn a_malformed_line_is_skipped() {
        assert_eq!(parse("1.0 0011 255,0"), None);
        assert_eq!(parse("# a comment"), None);
    }
}
