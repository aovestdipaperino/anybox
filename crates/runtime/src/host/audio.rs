//! Audio output (spec 12.3): the PCM the mixer releases (16-bit stereo at 44.1 kHz) is queued as
//! `f32`, at most two seconds of it, and a `cpal` stream on the default device plays it,
//! resampled to the device's rate. The stream waits for 100 ms to be queued before it plays, at
//! the start and whenever the queue runs dry, so it does not stutter.
//!
//! The title bar's volume slider scales what is played, apart from the system's own volume.
//!
//! `ANYXBOX_SOUND=0` turns sound off, `ANYXBOX_AUDIO_OUT=file` also writes the raw PCM there, and
//! `ANYXBOX_TRACE_AUDIO` prints how many frames came and went each second.

use std::collections::VecDeque;
use std::io::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// The mixer's rate.
const RATE: f64 = 44_100.0;
/// The most frames kept: two seconds.
const MAX_QUEUED: usize = 2 * 44_100;
/// Frames queued before playing starts: 100 ms.
const PRIME: usize = 4410;
/// After the queue runs dry while playing, it waits for this much more before playing again, up
/// to [`MOST_PRIME`]: the title's threads were late by more than the margin, so the margin grows
/// rather than the sound breaking again and again.
const PRIME_STEP: usize = 2205;
const MOST_PRIME: usize = 3 * 4410;

/// The frames waiting to be played, and the output's state.
#[derive(Debug)]
struct Queue {
    frames: VecDeque<[f32; 2]>,
    /// Whether the output is playing, rather than waiting for the queue to fill.
    primed: bool,
    /// How far the output is between the first two frames.
    phase: f64,
    /// The frames queued before playing (again).
    prime: usize,
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue {
    frames: VecDeque::new(),
    primed: false,
    phase: 0.0,
    prime: PRIME,
});
/// Times the queue ran dry while playing, for `ANYXBOX_TRACE_AUDIO`.
static DRY: AtomicU64 = AtomicU64::new(0);

/// Set when the console turns off: from then on the title's sound is not played.
static MUTED: AtomicBool = AtomicBool::new(false);
/// The gain the title bar's slider sets, as `f32` bits: 1.0 at first.
static GAIN: AtomicU32 = AtomicU32::new(0x3f80_0000);
/// Frames queued and played, for `ANYXBOX_TRACE_AUDIO`.
static QUEUED: AtomicU64 = AtomicU64::new(0);
static PLAYED: AtomicU64 = AtomicU64::new(0);
/// `ANYXBOX_AUDIO_OUT`, once opened.
static FILE: Mutex<Option<std::io::BufWriter<std::fs::File>>> = Mutex::new(None);

/// Whether sound is on: unless `ANYXBOX_SOUND=0`, read once, as the mixer pushes often.
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ANYXBOX_SOUND").map_or(true, |v| v != "0"))
}

fn queue() -> std::sync::MutexGuard<'static, Queue> {
    QUEUE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Queues PCM the mixer released.
pub fn push(pcm: &[[i16; 2]]) {
    if pcm.is_empty() || !enabled() || MUTED.load(Ordering::Relaxed) {
        return;
    }
    if let Some(file) = FILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_mut()
    {
        let bytes: Vec<u8> = pcm.iter().flatten().flat_map(|s| s.to_le_bytes()).collect();
        let _ = file.write_all(&bytes);
    }
    QUEUED.fetch_add(pcm.len() as u64, Ordering::Relaxed);
    let mut queue = queue();
    queue.frames.extend(
        pcm.iter()
            .map(|&[l, r]| [f32::from(l) / 32_768.0, f32::from(r) / 32_768.0]),
    );
    let excess = queue.frames.len().saturating_sub(MAX_QUEUED);
    queue.frames.drain(..excess);
}

/// Sets the volume, from 0 (silent) to 1 (as the mixer made it). The slider moves evenly in
/// loudness, so the gain is its square.
pub fn set_volume(volume: f32) {
    let volume = volume.clamp(0.0, 1.0);
    GAIN.store((volume * volume).to_bits(), Ordering::Relaxed);
}

/// Drops the sound queued but not played yet, as the title goes back to a perma snapshot.
pub fn flush() {
    let mut queue = queue();
    queue.frames.clear();
    queue.primed = false;
}

/// Silences the title for good, as switching off does; sounds of the console's own, such as
/// [`bump`], still play.
pub fn mute() {
    MUTED.store(true, Ordering::Relaxed);
    let mut queue = queue();
    queue.frames.clear();
    queue.primed = false;
}

/// The thump of a CRT switching off as its last dot flares: a low tone falling in pitch and
/// dying fast, with a soft click at its start.
///
/// Copied from the author's original `anybox-tk` code, with the author's permission.
pub fn bump() {
    queue().frames.extend(thump());
}

/// The thump's samples, 0.4 s of them.
fn thump() -> Vec<[f32; 2]> {
    // The source rate, and 0.4 s of it.
    const RATE: f32 = 44_100.0;
    const FRAMES: usize = 17_640;
    let mut noise = 0x1234_5678_u32;
    let mut low = 0.0f32;
    let mut phase = 0.0f32;
    (0..FRAMES)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / RATE;
            // 60 Hz falling to 40 Hz, dying in a tenth of a second.
            phase += std::f32::consts::TAU * (40.0 + 20.0 * (-t / 0.1).exp()) / RATE;
            let thud = phase.sin() * (-t / 0.09).exp() * 0.9;
            // The click: a short burst of noise, smoothed so it is soft.
            noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_precision_loss)]
            let white = (noise >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            low += (white - low) * 0.15;
            let click = low * (-t / 0.006).exp() * 0.5;
            // The last 50 ms fade to silence, so the sound does not stop on a click.
            #[allow(clippy::cast_precision_loss)]
            let fade = ((FRAMES - i) as f32 / 2_205.0).min(1.0);
            let sample = ((thud + click) * fade).clamp(-1.0, 1.0);
            [sample, sample]
        })
        .collect()
}

/// Fills `out`, `channels` samples a frame at `rate`, from the queue: left on even channels,
/// right on odd ones, resampled by linear interpolation and scaled by `gain`; silence while it is
/// not primed.
fn fill(queue: &mut Queue, out: &mut [f32], channels: usize, rate: f64, gain: f32) {
    let step = RATE / rate;
    for frame in out.chunks_mut(channels.max(1)) {
        if !queue.primed && queue.frames.len() >= queue.prime {
            queue.primed = true;
        }
        if !queue.primed || queue.frames.len() < 2 {
            if queue.primed {
                queue.prime = (queue.prime + PRIME_STEP).min(MOST_PRIME);
                DRY.fetch_add(1, Ordering::Relaxed);
            }
            queue.primed = false;
            frame.fill(0.0);
            continue;
        }
        let (a, b) = (queue.frames[0], queue.frames[1]);
        // Within 0..1.
        #[allow(clippy::cast_possible_truncation)]
        let t = queue.phase as f32;
        let sample = [
            (a[0] + (b[0] - a[0]) * t) * gain,
            (a[1] + (b[1] - a[1]) * t) * gain,
        ];
        for (channel, slot) in frame.iter_mut().enumerate() {
            *slot = sample[channel % 2];
        }
        queue.phase += step;
        while queue.phase >= 1.0 && !queue.frames.is_empty() {
            queue.phase -= 1.0;
            queue.frames.pop_front();
            PLAYED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Starts the output on a thread that keeps the stream alive. Does nothing when sound is off.
pub fn start() {
    if !enabled() {
        return;
    }
    if let Some(path) = std::env::var_os("ANYXBOX_AUDIO_OUT") {
        match std::fs::File::create(&path) {
            Ok(file) => {
                *FILE
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    Some(std::io::BufWriter::new(file));
            }
            Err(error) => eprintln!("anyxbox: {}: {error}", path.to_string_lossy()),
        }
    }
    if std::env::var_os("ANYXBOX_TRACE_AUDIO").is_some() {
        std::thread::spawn(|| {
            let start = Instant::now();
            loop {
                std::thread::sleep(Duration::from_secs(1));
                eprintln!(
                    "anyxbox: audio at {:.0} s: {} frames queued, {} played, ran dry {} times, \
                     {} waiting",
                    start.elapsed().as_secs_f64(),
                    QUEUED.swap(0, Ordering::Relaxed),
                    PLAYED.swap(0, Ordering::Relaxed),
                    DRY.swap(0, Ordering::Relaxed),
                    queue().frames.len()
                );
            }
        });
    }
    let spawned = std::thread::Builder::new()
        .name("anyxbox-audio".to_owned())
        .spawn(|| match open() {
            Ok(stream) => {
                // The stream plays while it exists, which is as long as the program.
                let _playing = stream;
                loop {
                    std::thread::park();
                }
            }
            Err(error) => eprintln!("anyxbox: no sound: {error}"),
        });
    if let Err(error) = spawned {
        eprintln!("anyxbox: no sound: {error}");
    }
}

/// Opens and starts the output stream on the default device, which must take `f32`.
fn open() -> Result<cpal::Stream, String> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("no output device")?;
    let config = device.default_output_config().map_err(|e| e.to_string())?;
    if config.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!(
            "the output takes {} samples, not f32",
            config.sample_format()
        ));
    }
    let channels = usize::from(config.channels());
    let rate = f64::from(config.sample_rate());
    let stream = device
        .build_output_stream(
            config.config(),
            move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let gain = f32::from_bits(GAIN.load(Ordering::Relaxed));
                fill(&mut queue(), out, channels, rate, gain);
            },
            |error| eprintln!("anyxbox: sound: {error}"),
            None,
        )
        .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue_of(frames: usize) -> Queue {
        Queue {
            frames: (0..frames)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let v = i as f32;
                    [v, -v]
                })
                .collect(),
            primed: false,
            phase: 0.0,
            prime: PRIME,
        }
    }

    #[test]
    fn the_bump_is_a_short_thud_that_dies_away() {
        let frames = thump();
        assert_eq!(frames.len(), 17_640, "0.4 s of sound");
        let peak = frames
            .iter()
            .take(4_410)
            .map(|f| f[0].abs())
            .fold(0.0, f32::max);
        let tail = frames
            .iter()
            .skip(13_000)
            .map(|f| f[0].abs())
            .fold(0.0, f32::max);
        let last = frames.last().map_or(1.0, |f| f[0].abs());
        assert!(peak > 0.5, "loud at first: {peak}");
        assert!(tail < 0.05, "faint by its last tenth of a second: {tail}");
        assert!(last < 0.001, "it ends in silence, without a click: {last}");
    }

    #[test]
    fn the_output_waits_for_100_ms_then_plays_left_and_right() {
        let mut queue = queue_of(PRIME - 1);
        let mut out = [1.0; 4];
        fill(&mut queue, &mut out, 2, RATE, 1.0);
        assert_eq!(out, [0.0; 4], "silent until primed");
        queue.frames.push_back([0.0; 2]);
        let mut out = [9.0; 6];
        fill(&mut queue, &mut out, 3, RATE, 1.0);
        assert_eq!(out, [0.0, -0.0, 0.0, 1.0, -1.0, 1.0]);
        assert_eq!(queue.frames.len(), PRIME - 2);
    }

    #[test]
    fn a_faster_device_interpolates_and_a_dry_queue_primes_again() {
        let mut queue = queue_of(PRIME);
        let mut out = [0.0; 6];
        fill(&mut queue, &mut out, 2, 2.0 * RATE, 1.0);
        assert_eq!(out, [0.0, 0.0, 0.5, -0.5, 1.0, -1.0]);
        queue.frames.truncate(1);
        let mut out = [1.0; 2];
        fill(&mut queue, &mut out, 2, RATE, 1.0);
        assert_eq!(out, [0.0; 2]);
        assert!(!queue.primed);
        assert_eq!(queue.prime, PRIME + PRIME_STEP, "it waits longer next time");
    }

    #[test]
    fn the_volume_scales_what_is_played() {
        let mut queue = queue_of(PRIME + 2);
        queue.frames[1] = [0.5, -1.0];
        let mut out = [0.0; 4];
        fill(&mut queue, &mut out, 2, RATE, 0.25);
        assert_eq!(out, [0.0, -0.0, 0.125, -0.25]);
    }
}
