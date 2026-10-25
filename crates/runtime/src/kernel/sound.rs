//! `DirectSound` replacements (spec 9.2, 9.3, 9.5): the objects and their vtables, voices
//! (buffers) and streams, their 3D settings, and the timeline that mixes them, completes stream
//! packets, and releases the mix to the audio output.
//!
//! Every replacement has the `Handler` signature, including those that cannot fail. Frame counts
//! become floats freely, as they are far below where precision is lost.
#![allow(clippy::unnecessary_wraps, clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

use std::collections::{HashMap, VecDeque};

use super::hooks::trap;
use super::mixer::{self, Frame, Listener, Source};
use super::time::BOOT_CLOCK;
use super::{Call, CallResult, Handler, KResult, Kernel};

/// The mixer's rate, in frames per second.
pub const RATE: u64 = 44_100;
/// The mix is released this far behind the clock (ticks).
const RELEASE_DELAY: u64 = 1_000_000;
/// The fewest frames the mix is released in.
const MIX_RUN: u64 = 64;
/// `E_OUTOFMEMORY`.
const OUT_OF_MEMORY: u32 = 0x8007_000E;
/// `XMEDIAPACKET_STATUS_PENDING`.
const PACKET_PENDING: u32 = 0x8000_000A;
/// A packet longer than this is ignored.
const MAX_PACKET: u32 = 1 << 20;
/// Slots in every vtable.
const VTABLE_SLOTS: u32 = 40;
/// `E_INVALIDARG`.
const INVALID_ARGUMENT: u32 = 0x8007_0057;
/// `DSBSTATUS_PLAYING`, `DSBSTATUS_PAUSED` and `DSBSTATUS_LOOPING`.
const STATUS_PLAYING: u32 = 1;
const STATUS_PAUSED: u32 = 2;
const STATUS_LOOPING: u32 = 4;
/// `DSBPAUSE_RESUME` and `DSBPAUSE_SYNCHPLAYBACK`, `Pause` modes (`DSBPAUSE_PAUSE` is 1).
const PAUSE_RESUME: u32 = 0;
const PAUSE_SYNCH_PLAYBACK: u32 = 2;
/// `DSBLOCK_ENTIREBUFFER`.
const LOCK_ENTIRE_BUFFER: u32 = 2;

/// A wave format: tag, channels, rate, bits per sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Format {
    pub tag: u16,
    pub channels: u16,
    pub rate: u32,
    pub bits: u16,
}

impl Format {
    /// Xbox ADPCM.
    const ADPCM: u16 = 0x69;
    const PCM: u16 = 1;

    /// Reads a `WAVEFORMATEX`.
    fn read(k: &Kernel, at: u32) -> KResult<Self> {
        Ok(Self {
            tag: k.r16(at)?,
            channels: k.r16(at + 2)?,
            rate: k.r32(at + 4)?,
            bits: k.r16(at + 14)?,
        })
    }

    /// Bytes of one frame (all channels) of PCM.
    fn frame_bytes(self) -> u32 {
        (u32::from(self.channels) * u32::from(self.bits) / 8).max(1)
    }

    /// The source frames in `bytes` of data.
    fn frames(self, bytes: u32) -> u64 {
        if self.tag == Self::ADPCM {
            // 36 bytes per channel per block, 65 samples each.
            let block = 36 * u64::from(self.channels.max(1));
            u64::from(bytes) / block * 65
        } else {
            u64::from(bytes / self.frame_bytes())
        }
    }

    /// The bytes of data `frames` source frames take, the inverse of `frames`: whole ADPCM
    /// blocks.
    fn bytes(self, frames: u64) -> u64 {
        if self.tag == Self::ADPCM {
            frames / 65 * 36 * u64::from(self.channels.max(1))
        } else {
            frames * u64::from(self.frame_bytes())
        }
    }
}

/// A voice: a buffer that plays from guest memory.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Voice {
    pub format: Format,
    pub data: u32,
    pub len: u32,
    /// The play region: start and length in bytes (0 is to the end).
    pub region: (u32, u32),
    pub frequency: u32,
    pub volume: i32,
    /// Hundredths of a decibel the voice plays below its volume (`SetHeadroom`).
    pub headroom: i32,
    pub looping: bool,
    pub playing: bool,
    /// How `Pause` holds the voice: `DSBPAUSE_PAUSE`, or `DSBPAUSE_SYNCHPLAYBACK` until
    /// `IDirectSound::SynchPlayback`; 0 when it is not paused.
    pub paused: u32,
    /// Source frames into the play region.
    pub position: f64,
    /// An ADPCM region decoded, its samples interleaved.
    decoded: Option<Vec<i16>>,
    /// Its 3D settings, and those waiting for the commit.
    pub source: Source,
    pending: Option<Source>,
}

impl Voice {
    /// The play region's start and length in bytes, within the data.
    fn region_bytes(&self) -> (u32, u32) {
        let start = self.region.0.min(self.len);
        let rest = self.len - start;
        let len = if self.region.1 == 0 {
            rest
        } else {
            self.region.1.min(rest)
        };
        (start, len)
    }

    /// The rate it plays at: its frequency, or its format's.
    fn rate(&self) -> u32 {
        if self.frequency == 0 {
            self.format.rate
        } else {
            self.frequency
        }
    }

    /// The source frames in the play region.
    fn region_frames(&self) -> u64 {
        match &self.decoded {
            Some(samples) => samples.len() as u64 / u64::from(self.format.channels.max(1)),
            None if self.format.tag == Format::ADPCM => 0,
            None => self.format.frames(self.region_bytes().1),
        }
    }

    /// Mixes `out.len()` frames of the voice into `out` from its position on, at its frequency
    /// times `doppler`, scaled by `gains`: looping over its region or stopping at its end.
    fn mix(&mut self, memory: &crate::mem::Memory, out: &mut [Frame], gains: Frame, doppler: f32) {
        let total = self.region_frames();
        if total == 0 {
            self.playing = false;
            return;
        }
        let step = f64::from(self.rate().max(1)) * f64::from(doppler) / RATE as f64;
        let samples = self.samples(memory, out.len(), step, total);
        for slot in out {
            // Positions are never negative.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let mut index = self.position as u64;
            if index >= total {
                if !self.looping {
                    self.playing = false;
                    return;
                }
                self.position %= total as f64;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let wrapped = self.position as u64;
                index = wrapped;
            }
            if let Some(frame) = samples.frame(index) {
                slot[0] += frame[0] * gains[0];
                slot[1] += frame[1] * gains[1];
            }
            self.position += step;
        }
    }

    /// The source frames the next `count` output frames can sample: the span from the position
    /// on and, when it loops past the region's end, the start it wraps to.
    fn samples(&self, memory: &crate::mem::Memory, count: usize, step: f64, total: u64) -> Samples {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (first, span) = (
            self.position as u64,
            (count as f64 * step).ceil() as u64 + 1,
        );
        if span >= total {
            return Samples {
                first: 0,
                frames: self.frames(memory, 0, total),
                head: Vec::new(),
            };
        }
        let first = first.min(total);
        let wrapped = if self.looping {
            (first + span).saturating_sub(total)
        } else {
            0
        };
        Samples {
            first,
            frames: self.frames(memory, first, (first + span).min(total)),
            head: self.frames(memory, 0, wrapped),
        }
    }

    /// Source frames `low..high` of the region, as stereo.
    fn frames(&self, memory: &crate::mem::Memory, low: u64, high: u64) -> Vec<Frame> {
        if high <= low {
            return Vec::new();
        }
        let channels = self.format.channels;
        if let Some(decoded) = &self.decoded {
            let width = usize::from(channels.max(1));
            let (low, high) = (
                usize::try_from(low).unwrap_or(usize::MAX),
                usize::try_from(high).unwrap_or(usize::MAX),
            );
            return decoded
                .get(low * width..high * width)
                .unwrap_or_default()
                .chunks_exact(width)
                .map(|frame| {
                    let left = f32::from(frame[0]);
                    [left, frame.get(1).map_or(left, |&r| f32::from(r))]
                })
                .collect();
        }
        let frame_bytes = u64::from(self.format.frame_bytes());
        let (start, _) = self.region_bytes();
        let mut bytes = vec![0; usize::try_from((high - low) * frame_bytes).unwrap_or(0)];
        let at = u64::from(self.data) + u64::from(start) + low * frame_bytes;
        u32::try_from(at)
            .ok()
            .filter(|&at| memory.read(at, &mut bytes))
            .and_then(|_| mixer::pcm_frames(&bytes, channels, self.format.bits))
            .unwrap_or_default()
    }
}

/// Source frames from frame `first` of a voice's region, and the start it wraps to.
struct Samples {
    first: u64,
    frames: Vec<Frame>,
    /// Frames from the region's start, for a voice that loops back to it.
    head: Vec<Frame>,
}

impl Samples {
    fn frame(&self, index: u64) -> Option<Frame> {
        if index < self.first {
            return self.head.get(usize::try_from(index).ok()?).copied();
        }
        let at = usize::try_from(index - self.first).ok()?;
        self.frames.get(at).copied()
    }
}

/// A packet queued on a stream: where its completion is written, and the frame it ends at.
#[derive(Debug, Serialize, Deserialize)]
struct Packet {
    length: u32,
    completed: u32,
    status: u32,
    event: u32,
    end: u64,
}

/// A stream: its format, the packets it holds, and the next free frame of its timeline.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Stream {
    pub format: Format,
    pub max_packets: u32,
    pub volume: i32,
    /// Each channel's stereo gains.
    pub routes: Vec<Frame>,
    packets: VecDeque<Packet>,
    next: u64,
}

/// The sound state: objects by guest address, the listener, and the mix.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Sound {
    pub voices: HashMap<u32, Voice>,
    pub streams: HashMap<u32, Stream>,
    pub listener: Listener,
    pending_listener: Option<Listener>,
    timeline: mixer::Timeline,
}

/// The timeline frame at clock tick `t`.
fn frame(t: u64) -> u64 {
    u128::from(t.saturating_sub(BOOT_CLOCK))
        .saturating_mul(u128::from(RATE))
        .checked_div(10_000_000)
        .and_then(|f| u64::try_from(f).ok())
        .unwrap_or(u64::MAX)
}

impl Kernel {
    /// Makes a COM object of `class`: an 8-byte heap block whose first dword is the class's
    /// vtable, built once (spec 9.2).
    fn new_object(&mut self, class: &'static str) -> KResult<Option<u32>> {
        let vtable = if let Some(&vtable) = self.hooks.vtables.get(class) {
            vtable
        } else {
            let vtable = self.heap.alloc(4 * VTABLE_SLOTS);
            if vtable == 0 {
                return Ok(None);
            }
            for slot in 0..VTABLE_SLOTS {
                let index = self.hooks.plan.len() + self.hooks.slots.len();
                if index >= super::hooks::MAX_HOOKS {
                    return Ok(None);
                }
                self.hooks.slots.push((class, slot));
                self.w32(vtable + 4 * slot, trap(index))?;
            }
            self.hooks.vtables.insert(class, vtable);
            vtable
        };
        let object = self.heap.alloc(8);
        if object == 0 {
            return Ok(None);
        }
        self.w32(object, vtable)?;
        Ok(Some(object))
    }

    /// Releases the timeline up to 100 ms behind now (spec 9.3): while a voice plays, every due
    /// frame, mixed with the voices, else what the streams have queued; with nothing queued the
    /// timeline follows the clock. The frames go through the limiter to the audio output. Then
    /// completes stream packets whose last frame has been released, writing their completed size
    /// and status and setting their event.
    pub fn service_sound(&mut self) {
        let now = self.clock.now();
        let released = frame(now.saturating_sub(RELEASE_DELAY));
        let from = self.sound.timeline.released;
        // Time is serviced on every kernel contact: the mix is released in runs of at least
        // 1.5 ms, well inside the 100 ms the output keeps queued.
        if released >= from + MIX_RUN {
            let voices = self.sound.voices.values().any(|v| v.playing);
            let queued = self
                .sound
                .streams
                .values()
                .map(|s| s.next)
                .max()
                .unwrap_or(0);
            let end = if voices {
                released
            } else {
                released.min(queued.max(from))
            };
            if end > from {
                let mut frames = self.sound.timeline.take(end);
                let listener = self.sound.listener;
                for voice in self.sound.voices.values_mut().filter(|v| v.playing) {
                    let [left, right] = voice.source.gains(&listener, voice.format.channels == 2);
                    let volume = mixer::gain(voice.volume.saturating_sub(voice.headroom));
                    let doppler = voice.source.doppler(&listener);
                    voice.mix(
                        &self.memory,
                        &mut frames,
                        [left * volume, right * volume],
                        doppler,
                    );
                }
                let pcm: Vec<[i16; 2]> = frames
                    .iter()
                    .map(|&[l, r]| [mixer::limit(l), mixer::limit(r)])
                    .collect();
                crate::host::audio::push(&pcm);
            }
            self.sound.timeline.skip_to(released);
        }
        let mut done = Vec::new();
        for stream in self.sound.streams.values_mut() {
            while stream.packets.front().is_some_and(|p| p.end <= released) {
                done.extend(stream.packets.pop_front());
            }
        }
        for packet in done {
            let _ = self.w32_opt(packet.completed, packet.length);
            let _ = self.w32_opt(packet.status, 0);
            if let Some(id) = self.object_of_handle(packet.event) {
                self.set_event(id);
            }
        }
    }
}

/// `DirectSoundCreate(guid, **ds, unknown)`.
fn create(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(object) = k.new_object("IDirectSound")? else {
        return Ok(CallResult::Value(OUT_OF_MEMORY));
    };
    k.w32(c.arg(1), object)?;
    Ok(CallResult::Value(0))
}

/// `DirectSoundCreateBuffer(*desc, **buffer)`.
fn create_buffer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    new_buffer(k, c.arg(0), c.arg(1))
}

/// `IDirectSound::CreateSoundBuffer(ds, *desc, **buffer, outer)`: the same, one argument later.
fn sound_create_buffer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    new_buffer(k, c.arg(1), c.arg(2))
}

/// Makes a buffer from the `DSBUFFERDESC` at `desc` and stores it through `out`: `+8` is its size
/// in bytes, `+12` the `WAVEFORMATEX*`. A buffer made with a size owns that much zeroed memory,
/// which the title fills through `Lock`; one made with none gets its data from `SetBufferData`.
fn new_buffer(k: &mut Kernel, desc: u32, out: u32) -> KResult<CallResult> {
    let bytes = k.r32(desc + 8)?;
    let data = if bytes == 0 { 0 } else { k.heap.alloc(bytes) };
    if bytes != 0 && data == 0 {
        return Ok(CallResult::Value(OUT_OF_MEMORY));
    }
    let Some(object) = k.new_object("IDirectSoundBuffer")? else {
        return Ok(CallResult::Value(OUT_OF_MEMORY));
    };
    let wave = k.r32(desc + 12)?;
    let mut format = if wave == 0 {
        Format::default()
    } else {
        Format::read(k, wave)?
    };
    if format.tag == Format::PCM && format.channels != 1 && data == 0 {
        // A PCM buffer is mono until `SetFormat` says otherwise; one with memory of its own plays
        // what the title locks and writes into it, in the format it was made with.
        format.channels = 0;
    }
    k.sound.voices.insert(
        object,
        Voice {
            format,
            data,
            len: bytes,
            ..Voice::default()
        },
    );
    k.w32(out, object)?;
    Ok(CallResult::Value(0))
}

/// `DSound_TimerObjectCreate(0, count, **object)`, our name for an XDK 5028 function: an object
/// told the time that makes the objects sounds play through. None of them plays anything yet:
/// each method only takes its arguments off the stack, and the two that make objects make them.
fn create_timer_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    new_into(k, "DSoundTimerObject", c.arg(2))
}

/// A getter whose out pointer is argument 1 (after `this`), answering 0.
fn write_zero_1(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w32(c.arg(1), 0)?;
    Ok(CallResult::Value(0))
}

/// A getter whose out pointer is argument 2, answering 0.
fn write_zero_2(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w32(c.arg(2), 0)?;
    Ok(CallResult::Value(0))
}

/// Makes an object of `class` and writes it to `out`.
fn new_into(k: &mut Kernel, class: &'static str, out: u32) -> KResult<CallResult> {
    let Some(object) = k.new_object(class)? else {
        return Ok(CallResult::Value(OUT_OF_MEMORY));
    };
    k.w32(out, object)?;
    Ok(CallResult::Value(0))
}

/// Method 5 of the timer object, `(this, *desc, **object)`: a sound described like a buffer.
fn timer_new_voice(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    new_into(k, "DSoundTimerVoice", c.arg(2))
}

/// Method 6 of the timer object, `(this, *desc, **object)`: a sound of the other kind.
fn timer_new_stream(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    new_into(k, "DSoundTimerStream", c.arg(2))
}

/// The bytes each method of the timer object and of the two kinds of object it makes takes off
/// the stack, `this` included, read from their `ret` in Marvel vs. Capcom 2; slots 0 and 1 are
/// `AddRef` and `Release`.
const TIMER_OBJECT_POPS: [u16; 14] = [4, 4, 8, 12, 20, 12, 12, 12, 12, 12, 4, 12, 8, 8];
const TIMER_VOICE_POPS: [u16; 39] = [
    4, 4, 4, 12, 8, 8, 8, 8, 8, 4, 24, 12, 16, 20, 20, 16, 20, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 20, 24, 28, 20, 20, 20, 20, 28, 28, 20, 12, 8,
];
const TIMER_STREAM_POPS: [u16; 36] = [
    4, 4, 4, 12, 8, 8, 12, 8, 8, 8, 4, 20, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20,
    24, 28, 20, 20, 20, 20, 28, 28, 12, 8,
];

/// A method of the timer object or of an object it made: `AddRef`, `Release`, the two that make
/// objects, and the rest, which do nothing.
fn timer_slot(class: &str, slot: u32) -> Option<(&'static str, Handler, u16)> {
    /// `CLASS::slotN` for every slot, as the call log names them.
    macro_rules! names {
        ($class:literal) => {
            names!($class; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
                27 28 29 30 31 32 33 34 35 36 37 38 39)
        };
        ($class:literal; $($n:literal)*) => {
            [$(concat!($class, "::slot", stringify!($n))),*]
        };
    }
    const OBJECT: [&str; 40] = names!("DSoundTimerObject");
    const VOICE: [&str; 40] = names!("DSoundTimerVoice");
    const STREAM: [&str; 40] = names!("DSoundTimerStream");
    let (pops, names): (&[u16], &[&'static str; 40]) = match class {
        "DSoundTimerObject" => (&TIMER_OBJECT_POPS, &OBJECT),
        "DSoundTimerVoice" => (&TIMER_VOICE_POPS, &VOICE),
        "DSoundTimerStream" => (&TIMER_STREAM_POPS, &STREAM),
        _ => return None,
    };
    let pops = *pops.get(slot as usize)?;
    let name = names[slot as usize];
    Some(match (class, slot) {
        (_, 0 | 1) => (name, one as Handler, pops),
        ("DSoundTimerObject", 5) => ("DSoundTimerObject::NewVoice", timer_new_voice, pops),
        ("DSoundTimerObject", 6) => ("DSoundTimerObject::NewStream", timer_new_stream, pops),
        // `GetStatus(this, *status)`: nothing plays.
        ("DSoundTimerVoice", 5) => ("DSoundTimerVoice::GetStatus", write_zero_1, pops),
        // `(this, index, *value)`: a value the object keeps for each index, none set here.
        ("DSoundTimerStream", 6) => ("DSoundTimerStream::slot6", write_zero_2, pops),
        _ => (name, ok, pops),
    })
}

/// `DirectSoundCreateStream(*desc, **stream)`: `DSSTREAMDESC +4` maximum packets, `+8` the
/// `WAVEFORMATEX*`.
fn create_stream(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(object) = k.new_object("IDirectSoundStream")? else {
        return Ok(CallResult::Value(OUT_OF_MEMORY));
    };
    let max_packets = k.r32(c.arg(0) + 4)?;
    let wave = k.r32(c.arg(0) + 8)?;
    let format = if wave == 0 {
        Format::default()
    } else {
        Format::read(k, wave)?
    };
    let mut routes = mixer::default_routes(format.channels);
    // `DSMIXBINS`: a count and `(bin, volume)` pairs; one pair a channel replaces the routes.
    let bins = k.r32(c.arg(0) + 20)?;
    if bins != 0 {
        let count = k.r32(bins)?;
        let pairs = k.r32(bins + 4)?;
        if count == u32::from(format.channels) && count > 0 && pairs != 0 {
            routes = (0..count)
                .map(|i| {
                    let bin = k.r32(pairs + 8 * i)?;
                    let gain = mixer::gain(k.r32(pairs + 8 * i + 4)?.cast_signed());
                    let [l, r] = mixer::bin_route(bin);
                    Ok([l * gain, r * gain])
                })
                .collect::<KResult<_>>()?;
        }
    }
    k.sound.streams.insert(
        object,
        Stream {
            format,
            max_packets,
            routes,
            ..Stream::default()
        },
    );
    k.w32(c.arg(1), object)?;
    Ok(CallResult::Value(0))
}

fn ok(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    Ok(CallResult::Value(0))
}

fn one(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    Ok(CallResult::Value(1))
}

/// Applies `f` to the voice at `c.arg(0)`, if it is one.
fn with_voice(k: &mut Kernel, c: &Call, f: impl FnOnce(&mut Voice, u64)) -> KResult<CallResult> {
    let now = k.clock.now();
    if let Some(voice) = k.sound.voices.get_mut(&c.arg(0)) {
        f(voice, now);
    }
    Ok(CallResult::Value(0))
}

/// `SetBufferData(buffer, data, length)`: resets the region, position and decoded data, and
/// stops.
fn set_buffer_data(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    with_voice(k, c, |v, _| {
        v.data = c.arg(1);
        v.len = c.arg(2);
        v.region = (0, 0);
        v.position = 0.0;
        v.decoded = None;
        v.playing = false;
    })
}

/// `SetPlayRegion(buffer, start, length)`: bytes, 0 is to the end; the position resets.
fn set_play_region(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    with_voice(k, c, |v, _| {
        v.region = (c.arg(1), c.arg(2));
        v.position = 0.0;
        v.decoded = None;
    })
}

/// `Play(buffer, reserved, reserved, flags)`: `DSBPLAY_LOOPING` (1), `DSBPLAY_FROMSTART` (2).
/// An ADPCM region is decoded the first time it plays.
fn play(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(voice) = k.sound.voices.get(&c.arg(0)) else {
        return Ok(CallResult::Value(0));
    };
    let decode = (voice.format.tag == Format::ADPCM && voice.decoded.is_none()).then(|| {
        let (start, len) = voice.region_bytes();
        (voice.data.wrapping_add(start), len, voice.format.channels)
    });
    let decoded = decode.map(|(at, len, channels)| {
        k.read(at, len)
            .map(|bytes| mixer::decode_adpcm(&bytes, channels))
            .unwrap_or_default()
    });
    with_voice(k, c, |v, _| {
        if decoded.is_some() {
            v.decoded = decoded;
        }
        let flags = c.arg(3);
        if flags & 2 != 0 || !v.playing {
            v.position = 0.0;
        }
        v.looping = flags & 1 != 0;
        v.playing = true;
        v.paused = 0;
    })
}

fn stop(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    with_voice(k, c, |v, _| {
        v.playing = false;
        v.paused = 0;
    })
}

/// `GetStatus(buffer, *status)`: playing or paused, and looping, or 0 when neither.
fn buffer_status(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let status = k.sound.voices.get(&c.arg(0)).map_or(0, |v| {
        let state = if v.playing {
            STATUS_PLAYING
        } else if v.paused != 0 {
            STATUS_PAUSED
        } else {
            return 0;
        };
        if v.looping {
            state | STATUS_LOOPING
        } else {
            state
        }
    });
    k.w32(c.arg(1), status)?;
    Ok(CallResult::Value(0))
}

/// `Pause(buffer, mode)`: `DSBPAUSE_PAUSE` holds a playing voice where it is,
/// `DSBPAUSE_SYNCHPLAYBACK` holds it until `IDirectSound::SynchPlayback`, and `DSBPAUSE_RESUME`
/// lets a held voice go on.
fn pause(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mode = c.arg(1);
    with_voice(k, c, |v, _| {
        if mode == PAUSE_RESUME {
            if v.paused != 0 {
                v.paused = 0;
                v.playing = true;
            }
        } else if v.playing || v.paused != 0 {
            v.playing = false;
            v.paused = mode;
        }
    })
}

/// `IDirectSound::SynchPlayback(ds)`: the voices paused for it start together.
fn synch_playback(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    for voice in k.sound.voices.values_mut() {
        if voice.paused == PAUSE_SYNCH_PLAYBACK {
            voice.paused = 0;
            voice.playing = true;
        }
    }
    Ok(CallResult::Value(0))
}

/// `Lock(buffer, offset, bytes, **first, *first_bytes, **second, *second_bytes, flags)`: the
/// title writes its sound through the pointers. The span is one part when it fits before the
/// buffer's end, else it wraps to the start as the second part. `DSBLOCK_ENTIREBUFFER` locks the
/// whole buffer whatever the offset and length. Null out pointers are skipped.
fn lock(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(voice) = k.sound.voices.get(&c.arg(0)) else {
        return Ok(CallResult::Value(INVALID_ARGUMENT));
    };
    let (data, len) = (voice.data, voice.len);
    let (mut offset, mut bytes) = (c.arg(1), c.arg(2));
    if c.arg(7) & LOCK_ENTIRE_BUFFER != 0 {
        (offset, bytes) = (0, len);
    }
    if offset > len || bytes > len {
        return Ok(CallResult::Value(INVALID_ARGUMENT));
    }
    let first = bytes.min(len - offset);
    let second = bytes - first;
    for (pointer, value) in [
        (c.arg(3), data.wrapping_add(offset)),
        (c.arg(4), first),
        (c.arg(5), if second == 0 { 0 } else { data }),
        (c.arg(6), second),
    ] {
        k.w32_opt(pointer, value)?;
    }
    Ok(CallResult::Value(0))
}

/// `GetCurrentPosition(buffer, *play, *write)`, bytes into the buffer: the play cursor where the
/// voice is, the write cursor 10 ms (on a whole frame) ahead of it, both wrapping at the end. A
/// title that streams into the buffer reads them to know how much it may write.
fn current_position(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(voice) = k.sound.voices.get(&c.arg(0)) else {
        return Ok(CallResult::Value(INVALID_ARGUMENT));
    };
    let len = u64::from(voice.len.max(1));
    let format = voice.format;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let frames = voice.position.max(0.0) as u64;
    let start = u64::from(voice.region_bytes().0);
    let play = (start + format.bytes(frames)) % len;
    let lead = u64::from(voice.rate() / 100) * u64::from(format.frame_bytes());
    let write = (play + lead) % len;
    // Both are below the buffer's length, a u32.
    let (play, write) = (
        u32::try_from(play).unwrap_or(0),
        u32::try_from(write).unwrap_or(0),
    );
    k.w32_opt(c.arg(1), play)?;
    k.w32_opt(c.arg(2), write)?;
    Ok(CallResult::Value(0))
}

/// `SetCurrentPosition(buffer, offset)`: the play cursor moves to `offset` bytes into the buffer,
/// the inverse of what `GetCurrentPosition` reports.
fn set_current_position(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let offset = c.arg(1);
    with_voice(k, c, |v, _| {
        let into = offset.min(v.len).saturating_sub(v.region_bytes().0);
        v.position = v.format.frames(into) as f64;
    })
}

/// `SetPitch(buffer, pitch)`: a frequency as a pitch, 4096 to the octave from 48 kHz.
fn set_pitch(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let pitch = f64::from(c.arg(1).cast_signed());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let frequency = (48_000.0 * (pitch / 4096.0).exp2()).round() as u32;
    with_voice(k, c, |v, _| v.frequency = frequency)
}

/// `SetHeadroom(buffer, headroom)`: hundredths of a decibel below the volume, at most 10,000.
fn set_headroom(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let headroom = i32::try_from(c.arg(1).min(10_000)).unwrap_or(0);
    with_voice(k, c, |v, _| v.headroom = headroom)
}

fn set_volume(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let volume = c.arg(1).cast_signed();
    if let Some(stream) = k.sound.streams.get_mut(&c.arg(0)) {
        stream.volume = volume;
    }
    with_voice(k, c, |v, _| v.volume = volume)
}

/// `SetFrequency(buffer, hz)`: 0 keeps the current rate.
fn set_frequency(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    with_voice(k, c, |v, _| {
        if c.arg(1) != 0 {
            v.frequency = c.arg(1);
        }
    })
}

/// `SetFormat(buffer, *wfx)`: PCM or ADPCM keep their channels, other tags are silent. The voice
/// plays at the format's rate until a later `SetFrequency`, and an ADPCM region is decoded again.
fn set_format(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mut format = Format::read(k, c.arg(1))?;
    if format.tag != Format::PCM && format.tag != Format::ADPCM {
        format.channels = 0;
    }
    with_voice(k, c, |v, _| {
        v.format = format;
        v.frequency = 0;
        v.decoded = None;
    })
}

/// `IDirectSoundStream::GetInfo(stream, *info)`: an `XMEDIAINFO`.
fn stream_info(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let frame_bytes = k
        .sound
        .streams
        .get(&c.arg(0))
        .map_or(4, |s| s.format.frame_bytes());
    let info = [5, frame_bytes, 0, 32 * frame_bytes];
    for (i, v) in info.iter().enumerate() {
        k.w32(c.arg(1) + 4 * u32::try_from(i).unwrap_or(0), *v)?;
    }
    Ok(CallResult::Value(0))
}

/// `IDirectSoundStream::GetStatus(stream, *status)`: ready while packets are below the maximum.
fn stream_status(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let ready = k
        .sound
        .streams
        .get(&c.arg(0))
        .is_some_and(|s| s.max_packets == 0 || s.packets.len() < s.max_packets as usize);
    k.w32(c.arg(1), u32::from(ready))?;
    Ok(CallResult::Value(0))
}

/// `IDirectSoundStream::Process(stream, *packet, *out)`: queues the packet's PCM on the stream's
/// timeline; it completes when its last frame has been released.
fn stream_process(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let packet = c.arg(1);
    let Ok(fields) = (0..5)
        .map(|i| k.r32(packet + 4 * i))
        .collect::<KResult<Vec<u32>>>()
    else {
        return Ok(CallResult::Value(0));
    };
    let (length, completed, status, event) = (fields[1], fields[2], fields[3], fields[4]);
    if length == 0 || length > MAX_PACKET {
        return Ok(CallResult::Value(0));
    }
    let Ok(data) = k.read(fields[0], length) else {
        return Ok(CallResult::Value(0));
    };
    k.w32_opt(status, PACKET_PENDING)?;
    let now = k.clock.now();
    let Some(stream) = k.sound.streams.get_mut(&c.arg(0)) else {
        return Ok(CallResult::Value(0));
    };
    let rate = u64::from(stream.format.rate.max(1));
    // The PCM as stereo at 44.1 kHz, scaled and routed; a format that cannot play is silent but
    // still takes its time.
    let mixed = stream_frames(stream, &data);
    let frames = if mixed.is_empty() {
        stream.format.frames(length) * RATE / rate
    } else {
        mixed.len() as u64
    };
    // A stream continues right after its last packet while that is still ahead of the release
    // point; one that starts or ran dry starts 100 ms from now.
    let released = frame(now.saturating_sub(RELEASE_DELAY));
    let start = if stream.next > released {
        stream.next
    } else {
        frame(now + RELEASE_DELAY).max(released)
    };
    stream.next = start + frames;
    k.sound.timeline.add(start, &mixed);
    let Some(stream) = k.sound.streams.get_mut(&c.arg(0)) else {
        return Ok(CallResult::Value(0));
    };
    stream.packets.push_back(Packet {
        length,
        completed,
        status,
        event,
        end: stream.next,
    });
    Ok(CallResult::Value(0))
}

/// A packet's PCM in `stream`'s format as frames to add to the timeline: stereo at 44.1 kHz,
/// scaled by the stream's volume and routed; empty when the format cannot play.
fn stream_frames(stream: &Stream, data: &[u8]) -> Vec<Frame> {
    let format = stream.format;
    if format.tag != Format::PCM {
        return Vec::new();
    }
    let Some(frames) = mixer::pcm_frames(data, format.channels, format.bits) else {
        return Vec::new();
    };
    let gain = mixer::gain(stream.volume);
    let routes = &stream.routes;
    mixer::resample(&frames, format.rate)
        .into_iter()
        .map(|[left, right]| {
            let channels = if routes.len() == 1 {
                [left, 0.0]
            } else {
                [left, right]
            };
            let mut out = [0.0; 2];
            for (sample, route) in channels.iter().zip(routes) {
                out[0] += sample * route[0] * gain;
                out[1] += sample * route[1] * gain;
            }
            out
        })
        .collect()
}

/// A float argument, passed as its bits.
fn float(c: &Call, n: usize) -> f32 {
    f32::from_bits(c.arg(n))
}

/// `DS3D_DEFERRED`: the setting waits for `CommitDeferredSettings`.
const DEFERRED: u32 = 1;

/// Changes the 3D settings of the voice at `c.arg(0)` with `f`, now or, when argument `apply`
/// is `DS3D_DEFERRED`, at the commit.
fn with_source(
    k: &mut Kernel,
    c: &Call,
    apply: usize,
    f: impl FnOnce(&mut Source),
) -> KResult<CallResult> {
    let deferred = c.arg(apply) == DEFERRED;
    with_voice(k, c, |v, _| {
        if deferred {
            let current = v.source;
            f(v.pending.get_or_insert(current));
        } else {
            f(&mut v.source);
        }
    })
}

/// Changes the listener with `f`, now or at the commit.
fn with_listener(
    k: &mut Kernel,
    c: &Call,
    apply: usize,
    f: impl FnOnce(&mut Listener),
) -> KResult<CallResult> {
    let sound = &mut k.sound;
    if c.arg(apply) == DEFERRED {
        let current = sound.listener;
        f(sound.pending_listener.get_or_insert(current));
    } else {
        f(&mut sound.listener);
    }
    Ok(CallResult::Value(0))
}

/// `IDirectSoundBuffer::SetPosition(buffer, x, y, z, apply)`.
fn source_position(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let position = [float(c, 1), float(c, 2), float(c, 3)];
    with_source(k, c, 4, |s| s.position = position)
}

fn min_distance(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let distance = float(c, 1);
    with_source(k, c, 2, |s| s.min_distance = distance)
}

fn max_distance(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let distance = float(c, 1);
    with_source(k, c, 2, |s| s.max_distance = distance)
}

fn source_rolloff(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let rolloff = float(c, 1);
    with_source(k, c, 2, |s| s.rolloff = rolloff)
}

fn mode(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mode = c.arg(1);
    with_source(k, c, 2, |s| s.mode = mode)
}

/// `IDirectSoundBuffer::SetVelocity(buffer, x, y, z, apply)`.
fn source_velocity(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let velocity = [float(c, 1), float(c, 2), float(c, 3)];
    with_source(k, c, 4, |s| s.velocity = velocity)
}

/// `SetConeAngles(buffer, inside, outside, apply)`: degrees.
fn cone_angles(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (inside, outside) = (c.arg(1), c.arg(2));
    with_source(k, c, 3, |s| {
        s.inside_cone = inside;
        s.outside_cone = outside;
    })
}

/// `SetConeOrientation(buffer, x, y, z, apply)`.
fn cone_orientation(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let orientation = [float(c, 1), float(c, 2), float(c, 3)];
    with_source(k, c, 4, |s| s.cone_orientation = orientation)
}

/// `SetConeOutsideVolume(buffer, volume, apply)`: hundredths of a dB.
fn cone_outside_volume(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let volume = c.arg(1).cast_signed();
    with_source(k, c, 2, |s| s.cone_outside_volume = volume)
}

fn source_distance_factor(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let factor = float(c, 1);
    with_source(k, c, 2, |s| s.distance_factor = factor)
}

fn source_doppler_factor(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let factor = float(c, 1);
    with_source(k, c, 2, |s| s.doppler_factor = factor)
}

/// Three floats at `at`.
fn vector(k: &Kernel, at: u32) -> KResult<[f32; 3]> {
    Ok([
        f32::from_bits(k.r32(at)?),
        f32::from_bits(k.r32(at + 4)?),
        f32::from_bits(k.r32(at + 8)?),
    ])
}

/// `IDirectSoundBuffer::SetAllParameters(buffer, *DS3DBUFFER, apply)`, in the public layout:
/// size, position, velocity, inside and outside cone angles, cone orientation, cone outside
/// volume, minimum and maximum distance, mode.
fn source_all(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(1);
    let read = Source {
        position: vector(k, at + 4)?,
        velocity: vector(k, at + 16)?,
        inside_cone: k.r32(at + 28)?,
        outside_cone: k.r32(at + 32)?,
        cone_orientation: vector(k, at + 36)?,
        cone_outside_volume: k.r32(at + 48)?.cast_signed(),
        min_distance: f32::from_bits(k.r32(at + 52)?),
        max_distance: f32::from_bits(k.r32(at + 56)?),
        mode: k.r32(at + 60)?,
        ..Source::default()
    };
    with_source(k, c, 2, |s| {
        *s = Source {
            rolloff: s.rolloff,
            distance_factor: s.distance_factor,
            doppler_factor: s.doppler_factor,
            ..read
        };
    })
}

/// `IDirectSound::SetVelocity(ds, x, y, z, apply)`.
fn listener_velocity(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let velocity = [float(c, 1), float(c, 2), float(c, 3)];
    with_listener(k, c, 4, |l| l.velocity = velocity)
}

fn listener_distance_factor(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let factor = float(c, 1);
    with_listener(k, c, 2, |l| l.distance_factor = factor)
}

fn listener_doppler_factor(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let factor = float(c, 1);
    with_listener(k, c, 2, |l| l.doppler_factor = factor)
}

/// `IDirectSound::SetAllParameters(ds, *DS3DLISTENER, apply)`, in the public layout: size,
/// position, velocity, front, top, distance, rolloff and Doppler factors.
fn listener_all(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(1);
    let read = Listener {
        position: vector(k, at + 4)?,
        velocity: vector(k, at + 16)?,
        front: vector(k, at + 28)?,
        top: vector(k, at + 40)?,
        distance_factor: f32::from_bits(k.r32(at + 52)?),
        rolloff: f32::from_bits(k.r32(at + 56)?),
        doppler_factor: f32::from_bits(k.r32(at + 60)?),
    };
    with_listener(k, c, 2, |l| *l = read)
}

/// `IDirectSound::SetPosition(ds, x, y, z, apply)`.
fn listener_position(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let position = [float(c, 1), float(c, 2), float(c, 3)];
    with_listener(k, c, 4, |l| l.position = position)
}

/// `IDirectSound::SetOrientation(ds, front xyz, top xyz, apply)`: an all-zero or non-finite
/// direction keeps the old orientation.
fn orientation(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let front = [float(c, 1), float(c, 2), float(c, 3)];
    let top = [float(c, 4), float(c, 5), float(c, 6)];
    let usable = |v: [f32; 3]| v.iter().all(|x| x.is_finite()) && v.iter().any(|&x| x != 0.0);
    if !usable(front) || !usable(top) {
        return Ok(CallResult::Value(0));
    }
    with_listener(k, c, 7, |l| {
        l.front = front;
        l.top = top;
    })
}

fn listener_rolloff(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let rolloff = float(c, 1);
    with_listener(k, c, 2, |l| l.rolloff = rolloff)
}

/// `IDirectSound::CommitDeferredSettings`: the deferred settings take effect.
fn commit(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    let sound = &mut k.sound;
    if let Some(listener) = sound.pending_listener.take() {
        sound.listener = listener;
    }
    for voice in sound.voices.values_mut() {
        if let Some(source) = voice.pending.take() {
            voice.source = source;
        }
    }
    Ok(CallResult::Value(0))
}

/// The replacement of a hooked `DirectSound` function, by name.
#[must_use]
pub fn handler(name: &str) -> Option<Handler> {
    let handler: Handler = match name {
        "DirectSoundCreate" => create,
        "DirectSoundCreateBuffer" => create_buffer,
        "IDirectSound::CreateSoundBuffer" => sound_create_buffer,
        "DirectSoundCreateStream" => create_stream,
        "DSound_TimerObjectCreate" => create_timer_object,
        "IDirectSoundBuffer::SetBufferData" => set_buffer_data,
        "IDirectSoundBuffer::SetPlayRegion" => set_play_region,
        // `PlayEx(buffer, timestamp, flags)` has `Play`'s stack layout; the timestamp (0 is
        // "now") is not modelled.
        "IDirectSoundBuffer::Play" | "IDirectSoundBuffer::PlayEx" => play,
        "IDirectSoundBuffer::Stop" => stop,
        "IDirectSoundBuffer::GetStatus" => buffer_status,
        "IDirectSoundBuffer::Pause" => pause,
        "IDirectSoundBuffer::Lock" => lock,
        "IDirectSoundBuffer::GetCurrentPosition" => current_position,
        "IDirectSoundBuffer::SetCurrentPosition" => set_current_position,
        "IDirectSoundBuffer::SetPitch" => set_pitch,
        "IDirectSoundBuffer::SetHeadroom" => set_headroom,
        "IDirectSoundBuffer::SetVolume" | "IDirectSoundStream::SetVolume" => set_volume,
        "IDirectSoundBuffer::SetFrequency" => set_frequency,
        "IDirectSoundBuffer::SetFormat" => set_format,
        "IDirectSound::SetPosition" => listener_position,
        "IDirectSound::SetOrientation" => orientation,
        "IDirectSound::SetRolloffFactor" => listener_rolloff,
        "IDirectSound::CommitDeferredSettings" => commit,
        "IDirectSound::SynchPlayback" => synch_playback,
        "IDirectSoundBuffer::SetPosition" => source_position,
        "IDirectSoundBuffer::SetMinDistance" => min_distance,
        "IDirectSoundBuffer::SetMaxDistance" => max_distance,
        "IDirectSoundBuffer::SetRolloffFactor" => source_rolloff,
        "IDirectSoundBuffer::SetMode" => mode,
        "IDirectSoundBuffer::SetVelocity" => source_velocity,
        "IDirectSoundBuffer::SetConeAngles" => cone_angles,
        "IDirectSoundBuffer::SetConeOrientation" => cone_orientation,
        "IDirectSoundBuffer::SetConeOutsideVolume" => cone_outside_volume,
        "IDirectSoundBuffer::SetDistanceFactor" => source_distance_factor,
        "IDirectSoundBuffer::SetDopplerFactor" => source_doppler_factor,
        "IDirectSoundBuffer::SetAllParameters" => source_all,
        "IDirectSound::SetVelocity" => listener_velocity,
        "IDirectSound::SetDistanceFactor" => listener_distance_factor,
        "IDirectSound::SetDopplerFactor" => listener_doppler_factor,
        "IDirectSound::SetAllParameters" => listener_all,
        // Accepted and ignored: there is no DSP, reverb or headphone virtualisation, loop
        // regions are not modelled (spec 9.2), and a buffer's voice is not routed through mix
        // bins: it plays on both front speakers.
        "IDirectSound::DownloadEffectsImage"
        | "IDirectSoundBuffer::SetLoopRegion"
        | "IDirectSound::SetMixBinHeadroom"
        | "IDirectSound::EnableHeadphones"
        | "IDirectSound::SetI3DL2Listener"
        | "IDirectSoundBuffer::SetMixBins"
        | "IDirectSoundBuffer::SetMixBinVolumes" => ok,
        _ => return None,
    };
    Some(handler)
}

/// A vtable slot the runtime knows: its name, handler, and the bytes it pops (spec 9.2).
#[must_use]
pub fn slot(class: &str, slot: u32) -> Option<(&'static str, Handler, u16)> {
    Some(match (class, slot) {
        ("IDirectSound", 1) => ("IDirectSound::AddRef", one as Handler, 4),
        ("IDirectSound", 2) => ("IDirectSound::Release", one, 4),
        ("IDirectSoundBuffer", 1) => ("IDirectSoundBuffer::AddRef", one, 4),
        ("IDirectSoundBuffer", 2) => ("IDirectSoundBuffer::Release", one, 4),
        ("IDirectSoundStream", 0) => ("IDirectSoundStream::AddRef", one, 4),
        ("IDirectSoundStream", 1) => ("IDirectSoundStream::Release", one, 4),
        ("IDirectSoundStream", 2) => ("IDirectSoundStream::GetInfo", stream_info, 8),
        ("IDirectSoundStream", 3) => ("IDirectSoundStream::GetStatus", stream_status, 8),
        ("IDirectSoundStream", 4) => ("IDirectSoundStream::Process", stream_process, 12),
        ("IDirectSoundStream", 5) => ("IDirectSoundStream::Discontinuity", ok, 4),
        ("IDirectSoundStream", 6) => ("IDirectSoundStream::Flush", ok, 4),
        _ => return timer_slot(class, slot),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::testing::kernel;

    #[test]
    fn a_stream_packet_completes_when_its_last_frame_is_released() {
        let mut k = kernel();
        let base = crate::mem::HEAP_BASE + 0x2F_0000;
        // DSSTREAMDESC with a WAVEFORMATEX: 16-bit stereo at 44.1 kHz.
        let (desc, wave, out, packet, completed, status) = (
            base,
            base + 0x40,
            base + 0x80,
            base + 0xC0,
            base + 0x100,
            base + 0x104,
        );
        k.w32(desc + 8, wave).unwrap();
        k.write(
            wave,
            &[1, 0, 2, 0, 0x44, 0xAC, 0, 0, 0, 0, 0, 0, 4, 0, 16, 0],
        )
        .unwrap();
        assert_eq!(
            create_stream(
                &mut k,
                &Call {
                    args: vec![desc, out],
                    esp: 0
                }
            ),
            Ok(CallResult::Value(0))
        );
        let stream = k.r32(out).unwrap();
        // 4410 frames: 100 ms.
        for (i, v) in [base + 0x200, 4410 * 4, completed, status, 0]
            .iter()
            .enumerate()
        {
            k.w32(packet + 4 * u32::try_from(i).unwrap(), *v).unwrap();
        }
        stream_process(
            &mut k,
            &Call {
                args: vec![stream, packet, 0],
                esp: 0,
            },
        )
        .unwrap();
        assert_eq!(k.r32(status), Ok(PACKET_PENDING));
        // It starts 100 ms ahead and is released 100 ms after it ends: 300 ms in all.
        k.clock.pass(2_990_000);
        k.service_sound();
        assert_eq!(k.r32(status), Ok(PACKET_PENDING));
        k.clock.pass(20_000);
        k.service_sound();
        assert_eq!(k.r32(status), Ok(0));
        assert_eq!(k.r32(completed), Ok(4410 * 4));
    }

    #[test]
    fn a_voice_plays_through_its_region_and_stops_unless_it_loops() {
        let mut k = kernel();
        let base = crate::mem::HEAP_BASE + 0x2F_0000;
        // 4410 frames of 16-bit mono at 44.1 kHz, all at 1000: 100 ms.
        let data: Vec<u8> = (0..4410).flat_map(|_| 1000i16.to_le_bytes()).collect();
        k.write(base, &data).unwrap();
        let voice = base + 0x4000;
        k.sound.voices.insert(
            voice,
            Voice {
                format: Format {
                    tag: 1,
                    channels: 1,
                    rate: 44_100,
                    bits: 16,
                },
                data: base,
                len: 4410 * 2,
                ..Voice::default()
            },
        );
        let call = |args: Vec<u32>| Call { args, esp: 0 };
        play(&mut k, &call(vec![voice, 0, 0, 0])).unwrap();
        // The release point is 100 ms behind the clock: 150 ms in, 50 ms have been mixed.
        k.clock.pass(1_500_000);
        k.service_sound();
        assert!(k.sound.voices[&voice].playing);
        assert_eq!(k.sound.timeline.released, 2205);
        k.clock.pass(1_000_000);
        k.service_sound();
        assert!(!k.sound.voices[&voice].playing, "it ended");
        play(&mut k, &call(vec![voice, 0, 0, 1])).unwrap();
        k.clock.pass(10_000_000);
        k.service_sound();
        let status = base + 0x5000;
        buffer_status(&mut k, &call(vec![voice, status])).unwrap();
        assert_eq!(k.r32(status), Ok(5), "looping still");
    }

    #[test]
    fn three_d_settings_read_from_a_ds3dbuffer_wait_for_the_commit_when_deferred() {
        let mut k = kernel();
        let base = crate::mem::HEAP_BASE + 0x2F_0000;
        let voice = base + 0x4000;
        k.sound.voices.insert(voice, Voice::default());
        let floats = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect()
        };
        let mut desc = 64u32.to_le_bytes().to_vec();
        desc.extend(floats(&[1.0, 2.0, 3.0, 0.0, 0.0, -5.0]));
        desc.extend(90u32.to_le_bytes());
        desc.extend(180u32.to_le_bytes());
        desc.extend(floats(&[0.0, 0.0, -1.0]));
        desc.extend((-600i32).to_le_bytes());
        desc.extend(floats(&[2.0, 50.0]));
        desc.extend(1u32.to_le_bytes());
        k.write(base, &desc).unwrap();
        let call = |args: Vec<u32>| Call { args, esp: 0 };
        source_all(&mut k, &call(vec![voice, base, DEFERRED])).unwrap();
        assert_eq!(k.sound.voices[&voice].source, Source::default(), "deferred");
        commit(&mut k, &call(vec![0])).unwrap();
        let source = k.sound.voices[&voice].source;
        assert_eq!(source.position, [1.0, 2.0, 3.0]);
        assert_eq!(source.velocity, [0.0, 0.0, -5.0]);
        assert_eq!((source.inside_cone, source.outside_cone), (90, 180));
        assert_eq!(source.cone_orientation, [0.0, 0.0, -1.0]);
        assert_eq!(source.cone_outside_volume, -600);
        assert_eq!((source.min_distance, source.max_distance), (2.0, 50.0));
        assert_eq!(source.mode, mixer::MODE_HEAD_RELATIVE);
        cone_angles(&mut k, &call(vec![voice, 30, 60, 0])).unwrap();
        let source = k.sound.voices[&voice].source;
        assert_eq!(
            (source.inside_cone, source.outside_cone),
            (30, 60),
            "at once"
        );
    }

    fn call(args: Vec<u32>) -> Call {
        Call { args, esp: 0 }
    }

    /// A buffer made with a size owns that memory, and `Lock` hands it out in one or two spans.
    #[test]
    fn a_sized_buffer_locks_in_spans_that_wrap() {
        let mut k = kernel();
        let base = crate::mem::HEAP_BASE + 0x2F_0000;
        let (desc, wave, out, ptrs) = (base, base + 0x40, base + 0x80, base + 0xC0);
        // DSBUFFERDESC: 0x100 bytes of 16-bit stereo PCM at 44.1 kHz.
        k.w32(desc + 8, 0x100).unwrap();
        k.w32(desc + 12, wave).unwrap();
        k.write(
            wave,
            &[1, 0, 2, 0, 0x44, 0xAC, 0, 0, 0, 0, 0, 0, 4, 0, 16, 0],
        )
        .unwrap();
        // `IDirectSound::CreateSoundBuffer(ds, desc, &buffer, outer)`.
        sound_create_buffer(&mut k, &call(vec![0, desc, out, 0])).unwrap();
        let buffer = k.r32(out).unwrap();
        let voice = &k.sound.voices[&buffer];
        let data = voice.data;
        assert!(data != 0 && voice.len == 0x100);
        assert_eq!(voice.format.channels, 2, "it keeps its channels");
        let lock_at = |k: &mut Kernel, offset, bytes, flags| {
            let result = lock(
                k,
                &call(vec![
                    buffer,
                    offset,
                    bytes,
                    ptrs,
                    ptrs + 4,
                    ptrs + 8,
                    ptrs + 12,
                    flags,
                ]),
            )
            .unwrap();
            let spans: Vec<u32> = (0..4).map(|i| k.r32(ptrs + 4 * i).unwrap()).collect();
            (result, spans)
        };
        assert_eq!(
            lock_at(&mut k, 0x10, 0x20, 0),
            (CallResult::Value(0), vec![data + 0x10, 0x20, 0, 0])
        );
        assert_eq!(
            lock_at(&mut k, 0xF0, 0x30, 0),
            (CallResult::Value(0), vec![data + 0xF0, 0x10, data, 0x20]),
            "past the end it wraps to the start"
        );
        assert_eq!(
            lock_at(&mut k, 0x40, 4, LOCK_ENTIRE_BUFFER).1,
            vec![data, 0x100, 0, 0]
        );
        assert_eq!(
            lock_at(&mut k, 0x200, 4, 0).0,
            CallResult::Value(INVALID_ARGUMENT)
        );
    }

    #[test]
    fn the_cursors_follow_the_voice_and_wrap() {
        let mut k = kernel();
        let base = crate::mem::HEAP_BASE + 0x2F_0000;
        let voice = base + 0x4000;
        k.sound.voices.insert(
            voice,
            Voice {
                format: Format {
                    tag: 1,
                    channels: 2,
                    rate: 44_100,
                    bits: 16,
                },
                data: base,
                len: 4000,
                ..Voice::default()
            },
        );
        set_current_position(&mut k, &call(vec![voice, 2000])).unwrap();
        assert!((k.sound.voices[&voice].position - 500.0).abs() < 1e-9);
        let (play, write) = (base + 0x5000, base + 0x5004);
        current_position(&mut k, &call(vec![voice, play, write])).unwrap();
        // 10 ms at 44.1 kHz is 441 frames of 4 bytes.
        assert_eq!((k.r32(play), k.r32(write)), (Ok(2000), Ok(3764)));
        set_current_position(&mut k, &call(vec![voice, 3000])).unwrap();
        current_position(&mut k, &call(vec![voice, play, write])).unwrap();
        assert_eq!((k.r32(play), k.r32(write)), (Ok(3000), Ok(764)), "wrapped");
    }

    #[test]
    fn pause_holds_a_voice_until_resumed_or_synchronised() {
        let mut k = kernel();
        let base = crate::mem::HEAP_BASE + 0x2F_0000;
        let (a, b, status) = (base, base + 0x10, base + 0x20);
        for voice in [a, b] {
            k.sound.voices.insert(
                voice,
                Voice {
                    playing: true,
                    looping: true,
                    ..Voice::default()
                },
            );
        }
        pause(&mut k, &call(vec![a, 1])).unwrap();
        pause(&mut k, &call(vec![b, PAUSE_SYNCH_PLAYBACK])).unwrap();
        buffer_status(&mut k, &call(vec![a, status])).unwrap();
        assert_eq!(k.r32(status), Ok(STATUS_PAUSED | STATUS_LOOPING));
        synch_playback(&mut k, &call(vec![0])).unwrap();
        assert!(!k.sound.voices[&a].playing && k.sound.voices[&b].playing);
        pause(&mut k, &call(vec![a, PAUSE_RESUME])).unwrap();
        buffer_status(&mut k, &call(vec![a, status])).unwrap();
        assert_eq!(k.r32(status), Ok(STATUS_PLAYING | STATUS_LOOPING));
        stop(&mut k, &call(vec![a])).unwrap();
        pause(&mut k, &call(vec![a, PAUSE_RESUME])).unwrap();
        assert!(
            !k.sound.voices[&a].playing,
            "a stopped voice does not resume"
        );
    }

    #[test]
    fn pitch_is_a_frequency_and_headroom_lowers_the_volume() {
        let mut k = kernel();
        let voice = crate::mem::HEAP_BASE + 0x2F_0000;
        k.sound.voices.insert(voice, Voice::default());
        set_pitch(&mut k, &call(vec![voice, (-4096i32).cast_unsigned()])).unwrap();
        assert_eq!(
            k.sound.voices[&voice].frequency, 24_000,
            "an octave below 48 kHz"
        );
        set_pitch(&mut k, &call(vec![voice, 0])).unwrap();
        assert_eq!(k.sound.voices[&voice].frequency, 48_000);
        set_headroom(&mut k, &call(vec![voice, 600])).unwrap();
        assert_eq!(k.sound.voices[&voice].headroom, 600);
        set_headroom(&mut k, &call(vec![voice, u32::MAX])).unwrap();
        assert_eq!(k.sound.voices[&voice].headroom, 10_000);
    }

    /// Dragon's Lair sets each effect's frequency to 48 kHz, then its format: the format's rate
    /// is the one it plays at, and an ADPCM region decoded in the old format is decoded again.
    #[test]
    fn a_format_brings_its_rate_after_a_frequency() {
        let mut k = kernel();
        let voice = crate::mem::HEAP_BASE + 0x2F_0000;
        let wave = voice + 0x100;
        // WAVEFORMATEX: Xbox ADPCM, mono, 22.05 kHz, 4 bits.
        let mut bytes = Vec::new();
        bytes.extend(Format::ADPCM.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(22_050u32.to_le_bytes());
        bytes.extend((22_050u32 * 36 / 64).to_le_bytes());
        bytes.extend(36u16.to_le_bytes());
        bytes.extend(4u16.to_le_bytes());
        k.write(wave, &bytes).unwrap();
        k.sound.voices.insert(
            voice,
            Voice {
                decoded: Some(vec![0; 130]),
                ..Voice::default()
            },
        );
        set_frequency(&mut k, &call(vec![voice, 48_000])).unwrap();
        set_format(&mut k, &call(vec![voice, wave])).unwrap();
        let v = &k.sound.voices[&voice];
        assert_eq!(v.rate(), 22_050, "the format's rate");
        assert!(v.decoded.is_none(), "decoded again in the new format");
        set_frequency(&mut k, &call(vec![voice, 30_000])).unwrap();
        assert_eq!(
            k.sound.voices[&voice].rate(),
            30_000,
            "a later frequency holds"
        );
    }

    /// Marvel vs. Capcom 2 calls the timer object it makes each tick, and makes sounds with it:
    /// it must get the objects, each method taking its arguments off the stack.
    #[test]
    fn a_timer_object_is_made_and_makes_sounds() {
        let mut k = kernel();
        let out = crate::mem::HEAP_BASE + 0x2F_0000;
        let result = create_timer_object(&mut k, &call(vec![0, 1000, out])).unwrap();
        assert!(matches!(result, CallResult::Value(0)));
        let object = k.r32(out).unwrap();
        assert_ne!(object, 0, "an object is written");
        assert_ne!(k.r32(object).unwrap(), 0, "with a vtable");
        // Method 3 is told a 64-bit time: this and two dwords.
        assert_eq!(slot("DSoundTimerObject", 3).map(|s| s.2), Some(12));
        let (name, new_voice, _) = slot("DSoundTimerObject", 5).unwrap();
        assert_eq!(name, "DSoundTimerObject::NewVoice");
        let voice = out + 4;
        new_voice(&mut k, &call(vec![object, 0, voice])).unwrap();
        assert_ne!(k.r32(voice).unwrap(), 0, "a sound object is written");
        assert_eq!(slot("DSoundTimerVoice", 38).map(|s| s.2), Some(8));
        assert!(
            slot("DSoundTimerVoice", 39).is_none(),
            "no method past the vtable"
        );
    }

    /// Lays out `entry` at `at` in `code`, its callees after it with each call pointed at the
    /// next; masked bytes are `int3`. Returns where the chain ends.
    fn lay_out(code: &mut [u8], entry: &xdk::Entry, at: usize) -> usize {
        let bytes = |pattern: &xdk::Pattern| -> Vec<u8> {
            pattern
                .to_string()
                .split_whitespace()
                .map(|token| u8::from_str_radix(token, 16).unwrap_or(0xCC))
                .collect()
        };
        let mut caller = at;
        let mut end = at + entry.pattern.len();
        code[at..end].copy_from_slice(&bytes(&entry.pattern));
        for (offset, callee) in &entry.callees {
            let rel = u32::try_from(end).unwrap() - u32::try_from(caller + offset + 4).unwrap();
            code[caller + offset..caller + offset + 4].copy_from_slice(&rel.to_le_bytes());
            caller = end;
            end += callee.len();
            code[caller..end].copy_from_slice(&bytes(callee));
        }
        end
    }

    /// The buffer thunks `BLiNX` links, all of one shape, are found by what they call: none is
    /// taken for another, as match numbers took `SetHeadroom` for `GetStatus`.
    #[test]
    fn thunks_of_one_shape_are_found_by_their_callees() {
        let entries = xdk::parse(super::super::hooks::SIGNATURES).unwrap();
        let names = [
            "IDirectSoundBuffer::SetPitch",
            "IDirectSoundBuffer::SetHeadroom",
            "IDirectSoundBuffer::SetMixBins",
            "IDirectSoundBuffer::SetMixBinVolumes",
            "IDirectSoundBuffer::SetFrequency",
            "IDirectSoundBuffer::Pause",
            "IDirectSoundBuffer::SetCurrentPosition",
            "IDirectSoundBuffer::GetCurrentPosition",
        ];
        let mut code = vec![0xCC; 0x1000];
        let mut at = 0;
        let mut starts = Vec::new();
        for name in names {
            let entry = entries.iter().find(|e| e.name == name).unwrap();
            starts.push(0x1_0000 + u32::try_from(at).unwrap());
            at = lay_out(&mut code, entry, at).next_multiple_of(16);
        }
        let found = xdk::search(&entries, &[(0x1_0000, &code)]);
        for (name, start) in names.iter().zip(starts) {
            let hits: Vec<u32> = found
                .hits
                .iter()
                .filter(|hit| entries[hit.entry].name == *name)
                .map(|hit| hit.address)
                .collect();
            assert_eq!(hits, [start], "{name}");
        }
        assert_eq!(found.hits.len(), names.len(), "nothing else is found");
    }

    #[test]
    fn mono_stream_pcm_goes_to_both_sides_scaled_by_its_volume() {
        let stream = Stream {
            format: Format {
                tag: 1,
                channels: 1,
                rate: 44_100,
                bits: 16,
            },
            volume: -2000,
            routes: mixer::default_routes(1),
            ..Stream::default()
        };
        let frames = stream_frames(&stream, &[0x10, 0x27, 0xF0, 0xD8]);
        assert_eq!(frames.len(), 2);
        assert!((frames[0][0] - 1000.0).abs() < 0.01 && (frames[0][1] - 1000.0).abs() < 0.01);
        assert!((frames[1][0] + 1000.0).abs() < 0.01);
    }
}
