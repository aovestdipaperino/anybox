//! The mixer (spec 9.3), Xbox ADPCM (9.4), 3D positioning and the limiter (9.5): the arithmetic,
//! kept apart from the `DirectSound` objects that use it. Beyond the spec, sound cones, velocity
//! and the Doppler shift follow the public `DirectSound` 3D model.
//!
//! Samples are `f32` in 16-bit units until the limiter rounds them; frame counts and volumes
//! become floats freely, as they are far below where precision is lost.
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

use std::collections::VecDeque;

/// A stereo frame.
pub type Frame = [f32; 2];

/// The gain of a volume in hundredths of a dB.
#[must_use]
pub fn gain(volume: i32) -> f32 {
    10f32.powf(volume.clamp(-10_000, 0) as f32 / 2000.0)
}

/// The limiter's knee: magnitudes up to here pass.
const KNEE: f32 = 23_000.0;
/// The room above the knee.
const ROOM: f32 = 9767.0;

/// A summed sample through the limiter, as 16-bit PCM.
#[must_use]
pub fn limit(sample: f32) -> i16 {
    let magnitude = sample.abs();
    let limited = if magnitude <= KNEE {
        magnitude
    } else {
        KNEE + ROOM * ((magnitude - KNEE) / ROOM).tanh()
    };
    // At most 32767 in magnitude, so the cast is exact.
    #[allow(clippy::cast_possible_truncation)]
    let rounded = limited.round().min(32_767.0) as i16;
    if sample < 0.0 { -rounded } else { rounded }
}

/// The stereo gains of mix bin `bin` (spec 9.3): front left and right, centre, back left and
/// right; the others (LFE, effects) are silent.
#[must_use]
pub fn bin_route(bin: u32) -> Frame {
    match bin {
        0 => [1.0, 0.0],
        1 => [0.0, 1.0],
        2 => [0.707, 0.707],
        4 => [0.707, 0.0],
        5 => [0.0, 0.707],
        _ => [0.0, 0.0],
    }
}

/// The default routes of a stream with `channels`: mono to both front speakers, stereo left and
/// right.
#[must_use]
pub fn default_routes(channels: u16) -> Vec<Frame> {
    match channels {
        1 => vec![[1.0, 1.0]],
        2 => vec![[1.0, 0.0], [0.0, 1.0]],
        _ => Vec::new(),
    }
}

/// One PCM sample of `bits` (8 unsigned or 16 signed little-endian) at `bytes`, in 16-bit units.
fn pcm_sample(bytes: &[u8], bits: u16) -> f32 {
    match bits {
        8 => f32::from((i16::from(bytes[0]) - 128) << 8),
        _ => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])),
    }
}

/// PCM frames of `channels` and `bits` as stereo (a mono sample on both sides); `None` for a
/// layout that is not 8- or 16-bit mono or stereo.
#[must_use]
pub fn pcm_frames(data: &[u8], channels: u16, bits: u16) -> Option<Vec<Frame>> {
    if !matches!(channels, 1 | 2) || !matches!(bits, 8 | 16) {
        return None;
    }
    let width = usize::from(bits / 8);
    Some(
        data.chunks_exact(width * usize::from(channels))
            .map(|frame| {
                let left = pcm_sample(frame, bits);
                let right = if channels == 2 {
                    pcm_sample(&frame[width..], bits)
                } else {
                    left
                };
                [left, right]
            })
            .collect(),
    )
}

/// `frames` at `rate` resampled to the mixer's 44.1 kHz by linear interpolation.
#[must_use]
pub fn resample(frames: &[Frame], rate: u32) -> Vec<Frame> {
    let Some(&last) = frames.last() else {
        return Vec::new();
    };
    let rate = u64::from(rate.max(1));
    let count = frames.len() as u64 * super::sound::RATE / rate;
    (0..count)
        .map(|j| {
            let at = j as f64 * rate as f64 / super::sound::RATE as f64;
            // Non-negative and below the frame count.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let i = at as usize;
            let fraction = (at - i as f64) as f32;
            let a = frames.get(i).copied().unwrap_or(last);
            let b = frames.get(i + 1).copied().unwrap_or(a);
            [
                a[0] + (b[0] - a[0]) * fraction,
                a[1] + (b[1] - a[1]) * fraction,
            ]
        })
        .collect()
}

/// The IMA ADPCM step sizes.
const STEPS: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];
/// How each code's low three bits move the step index.
const INDEX_MOVES: [isize; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];
/// Bytes of one channel's part of a block.
pub const ADPCM_BLOCK: usize = 36;

/// One channel's decoder state.
#[derive(Debug, Clone, Copy)]
struct Adpcm {
    sample: i32,
    index: usize,
}

impl Adpcm {
    fn decode(&mut self, code: u8) -> i16 {
        let step = STEPS[self.index];
        let mut delta = step >> 3;
        if code & 4 != 0 {
            delta += step;
        }
        if code & 2 != 0 {
            delta += step >> 1;
        }
        if code & 1 != 0 {
            delta += step >> 2;
        }
        self.sample = if code & 8 != 0 {
            self.sample - delta
        } else {
            self.sample + delta
        }
        .clamp(-32_768, 32_767);
        self.index = self
            .index
            .saturating_add_signed(INDEX_MOVES[usize::from(code & 7)])
            .min(88);
        // Clamped to 16 bits above.
        #[allow(clippy::cast_possible_truncation)]
        let sample = self.sample as i16;
        sample
    }
}

/// Decodes Xbox ADPCM of `channels`: blocks of 36 bytes a channel, 65 samples a channel each,
/// interleaved. A partial block at the end is left out.
#[must_use]
pub fn decode_adpcm(data: &[u8], channels: u16) -> Vec<i16> {
    let channels = usize::from(channels.max(1));
    let mut out = Vec::with_capacity(data.len() / ADPCM_BLOCK * 65 * 2);
    let mut block_samples = vec![0i16; 65 * channels];
    for block in data.chunks_exact(ADPCM_BLOCK * channels) {
        let (headers, codes) = block.split_at(4 * channels);
        let mut states: Vec<Adpcm> = headers
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .map(|(channel, header)| {
                let sample = i16::from_le_bytes([header[0], header[1]]);
                block_samples[channel] = sample;
                Adpcm {
                    sample: i32::from(sample),
                    index: usize::from(header[2]).min(88),
                }
            })
            .collect();
        // Each channel's codes come in groups of 4 bytes (8 samples), channel after channel.
        for (group, bytes) in codes.as_chunks::<4>().0.iter().enumerate() {
            let channel = group % channels;
            let first = 1 + group / channels * 8;
            for (i, &byte) in bytes.iter().enumerate() {
                for (half, code) in [byte & 0xF, byte >> 4].into_iter().enumerate() {
                    let n = first + 2 * i + half;
                    block_samples[n * channels + channel] = states[channel].decode(code);
                }
            }
        }
        out.extend_from_slice(&block_samples);
    }
    out
}

/// A 3D vector.
pub type Vector = [f32; 3];

fn sub(a: Vector, b: Vector) -> Vector {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vector, b: Vector) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: Vector) -> f32 {
    dot(a, a).sqrt()
}

/// The speed of sound, in metres a second.
const SPEED_OF_SOUND: f32 = 343.0;

/// The listener (spec 9.5): left-handed axes, x right, y up, z ahead. The distance factor is
/// metres per unit; the Doppler factor scales the shift (0 is none, 1 the real one).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Listener {
    pub position: Vector,
    pub velocity: Vector,
    pub front: Vector,
    pub top: Vector,
    pub rolloff: f32,
    pub distance_factor: f32,
    pub doppler_factor: f32,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            front: [0.0, 0.0, 1.0],
            top: [0.0, 1.0, 0.0],
            rolloff: 1.0,
            distance_factor: 1.0,
            doppler_factor: 1.0,
        }
    }
}

/// `DS3DMODE_HEADRELATIVE`.
pub const MODE_HEAD_RELATIVE: u32 = 1;
/// `DS3DMODE_DISABLE`.
pub const MODE_DISABLE: u32 = 2;

/// A voice's 3D settings. The cone: inside it (an angle in degrees around its orientation) the
/// sound is at full volume, outside the outer angle at the outside volume (hundredths of a dB),
/// and between the two the volume moves from one to the other. The distance and Doppler factors
/// multiply the listener's.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub position: Vector,
    pub velocity: Vector,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: f32,
    pub mode: u32,
    pub inside_cone: u32,
    pub outside_cone: u32,
    pub cone_orientation: Vector,
    pub cone_outside_volume: i32,
    pub distance_factor: f32,
    pub doppler_factor: f32,
}

impl Default for Source {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            min_distance: 1.0,
            max_distance: 1e9,
            rolloff: 1.0,
            mode: 0,
            inside_cone: 360,
            outside_cone: 360,
            cone_orientation: [0.0, 0.0, 1.0],
            cone_outside_volume: 0,
            distance_factor: 1.0,
            doppler_factor: 1.0,
        }
    }
}

impl Source {
    /// The source's offset from the listener, and the listener's right.
    fn offset(&self, listener: &Listener) -> (Vector, Vector) {
        if self.mode == MODE_HEAD_RELATIVE {
            (self.position, [1.0, 0.0, 0.0])
        } else {
            let right = cross(listener.top, listener.front);
            let norm = length(right);
            let right = if norm > 0.0 && norm.is_finite() {
                [right[0] / norm, right[1] / norm, right[2] / norm]
            } else {
                [1.0, 0.0, 0.0]
            };
            (sub(self.position, listener.position), right)
        }
    }

    /// The gain of the cone toward a listener at `offset` from the source's position (as
    /// [`Source::offset`] gives it, from the listener).
    fn cone_gain(&self, offset: Vector) -> f32 {
        let (inside, outside) = (
            self.inside_cone.min(360) as f32 / 2.0,
            self.outside_cone.min(360) as f32 / 2.0,
        );
        let (distance, facing) = (length(offset), length(self.cone_orientation));
        if inside >= 180.0 || distance == 0.0 || facing == 0.0 || !facing.is_finite() {
            return 1.0;
        }
        // The angle between where the source faces and the way to the listener.
        let toward = [-offset[0], -offset[1], -offset[2]];
        let cosine = (dot(self.cone_orientation, toward) / (facing * distance)).clamp(-1.0, 1.0);
        let angle = cosine.acos().to_degrees();
        let volume = if angle <= inside {
            0.0
        } else if angle >= outside.max(inside) {
            self.cone_outside_volume as f32
        } else {
            self.cone_outside_volume as f32 * (angle - inside) / (outside - inside)
        };
        10f32.powf(volume.clamp(-10_000.0, 0.0) / 2000.0)
    }

    /// The ratio by which the Doppler shift changes this source's frequency for `listener`:
    /// above 1 as they close in, below as they draw apart. Each speed along the line between
    /// them is held under half the speed of sound.
    #[must_use]
    pub fn doppler(&self, listener: &Listener) -> f32 {
        let factor = self.doppler_factor * listener.doppler_factor;
        let (offset, _) = self.offset(listener);
        let distance = length(offset);
        if self.mode == MODE_DISABLE || factor == 0.0 || distance == 0.0 || !factor.is_finite() {
            return 1.0;
        }
        let metres = self.distance_factor * listener.distance_factor;
        let sound = SPEED_OF_SOUND / if metres > 0.0 { metres } else { 1.0 };
        let way = [
            offset[0] / distance,
            offset[1] / distance,
            offset[2] / distance,
        ];
        let listener_velocity = if self.mode == MODE_HEAD_RELATIVE {
            [0.0; 3]
        } else {
            listener.velocity
        };
        let limit = sound / 2.0;
        let toward_source = (dot(listener_velocity, way) * factor).clamp(-limit, limit);
        let toward_listener = (-dot(self.velocity, way) * factor).clamp(-limit, limit);
        let ratio = (sound + toward_source) / (sound - toward_listener);
        if ratio.is_finite() { ratio } else { 1.0 }
    }

    /// The left and right gains of this source heard by `listener`; `stereo` sources are not
    /// panned.
    #[must_use]
    pub fn gains(&self, listener: &Listener, stereo: bool) -> Frame {
        if self.mode == MODE_DISABLE {
            return [1.0, 1.0];
        }
        let (offset, right) = self.offset(listener);
        let distance = length(offset);
        let clamped = distance.min(self.max_distance);
        let minimum = self.min_distance;
        let attenuation = if clamped <= minimum {
            1.0
        } else {
            minimum / (minimum + self.rolloff * listener.rolloff * (clamped - minimum))
        } * self.cone_gain(offset);
        if stereo || distance == 0.0 {
            return [attenuation, attenuation];
        }
        let bearing = (dot(offset, right) / distance).clamp(-1.0, 1.0);
        let angle = (bearing + 1.0) * std::f32::consts::FRAC_PI_4;
        let sqrt2 = std::f32::consts::SQRT_2;
        [
            (angle.cos() * sqrt2).min(1.0) * attenuation,
            (angle.sin() * sqrt2).min(1.0) * attenuation,
        ]
    }
}

/// The mix not yet released: frame `released` onwards.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Timeline {
    /// The first frame not released.
    pub released: u64,
    pending: VecDeque<Frame>,
}

impl Timeline {
    /// Adds `frames` from frame `at`; any part before the release point is dropped.
    pub fn add(&mut self, at: u64, frames: &[Frame]) {
        let skip = usize::try_from(self.released.saturating_sub(at)).unwrap_or(usize::MAX);
        let Some(frames) = frames.get(skip..) else {
            return;
        };
        let start = usize::try_from(at.max(self.released) - self.released).unwrap_or(usize::MAX);
        let end = start.saturating_add(frames.len());
        if self.pending.len() < end {
            self.pending.resize(end, [0.0; 2]);
        }
        for (slot, frame) in self.pending.range_mut(start..end).zip(frames) {
            slot[0] += frame[0];
            slot[1] += frame[1];
        }
    }

    /// Takes the summed frames from the release point up to `end`, moving the release point
    /// there.
    pub fn take(&mut self, end: u64) -> Vec<Frame> {
        let count = usize::try_from(end.saturating_sub(self.released)).unwrap_or(usize::MAX);
        let mut out: Vec<Frame> = self
            .pending
            .drain(..count.min(self.pending.len()))
            .collect();
        out.resize(count, [0.0; 2]);
        self.released = self.released.max(end);
        out
    }

    /// Moves the release point to `end` without taking anything, as when the timeline follows
    /// the clock with nothing queued.
    pub fn skip_to(&mut self, end: u64) {
        let _ = self.take(end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gains_and_the_limiter() {
        assert!((gain(0) - 1.0).abs() < 1e-6);
        assert!((gain(-2000) - 0.1).abs() < 1e-6);
        assert_eq!(gain(-20_000), gain(-10_000));
        assert_eq!(limit(1000.0), 1000);
        assert_eq!(limit(-23_000.0), -23_000);
        assert_eq!(limit(100_000.0), 32_767);
        let above = limit(30_000.0);
        assert!(above > 23_000 && above < 30_000, "{above}");
        assert_eq!(limit(-30_000.0), -above);
    }

    #[test]
    fn pcm_becomes_stereo_and_resamples() {
        let mono8 = pcm_frames(&[128, 255, 0], 1, 8).unwrap();
        assert_eq!(
            mono8,
            vec![[0.0, 0.0], [32512.0, 32512.0], [-32768.0, -32768.0]]
        );
        let stereo16 = pcm_frames(&[1, 0, 0xFF, 0xFF], 2, 16).unwrap();
        assert_eq!(stereo16, vec![[1.0, -1.0]]);
        assert_eq!(pcm_frames(&[0; 6], 3, 16), None);
        // 22.05 kHz doubles, interpolating between frames.
        let up = resample(&[[0.0, 0.0], [100.0, -100.0]], 22_050);
        assert_eq!(
            up,
            vec![[0.0, 0.0], [50.0, -50.0], [100.0, -100.0], [100.0, -100.0]]
        );
    }

    #[test]
    fn an_adpcm_block_starts_with_its_header_sample() {
        // Mono: the header (sample 1000, index 0), then 32 bytes of codes.
        let mut block = vec![0xE8, 0x03, 0, 0];
        block.extend([0x77; 32]);
        let samples = decode_adpcm(&block, 1);
        assert_eq!(samples.len(), 65);
        assert_eq!(samples[0], 1000);
        // Code 7 at step 7: 7 >> 3 + 7 + 3 + 1 = 11 up, and the index climbs.
        assert_eq!(samples[1], 1011);
        assert!(samples.windows(2).all(|w| w[1] >= w[0]), "rising");
        // Stereo: the channels' headers, then 4 bytes each in turn.
        let mut stereo = vec![0x10, 0, 0, 0, 0xF0, 0xFF, 0, 0];
        for group in 0..16 {
            stereo.extend([if group % 2 == 0 { 0x77 } else { 0xFF }; 4]);
        }
        let samples = decode_adpcm(&stereo, 2);
        assert_eq!(samples.len(), 130);
        assert_eq!((samples[0], samples[1]), (16, -16));
        assert!(samples[2] > 16 && samples[3] < -16);
        // A partial block is left out.
        assert_eq!(decode_adpcm(&block[..35], 1), Vec::<i16>::new());
    }

    #[test]
    fn sources_pan_and_attenuate() {
        let listener = Listener::default();
        let ahead = Source {
            position: [0.0, 0.0, 1.0],
            ..Source::default()
        };
        let [l, r] = ahead.gains(&listener, false);
        assert!((l - 1.0).abs() < 1e-5 && (r - 1.0).abs() < 1e-5);
        let right = Source {
            position: [2.0, 0.0, 0.0],
            ..Source::default()
        };
        let [l, r] = right.gains(&listener, false);
        // Twice the minimum distance: half, all on the right.
        assert!(l.abs() < 1e-5 && (r - 0.5).abs() < 1e-5, "{l} {r}");
        assert_eq!(right.gains(&listener, true), [0.5, 0.5]);
        let off = Source {
            mode: MODE_DISABLE,
            ..right
        };
        assert_eq!(off.gains(&listener, false), [1.0, 1.0]);
    }

    #[test]
    fn a_cone_quietens_a_source_facing_away() {
        let listener = Listener::default();
        // At (0, 0, 1), facing on along z: away from the listener at the origin.
        let away = Source {
            position: [0.0, 0.0, 1.0],
            inside_cone: 90,
            outside_cone: 180,
            cone_outside_volume: -2000,
            ..Source::default()
        };
        let [l, r] = away.gains(&listener, true);
        assert!((l - 0.1).abs() < 1e-4 && (r - 0.1).abs() < 1e-4, "{l}");
        let facing = Source {
            cone_orientation: [0.0, 0.0, -1.0],
            ..away
        };
        assert_eq!(facing.gains(&listener, true), [1.0, 1.0]);
        // Side on: 90 degrees, the outer cone's edge.
        let side = Source {
            cone_orientation: [1.0, 0.0, 0.0],
            ..away
        };
        let [l, _] = side.gains(&listener, true);
        assert!((l - 0.1).abs() < 1e-4, "{l}");
        let between = Source {
            outside_cone: 270,
            ..side
        };
        // 90 degrees, halfway from 45 to 135: half of -20 dB.
        let [l, _] = between.gains(&listener, true);
        assert!((l - 10f32.powf(-0.5)).abs() < 1e-4, "{l}");
    }

    #[test]
    fn closing_in_raises_the_pitch_and_drawing_apart_lowers_it() {
        let listener = Listener::default();
        let approaching = Source {
            position: [0.0, 0.0, 100.0],
            velocity: [0.0, 0.0, -34.3],
            ..Source::default()
        };
        let ratio = approaching.doppler(&listener);
        assert!((ratio - 343.0 / (343.0 - 34.3)).abs() < 1e-4, "{ratio}");
        let receding = Source {
            velocity: [0.0, 0.0, 34.3],
            ..approaching
        };
        assert!(receding.doppler(&listener) < 1.0);
        let walking_toward = Listener {
            velocity: [0.0, 0.0, 34.3],
            ..listener
        };
        let still = Source {
            velocity: [0.0; 3],
            ..approaching
        };
        assert!((still.doppler(&walking_toward) - 1.1).abs() < 1e-4);
        let none = Listener {
            doppler_factor: 0.0,
            ..listener
        };
        assert_eq!(approaching.doppler(&none), 1.0);
    }

    #[test]
    fn the_timeline_sums_and_releases() {
        let mut timeline = Timeline::default();
        timeline.add(2, &[[1.0, 1.0], [2.0, 2.0]]);
        timeline.add(3, &[[10.0, 0.0]]);
        assert_eq!(timeline.take(2), vec![[0.0, 0.0]; 2]);
        assert_eq!(timeline.take(5), vec![[1.0, 1.0], [12.0, 2.0], [0.0, 0.0]]);
        timeline.add(4, &[[5.0, 5.0], [6.0, 6.0]]);
        assert_eq!(timeline.take(6), vec![[6.0, 6.0]]);
    }
}
