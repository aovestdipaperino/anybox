//! The fade through black and white that hibernating and resuming play over the last picture:
//! colour to black-and-white to black going out, and the same run backwards coming in.

use std::time::Instant;

/// How long the picture takes to lose its colour (or to regain it, coming in).
const DESATURATE_SECS: f64 = 0.6;
/// How long the black-and-white picture takes to go dark (or to come out of the dark).
const DARKEN_SECS: f64 = 0.6;

/// The 0RGB pixel `rgb` with `colour` of its colour (1 full, 0 grey, by Rec. 601 luma) at `light`
/// of its brightness (1 full, 0 black).
pub(crate) fn shade(rgb: u32, colour: f32, light: f32) -> u32 {
    let [_, r, g, b] = rgb.to_be_bytes();
    let (r, g, b) = (f32::from(r), f32::from(g), f32::from(b));
    let grey = 0.299 * r + 0.587 * g + 0.114 * b;
    let channel = |c: f32| {
        let value = ((grey + (c - grey) * colour) * light)
            .round()
            .clamp(0.0, 255.0);
        // Clamped to a byte just above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let byte = value as u32;
        byte
    };
    channel(r) << 16 | channel(g) << 8 | channel(b)
}

/// The (colour, light) levels `elapsed_secs` into the fade, or `None` once it is over. Going out,
/// colour falls from 1 to 0 at full light, then light falls from 1 to 0; coming in is that run
/// backwards.
pub(crate) fn levels(elapsed_secs: f64, inward: bool) -> Option<(f32, f32)> {
    let total = DESATURATE_SECS + DARKEN_SECS;
    if elapsed_secs >= total {
        return None;
    }
    let t = if inward {
        total - elapsed_secs
    } else {
        elapsed_secs
    }
    .max(0.0);
    #[allow(clippy::cast_possible_truncation)]
    let levels = if t < DESATURATE_SECS {
        (1.0 - t / DESATURATE_SECS, 1.0)
    } else {
        (0.0, 1.0 - (t - DESATURATE_SECS) / DARKEN_SECS)
    };
    #[allow(clippy::cast_possible_truncation)]
    Some((levels.0 as f32, levels.1 as f32))
}

/// A picture fading out to black or in from it, started when it was made.
#[derive(Debug)]
pub(crate) struct Fade {
    started: Instant,
    picture: Vec<u32>,
    inward: bool,
}

impl Fade {
    /// Starts fading `pixels`, a 0RGB picture, from colour to black.
    pub(crate) fn out(pixels: &[u32]) -> Self {
        Self::start(pixels, false)
    }

    /// Starts fading `pixels`, a 0RGB picture, in from black to colour.
    pub(crate) fn inward(pixels: &[u32]) -> Self {
        Self::start(pixels, true)
    }

    fn start(pixels: &[u32], inward: bool) -> Self {
        Self {
            started: Instant::now(),
            picture: pixels.to_vec(),
            inward,
        }
    }

    /// The picture as it looks now; `None` once the fade is over.
    pub(crate) fn frame(&self) -> Option<Vec<u32>> {
        let (colour, light) = levels(self.started.elapsed().as_secs_f64(), self.inward)?;
        Some(
            self.picture
                .iter()
                .map(|&rgb| shade(rgb, colour, light))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shade_keeps_white_and_full_colour_and_blackens_at_no_light() {
        assert_eq!(shade(0x00FF_FFFF, 0.0, 1.0), 0x00FF_FFFF);
        assert_eq!(shade(0x0012_3456, 1.0, 1.0), 0x0012_3456);
        assert_eq!(shade(0x00FF_8040, 1.0, 0.0), 0);
        assert_eq!(shade(0x00FF_8040, 0.0, 0.0), 0);
    }

    #[test]
    fn shade_turns_pure_red_into_its_luma_grey() {
        assert_eq!(shade(0x00FF_0000, 0.0, 1.0), 0x004C_4C4C);
    }

    #[test]
    fn levels_follow_the_two_stages_going_out() {
        let at = |t| levels(t, false).unwrap();
        assert_eq!(at(0.0), (1.0, 1.0));
        assert!((at(0.3).0 - 0.5).abs() < 1e-6 && (at(0.3).1 - 1.0).abs() < 1e-6);
        assert_eq!(at(0.6), (0.0, 1.0));
        assert!(at(0.9).0.abs() < 1e-6 && (at(0.9).1 - 0.5).abs() < 1e-6);
        assert_eq!(levels(1.2, false), None);
        assert_eq!(levels(5.0, false), None);
    }

    #[test]
    fn levels_coming_in_are_going_out_backwards() {
        assert_eq!(levels(0.0, true), Some((0.0, 0.0)));
        assert_eq!(levels(1.2, true), None);
        for t in [0.0, 0.3, 0.6, 0.9, 1.1] {
            let inward = levels(t, true).unwrap();
            // Out at 1.2 s is over, which is black.
            let outward = levels(1.2 - t, false).unwrap_or((0.0, 0.0));
            assert!(
                (inward.0 - outward.0).abs() < 1e-5 && (inward.1 - outward.1).abs() < 1e-5,
                "{t}"
            );
        }
    }

    #[test]
    fn a_fade_makes_frames_of_the_picture_size() {
        let fade = Fade::out(&[0x00FF_0000; 6]);
        let frame = fade.frame().unwrap();
        assert_eq!(frame.len(), 6);
    }
}
