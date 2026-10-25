//! What the window shows besides the title's images: the boot logo, warming up and switching off
//! as a CRT does.
//!
//! Copied from the author's original `anybox-tk` code (not derived from xemu), with the author's
//! permission; the frame-rate text it also drew is not used, the frame rate is in the title.

/// The anyxbox logo, shown until the title shows its first image.
const LOGO: &[u8] = include_bytes!("../../assets/logo.png");

/// The logo centred on black in a `width` x `height` image of 0RGB pixels; plain black if it
/// cannot be decoded.
pub(crate) fn boot_image(width: usize, height: usize) -> Vec<u32> {
    let mut pixels = vec![0u32; width * height];
    let Some((logo_width, logo_height, rgba)) = decode_logo() else {
        return pixels;
    };
    let left = width.saturating_sub(logo_width) / 2;
    let top = height.saturating_sub(logo_height) / 2;
    for y in 0..logo_height.min(height) {
        for x in 0..logo_width.min(width) {
            let at = (y * logo_width + x) * 4;
            let [r, g, b, a] = [rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]];
            // Over black, a translucent pixel is its colour scaled by its alpha.
            let channel = |c: u8| u32::from(c) * u32::from(a) / 255;
            pixels[(top + y) * width + left + x] = channel(r) << 16 | channel(g) << 8 | channel(b);
        }
    }
    pixels
}

/// The logo centred on black in a `width` x `height` image, as RGBA8.
fn boot_rgba(width: usize, height: usize) -> image::RgbaImage {
    let pixels = boot_image(width, height);
    let mut image = image::RgbaImage::new(
        u32::try_from(width).unwrap_or(0),
        u32::try_from(height).unwrap_or(0),
    );
    for (pixel, rgb) in image.pixels_mut().zip(pixels) {
        let [_, r, g, b] = rgb.to_be_bytes();
        *pixel = image::Rgba([r, g, b, 255]);
    }
    image
}

/// A CRT turning on or off, from `crt-off`: switching off, the picture collapses to a bright
/// line, the line to a dot, and the dot fades; warming up is the same, backwards.
pub(crate) struct Crt {
    started: std::time::Instant,
    picture: image::RgbaImage,
    config: crt_off::Config,
    /// Whether it is warming up rather than switching off.
    warming: bool,
}

impl std::fmt::Debug for Crt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Crt")
            .field("warming", &self.warming)
            .finish_non_exhaustive()
    }
}

impl Crt {
    /// Starts warming up a `width` x `height` logo.
    pub(crate) fn warm_up(width: usize, height: usize) -> Self {
        Self {
            started: std::time::Instant::now(),
            picture: boot_rgba(width, height),
            config: crt_off::Config::default(),
            warming: true,
        }
    }

    /// Starts switching off `pixels`, a `width` x `height` 0RGB picture.
    pub(crate) fn switch_off(pixels: &[u32], width: usize, height: usize) -> Self {
        let mut picture = image::RgbaImage::new(
            u32::try_from(width).unwrap_or(0),
            u32::try_from(height).unwrap_or(0),
        );
        for (pixel, &rgb) in picture.pixels_mut().zip(pixels) {
            let [_, r, g, b] = rgb.to_be_bytes();
            *pixel = image::Rgba([r, g, b, 255]);
        }
        Self {
            started: std::time::Instant::now(),
            picture,
            config: crt_off::Config::default(),
            warming: false,
        }
    }

    /// The image to show now from the warm-up in `slot`: its frame while it runs, the logo
    /// itself as it ends, which empties the slot, and `None` when there is none.
    pub(crate) fn step(slot: &mut Option<Self>, width: usize, height: usize) -> Option<Vec<u32>> {
        let frame = slot.as_ref()?.frame();
        if frame.is_none() {
            *slot = None;
        }
        Some(frame.unwrap_or_else(|| boot_image(width, height)))
    }

    /// Whether the effect has reached its dot: the line has shrunk to the centre.
    pub(crate) fn at_dot(&self) -> bool {
        let into_dot = self.config.vstretch_secs + self.config.hstretch_secs;
        let elapsed = self.started.elapsed().as_secs_f64();
        if self.warming {
            elapsed <= self.config.duration() - into_dot
        } else {
            elapsed >= into_dot
        }
    }

    /// The image to show now as 0RGB pixels, or `None` once the effect is over.
    pub(crate) fn frame(&self) -> Option<Vec<u32>> {
        let elapsed = self.started.elapsed().as_secs_f64();
        let duration = self.config.duration();
        if elapsed >= duration {
            return None;
        }
        let at = if self.warming {
            duration - elapsed
        } else {
            elapsed
        };
        let frame = crt_off::render_frame(&self.picture, &self.config, at);
        Some(
            frame
                .pixels()
                .map(|p| u32::from(p[0]) << 16 | u32::from(p[1]) << 8 | u32::from(p[2]))
                .collect(),
        )
    }
}

/// The logo as RGBA8, with its width and height.
fn decode_logo() -> Option<(usize, usize, Vec<u8>)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(LOGO));
    decoder.set_transformations(
        png::Transformations::normalize_to_color8() | png::Transformations::ALPHA,
    );
    let mut reader = decoder.read_info().ok()?;
    let mut buffer = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buffer).ok()?;
    buffer.truncate(info.buffer_size());
    // After the transformations every pixel is four bytes, greyscale ones included.
    let (width, height) = (info.width as usize, info.height as usize);
    (buffer.len() == width * height * 4).then_some((width, height, buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_warm_up_starts_dark_and_ends() {
        let warm_up = Crt::warm_up(64, 48);
        let first = warm_up.frame().expect("it has just started");
        assert!(
            first.iter().filter(|&&p| p != 0).count() < 64 * 48 / 10,
            "mostly dark at first"
        );
        let done = Crt {
            started: std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(10))
                .unwrap(),
            ..warm_up
        };
        assert_eq!(done.frame(), None);
    }

    #[test]
    fn switching_off_starts_on_the_picture() {
        let picture = vec![0x0012_3456; 64 * 48];
        let first = Crt::switch_off(&picture, 64, 48).frame().unwrap();
        assert!(
            first.iter().filter(|&&p| p != 0).count() > 64 * 48 / 2,
            "mostly lit at first"
        );
    }

    #[test]
    fn the_boot_image_has_the_logo_in_the_middle() {
        let (width, height) = (640, 480);
        let pixels = boot_image(width, height);
        assert_eq!(pixels.len(), width * height);
        assert_eq!(pixels[0], 0, "the corner is black");
        let lit = pixels.iter().filter(|&&p| p != 0).count();
        assert!(
            lit > width * height / 20,
            "the logo shows: {lit} pixels lit"
        );
        let green = pixels
            .iter()
            .any(|&p| (p >> 8 & 0xFF) > 150 && (p >> 16 & 0xFF) < 120);
        assert!(green, "the logo's green is there");
    }
}
