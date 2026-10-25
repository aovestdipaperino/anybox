//! The app's icon: a PNG made from the image the user gives, or, without one, from the title's
//! own image in its executable.

use std::path::Path;
use std::process::Command;

use crate::progress::{Reporter, note};

/// Writes the icon at `source`, a file or an `http(s)` link, to `png` as a PNG.
pub(crate) fn write_icon(source: &str, png: &Path, reporter: &dyn Reporter) -> Result<(), String> {
    let run = |command: &mut Command, what: &str| -> Result<(), String> {
        let output = command.output().map_err(|error| error.to_string())?;
        if output.status.success() {
            Ok(())
        } else {
            let messages = String::from_utf8_lossy(&output.stderr);
            Err(format!("{what} {source} failed\n{}", messages.trim_end()))
        }
    };
    let downloaded = png.with_extension("download");
    let file = if source.starts_with("http://") || source.starts_with("https://") {
        run(
            Command::new("curl")
                .args(["-fsSL", "-o"])
                .arg(&downloaded)
                .arg(source),
            "downloading the icon",
        )?;
        downloaded.as_path()
    } else {
        Path::new(source)
    };
    // sips reads any image format macOS knows, so the source need not be a PNG.
    let converted = run(
        Command::new("sips")
            .args(["-s", "format", "png"])
            .arg(file)
            .arg("--out")
            .arg(png)
            .stdout(std::process::Stdio::null()),
        "converting the icon",
    );
    let _ = std::fs::remove_file(&downloaded);
    converted?;
    // The largest icon macOS uses is 1024 pixels (512 at 2x). A bigger source is scaled down to
    // that so the app does not carry more than it can show; a smaller one is kept as it is, and
    // the packaging scales it up, since no sharper picture exists.
    let size = run_output(
        Command::new("sips")
            .args(["-g", "pixelWidth", "-g", "pixelHeight"])
            .arg(png),
    )
    .and_then(|text| largest_side(&text));
    if size.is_some_and(|side| side > ICON_SIDE) {
        run(
            Command::new("sips")
                .args(["-Z", &ICON_SIDE.to_string()])
                .arg(png)
                .stdout(std::process::Stdio::null()),
            "scaling the icon",
        )?;
    }
    if let Some(side) = size {
        note(
            reporter,
            format!(
                "icon: {side} pixels{}",
                if side < ICON_SIDE {
                    " (smaller than the 1024 macOS can show; it will look soft at the largest sizes)"
                } else {
                    ""
                }
            ),
        );
    }
    Ok(())
}

/// Writes the icon of a title the user gave none for to `png`: the title image the executable
/// carries (`title_image`, its `$$XTIMAGE` section, which the dashboard shows), or a plain one
/// when it has none that reads.
pub(crate) fn write_default_icon(
    title_image: Option<&[u8]>,
    png: &Path,
    reporter: &dyn Reporter,
) -> Result<(), String> {
    let (side, pixels) = if let Some(decoded) = title_image.and_then(decode_title_image) {
        note(reporter, "icon: the title's own image");
        decoded
    } else {
        note(
            reporter,
            "icon: none given and the title has no image; a plain one",
        );
        (
            PLAIN_SIDE,
            vec![PLAIN_COLOUR; (PLAIN_SIDE * PLAIN_SIDE) as usize],
        )
    };
    let bmp = png.with_extension("bmp");
    std::fs::write(&bmp, bitmap(side, side, &pixels)).map_err(|error| error.to_string())?;
    let written = write_icon(&bmp.to_string_lossy(), png, reporter);
    let _ = std::fs::remove_file(&bmp);
    written
}

/// The side, in pixels, of the plain icon: the packaging scales it to every size.
const PLAIN_SIDE: u32 = 256;
/// The plain icon's colour, RGB.
const PLAIN_COLOUR: [u8; 3] = [0x2F, 0x6B, 0x3A];

/// A title image: an `XPR0` resource holding one square DXT1, DXT3 or DXT5 texture. Returns its
/// side and its pixels, RGB, row by row from the top; the alpha is dropped (a transparent DXT1
/// texel is black).
pub(crate) fn decode_title_image(xpr: &[u8]) -> Option<(u32, Vec<[u8; 3]>)> {
    let word = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(xpr.get(at..at + 4)?.try_into().ok()?))
    };
    if xpr.get(..4)? != b"XPR0" {
        return None;
    }
    let data_start = usize::try_from(word(8)?).ok()?;
    // The texture's resource header follows the XPR header: common, data, lock, format, size.
    let format = word(12 + 12)?;
    let block_bytes = match (format >> 8) & 0xFF {
        0x0C => 8,
        0x0E | 0x0F => 16,
        _ => return None,
    };
    let (u, v) = ((format >> 20) & 0xF, (format >> 24) & 0xF);
    if u != v || !(2..=10).contains(&u) {
        return None;
    }
    let side = 1u32 << u;
    let blocks = side / 4;
    let data = xpr.get(data_start..)?;
    let mut pixels = vec![[0u8; 3]; (side * side) as usize];
    for by in 0..blocks {
        for bx in 0..blocks {
            let at = ((by * blocks + bx) * block_bytes) as usize;
            // DXT3 and DXT5 keep their alpha first; the colour half is DXT1's, always opaque.
            let colour = data.get(at + block_bytes as usize - 8..at + block_bytes as usize)?;
            let texels = dxt_colour_block(colour, block_bytes == 8);
            for (index, texel) in (0u32..).zip(texels) {
                let (x, y) = (bx * 4 + index % 4, by * 4 + index / 4);
                pixels[(y * side + x) as usize] = texel;
            }
        }
    }
    Some((side, pixels))
}

/// The sixteen texels of a DXT colour block, row by row. In a DXT1 block whose first colour is
/// not the larger, the third colour is the midpoint and the fourth transparent (black here).
fn dxt_colour_block(block: &[u8], dxt1: bool) -> [[u8; 3]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let rgb = |c: u16| -> [u32; 3] {
        let (r, g, b) = (
            u32::from(c >> 11),
            u32::from((c >> 5) & 0x3F),
            u32::from(c & 0x1F),
        );
        [
            (r << 3) | (r >> 2),
            (g << 2) | (g >> 4),
            (b << 3) | (b >> 2),
        ]
    };
    let (a, b) = (rgb(c0), rgb(c1));
    let mix = |wa: u32, wb: u32| -> [u8; 3] {
        let channel = |i: usize| u8::try_from((a[i] * wa + b[i] * wb) / (wa + wb)).unwrap_or(0);
        [channel(0), channel(1), channel(2)]
    };
    let palette = if c0 > c1 || !dxt1 {
        [mix(1, 0), mix(0, 1), mix(2, 1), mix(1, 2)]
    } else {
        [mix(1, 0), mix(0, 1), mix(1, 1), [0, 0, 0]]
    };
    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    std::array::from_fn(|texel| palette[((indices >> (2 * texel)) & 3) as usize])
}

/// A 24-bit BMP of `pixels`, RGB rows from the top, which `sips` reads.
fn bitmap(width: u32, height: u32, pixels: &[[u8; 3]]) -> Vec<u8> {
    let row = (width * 3).next_multiple_of(4);
    let image = row * height;
    let mut out = Vec::with_capacity(54 + image as usize);
    out.extend(b"BM");
    out.extend((54 + image).to_le_bytes());
    out.extend([0; 4]);
    out.extend(54u32.to_le_bytes());
    out.extend(40u32.to_le_bytes());
    out.extend(width.to_le_bytes());
    out.extend(height.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(24u16.to_le_bytes());
    out.extend([0; 4]);
    out.extend(image.to_le_bytes());
    out.extend([0; 16]);
    // The rows go bottom up, each pixel blue, green, red.
    for y in (0..height).rev() {
        let start = out.len();
        for x in 0..width {
            let [r, g, b] = pixels[(y * width + x) as usize];
            out.extend([b, g, r]);
        }
        out.resize(start + row as usize, 0);
    }
    out
}

/// The side, in pixels, of the largest icon macOS shows.
pub(crate) const ICON_SIDE: u32 = 1024;

/// The output of `command`, or `None` if it did not run or failed.
pub(crate) fn run_output(command: &mut Command) -> Option<String> {
    let output = command.output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The larger of the width and height in the output of `sips -g pixelWidth -g pixelHeight`.
pub(crate) fn largest_side(report: &str) -> Option<u32> {
    report
        .lines()
        .filter_map(|line| {
            let (key, value) = line.trim().split_once(':')?;
            key.starts_with("pixel")
                .then(|| value.trim().parse().ok())?
        })
        .max()
}

#[cfg(test)]
mod icon_tests {
    use super::{bitmap, decode_title_image, largest_side};

    /// An `XPR0` title image of side `1 << log2` in DXT1, every block `block`.
    fn title_image(log2: u32, block: [u8; 8]) -> Vec<u8> {
        let side = 1usize << log2;
        let mut xpr = b"XPR0".to_vec();
        xpr.extend(0u32.to_le_bytes());
        xpr.extend(2048u32.to_le_bytes());
        xpr.extend([0; 12]);
        xpr.extend((0x0000_0C29 | (log2 << 20) | (log2 << 24)).to_le_bytes());
        xpr.resize(2048, 0xAD);
        for _ in 0..(side / 4) * (side / 4) {
            xpr.extend(block);
        }
        xpr
    }

    #[test]
    fn a_dxt1_title_image_decodes_to_its_colours() {
        // Pure red and pure blue, then texels 0, 1, 2 and 3 in the first row.
        let block = [0x00, 0xF8, 0x1F, 0x00, 0b1110_0100, 0, 0, 0];
        let (side, pixels) = decode_title_image(&title_image(3, block)).unwrap();
        assert_eq!(side, 8);
        assert_eq!(pixels.len(), 64);
        assert_eq!(pixels[0], [255, 0, 0]);
        assert_eq!(pixels[1], [0, 0, 255]);
        assert_eq!(pixels[2], [170, 0, 85], "two thirds of the first colour");
        assert_eq!(pixels[3], [85, 0, 170]);
        assert_eq!(pixels[4], [255, 0, 0], "the next block starts over");
        assert_eq!(
            pixels[8],
            [255, 0, 0],
            "the second row's texels are index 0"
        );
        // The first colour the smaller: the third is the midpoint and the fourth black.
        let block = [0x1F, 0x00, 0x00, 0xF8, 0b1110_0100, 0, 0, 0];
        let (_, pixels) = decode_title_image(&title_image(2, block)).unwrap();
        assert_eq!(
            &pixels[..4],
            &[[0, 0, 255], [255, 0, 0], [127, 0, 127], [0, 0, 0]]
        );
    }

    #[test]
    fn what_is_not_a_square_dxt_texture_is_refused() {
        assert_eq!(decode_title_image(b"XPR1"), None);
        let mut swizzled = title_image(3, [0; 8]);
        swizzled[25] = 0x06; // A8R8G8B8
        assert_eq!(decode_title_image(&swizzled), None);
        let truncated = title_image(5, [0; 8]);
        assert_eq!(decode_title_image(&truncated[..2100]), None);
    }

    #[test]
    fn the_bitmap_has_padded_rows_bottom_up() {
        let bmp = bitmap(1, 2, &[[1, 2, 3], [4, 5, 6]]);
        assert_eq!(&bmp[..2], b"BM");
        assert_eq!(bmp.len(), 54 + 8);
        assert_eq!(&bmp[54..62], &[6, 5, 4, 0, 3, 2, 1, 0]);
    }

    #[test]
    fn the_larger_side_is_read_from_a_sips_report() {
        let report = "/tmp/icon.png\n  pixelWidth: 256\n  pixelHeight: 300\n";
        assert_eq!(largest_side(report), Some(300));
        assert_eq!(largest_side("nothing here"), None);
    }
}
