//! The video overlay (spec 10.6): a packed 4:2:2 picture, scaled by nearest sampling and
//! converted to RGB.

use serde::{Deserialize, Serialize};

use super::{Bus, Gpu, guest_address};

/// Each overlay buffer's registers are these plus 4 times the buffer number.
const OFFSET: u32 = 0x00_8920;
const SIZE_IN: u32 = 0x00_8928;
const DS_DX: u32 = 0x00_8938;
const DT_DY: u32 = 0x00_8940;
const SIZE_OUT: u32 = 0x00_8950;
const FORMAT: u32 = 0x00_8958;
/// The most source bytes a picture reads.
const MAX_SOURCE: u32 = 16 << 20;
/// 1.0 in the 12.20 fixed point of the steps.
const ONE: u64 = 1 << 20;

/// An overlay picture: its size and RGB bytes, row by row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// BT.601 limited-range YCbCr to RGB.
#[must_use]
pub fn rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let (y, cb, cr) = (i32::from(y) - 16, i32::from(cb) - 128, i32::from(cr) - 128);
    let clamp = |v: i32| u8::try_from(v.clamp(0, 255)).unwrap_or(0);
    [
        clamp((298 * y + 409 * cr + 128) >> 8),
        clamp((298 * y - 100 * cb - 208 * cr + 128) >> 8),
        clamp((298 * y + 516 * cb + 128) >> 8),
    ]
}

/// The picture overlay buffer `buffer` holds, if its source is packed 4:2:2 and readable.
pub(super) fn picture(gpu: &Gpu, buffer: u32, bus: &dyn Bus) -> Option<Picture> {
    let reg = |base: u32| gpu.reg(base + 4 * buffer);
    let format = reg(FORMAT);
    if (format >> 16) & 3 != 1 {
        return None;
    }
    let pitch = format & 0x1FFF;
    let size_in = reg(SIZE_IN);
    let (in_w, in_h) = (size_in & 0x7FF, (size_in >> 16) & 0x7FF);
    if in_w == 0 || in_h == 0 {
        return None;
    }
    let size_out = reg(SIZE_OUT);
    let (out_w, out_h) = match (size_out & 0xFFF, (size_out >> 16) & 0xFFF) {
        (0, _) | (_, 0) => (in_w, in_h),
        size => size,
    };
    let step = |v: u32| if v == 0 { ONE } else { u64::from(v) };
    let (dx, dy) = (step(reg(DS_DX)), step(reg(DT_DY)));
    let len = pitch.checked_mul(in_h).filter(|&n| n <= MAX_SOURCE)?;
    let mut source = vec![0u8; len as usize];
    if !bus.read(guest_address(reg(OFFSET)), &mut source) {
        return None;
    }
    let mut out = Vec::with_capacity((out_w * out_h * 3) as usize);
    for y in 0..u64::from(out_h) {
        let sy = ((y * dy) >> 20).min(u64::from(in_h - 1));
        for x in 0..u64::from(out_w) {
            let sx = ((x * dx) >> 20).min(u64::from(in_w - 1));
            // Y0 Cb Y1 Cr per pixel pair.
            let pair = usize::try_from(sy * u64::from(pitch) + (sx & !1) * 2).unwrap_or(usize::MAX);
            let Some(&[y0, cb, y1, cr]) = source
                .get(pair..pair + 4)
                .and_then(|s| s.first_chunk::<4>())
            else {
                out.extend([0, 0, 0]);
                continue;
            };
            out.extend(rgb(if sx & 1 == 0 { y0 } else { y1 }, cb, cr));
        }
    }
    Some(Picture {
        width: out_w,
        height: out_h,
        rgb: out,
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::FlatBus;
    use super::*;
    use crate::gpu::PVIDEO_BUFFER;

    #[test]
    fn colours_convert_by_bt601_limited_range() {
        assert_eq!(rgb(16, 128, 128), [0, 0, 0]);
        assert_eq!(rgb(235, 128, 128), [255, 255, 255]);
        assert_eq!(rgb(81, 90, 240), [255, 0, 0]);
    }

    #[test]
    fn submitting_a_buffer_makes_a_picture_scaled_by_the_steps() {
        let bus = FlatBus::new();
        let mut gpu = Gpu::new();
        // A 2x1 YUYV source: black then white, at 0x10000.
        bus.put32(0x1_0000, u32::from_le_bytes([16, 128, 235, 128]));
        gpu.write(OFFSET, 4, 0x1_0000, &bus);
        gpu.write(SIZE_IN, 4, 1 << 16 | 2, &bus);
        gpu.write(SIZE_OUT, 4, 1 << 16 | 4, &bus);
        gpu.write(DS_DX, 4, 1 << 19, &bus); // half a source pixel per output pixel
        gpu.write(FORMAT, 4, 1 << 16 | 4, &bus);
        gpu.write(PVIDEO_BUFFER, 4, 1, &bus);
        let pictures = gpu.take_pictures();
        assert_eq!(pictures.len(), 1);
        assert_eq!((pictures[0].width, pictures[0].height), (4, 1));
        assert_eq!(
            pictures[0].rgb,
            [0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255]
        );
        assert_eq!(gpu.read(PVIDEO_BUFFER, 4), 1, "the buffer is in use");
    }
}
