//! Pixel state: what happens to a primitive's fragments and where they land. The surface it is
//! drawn into (its size, anti-aliasing and clipping), clears, the colour mask, the blend
//! equation and constant, the stencil test, face culling, the depth range, polygon modes and
//! colour keys. Also the visibility test's pixel count.
//!
//! Method numbers and their fields are hardware facts; their meaning was written down from the
//! reference runtime's behaviour (`anybox-tk`, bug log rows 28, 31 and 32, see
//! `CLEAN-ROOM-LOG.md`) and implemented here afresh.
//!
//! Register fields and pixel counts are small integers, which become floats freely.
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

use super::Command;
use super::draw::{Kind, Primitive, Vertex};
use super::fifo::KELVIN;
use super::texture::{Mode, Shader};

// Kelvin methods [F].
pub(crate) const SURFACE_CLIP_HORIZONTAL: u32 = 0x0200;
pub(crate) const SURFACE_CLIP_VERTICAL: u32 = 0x0204;
pub(crate) const SURFACE_FORMAT: u32 = 0x0208;
const CONTROL0: u32 = 0x0290;
const LIGHT_CONTROL: u32 = 0x0294;
const WINDOW_CLIP_TYPE: u32 = 0x02B4;
const WINDOW_CLIP_HORIZONTAL: u32 = 0x02C0;
const WINDOW_CLIP_VERTICAL: u32 = 0x02E0;
const CULL_FACE_ENABLE: u32 = 0x0308;
const STENCIL_TEST_ENABLE: u32 = 0x032C;
const BLEND_SOURCE: u32 = 0x0344;
const BLEND_DESTINATION: u32 = 0x0348;
const BLEND_COLOUR: u32 = 0x034C;
const BLEND_EQUATION: u32 = 0x0350;
const COLOUR_MASK: u32 = 0x0358;
const STENCIL_MASK: u32 = 0x0360;
const STENCIL_FUNC: u32 = 0x0364;
const STENCIL_FUNC_REF: u32 = 0x0368;
const STENCIL_FUNC_MASK: u32 = 0x036C;
const STENCIL_OP_FAIL: u32 = 0x0370;
const STENCIL_OP_ZFAIL: u32 = 0x0374;
const STENCIL_OP_ZPASS: u32 = 0x0378;
const FRONT_POLYGON_MODE: u32 = 0x038C;
const BACK_POLYGON_MODE: u32 = 0x0390;
const CLIP_MIN: u32 = 0x0394;
const CLIP_MAX: u32 = 0x0398;
const CULL_FACE: u32 = 0x039C;
const FRONT_FACE: u32 = 0x03A0;
const SPECULAR_ENABLE: u32 = 0x03B8;
const COLOUR_KEY_COLOUR: u32 = 0x0AE0;
const ZMIN_MAX_CONTROL: u32 = 0x1D78;
const ZSTENCIL_CLEAR_VALUE: u32 = 0x1D8C;
const COLOUR_CLEAR_VALUE: u32 = 0x1D90;
pub(crate) const CLEAR_SURFACE: u32 = 0x1D94;
const CLEAR_RECT_HORIZONTAL: u32 = 0x1D98;
const CLEAR_RECT_VERTICAL: u32 = 0x1D9C;
const SET_BEGIN_END: u32 = 0x17FC;

/// GL enums the registers take.
const GL_ALWAYS: u32 = 0x207;
const GL_FUNC_ADD: u32 = 0x8006;
const GL_KEEP: u32 = 0x1E00;
const GL_REPLACE: u32 = 0x1E01;
const GL_CW: u32 = 0x900;
const GL_CCW: u32 = 0x901;
const GL_FRONT: u32 = 0x404;
const GL_BACK: u32 = 0x405;
const GL_FRONT_AND_BACK: u32 = 0x408;
/// `SET_FRONT_POLYGON_MODE` values.
const POLYGON_POINT: u32 = 0x1B00;
const POLYGON_LINE: u32 = 0x1B01;
const POLYGON_FILL: u32 = 0x1B02;

/// The blend factors the factor methods take; any other value is dropped and the factor kept.
const BLEND_FACTORS: [u32; 15] = [
    0, 1, 0x300, 0x301, 0x302, 0x303, 0x304, 0x305, 0x306, 0x307, 0x308, 0x8001, 0x8002, 0x8003,
    0x8004,
];
/// The equations `SET_BLEND_EQUATION` takes: subtract, reverse subtract, add, min, max, then the
/// hardware's signed reverse subtract and signed add.
const BLEND_EQUATIONS: [u32; 7] = [0x800A, 0x800B, 0x8006, 0x8007, 0x8008, 0xF005, 0xF006];

/// `SET_SURFACE_FORMAT` fields: the type in bits 8..11 (2 swizzled), the zeta format in bits
/// 4..7 (1 Z16, 2 Z24S8), the anti-aliasing in bits 12..15.
const SURFACE_SWIZZLED: u32 = 2;
const ZETA_Z16: u32 = 1;
/// `SET_CONTROL0`: bit 12 a floating-point depth buffer, bit 16 w-buffering.
const CONTROL0_Z_FLOAT: u32 = 1 << 12;
const CONTROL0_Z_PERSPECTIVE: u32 = 1 << 16;
/// `SET_LIGHT_CONTROL` bit 17: the specular colour keeps its alpha.
const LIGHT_CONTROL_SPECULAR_ALPHA: u32 = 1 << 17;
/// The largest depth of the floating-point Z16 and Z24 buffers.
const F16_MAX: f32 = 511.9375;
const F24_MAX: f32 = 1.0e30;
/// The largest integer depth of a Z24 buffer.
const Z24_MAX: f32 = 16_777_215.0;
/// `CLEAR_SURFACE` bits: depth, stencil, then red, green, blue and alpha from bit 4.
const CLEAR_Z: u32 = 1;
const CLEAR_STENCIL: u32 = 2;
const CLEAR_COLOUR: u32 = 0xF0;
/// The window clip has eight rectangles.
pub const WINDOW_CLIPS: usize = 8;

/// The surface a primitive is drawn into, in the surface's own pixels: an anti-aliased surface
/// has `samples` of them across and down for each pixel the title's coordinates address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Surface {
    /// Width and height: a linear surface's is its clip's size times the samples, plus the
    /// clip's first pixel; a swizzled one's the format's powers of two times the samples.
    pub size: [u32; 2],
    /// The anti-aliasing factor across and down: 1x1, 2x1 or 2x2.
    pub samples: [u32; 2],
    /// The surface clip as a scissor: first pixel across and down, then width and height.
    pub scissor: [u32; 4],
    pub window: WindowClip,
}

impl Surface {
    /// The size the title's screen coordinates address.
    #[must_use]
    pub fn logical_size(&self) -> [f32; 2] {
        [0, 1].map(|i| self.size[i] as f32 / self.samples[i].max(1) as f32)
    }
}

impl Default for Surface {
    /// A 640x480 surface without anti-aliasing or clipping.
    fn default() -> Self {
        Self {
            size: [640, 480],
            samples: [1, 1],
            scissor: [0, 0, 640, 480],
            window: WindowClip::OPEN,
        }
    }
}

/// The window clip: fragments are drawn only inside one of the rectangles, or with `exclusive`
/// only outside all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowClip {
    pub exclusive: bool,
    /// Each rectangle's first pixel across and down, then the pixel after its last, in the
    /// surface's pixels.
    pub rects: [[u32; 4]; WINDOW_CLIPS],
}

impl WindowClip {
    /// A window clip that clips nothing.
    pub const OPEN: Self = Self {
        exclusive: false,
        rects: [[0, 0, u32::MAX, u32::MAX]; WINDOW_CLIPS],
    };
}

/// The stencil test and what a fragment does to the stencil.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Stencil {
    /// The comparison of the reference with the stencil, both through the read mask (GL enum).
    pub function: u32,
    pub reference: u8,
    pub read_mask: u8,
    /// The stencil bits a fragment may change.
    pub write_mask: u8,
    /// What a fragment that fails the stencil test, passes it but fails the depth test, or
    /// passes both does (GL enums: `0x1E00` keep, 0 zero, `0x1E01` replace, `0x1E02` and
    /// `0x1E03` saturating increment and decrement, `0x150A` invert, `0x8507` and `0x8508`
    /// wrapping increment and decrement).
    pub fail: u32,
    pub depth_fail: u32,
    pub pass: u32,
}

impl Default for Stencil {
    /// Always passes, all bits, keeps the stencil.
    fn default() -> Self {
        Self {
            function: GL_ALWAYS,
            reference: 0,
            read_mask: 0xFF,
            write_mask: 0xFF,
            fail: GL_KEEP,
            depth_fail: GL_KEEP,
            pass: GL_KEEP,
        }
    }
}

/// Which triangles are dropped, by their winding on the screen (x right, y down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cull {
    Clockwise,
    CounterClockwise,
    /// Front and back: every triangle.
    All,
}

/// The depth range fragments are kept in, in 0..1 of the depth buffer (`SET_CLIP_MIN` and
/// `_MAX` over the buffer's largest depth), whether depth outside it is clamped into it
/// (`SET_ZMIN_MAX_CONTROL`) rather than the fragment dropped, and with w-buffering the largest
/// depth a fragment's `w` is divided by.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DepthRange {
    pub min: f32,
    pub max: f32,
    pub clamp: bool,
    pub w_buffer: Option<f32>,
}

impl Default for DepthRange {
    /// The whole buffer, clamped: what a clear writes into.
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            clamp: true,
            w_buffer: None,
        }
    }
}

/// A primitive's pixel state, as the registers left it when it began.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pixel {
    /// The surface drawn into: its size, anti-aliasing and clipping.
    pub target: Surface,
    /// `SET_COLOR_MASK`: alpha in bit 24, red 16, green 8, blue 0.
    pub colour_mask: u32,
    /// `SET_BLEND_EQUATION` (GL enum, or the signed forms `0xF005` and `0xF006`).
    pub blend_equation: u32,
    /// `SET_BLEND_COLOR`, `0xAARRGGBB`.
    pub blend_colour: u32,
    /// The stencil test, when on.
    pub stencil: Option<Stencil>,
    /// The faces culled, when culling is on.
    pub cull: Option<Cull>,
    pub depth_range: DepthRange,
    /// `SET_COLOR_KEY_COLOR` of the four stages.
    pub colour_keys: [u32; 4],
}

impl Default for Pixel {
    fn default() -> Self {
        PixelRegs::default().pixel()
    }
}

/// The pixel registers, as the commands so far left them.
#[derive(Debug, Clone)]
pub(crate) struct PixelRegs {
    surface_format: u32,
    surface_clip: [u32; 2],
    window_type: u32,
    window_horizontal: [u32; WINDOW_CLIPS],
    window_vertical: [u32; WINDOW_CLIPS],
    colour_mask: u32,
    /// Blend source and destination factors, kept across invalid values.
    pub(crate) blend: (u32, u32),
    blend_equation: u32,
    blend_colour: u32,
    stencil_on: bool,
    stencil: Stencil,
    cull: (bool, u32, u32),
    specular: (bool, u32),
    polygon_modes: [u32; 2],
    control0: u32,
    clip_range: [f32; 2],
    zmin_max_control: u32,
    zeta_clear: u32,
    colour_clear: u32,
    clear_rect: [u32; 2],
    colour_keys: [u32; 4],
}

impl Default for PixelRegs {
    fn default() -> Self {
        Self {
            // Linear, Z24S8, A8R8G8B8.
            surface_format: 0x128,
            surface_clip: [640 << 16, 480 << 16],
            window_type: 0,
            // Open rectangles, so that commands from before the title set them draw.
            window_horizontal: [0x0FFF_0000; WINDOW_CLIPS],
            window_vertical: [0x0FFF_0000; WINDOW_CLIPS],
            colour_mask: 0x0101_0101,
            blend: (0x302, 0x303),
            blend_equation: GL_FUNC_ADD,
            blend_colour: 0,
            stencil_on: false,
            stencil: Stencil::default(),
            cull: (false, GL_BACK, GL_CW),
            specular: (false, 0),
            polygon_modes: [POLYGON_FILL; 2],
            control0: 0,
            // Until the title sets it, the clip maximum is the buffer's largest depth, whatever
            // the buffer's format.
            clip_range: [0.0, f32::INFINITY],
            zmin_max_control: 0,
            zeta_clear: 0xFFFF_FF00,
            colour_clear: 0,
            clear_rect: [639 << 16, 479 << 16],
            colour_keys: [0; 4],
        }
    }
}

impl PixelRegs {
    /// Takes one method; the others leave the pixel state alone.
    pub(crate) fn set(&mut self, method: u32, value: u32) {
        let float = f32::from_bits(value);
        let [byte, ..] = value.to_le_bytes();
        match method {
            SURFACE_FORMAT => self.surface_format = value,
            SURFACE_CLIP_HORIZONTAL => self.surface_clip[0] = value,
            SURFACE_CLIP_VERTICAL => self.surface_clip[1] = value,
            WINDOW_CLIP_TYPE => self.window_type = value,
            // A rectangle's word also fills the slots after it.
            m if (WINDOW_CLIP_HORIZONTAL..WINDOW_CLIP_HORIZONTAL + 32).contains(&m) => {
                let slot = ((m - WINDOW_CLIP_HORIZONTAL) / 4) as usize;
                self.window_horizontal[slot..].fill(value);
            }
            m if (WINDOW_CLIP_VERTICAL..WINDOW_CLIP_VERTICAL + 32).contains(&m) => {
                let slot = ((m - WINDOW_CLIP_VERTICAL) / 4) as usize;
                self.window_vertical[slot..].fill(value);
            }
            COLOUR_MASK => self.colour_mask = value,
            BLEND_SOURCE if BLEND_FACTORS.contains(&value) => self.blend.0 = value,
            BLEND_DESTINATION if BLEND_FACTORS.contains(&value) => self.blend.1 = value,
            BLEND_EQUATION if BLEND_EQUATIONS.contains(&value) => self.blend_equation = value,
            BLEND_COLOUR => self.blend_colour = value,
            STENCIL_TEST_ENABLE => self.stencil_on = value != 0,
            // The comparison is the enum's low nibble; reference and masks are bytes.
            STENCIL_FUNC => self.stencil.function = 0x200 | value & 0xF,
            STENCIL_FUNC_REF => self.stencil.reference = byte,
            STENCIL_FUNC_MASK => self.stencil.read_mask = byte,
            STENCIL_MASK => self.stencil.write_mask = byte,
            STENCIL_OP_FAIL => self.stencil.fail = value,
            STENCIL_OP_ZFAIL => self.stencil.depth_fail = value,
            STENCIL_OP_ZPASS => self.stencil.pass = value,
            CULL_FACE_ENABLE => self.cull.0 = value != 0,
            CULL_FACE => self.cull.1 = value,
            FRONT_FACE => self.cull.2 = value,
            SPECULAR_ENABLE => self.specular.0 = value != 0,
            LIGHT_CONTROL => self.specular.1 = value,
            FRONT_POLYGON_MODE | BACK_POLYGON_MODE
                if (POLYGON_POINT..=POLYGON_FILL).contains(&value) =>
            {
                self.polygon_modes[usize::from(method == BACK_POLYGON_MODE)] = value;
            }
            CONTROL0 => self.control0 = value,
            CLIP_MIN => self.clip_range[0] = float,
            CLIP_MAX => self.clip_range[1] = float,
            ZMIN_MAX_CONTROL => self.zmin_max_control = value,
            ZSTENCIL_CLEAR_VALUE => self.zeta_clear = value,
            COLOUR_CLEAR_VALUE => self.colour_clear = value,
            CLEAR_RECT_HORIZONTAL => self.clear_rect[0] = value,
            CLEAR_RECT_VERTICAL => self.clear_rect[1] = value,
            m if (COLOUR_KEY_COLOUR..COLOUR_KEY_COLOUR + 16).contains(&m) => {
                self.colour_keys[((m - COLOUR_KEY_COLOUR) / 4) as usize] = value;
            }
            _ => {}
        }
    }

    /// The largest depth of the depth buffer, which depth is divided by to bring it to 0..1:
    /// the integer maximum of Z16 or Z24, or the largest float of the floating-point formats.
    pub(crate) fn zmax(&self) -> f32 {
        let float = self.control0 & CONTROL0_Z_FLOAT != 0;
        match (self.zeta_format() == ZETA_Z16, float) {
            (true, false) => f32::from(u16::MAX),
            (true, true) => F16_MAX,
            (false, false) => Z24_MAX,
            (false, true) => F24_MAX,
        }
    }

    fn zeta_format(&self) -> u32 {
        (self.surface_format >> 4) & 0xF
    }

    /// How the second colour reaches the combiners: `None` while specular is off (it is then
    /// black with alpha 1), else whether its alpha is replaced by 1.
    pub(crate) fn specular(&self) -> Option<bool> {
        let (on, light_control) = self.specular;
        on.then_some(light_control & LIGHT_CONTROL_SPECULAR_ALPHA == 0)
    }

    /// The anti-aliasing factor of the surface format, across and down.
    fn samples(&self) -> [u32; 2] {
        match (self.surface_format >> 12) & 0xF {
            1 => [2, 1],
            2 => [2, 2],
            _ => [1, 1],
        }
    }

    /// The surface the next primitives draw into. A linear surface's size is its clip's,
    /// scaled by the anti-aliasing factor, with the clip's first pixel added after the scale.
    fn target(&self) -> Surface {
        let samples = self.samples();
        let [[x, width], [y, height]] = self.surface_clip.map(|word| [word & 0xFFFF, word >> 16]);
        let size = if (self.surface_format >> 8) & 0xF == SURFACE_SWIZZLED {
            let side = |shift: u32| 1u32 << ((self.surface_format >> shift) & 0xFF).min(12);
            [side(16) * samples[0], side(24) * samples[1]]
        } else {
            [width * samples[0] + x, height * samples[1] + y]
        };
        let mut rects = [[0; 4]; WINDOW_CLIPS];
        for (rect, (h, v)) in rects
            .iter_mut()
            .zip(self.window_horizontal.iter().zip(&self.window_vertical))
        {
            // First and last pixel in 12-bit fields at bits 0 and 16.
            let edges = |word: u32, scale: u32| {
                [(word & 0xFFF) * scale, (((word >> 16) & 0xFFF) + 1) * scale]
            };
            let ([left, right], [top, bottom]) = (edges(*h, samples[0]), edges(*v, samples[1]));
            *rect = [left, top, right, bottom];
        }
        Surface {
            size: size.map(|side| side.max(1)),
            samples,
            scissor: [
                x * samples[0],
                y * samples[1],
                width * samples[0],
                height * samples[1],
            ],
            window: WindowClip {
                exclusive: self.window_type & 1 != 0,
                rects,
            },
        }
    }

    /// The faces culled: the culled face is the back one unless it is the front, and the front
    /// face winds clockwise unless it is `GL_CCW`.
    fn cull(&self) -> Option<Cull> {
        let (on, face, front) = self.cull;
        if !on {
            return None;
        }
        let front_clockwise = front != GL_CCW;
        Some(match face {
            GL_FRONT_AND_BACK => Cull::All,
            GL_FRONT if front_clockwise => Cull::Clockwise,
            GL_FRONT => Cull::CounterClockwise,
            _ if front_clockwise => Cull::CounterClockwise,
            _ => Cull::Clockwise,
        })
    }

    fn depth_range(&self) -> DepthRange {
        let zmax = self.zmax();
        DepthRange {
            min: self.clip_range[0] / zmax,
            max: if self.clip_range[1].is_finite() {
                self.clip_range[1] / zmax
            } else {
                1.0
            },
            clamp: (self.zmin_max_control >> 4) & 0xF == 1,
            w_buffer: (self.control0 & CONTROL0_Z_PERSPECTIVE != 0).then_some(zmax),
        }
    }

    /// The pixel state of a primitive beginning now.
    pub(crate) fn pixel(&self) -> Pixel {
        Pixel {
            target: self.target(),
            colour_mask: self.colour_mask,
            blend_equation: self.blend_equation,
            blend_colour: self.blend_colour,
            stencil: self.stencil_on.then_some(self.stencil),
            cull: self.cull(),
            depth_range: self.depth_range(),
            colour_keys: self.colour_keys,
        }
    }

    /// `SET_COLOR_CLEAR_VALUE` as RGBA in 0..1, read in the surface's colour format; `None` for
    /// the one- and two-channel formats, which are not cleared.
    fn clear_colour(&self) -> Option<[f32; 4]> {
        let value = self.colour_clear;
        let field = |shift: u32, bits: u32| {
            let max = (1u32 << bits) - 1;
            ((value >> shift) & max) as f32 / max as f32
        };
        let format = self.surface_format & 0xF;
        let [r, g, b] = match format {
            1 | 2 => [field(10, 5), field(5, 5), field(0, 5)],
            3 => [field(11, 5), field(5, 6), field(0, 5)],
            4..=8 => [field(16, 8), field(8, 8), field(0, 8)],
            _ => return None,
        };
        let seven = (value >> 24) & 0x7F;
        let a = match format {
            // X1R5G5B5 and X8R8G8B8 with the unused bits zero.
            1 | 4 => 0.0,
            // X1A7R8G8B8: seven bits of alpha, the top one zero or one.
            6 => seven as f32 / 255.0,
            7 => (0x80 | seven) as f32 / 255.0,
            8 => field(24, 8),
            _ => 1.0,
        };
        Some([r, g, b, a])
    }

    /// The depth and stencil a clear writes: depth then stencil for Z24S8, depth alone for
    /// Z16, as a fraction of the buffer's largest depth.
    fn clear_depth_stencil(&self) -> (f32, u8) {
        let [s, z0, z1, z2] = self.zeta_clear.to_le_bytes();
        let float = self.control0 & CONTROL0_Z_FLOAT != 0;
        if self.zeta_format() == ZETA_Z16 {
            let z = u16::from_le_bytes([s, z0]);
            let depth = if float {
                f16_depth(z) / F16_MAX
            } else {
                f32::from(z) / f32::from(u16::MAX)
            };
            (depth, 0)
        } else if float {
            (f24_depth(u32::from_le_bytes([z0, z1, z2, 0])) / F24_MAX, s)
        } else {
            let z = u32::from_le_bytes([z0, z1, z2, 0]);
            (z as f32 / Z24_MAX, s)
        }
    }

    /// `CLEAR_SURFACE` with `flags` as a quad over the clear rectangle, which is also its
    /// scissor (neither the surface clip nor the window clip applies): it writes the clear depth
    /// where depth is cleared, the clear stencil where the stencil is, and the clear colour in
    /// the channels the colour bits name, in colour buffer `surface`. `None` when nothing is
    /// cleared.
    pub(crate) fn clear(&self, flags: u32, surface: u32) -> Option<Primitive> {
        let colour = if flags & CLEAR_COLOUR == 0 {
            None
        } else {
            self.clear_colour()
        };
        if flags & (CLEAR_Z | CLEAR_STENCIL) == 0 && colour.is_none() {
            return None;
        }
        let colour_mask = [(4, 16), (5, 8), (6, 0), (7, 24)]
            .iter()
            .filter(|(bit, _)| colour.is_some() && flags & 1 << bit != 0)
            .fold(0, |mask, (_, shift)| mask | 1 << shift);
        let (depth, stencil) = self.clear_depth_stencil();
        // First and last pixel in 12-bit fields at bits 0 and 16, in the title's pixels.
        let first = |word: u32| word & 0xFFF;
        let last = |word: u32| (word >> 16) & 0xFFF;
        let [h, v] = self.clear_rect;
        let (left, right) = (first(h) as f32, last(h) as f32 + 1.0);
        let (top, bottom) = (first(v) as f32, last(v) as f32 + 1.0);
        let target = self.target();
        let [sx, sy] = target.samples;
        let span = |word: u32, scale: u32| {
            [
                first(word) * scale,
                (last(word) + 1).saturating_sub(first(word)) * scale,
            ]
        };
        let ([x, width], [y, height]) = (span(h, sx), span(v, sy));
        let corner = |x: f32, y: f32| Vertex {
            x,
            y,
            depth,
            w: 1.0,
            colour: colour.unwrap_or([0.0; 4]),
            texture: [[0.0, 0.0, 0.0, 1.0]; 4],
            fog: 1.0,
            specular: [0.0, 0.0, 0.0, 1.0],
        };
        Some(Primitive {
            kind: Kind::Quads,
            vertices: vec![
                corner(left, top),
                corner(right, top),
                corner(right, bottom),
                corner(left, bottom),
            ],
            textures: [None; 4],
            // Untextured stages pass white, so the fragment is the clear colour.
            modes: [Mode::Sample; 4],
            shader: Shader::default(),
            surface,
            depth: Some(super::draw::Depth {
                function: GL_ALWAYS,
                write: flags & CLEAR_Z != 0,
            }),
            blend: None,
            fog: None,
            combiner: None,
            alpha_test: None,
            point_size: 1.0,
            pixel: Pixel {
                target: Surface {
                    scissor: [x, y, width, height],
                    window: WindowClip::OPEN,
                    ..target
                },
                colour_mask,
                blend_equation: GL_FUNC_ADD,
                blend_colour: 0,
                stencil: (flags & CLEAR_STENCIL != 0).then_some(Stencil {
                    reference: stencil,
                    pass: GL_REPLACE,
                    ..Stencil::default()
                }),
                // A clear covers its rectangle whichever way it winds.
                cull: None,
                depth_range: DepthRange::default(),
                colour_keys: [0; 4],
            },
            clear: true,
        })
    }

    /// The polygon modes and front face a primitive beginning now is rasterised with.
    pub(crate) fn raster(&self) -> Raster {
        Raster {
            polygon_modes: self.polygon_modes,
            front_clockwise: self.cull.2 != GL_CCW,
        }
    }
}

/// A floating-point Z16 depth: no sign, five exponent bits and eleven of mantissa.
fn f16_depth(z: u16) -> f32 {
    if z == 0 {
        return 0.0;
    }
    f32::from_bits((u32::from(z) << 11) + 0x3C00_0000)
}

/// A floating-point Z24 depth: a float without its sign and its lowest seven mantissa bits.
fn f24_depth(z: u32) -> f32 {
    f32::from_bits((z & 0xFF_FFFF) << 7)
}

/// The second colour as the rasteriser takes it: clamped to 0..1, NaN as 1, its alpha 1 when the
/// title drops it.
#[must_use]
pub fn specular_output(value: [f32; 4], ignore_alpha: bool) -> [f32; 4] {
    let mut out = value.map(|c| if c.is_nan() { 1.0 } else { c.clamp(0.0, 1.0) });
    if ignore_alpha {
        out[3] = 1.0;
    }
    out
}

/// The second colour while specular is off.
pub const NO_SPECULAR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// How a primitive's faces become what is drawn: the polygon modes of its front and back faces
/// and which winding is the front.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Raster {
    polygon_modes: [u32; 2],
    front_clockwise: bool,
}

/// Whether the face through `face`'s vertices winds clockwise on the screen (y down).
fn winds_clockwise(vertices: &[Vertex], face: &[usize]) -> bool {
    let area: f32 = (0..face.len())
        .map(|i| {
            let (a, b) = (&vertices[face[i]], &vertices[face[(i + 1) % face.len()]]);
            a.x * b.y - b.x * a.y
        })
        .sum();
    area > 0.0
}

impl Raster {
    /// Pushes what `primitive` draws to `out`: as it is when its faces are filled (or it has
    /// none), else each face filled, or as its edges (lines) or corners (points) by the mode of
    /// the side it shows. Faces drawn as lines or points are culled here, as the hardware culls
    /// before it rasterises, since lines and points are never culled later. Corners become
    /// squares of the point size, and edges wider than a pixel (`line_width`) quads.
    pub(crate) fn assemble(
        self,
        mut primitive: Primitive,
        line_width: f32,
        out: &mut Vec<Primitive>,
    ) {
        let fill = self.polygon_modes == [POLYGON_FILL; 2];
        let n = primitive.vertices.len();
        let faces: Vec<Vec<usize>> = match primitive.kind {
            _ if fill => Vec::new(),
            Kind::Triangles => (0..n / 3)
                .map(|t| vec![3 * t, 3 * t + 1, 3 * t + 2])
                .collect(),
            Kind::TriangleStrip => (2..n)
                .map(|i| {
                    if i % 2 == 0 {
                        vec![i - 2, i - 1, i]
                    } else {
                        vec![i - 1, i - 2, i]
                    }
                })
                .collect(),
            Kind::TriangleFan => (2..n).map(|i| vec![0, i - 1, i]).collect(),
            Kind::Quads => (0..n / 4).map(|q| (4 * q..4 * q + 4).collect()).collect(),
            _ => Vec::new(),
        };
        if faces.is_empty() {
            out.push(primitive);
            return;
        }
        let (mut filled, mut lines, mut points) = (Vec::new(), Vec::new(), Vec::new());
        for face in faces {
            let mode = if self.polygon_modes[0] == self.polygon_modes[1] {
                self.polygon_modes[0]
            } else {
                let clockwise = winds_clockwise(&primitive.vertices, &face);
                self.polygon_modes[usize::from(clockwise != self.front_clockwise)]
            };
            if mode == POLYGON_FILL {
                filled.extend((2..face.len()).flat_map(|i| [face[0], face[i - 1], face[i]]));
                continue;
            }
            let culled = match primitive.pixel.cull {
                None => false,
                Some(Cull::All) => true,
                Some(Cull::Clockwise) => winds_clockwise(&primitive.vertices, &face),
                Some(Cull::CounterClockwise) => !winds_clockwise(&primitive.vertices, &face),
            };
            if culled {
                continue;
            }
            if mode == POLYGON_LINE {
                lines.extend((0..face.len()).flat_map(|i| [face[i], face[(i + 1) % face.len()]]));
            } else {
                points.extend(face);
            }
        }
        let vertices = std::mem::take(&mut primitive.vertices);
        let mut with = |kind: Kind, indices: &[usize], cull: Option<Cull>| {
            if !indices.is_empty() {
                let mut part = primitive.clone();
                part.kind = kind;
                part.vertices = indices.iter().map(|&i| vertices[i]).collect();
                part.pixel.cull = cull;
                out.push(part);
            }
        };
        let cull = primitive.pixel.cull;
        with(Kind::Triangles, &filled, cull);
        if line_width > 1.0 {
            let quads: Vec<Vertex> = lines
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&[a, b]| super::draw::wide_line(vertices[a], vertices[b], line_width))
                .collect();
            push_quads(&primitive, quads, out);
        } else {
            with(Kind::Lines, &lines, None);
        }
        let size = primitive.point_size;
        let squares: Vec<Vertex> = points
            .iter()
            .flat_map(|&i| super::draw::point_quad(vertices[i], size))
            .collect();
        push_quads(&primitive, squares, out);
    }
}

/// Pushes `primitive` as the quads `vertices` make, which are never culled, if there are any.
fn push_quads(primitive: &Primitive, vertices: Vec<Vertex>, out: &mut Vec<Primitive>) {
    if !vertices.is_empty() {
        let mut part = primitive.clone();
        part.kind = Kind::Quads;
        part.vertices = vertices;
        part.pixel.cull = None;
        out.push(part);
    }
}

/// What a visibility test reports when its draws decode to nothing: a whole 640x480 surface,
/// so that a title never takes what the model cannot see for hidden.
pub const WHOLE_SURFACE: u32 = 640 * 480;

/// The pixels the draws in `commands` cover, as a visibility test counts them: each triangle's
/// pixel centres inside the surface and its scissor, after clipping at the eye and, when the
/// depth range drops what is outside it, at the range, times the samples of an anti-aliased
/// surface. Culled faces count nothing; a pixel two triangles cover counts twice. The depth and
/// stencil tests, the alpha test and colour keys are not modelled, so a hidden fragment counts.
/// Draws that decode to nothing count as [`WHOLE_SURFACE`].
#[must_use]
pub fn visible_pixels(commands: &[Command]) -> u32 {
    let draws = commands
        .iter()
        .any(|c| c.class == KELVIN && c.method == SET_BEGIN_END && c.value != 0);
    let found = super::draw::primitives(commands);
    if draws && found.iter().all(|p| p.vertices.is_empty()) {
        return WHOLE_SURFACE;
    }
    let total: u64 = found
        .iter()
        .map(|p| {
            let [sx, sy] = p.pixel.target.samples;
            u64::from(sx.max(1)) * u64::from(sy.max(1)) * logical_pixels(p)
        })
        .sum();
    u32::try_from(total).unwrap_or(u32::MAX)
}

/// A clip-space position: pixels times `w`, depth times `w`, `w`.
type Clipped = [f32; 4];

/// The pixels one primitive covers in the title's coordinates, inside the surface's scissor.
fn logical_pixels(primitive: &Primitive) -> u64 {
    let target = primitive.pixel.target;
    let size = target.logical_size();
    let logical = |value: u32, axis: usize| (value / target.samples[axis].max(1)) as f32;
    let [x, y, width, height] = target.scissor;
    let bounds = [
        logical(x, 0).min(size[0]),
        logical(y, 1).min(size[1]),
        logical(x.saturating_add(width), 0).min(size[0]),
        logical(y.saturating_add(height), 1).min(size[1]),
    ];
    let range = primitive.pixel.depth_range;
    let planes = clip_planes(
        bounds,
        (!range.clamp && range.w_buffer.is_none()).then_some([range.min, range.max]),
    );
    let clip = |v: &Vertex| [v.x * v.w, v.y * v.w, v.depth * v.w, v.w];
    let vertices = &primitive.vertices;
    if primitive.kind == Kind::Lines {
        return vertices
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[a, b]| line_pixels([clip(a), clip(b)], &planes, bounds))
            .sum();
    }
    triangles(primitive.kind, vertices.len())
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let polygon = clip_polygon(t.iter().map(|&i| clip(&vertices[i])).collect(), &planes);
            let screen: Vec<[f32; 2]> =
                polygon.iter().map(|v| [v[0] / v[3], v[1] / v[3]]).collect();
            if screen.len() < 3 {
                return 0;
            }
            let area: f32 = (0..screen.len())
                .map(|i| {
                    let ([x0, y0], [x1, y1]) = (screen[i], screen[(i + 1) % screen.len()]);
                    x0 * y1 - x1 * y0
                })
                .sum();
            let culled = match primitive.pixel.cull {
                None => false,
                Some(Cull::All) => true,
                Some(Cull::Clockwise) => area > 0.0,
                Some(Cull::CounterClockwise) => area < 0.0,
            };
            if culled {
                return 0;
            }
            (2..screen.len())
                .map(|i| triangle_pixels([screen[0], screen[i - 1], screen[i]], bounds))
                .sum()
        })
        .sum()
}

/// A primitive's triangles as index triples, strips wound as their first triangle.
fn triangles(kind: Kind, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    match kind {
        Kind::Triangles => out.extend(0..n - n % 3),
        Kind::TriangleStrip => {
            for i in 2..n {
                if i % 2 == 0 {
                    out.extend([i - 2, i - 1, i]);
                } else {
                    out.extend([i - 1, i - 2, i]);
                }
            }
        }
        Kind::TriangleFan => (2..n).for_each(|i| out.extend([0, i - 1, i])),
        Kind::Quads => {
            for b in (0..n - n % 4).step_by(4) {
                out.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
            }
        }
        _ => {}
    }
    out
}

/// The planes a clip-space position must be on the inner side of, each with how far inside it
/// must be: in front of the eye (a hair off it), inside `bounds` (left, top, right, bottom),
/// and inside the depth range when one drops what is outside.
fn clip_planes(bounds: [f32; 4], range: Option<[f32; 2]>) -> Vec<([f32; 4], f32)> {
    let [left, top, right, bottom] = bounds;
    let mut planes = vec![
        ([0.0, 0.0, 0.0, 1.0], 1e-6),
        ([1.0, 0.0, 0.0, -left], 0.0),
        ([-1.0, 0.0, 0.0, right], 0.0),
        ([0.0, 1.0, 0.0, -top], 0.0),
        ([0.0, -1.0, 0.0, bottom], 0.0),
    ];
    if let Some([near, far]) = range {
        planes.push(([0.0, 0.0, 1.0, -near], 0.0));
        planes.push(([0.0, 0.0, -1.0, far], 0.0));
    }
    planes
}

/// How far `v` is on the inner side of a plane.
fn distance((plane, margin): ([f32; 4], f32), v: Clipped) -> f32 {
    plane.iter().zip(v).map(|(p, c)| p * c).sum::<f32>() - margin
}

/// A convex polygon clipped to the planes, one at a time.
fn clip_polygon(mut polygon: Vec<Clipped>, planes: &[([f32; 4], f32)]) -> Vec<Clipped> {
    if polygon.iter().flatten().any(|c| !c.is_finite()) {
        return Vec::new();
    }
    for &plane in planes {
        let mut kept = Vec::with_capacity(polygon.len() + 1);
        for i in 0..polygon.len() {
            let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
            let (da, db) = (distance(plane, a), distance(plane, b));
            if da >= 0.0 {
                kept.push(a);
            }
            if (da >= 0.0) != (db >= 0.0) {
                let t = da / (da - db);
                kept.push(std::array::from_fn(|c| a[c] + t * (b[c] - a[c])));
            }
        }
        polygon = kept;
        if polygon.is_empty() {
            break;
        }
    }
    polygon
}

/// The pixel centres inside a triangle on the surface. A centre on an edge counts for one of
/// the two triangles that share it (the top-left rule).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn triangle_pixels(corners: [[f32; 2]; 3], bounds: [f32; 4]) -> u64 {
    let [a, b, c] = corners;
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if !area.is_normal() {
        return 0;
    }
    // Wound so that every edge function is positive inside.
    let [a, b, c] = if area > 0.0 { [a, b, c] } else { [a, c, b] };
    let edges = [(a, b), (b, c), (c, a)];
    // Held to the surface first, so the conversion is exact.
    let pixel = |value: f32, low: f32, high: f32| value.clamp(low.max(0.0), high.max(0.0)) as u32;
    let lowest = |i: usize| a[i].min(b[i]).min(c[i]).floor();
    let highest = |i: usize| a[i].max(b[i]).max(c[i]).ceil();
    let (left, right) = (
        pixel(lowest(0), bounds[0], bounds[2]),
        pixel(highest(0), bounds[0], bounds[2]),
    );
    let (top, bottom) = (
        pixel(lowest(1), bounds[1], bounds[3]),
        pixel(highest(1), bounds[1], bounds[3]),
    );
    let mut count = 0;
    for y in top..bottom {
        let py = y as f32 + 0.5;
        for x in left..right {
            let px = x as f32 + 0.5;
            let inside = edges.iter().all(|&(p, q)| {
                let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
                let e = dx * (py - p[1]) - dy * (px - p[0]);
                e > 0.0 || (e == 0.0 && (dy < 0.0 || (dy == 0.0 && dx > 0.0)))
            });
            count += u64::from(inside);
        }
    }
    count
}

/// The pixels a line between two clip-space ends covers: one a step along its longer axis.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn line_pixels(mut ends: [Clipped; 2], planes: &[([f32; 4], f32)], bounds: [f32; 4]) -> u64 {
    if ends.iter().flatten().any(|c| !c.is_finite()) {
        return 0;
    }
    for &plane in planes {
        let (da, db) = (distance(plane, ends[0]), distance(plane, ends[1]));
        if da < 0.0 && db < 0.0 {
            return 0;
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            let cut = std::array::from_fn(|c| ends[0][c] + t * (ends[1][c] - ends[0][c]));
            ends[usize::from(da >= 0.0)] = cut;
        }
    }
    let at = |e: Clipped| [e[0] / e[3], e[1] / e[3]];
    let ([x0, y0], [x1, y1]) = (at(ends[0]), at(ends[1]));
    let steps = (x1 - x0).abs().max((y1 - y0).abs()).round();
    if steps.is_finite() {
        steps.clamp(
            1.0,
            (bounds[2] - bounds[0] + bounds[3] - bounds[1]).max(1.0),
        ) as u64
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::draw::primitives;

    fn kelvin(method: u32, value: u32) -> Command {
        Command {
            class: KELVIN,
            method,
            value,
        }
    }

    fn float(method: u32, value: f32) -> Command {
        kelvin(method, value.to_bits())
    }

    /// Position: three floats.
    fn position() -> Command {
        kelvin(0x1760, 3 << 4 | 2)
    }

    /// The square from `(x, y)` to `(x + size, y + size)` as a quad, wound clockwise on the
    /// screen.
    fn square(x: f32, y: f32, size: f32) -> Vec<Command> {
        let mut c = vec![kelvin(SET_BEGIN_END, 8)];
        for (cx, cy) in [(x, y), (x + size, y), (x + size, y + size), (x, y + size)] {
            c.extend([float(0x1818, cx), float(0x1818, cy), float(0x1818, 0.0)]);
        }
        c.push(kelvin(SET_BEGIN_END, 0));
        c
    }

    fn drawn_after(state: &[Command]) -> Primitive {
        let mut commands = vec![position()];
        commands.extend_from_slice(state);
        commands.extend(square(0.0, 0.0, 10.0));
        primitives(&commands).pop().expect("a primitive")
    }

    #[test]
    fn blend_factors_and_equations_outside_the_hardwares_are_dropped() {
        let p = drawn_after(&[
            kelvin(0x0304, 1),
            kelvin(BLEND_SOURCE, 0x8003),
            kelvin(BLEND_DESTINATION, 0x1234),
            kelvin(BLEND_EQUATION, 0xF005),
            kelvin(BLEND_EQUATION, 0x9999),
            kelvin(BLEND_COLOUR, 0x80FF_0033),
            kelvin(COLOUR_MASK, 0x0100_0000),
        ]);
        assert_eq!(p.blend, Some((0x8003, 0x303)));
        assert_eq!(p.pixel.blend_equation, 0xF005);
        assert_eq!(p.pixel.blend_colour, 0x80FF_0033);
        assert_eq!(p.pixel.colour_mask, 0x0100_0000);
    }

    #[test]
    fn a_primitive_carries_the_stencil_test_while_it_is_on() {
        let state = [
            kelvin(STENCIL_FUNC, 0x202),
            kelvin(STENCIL_FUNC_REF, 0x180),
            kelvin(STENCIL_FUNC_MASK, 0xF0),
            kelvin(STENCIL_MASK, 0x0F),
            kelvin(STENCIL_OP_FAIL, 0x8507),
            kelvin(STENCIL_OP_ZFAIL, 0x8508),
            kelvin(STENCIL_OP_ZPASS, 0x1E02),
        ];
        assert_eq!(drawn_after(&state).pixel.stencil, None);
        let mut on = state.to_vec();
        on.push(kelvin(STENCIL_TEST_ENABLE, 1));
        assert_eq!(
            drawn_after(&on).pixel.stencil,
            Some(Stencil {
                function: 0x202,
                reference: 0x80,
                read_mask: 0xF0,
                write_mask: 0x0F,
                fail: 0x8507,
                depth_fail: 0x8508,
                pass: 0x1E02,
            })
        );
    }

    #[test]
    fn culling_follows_the_culled_face_and_the_front_face() {
        let cull = |face, front| {
            drawn_after(&[
                kelvin(CULL_FACE_ENABLE, 1),
                kelvin(CULL_FACE, face),
                kelvin(FRONT_FACE, front),
            ])
            .pixel
            .cull
        };
        assert_eq!(drawn_after(&[]).pixel.cull, None);
        assert_eq!(cull(GL_BACK, GL_CW), Some(Cull::CounterClockwise));
        assert_eq!(cull(GL_BACK, GL_CCW), Some(Cull::Clockwise));
        assert_eq!(cull(GL_FRONT, GL_CW), Some(Cull::Clockwise));
        assert_eq!(cull(GL_FRONT_AND_BACK, GL_CW), Some(Cull::All));
    }

    #[test]
    fn the_surface_is_sized_by_its_clip_and_anti_aliasing() {
        let p = drawn_after(&[
            kelvin(SURFACE_CLIP_HORIZONTAL, 320 << 16 | 8),
            kelvin(SURFACE_CLIP_VERTICAL, 240 << 16 | 4),
            kelvin(SURFACE_FORMAT, 0x2128),
        ]);
        let target = p.pixel.target;
        assert_eq!(target.size, [648, 484]);
        assert_eq!(target.samples, [2, 2]);
        assert_eq!(target.scissor, [16, 8, 640, 480]);
        assert_eq!(target.logical_size(), [324.0, 242.0]);
        // A swizzled surface is its powers of two.
        let p = drawn_after(&[kelvin(SURFACE_FORMAT, 7 << 24 | 8 << 16 | 0x1228)]);
        assert_eq!(p.pixel.target.size, [512, 128]);
        assert_eq!(p.pixel.target.samples, [2, 1]);
    }

    #[test]
    fn the_window_clip_rectangles_fill_the_slots_after_them_and_scale_with_the_samples() {
        let p = drawn_after(&[
            kelvin(SURFACE_FORMAT, 0x2128),
            kelvin(WINDOW_CLIP_TYPE, 1),
            kelvin(WINDOW_CLIP_HORIZONTAL + 8, (99 << 16) + 10),
            kelvin(WINDOW_CLIP_VERTICAL + 8, (49 << 16) + 20),
        ]);
        let window = p.pixel.target.window;
        assert!(window.exclusive);
        assert_eq!(window.rects[0], [0, 0, 0x1000 * 2, 0x1000 * 2]);
        for rect in &window.rects[2..] {
            assert_eq!(*rect, [20, 40, 200, 100]);
        }
    }

    #[test]
    fn a_clear_is_a_quad_that_writes_depth_stencil_and_the_named_channels() {
        let mut commands = vec![
            kelvin(ZSTENCIL_CLEAR_VALUE, 0x8000_0080),
            kelvin(COLOUR_CLEAR_VALUE, 0x40FF_8000),
            kelvin(CLEAR_RECT_HORIZONTAL, (99 << 16) + 10),
            kelvin(CLEAR_RECT_VERTICAL, (49 << 16) + 20),
        ];
        commands.push(kelvin(CLEAR_SURFACE, CLEAR_Z | CLEAR_STENCIL | 0x10 | 0x80));
        let clear = primitives(&commands).pop().expect("a clear");
        assert_eq!(clear.kind, Kind::Quads);
        let v = clear.vertices[2];
        assert_eq!((v.x, v.y), (100.0, 50.0));
        assert!((v.depth - 0.5).abs() < 1e-3, "0x800000 of a Z24 buffer");
        assert_eq!(v.colour, [1.0, 128.0 / 255.0, 0.0, 64.0 / 255.0]);
        assert_eq!(
            clear.depth.map(|d| (d.function, d.write)),
            Some((GL_ALWAYS, true))
        );
        let stencil = clear.pixel.stencil.expect("the stencil is written");
        assert_eq!((stencil.reference, stencil.pass), (0x80, GL_REPLACE));
        // Red and alpha only.
        assert_eq!(clear.pixel.colour_mask, 1 << 16 | 1 << 24);
        assert_eq!(clear.pixel.target.scissor, [10, 20, 90, 30]);
        // Colour alone; X8R8G8B8 clears alpha to zero; nothing named, nothing drawn.
        let colour = [
            kelvin(SURFACE_FORMAT, 0x124),
            kelvin(COLOUR_CLEAR_VALUE, 0xFF00_00FF),
            kelvin(CLEAR_SURFACE, CLEAR_COLOUR),
        ];
        let clear = primitives(&colour).pop().expect("a colour clear");
        assert_eq!(clear.vertices[0].colour, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(clear.depth.map(|d| d.write), Some(false));
        assert!(clear.pixel.stencil.is_none());
        assert_eq!(primitives(&[kelvin(CLEAR_SURFACE, 0)]), []);
    }

    /// Dragon's Lair clears the screen to black every frame of its movies, which play on the
    /// video overlay: a frame of clears alone must not cover them.
    #[test]
    fn a_frame_of_clears_alone_shows_nothing() {
        let clear = [
            kelvin(COLOUR_CLEAR_VALUE, 0xFF00_0000),
            kelvin(CLEAR_SURFACE, CLEAR_COLOUR),
        ];
        let cleared = primitives(&clear);
        assert!(cleared[0].clear);
        assert!(!crate::gpu::draw::shows_anything(&cleared));
        let drawn = [cleared[0].clone(), drawn_after(&[])];
        assert!(!drawn[1].clear);
        assert!(crate::gpu::draw::shows_anything(&drawn));
        assert!(!crate::gpu::draw::shows_anything(&[]));
    }

    #[test]
    fn depth_is_over_the_buffers_largest_and_the_clip_range_is_kept() {
        let mut commands = vec![
            position(),
            float(CLIP_MIN, 0.25 * Z24_MAX),
            float(CLIP_MAX, 0.5 * Z24_MAX),
            kelvin(ZMIN_MAX_CONTROL, 1 << 4),
            kelvin(SET_BEGIN_END, 5),
        ];
        for _ in 0..3 {
            commands.extend([
                float(0x1818, 0.0),
                float(0x1818, 0.0),
                float(0x1818, Z24_MAX / 4.0),
            ]);
        }
        commands.push(kelvin(SET_BEGIN_END, 0));
        let p = primitives(&commands).pop().unwrap();
        assert!((p.vertices[0].depth - 0.25).abs() < 1e-6);
        let range = p.pixel.depth_range;
        assert!((range.min - 0.25).abs() < 1e-6 && (range.max - 0.5).abs() < 1e-6);
        assert!(range.clamp && range.w_buffer.is_none());
        // Z16 float with w-buffering.
        let mut regs = PixelRegs::default();
        regs.set(SURFACE_FORMAT, 0x118);
        regs.set(CONTROL0, CONTROL0_Z_FLOAT | CONTROL0_Z_PERSPECTIVE);
        assert_eq!(regs.zmax(), F16_MAX);
        assert_eq!(regs.depth_range().w_buffer, Some(F16_MAX));
        assert!(!regs.depth_range().clamp);
        // The default range is the whole buffer, float formats included.
        assert_eq!(regs.depth_range().max, 1.0);
        regs.set(CONTROL0, 0);
        assert_eq!(PixelRegs::default().depth_range().max, 1.0);
        assert_eq!(f16_depth(0), 0.0);
        assert_eq!(f24_depth(0x7F_0000), f32::from_bits(0x7F_0000 << 7));
    }

    #[test]
    fn polygon_modes_draw_faces_as_edges_or_corners() {
        let lines = [
            kelvin(FRONT_POLYGON_MODE, POLYGON_LINE),
            kelvin(BACK_POLYGON_MODE, POLYGON_LINE),
        ];
        let mut commands = vec![position()];
        commands.extend(lines);
        commands.extend(square(0.0, 0.0, 10.0));
        let out = primitives(&commands);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, Kind::Lines);
        assert_eq!(out[0].vertices.len(), 8, "four edges");
        // Front faces (clockwise) as points, back faces filled: the square is a front face.
        let mut commands = vec![position(), kelvin(FRONT_POLYGON_MODE, POLYGON_POINT)];
        commands.extend(square(0.0, 0.0, 10.0));
        let out = primitives(&commands);
        // Corners are squares of the point size, never culled.
        assert_eq!(out[0].kind, Kind::Quads);
        assert_eq!(out[0].vertices.len(), 16);
        assert_eq!(out[0].pixel.cull, None);
        // A culled face drawn as lines is dropped here.
        let mut commands = vec![
            position(),
            kelvin(CULL_FACE_ENABLE, 1),
            kelvin(CULL_FACE, GL_FRONT),
        ];
        commands.extend(lines);
        commands.extend(square(0.0, 0.0, 10.0));
        assert_eq!(primitives(&commands), []);
    }

    #[test]
    fn a_visibility_test_counts_the_pixels_its_draws_cover() {
        let mut commands = vec![position()];
        commands.extend(square(10.0, 10.0, 10.0));
        assert_eq!(visible_pixels(&commands), 100);
        // Off the surface counts nothing; half off counts the half on it.
        let mut commands = vec![position()];
        commands.extend(square(-100.0, 10.0, 20.0));
        commands.extend(square(630.0, 470.0, 20.0));
        assert_eq!(visible_pixels(&commands), 100);
        // Anti-aliasing counts every sample; a culled face counts nothing.
        let mut commands = vec![position(), kelvin(SURFACE_FORMAT, 0x2128)];
        commands.extend(square(0.0, 0.0, 4.0));
        assert_eq!(visible_pixels(&commands), 64);
        let mut commands = vec![
            position(),
            kelvin(CULL_FACE_ENABLE, 1),
            kelvin(CULL_FACE, GL_FRONT),
        ];
        commands.extend(square(0.0, 0.0, 4.0));
        assert_eq!(visible_pixels(&commands), 0);
        // Draws that decode to nothing count the whole surface.
        let mut commands = vec![kelvin(0x1760, 2)];
        commands.extend(square(0.0, 0.0, 4.0));
        assert_eq!(visible_pixels(&commands), WHOLE_SURFACE);
        assert_eq!(visible_pixels(&[]), 0);
    }
}
