//! Textures (spec 10.9): a stage's registers as a texture description, where its bytes are, and
//! their decoding to RGBA.

use serde::{Deserialize, Serialize};

/// A texture stage's registers, from `0x1B00 + 0x40 * stage`.
pub const STAGE_BASE: u32 = 0x1B00;
pub const STAGE_STRIDE: u32 = 0x40;

/// The register defaults (spec 10.9).
pub const DEFAULT_ADDRESS: u32 = 0x0001_0101;
pub const DEFAULT_CONTROL0: u32 = 0xFFF << 6;
pub const DEFAULT_FILTER: u32 = 0x0202_0000;

/// How a format's texels are stored and decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Y8,
    Ay8,
    A1r5g5b5,
    X1r5g5b5,
    A4r4g4b4,
    R5g6b5,
    A8r8g8b8,
    X8r8g8b8,
    /// An index into the palette.
    I8,
    Dxt1,
    Dxt3,
    Dxt5,
    A8,
    A8y8,
    G8b8,
    R8b8,
    Y16,
    A8b8g8r8,
    B8g8r8a8,
    R8g8b8a8,
    /// Linear 4:2:2 video, two texels in four bytes Y0 Cb Y1 Cr (`LC_IMAGE_CR8YB8CB8YA8`).
    Yuy2,
    /// The same, with the bytes Cb Y0 Cr Y1 (`LC_IMAGE_YB8CR8YA8CB8`).
    Uyvy,
}

/// A colour format: its kind, whether it is linear (`LU_*`, one level, rows `pitch` apart) rather
/// than swizzled, and bytes per texel (per 4x4 block for DXT).
#[must_use]
pub fn format(code: u32) -> Option<(Kind, bool, u32)> {
    use Kind::{
        A1r5g5b5, A4r4g4b4, A8, A8b8g8r8, A8r8g8b8, A8y8, Ay8, B8g8r8a8, Dxt1, Dxt3, Dxt5, G8b8,
        I8, R5g6b5, R8b8, R8g8b8a8, Uyvy, X1r5g5b5, X8r8g8b8, Y8, Y16, Yuy2,
    };
    Some(match code {
        0x00 => (Y8, false, 1),
        0x13 => (Y8, true, 1),
        0x01 => (Ay8, false, 1),
        0x1B => (Ay8, true, 1),
        0x02 => (A1r5g5b5, false, 2),
        0x10 => (A1r5g5b5, true, 2),
        0x03 => (X1r5g5b5, false, 2),
        0x1C => (X1r5g5b5, true, 2),
        0x04 => (A4r4g4b4, false, 2),
        0x1D => (A4r4g4b4, true, 2),
        0x05 => (R5g6b5, false, 2),
        0x11 => (R5g6b5, true, 2),
        0x06 => (A8r8g8b8, false, 4),
        0x12 => (A8r8g8b8, true, 4),
        0x07 => (X8r8g8b8, false, 4),
        0x1E => (X8r8g8b8, true, 4),
        0x0B => (I8, false, 1),
        0x0C => (Dxt1, false, 8),
        0x0E => (Dxt3, false, 16),
        0x0F => (Dxt5, false, 16),
        0x19 => (A8, false, 1),
        0x1F => (A8, true, 1),
        0x1A => (A8y8, false, 2),
        0x20 => (A8y8, true, 2),
        0x28 => (G8b8, false, 2),
        0x17 => (G8b8, true, 2),
        0x29 => (R8b8, false, 2),
        0x32 => (Y16, false, 2),
        0x35 => (Y16, true, 2),
        0x3A => (A8b8g8r8, false, 4),
        0x3F => (A8b8g8r8, true, 4),
        0x3B => (B8g8r8a8, false, 4),
        0x40 => (B8g8r8a8, true, 4),
        0x3C => (R8g8b8a8, false, 4),
        0x41 => (R8g8b8a8, true, 4),
        0x24 => (Yuy2, true, 2),
        0x25 => (Uyvy, true, 2),
        _ => return None,
    })
}

/// A texture as the renderer needs it described: its identity is every field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Texture {
    /// The guest address of the top level.
    pub offset: u32,
    pub code: u32,
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    /// Row pitch in bytes of a linear texture (0 for swizzled ones).
    pub pitch: u32,
    /// The palette's guest address and its number of colours (0 without one).
    pub palette: u32,
    pub palette_colours: u32,
    /// The address modes (u, v, p), filter, border colour, and alpha kill: sampling state.
    pub address: u32,
    pub filter: u32,
    pub border: u32,
    pub alpha_kill: bool,
    /// Control 0: the min and max LOD clamps (bits 18..29 and 6..17, 4.8 fixed point).
    pub control0: u32,
    /// The slices of a 3D texture (1 for the others).
    pub depth: u32,
    /// A cube map: six faces, +X, -X, +Y, -Y, +Z and -Z, each a mip chain.
    pub cube: bool,
}

/// What each face of a cube map's mip chain is aligned to, in bytes.
pub const CUBE_FACE_ALIGNMENT: u32 = 128;

/// One stage's registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRegs {
    pub offset: u32,
    pub format: u32,
    pub address: u32,
    pub control0: u32,
    pub control1: u32,
    pub filter: u32,
    pub rect: u32,
    pub palette: u32,
    pub border: u32,
}

impl Default for StageRegs {
    fn default() -> Self {
        Self {
            offset: 0,
            format: 0,
            address: DEFAULT_ADDRESS,
            control0: DEFAULT_CONTROL0,
            control1: 0,
            filter: DEFAULT_FILTER,
            rect: 0,
            palette: 0,
            border: 0,
        }
    }
}

impl StageRegs {
    /// Applies a write to register `offset` within the stage; false for an offset that is not one.
    pub fn set(&mut self, offset: u32, value: u32) -> bool {
        let field = match offset {
            0x00 => &mut self.offset,
            0x04 => &mut self.format,
            0x08 => &mut self.address,
            0x0C => &mut self.control0,
            0x10 => &mut self.control1,
            0x14 => &mut self.filter,
            0x1C => &mut self.rect,
            0x20 => &mut self.palette,
            0x24 => &mut self.border,
            _ => return false,
        };
        *field = value;
        true
    }

    /// The texture the stage describes, if its format is one the model decodes.
    ///
    /// The format word: bit 2 a cube map, bits 4..7 the dimensionality (3 for a 3D texture),
    /// 8..15 the colour format, 16..19 the levels, and the log2 width, height and depth in bits
    /// 20..23, 24..27 and 28..31. Linear textures are flat whatever the word says.
    #[must_use]
    pub fn texture(&self) -> Option<Texture> {
        let code = (self.format >> 8) & 0xFF;
        let (kind, linear, _) = format(code)?;
        let (log_width, log_height) = ((self.format >> 20) & 0xF, (self.format >> 24) & 0xF);
        let mut width = 1 << log_width;
        let mut height = 1 << log_height;
        let volume = !linear && (self.format >> 4) & 0xF == 3;
        let cube = !linear && !volume && self.format & 1 << 2 != 0;
        let (levels, pitch) = if linear {
            if self.rect != 0 {
                width = (self.rect >> 16).max(1);
                height = (self.rect & 0xFFFF).max(1);
            }
            (1, self.control1 >> 16)
        } else {
            let most = 32 - width.max(height).leading_zeros();
            let mut levels = ((self.format >> 16) & 0xF).clamp(1, most);
            if volume {
                // A 3D texture's levels stay at least 4x4.
                levels = if log_width < 2 || log_height < 2 {
                    1
                } else {
                    levels.min(log_width.min(log_height) - 1).max(1)
                };
            }
            // The max LOD clamp is 4.8 fixed point.
            let clamp = ((self.control0 >> 6) & 0xFFF) >> 8;
            (levels.min(clamp + 1), 0)
        };
        let (palette, palette_colours) = if kind == Kind::I8 {
            (self.palette & !0x3F, 256 >> ((self.palette >> 2) & 3))
        } else {
            (0, 0)
        };
        Some(Texture {
            offset: self.offset,
            code,
            width,
            height,
            levels,
            pitch,
            palette,
            palette_colours,
            address: self.address,
            filter: self.filter,
            border: self.border,
            alpha_kill: self.control0 & 1 << 2 != 0,
            control0: self.control0,
            depth: if volume { 1 << (self.format >> 28) } else { 1 },
            cube,
        })
    }
}

/// `SET_SHADER_STAGE_PROGRAM`: each stage's texture mode, five bits a stage from the bottom.
pub const SHADER_STAGE_PROGRAM: u32 = 0x1E70;
/// `SET_DOT_RGBMAPPING`: how stages 1 to 3 read the texel their dot product takes, four bits a
/// stage from the bottom.
pub const DOT_RGBMAPPING: u32 = 0x1E74;
/// `SET_SHADER_OTHER_STAGE_INPUT`: the stage whose texel stage 2 reads (bits 16..19) and stage 3
/// reads (bits 20..23); stage 1 reads stage 0's.
pub const SHADER_OTHER_STAGE_INPUT: u32 = 0x1E78;
/// Within a stage's registers: the bump matrix (four floats), and the bump luminance scale and
/// offset.
const BUMP_MATRIX: u32 = 0x28;
const BUMP_SCALE: u32 = 0x38;
const BUMP_OFFSET: u32 = 0x3C;

/// What a texture stage gives the combiners, from its mode in the stage program. The number is
/// the shader's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Mode {
    /// The texture at `s/q`, `t/q` (`PROJECT2D`); a cube map toward `(1, t, -s)` normalised, a
    /// 3D texture at `r` = 0.
    #[default]
    Sample = 0,
    /// Black with alpha 1: no program for the stage (`PROGRAM_NONE`), or its texture is off.
    Off = 1,
    /// The coordinates `s, t, r, q`, clamped to 0..1 (`PASS_THROUGH`).
    Coordinates = 2,
    /// Zero: `CLIP_PLANE` (its discards are not modelled), `BRDF`, `DOT_RFLCT_SPEC_CONST`, and a
    /// mode on a stage that cannot take it.
    Zero = 3,
    /// The texture at `s/q, t/q, r/q` (`PROJECT3D`).
    Project3d = 4,
    /// The cube map toward `(s, t, r)` (`CUBE_MAP`).
    CubeMap = 5,
    /// The texture at `(s, t)` moved by the input stage's blue and green, read signed and turned
    /// by the stage's bump matrix (`BUMPENVMAP`, stages 1 to 3).
    BumpEnvMap = 6,
    /// As [`Mode::BumpEnvMap`], the texel then scaled by `scale * red + offset` of the input
    /// (`BUMPENVMAP_LUMINANCE`).
    BumpEnvMapLuminance = 7,
    /// Zero; the dot product of `(s, t, r)` with the input texel feeds the stages after
    /// (`DOT_PRODUCT`, stages 1 and 2).
    DotProduct = 8,
    /// The texture at the previous stage's dot product and this one's (`DOT_ST`, stages 2, 3).
    DotSt = 9,
    /// Zero; the dot product would be the depth, which is not written (`DOT_ZW`, stages 2, 3).
    DotZw = 10,
    /// The cube map toward the normal of the dot products of stages 1 to 3 (`DOT_RFLCT_DIFF`,
    /// stage 2).
    DotReflectDiffuse = 11,
    /// The cube map toward the eye vector (the `q` of stages 1 to 3) reflected about that normal
    /// (`DOT_RFLCT_SPEC`, stage 3).
    DotReflectSpecular = 12,
    /// The 3D texture at the dot products of stages 1 to 3 (`DOT_STR_3D`, stage 3).
    DotStr3d = 13,
    /// The cube map toward the dot products of stages 1 to 3 (`DOT_STR_CUBE`, stage 3).
    DotStrCube = 14,
    /// The texture at the input texel's alpha and red (`DEPENDENT_AR`, stages 1 to 3).
    DependentAr = 15,
    /// The texture at the input texel's green and blue (`DEPENDENT_GB`, stages 1 to 3).
    DependentGb = 16,
}

impl Mode {
    /// The mode of `stage` in stage program `program`; `on` is the stage's enable bit (control 0
    /// bit 30), without which a mode that reads a texture reads none (black). With no program
    /// written yet (`None`), stages 0 and 1 sample, and stages 2 and 3 sample while on.
    #[must_use]
    pub fn of(program: Option<u32>, stage: usize, on: bool) -> Self {
        let stage = stage.min(3);
        let Some(program) = program else {
            return if stage < 2 || on {
                Self::Sample
            } else {
                Self::Off
            };
        };
        let mode = match (program >> (5 * stage)) & 0x1F {
            0 => return Self::Off,
            4 => return Self::Coordinates,
            0x11 if stage == 1 || stage == 2 => return Self::DotProduct,
            0x0A if stage >= 2 => return Self::DotZw,
            5 | 8 | 0x0A | 0x11 | 0x12 => return Self::Zero,
            2 => Self::Project3d,
            3 => Self::CubeMap,
            6 if stage >= 1 => Self::BumpEnvMap,
            7 if stage >= 1 => Self::BumpEnvMapLuminance,
            9 if stage >= 2 => Self::DotSt,
            0x0B if stage == 2 => Self::DotReflectDiffuse,
            0x0C if stage == 3 => Self::DotReflectSpecular,
            0x0D if stage == 3 => Self::DotStr3d,
            0x0E if stage == 3 => Self::DotStrCube,
            0x0F if stage >= 1 => Self::DependentAr,
            0x10 if stage >= 1 => Self::DependentGb,
            // PROJECT2D, and a mode the stage cannot take.
            _ => Self::Sample,
        };
        if on { mode } else { Self::Off }
    }

    /// Whether the stage reads its texture.
    #[must_use]
    pub fn samples(self) -> bool {
        !matches!(
            self,
            Self::Off | Self::Coordinates | Self::Zero | Self::DotProduct | Self::DotZw
        )
    }
}

/// What the modes that move coordinates read besides the textures: the dot mappings, the input
/// stages, and each stage's bump environment.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Shader {
    /// `SET_DOT_RGBMAPPING`.
    pub dot_mapping: u32,
    /// `SET_SHADER_OTHER_STAGE_INPUT`.
    pub other_input: u32,
    /// Each stage's bump matrix `[m00, m01, m10, m11]`: offsets `(du, dv)` move the coordinates
    /// by `(m00 du + m10 dv, m01 du + m11 dv)`. Stage 0 has none.
    pub bump_matrix: [[f32; 4]; 4],
    /// Each stage's bump luminance scale and offset (stage 0 has none).
    pub bump_scale: [f32; 4],
    pub bump_offset: [f32; 4],
}

impl Shader {
    /// Applies a write to `method`; false for a method that is none of these registers.
    pub fn set(&mut self, method: u32, value: u32) -> bool {
        match method {
            DOT_RGBMAPPING => self.dot_mapping = value,
            SHADER_OTHER_STAGE_INPUT => self.other_input = value,
            m if (STAGE_BASE..STAGE_BASE + 4 * STAGE_STRIDE).contains(&m) => {
                let stage = ((m - STAGE_BASE) / STAGE_STRIDE) as usize;
                let float = f32::from_bits(value);
                match (m - STAGE_BASE) % STAGE_STRIDE {
                    // The words come as m00, m01, m11, m10.
                    r @ BUMP_MATRIX..0x38 => {
                        let slot = [0, 1, 3, 2][((r - BUMP_MATRIX) / 4) as usize];
                        if stage > 0 {
                            self.bump_matrix[stage][slot] = float;
                        }
                    }
                    BUMP_SCALE if stage > 0 => self.bump_scale[stage] = float,
                    BUMP_OFFSET if stage > 0 => self.bump_offset[stage] = float,
                    BUMP_SCALE | BUMP_OFFSET => {}
                    _ => return false,
                }
            }
            _ => return false,
        }
        true
    }

    /// How `stage` maps the texel its dot product reads: 0 unsigned (0..1), 1 to 3 the signed
    /// mappings, 4 `HILO_1`; stage 0 has none.
    #[must_use]
    pub fn mapping(&self, stage: usize) -> u32 {
        match stage {
            1..=3 => (self.dot_mapping >> (4 * (stage - 1))) & 0xF,
            _ => 0,
        }
    }

    /// The stage whose texel `stage` reads: stage 0's for stages 0 and 1, the register's for
    /// stages 2 and 3 (an earlier stage).
    #[must_use]
    pub fn input(&self, stage: usize) -> usize {
        let field = |shift: u32| ((self.other_input >> shift) & 0xF) as usize;
        match stage {
            2 => field(16).min(1),
            3 => field(20).min(2),
            _ => 0,
        }
    }
}

/// One decoded level: its size and RGBA bytes, row by row, and the layers it holds one after
/// the other (a 3D texture's slices or a cube map's faces; 1 for a flat texture).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Level {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub rgba: Vec<u8>,
}

/// A level's size in bytes.
fn level_bytes(kind: Kind, bpp: u32, width: u32, height: u32) -> u32 {
    if matches!(kind, Kind::Dxt1 | Kind::Dxt3 | Kind::Dxt5) {
        width.div_ceil(4) * height.div_ceil(4) * bpp
    } else {
        width * height * bpp
    }
}

/// How a texture is sampled: flat, as a 3D texture, or as a cube map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Shape {
    #[default]
    Flat,
    Volume,
    /// Six square faces (a cube map whose faces are not square is read as flat, its first face).
    Cube,
}

impl Texture {
    /// How the texture is sampled.
    #[must_use]
    pub fn shape(&self) -> Shape {
        if self.depth > 1 {
            Shape::Volume
        } else if self.cube && self.width == self.height {
            Shape::Cube
        } else {
            Shape::Flat
        }
    }

    /// Whether it is linear (`LU_*`, `LC_*`): one level, rows `pitch` apart, coordinates in
    /// texels.
    #[must_use]
    pub fn linear(&self) -> bool {
        format(self.code).is_some_and(|(_, linear, _)| linear)
    }

    /// The slices of level `level` of a 3D texture, which halve with each level; 1 for the
    /// others.
    #[must_use]
    pub fn level_depth(&self, level: u32) -> u32 {
        (self.depth >> level.min(31)).max(1)
    }

    /// The bytes of one mip chain, every slice of every level: the whole texture, or one face of
    /// a cube map.
    fn chain_len(&self, kind: Kind, bpp: u32) -> u32 {
        (0..self.levels)
            .map(|l| {
                level_bytes(
                    kind,
                    bpp,
                    (self.width >> l).max(1),
                    (self.height >> l).max(1),
                ) * self.level_depth(l)
            })
            .sum()
    }

    /// The bytes of every level (top first), as one span from `offset`. A cube map's faces each
    /// start on a [`CUBE_FACE_ALIGNMENT`] boundary.
    #[must_use]
    pub fn byte_len(&self) -> u32 {
        let Some((kind, linear, bpp)) = format(self.code) else {
            return 0;
        };
        if linear {
            let pitch = if self.pitch == 0 {
                self.width * bpp
            } else {
                self.pitch
            };
            return pitch * self.height;
        }
        let chain = self.chain_len(kind, bpp);
        if self.cube {
            chain.next_multiple_of(CUBE_FACE_ALIGNMENT) * 5 + chain
        } else {
            chain
        }
    }

    /// Decodes every level from the texture's `bytes` (and, for a palettised one, `palette`).
    /// A cube map's levels hold the six faces, a 3D texture's its slices.
    #[must_use]
    pub fn decode(&self, bytes: &[u8], palette: &[u8]) -> Option<Vec<Level>> {
        let (kind, linear, bpp) = format(self.code)?;
        if !self.cube {
            return self.decode_chain(kind, linear, bpp, bytes, palette);
        }
        let face = self
            .chain_len(kind, bpp)
            .next_multiple_of(CUBE_FACE_ALIGNMENT) as usize;
        let mut levels = self.decode_chain(kind, linear, bpp, bytes, palette)?;
        for index in 1..6 {
            let faces =
                self.decode_chain(kind, linear, bpp, bytes.get(index * face..)?, palette)?;
            for (level, more) in levels.iter_mut().zip(faces) {
                level.rgba.extend(more.rgba);
                level.layers += 1;
            }
        }
        Some(levels)
    }

    /// Decodes one mip chain from the start of `bytes`.
    fn decode_chain(
        &self,
        kind: Kind,
        linear: bool,
        bpp: u32,
        bytes: &[u8],
        palette: &[u8],
    ) -> Option<Vec<Level>> {
        let mut levels = Vec::new();
        let mut at = 0usize;
        for level in 0..self.levels {
            let (w, h) = ((self.width >> level).max(1), (self.height >> level).max(1));
            let depth = self.level_depth(level);
            let len = (level_bytes(kind, bpp, w, h) * depth) as usize;
            let data = if linear {
                bytes
            } else {
                bytes.get(at..at + len)?
            };
            at += len;
            let mut rgba = Vec::with_capacity((w * h * depth * 4) as usize);
            for z in 0..depth {
                rgba.extend(self.decode_slice(
                    kind,
                    linear,
                    bpp,
                    data,
                    (w, h, depth),
                    z,
                    palette,
                )?);
            }
            levels.push(Level {
                width: w,
                height: h,
                layers: depth,
                rgba,
            });
        }
        Some(levels)
    }

    /// Decodes slice `z` of a level of `size` (width, height, depth) from its bytes `data`.
    #[allow(clippy::too_many_arguments)]
    fn decode_slice(
        &self,
        kind: Kind,
        linear: bool,
        bpp: u32,
        data: &[u8],
        (w, h, depth): (u32, u32, u32),
        z: u32,
        palette: &[u8],
    ) -> Option<Vec<u8>> {
        if matches!(kind, Kind::Dxt1 | Kind::Dxt3 | Kind::Dxt5) {
            if depth == 1 {
                return dxt(kind, data, w, h);
            }
            // A 3D texture's blocks go by groups of four slices: each block position holds the
            // group's slices of it one after the other.
            let (across, down) = (w.div_ceil(4), h.div_ceil(4));
            let group = z / 4 * 4;
            let deep = (depth - group).min(4);
            let size = bpp as usize;
            let mut blocks = vec![0u8; (across * down) as usize * size];
            for index in 0..across * down {
                let from = ((group * across * down + index * deep + z - group) as usize) * size;
                let to = index as usize * size;
                blocks[to..to + size].copy_from_slice(data.get(from..from + size)?);
            }
            return dxt(kind, &blocks, w, h);
        }
        let pitch = if self.pitch == 0 { w * bpp } else { self.pitch };
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                if matches!(kind, Kind::Yuy2 | Kind::Uyvy) {
                    // Two texels share four bytes and their chroma.
                    let at = (y * pitch + (x & !1) * 2) as usize;
                    let pair = data.get(at..at + 4)?;
                    let (luma, cb, cr) = if kind == Kind::Yuy2 {
                        (pair[usize::from(x & 1 != 0) * 2], pair[1], pair[3])
                    } else {
                        (pair[1 + usize::from(x & 1 != 0) * 2], pair[0], pair[2])
                    };
                    let [red, green, blue] = super::overlay::rgb(luma, cb, cr);
                    rgba.extend([red, green, blue, 255]);
                    continue;
                }
                let index = if linear {
                    (y * pitch + x * bpp) as usize
                } else if depth > 1 {
                    (morton3(x, y, z, (w, h, depth)) * bpp) as usize
                } else {
                    (morton(x, y, w, h) * bpp) as usize
                };
                let texel = data.get(index..index + bpp as usize)?;
                rgba.extend(decode_texel(kind, texel, palette));
            }
        }
        Some(rgba)
    }
}

/// The index of texel `(x, y)` in a swizzled `width` by `height` level: walking bit positions
/// from 0, while either dimension has bits left, x's bit (if `1 << bit < width`) then y's bit
/// (if `1 << bit < height`) go into successive index bits.
#[must_use]
pub fn morton(x: u32, y: u32, width: u32, height: u32) -> u32 {
    let mut index = 0;
    let mut out = 0;
    let mut bit = 0;
    while (1 << bit) < width || (1 << bit) < height {
        if (1 << bit) < width {
            index |= ((x >> bit) & 1) << out;
            out += 1;
        }
        if (1 << bit) < height {
            index |= ((y >> bit) & 1) << out;
            out += 1;
        }
        bit += 1;
    }
    index
}

/// The index of texel `(x, y, z)` in a swizzled `size` (width, height, depth) level of a 3D
/// texture: as [`morton`], with x's, then y's, then z's bit at each bit position while that
/// dimension has bits left.
#[must_use]
pub fn morton3(x: u32, y: u32, z: u32, (width, height, depth): (u32, u32, u32)) -> u32 {
    let mut index = 0;
    let mut out = 0;
    let mut bit = 0;
    while (1 << bit) < width || (1 << bit) < height || (1 << bit) < depth {
        for (value, size) in [(x, width), (y, height), (z, depth)] {
            if (1 << bit) < size {
                index |= ((value >> bit) & 1) << out;
                out += 1;
            }
        }
        bit += 1;
    }
    index
}

/// Scales an `n`-bit channel to 8 bits.
fn scale(v: u16, bits: u32) -> u8 {
    u8::try_from(u32::from(v) * 255 / ((1 << bits) - 1)).unwrap_or(255)
}

/// Expands 5 or 6 bits to 8 by replicating the high bits.
fn replicate(v: u16, bits: u32) -> u8 {
    let v = u32::from(v);
    u8::try_from((v << (8 - bits)) | (v >> (2 * bits - 8))).unwrap_or(255)
}

/// RGBA of one texel from its little-endian bytes.
fn decode_texel(kind: Kind, b: &[u8], palette: &[u8]) -> [u8; 4] {
    let half = || u16::from_le_bytes([b[0], b[1]]);
    match kind {
        Kind::Y8 => [b[0], b[0], b[0], 255],
        Kind::Ay8 => [b[0], b[0], b[0], b[0]],
        Kind::A1r5g5b5 | Kind::X1r5g5b5 => {
            let v = half();
            let alpha = if kind == Kind::X1r5g5b5 || v & 0x8000 != 0 {
                255
            } else {
                0
            };
            [
                scale((v >> 10) & 31, 5),
                scale((v >> 5) & 31, 5),
                scale(v & 31, 5),
                alpha,
            ]
        }
        Kind::A4r4g4b4 => {
            let v = half();
            [
                scale((v >> 8) & 15, 4),
                scale((v >> 4) & 15, 4),
                scale(v & 15, 4),
                scale(v >> 12, 4),
            ]
        }
        Kind::R5g6b5 => {
            let v = half();
            [
                replicate(v >> 11, 5),
                replicate((v >> 5) & 63, 6),
                replicate(v & 31, 5),
                255,
            ]
        }
        Kind::A8r8g8b8 => [b[2], b[1], b[0], b[3]],
        Kind::X8r8g8b8 => [b[2], b[1], b[0], 255],
        Kind::I8 => {
            let p = usize::from(b[0]) * 4;
            palette
                .get(p..p + 4)
                .map_or([0; 4], |p| [p[2], p[1], p[0], p[3]])
        }
        Kind::A8 => [255, 255, 255, b[0]],
        Kind::A8y8 => [b[0], b[0], b[0], b[1]],
        Kind::G8b8 => [b[0], b[1], b[0], b[1]],
        Kind::R8b8 => [b[1], b[0], b[0], b[1]],
        Kind::Y16 => [255, b[1], b[0], 255],
        Kind::A8b8g8r8 => [b[0], b[1], b[2], b[3]],
        Kind::B8g8r8a8 => [b[1], b[2], b[3], b[0]],
        Kind::R8g8b8a8 => [b[3], b[2], b[1], b[0]],
        Kind::Dxt1 | Kind::Dxt3 | Kind::Dxt5 | Kind::Yuy2 | Kind::Uyvy => [0; 4],
    }
}

/// The four colours of a DXT colour block; `four` forces the four-colour mode.
fn dxt_colours(block: &[u8], four: bool) -> [[u8; 4]; 4] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let rgb = |c: u16| {
        [
            replicate(c >> 11, 5),
            replicate((c >> 5) & 63, 6),
            replicate(c & 31, 5),
        ]
    };
    let (a, b) = (rgb(c0), rgb(c1));
    let mix = |f: fn(u16, u16) -> u16| -> [u8; 4] {
        let channel = |i: usize| u8::try_from(f(u16::from(a[i]), u16::from(b[i]))).unwrap_or(255);
        [channel(0), channel(1), channel(2), 255]
    };
    if four || c0 > c1 {
        [
            [a[0], a[1], a[2], 255],
            [b[0], b[1], b[2], 255],
            mix(|a, b| (2 * a + b) / 3),
            mix(|a, b| (a + 2 * b) / 3),
        ]
    } else {
        // Punch-through: the fourth colour is transparent black.
        [
            [a[0], a[1], a[2], 255],
            [b[0], b[1], b[2], 255],
            mix(u16::midpoint),
            [0, 0, 0, 0],
        ]
    }
}

/// Decodes a DXT level; blocks past the edge are decoded at 4x4 and cropped.
fn dxt(kind: Kind, data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let block_bytes = if kind == Kind::Dxt1 { 8 } else { 16 };
    let (bw, bh) = (width.div_ceil(4), height.div_ceil(4));
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for by in 0..bh {
        for bx in 0..bw {
            let at = ((by * bw + bx) * block_bytes) as usize;
            let block = data.get(at..at + block_bytes as usize)?;
            let (alpha, colour) = if kind == Kind::Dxt1 {
                (&block[..0], block)
            } else {
                block.split_at(8)
            };
            let colours = dxt_colours(colour, kind != Kind::Dxt1);
            let indices = u32::from_le_bytes([colour[4], colour[5], colour[6], colour[7]]);
            let alphas = match kind {
                Kind::Dxt5 => dxt5_alphas(alpha),
                _ => [0; 8],
            };
            let alpha_bits = u64::from_le_bytes(alpha.try_into().unwrap_or([0; 8]));
            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx * 4 + px, by * 4 + py);
                    if x >= width || y >= height {
                        continue;
                    }
                    let i = py * 4 + px;
                    let mut texel = colours[((indices >> (2 * i)) & 3) as usize];
                    match kind {
                        Kind::Dxt3 => {
                            texel[3] =
                                u8::try_from(((alpha_bits >> (4 * i)) & 15) * 17).unwrap_or(255);
                        }
                        Kind::Dxt5 => {
                            texel[3] = alphas[((alpha_bits >> (16 + 3 * i)) & 7) as usize];
                        }
                        _ => {}
                    }
                    let out = ((y * width + x) * 4) as usize;
                    rgba[out..out + 4].copy_from_slice(&texel);
                }
            }
        }
    }
    Some(rgba)
}

/// The eight alphas of a DXT5 alpha block: interpolated between the two endpoints, eight ways
/// when the first is larger, else six ways plus 0 and 255.
fn dxt5_alphas(block: &[u8]) -> [u8; 8] {
    let (a0, a1) = (u16::from(block[0]), u16::from(block[1]));
    let lerp = |i: u16, n: u16| u8::try_from(((n - i) * a0 + i * a1) / n).unwrap_or(255);
    if a0 > a1 {
        [
            block[0],
            block[1],
            lerp(1, 7),
            lerp(2, 7),
            lerp(3, 7),
            lerp(4, 7),
            lerp(5, 7),
            lerp(6, 7),
        ]
    } else {
        [
            block[0],
            block[1],
            lerp(1, 5),
            lerp(2, 5),
            lerp(3, 5),
            lerp(4, 5),
            0,
            255,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_order_interleaves_x_then_y_while_both_have_bits() {
        assert_eq!(morton(1, 0, 4, 4), 1);
        assert_eq!(morton(0, 1, 4, 4), 2);
        assert_eq!(morton(1, 1, 4, 4), 3);
        assert_eq!(morton(2, 0, 4, 4), 4);
        assert_eq!(morton(3, 3, 4, 4), 15);
        // A wide level: once y runs out of bits, x's bits follow on their own.
        assert_eq!(morton(2, 0, 8, 2), 4);
        assert_eq!(morton(4, 0, 8, 2), 8);
        assert_eq!(morton(0, 1, 8, 2), 2);
        assert_eq!(morton(7, 1, 8, 2), 15);
    }

    #[test]
    fn hand_made_texels_decode() {
        assert_eq!(
            decode_texel(Kind::A8r8g8b8, &[1, 2, 3, 4], &[]),
            [3, 2, 1, 4]
        );
        assert_eq!(
            decode_texel(Kind::X8r8g8b8, &[1, 2, 3, 4], &[]),
            [3, 2, 1, 255]
        );
        assert_eq!(
            decode_texel(Kind::R5g6b5, &0xF800u16.to_le_bytes(), &[]),
            [255, 0, 0, 255]
        );
        assert_eq!(
            decode_texel(Kind::R5g6b5, &0x07E0u16.to_le_bytes(), &[]),
            [0, 255, 0, 255]
        );
        assert_eq!(
            decode_texel(Kind::A1r5g5b5, &0x801Fu16.to_le_bytes(), &[]),
            [0, 0, 255, 255]
        );
        assert_eq!(
            decode_texel(Kind::A1r5g5b5, &0x001Fu16.to_le_bytes(), &[]),
            [0, 0, 255, 0]
        );
        assert_eq!(
            decode_texel(Kind::X1r5g5b5, &0x0000u16.to_le_bytes(), &[]),
            [0, 0, 0, 255]
        );
        assert_eq!(
            decode_texel(Kind::A4r4g4b4, &0xF0F0u16.to_le_bytes(), &[]),
            [0, 255, 0, 255]
        );
        assert_eq!(decode_texel(Kind::Y8, &[7], &[]), [7, 7, 7, 255]);
        assert_eq!(decode_texel(Kind::Ay8, &[7], &[]), [7, 7, 7, 7]);
        assert_eq!(decode_texel(Kind::A8, &[7], &[]), [255, 255, 255, 7]);
        assert_eq!(decode_texel(Kind::A8y8, &[7, 9], &[]), [7, 7, 7, 9]);
        assert_eq!(decode_texel(Kind::G8b8, &[7, 9], &[]), [7, 9, 7, 9]);
        assert_eq!(decode_texel(Kind::R8b8, &[7, 9], &[]), [9, 7, 7, 9]);
        assert_eq!(decode_texel(Kind::Y16, &[7, 9], &[]), [255, 9, 7, 255]);
        assert_eq!(
            decode_texel(Kind::A8b8g8r8, &[1, 2, 3, 4], &[]),
            [1, 2, 3, 4]
        );
        assert_eq!(
            decode_texel(Kind::B8g8r8a8, &[1, 2, 3, 4], &[]),
            [2, 3, 4, 1]
        );
        assert_eq!(
            decode_texel(Kind::R8g8b8a8, &[1, 2, 3, 4], &[]),
            [4, 3, 2, 1]
        );
        assert_eq!(
            decode_texel(Kind::I8, &[1], &[0, 0, 0, 0, 1, 2, 3, 4]),
            [3, 2, 1, 4]
        );
    }

    #[test]
    fn a_swizzled_texture_decodes_every_level_in_morton_order() {
        let stage = StageRegs {
            // A8R8G8B8 (6), two levels, 2x2.
            format: 6 << 8 | 2 << 16 | 1 << 20 | 1 << 24,
            ..StageRegs::default()
        };
        let texture = stage.texture().unwrap();
        assert_eq!((texture.width, texture.height, texture.levels), (2, 2, 2));
        assert_eq!(texture.byte_len(), 16 + 4);
        // Memory order: (0,0) (1,0) (0,1) (1,1), then the 1x1 level.
        let bytes: Vec<u8> = (0u8..5).flat_map(|i| [i, 0, 0, 255]).collect();
        let levels = texture.decode(&bytes, &[]).unwrap();
        let blue: Vec<u8> = levels[0].rgba.chunks(4).map(|p| p[2]).collect();
        assert_eq!(blue, [0, 1, 2, 3]);
        assert_eq!(levels[1].rgba, [0, 0, 4, 255]);
    }

    #[test]
    fn a_linear_texture_takes_its_pitch_and_image_rect() {
        let stage = StageRegs {
            format: 0x12 << 8,
            control1: 12 << 16,
            rect: 2 << 16 | 2,
            ..StageRegs::default()
        };
        let texture = stage.texture().unwrap();
        assert_eq!(
            (texture.width, texture.height, texture.levels, texture.pitch),
            (2, 2, 1, 12)
        );
        let mut bytes = vec![0u8; 24];
        bytes[12..16].copy_from_slice(&[9, 0, 0, 255]); // (0, 1)
        let levels = texture.decode(&bytes, &[]).unwrap();
        assert_eq!(&levels[0].rgba[8..12], &[0, 0, 9, 255]);
    }

    #[test]
    fn levels_are_clamped_by_the_size_and_the_max_lod() {
        let stage = StageRegs {
            format: 6 << 8 | 15 << 16 | 2 << 20 | 1 << 24, // 4x2, 15 levels asked
            ..StageRegs::default()
        };
        assert_eq!(stage.texture().unwrap().levels, 3);
        let clamped = StageRegs {
            control0: 0x100 << 6, // max LOD 1.0
            ..stage
        };
        assert_eq!(clamped.texture().unwrap().levels, 2);
    }

    #[test]
    fn a_palette_holds_256_or_fewer_colours() {
        let stage = StageRegs {
            format: 0x0B << 8,
            palette: 0x1_2345 | 1 << 2,
            ..StageRegs::default()
        };
        let texture = stage.texture().unwrap();
        assert_eq!(
            (texture.palette, texture.palette_colours),
            (0x1_2340 & !0x3F, 128)
        );
    }

    #[test]
    fn dxt1_blocks_have_a_punch_through_mode() {
        // c0 = red, c1 = blue, c0 > c1: four colours; texel 0 index 0, texel 1 index 2.
        let red = 0xF800u16;
        let blue = 0x001Fu16;
        let mut block = Vec::new();
        block.extend(red.to_le_bytes());
        block.extend(blue.to_le_bytes());
        block.extend(0b1000u32.to_le_bytes());
        let rgba = dxt(Kind::Dxt1, &block, 4, 4).unwrap();
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[170, 0, 85, 255]);
        // c0 <= c1: index 3 is transparent black.
        let mut block = Vec::new();
        block.extend(blue.to_le_bytes());
        block.extend(red.to_le_bytes());
        block.extend(3u32.to_le_bytes());
        let rgba = dxt(Kind::Dxt1, &block, 4, 4).unwrap();
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn dxt3_and_dxt5_carry_alpha() {
        let mut colour = Vec::new();
        colour.extend(0xFFFFu16.to_le_bytes());
        colour.extend(0xFFFFu16.to_le_bytes());
        colour.extend(0u32.to_le_bytes());
        let mut dxt3 = 0x0000_0000_0000_00F5u64.to_le_bytes().to_vec();
        dxt3.extend(&colour);
        let rgba = dxt(Kind::Dxt3, &dxt3, 4, 4).unwrap();
        assert_eq!((rgba[3], rgba[7]), (5 * 17, 15 * 17));
        // DXT5: a0 = 200, a1 = 100; texel 0 index 1 (a1), texel 1 index 2 ((6 * 200 + 100) / 7).
        let mut dxt5 = vec![200, 100];
        let indices: u64 = 1 | 2 << 3;
        dxt5.extend(&indices.to_le_bytes()[..6]);
        dxt5.extend(&colour);
        let rgba = dxt(Kind::Dxt5, &dxt5, 4, 4).unwrap();
        assert_eq!((rgba[3], rgba[7]), (100, 185));
        assert_eq!(dxt5_alphas(&[100, 200, 0, 0, 0, 0, 0, 0])[6..], [0, 255]);
    }

    #[test]
    fn a_small_dxt_level_is_cropped() {
        let block = [0u8; 8];
        assert_eq!(dxt(Kind::Dxt1, &block, 2, 1).unwrap().len(), 8);
    }

    #[test]
    fn dxt5_alpha_codes_0_and_1_are_the_two_endpoints() {
        // Texels 0 to 7 take alpha codes 0 to 7; the colour block is all black.
        let codes = (0..8u64).fold(0, |bits, code| bits | code << (3 * code));
        let alphas = |a0: u8, a1: u8| {
            let mut block = vec![a0, a1];
            block.extend(&codes.to_le_bytes()[..6]);
            block.extend([0u8; 8]);
            let rgba = dxt(Kind::Dxt5, &block, 4, 4).unwrap();
            (0..8).map(|t| rgba[t * 4 + 3]).collect::<Vec<_>>()
        };
        // a0 > a1: six steps between the endpoints.
        assert_eq!(alphas(200, 100), [200, 100, 185, 171, 157, 142, 128, 114]);
        // a0 <= a1: four steps, then fully transparent and fully opaque.
        assert_eq!(alphas(100, 200), [100, 200, 120, 140, 160, 180, 0, 255]);
        assert_eq!(alphas(0, 0), [0, 0, 0, 0, 0, 0, 0, 255]);
    }

    #[test]
    fn video_textures_share_chroma_between_two_texels() {
        // Two texels, luma 16 (black) and 235 (white), no chroma, in both byte orders; then a
        // red pair (Y 81, Cb 90, Cr 240).
        for (code, bytes) in [
            (0x24, [16, 128, 235, 128, 81, 90, 81, 240]),
            (0x25, [128, 16, 128, 235, 90, 81, 240, 81]),
        ] {
            let stage = StageRegs {
                format: code << 8,
                control1: 8 << 16,
                rect: 4 << 16 | 1,
                ..StageRegs::default()
            };
            let texture = stage.texture().unwrap();
            assert!(texture.linear());
            assert_eq!(texture.byte_len(), 8);
            let rgba = &texture.decode(&bytes, &[]).unwrap()[0].rgba;
            assert_eq!(&rgba[..8], &[0, 0, 0, 255, 255, 255, 255, 255]);
            for texel in rgba[8..].chunks(4) {
                assert!(texel[0] > 250 && texel[1] < 5 && texel[2] < 5, "{texel:?}");
            }
        }
    }

    #[test]
    fn a_3d_texture_interleaves_z_after_x_and_y() {
        // A8 (0x19), 3D, 2x2x2, one level.
        let stage = StageRegs {
            format: 1 << 28 | 1 << 24 | 1 << 20 | 1 << 16 | 0x19 << 8 | 3 << 4,
            ..StageRegs::default()
        };
        let t = stage.texture().unwrap();
        assert_eq!(
            (t.width, t.height, t.depth, t.levels, t.cube),
            (2, 2, 2, 1, false)
        );
        assert_eq!((t.shape(), t.byte_len()), (Shape::Volume, 8));
        let levels = t.decode(&[0, 1, 2, 3, 4, 5, 6, 7], &[]).unwrap();
        assert_eq!(levels[0].layers, 2);
        // Texel (x, y, z) is byte x | y << 1 | z << 2; slice 1 follows slice 0.
        let alpha = |x: usize, y: usize, z: usize| levels[0].rgba[((z * 2 + y) * 2 + x) * 4 + 3];
        assert_eq!(
            [
                alpha(1, 0, 0),
                alpha(0, 1, 0),
                alpha(1, 0, 1),
                alpha(1, 1, 1)
            ],
            [1, 2, 5, 7]
        );
        assert_eq!(morton3(1, 1, 1, (4, 2, 2)), 7);
        assert_eq!(morton3(2, 0, 0, (4, 2, 2)), 8);
    }

    #[test]
    fn a_3d_textures_levels_stay_at_least_4x4() {
        let volume = |log_w: u32, log_h: u32| StageRegs {
            format: 2 << 28 | log_h << 24 | log_w << 20 | 9 << 16 | 6 << 8 | 3 << 4,
            ..StageRegs::default()
        };
        assert_eq!(volume(1, 3).texture().unwrap().levels, 1);
        assert_eq!(volume(2, 2).texture().unwrap().levels, 1);
        assert_eq!(volume(3, 4).texture().unwrap().levels, 2);
        // The slices halve with the levels.
        let t = volume(3, 3).texture().unwrap();
        assert_eq!((t.depth, t.level_depth(1), t.level_depth(5)), (4, 2, 1));
        assert_eq!(t.byte_len(), (8 * 8 * 4 + 4 * 4 * 2) * 4);
    }

    #[test]
    fn a_3d_dxt_texture_keeps_each_blocks_slices_together() {
        // DXT1, 3D, 4x4x2: the first block is slice 0 (red), the second slice 1 (blue).
        let stage = StageRegs {
            format: 1 << 28 | 2 << 24 | 2 << 20 | 1 << 16 | 0x0C << 8 | 3 << 4,
            ..StageRegs::default()
        };
        let t = stage.texture().unwrap();
        assert_eq!((t.depth, t.levels, t.byte_len()), (2, 1, 16));
        let mut data = vec![0x00, 0xF8, 0x00, 0xF8, 0, 0, 0, 0];
        data.extend([0x1F, 0x00, 0x1F, 0x00, 0, 0, 0, 0]);
        let level = &t.decode(&data, &[]).unwrap()[0];
        assert_eq!(level.rgba.len(), 4 * 4 * 2 * 4);
        assert_eq!(&level.rgba[..4], &[255, 0, 0, 255]);
        assert_eq!(&level.rgba[64..68], &[0, 0, 255, 255]);
    }

    #[test]
    fn a_cube_map_has_six_aligned_faces_decoded_level_by_level() {
        // A8, a cube map of 2x2 faces with two levels: 5 bytes a face, faces 128 bytes apart.
        let stage = StageRegs {
            format: 1 << 24 | 1 << 20 | 2 << 16 | 0x19 << 8 | 2 << 4 | 1 << 2,
            ..StageRegs::default()
        };
        let t = stage.texture().unwrap();
        assert!(t.cube);
        assert_eq!((t.shape(), t.byte_len()), (Shape::Cube, 5 * 128 + 5));
        let mut data = vec![0; 5 * 128 + 5];
        for face in 0..6 {
            let value = u8::try_from(10 * face).unwrap();
            data[face * 128..][..5].fill(value);
            data[face * 128 + 4] = value + 1;
        }
        let levels = t.decode(&data, &[]).unwrap();
        assert_eq!((levels[0].layers, levels[1].layers), (6, 6));
        assert_eq!(
            (levels[0].rgba.len(), levels[1].rgba.len()),
            (6 * 16, 6 * 4)
        );
        assert_eq!(levels[0].rgba[16 + 3], 10);
        let small: Vec<u8> = levels[1].rgba.iter().skip(3).step_by(4).copied().collect();
        assert_eq!(small, [1, 11, 21, 31, 41, 51]);
        // A linear format is never a cube map or 3D.
        let linear = StageRegs {
            format: 0x12 << 8 | 3 << 4 | 1 << 2,
            ..StageRegs::default()
        };
        assert_eq!(linear.texture().unwrap().shape(), Shape::Flat);
    }

    #[test]
    fn the_stage_modes_are_told_apart_by_stage() {
        // PROJECT3D, BUMPENVMAP, DOT_RFLCT_DIFF and DOT_RFLCT_SPEC.
        let program = Some(2 | 6 << 5 | 0x0B << 10 | 0x0C << 15);
        let modes = [0, 1, 2, 3].map(|stage| Mode::of(program, stage, true));
        assert_eq!(
            modes,
            [
                Mode::Project3d,
                Mode::BumpEnvMap,
                Mode::DotReflectDiffuse,
                Mode::DotReflectSpecular
            ]
        );
        // A stage that cannot take its mode samples as PROJECT2D, or gives zero for a dot
        // product; a stage that is off reads nothing, but a dot product needs no texture.
        assert_eq!(Mode::of(Some(6), 0, true), Mode::Sample);
        assert_eq!(Mode::of(Some(0x11 << 15), 3, true), Mode::Zero);
        assert_eq!(Mode::of(Some(0x0F << 5), 1, false), Mode::Off);
        assert_eq!(Mode::of(Some(0x11 << 5), 1, false), Mode::DotProduct);
        assert_eq!(Mode::of(Some(0), 0, true), Mode::Off);
        assert_eq!(Mode::of(Some(4), 0, true), Mode::Coordinates);
        assert!(!Mode::DotProduct.samples() && Mode::DotStrCube.samples());
        // Before a program is written stages 0 and 1 sample, 2 and 3 only while on.
        assert_eq!(Mode::of(None, 1, false), Mode::Sample);
        assert_eq!(Mode::of(None, 2, false), Mode::Off);
        assert_eq!(Mode::of(None, 3, true), Mode::Sample);
    }

    #[test]
    fn the_bump_environment_takes_its_words_in_the_hardwares_order() {
        let mut shader = Shader::default();
        for (k, value) in (0u32..).zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!(shader.set(
                STAGE_BASE + 3 * STAGE_STRIDE + 0x28 + 4 * k,
                value.to_bits()
            ));
            // Stage 0 has no bump environment.
            assert!(shader.set(STAGE_BASE + 0x28 + 4 * k, value.to_bits()));
        }
        assert!(shader.set(STAGE_BASE + 2 * STAGE_STRIDE + 0x3C, 0.25f32.to_bits()));
        assert!(shader.set(STAGE_BASE + STAGE_STRIDE + 0x38, 0.5f32.to_bits()));
        assert_eq!(shader.bump_matrix[3], [1.0, 2.0, 4.0, 3.0]);
        assert_eq!(shader.bump_matrix[0], [0.0; 4]);
        assert_eq!((shader.bump_offset[2], shader.bump_scale[1]), (0.25, 0.5));
        assert!(!shader.set(STAGE_BASE + 4, 0));
        assert!(shader.set(DOT_RGBMAPPING, 0x421));
        assert!(shader.set(SHADER_OTHER_STAGE_INPUT, 0x0021_0000));
        assert_eq!(
            (shader.mapping(1), shader.mapping(3), shader.mapping(0)),
            (1, 4, 0)
        );
        assert_eq!(
            (shader.input(1), shader.input(2), shader.input(3)),
            (0, 1, 2)
        );
    }
}
