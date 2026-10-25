//! The fixed-function vertex pipeline: what runs when the transform execution mode is not a
//! vertex program. Skinning gives the eye-space position and normal; texture coordinates are
//! generated and go through the texture matrices; the lights and the material colour the
//! vertex.
//!
//! The matrices, the texture-generation planes and the eye position live in the program
//! constants (`program::transform_constant`), which the caller passes in; the lights, the
//! material and the modes are methods kept here. Direct3D folds the material colours into the
//! light colours it sends, so the material sources only say which vertex colour, if any,
//! scales each light term.

use serde::{Deserialize, Serialize};

type Vec3 = [f32; 3];
type Vec4 = [f32; 4];

// Methods [F].
const LIGHT_CONTROL: u32 = 0x0294;
const COLOUR_MATERIAL: u32 = 0x0298;
const LIGHTING_ENABLE: u32 = 0x0314;
const SKIN_MODE: u32 = 0x0328;
const NORMALIZATION_ENABLE: u32 = 0x03A4;
const MATERIAL_EMISSION: u32 = 0x03A8;
const MATERIAL_ALPHA: u32 = 0x03B4;
const SPECULAR_ENABLE: u32 = 0x03B8;
const LIGHT_ENABLE_MASK: u32 = 0x03BC;
const TEXGEN: u32 = 0x03C0;
const TEXTURE_MATRIX_ENABLE: u32 = 0x0420;
const TEXGEN_VIEW_MODEL: u32 = 0x09CC;
const SPECULAR_PARAMS: u32 = 0x09E0;
const SCENE_AMBIENT: u32 = 0x0A10;
const LIGHTS: u32 = 0x1000;
const LIGHT_STRIDE: u32 = 0x80;

/// `SET_LIGHT_CONTROL` bits: the specular colour kept as the second colour (else added to the
/// diffuse one), the viewer at the eye position (else at infinity), and the second colour's
/// alpha kept (else 1).
const SEPARATE_SPECULAR: u32 = 1;
const LOCAL_VIEWER: u32 = 1 << 16;
const SPECULAR_ALPHA: u32 = 1 << 17;

// Constant rows: model-view matrix k at 8 + 8k with its inverse four rows on, the eye, stage
// s's texture-generation planes (S, T, R, Q) and texture matrix eight rows apart.
const MODEL_VIEW: usize = 0x08;
const INVERSE_MODEL_VIEW: usize = 0x0C;
const EYE: usize = 0x38;
const TEXGEN_PLANES: usize = 0x40;
const TEXTURE_MATRIX: usize = 0x44;

// Attributes.
pub const POSITION: usize = 0;
pub const WEIGHT: usize = 1;
pub const NORMAL: usize = 2;
pub const DIFFUSE: usize = 3;
pub const SPECULAR: usize = 4;
pub const FOG_COORDINATE: usize = 5;
pub const TEXTURE0: usize = 9;

/// The coefficients `(L, M, N)` of the power approximation `(x + L) / (M x + N)` the engine
/// starts with, about a power of 1.
const POWER_ONE: Vec3 = [0.0, -0.494_592, 1.494_592];

// Where a light term's colour comes from (`SET_COLOR_MATERIAL`, two bits each: emission,
// ambient, diffuse, specular).
const FROM_MATERIAL: u32 = 0;
const FROM_DIFFUSE: u32 = 1;
const FROM_SPECULAR: u32 = 2;

/// How one texture coordinate is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
enum Texgen {
    /// The vertex's own.
    #[default]
    Own,
    /// The plane against the eye-space position.
    EyeLinear,
    /// The plane against the vertex's own position.
    ObjectLinear,
    /// The reflected view direction mapped onto a sphere (S and T only).
    SphereMap,
    /// The eye-space normal (S, T and R).
    NormalMap,
    /// The reflected view direction (S, T and R).
    ReflectionMap,
}

impl Texgen {
    /// The mode a `SET_TEXGEN_*` value selects (the GL enums), or `None` for one the engine
    /// does not take.
    fn from_method(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Own,
            0x2400 => Self::EyeLinear,
            0x2401 => Self::ObjectLinear,
            0x2402 => Self::SphereMap,
            0x8511 => Self::NormalMap,
            0x8512 => Self::ReflectionMap,
            _ => return None,
        })
    }
}

/// One light's registers, as the floats of its method block (`0x80` bytes from `0x1000 +
/// 0x80 * light`), from 0x00: ambient, diffuse and specular colours (three each), range,
/// half vector (three), direction towards the light (three), spot falloff (three), spot
/// direction with the cone's offset (four), eye-space position (three), constant, linear and
/// quadratic attenuation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Light([f32; 29]);

impl Default for Light {
    fn default() -> Self {
        let mut light = Self([0.0; 29]);
        light.0[Self::FALLOFF / 4..Self::FALLOFF / 4 + 3].copy_from_slice(&POWER_ONE);
        light
    }
}

impl Light {
    const AMBIENT: usize = 0x00;
    const DIFFUSE: usize = 0x0C;
    const SPECULAR: usize = 0x18;
    const RANGE: usize = 0x24;
    const HALF_VECTOR: usize = 0x28;
    const DIRECTION: usize = 0x34;
    const FALLOFF: usize = 0x40;
    const SPOT: usize = 0x4C;
    const POSITION: usize = 0x5C;
    const ATTENUATION: usize = 0x68;

    fn three(&self, offset: usize) -> Vec3 {
        let at = offset / 4;
        [self.0[at], self.0[at + 1], self.0[at + 2]]
    }
}

/// What a light gives a vertex: its attenuation, and the normal's cosines to the light and to
/// the half vector, both at least 0.
struct Reach {
    attenuation: f32,
    diffuse: f32,
    highlight: f32,
}

/// The pipeline's state. Each flag is an enable register of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Pipeline {
    lighting: bool,
    /// `SET_LIGHT_ENABLE_MASK`: two bits a light, 0 off, 1 infinite, 2 local, 3 spot.
    light_mask: u32,
    colour_material: u32,
    normalize: bool,
    /// `SET_SKIN_MODE`.
    skin: u32,
    emission: Vec3,
    alpha: f32,
    scene_ambient: Vec3,
    /// The specular power's coefficients.
    power: Vec3,
    lights: [Light; 8],
    texgen: [[Texgen; 4]; 4],
    texture_matrix: [bool; 4],
    infinite_viewer: bool,
    light_control: u32,
    specular_enable: bool,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            lighting: false,
            light_mask: 0,
            colour_material: 0,
            normalize: false,
            skin: 0,
            emission: [0.0; 3],
            alpha: 0.0,
            scene_ambient: [0.0; 3],
            power: POWER_ONE,
            lights: [Light::default(); 8],
            texgen: [[Texgen::Own; 4]; 4],
            texture_matrix: [false; 4],
            infinite_viewer: false,
            light_control: 0,
            specular_enable: false,
        }
    }
}

/// What the pipeline gives one vertex.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Output {
    /// The position the composite matrix transforms: the vertex's own without skinning, the
    /// skinned eye-space one with it.
    pub position: Vec4,
    /// The eye-space position.
    pub eye: Vec4,
    /// The diffuse and specular colours, not clamped.
    pub diffuse: Vec4,
    pub specular: Vec4,
    /// The four stages' texture coordinates.
    pub texture: [Vec4; 4],
}

fn dot3(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dot4(a: Vec4, b: Vec4) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn xyz(v: Vec4) -> Vec3 {
    [v[0], v[1], v[2]]
}

fn plus(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn times(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn scaled(a: Vec3, s: f32) -> Vec3 {
    a.map(|c| c * s)
}

/// The unit vector along `a`; a zero vector gives NaNs, as on the GPU.
fn unit(a: Vec3) -> Vec3 {
    scaled(a, 1.0 / dot3(a, a).sqrt())
}

/// `incident` reflected about `normal`.
fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    plus(incident, scaled(normal, -2.0 * dot3(normal, incident)))
}

/// The engine's approximation of a power (the specular power, a spot's falloff):
/// `(x + L) / (M x + N)` in `0..=limit`, 0 for `x` at or below `-L` or a negative
/// denominator, `limit` for a zero one.
fn power(x: f32, [l, m, n]: Vec3, limit: f32) -> f32 {
    let denominator = m * x + n;
    if x <= -l || denominator < 0.0 {
        0.0
    } else if denominator == 0.0 {
        limit
    } else {
        ((x + l) / denominator).clamp(0.0, limit)
    }
}

impl Pipeline {
    /// Takes one method; returns whether it was one of the pipeline's.
    pub fn set(&mut self, method: u32, value: u32) -> bool {
        let float = f32::from_bits(value);
        let index = |base: u32| ((method - base) / 4) as usize;
        match method {
            LIGHT_CONTROL => self.light_control = value,
            SPECULAR_ENABLE => self.specular_enable = value != 0,
            LIGHTING_ENABLE => self.lighting = value != 0,
            LIGHT_ENABLE_MASK => self.light_mask = value & 0xFFFF,
            COLOUR_MATERIAL => self.colour_material = value,
            NORMALIZATION_ENABLE => self.normalize = value != 0,
            SKIN_MODE => self.skin = value & 7,
            m if (MATERIAL_EMISSION..MATERIAL_ALPHA).contains(&m) => {
                self.emission[index(MATERIAL_EMISSION)] = float;
            }
            MATERIAL_ALPHA => self.alpha = float,
            m if (TEXGEN..TEXGEN + 0x40).contains(&m) => {
                let n = index(TEXGEN);
                if let Some(mode) = Texgen::from_method(value) {
                    self.texgen[n / 4][n % 4] = mode;
                }
            }
            m if (TEXTURE_MATRIX_ENABLE..TEXTURE_MATRIX_ENABLE + 16).contains(&m) => {
                self.texture_matrix[index(TEXTURE_MATRIX_ENABLE)] = value != 0;
            }
            TEXGEN_VIEW_MODEL => self.infinite_viewer = value & 1 != 0,
            m if (SPECULAR_PARAMS..SPECULAR_PARAMS + 12).contains(&m) => {
                self.power[index(SPECULAR_PARAMS)] = float;
            }
            m if (SCENE_AMBIENT..SCENE_AMBIENT + 12).contains(&m) => {
                self.scene_ambient[index(SCENE_AMBIENT)] = float;
            }
            m if (LIGHTS..LIGHTS + 8 * LIGHT_STRIDE).contains(&m) => {
                let light = &mut self.lights[((m - LIGHTS) / LIGHT_STRIDE) as usize];
                let Some(slot) = light.0.get_mut(((m - LIGHTS) % LIGHT_STRIDE / 4) as usize) else {
                    return false;
                };
                *slot = float;
            }
            _ => return false,
        }
        true
    }

    /// Runs the pipeline on one vertex's attributes with the program constants `constants`.
    #[must_use]
    pub fn run(&self, inputs: &[Vec4; 16], constants: &[Vec4]) -> Output {
        let row = |r: usize| constants.get(r).copied().unwrap_or([0.0; 4]);
        let (eye, normal) = self.skinned(inputs, &row);
        let normal = if self.normalize { unit(normal) } else { normal };
        let texture = [0, 1, 2, 3].map(|stage| self.coordinates(stage, inputs, &row, eye, normal));
        let (mut diffuse, mut specular) = if self.lighting {
            self.lit(inputs, &row, eye, normal)
        } else {
            (inputs[DIFFUSE], inputs[SPECULAR])
        };
        if self.specular_enable {
            if self.light_control & SEPARATE_SPECULAR == 0 {
                // The highlight joins the diffuse colour; the second colour is the vertex's.
                if self.lighting {
                    for (d, s) in diffuse.iter_mut().zip(specular).take(3) {
                        *d += s;
                    }
                }
                specular = inputs[SPECULAR];
            }
            if self.light_control & SPECULAR_ALPHA == 0 {
                specular[3] = 1.0;
            }
        } else {
            specular = [0.0, 0.0, 0.0, 1.0];
        }
        Output {
            position: if self.skin == 0 {
                inputs[POSITION]
            } else {
                eye
            },
            eye,
            diffuse,
            specular,
            texture,
        }
    }

    /// The eye-space position and normal. Without skinning, through the first model-view
    /// matrix (each output component a row against the position) and the first three rows of
    /// its inverse (against the normal). With it, the sum over 2, 3 or 4 matrices weighted by
    /// attribute 1's components: modes 1, 3 and 5 make the last weight 1 minus the others,
    /// modes 2, 4 and 6 take every weight as given.
    fn skinned(&self, inputs: &[Vec4; 16], row: &dyn Fn(usize) -> Vec4) -> (Vec4, Vec3) {
        let position = inputs[POSITION];
        let normal = xyz(inputs[NORMAL]);
        let through = |matrix: usize| -> (Vec4, Vec3) {
            let base = 8 * matrix;
            (
                [0, 1, 2, 3].map(|i| dot4(row(MODEL_VIEW + base + i), position)),
                [0, 1, 2].map(|i| dot3(xyz(row(INVERSE_MODEL_VIEW + base + i)), normal)),
            )
        };
        let (matrices, made_up) = match self.skin {
            1..=6 => ((self.skin as usize).div_ceil(2) + 1, self.skin % 2 == 1),
            _ => return through(0),
        };
        let weights = inputs[WEIGHT];
        let mut eye = [0.0; 4];
        let mut eye_normal = [0.0; 3];
        let mut left = 1.0;
        for (matrix, &given) in weights.iter().enumerate().take(matrices) {
            let weight = if made_up && matrix + 1 == matrices {
                left
            } else {
                left -= given;
                given
            };
            let (p, n) = through(matrix);
            for (e, c) in eye.iter_mut().zip(p) {
                *e += c * weight;
            }
            for (e, c) in eye_normal.iter_mut().zip(n) {
                *e += c * weight;
            }
        }
        (eye, eye_normal)
    }

    /// Stage `stage`'s coordinates: S, T, R and Q each the vertex's own or generated, then
    /// through the stage's texture matrix when it is on.
    fn coordinates(
        &self,
        stage: usize,
        inputs: &[Vec4; 16],
        row: &dyn Fn(usize) -> Vec4,
        eye: Vec4,
        normal: Vec3,
    ) -> Vec4 {
        let own = inputs[TEXTURE0 + stage];
        // The view direction (towards the vertex) reflected about the normal.
        let reflected = || {
            let view = if self.infinite_viewer {
                [0.0, 0.0, 1.0]
            } else {
                unit(xyz(eye))
            };
            reflect(view, normal)
        };
        let mut out = own;
        for (component, value) in out.iter_mut().enumerate() {
            let plane = || row(TEXGEN_PLANES + 8 * stage + component);
            *value = match self.texgen[stage][component] {
                Texgen::EyeLinear => dot4(plane(), eye),
                Texgen::ObjectLinear => dot4(plane(), inputs[POSITION]),
                Texgen::SphereMap if component < 2 => {
                    let r = reflected();
                    let toward = plus(r, [0.0, 0.0, 1.0]);
                    r[component] / (2.0 * dot3(toward, toward).sqrt()) + 0.5
                }
                Texgen::NormalMap if component < 3 => normal[component],
                Texgen::ReflectionMap if component < 3 => reflected()[component],
                _ => own[component],
            };
        }
        if self.texture_matrix[stage] {
            let t = out;
            out = [0, 1, 2, 3].map(|i| dot4(row(TEXTURE_MATRIX + 8 * stage + i), t));
        }
        out
    }

    /// How light `index` of `kind` reaches a vertex at eye-space `point` with `normal`, or
    /// `None` when it adds nothing. `to_viewer` is the unit direction to a local viewer.
    fn reach(
        &self,
        index: usize,
        kind: u32,
        point: Vec3,
        normal: Vec3,
        to_viewer: Option<Vec3>,
    ) -> Option<Reach> {
        let light = &self.lights[index];
        let (attenuation, direction, half) = match kind {
            // Infinite: the direction register points at the light; its half vector is the
            // register's unless the viewer is local.
            1 => {
                let direction = unit(light.three(Light::DIRECTION));
                let half = to_viewer.map_or(light.three(Light::HALF_VECTOR), |v| {
                    unit(plus(direction, v))
                });
                (1.0, direction, half)
            }
            2 | 3 => {
                let position = light.three(Light::POSITION);
                let to_light = [0, 1, 2].map(|i| position[i] - point[i]);
                let distance = dot3(to_light, to_light).sqrt();
                if distance.is_nan() || distance > light.0[Light::RANGE / 4] {
                    return None;
                }
                let direction = unit(to_light);
                let [constant, linear, quadratic] = light.three(Light::ATTENUATION);
                let mut attenuation =
                    1.0 / (constant + linear * distance + quadratic * distance * distance);
                if kind == 3 {
                    let at = Light::SPOT / 4;
                    let spot = &light.0[at..at + 4];
                    let cone = dot3([spot[0], spot[1], spot[2]], direction) + spot[3];
                    if cone <= 0.0 {
                        attenuation = 0.0;
                    } else if cone < 1.0 {
                        attenuation *= power(cone, light.three(Light::FALLOFF), 1.0);
                    }
                }
                let half = unit(plus(direction, to_viewer.unwrap_or([0.0; 3])));
                (attenuation, direction, half)
            }
            _ => return None,
        };
        Some(Reach {
            attenuation,
            diffuse: dot3(normal, direction).max(0.0),
            highlight: dot3(normal, half).max(0.0),
        })
    }

    /// The lit diffuse and specular colours: a base from the emission and the scene ambient,
    /// then each light's ambient, diffuse and specular terms, each scaled by the vertex colour
    /// its material source names (the material itself leaves the term as the light gives it).
    fn lit(
        &self,
        inputs: &[Vec4; 16],
        row: &dyn Fn(usize) -> Vec4,
        eye: Vec4,
        normal: Vec3,
    ) -> (Vec4, Vec4) {
        let source = |shift: u32| (self.colour_material >> shift) & 3;
        let (emission, ambient, diffuse, specular) = (source(0), source(2), source(4), source(6));
        let (vertex_diffuse, vertex_specular) = (inputs[DIFFUSE], inputs[SPECULAR]);
        let vertex = |source: u32| {
            if source == FROM_DIFFUSE {
                xyz(vertex_diffuse)
            } else {
                xyz(vertex_specular)
            }
        };
        // The scene ambient register stands for the material's own terms. A vertex colour as
        // the emission or ambient source adds that colour times the emission register.
        let base = match (emission, ambient) {
            (FROM_MATERIAL, FROM_MATERIAL) => self.scene_ambient,
            (FROM_MATERIAL, from) | (from, FROM_MATERIAL) => {
                plus(self.scene_ambient, times(vertex(from), self.emission))
            }
            (from_emission, from_ambient) => plus(
                vertex(from_emission),
                times(vertex(from_ambient), self.emission),
            ),
        };
        let alpha = match diffuse {
            FROM_MATERIAL => self.alpha,
            FROM_SPECULAR => vertex_specular[3],
            _ => vertex_diffuse[3],
        };
        let term = |source: u32, colour: Vec3| match source {
            FROM_MATERIAL => colour,
            FROM_DIFFUSE => times(xyz(vertex_diffuse), colour),
            FROM_SPECULAR => times(xyz(vertex_specular), colour),
            _ => [0.0; 3],
        };
        let point = xyz(eye).map(|c| c / eye[3]);
        let to_viewer = (self.light_control & LOCAL_VIEWER != 0).then(|| {
            let viewer = row(EYE);
            unit([0, 1, 2].map(|i| viewer[i] / viewer[3] - point[i]))
        });
        let mut lit_diffuse = base;
        let mut lit_specular = [0.0; 3];
        for index in 0..8 {
            let kind = (self.light_mask >> (2 * index)) & 3;
            let Some(reach) = self.reach(index, kind, point, normal, to_viewer) else {
                continue;
            };
            let light = &self.lights[index];
            let highlight = if reach.diffuse > 0.0 {
                power(reach.highlight, self.power, 10_000.0)
            } else {
                0.0
            };
            let a = reach.attenuation;
            lit_diffuse = plus(
                lit_diffuse,
                term(ambient, scaled(light.three(Light::AMBIENT), a)),
            );
            lit_diffuse = plus(
                lit_diffuse,
                term(
                    diffuse,
                    scaled(light.three(Light::DIFFUSE), a * reach.diffuse),
                ),
            );
            lit_specular = plus(
                lit_specular,
                term(
                    specular,
                    scaled(light.three(Light::SPECULAR), a * highlight),
                ),
            );
        }
        let [r, g, b] = lit_diffuse;
        let [sr, sg, sb] = lit_specular;
        ([r, g, b, alpha], [sr, sg, sb, vertex_specular[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constants with identity model-view and inverse model-view matrices.
    fn identity() -> Vec<Vec4> {
        let mut c = vec![[0.0; 4]; 192];
        for i in 0..4 {
            c[MODEL_VIEW + i][i] = 1.0;
            c[INVERSE_MODEL_VIEW + i][i] = 1.0;
        }
        c
    }

    /// A vertex at `position` with `normal`, `diffuse`, and a red specular of alpha 0.5.
    fn vertex(position: Vec4, normal: Vec3, diffuse: Vec4) -> [Vec4; 16] {
        let mut v = [[0.0, 0.0, 0.0, 1.0]; 16];
        v[POSITION] = position;
        v[NORMAL] = [normal[0], normal[1], normal[2], 1.0];
        v[DIFFUSE] = diffuse;
        v[SPECULAR] = [1.0, 0.0, 0.0, 0.5];
        v
    }

    fn floats(pipeline: &mut Pipeline, method: u32, values: &[f32]) {
        for (i, value) in (0u32..).zip(values) {
            assert!(pipeline.set(method + 4 * i, value.to_bits()), "{method:#x}");
        }
    }

    fn near(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn unlit_vertices_keep_their_colours() {
        let mut p = Pipeline::default();
        let v = vertex([1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 1.0], [0.2, 0.4, 0.6, 0.8]);
        let out = p.run(&v, &identity());
        assert_eq!(out.diffuse, [0.2, 0.4, 0.6, 0.8]);
        assert_eq!(out.specular, [0.0, 0.0, 0.0, 1.0], "specular is off");
        assert_eq!(out.position, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(out.eye, [1.0, 2.0, 3.0, 1.0]);
        p.set(SPECULAR_ENABLE, 1);
        assert_eq!(p.run(&v, &identity()).specular, [1.0, 0.0, 0.0, 1.0]);
        p.set(LIGHT_CONTROL, SPECULAR_ALPHA);
        assert_eq!(p.run(&v, &identity()).specular, [1.0, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn an_infinite_light_shades_by_the_cosine_to_the_normal() {
        let mut p = Pipeline::default();
        p.set(LIGHTING_ENABLE, 1);
        p.set(LIGHT_ENABLE_MASK, 1);
        floats(&mut p, SCENE_AMBIENT, &[0.1, 0.1, 0.1]);
        p.set(MATERIAL_ALPHA, 0.75f32.to_bits());
        floats(&mut p, LIGHTS, &[0.0, 0.0, 0.2]);
        floats(&mut p, LIGHTS + 0x0C, &[1.0, 0.5, 0.0]);
        // Sixty degrees from the normal: cosine one half.
        floats(&mut p, LIGHTS + 0x34, &[0.0, 3f32.sqrt() / 2.0, 0.5]);
        let v = vertex([0.0, 0.0, -5.0, 1.0], [0.0, 0.0, 1.0], [0.0; 4]);
        let out = p.run(&v, &identity());
        assert!(
            near(&out.diffuse, &[0.6, 0.35, 0.3, 0.75]),
            "{:?}",
            out.diffuse
        );
        // The vertex's diffuse colour as the diffuse source scales the light's.
        p.set(COLOUR_MATERIAL, FROM_DIFFUSE << 4);
        let v = vertex(
            [0.0, 0.0, -5.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.5, 1.0, 1.0, 0.25],
        );
        let out = p.run(&v, &identity());
        assert!(
            near(&out.diffuse, &[0.35, 0.35, 0.3, 0.25]),
            "{:?}",
            out.diffuse
        );
    }

    #[test]
    fn a_local_light_attenuates_and_ends_at_its_range() {
        let mut p = Pipeline::default();
        p.set(LIGHTING_ENABLE, 1);
        p.set(LIGHT_ENABLE_MASK, 2 << 2);
        let light = LIGHTS + LIGHT_STRIDE;
        floats(&mut p, light + 0x0C, &[1.0, 1.0, 1.0]);
        floats(&mut p, light + 0x24, &[10.0]);
        floats(&mut p, light + 0x5C, &[0.0, 0.0, 4.0]);
        floats(&mut p, light + 0x68, &[1.0, 0.0, 0.25]);
        let v = vertex([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0; 4]);
        // At distance 4: 1 / (1 + 0.25 * 16).
        assert!(near(&p.run(&v, &identity()).diffuse[..3], &[0.2; 3]));
        floats(&mut p, light + 0x24, &[3.0]);
        assert_eq!(p.run(&v, &identity()).diffuse[..3], [0.0; 3]);
    }

    #[test]
    fn a_spot_light_lights_inside_its_cone_only() {
        let mut p = Pipeline::default();
        p.set(LIGHTING_ENABLE, 1);
        p.set(LIGHT_ENABLE_MASK, 3);
        floats(&mut p, LIGHTS + 0x0C, &[1.0, 1.0, 1.0]);
        floats(&mut p, LIGHTS + 0x24, &[100.0]);
        floats(&mut p, LIGHTS + 0x5C, &[0.0, 0.0, 4.0]);
        floats(&mut p, LIGHTS + 0x68, &[1.0, 0.0, 0.0]);
        floats(&mut p, LIGHTS + 0x4C, &[0.0, 0.0, 1.0, 0.0]);
        let v = vertex([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0; 4]);
        assert!(near(&p.run(&v, &identity()).diffuse[..3], &[1.0; 3]));
        floats(&mut p, LIGHTS + 0x4C, &[0.0, 0.0, -1.0, 0.0]);
        assert_eq!(p.run(&v, &identity()).diffuse[..3], [0.0; 3]);
        // Half way into the cone, the falloff's power: with (0, 0, 1) it is the cone itself.
        floats(&mut p, LIGHTS + 0x4C, &[0.0, 0.0, 1.0, -0.5]);
        floats(&mut p, LIGHTS + 0x40, &[0.0, 0.0, 1.0]);
        assert!(near(&p.run(&v, &identity()).diffuse[..3], &[0.5; 3]));
    }

    #[test]
    fn the_highlight_is_separate_or_added_and_the_viewer_may_be_local() {
        let mut p = Pipeline::default();
        p.set(LIGHTING_ENABLE, 1);
        p.set(LIGHT_ENABLE_MASK, 1);
        p.set(SPECULAR_ENABLE, 1);
        floats(&mut p, LIGHTS + 0x18, &[1.0, 1.0, 1.0]);
        floats(&mut p, LIGHTS + 0x34, &[0.0, 0.0, 1.0]);
        floats(&mut p, LIGHTS + 0x28, &[0.0, 0.0, 1.0]);
        floats(&mut p, SPECULAR_PARAMS, &[0.0, 0.0, 1.0]);
        let v = vertex([0.0, 0.0, -1.0, 1.0], [0.0, 0.6, 0.8], [0.0; 4]);
        p.set(LIGHT_CONTROL, SEPARATE_SPECULAR | SPECULAR_ALPHA);
        let out = p.run(&v, &identity());
        assert!(
            near(&out.specular, &[0.8, 0.8, 0.8, 0.5]),
            "{:?}",
            out.specular
        );
        p.set(LIGHT_CONTROL, SPECULAR_ALPHA);
        let out = p.run(&v, &identity());
        assert!(near(&out.diffuse[..3], &[0.8; 3]), "{:?}", out.diffuse);
        assert_eq!(out.specular, [1.0, 0.0, 0.0, 0.5]);
        // A viewer at the origin, straight above the vertex: the half vector is (0, 0, 1).
        let mut c = identity();
        c[EYE] = [0.0, 0.0, 0.0, 1.0];
        p.set(LIGHT_CONTROL, SEPARATE_SPECULAR | LOCAL_VIEWER);
        assert!(near(&p.run(&v, &c).specular[..3], &[0.8; 3]));
    }

    #[test]
    fn texgen_planes_and_the_texture_matrix_make_coordinates() {
        let mut p = Pipeline::default();
        let mut c = identity();
        // Stage 1: S object linear on (1, 0, 0, 0.5), T eye linear on (0, 1, 0, 0).
        p.set(TEXGEN + 16, 0x2401);
        p.set(TEXGEN + 20, 0x2400);
        c[TEXGEN_PLANES + 8] = [1.0, 0.0, 0.0, 0.5];
        c[TEXGEN_PLANES + 9] = [0.0, 1.0, 0.0, 0.0];
        c[MODEL_VIEW + 1][3] = 2.0;
        let v = vertex([3.0, 4.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0; 4]);
        assert_eq!(p.run(&v, &c).texture[1], [3.5, 6.0, 0.0, 1.0]);
        // Stage 1's texture matrix swaps S and T.
        p.set(TEXTURE_MATRIX_ENABLE + 4, 1);
        c[TEXTURE_MATRIX + 8] = [0.0, 1.0, 0.0, 0.0];
        c[TEXTURE_MATRIX + 9] = [1.0, 0.0, 0.0, 0.0];
        c[TEXTURE_MATRIX + 10] = [0.0, 0.0, 1.0, 0.0];
        c[TEXTURE_MATRIX + 11] = [0.0, 0.0, 0.0, 1.0];
        assert_eq!(p.run(&v, &c).texture[1], [6.0, 3.5, 0.0, 1.0]);
        assert_eq!(
            p.run(&v, &c).texture[0],
            v[TEXTURE0],
            "stage 0 keeps its own"
        );
        // An unknown mode leaves the last one.
        p.set(TEXGEN + 16, 0x1234);
        assert_eq!(p.texgen[1][0], Texgen::ObjectLinear);
    }

    #[test]
    fn normal_sphere_and_reflection_maps_follow_the_normal() {
        let mut p = Pipeline::default();
        for component in 0..3 {
            p.set(TEXGEN + 4 * component, 0x8511);
            p.set(TEXGEN + 32 + 4 * component, 0x8512);
        }
        p.set(TEXGEN + 16, 0x2402);
        p.set(TEXGEN + 20, 0x2402);
        p.set(TEXGEN_VIEW_MODEL, 1);
        let v = vertex([0.0, 0.0, -1.0, 1.0], [0.6, 0.0, 0.8], [0.0; 4]);
        let out = p.run(&v, &identity());
        assert_eq!(out.texture[0][..3], [0.6, 0.0, 0.8]);
        // (0, 0, 1) about (0.6, 0, 0.8): (-0.96, 0, -0.28).
        assert!(
            near(&out.texture[2][..3], &[-0.96, 0.0, -0.28]),
            "{:?}",
            out.texture[2]
        );
        // The sphere map: r / (2 |r + (0, 0, 1)|) + 0.5, |(-0.96, 0, 0.72)| = 1.2.
        assert!(
            near(&out.texture[1][..2], &[-0.96 / 2.4 + 0.5, 0.5]),
            "{:?}",
            out.texture[1]
        );
    }

    #[test]
    fn skinning_blends_the_model_view_matrices() {
        let mut p = Pipeline::default();
        let mut c = identity();
        for i in 0..4 {
            c[MODEL_VIEW + 8 + i][i] = 1.0;
            c[INVERSE_MODEL_VIEW + 8 + i][i] = 1.0;
        }
        c[MODEL_VIEW + 8][3] = 10.0;
        // Two matrices, the second weight made up: 0.25 and 0.75.
        p.set(SKIN_MODE, 1);
        let mut v = vertex([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0; 4]);
        v[WEIGHT] = [0.25, 0.5, 0.0, 1.0];
        let out = p.run(&v, &c);
        assert_eq!(out.position, [8.5, 0.0, 0.0, 1.0]);
        assert_eq!(out.eye, out.position);
        // Mode 2 takes both weights as given: 0.25 and 0.5.
        p.set(SKIN_MODE, 2);
        assert_eq!(p.run(&v, &c).eye, [5.75, 0.0, 0.0, 0.75]);
        // Off: the composite matrix takes the vertex's own position.
        p.set(SKIN_MODE, 0);
        assert_eq!(p.run(&v, &c).position, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn the_power_approximation() {
        assert_eq!(power(0.5, [0.0, 0.0, 1.0], 1.0), 0.5);
        assert_eq!(power(-0.5, [0.0, 0.0, 1.0], 1.0), 0.0);
        assert_eq!(power(0.5, [0.0, 0.0, 0.0], 7.0), 7.0);
        assert_eq!(power(0.5, [0.0, -4.0, 1.0], 7.0), 0.0);
        assert!((power(1.0, POWER_ONE, 10.0) - 1.0).abs() < 1e-5);
    }
}
