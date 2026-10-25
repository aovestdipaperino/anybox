//! The texture stages' modes (`SET_SHADER_STAGE_PROGRAM`): where each stage reads its texture
//! and what it gives the combiners. This is the reference for the shader's stage code, which
//! mirrors it; keep the two in step.
//!
//! The stages run in order, each reading what the earlier ones gave: the texel of its input
//! stage (stage 0's for stage 1, `SET_SHADER_OTHER_STAGE_INPUT`'s for stages 2 and 3) and the
//! dot products the earlier stages made.

use crate::gpu::texture::{Mode, Shader, Shape};

type Vec4 = [f32; 4];
type Vec3 = [f32; 3];

/// A texel channel as the signed value the bump modes read it as: the byte two's complement,
/// over 127.
#[must_use]
pub fn signed(x: f32) -> f32 {
    let v = x * 255.0;
    if v >= 128.0 {
        (v - 256.0) / 127.0
    } else {
        v / 127.0
    }
}

/// A texel as `stage`'s dot product reads it, by the stage's dot mapping: 0 as it is (0..1); 1
/// `(v - 128) / 127`; 2 `(v + 0.5) / 127.5` below 128 and `(v - 255.5) / 127.5` from it; 3 the
/// byte two's complement over 127; 4 (`HILO_1`) alpha and red as one 16-bit value, green and
/// blue as another, then 1. The hemisphere mappings read it as it is. `v` is the channel's byte.
#[must_use]
pub fn dot_mapped(texel: Vec4, stage: usize, shader: &Shader) -> Vec3 {
    let rgb = [texel[0], texel[1], texel[2]];
    match shader.mapping(stage) {
        1 => rgb.map(|c| (c * 255.0 - 128.0) / 127.0),
        2 => rgb.map(|c| {
            let v = c * 255.0;
            if v >= 128.0 {
                (v - 255.5) / 127.5
            } else {
                (v + 0.5) / 127.5
            }
        }),
        3 => rgb.map(signed),
        4 => {
            // The channels are colours, so they fit in a byte.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let byte = |c: f32| u32::from((c.clamp(0.0, 1.0) * 255.0).round() as u8);
            let pair = |high: f32, low: f32| (byte(high) << 8 | byte(low)) as f32 / 65535.0;
            [pair(texel[3], texel[0]), pair(texel[1], texel[2]), 1.0]
        }
        _ => rgb,
    }
}

/// A direction's place on the face of a cube it points to, from -1 to 1 on that face: the
/// lookup a cube-map mode makes on a texture that is not a cube map. The major axis is the
/// strictly largest component (z on a tie); +x gives `(-z, y)`, -x `(z, y)`, +y `(x, -z)`, -y
/// `(x, z)`, +z `(x, y)` and -z `(-x, y)`, each over that component's size.
#[must_use]
pub fn cube_to_flat(c: Vec3) -> [f32; 2] {
    let a = c.map(f32::abs);
    let (u, v, m) = if a[0] > a[1] && a[0] > a[2] {
        if c[0] > 0.0 {
            (-c[2], c[1], a[0])
        } else {
            (c[2], c[1], a[0])
        }
    } else if a[1] > a[0] && a[1] > a[2] {
        if c[1] > 0.0 {
            (c[0], -c[2], a[1])
        } else {
            (c[0], c[2], a[1])
        }
    } else if c[2] > 0.0 {
        (c[0], c[1], a[2])
    } else {
        (-c[0], c[1], a[2])
    };
    [u / m, v / m]
}

fn dot3(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// What the stages computed so far for one fragment.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Fragment {
    /// Each stage's coordinates `s, t, r, q`.
    pub coordinates: [Vec4; 4],
    /// What each stage gave, once it ran.
    pub texels: [Vec4; 4],
    /// Each stage's dot product, once it ran.
    pub dots: [f32; 4],
}

impl Fragment {
    /// `stage`'s dot product of its `s, t, r` with its input texel, dot-mapped.
    #[must_use]
    pub fn dot(&self, stage: usize, shader: &Shader) -> f32 {
        let p = self.coordinates[stage];
        let texel = self.texels[shader.input(stage)];
        dot3([p[0], p[1], p[2]], dot_mapped(texel, stage, shader))
    }

    /// Where `stage` in `mode` reads its texture of shape `shape`: `x, y` for a flat lookup
    /// (before linear textures' scaling), `x, y, z` for a 3D one, the direction for a cube map.
    /// The stage's dot product must already be in `dots`.
    #[must_use]
    pub fn coordinates(&self, stage: usize, mode: Mode, shape: Shape, shader: &Shader) -> Vec3 {
        let p = self.coordinates[stage];
        let source = self.texels[shader.input(stage)];
        let d = self.dots;
        let toward = |c: Vec3| {
            if shape == Shape::Cube {
                c
            } else {
                let [u, v] = cube_to_flat(c);
                [u, v, 0.0]
            }
        };
        match mode {
            Mode::Sample => {
                let (s, t) = (p[0] / p[3], p[1] / p[3]);
                if shape == Shape::Cube {
                    let length = (1.0 + t * t + s * s).sqrt();
                    [1.0 / length, t / length, -s / length]
                } else {
                    [s, t, 0.0]
                }
            }
            Mode::Project3d => [p[0] / p[3], p[1] / p[3], p[2] / p[3]],
            Mode::CubeMap => toward([p[0], p[1], p[2]]),
            Mode::BumpEnvMap | Mode::BumpEnvMapLuminance => {
                let m = shader.bump_matrix[stage];
                let (du, dv) = (signed(source[2]), signed(source[1]));
                [
                    p[0] + m[0] * du + m[2] * dv,
                    p[1] + m[1] * du + m[3] * dv,
                    p[2],
                ]
            }
            Mode::DotSt => [d[stage - 1], d[stage], 0.0],
            Mode::DotReflectDiffuse => {
                // Stage 3 has not run: its dot product is made here.
                toward([d[1], d[2], self.dot(3, shader)])
            }
            Mode::DotReflectSpecular => {
                let n = [d[1], d[2], d[3]];
                let e = [
                    self.coordinates[1][3],
                    self.coordinates[2][3],
                    self.coordinates[3][3],
                ];
                let k = 2.0 * dot3(n, e) / dot3(n, n);
                toward([k * n[0] - e[0], k * n[1] - e[1], k * n[2] - e[2]])
            }
            Mode::DotStr3d => [d[1], d[2], d[3]],
            Mode::DotStrCube => toward([d[1], d[2], d[3]]),
            Mode::DependentAr => [source[3], source[0], 0.0],
            Mode::DependentGb => [source[1], source[2], 0.0],
            Mode::Off | Mode::Coordinates | Mode::Zero | Mode::DotProduct | Mode::DotZw => {
                [p[0], p[1], p[2]]
            }
        }
    }

    /// What `stage` in `mode` gives the combiners, from the texel its lookup found.
    #[must_use]
    pub fn staged(&self, stage: usize, mode: Mode, texel: Vec4, shader: &Shader) -> Vec4 {
        match mode {
            Mode::Off => [0.0, 0.0, 0.0, 1.0],
            Mode::Coordinates => self.coordinates[stage].map(|c| c.clamp(0.0, 1.0)),
            Mode::Zero | Mode::DotProduct | Mode::DotZw => [0.0; 4],
            Mode::BumpEnvMapLuminance => {
                let red = self.texels[shader.input(stage)][0];
                let l = shader.bump_scale[stage] * red + shader.bump_offset[stage];
                texel.map(|c| c * l)
            }
            _ => texel,
        }
    }
}

/// Which lookup a stage in `mode` makes on a texture of `shape`: the cube map for the modes that
/// look toward a direction or for `PROJECT2D`, the 3D texture for the modes that give three
/// coordinates or for `PROJECT2D` and the bump modes, else flat.
#[must_use]
pub fn lookup(mode: Mode, shape: Shape) -> Shape {
    use Mode::{
        BumpEnvMap, BumpEnvMapLuminance, CubeMap, DotReflectDiffuse, DotReflectSpecular, DotStr3d,
        DotStrCube, Project3d, Sample,
    };
    match shape {
        Shape::Cube
            if matches!(
                mode,
                Sample | CubeMap | DotReflectDiffuse | DotReflectSpecular | DotStrCube
            ) =>
        {
            Shape::Cube
        }
        Shape::Volume
            if matches!(
                mode,
                Sample | Project3d | BumpEnvMap | BumpEnvMapLuminance | DotStr3d
            ) =>
        {
            Shape::Volume
        }
        _ => Shape::Flat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn signed_channels_and_the_dot_mappings() {
        assert!(close(&[signed(1.0), signed(0.0)], &[-1.0 / 127.0, 0.0]));
        assert!(close(&[signed(127.0 / 255.0)], &[1.0]));
        let texel = [1.0, 0.0, 128.0 / 255.0, 1.0];
        let mut shader = Shader {
            dot_mapping: 0x321,
            ..Shader::default()
        };
        assert!(close(&dot_mapped(texel, 0, &shader), &texel[..3]));
        assert!(close(
            &dot_mapped(texel, 1, &shader),
            &[127.0 / 127.0, -128.0 / 127.0, 0.0]
        ));
        assert!(close(
            &dot_mapped(texel, 2, &shader),
            &[-0.5 / 127.5, 0.5 / 127.5, -127.5 / 127.5]
        ));
        assert!(close(
            &dot_mapped(texel, 3, &shader),
            &[-1.0 / 127.0, 0.0, -128.0 / 127.0]
        ));
        shader.dot_mapping = 4;
        assert!(close(
            &dot_mapped([1.0, 0.0, 1.0, 0.0], 1, &shader),
            &[255.0 / 65535.0, 255.0 / 65535.0, 1.0]
        ));
    }

    #[test]
    fn a_direction_lands_on_the_face_it_points_to() {
        assert!(close(&cube_to_flat([2.0, 1.0, 0.5]), &[-0.25, 0.5]));
        assert!(close(&cube_to_flat([-2.0, 1.0, 0.5]), &[0.25, 0.5]));
        assert!(close(&cube_to_flat([0.5, 2.0, 1.0]), &[0.25, -0.5]));
        assert!(close(&cube_to_flat([0.5, -2.0, 1.0]), &[0.25, 0.5]));
        assert!(close(&cube_to_flat([0.5, 1.0, 2.0]), &[0.25, 0.5]));
        assert!(close(&cube_to_flat([0.5, 1.0, -2.0]), &[-0.25, 0.5]));
    }

    #[test]
    fn projective_modes_divide_by_q() {
        let f = Fragment {
            coordinates: [[1.0, 2.0, 3.0, 2.0], [0.0; 4], [0.0; 4], [0.0; 4]],
            ..Fragment::default()
        };
        let shader = Shader::default();
        assert!(close(
            &f.coordinates(0, Mode::Sample, Shape::Flat, &shader),
            &[0.5, 1.0, 0.0]
        ));
        assert!(close(
            &f.coordinates(0, Mode::Project3d, Shape::Volume, &shader),
            &[0.5, 1.0, 1.5]
        ));
        // PROJECT2D on a cube map looks toward (1, t, -s).
        let n = (1.0f32 + 1.0 + 0.25).sqrt();
        assert!(close(
            &f.coordinates(0, Mode::Sample, Shape::Cube, &shader),
            &[1.0 / n, 1.0 / n, -0.5 / n]
        ));
        assert!(close(
            &f.coordinates(0, Mode::CubeMap, Shape::Flat, &shader),
            &[1.0 / 3.0, 2.0 / 3.0, 0.0]
        ));
    }

    #[test]
    fn bump_mapping_turns_the_input_offsets_by_the_matrix_and_scales_by_luminance() {
        let mut shader = Shader::default();
        shader.bump_matrix[1] = [0.5, 0.25, 2.0, 1.0];
        shader.bump_scale[1] = 0.5;
        shader.bump_offset[1] = 0.25;
        let f = Fragment {
            coordinates: [[0.0; 4], [0.1, 0.2, 0.3, 1.0], [0.0; 4], [0.0; 4]],
            // du = 1 (blue 127), dv = -1 (green 129).
            texels: [
                [1.0, 129.0 / 255.0, 127.0 / 255.0, 1.0],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
            ],
            ..Fragment::default()
        };
        assert!(close(
            &f.coordinates(1, Mode::BumpEnvMap, Shape::Flat, &shader),
            &[0.1 + 0.5 - 2.0, 0.2 + 0.25 - 1.0, 0.3]
        ));
        let texel = f.staged(1, Mode::BumpEnvMapLuminance, [1.0; 4], &shader);
        assert!(close(&texel, &[0.75; 4]));
    }

    #[test]
    fn dot_products_feed_the_later_stages() {
        let shader = Shader {
            // Stage 3 reads stage 0's texel.
            other_input: 0,
            ..Shader::default()
        };
        let mut f = Fragment {
            coordinates: [
                [0.0; 4],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            texels: [[0.2, 0.4, 0.6, 1.0], [0.0; 4], [0.0; 4], [0.0; 4]],
            ..Fragment::default()
        };
        for stage in 1..4 {
            f.dots[stage] = f.dot(stage, &shader);
        }
        assert!(close(&f.dots, &[0.0, 0.2, 0.4, 0.6]));
        assert!(close(
            &f.coordinates(2, Mode::DotSt, Shape::Flat, &shader),
            &[0.2, 0.4, 0.0]
        ));
        assert!(close(
            &f.coordinates(3, Mode::DotStr3d, Shape::Volume, &shader),
            &[0.2, 0.4, 0.6]
        ));
        assert!(close(
            &f.coordinates(2, Mode::DotReflectDiffuse, Shape::Cube, &shader),
            &[0.2, 0.4, 0.6]
        ));
        // The eye vector (0, 1, 0) reflected about the normal (0.2, 0.4, 0.6).
        let n = [0.2f32, 0.4, 0.6];
        let k = 2.0 * 0.4 / (0.04 + 0.16 + 0.36);
        assert!(close(
            &f.coordinates(3, Mode::DotReflectSpecular, Shape::Cube, &shader),
            &[k * n[0], k * n[1] - 1.0, k * n[2]]
        ));
    }

    #[test]
    fn dependent_reads_take_two_channels_of_the_input() {
        let f = Fragment {
            texels: [[0.1, 0.2, 0.3, 0.4], [0.0; 4], [0.0; 4], [0.0; 4]],
            ..Fragment::default()
        };
        let shader = Shader::default();
        assert!(close(
            &f.coordinates(1, Mode::DependentAr, Shape::Flat, &shader),
            &[0.4, 0.1, 0.0]
        ));
        assert!(close(
            &f.coordinates(1, Mode::DependentGb, Shape::Flat, &shader),
            &[0.2, 0.3, 0.0]
        ));
    }

    #[test]
    fn each_mode_takes_the_lookup_its_texture_allows() {
        assert_eq!(lookup(Mode::Sample, Shape::Cube), Shape::Cube);
        assert_eq!(lookup(Mode::DependentAr, Shape::Cube), Shape::Flat);
        assert_eq!(lookup(Mode::BumpEnvMap, Shape::Volume), Shape::Volume);
        assert_eq!(lookup(Mode::CubeMap, Shape::Volume), Shape::Flat);
        assert_eq!(lookup(Mode::DotStrCube, Shape::Flat), Shape::Flat);
    }

    #[test]
    fn the_stages_that_give_no_texel() {
        let f = Fragment {
            coordinates: [[2.0, 0.5, -1.0, 1.0], [0.0; 4], [0.0; 4], [0.0; 4]],
            ..Fragment::default()
        };
        let shader = Shader::default();
        let texel = [0.3; 4];
        assert_eq!(f.staged(0, Mode::Off, texel, &shader), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            f.staged(0, Mode::Coordinates, texel, &shader),
            [1.0, 0.5, 0.0, 1.0]
        );
        assert_eq!(f.staged(0, Mode::DotProduct, texel, &shader), [0.0; 4]);
        assert_eq!(f.staged(0, Mode::DependentGb, texel, &shader), texel);
    }
}
