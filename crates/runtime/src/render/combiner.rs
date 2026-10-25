//! Register combiners (spec 11.5): an evaluator that is the reference for the shader, and the
//! uniform block the shader reads.
//!
//! The semantics are NVIDIA's register combiners (`GL_NV_register_combiners` and
//! `GL_NV_register_combiners2`); the word encoding follows the spec, which asks for confirmation on
//! hardware.

use crate::gpu::draw::Combiner;
use crate::gpu::texture::{Mode, Shader, Shape};

type Vec4 = [f32; 4];

/// What a fragment starts with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fragment {
    pub texture0: Vec4,
    pub texture1: Vec4,
    /// What stages 2 and 3 give (black with alpha 1 when off).
    pub texture2: Vec4,
    pub texture3: Vec4,
    pub colour: Vec4,
    /// `(r, g, b, 1 - fog)`, or zero without fog.
    pub fog: Vec4,
    /// The specular colour: register 5.
    pub specular: Vec4,
}

/// `0xAARRGGBB` as RGBA in 0..1.
#[must_use]
pub fn colour(argb: u32) -> Vec4 {
    let [b, g, r, a] = argb.to_le_bytes();
    [r, g, b, a].map(|c| f32::from(c) / 255.0)
}

/// An input mapping (bits 5..7 of an input byte).
fn map(mapping: u32, x: f32) -> f32 {
    let positive = x.max(0.0);
    match mapping {
        0 => positive,
        1 => 1.0 - x.clamp(0.0, 1.0),
        2 => 2.0 * positive - 1.0,
        3 => 1.0 - 2.0 * positive,
        4 => positive - 0.5,
        5 => 0.5 - positive,
        6 => x,
        _ => -x,
    }
}

/// A general-stage colour input: register bits 0..3, bit 4 the alpha replicated, mapped.
fn colour_input(byte: u32, regs: &[Vec4; 16]) -> [f32; 3] {
    let r = regs[(byte & 0xF) as usize];
    let v = if byte & 0x10 != 0 {
        [r[3]; 3]
    } else {
        [r[0], r[1], r[2]]
    };
    v.map(|x| map(byte >> 5 & 7, x))
}

/// A general-stage alpha input: bit 4 selects alpha, else blue.
fn alpha_input(byte: u32, regs: &[Vec4; 16]) -> f32 {
    let r = regs[(byte & 0xF) as usize];
    map(byte >> 5 & 7, if byte & 0x10 != 0 { r[3] } else { r[2] })
}

/// Scale and bias (bits 15..17 of an output word), then the clamp to -1..1.
fn scale(word: u32, x: f32) -> f32 {
    let y = match (word >> 15) & 7 {
        1 => x - 0.5,
        2 => x * 2.0,
        3 => (x - 0.5) * 2.0,
        4 => x * 4.0,
        6 => x * 0.5,
        _ => x,
    };
    y.clamp(-1.0, 1.0)
}

/// The mux: spare 0's alpha at least 0.5 (control bit 0), else the low bit of its 8-bit value.
fn mux(control: u32, regs: &[Vec4; 16]) -> bool {
    let a = regs[12][3];
    if control & 1 != 0 {
        a >= 0.5
    } else {
        // The 8-bit value's low bit; `a` is a colour, so it fits.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let byte = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        byte & 1 != 0
    }
}

/// The bytes A, B, C, D of an inputs word.
fn bytes(word: u32) -> [u32; 4] {
    [word >> 24, word >> 16 & 0xFF, word >> 8 & 0xFF, word & 0xFF]
}

/// Runs general stage `i` on `regs`.
fn general_stage(c: &Combiner, i: usize, regs: &mut [Vec4; 16]) {
    let own0 = c.control & 1 << 4 != 0;
    let own1 = c.control & 1 << 8 != 0;
    regs[1] = colour(c.constant0[if own0 { i } else { 0 }]);
    regs[2] = colour(c.constant1[if own1 { i } else { 0 }]);
    let mux = mux(c.control, regs);

    // Colour: from the registers as the stage found them.
    let out = c.colour_outputs[i];
    let [a, b, cc, d] = bytes(c.colour_inputs[i]).map(|byte| colour_input(byte, regs));
    let product = |x: [f32; 3], y: [f32; 3], dot: bool| -> [f32; 3] {
        if dot {
            [x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; 3]
        } else {
            [x[0] * y[0], x[1] * y[1], x[2] * y[2]]
        }
    };
    let ab = product(a, b, out & 1 << 13 != 0);
    let cd = product(cc, d, out & 1 << 12 != 0);
    let sum: [f32; 3] = if out & 1 << 14 != 0 {
        if mux { cd } else { ab }
    } else {
        [ab[0] + cd[0], ab[1] + cd[1], ab[2] + cd[2]]
    };
    let (ab, cd, sum) = (
        ab.map(|x| scale(out, x)),
        cd.map(|x| scale(out, x)),
        sum.map(|x| scale(out, x)),
    );

    // Alpha, likewise.
    let alpha_out = c.alpha_outputs[i];
    let [aa, ab_, ac, ad] = bytes(c.alpha_inputs[i]).map(|byte| alpha_input(byte, regs));
    let alpha_ab = aa * ab_;
    let alpha_cd = ac * ad;
    let alpha_sum = if alpha_out & 1 << 14 != 0 {
        if mux { alpha_cd } else { alpha_ab }
    } else {
        alpha_ab + alpha_cd
    };
    let (alpha_ab, alpha_cd, alpha_sum) = (
        scale(alpha_out, alpha_ab),
        scale(alpha_out, alpha_cd),
        scale(alpha_out, alpha_sum),
    );

    // Then the writes: rgb, then the blue-to-alpha copies, then alpha.
    let destination = |word: u32, shift: u32| (word >> shift & 0xF) as usize;
    for (shift, value) in [(0, cd), (4, ab), (8, sum)] {
        let dst = destination(out, shift);
        if dst != 0 {
            regs[dst][..3].copy_from_slice(&value);
        }
    }
    if out & 1 << 18 != 0 && destination(out, 0) != 0 {
        regs[destination(out, 0)][3] = cd[2];
    }
    if out & 1 << 19 != 0 && destination(out, 4) != 0 {
        regs[destination(out, 4)][3] = ab[2];
    }
    for (shift, value) in [(0, alpha_cd), (4, alpha_ab), (8, alpha_sum)] {
        let dst = destination(alpha_out, shift);
        if dst != 0 {
            regs[dst][3] = value;
        }
    }
}

/// A final-combiner input: register, alpha replicated with bit 4, clamped, inverted with bit 5.
fn final_input(byte: u32, regs: &[Vec4; 16]) -> [f32; 3] {
    let r = regs[(byte & 0xF) as usize];
    let v = if byte & 0x10 != 0 {
        [r[3]; 3]
    } else {
        [r[0], r[1], r[2]]
    };
    v.map(|x| {
        let x = x.clamp(0.0, 1.0);
        if byte & 0x20 != 0 { 1.0 - x } else { x }
    })
}

/// The registers a fragment starts with (spec 11.5).
#[must_use]
pub fn registers(f: &Fragment) -> [Vec4; 16] {
    let mut regs = [[0.0; 4]; 16];
    regs[3] = [f.fog[0], f.fog[1], f.fog[2], 1.0 - f.fog[3]];
    regs[4] = f.colour;
    regs[5] = f.specular;
    regs[8] = f.texture0;
    regs[9] = f.texture1;
    regs[10] = f.texture2;
    regs[11] = f.texture3;
    regs[12] = [0.0, 0.0, 0.0, f.texture0[3]];
    regs
}

/// Runs a combiner program over a fragment: the general stages, then the final combiner.
#[must_use]
pub fn evaluate(c: &Combiner, f: &Fragment) -> Vec4 {
    let mut regs = registers(f);
    let stages = (c.control & 0xFF).min(8) as usize;
    for i in 0..stages {
        general_stage(c, i, &mut regs);
    }
    regs[1] = colour(c.final_constants[0]);
    regs[2] = colour(c.final_constants[1]);
    let flags = c.final1 & 0xFF;
    let spare0 = regs[12];
    let specular = regs[5];
    let complement = |x: f32, on: bool| if on { 1.0 - x.clamp(0.0, 1.0) } else { x };
    for k in 0..3 {
        regs[14][k] = complement(specular[k], flags & 1 << 6 != 0)
            + complement(spare0[k], flags & 1 << 5 != 0);
    }
    let [e, f_, _, _] = bytes(c.final1).map(|byte| final_input(byte, &regs));
    regs[15] = [e[0] * f_[0], e[1] * f_[1], e[2] * f_[2], 0.0];
    let [a, b, cc, d] = bytes(c.final0).map(|byte| final_input(byte, &regs));
    let g = c.final1 >> 8 & 0xFF;
    let alpha = final_input(g | 0x10, &regs)[0];
    let rgb: [f32; 3] =
        std::array::from_fn(|k| (a[k] * b[k] + (1.0 - a[k]) * cc[k] + d[k]).clamp(0.0, 1.0));
    [rgb[0], rgb[1], rgb[2], alpha.clamp(0.0, 1.0)]
}

/// The shader's per-batch parameters, as bytes of its uniform block.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Params {
    pub combiner: Option<Combiner>,
    /// Alpha test function (GL enum) and reference.
    pub alpha_test: Option<(u32, u8)>,
    /// Alpha kill per stage.
    pub alpha_kill: [bool; 4],
    /// Mip bias per stage, in levels.
    pub bias: [f32; 4],
    /// Each stage's mode and the shape of its texture (flat without one).
    pub modes: [Mode; 4],
    pub shapes: [Shape; 4],
    /// What brings each stage's flat coordinates to 0..1: one over the size of a linear
    /// texture, which takes them in texels; else 1.
    pub scales: [[f32; 2]; 4],
    /// The dot mappings, input stages and bump environments.
    pub shader: Shader,
    /// Each stage's colour key mode (0 none, 1 clears a matching texel's alpha, 2 the whole
    /// texel, 3 drops the fragment), key, and the texel bits compared.
    pub colour_keys: [(u32, u32, u32); 4],
    /// The window clip: `Some((exclusive, rectangles))` when the shader tests it.
    pub window: Option<(bool, [[u32; 4]; 8])>,
    /// Whether depth outside the range drops the fragment.
    pub drop_outside: bool,
    /// With w-buffering, the depth buffer's largest depth.
    pub w_buffer: Option<f32>,
    /// The depth range, in 0..1 of the buffer.
    pub depth_range: [f32; 2],
}

/// The size of the uniform block.
pub const PARAMS_SIZE: usize = 800;

impl Params {
    /// The uniform block's bytes (the layout of `Params` in `shader.wgsl`).
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let c = self.combiner.unwrap_or_default();
        let (func, reference) = self.alpha_test.unwrap_or((0x207, 0));
        let mut words: Vec<u32> = vec![
            u32::from(self.combiner.is_some()),
            (c.control & 0xFF).min(8),
            c.control,
            c.final1,
            c.final0,
            if self.alpha_test.is_some() {
                func
            } else {
                0x207
            },
            u32::from(reference),
            (0..4).fold(0, |bits, s| bits | u32::from(self.alpha_kill[s]) << s),
            self.bias[0].to_bits(),
            self.bias[1].to_bits(),
            self.bias[2].to_bits(),
            self.bias[3].to_bits(),
        ];
        for array in [
            c.colour_inputs,
            c.colour_outputs,
            c.alpha_inputs,
            c.alpha_outputs,
        ] {
            words.extend(array);
        }
        for value in c
            .constant0
            .iter()
            .chain(&c.constant1)
            .chain(&c.final_constants)
        {
            words.extend(colour(*value).map(f32::to_bits));
        }
        // The texture stages: modes and shapes a byte each, the dot mappings, the input stages;
        // the coordinate scales; the bump matrices; the bump scales, then offsets.
        let stage_words = (0..4).fold(0, |bits, s| {
            let shape = match self.shapes[s] {
                Shape::Flat => 0,
                Shape::Volume => 1 << 5,
                Shape::Cube => 1 << 6,
            };
            bits | (self.modes[s] as u32 | shape) << (8 * s)
        });
        let inputs = u32::try_from(self.shader.input(2) | self.shader.input(3) << 4).unwrap_or(0);
        words.extend([stage_words, self.shader.dot_mapping, inputs, 0]);
        let floats = self
            .scales
            .as_flattened()
            .iter()
            .chain(self.shader.bump_matrix.as_flattened())
            .chain(&self.shader.bump_scale)
            .chain(&self.shader.bump_offset);
        words.extend(floats.map(|f| f.to_bits()));
        // The pixel words: the colour keys, their masks; the key modes, window clip, depth flags
        // and largest depth; the depth range; then the window rectangles.
        words.extend(self.colour_keys.map(|(_, key, _)| key));
        words.extend(self.colour_keys.map(|(_, _, mask)| mask));
        words.extend([
            (0..4).fold(0, |bits, s| bits | (self.colour_keys[s].0 & 3) << (2 * s)),
            self.window
                .map_or(0, |(exclusive, _)| 1 + u32::from(exclusive)),
            u32::from(self.drop_outside) | u32::from(self.w_buffer.is_some()) << 1,
            self.w_buffer.unwrap_or(1.0).to_bits(),
            self.depth_range[0].to_bits(),
            self.depth_range[1].to_bits(),
            0,
            0,
        ]);
        let rects = self.window.map_or([[0; 4]; 8], |(_, rects)| rects);
        words.extend(rects.iter().flatten());
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        debug_assert_eq!(bytes.len(), PARAMS_SIZE);
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragment() -> Fragment {
        Fragment {
            texture0: [0.5, 0.25, 1.0, 0.75],
            texture1: [1.0; 4],
            texture2: [0.0, 0.0, 0.0, 1.0],
            texture3: [0.0, 0.0, 0.0, 1.0],
            colour: [0.2, 0.4, 0.6, 0.8],
            fog: [0.0; 4],
            specular: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn the_specular_colour_is_register_5_and_in_the_final_sum() {
        let f = Fragment {
            specular: [0.25, 0.5, 0.0, 1.0],
            ..fragment()
        };
        // Final A = register 5 times B = 1.
        assert_eq!(evaluate(&final_only(5, 0), &f)[..3], [0.25, 0.5, 0.0]);
        // Register 14 = specular + spare 0 (zero here), as D.
        let sum = Combiner {
            final0: 14,
            ..Combiner::default()
        };
        assert_eq!(evaluate(&sum, &f)[..3], [0.25, 0.5, 0.0]);
    }

    /// A program whose final combiner outputs `A * B` with A = register `a`, B = 1 (zero
    /// inverted), C = D = 0, and alpha from register `g`.
    fn final_only(a: u32, g: u32) -> Combiner {
        Combiner {
            final0: a << 24 | 0x20 << 16,
            final1: g << 8,
            ..Combiner::default()
        }
    }

    #[test]
    fn the_input_mappings() {
        let cases = [
            (0, -0.5, 0.0),
            (0, 0.5, 0.5),
            (1, 0.25, 0.75),
            (1, 2.0, 0.0),
            (2, 0.75, 0.5),
            (3, 0.75, -0.5),
            (4, 0.25, -0.25),
            (5, 0.25, 0.25),
            (6, -0.5, -0.5),
            (7, 0.5, -0.5),
        ];
        for (mapping, x, y) in cases {
            assert!(
                (map(mapping, x) - y).abs() < 1e-6,
                "mapping {mapping} of {x}"
            );
        }
    }

    #[test]
    fn the_scales_and_biases_then_the_clamp() {
        let cases = [
            (0, 0.3, 0.3),
            (1, 0.3, -0.2),
            (2, 0.3, 0.6),
            (3, 0.3, -0.4),
            (4, 0.3, 1.0),
            (6, 0.3, 0.15),
            (2, -0.8, -1.0),
        ];
        for (s, x, y) in cases {
            assert!((scale(s << 15, x) - y).abs() < 1e-6, "scale {s} of {x}");
        }
    }

    #[test]
    fn the_final_combiner_passes_a_register_through() {
        let out = evaluate(&final_only(8, 4), &fragment());
        assert_eq!(&out[..3], &[0.5, 0.25, 1.0]);
        assert!(
            (out[3] - 0.8).abs() < 1e-6,
            "alpha from the diffuse register"
        );
    }

    #[test]
    fn the_final_combiner_lerps_and_adds() {
        // A = diffuse alpha (0.8), B = texture 0, C = diffuse, D = zero.
        let c = Combiner {
            final0: (0x10 | 4) << 24 | 8 << 16 | 4 << 8,
            ..Combiner::default()
        };
        let f = fragment();
        let out = evaluate(&c, &f);
        for (k, got) in out.iter().take(3).enumerate() {
            let want = 0.8 * f.texture0[k] + 0.2 * f.colour[k];
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn a_general_stage_modulates_into_spare0() {
        // Stage 0: spare0.rgb = texture0 * diffuse (AB to register 12); final outputs spare0.
        let c = Combiner {
            colour_inputs: [8 << 24 | 4 << 16, 0, 0, 0, 0, 0, 0, 0],
            colour_outputs: [12 << 4, 0, 0, 0, 0, 0, 0, 0],
            control: 1,
            ..final_only(12, 0)
        };
        let f = fragment();
        let out = evaluate(&c, &f);
        for (k, got) in out.iter().take(3).enumerate() {
            assert!((got - f.texture0[k] * f.colour[k]).abs() < 1e-6);
        }
    }

    #[test]
    fn dot_products_mux_and_blue_to_alpha() {
        // AB = dot(texture0, texture0) into register 13 with blue copied to its alpha.
        let c = Combiner {
            colour_inputs: [8 << 24 | 8 << 16, 0, 0, 0, 0, 0, 0, 0],
            colour_outputs: [13 << 4 | 1 << 13 | 1 << 19, 0, 0, 0, 0, 0, 0, 0],
            control: 1,
            ..final_only(13, 13)
        };
        let f = fragment();
        let dot = (0.5f32 * 0.5 + 0.25 * 0.25 + 1.0).min(1.0);
        let out = evaluate(&c, &f);
        assert!((out[0] - dot).abs() < 1e-6 && (out[3] - dot).abs() < 1e-6);
        // Mux: spare0 alpha is texture0's alpha (0.75 >= 0.5 with control bit 0): CD is chosen.
        let m = Combiner {
            colour_inputs: [4 << 24 | 0x20 << 16 | 8 << 8 | 0x20, 0, 0, 0, 0, 0, 0, 0],
            colour_outputs: [13 << 8 | 1 << 14, 0, 0, 0, 0, 0, 0, 0],
            control: 1,
            ..final_only(13, 0)
        };
        assert_eq!(&evaluate(&m, &f)[..3], &f.texture0[..3]);
    }

    #[test]
    fn outputs_are_written_after_all_inputs_are_read() {
        // CD = texture0 * 1 into register 8; AB = texture0 * 1 into register 13: AB must see the
        // old texture0 even though CD writes register 8 first. Both are then scaled by 2.
        let c = Combiner {
            colour_inputs: [8 << 24 | 0x20 << 16 | 8 << 8 | 0x20, 0, 0, 0, 0, 0, 0, 0],
            colour_outputs: [8 | 13 << 4 | 2 << 15, 0, 0, 0, 0, 0, 0, 0],
            control: 1,
            ..final_only(13, 0)
        };
        let out = evaluate(&c, &fragment());
        assert_eq!(&out[..3], &[1.0, 0.5, 1.0]);
    }

    #[test]
    fn constants_are_stage_zeros_unless_the_control_gives_each_its_own() {
        let mut c = Combiner {
            colour_inputs: [0, 1 << 24 | 0x20 << 16, 0, 0, 0, 0, 0, 0],
            colour_outputs: [0, 13 << 4, 0, 0, 0, 0, 0, 0],
            constant0: [0xFF00_00FF, 0xFFFF_0000, 0, 0, 0, 0, 0, 0],
            control: 2,
            ..final_only(13, 0)
        };
        assert_eq!(
            &evaluate(&c, &fragment())[..3],
            &[0.0, 0.0, 1.0],
            "stage 0's constant"
        );
        c.control |= 1 << 4;
        assert_eq!(
            &evaluate(&c, &fragment())[..3],
            &[1.0, 0.0, 0.0],
            "stage 1's own constant"
        );
    }

    #[test]
    fn the_uniform_block_has_its_size() {
        assert_eq!(Params::default().bytes().len(), PARAMS_SIZE);
    }
}
