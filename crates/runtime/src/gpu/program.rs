//! Vertex programs (spec 10.10): the interpreter that runs a program over one vertex.
//!
//! The bit positions follow the spec, which marks them as needing independent confirmation.

/// Instruction slots, constants, inputs, temporaries and outputs.
pub const SLOTS: usize = 136;
pub const CONSTANTS: usize = 192;
const REGISTERS: usize = 16;
/// The temporary that is the output position under another name: a program reads its position
/// back through it (for the divide and the viewport step) and may write it.
const POSITION_TEMP: usize = 12;

/// Output numbers.
pub const POSITION: usize = 0;
pub const DIFFUSE: usize = 3;
pub const FOG: usize = 5;
pub const POINT_SIZE: usize = 6;
pub const TEXTURE0: usize = 9;

use serde::{Deserialize, Serialize};

use std::sync::Arc;

type Vec4 = [f32; 4];

/// The program constant a fixed-function transform register shares its storage with, as
/// `(row, component)`, or `None` for any other method. The engine keeps the transform state and
/// the program constants in one store, so a program reads the viewport the title set (row 58 the
/// scale, 59 the offset), the matrices, the eye and the fog plane. A block of matrices takes four
/// rows each, eight rows apart; each row is four floats.
#[must_use]
pub fn transform_constant(method: u32) -> Option<(usize, usize)> {
    // (first method, end, first row) of the blocks of 4x4 matrices: model-view, inverse
    // model-view, texture matrices, texture-generation planes.
    const MATRICES: [(u32, u32, usize); 4] = [
        (0x0480, 0x0580, 0x08),
        (0x0580, 0x0680, 0x0C),
        (0x06C0, 0x07C0, 0x44),
        (0x0840, 0x0940, 0x40),
    ];
    // (first method, end, first row) of the runs of consecutive rows: the block at 0x0440, the
    // composite matrix, the fog plane, the viewport offset, the eye position, the viewport
    // scale.
    const ROWS: [(u32, u32, usize); 6] = [
        (0x0440, 0x0480, 0x04),
        (0x0680, 0x06C0, 0x00),
        (0x09D0, 0x09E0, 0x39),
        (0x0A20, 0x0A30, 0x3B),
        (0x0A50, 0x0A60, 0x38),
        (0x0AF0, 0x0B00, 0x3A),
    ];
    for (first, end, row) in MATRICES {
        if (first..end).contains(&method) {
            let word = ((method - first) / 4) as usize;
            let (matrix, within) = (word / 16, word % 16);
            return Some((row + 8 * matrix + within / 4, within % 4));
        }
    }
    ROWS.iter()
        .find(|(first, end, _)| (*first..*end).contains(&method))
        .map(|&(first, _, row)| {
            let word = ((method - first) / 4) as usize;
            (row + word / 4, word % 4)
        })
}

/// `field(word, shift, bits)`.
fn field(insn: &[u32; 4], word: usize, shift: u32, bits: u32) -> u32 {
    (insn[word] >> shift) & ((1 << bits) - 1)
}

/// The state a program runs with: instructions, constants, and its start slot. The constants
/// are shared, copy on write, with the draws that run their vertices later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub instructions: Vec<[u32; 4]>,
    pub constants: Arc<Vec<Vec4>>,
    pub start: usize,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            instructions: vec![[0; 4]; SLOTS],
            constants: Arc::new(vec![[0.0; 4]; CONSTANTS]),
            start: 0,
        }
    }
}

impl Program {
    /// The program from its start slot, decoded: what each vertex runs.
    #[must_use]
    pub fn decode(&self) -> Vec<Op> {
        decode(&self.instructions, self.start)
    }
}

/// An operand's source, decoded: where it reads (1 a temporary, 2 the input, 3 the constant,
/// anything else zero), the component each of x, y, z and w takes, and the negation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Source {
    mux: u8,
    register: u8,
    swizzle: [u8; 4],
    negate: bool,
}

impl Source {
    /// A source that reads zero, for an operand no running unit uses.
    const ZERO: Self = Self {
        mux: 0,
        register: 0,
        swizzle: [0, 1, 2, 3],
        negate: false,
    };
}

/// One instruction, decoded once so that each vertex only runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Op {
    /// The vector operation (13 is ARL), and the scalar one.
    mac: u8,
    ilu: u8,
    /// Operands A, B and C.
    sources: [Source; 3],
    input: u8,
    /// The constant it reads, before the address register's offset, and whether that applies.
    constant: i32,
    relative: bool,
    /// The temporary and mask each unit's result goes to, when it writes one.
    mac_write: Option<(u8, u8)>,
    ilu_write: Option<(u8, u8)>,
    /// The output register and mask, when it writes one, and whether it takes the scalar
    /// result rather than the vector one.
    output: Option<(u8, u8)>,
    output_from_ilu: bool,
}

/// Whether vector operation `op` gives a result (ARL, 13, writes the address register only).
fn mac_runs(op: u32) -> bool {
    matches!(op, 1..=12)
}

/// Whether scalar operation `op` gives a result.
fn ilu_runs(op: u32) -> bool {
    matches!(op, 1..=7)
}

/// Decodes `instructions` from `start` up to the one marked final, or their end. Instructions
/// that do nothing are left out.
#[must_use]
pub fn decode(instructions: &[[u32; 4]], start: usize) -> Vec<Op> {
    let mut ops = Vec::new();
    for insn in instructions.iter().skip(start) {
        let mac_op = field(insn, 1, 21, 4);
        let ilu_op = field(insn, 1, 25, 3);
        if mac_runs(mac_op) || ilu_runs(ilu_op) || mac_op == 13 {
            ops.push(decode_one(insn, mac_op, ilu_op));
        }
        if insn[3] & 1 != 0 {
            break;
        }
    }
    ops
}

#[allow(clippy::cast_possible_truncation)]
fn decode_one(insn: &[u32; 4], mac_op: u32, ilu_op: u32) -> Op {
    let source = |negate: u32, swizzle: u32, mux: u32, register: u32, scalar: bool| {
        // Four 2-bit selectors, x in the top pair; the scalar ops read C with x replicated.
        let select = |i: u32| ((swizzle >> (6 - 2 * i)) & 3) as u8;
        Source {
            mux: mux as u8,
            register: register as u8,
            swizzle: if scalar {
                [select(0); 4]
            } else {
                [select(0), select(1), select(2), select(3)]
            },
            negate: negate != 0,
        }
    };
    let a = source(
        field(insn, 1, 8, 1),
        field(insn, 1, 0, 8),
        field(insn, 2, 26, 2),
        field(insn, 2, 28, 4),
        false,
    );
    let b = source(
        field(insn, 2, 25, 1),
        field(insn, 2, 17, 8),
        field(insn, 2, 11, 2),
        field(insn, 2, 13, 4),
        false,
    );
    let c = source(
        field(insn, 2, 10, 1),
        field(insn, 2, 2, 8),
        field(insn, 3, 28, 2),
        field(insn, 2, 0, 2) << 2 | field(insn, 3, 30, 2),
        matches!(ilu_op, 2..=6),
    );
    // Which operands the running units read; the others need not be fetched.
    let (uses_b, uses_c) = match mac_op {
        2 | 5..=12 => (true, false),
        3 => (false, true),
        4 => (true, true),
        _ => (false, false),
    };
    let (mac, ilu) = (mac_runs(mac_op), ilu_runs(ilu_op));
    let uses_c = uses_c || ilu;
    let uses_a = mac || mac_op == 13;
    let temp = field(insn, 3, 20, 4) as u8;
    let both = mac && ilu;
    let mac_mask = field(insn, 3, 24, 4) as u8;
    let ilu_mask = field(insn, 3, 16, 4) as u8;
    let out_mask = field(insn, 3, 12, 4) as u8;
    let out_from_ilu = field(insn, 3, 2, 1) != 0;
    let out_unit_runs = if out_from_ilu { ilu } else { mac };
    Op {
        mac: mac_op as u8,
        ilu: ilu_op as u8,
        sources: [
            if uses_a { a } else { Source::ZERO },
            if uses_b { b } else { Source::ZERO },
            if uses_c { c } else { Source::ZERO },
        ],
        input: field(insn, 1, 9, 4) as u8,
        constant: field(insn, 1, 13, 8).cast_signed(),
        relative: insn[3] & 2 != 0,
        // With both units running, the scalar result goes to R1, and the vector one is not
        // written when its temporary is R1 too.
        mac_write: (mac && !(both && temp == 1) && mac_mask != 0).then_some((temp, mac_mask)),
        ilu_write: (ilu && ilu_mask != 0).then_some((if both { 1 } else { temp }, ilu_mask)),
        output: (out_mask != 0 && field(insn, 3, 11, 1) != 0 && out_unit_runs)
            .then_some(((field(insn, 3, 3, 8) & 0xF) as u8, out_mask)),
        output_from_ilu: out_from_ilu,
    }
}

/// The registers of one run.
struct Machine<'a> {
    constants: &'a [Vec4],
    inputs: &'a [Vec4; 16],
    temps: [Vec4; REGISTERS],
    address: i32,
    outputs: [Vec4; 16],
}

impl Machine<'_> {
    fn read(&self, source: Source, constant: Option<usize>, input: usize) -> Vec4 {
        let value = match source.mux {
            1 if usize::from(source.register) == POSITION_TEMP => self.outputs[POSITION],
            1 => self.temps[usize::from(source.register)],
            2 => self.inputs[input],
            3 => constant
                .and_then(|c| self.constants.get(c))
                .copied()
                .unwrap_or([0.0; 4]),
            _ => return [0.0; 4],
        };
        let s = source.swizzle.map(usize::from);
        let out = [value[s[0]], value[s[1]], value[s[2]], value[s[3]]];
        if source.negate { out.map(|x| -x) } else { out }
    }

    /// Temporary `index` for writing; R12 is the output position.
    fn temp_mut(&mut self, index: usize) -> &mut Vec4 {
        if index == POSITION_TEMP {
            &mut self.outputs[POSITION]
        } else {
            &mut self.temps[index]
        }
    }
}

/// Writes `value` into `target` under a 4-bit mask whose bit 3 is x.
fn write_masked(target: &mut Vec4, value: Vec4, mask: u8) {
    for (i, slot) in target.iter_mut().enumerate() {
        if mask & (8 >> i) != 0 {
            *slot = value[i];
        }
    }
}

fn dot3(a: Vec4, b: Vec4) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A product of the multiplier: 0 when either factor is 0, even against infinity or NaN, so a
/// lighting term scaled by a zero weight stays 0.
fn times(x: f32, y: f32) -> f32 {
    if x == 0.0 || y == 0.0 { 0.0 } else { x * y }
}

/// The vector (MAC) unit.
fn mac(op: u32, a: Vec4, b: Vec4, c: Vec4) -> Option<Vec4> {
    let each = |f: fn(f32, f32) -> f32, x: Vec4, y: Vec4| {
        [f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2]), f(x[3], y[3])]
    };
    Some(match op {
        1 => a,
        2 => each(times, a, b),
        3 => each(|x, y| x + y, a, c),
        4 => [
            times(a[0], b[0]) + c[0],
            times(a[1], b[1]) + c[1],
            times(a[2], b[2]) + c[2],
            times(a[3], b[3]) + c[3],
        ],
        5 => [dot3(a, b); 4],
        6 => [dot3(a, b) + b[3]; 4],
        7 => [dot3(a, b) + a[3] * b[3]; 4],
        8 => [1.0, a[1] * b[1], a[2], b[3]],
        9 => each(f32::min, a, b),
        10 => each(f32::max, a, b),
        11 => each(|x, y| f32::from(u8::from(x < y)), a, b),
        12 => each(|x, y| f32::from(u8::from(x >= y)), a, b),
        _ => return None,
    })
}

/// The scalar (ILU) unit, on operand C.
fn ilu(op: u32, c: Vec4) -> Option<Vec4> {
    let x = c[0];
    Some(match op {
        1 => c,
        2 => [1.0 / x; 4],
        3 => {
            let r = 1.0 / x;
            let magnitude = r.abs().clamp(5.421_01e-20, 1.844_674_4e19);
            [magnitude.copysign(r); 4]
        }
        4 => [1.0 / x.abs().sqrt(); 4],
        5 => {
            let floor = x.floor();
            [floor.exp2(), x - floor, x.exp2(), 1.0]
        }
        6 => {
            if x == 0.0 {
                [f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 1.0]
            } else {
                let e = x.abs().log2().floor();
                [e, x.abs() / e.exp2(), x.abs().log2(), 1.0]
            }
        }
        7 => {
            let specular = if c[0] > 0.0 {
                c[1].max(0.0).powf(c[3].clamp(-128.0, 128.0))
            } else {
                0.0
            };
            [1.0, c[0].max(0.0), specular, 1.0]
        }
        _ => return None,
    })
}

/// Runs `program` from its start slot for one vertex with `inputs`; returns the 16 outputs, which
/// start as `(0, 0, 0, 1)`. Decodes the program first: a caller running many vertices decodes
/// once and calls [`run_decoded`].
#[must_use]
pub fn run(program: &Program, inputs: &[Vec4; 16]) -> [Vec4; 16] {
    run_decoded(&program.decode(), &program.constants, inputs)
}

/// Runs a program [`decode`] made for one vertex with `inputs` and the program's `constants`.
#[must_use]
pub fn run_decoded(ops: &[Op], constants: &[Vec4], inputs: &[Vec4; 16]) -> [Vec4; 16] {
    let mut m = Machine {
        constants,
        inputs,
        temps: [[0.0; 4]; REGISTERS],
        address: 0,
        outputs: [[0.0, 0.0, 0.0, 1.0]; 16],
    };
    for op in ops {
        step(&mut m, op);
    }
    m.outputs
}

/// Runs one instruction: both units read before either writes.
fn step(m: &mut Machine<'_>, op: &Op) {
    let mut constant = op.constant;
    if op.relative {
        constant += m.address;
    }
    let constant = usize::try_from(constant).ok().filter(|&c| c < CONSTANTS);
    let input = usize::from(op.input);
    let [a, b, c] = op.sources;
    let (va, vb, vc) = (
        m.read(a, constant, input),
        m.read(b, constant, input),
        m.read(c, constant, input),
    );
    let mac_op = u32::from(op.mac);
    let mac_result = if mac_op == 13 {
        // ARL: no register write. The floor is nudged up by 0.001, so a value just short of an
        // integer (an index byte scaled by 1/255 and back) gives that integer.
        m.address = (va[0] + 0.001).floor().clamp(-512.0, 511.0) as i32;
        None
    } else {
        mac(mac_op, va, vb, vc)
    };
    let ilu_result = ilu(u32::from(op.ilu), vc);
    if let (Some(value), Some((temp, mask))) = (mac_result, op.mac_write) {
        write_masked(m.temp_mut(usize::from(temp)), value, mask);
    }
    if let (Some(value), Some((temp, mask))) = (ilu_result, op.ilu_write) {
        write_masked(m.temp_mut(usize::from(temp)), value, mask);
    }
    let out_value = if op.output_from_ilu {
        ilu_result
    } else {
        mac_result
    };
    if let (Some((index, mask)), Some(value)) = (op.output, out_value) {
        let index = usize::from(index);
        if index == FOG {
            // Scalar: x takes the first component the mask selects.
            if let Some(first) = (0..4).find(|i| mask & (8 >> i) != 0) {
                m.outputs[FOG][0] = value[first];
            }
        } else {
            write_masked(&mut m.outputs[index], value, mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identity swizzle `xyzw`.
    const XYZW: u32 = 0b00_01_10_11;

    /// Builds an instruction from its fields.
    #[derive(Default)]
    struct Insn {
        ilu: u32,
        mac: u32,
        constant: u32,
        input: u32,
        a: (u32, u32),
        b: (u32, u32),
        c: (u32, u32),
        mac_mask: u32,
        temp: u32,
        ilu_mask: u32,
        out_mask: u32,
        out: u32,
        out_ilu: bool,
        last: bool,
    }

    impl Insn {
        /// Operands are (mux, register), all with the identity swizzle.
        fn words(&self) -> [u32; 4] {
            let mut w = [0u32; 4];
            w[1] = self.ilu << 25 | self.mac << 21 | self.constant << 13 | self.input << 9 | XYZW;
            w[2] = self.a.0 << 26
                | self.a.1 << 28
                | XYZW << 17
                | self.b.0 << 11
                | self.b.1 << 13
                | XYZW << 2;
            w[2] |= self.c.1 >> 2;
            w[3] = self.c.0 << 28
                | (self.c.1 & 3) << 30
                | self.mac_mask << 24
                | self.temp << 20
                | self.ilu_mask << 16;
            w[3] |= self.out_mask << 12
                | 1 << 11
                | self.out << 3
                | u32::from(self.out_ilu) << 2
                | u32::from(self.last);
            w
        }
    }

    fn inputs() -> [Vec4; 16] {
        let mut inputs = [[0.0; 4]; 16];
        inputs[0] = [1.0, 2.0, 3.0, 1.0];
        inputs
    }

    #[test]
    fn a_dp4_transform_writes_the_position() {
        let mut program = Program::default();
        // c0 = (2, 0, 0, 0): o0.x = dp4(v0, c0); then MOV o3 = v0, last.
        Arc::make_mut(&mut program.constants)[0] = [2.0, 0.0, 0.0, 0.5];
        program.instructions[0] = Insn {
            mac: 7,
            a: (2, 0),
            b: (3, 0),
            out_mask: 8,
            out: 0,
            ..Insn::default()
        }
        .words();
        program.instructions[1] = Insn {
            mac: 1,
            a: (2, 0),
            out_mask: 15,
            out: 3,
            last: true,
            ..Insn::default()
        }
        .words();
        program.instructions[2] = Insn {
            mac: 1,
            a: (2, 0),
            out_mask: 15,
            out: 4,
            ..Insn::default()
        }
        .words();
        let out = run(&program, &inputs());
        assert_eq!(out[POSITION], [2.5, 0.0, 0.0, 1.0]);
        assert_eq!(out[DIFFUSE], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(
            out[4],
            [0.0, 0.0, 0.0, 1.0],
            "the program ended at the final bit"
        );
    }

    #[test]
    fn mad_through_a_temporary() {
        let mut program = Program::default();
        Arc::make_mut(&mut program.constants)[5] = [10.0, 10.0, 10.0, 10.0];
        // R2 = v0 * v0 + c5; o0 = R2.
        program.instructions[0] = Insn {
            mac: 4,
            constant: 5,
            a: (2, 0),
            b: (2, 0),
            c: (3, 0),
            mac_mask: 15,
            temp: 2,
            ..Insn::default()
        }
        .words();
        program.instructions[1] = Insn {
            mac: 1,
            a: (1, 2),
            out_mask: 15,
            out: 0,
            last: true,
            ..Insn::default()
        }
        .words();
        assert_eq!(run(&program, &inputs())[0], [11.0, 14.0, 19.0, 11.0]);
    }

    #[test]
    fn both_units_running_send_the_scalar_result_to_r1() {
        let mut program = Program::default();
        // MAC: R1 = v0 (suppressed: both run and the temporary is R1); ILU: RCP(v0.x) to R1.
        program.instructions[0] = Insn {
            mac: 1,
            ilu: 2,
            a: (2, 0),
            c: (2, 0),
            mac_mask: 15,
            ilu_mask: 15,
            temp: 1,
            ..Insn::default()
        }
        .words();
        program.instructions[1] = Insn {
            mac: 1,
            a: (1, 1),
            out_mask: 15,
            out: 0,
            last: true,
            ..Insn::default()
        }
        .words();
        assert_eq!(run(&program, &inputs())[0], [1.0; 4]);
    }

    #[test]
    fn the_fog_output_is_scalar() {
        let mut program = Program::default();
        program.instructions[0] = Insn {
            mac: 1,
            a: (2, 0),
            out_mask: 0b0100,
            out: 5,
            last: true,
            ..Insn::default()
        }
        .words();
        assert_eq!(run(&program, &inputs())[FOG][0], 2.0);
    }

    #[test]
    fn arl_offsets_constant_reads() {
        let mut program = Program::default();
        Arc::make_mut(&mut program.constants)[3] = [7.0; 4];
        let mut inputs = inputs();
        inputs[1] = [3.0, 0.0, 0.0, 0.0];
        program.instructions[0] = Insn {
            mac: 13,
            input: 1,
            a: (2, 0),
            ..Insn::default()
        }
        .words();
        let mut read = Insn {
            mac: 1,
            a: (3, 0),
            out_mask: 15,
            out: 0,
            last: true,
            ..Insn::default()
        }
        .words();
        read[3] |= 2; // relative to A0.x
        program.instructions[1] = read;
        assert_eq!(run(&program, &inputs)[0], [7.0; 4]);
    }

    #[test]
    fn r12_is_the_output_position() {
        let mut program = Program::default();
        Arc::make_mut(&mut program.constants)[0] = [2.0, 2.0, 2.0, 2.0];
        // MOV oPos, v0; MUL R12.xyz, R12, c0 (the viewport step reads oPos back), last.
        program.instructions[0] = Insn {
            mac: 1,
            a: (2, 0),
            out_mask: 15,
            out: 0,
            ..Insn::default()
        }
        .words();
        program.instructions[1] = Insn {
            mac: 2,
            a: (1, 12),
            b: (3, 0),
            mac_mask: 0b1110,
            temp: 12,
            last: true,
            ..Insn::default()
        }
        .words();
        assert_eq!(run(&program, &inputs())[POSITION], [2.0, 4.0, 6.0, 1.0]);
    }

    #[test]
    fn a_product_with_zero_is_zero_even_against_infinity() {
        let a = [0.0, f32::INFINITY, 2.0, f32::NAN];
        let b = [f32::INFINITY, 0.0, 3.0, 0.0];
        assert_eq!(mac(2, a, b, b), Some([0.0, 0.0, 6.0, 0.0]));
        assert_eq!(mac(4, a, b, [1.0; 4]), Some([1.0, 1.0, 7.0, 1.0]));
    }

    #[test]
    fn arl_rounds_a_value_just_short_of_an_integer_up() {
        let mut program = Program::default();
        Arc::make_mut(&mut program.constants)[3] = [7.0; 4];
        let mut inputs = inputs();
        inputs[1] = [2.9995, 0.0, 0.0, 0.0];
        program.instructions[0] = Insn {
            mac: 13,
            input: 1,
            a: (2, 0),
            ..Insn::default()
        }
        .words();
        let mut read = Insn {
            mac: 1,
            a: (3, 0),
            out_mask: 15,
            out: 0,
            last: true,
            ..Insn::default()
        }
        .words();
        read[3] |= 2;
        program.instructions[1] = read;
        assert_eq!(run(&program, &inputs)[0], [7.0; 4]);
    }

    #[test]
    fn an_output_from_an_idle_unit_is_not_written() {
        let mut program = Program::default();
        // MOV oPos, v0; then a MAC MOV whose output names the idle ILU.
        program.instructions[0] = Insn {
            mac: 1,
            a: (2, 0),
            out_mask: 15,
            out: 0,
            ..Insn::default()
        }
        .words();
        program.instructions[1] = Insn {
            mac: 1,
            a: (3, 0),
            out_mask: 15,
            out: 0,
            out_ilu: true,
            last: true,
            ..Insn::default()
        }
        .words();
        assert_eq!(run(&program, &inputs())[0], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn transform_registers_share_the_constants() {
        // The composite matrix is rows 0..3; viewport scale and offset rows 58 and 59.
        assert_eq!(transform_constant(0x0680), Some((0, 0)));
        assert_eq!(transform_constant(0x06BC), Some((3, 3)));
        assert_eq!(transform_constant(0x0AF4), Some((58, 1)));
        assert_eq!(transform_constant(0x0A2C), Some((59, 3)));
        // The second model-view matrix's second row; the inverse of the first.
        assert_eq!(transform_constant(0x04C0 + 16 + 8), Some((0x11, 2)));
        assert_eq!(transform_constant(0x0580), Some((0x0C, 0)));
        // Stage 3's texture matrix, stage 1's texgen R plane, the eye, the fog plane.
        assert_eq!(transform_constant(0x0780), Some((0x44 + 24, 0)));
        assert_eq!(transform_constant(0x0880 + 32), Some((0x48 + 2, 0)));
        assert_eq!(transform_constant(0x0A50), Some((0x38, 0)));
        assert_eq!(transform_constant(0x09DC), Some((0x39, 3)));
        assert_eq!(transform_constant(0x0B00), None);
        assert_eq!(transform_constant(0x07C0), None);
    }

    #[test]
    fn the_scalar_ops() {
        assert_eq!(ilu(2, [4.0; 4]), Some([0.25; 4]));
        assert_eq!(ilu(2, [0.0; 4]).unwrap()[0], f32::INFINITY);
        assert_eq!(ilu(4, [-4.0; 4]), Some([0.5; 4]));
        assert_eq!(
            ilu(5, [2.5, 0.0, 0.0, 0.0]).unwrap(),
            [4.0, 0.5, 2.5f32.exp2(), 1.0]
        );
        assert_eq!(ilu(6, [8.0; 4]).unwrap(), [3.0, 1.0, 3.0, 1.0]);
        assert_eq!(ilu(6, [0.0; 4]).unwrap()[1], 1.0);
        assert_eq!(ilu(7, [0.5, 2.0, 0.0, 2.0]).unwrap(), [1.0, 0.5, 4.0, 1.0]);
        assert_eq!(ilu(7, [-0.5, 2.0, 0.0, 2.0]).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(ilu(3, [1e30; 4]).unwrap()[0], 5.421_01e-20);
    }

    #[test]
    fn the_vector_ops() {
        let (a, b) = ([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(mac(5, a, b, b), Some([12.0; 4]));
        assert_eq!(mac(6, a, b, b), Some([14.0; 4]));
        assert_eq!(mac(8, a, b, b), Some([1.0, 4.0, 3.0, 2.0]));
        assert_eq!(mac(11, a, b, b), Some([1.0, 0.0, 0.0, 0.0]));
        assert_eq!(mac(12, a, b, b), Some([0.0, 1.0, 1.0, 1.0]));
    }
}
