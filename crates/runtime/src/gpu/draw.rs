//! Draw decoding (spec 10.8): a frame's commands, applied in order, as primitives for the
//! renderer.

use serde::{Deserialize, Serialize};

use super::Command;
use super::fifo::KELVIN;
use super::fixed::{self, Pipeline};
use super::pixel::{self, Pixel, PixelRegs};
use super::program::{self, Op, Program};
use super::texture::{
    Mode, SHADER_STAGE_PROGRAM, STAGE_BASE, STAGE_STRIDE, Shader, StageRegs, Texture,
};
use std::sync::Arc;

// Kelvin methods [F].
const SET_BEGIN_END: u32 = 0x17FC;
const INLINE_ARRAY: u32 = 0x1818;
const ARRAY_FORMAT: u32 = 0x1760;
const MATRIX: u32 = 0x0680;
const VIEWPORT_OFFSET: u32 = 0x0A20;
const DEPTH_ENABLE: u32 = 0x030C;
const DEPTH_FUNC: u32 = 0x0354;
const DEPTH_WRITE: u32 = 0x035C;
const BLEND_ENABLE: u32 = 0x0304;
const ALPHA_ENABLE: u32 = 0x0300;
const ALPHA_FUNC: u32 = 0x033C;
const ALPHA_REF: u32 = 0x0340;
const FOG_MODE: u32 = 0x029C;
const FOG_GEN_MODE: u32 = 0x02A0;
const FOG_ENABLE: u32 = 0x02A4;
const FOG_COLOUR: u32 = 0x02A8;
const FOG_BASE: u32 = 0x09C0;
const FOG_SCALE: u32 = 0x09C4;
const POINT_SIZE: u32 = 0x043C;
/// `SET_SURFACE_COLOR_OFFSET`: the colour buffer the next primitives draw into.
const SURFACE_COLOR_OFFSET: u32 = 0x0210;
const POINT_PARAMS_ENABLE: u32 = 0x0318;
const POINT_PARAMS: u32 = 0x0A30;
const LINE_WIDTH: u32 = 0x0380;
/// How far a line is pulled toward the eye, in depth, so that outlines pass the depth test.
const LINE_PULL: f32 = 0.001;
/// The largest point size, in pixels.
const POINT_SIZE_MAX: f32 = 63.875;
const COMBINER_ALPHA_IN: u32 = 0x0260;
const COMBINER_ALPHA_OUT: u32 = 0x0AA0;
const COMBINER_COLOUR_IN: u32 = 0x0AC0;
const COMBINER_COLOUR_OUT: u32 = 0x1E40;
const COMBINER_CONSTANT0: u32 = 0x0A60;
const COMBINER_CONSTANT1: u32 = 0x0A80;
const FINAL_COMBINER0: u32 = 0x0288;
const FINAL_COMBINER1: u32 = 0x028C;
const FINAL_CONSTANT0: u32 = 0x1E20;
const FINAL_CONSTANT1: u32 = 0x1E24;
const COMBINER_CONTROL: u32 = 0x1E60;
const TRANSFORM_MODE: u32 = 0x1E94;
const PROGRAM_LOAD: u32 = 0x1E9C;
const PROGRAM_START: u32 = 0x1EA0;
const CONSTANT_LOAD: u32 = 0x1EA4;
const PROGRAM_WORDS: std::ops::Range<u32> = 0x0B00..0x0B80;
const CONSTANT_WORDS: std::ops::Range<u32> = 0x0B80..0x0C00;
/// `SET_VERTEX_DATA2F_M`, `2S`, `4UB`, `4S_M`, `4F_M` and the end of their runs: an attribute's
/// inline value.
const VERTEX_DATA2F: u32 = 0x1880;
const VERTEX_DATA2S: u32 = 0x1900;
const VERTEX_DATA4UB: u32 = 0x1940;
const VERTEX_DATA4S: u32 = 0x1980;
const VERTEX_DATA4F: u32 = 0x1A00;
const VERTEX_DATA_END: u32 = 0x1B00;

/// `SET_FOG_MODE` values: the equation that turns a fog distance into the factor.
const FOG_LINEAR: u32 = 0x2601;
const FOG_EXP: u32 = 0x0800;
const FOG_EXP2: u32 = 0x0801;
const FOG_EXP_ABS: u32 = 0x0802;
const FOG_EXP2_ABS: u32 = 0x0803;
const FOG_LINEAR_ABS: u32 = 0x0804;

/// `SET_FOG_GEN_MODE` values: where the fixed function takes the fog distance from.
const FOG_FROM_SPECULAR_ALPHA: u32 = 0;
const FOG_RADIAL: u32 = 1;
const FOG_ABS_PLANAR: u32 = 3;
const FOG_FROM_COORDINATE: u32 = 6;
const FOG_PLANAR: u32 = 2;

/// The program-constant row of the fog plane.
const FOG_PLANE_ROW: usize = 0x39;

/// GL enums used as defaults.
const GL_LEQUAL: u32 = 0x203;
const GL_ALWAYS: u32 = 0x207;

/// What a primitive's vertices make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Points,
    Lines,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads,
}

/// Every kind `SET_BEGIN_END` takes. Those [`Kind`] has no name for, points, and lines wider
/// than a pixel are assembled into the others once the vertices are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Begin {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads,
    QuadStrip,
    Polygon,
}

impl Begin {
    /// Whether the primitive is made of faces, which polygon modes and culling apply to.
    fn has_faces(self) -> bool {
        !matches!(
            self,
            Self::Points | Self::Lines | Self::LineLoop | Self::LineStrip
        )
    }

    fn from_value(value: u32) -> Option<Self> {
        Some(match value {
            1 => Self::Points,
            2 => Self::Lines,
            3 => Self::LineLoop,
            4 => Self::LineStrip,
            5 => Self::Triangles,
            6 => Self::TriangleStrip,
            7 => Self::TriangleFan,
            8 => Self::Quads,
            9 => Self::QuadStrip,
            10 => Self::Polygon,
            _ => return None,
        })
    }

    /// The kind its vertices are drawn as, before points and wide lines become quads.
    fn kind(self) -> Kind {
        match self {
            Self::Points => Kind::Points,
            Self::Lines | Self::LineLoop | Self::LineStrip => Kind::Lines,
            Self::Triangles => Kind::Triangles,
            Self::TriangleStrip | Self::QuadStrip => Kind::TriangleStrip,
            Self::TriangleFan | Self::Polygon => Kind::TriangleFan,
            Self::Quads => Kind::Quads,
        }
    }
}

/// A transformed vertex: screen position, depth (over the depth buffer's largest) and w; colour;
/// the four stages' texture coordinates `s, t, r, q` (`(0, 0, 0, 1)` where none are given); the
/// fraction of its colour that survives fog; and the specular colour, black with alpha 1 while
/// specular is off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub depth: f32,
    pub w: f32,
    pub colour: [f32; 4],
    pub texture: [[f32; 4]; 4],
    pub fog: f32,
    pub specular: [f32; 4],
}

/// The register-combiner program in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Combiner {
    pub alpha_inputs: [u32; 8],
    pub alpha_outputs: [u32; 8],
    pub colour_inputs: [u32; 8],
    pub colour_outputs: [u32; 8],
    pub constant0: [u32; 8],
    pub constant1: [u32; 8],
    pub final0: u32,
    pub final1: u32,
    pub final_constants: [u32; 2],
    /// Low byte: the number of stages (at most 8); the rest flags.
    pub control: u32,
}

/// A depth test: function (GL enum) and whether depth is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Depth {
    pub function: u32,
    pub write: bool,
}

/// One primitive and the state it is drawn with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    pub kind: Kind,
    pub vertices: Vec<Vertex>,
    /// The four stages' textures, when on and their mode reads one.
    pub textures: [Option<Texture>; 4],
    /// What each stage gives the combiners, and what the modes that move coordinates read.
    pub modes: [Mode; 4],
    pub shader: Shader,
    /// The colour buffer it is drawn into (`SET_SURFACE_COLOR_OFFSET`): a title may draw into
    /// one off the screen and sample it as a texture afterwards.
    pub surface: u32,
    pub depth: Option<Depth>,
    /// Source and destination factors (GL enums), when blending.
    pub blend: Option<(u32, u32)>,
    /// The fog colour (`0xAABBGGRR`), when fogged.
    pub fog: Option<u32>,
    pub combiner: Option<Combiner>,
    /// Function (GL enum) and reference, when alpha testing.
    pub alpha_test: Option<(u32, u8)>,
    pub point_size: f32,
    /// Where its fragments land and what they do there.
    pub pixel: Pixel,
    /// Whether it is a `CLEAR_SURFACE`, which alone does not make a frame to show.
    pub clear: bool,
}

/// The registers' values (0 until written): the methods below 0x2000 in a table, any others
/// aside.
struct Registers {
    table: Box<[u32; 0x800]>,
    others: std::collections::HashMap<u32, u32>,
}

impl Default for Registers {
    fn default() -> Self {
        Self {
            table: Box::new([0; 0x800]),
            others: std::collections::HashMap::new(),
        }
    }
}

impl Registers {
    fn get(&self, method: u32) -> u32 {
        match self.table.get(method as usize / 4) {
            Some(&value) if method.is_multiple_of(4) => value,
            _ => self.others.get(&method).copied().unwrap_or(0),
        }
    }

    fn insert(&mut self, method: u32, value: u32) {
        match self.table.get_mut(method as usize / 4) {
            Some(slot) if method.is_multiple_of(4) => *slot = value,
            _ => {
                self.others.insert(method, value);
            }
        }
    }
}

/// The state commands build up.
struct State {
    regs: Registers,
    /// The enabled attributes of the vertices, as the formats are now.
    layout: Arc<Vec<(usize, u32, u32, usize)>>,
    stages: [StageRegs; 4],
    stage_on: [bool; 4],
    /// `SET_SHADER_STAGE_PROGRAM`, once written.
    stage_program: Option<u32>,
    shader: Shader,
    combiner_set: bool,
    program: Program,
    /// The program from its start, decoded on first use and dropped when a word of it or its
    /// start changes: every vertex runs it, and titles load the same programs again and again.
    decoded: Option<Arc<Vec<Op>>>,
    /// The slots the next streamed program and constant words go to.
    program_slot: usize,
    constant_slot: usize,
    /// The fixed-function pipeline's lights, material and texture-coordinate generation.
    pipeline: Pipeline,
    /// What each attribute the vertices do not carry reads as, and which have been set.
    inline: InlineValues,
    pixel: PixelRegs,
}

impl State {
    fn new() -> Self {
        let mut regs = Registers::default();
        for i in 0..16 {
            regs.insert(ARRAY_FORMAT + 4 * i, 2);
        }
        for k in [0u32, 5, 10, 15] {
            regs.insert(MATRIX + 4 * k, 1.0f32.to_bits());
        }
        regs.insert(DEPTH_FUNC, GL_LEQUAL);
        regs.insert(DEPTH_WRITE, 1);
        regs.insert(ALPHA_FUNC, GL_ALWAYS);
        regs.insert(FOG_BASE, 1.0f32.to_bits());
        regs.insert(FOG_MODE, FOG_LINEAR);
        regs.insert(FOG_GEN_MODE, FOG_PLANAR);
        let layout = Arc::new(layout(|i| regs.get(ARRAY_FORMAT + 4 * i)));
        Self {
            regs,
            layout,
            stages: [StageRegs::default(); 4],
            stage_on: [true, false, false, false],
            stage_program: None,
            shader: Shader::default(),
            combiner_set: false,
            program: Program::default(),
            decoded: None,
            program_slot: 0,
            constant_slot: 0,
            pipeline: Pipeline::default(),
            inline: InlineValues::default(),
            pixel: PixelRegs::default(),
        }
    }

    fn get(&self, method: u32) -> u32 {
        self.regs.get(method)
    }

    fn float(&self, method: u32) -> f32 {
        f32::from_bits(self.get(method))
    }

    fn apply(&mut self, method: u32, value: u32) {
        self.regs.insert(method, value);
        if (ARRAY_FORMAT..ARRAY_FORMAT + 4 * 16).contains(&method) {
            self.layout = Arc::new(layout(|i| self.get(ARRAY_FORMAT + 4 * i)));
        }
        self.pixel.set(method, value);
        self.pipeline.set(method, value);
        self.inline.set(method, value);
        if let Some((row, component)) = program::transform_constant(method)
            && let Some(constant) = Arc::make_mut(&mut self.program.constants).get_mut(row)
        {
            constant[component] = f32::from_bits(value);
        }
        if (STAGE_BASE..STAGE_BASE + 4 * STAGE_STRIDE).contains(&method) {
            let stage = ((method - STAGE_BASE) / STAGE_STRIDE) as usize;
            let offset = (method - STAGE_BASE) % STAGE_STRIDE;
            self.stages[stage].set(offset, value);
            if offset == 0x0C {
                self.stage_on[stage] = value & 1 << 30 != 0;
            }
        }
        self.shader.set(method, value);
        match method {
            COMBINER_CONTROL => self.combiner_set = true,
            SHADER_STAGE_PROGRAM => self.stage_program = Some(value),
            PROGRAM_LOAD => self.program_slot = value as usize,
            CONSTANT_LOAD => self.constant_slot = value as usize,
            PROGRAM_START => {
                if self.program.start != value as usize {
                    self.program.start = value as usize;
                    self.decoded = None;
                }
            }
            // The method's address names the word; the fourth moves to the next slot.
            m if PROGRAM_WORDS.contains(&m) => {
                let word = ((m - PROGRAM_WORDS.start) / 4 % 4) as usize;
                if let Some(insn) = self.program.instructions.get_mut(self.program_slot)
                    && insn[word] != value
                {
                    insn[word] = value;
                    self.decoded = None;
                }
                if word == 3 {
                    self.program_slot += 1;
                }
            }
            m if CONSTANT_WORDS.contains(&m) => {
                let word = ((m - CONSTANT_WORDS.start) / 4 % 4) as usize;
                if let Some(constant) =
                    Arc::make_mut(&mut self.program.constants).get_mut(self.constant_slot)
                {
                    constant[word] = f32::from_bits(value);
                }
                if word == 3 {
                    self.constant_slot += 1;
                }
            }
            _ => {}
        }
    }

    /// Each enabled attribute: its number, type, size and words per vertex.
    fn layout(&self) -> &[(usize, u32, u32, usize)] {
        &self.layout
    }

    /// Leaves each attribute the primitive's vertices carried at its last vertex's value, as
    /// the engine does after a draw.
    fn carry(&mut self, words: &[u32]) {
        self.inline.carry(&self.layout, words);
    }

    fn combiner(&self) -> Combiner {
        let array = |base: u32| -> [u32; 8] {
            std::array::from_fn(|i| self.get(base + 4 * u32::try_from(i).unwrap_or(0)))
        };
        Combiner {
            alpha_inputs: array(COMBINER_ALPHA_IN),
            alpha_outputs: array(COMBINER_ALPHA_OUT),
            colour_inputs: array(COMBINER_COLOUR_IN),
            colour_outputs: array(COMBINER_COLOUR_OUT),
            constant0: array(COMBINER_CONSTANT0),
            constant1: array(COMBINER_CONSTANT1),
            final0: self.get(FINAL_COMBINER0),
            final1: self.get(FINAL_COMBINER1),
            final_constants: [self.get(FINAL_CONSTANT0), self.get(FINAL_CONSTANT1)],
            control: self.get(COMBINER_CONTROL),
        }
    }

    /// Whether vertices are fogged: fog is on and the final combiner's input A is the fog
    /// register (3).
    fn fogged(&self) -> bool {
        self.get(FOG_ENABLE) != 0 && (self.get(FINAL_COMBINER0) >> 24) & 0xF == 3
    }

    /// Each stage's mode.
    fn modes(&self) -> [Mode; 4] {
        std::array::from_fn(|s| Mode::of(self.stage_program, s, self.stage_on[s]))
    }

    /// Each stage's texture, when the stage is on and its mode reads one.
    fn textures(&self) -> [Option<Texture>; 4] {
        let modes = self.modes();
        std::array::from_fn(|s| {
            if self.stage_on[s] && modes[s].samples() {
                self.stages[s].texture()
            } else {
                None
            }
        })
    }

    fn primitive(&self, kind: Kind) -> Primitive {
        Primitive {
            kind,
            vertices: Vec::new(),
            textures: self.textures(),
            modes: self.modes(),
            shader: self.shader,
            surface: self.get(SURFACE_COLOR_OFFSET),
            depth: (self.get(DEPTH_ENABLE) != 0).then(|| Depth {
                function: self.get(DEPTH_FUNC),
                write: self.get(DEPTH_WRITE) != 0,
            }),
            blend: (self.get(BLEND_ENABLE) != 0).then_some(self.pixel.blend),
            fog: self.fogged().then(|| self.get(FOG_COLOUR)),
            combiner: self.combiner_set.then(|| self.combiner()),
            alpha_test: (self.get(ALPHA_ENABLE) != 0)
                .then(|| (self.get(ALPHA_FUNC), (self.get(ALPHA_REF) & 0xFF) as u8)),
            point_size: self.point_size(),
            pixel: self.pixel.pixel(),
            clear: false,
        }
    }

    /// `SET_POINT_SIZE`, in pixels (nine bits in eighths).
    fn point_size(&self) -> f32 {
        f32::from(u16::try_from(self.get(POINT_SIZE) & 0x1FF).unwrap_or(0)) / 8.0
    }

    /// `SET_LINE_WIDTH`, in pixels (nine bits in eighths).
    fn line_width(&self) -> f32 {
        f32::from(u16::try_from(self.get(LINE_WIDTH) & 0x1FF).unwrap_or(0)) / 8.0
    }

    /// The size of a fixed-function point at eye-space `eye`: `SET_POINT_SIZE`, at least a
    /// pixel; with the point parameters on, attenuated by its distance `d` from the eye. The
    /// eight parameters are the constant, linear and quadratic attenuation, a scale, two
    /// unused, a bias and the smallest size: `((1 / sqrt(a + b d + c d^2) + bias) * scale +
    /// smallest)`, kept between the smallest size and `scale + smallest`, both at most 63.875.
    fn fixed_point_size(&self, eye: [f32; 4]) -> f32 {
        if self.get(POINT_PARAMS_ENABLE) == 0 {
            return self.point_size().max(1.0);
        }
        let p: [f32; 8] =
            std::array::from_fn(|i| self.float(POINT_PARAMS + 4 * u32::try_from(i).unwrap_or(0)));
        let d = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
        let smallest = p[7].min(POINT_SIZE_MAX);
        let largest = (p[3] + smallest).min(POINT_SIZE_MAX).max(smallest);
        let attenuated = 1.0 / (p[0] + p[1] * d + p[2] * d * d).sqrt();
        ((attenuated + p[6]) * p[3] + p[7]).clamp(smallest, largest)
    }

    /// Whether the vertices run the vertex program.
    fn programmed(&self) -> bool {
        self.get(TRANSFORM_MODE) & 3 == 2
    }

    /// The fixed-function vertices of `words` in the inline layout, each with the size of the
    /// point it is drawn as when it is one.
    fn vertices(&self, words: &[u32]) -> Vec<(Vertex, f32)> {
        let mut defaults = self.inline.values;
        // The fixed function draws white while the title has given no diffuse colour at all.
        if self.inline.set & 1 << fixed::DIFFUSE == 0 {
            defaults[fixed::DIFFUSE] = [1.0; 4];
        }
        let mut out = Vec::new();
        for_each_input(self.layout(), &defaults, words, |inputs| {
            out.push(self.fixed(inputs));
        });
        out
    }

    /// What running the vertex program over the vertices of `words` needs, as the state is
    /// now: the program decoded, a snapshot of its constants, and the state its outputs are
    /// read with.
    fn program_vertices(&mut self, words: Vec<u32>) -> ProgramVertices {
        let program = &self.program;
        let ops = Arc::clone(
            self.decoded
                .get_or_insert_with(|| Arc::new(program.decode())),
        );
        ProgramVertices {
            ops,
            constants: Arc::clone(&self.program.constants),
            layout: Arc::clone(&self.layout),
            defaults: self.inline.values,
            words,
            zmax: self.pixel.zmax(),
            fog: self.fogged().then(|| {
                (
                    self.get(FOG_MODE),
                    self.float(FOG_BASE),
                    self.float(FOG_SCALE),
                )
            }),
            specular: self.pixel.specular(),
            point_size: if self.get(POINT_PARAMS_ENABLE) != 0 {
                None
            } else if self.point_size() > 0.0 {
                Some(self.point_size())
            } else {
                Some(1.0)
            },
        }
    }

    /// How much of a vertex's colour survives fog at `distance`.
    fn fog_factor(&self, distance: f32) -> f32 {
        fog_factor(
            self.get(FOG_MODE),
            self.float(FOG_BASE),
            self.float(FOG_SCALE),
            distance,
        )
    }

    /// The fixed-function fog distance of a vertex with `inputs` at eye-space `eye`, as
    /// `SET_FOG_GEN_MODE` says: the specular colour's alpha, the distance from the eye, the
    /// fog plane against the eye-space position (planar, the default, which is minus the eye
    /// depth with Direct3D's plane) or its magnitude, or the fog coordinate (attribute 5).
    fn fog_distance(&self, inputs: &[[f32; 4]; 16], eye: [f32; 4]) -> f32 {
        let plane = self.program.constants[FOG_PLANE_ROW];
        let planar = plane[0] * eye[0] + plane[1] * eye[1] + plane[2] * eye[2] + plane[3];
        match self.get(FOG_GEN_MODE) {
            FOG_FROM_SPECULAR_ALPHA => inputs[fixed::SPECULAR][3].clamp(0.0, 1.0),
            FOG_RADIAL => (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt(),
            FOG_ABS_PLANAR => planar.abs(),
            FOG_FROM_COORDINATE => inputs[fixed::FOG_COORDINATE][0],
            _ => planar,
        }
    }

    /// The specular colour of a vertex whose second colour is `value`.
    fn specular(&self, value: [f32; 4]) -> [f32; 4] {
        self.pixel
            .specular()
            .map_or(pixel::NO_SPECULAR, |ignore_alpha| {
                pixel::specular_output(value, ignore_alpha)
            })
    }

    /// The fixed function: the pipeline's colours, texture coordinates and eye-space
    /// position, then the composite matrix, the divide and the viewport offset.
    fn fixed(&self, inputs: &[[f32; 4]; 16]) -> (Vertex, f32) {
        let out = self.pipeline.run(inputs, &self.program.constants);
        let position = out.position;
        let row = |r: u32| {
            (0..4)
                .map(|c| self.float(MATRIX + 4 * (4 * r + c)) * position[c as usize])
                .sum::<f32>()
        };
        let (px, py, pz, w) = (row(0), row(1), row(2), row(3));
        let divisor = if w == 0.0 { 1.0 } else { w };
        let fog = if self.fogged() {
            self.fog_factor(self.fog_distance(inputs, out.eye))
        } else {
            1.0
        };
        let vertex = Vertex {
            x: px / divisor + self.float(VIEWPORT_OFFSET),
            y: py / divisor + self.float(VIEWPORT_OFFSET + 4),
            depth: pz / divisor / self.pixel.zmax(),
            w,
            colour: out
                .diffuse
                .map(|c| if c.is_nan() { 1.0 } else { c.clamp(0.0, 1.0) }),
            texture: out.texture,
            fog,
            specular: self.specular(out.specular),
        };
        (vertex, self.fixed_point_size(out.eye))
    }
}

/// Calls `f` with the attributes of each vertex of `words` in `layout`; attributes the vertices
/// do not carry read as `defaults`.
fn for_each_input(
    layout: &[(usize, u32, u32, usize)],
    defaults: &[[f32; 4]; 16],
    words: &[u32],
    mut f: impl FnMut(&[[f32; 4]; 16]),
) {
    let stride: usize = layout.iter().map(|a| a.3).sum();
    if stride == 0 {
        return;
    }
    for chunk in words.chunks_exact(stride) {
        let mut inputs = *defaults;
        let mut at = 0;
        for &(i, kind, size, n) in layout {
            inputs[i] = attribute(kind, size, &chunk[at..at + n]);
            at += n;
        }
        f(&inputs);
    }
}

/// A programmed draw's vertices, still to run: the program and the state its outputs are read
/// with, as they were when the title drew it, and the vertices' words.
struct ProgramVertices {
    ops: Arc<Vec<Op>>,
    constants: Arc<Vec<[f32; 4]>>,
    layout: Arc<Vec<(usize, u32, u32, usize)>>,
    /// What the attributes the vertices do not carry read as.
    defaults: [[f32; 4]; 16],
    words: Vec<u32>,
    zmax: f32,
    /// The fog mode, base and scale, when fogged.
    fog: Option<(u32, f32, f32)>,
    /// Whether specular is on, and whether it drops its alpha.
    specular: Option<bool>,
    /// The size of points drawn, or `None` for the program's `oPts`.
    point_size: Option<f32>,
}

impl ProgramVertices {
    /// Runs the program over each vertex.
    fn run(&self) -> Vec<(Vertex, f32)> {
        let mut out = Vec::new();
        for_each_input(&self.layout, &self.defaults, &self.words, |inputs| {
            out.push(self.vertex(inputs));
        });
        out
    }

    /// The vertex-program path: position, colour, texture coordinates and fog from its outputs;
    /// the point size is `SET_POINT_SIZE` (1 when 0), or the program's `oPts` with the point
    /// parameters on.
    fn vertex(&self, inputs: &[[f32; 4]; 16]) -> (Vertex, f32) {
        let out = program::run_decoded(&self.ops, &self.constants, inputs);
        let p = out[program::POSITION];
        let colour = out[program::DIFFUSE].map(|c| c.clamp(0.0, 1.0));
        let t = program::TEXTURE0;
        // The position is in screen pixels with the viewport offset in. A screen-space program
        // fed without a reciprocal w leaves w at 0 or 1/0: w is then 1, as a divisor and for
        // the renderer. Depth beyond the buffer is left for the renderer to clip.
        let w = if p[3].is_finite() && p[3] != 0.0 {
            p[3]
        } else {
            1.0
        };
        let vertex = Vertex {
            x: p[0],
            y: p[1],
            depth: p[2] / self.zmax,
            w,
            colour,
            texture: [out[t], out[t + 1], out[t + 2], out[t + 3]],
            // The program's fog output is a distance, which the fog mode turns into the factor.
            fog: self.fog.map_or(1.0, |(mode, base, scale)| {
                fog_factor(mode, base, scale, out[program::FOG][0])
            }),
            specular: self.specular.map_or(pixel::NO_SPECULAR, |ignore_alpha| {
                pixel::specular_output(out[SPECULAR], ignore_alpha)
            }),
        };
        let size = self.point_size.unwrap_or(out[program::POINT_SIZE][0]);
        (vertex, size)
    }
}

/// A primitive whose vertices are still to come, and how they are assembled.
struct Open {
    begin: Begin,
    primitive: Primitive,
    raster: pixel::Raster,
    line_width: f32,
}

impl Open {
    /// Pushes what the primitive draws with `vertices`; faces may be drawn as their edges or
    /// corners.
    fn finish(&self, vertices: &[(Vertex, f32)], out: &mut Vec<Primitive>) {
        let mut assembled = Vec::new();
        assemble(
            self.begin,
            self.primitive.clone(),
            vertices,
            self.line_width,
            &mut assembled,
        );
        for primitive in assembled {
            if self.begin.has_faces() {
                self.raster.assemble(primitive, self.line_width, out);
            } else {
                out.push(primitive);
            }
        }
    }
}

/// A programmed draw run after the frame's commands are in, and where its primitives go.
struct Deferred {
    /// How many primitives come before it.
    at: usize,
    open: Open,
    vertices: ProgramVertices,
}

impl Deferred {
    fn primitives(&self) -> Vec<Primitive> {
        let vertices = self.vertices.run();
        let mut out = Vec::new();
        self.open.finish(&vertices, &mut out);
        out
    }

    fn vertex_count(&self) -> usize {
        let stride: usize = self.vertices.layout.iter().map(|a| a.3).sum();
        self.vertices.words.len() / stride.max(1)
    }
}

/// Programmed vertices fewer than this in a frame run on the one thread.
const PARALLEL_VERTICES: usize = 2048;

/// Runs the deferred draws' programs on up to eight threads, each taking the next draw not yet
/// taken, and puts their primitives in `out` where they were drawn.
fn run_deferred(out: Vec<Primitive>, deferred: &[Deferred]) -> Vec<Primitive> {
    if deferred.is_empty() {
        return out;
    }
    let total: usize = deferred.iter().map(Deferred::vertex_count).sum();
    let threads = if total < PARALLEL_VERTICES {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(8)
            .min(deferred.len())
    };
    let mut results: Vec<Vec<Primitive>> = if threads <= 1 {
        deferred.iter().map(Deferred::primitives).collect()
    } else {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let work = || {
            let mut done = Vec::new();
            loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(d) = deferred.get(i) else {
                    return done;
                };
                done.push((i, d.primitives()));
            }
        };
        let mut results = vec![Vec::new(); deferred.len()];
        std::thread::scope(|scope| {
            let handles: Vec<_> = (1..threads).map(|_| scope.spawn(work)).collect();
            let mine = work();
            for (i, primitives) in handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("a vertex-program thread"))
                .chain(mine)
            {
                results[i] = primitives;
            }
        });
        results
    };
    let added: usize = results.iter().map(Vec::len).sum();
    let mut merged = Vec::with_capacity(out.len() + added);
    let mut before = out.into_iter();
    let mut taken = 0;
    for (d, primitives) in deferred.iter().zip(results.iter_mut()) {
        merged.extend(before.by_ref().take(d.at - taken));
        taken = d.at;
        merged.append(primitives);
    }
    merged.extend(before);
    merged
}

/// The specular colour's vertex-program output.
const SPECULAR: usize = 4;

/// An attribute's value as four floats (missing components 0, w 1): packed colours as RGBA in
/// 0..1, type 4 bytes / 255, type 1 signed 16-bit values over 32767 (at least -1), type 5 signed
/// 16-bit values as they are, type 6 a normal packed in 11, 11 and 10 bits, floats as they are.
fn attribute(kind: u32, size: u32, words: &[u32]) -> [f32; 4] {
    let mut value = [0.0, 0.0, 0.0, 1.0];
    match kind {
        0 => {
            let [blue, green, red, alpha] = words[0].to_le_bytes();
            value = [red, green, blue, alpha].map(|c| f32::from(c) / 255.0);
        }
        4 => {
            for (i, byte) in words[0]
                .to_le_bytes()
                .iter()
                .take(size as usize)
                .enumerate()
            {
                value[i] = f32::from(*byte) / 255.0;
            }
        }
        1 | 5 => {
            for i in 0..(size as usize).min(4) {
                let half = ((words[i / 2] >> (16 * (i % 2))) & 0xFFFF) as u16;
                let half = f32::from(half.cast_signed());
                value[i] = if kind == 1 {
                    (half / 32767.0).max(-1.0)
                } else {
                    half
                };
            }
        }
        6 => value = packed_normal(words[0]),
        _ => {
            for i in 0..(size as usize).min(4) {
                value[i] = f32::from_bits(words[i]);
            }
        }
    }
    value
}

/// The words an attribute of array format `format` takes in a vertex, as [`layout`] counts them.
pub(crate) fn attribute_words(format: u32) -> usize {
    let (kind, size) = (format & 0xF, (format >> 4) & 0xF);
    match kind {
        _ if size == 0 => 0,
        0 | 4 | 6 => 1,
        1 | 5 => size.div_ceil(2) as usize,
        _ => size as usize,
    }
}

/// Each enabled attribute of the formats `format(i)` gives: its number, type, size and words
/// per vertex.
fn layout(format: impl Fn(u32) -> u32) -> Vec<(usize, u32, u32, usize)> {
    (0..16)
        .filter_map(|i| {
            let format = format(i);
            let (kind, size) = (format & 0xF, (format >> 4) & 0xF);
            (size != 0).then(|| {
                let words = match kind {
                    0 | 4 | 6 => 1,
                    1 | 5 => size.div_ceil(2),
                    _ => size,
                };
                (i as usize, kind, size, words as usize)
            })
        })
        .collect()
}

/// Each attribute's inline value: what a program and the fixed function read for an attribute
/// the vertices do not carry. `SET_VERTEX_DATA*` sets it, and a draw leaves each attribute it
/// carried at its last vertex's value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct InlineValues {
    values: [[f32; 4]; 16],
    /// The attributes set since reset, a bit each.
    set: u16,
}

impl Default for InlineValues {
    fn default() -> Self {
        Self {
            values: [[0.0, 0.0, 0.0, 1.0]; 16],
            set: 0,
        }
    }
}

impl InlineValues {
    /// Takes a `SET_VERTEX_DATA*` method; returns whether it was one. The two-float form sets
    /// its component and z and w to 0 and 1; two shorts are x and y with z and w 0 and 1; four
    /// bytes are the components in byte order over 255; four shorts and four floats set their
    /// components only. Attributes are two words apart in the two-float and four-short runs,
    /// four in the four-float run, one in the others.
    fn set(&mut self, method: u32, value: u32) -> bool {
        if !(VERTEX_DATA2F..VERTEX_DATA_END).contains(&method) {
            return false;
        }
        let float = f32::from_bits(value);
        let [b0, b1, b2, b3] = value.to_le_bytes();
        let low = f32::from(i16::from_le_bytes([b0, b1]));
        let high = f32::from(i16::from_le_bytes([b2, b3]));
        let (run, base) = [
            (VERTEX_DATA4F, 0),
            (VERTEX_DATA4S, 1),
            (VERTEX_DATA4UB, 2),
            (VERTEX_DATA2S, 3),
            (VERTEX_DATA2F, 4),
        ]
        .into_iter()
        .find(|&(start, _)| method >= start)
        .unwrap_or((VERTEX_DATA2F, 4));
        let word = ((method - run) / 4) as usize;
        let attribute = match base {
            0 => {
                self.values[word / 4][word % 4] = float;
                word / 4
            }
            1 => {
                let at = word % 2 * 2;
                self.values[word / 2][at..at + 2].copy_from_slice(&[low, high]);
                word / 2
            }
            2 => {
                self.values[word] = [b0, b1, b2, b3].map(|b| f32::from(b) / 255.0);
                word
            }
            3 => {
                self.values[word] = [low, high, 0.0, 1.0];
                word
            }
            _ => {
                let v = &mut self.values[word / 2];
                v[word % 2] = float;
                v[2] = 0.0;
                v[3] = 1.0;
                word / 2
            }
        };
        self.set |= 1 << attribute;
        true
    }

    /// Leaves each attribute of `layout` at its value in the last whole vertex of `words`. A
    /// packed colour is kept in its bytes' order (blue first), as the engine stores it, not in
    /// the red-first order a vertex reads it in.
    fn carry(&mut self, layout: &[(usize, u32, u32, usize)], words: &[u32]) {
        let stride: usize = layout.iter().map(|a| a.3).sum();
        if stride == 0 || words.len() < stride {
            return;
        }
        let last = &words[(words.len() / stride - 1) * stride..];
        let mut at = 0;
        for &(i, kind, size, n) in layout {
            let own = &last[at..at + n];
            self.values[i] = if kind == 0 {
                let mut value = [0.0, 0.0, 0.0, 1.0];
                for (c, byte) in value
                    .iter_mut()
                    .zip(own[0].to_le_bytes())
                    .take(size as usize)
                {
                    *c = f32::from(byte) / 255.0;
                }
                value
            } else {
                attribute(kind, size, own)
            };
            self.set |= 1 << i;
            at += n;
        }
    }
}

/// The inline values the commands so far left, for the GPU model to start each frame with:
/// a frame's own commands carry only the changes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct InlineShadow {
    values: InlineValues,
    formats: [u32; 16],
    /// The open primitive's last vertex, as a ring of its words, where the next word goes in
    /// it, and how many words came.
    vertex: Vec<u32>,
    at: usize,
    words: usize,
    open: bool,
}

impl InlineShadow {
    /// Takes one vertex word (`INLINE_ARRAY`): the most frequent method by far, so it skips
    /// every other test.
    #[inline]
    pub(crate) fn take_vertex_word(&mut self, value: u32) {
        if self.open && !self.vertex.is_empty() {
            self.vertex[self.at] = value;
            self.at += 1;
            if self.at == self.vertex.len() {
                self.at = 0;
            }
            self.words += 1;
        }
    }

    /// Takes vertex words, as [`InlineShadow::take_vertex_word`] does each.
    pub(crate) fn take_vertex_words(&mut self, words: &[u32]) {
        let stride = self.vertex.len();
        if !self.open || stride == 0 {
            return;
        }
        // Only the words of the last whole ring turn matter.
        let skip = words.len().saturating_sub(stride);
        self.at = (self.at + skip) % stride;
        self.words += skip;
        for &word in &words[skip..] {
            self.take_vertex_word(word);
        }
    }

    /// Takes one 3D method.
    pub(crate) fn take(&mut self, method: u32, value: u32) {
        if method == INLINE_ARRAY {
            self.take_vertex_word(value);
            return;
        }
        if self.values.set(method, value) {
            return;
        }
        match method {
            m if (ARRAY_FORMAT..ARRAY_FORMAT + 64).contains(&m) => {
                self.formats[((m - ARRAY_FORMAT) / 4) as usize] = value;
            }
            SET_BEGIN_END if value != 0 => {
                let stride = self.formats.iter().map(|&f| attribute_words(f)).sum();
                self.vertex.clear();
                self.vertex.resize(stride, 0);
                self.at = 0;
                self.words = 0;
                self.open = true;
            }
            SET_BEGIN_END => {
                let stride = self.vertex.len();
                if self.open
                    && stride > 0
                    && self.words >= stride
                    && self.words.is_multiple_of(stride)
                {
                    let layout = layout(|i| self.formats[i as usize]);
                    self.values.carry(&layout, &self.vertex);
                }
                self.open = false;
            }
            _ => {}
        }
    }

    /// The commands that set each value that is not the default: four floats an attribute.
    pub(crate) fn replay(&self) -> Vec<Command> {
        let mut out = Vec::new();
        for (attribute, value) in (0u32..).zip(&self.values.values) {
            if self.values.set & 1 << attribute == 0 {
                continue;
            }
            for (component, float) in (0u32..).zip(value) {
                out.push(Command {
                    class: KELVIN,
                    method: VERTEX_DATA4F + 16 * attribute + 4 * component,
                    value: float.to_bits(),
                });
            }
        }
        out
    }
}

/// The fraction of a vertex's colour that survives fog at `distance`, by the `SET_FOG_MODE`
/// equation over the two fog parameters, which carry a bias the equations take back out:
/// linear `base + d * scale - 1`, EXP `base + 2^(16 * d * scale) - 1.5`, EXP2 `base +
/// 2^(-32 * d^2 * scale^2) - 1.5`; the ABS modes take the result's magnitude. Clamped to 0..1;
/// an infinite distance, or a NaN result, is no fog.
fn fog_factor(mode: u32, base: f32, scale: f32, distance: f32) -> f32 {
    if distance.is_infinite() {
        return 1.0;
    }
    let factor = match mode {
        FOG_EXP | FOG_EXP_ABS => base + (16.0 * distance * scale).exp2() - 1.5,
        FOG_EXP2 | FOG_EXP2_ABS => {
            base + (-32.0 * distance * distance * scale * scale).exp2() - 1.5
        }
        _ => base + distance * scale - 1.0,
    };
    let factor = if matches!(mode, FOG_LINEAR_ABS | FOG_EXP_ABS | FOG_EXP2_ABS) {
        factor.abs()
    } else {
        factor
    };
    if factor.is_nan() {
        1.0
    } else {
        factor.clamp(0.0, 1.0)
    }
}

/// A normal packed in one word (type 6): x in bits 0..10 and y in 11..21, signed and over 1023,
/// z in 22..31, signed and over 511; w 1.
fn packed_normal(word: u32) -> [f32; 4] {
    let signed = |shift: u32, bits: u32| {
        // The field moved to the top of the word, then shifted back down with its sign.
        let top = (word << (32 - shift - bits)).cast_signed();
        f32::from(i16::try_from(top >> (32 - bits)).unwrap_or(0))
    };
    [
        signed(0, 11) / 1023.0,
        signed(11, 11) / 1023.0,
        signed(22, 10) / 511.0,
        1.0,
    ]
}

/// Whether a frame of `primitives` draws on the surface it shows (that of its last primitive)
/// more than clears: a frame that only clears the screen leaves the video overlay showing.
#[must_use]
pub fn shows_anything(primitives: &[Primitive]) -> bool {
    primitives.last().is_some_and(|last| {
        primitives
            .iter()
            .any(|p| p.surface == last.surface && !p.clear)
    })
}

/// The primitives of a frame's commands, applying state in order. Programmed draws run their
/// vertices once the commands are in, on several threads.
#[must_use]
pub fn primitives(commands: &[Command]) -> Vec<Primitive> {
    let mut state = State::new();
    let mut open: Option<Open> = None;
    // One buffer for every primitive's words, emptied between them.
    let mut words = Vec::new();
    let begins = commands
        .iter()
        .filter(|c| c.method == SET_BEGIN_END && c.value != 0)
        .count();
    let mut out = Vec::with_capacity(begins);
    let mut deferred = Vec::with_capacity(begins);
    for command in commands.iter().filter(|c| c.class == KELVIN) {
        match command.method {
            SET_BEGIN_END if command.value == 0 => {
                if let Some(mut primitive) = open.take() {
                    primitive.line_width = state.line_width();
                    if state.programmed() {
                        // The inline values the vertices read are taken before the carry.
                        let vertices = state.program_vertices(words.clone());
                        state.carry(&words);
                        deferred.push(Deferred {
                            at: out.len(),
                            open: primitive,
                            vertices,
                        });
                    } else {
                        let vertices = state.vertices(&words);
                        primitive.finish(&vertices, &mut out);
                        state.carry(&words);
                    }
                    words.clear();
                }
            }
            SET_BEGIN_END => {
                words.clear();
                open = Begin::from_value(command.value).map(|begin| Open {
                    begin,
                    primitive: state.primitive(begin.kind()),
                    raster: state.pixel.raster(),
                    line_width: 0.0,
                });
            }
            pixel::CLEAR_SURFACE => {
                let surface = state.get(SURFACE_COLOR_OFFSET);
                out.extend(state.pixel.clear(command.value, surface));
            }
            INLINE_ARRAY => {
                if open.is_some() {
                    words.push(command.value);
                }
            }
            method => state.apply(method, command.value),
        }
    }
    run_deferred(out, &deferred)
}

/// Pushes what a primitive of kind `begin` with `vertices` (each with its point size) draws:
/// line strips and loops as line pairs, quad strips as triangle strips (an odd last vertex
/// dropped), polygons as fans, points as squares of their size around them, and lines wider
/// than a pixel as quads along them. Every assembled quad keeps its vertex's colours and
/// coordinates.
fn assemble(
    begin: Begin,
    mut primitive: Primitive,
    vertices: &[(Vertex, f32)],
    line_width: f32,
    out: &mut Vec<Primitive>,
) {
    let n = vertices.len();
    let lines: Vec<[usize; 2]> = match begin {
        Begin::Points => {
            // Squares facing the screen, never culled.
            primitive.kind = Kind::Quads;
            primitive.pixel.cull = None;
            primitive.vertices = vertices
                .iter()
                .flat_map(|&(v, size)| point_quad(v, size))
                .collect();
            out.push(primitive);
            return;
        }
        Begin::Lines => (0..n / 2).map(|i| [2 * i, 2 * i + 1]).collect(),
        Begin::LineStrip | Begin::LineLoop => {
            let mut lines: Vec<[usize; 2]> = (1..n).map(|i| [i - 1, i]).collect();
            if begin == Begin::LineLoop && n > 2 {
                lines.push([n - 1, 0]);
            }
            lines
        }
        Begin::QuadStrip => {
            primitive.vertices = vertices[..n - n % 2].iter().map(|v| v.0).collect();
            out.push(primitive);
            return;
        }
        _ => {
            primitive.vertices = vertices.iter().map(|v| v.0).collect();
            out.push(primitive);
            return;
        }
    };
    if line_width > 1.0 {
        primitive.kind = Kind::Quads;
        primitive.pixel.cull = None;
        primitive.vertices = lines
            .iter()
            .flat_map(|&[a, b]| wide_line(vertices[a].0, vertices[b].0, line_width))
            .collect();
    } else {
        primitive.vertices = lines.iter().flatten().map(|&i| vertices[i].0).collect();
    }
    out.push(primitive);
}

/// A point as the square of `size` pixels (at least one) around it.
pub(super) fn point_quad(vertex: Vertex, size: f32) -> [Vertex; 4] {
    let half = size.max(1.0) / 2.0;
    [(-half, -half), (half, -half), (half, half), (-half, half)].map(|(dx, dy)| Vertex {
        x: vertex.x + dx,
        y: vertex.y + dy,
        ..vertex
    })
}

/// A line `width` pixels wide as a quad along it, each end with its vertex's colours, pulled
/// toward the eye as thin lines are. A line of no length draws nothing.
pub(super) fn wide_line(a: Vertex, b: Vertex, width: f32) -> Vec<Vertex> {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let length = dx.hypot(dy);
    if !length.is_normal() {
        return Vec::new();
    }
    let (nx, ny) = (-dy / length * width / 2.0, dx / length * width / 2.0);
    let side = |v: Vertex, sign: f32| Vertex {
        x: v.x + sign * nx,
        y: v.y + sign * ny,
        depth: v.depth - LINE_PULL,
        ..v
    };
    vec![side(a, 1.0), side(b, 1.0), side(b, -1.0), side(a, -1.0)]
}

/// The distinct textures bound on any of the four stages at each primitive start, in first-bound
/// order.
#[must_use]
pub fn bound(commands: &[Command]) -> Vec<Texture> {
    let mut state = State::new();
    let mut textures: Vec<Texture> = Vec::new();
    for command in commands.iter().filter(|c| c.class == KELVIN) {
        if command.method == SET_BEGIN_END && Begin::from_value(command.value).is_some() {
            for texture in state.textures().into_iter().flatten() {
                if !textures.contains(&texture) {
                    textures.push(texture);
                }
            }
        } else if command.method != INLINE_ARRAY && command.method != SET_BEGIN_END {
            state.apply(command.method, command.value);
        }
    }
    textures
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn a_triangle_through_the_fixed_function_transform() {
        let mut commands = vec![
            // Position: three floats; diffuse in attribute 3 as a packed colour.
            kelvin(ARRAY_FORMAT, 3 << 4 | 2),
            kelvin(ARRAY_FORMAT + 12, 4 << 4),
            // Scale x by 2, offset the viewport by (10, 20).
            float(MATRIX, 2.0),
            float(VIEWPORT_OFFSET, 10.0),
            float(VIEWPORT_OFFSET + 4, 20.0),
            kelvin(SET_BEGIN_END, 5),
        ];
        for (x, y) in [(1.0f32, 2.0f32), (3.0, 4.0), (5.0, 6.0)] {
            commands.extend([
                float(INLINE_ARRAY, x),
                float(INLINE_ARRAY, y),
                float(INLINE_ARRAY, 0.5),
            ]);
            commands.push(kelvin(INLINE_ARRAY, 0x80FF_0000));
        }
        commands.push(kelvin(SET_BEGIN_END, 0));
        let primitives = primitives(&commands);
        assert_eq!(primitives.len(), 1);
        let p = &primitives[0];
        assert_eq!(p.kind, Kind::Triangles);
        assert_eq!(p.vertices.len(), 3);
        let v = p.vertices[1];
        // Depth is over the depth buffer's largest, a Z24 one's by default.
        assert_eq!((v.x, v.y, v.depth), (16.0, 24.0, 0.5 / 16_777_215.0));
        assert_eq!(v.colour, [1.0, 0.0, 0.0, f32::from(0x80u8) / 255.0]);
        assert!(
            p.depth.is_none() && p.blend.is_none() && p.alpha_test.is_none() && p.fog.is_none()
        );
        // Stage 0 starts on with its registers' defaults (format 0 is Y8, 1x1); stage 1 off.
        assert_eq!(
            p.textures[0].map(|t| (t.code, t.width, t.height)),
            Some((0, 1, 1))
        );
        assert_eq!(p.textures[1], None);
    }

    #[test]
    fn white_without_a_packed_colour_and_state_snapshots_at_the_start() {
        let commands = vec![
            kelvin(ARRAY_FORMAT, 3 << 4 | 2),
            kelvin(DEPTH_ENABLE, 1),
            kelvin(BLEND_ENABLE, 1),
            kelvin(ALPHA_ENABLE, 1),
            kelvin(ALPHA_REF, 0x1_80),
            kelvin(POINT_SIZE, 16),
            kelvin(SET_BEGIN_END, 1),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            kelvin(DEPTH_ENABLE, 0),
            kelvin(SET_BEGIN_END, 0),
        ];
        let p = &primitives(&commands)[0];
        assert_eq!(p.vertices[0].colour, [1.0; 4]);
        assert_eq!(
            p.depth,
            Some(Depth {
                function: GL_LEQUAL,
                write: true
            })
        );
        assert_eq!(p.blend, Some((0x302, 0x303)));
        assert_eq!(p.alpha_test, Some((GL_ALWAYS, 0x80)));
        assert_eq!(p.point_size, 2.0);
    }

    #[test]
    fn fog_needs_the_fog_register_as_the_final_combiners_input_a() {
        // Radial fog: the vertex is 2 from the eye (the first model-view matrix moves it).
        let base = vec![
            kelvin(ARRAY_FORMAT, 3 << 4 | 2),
            kelvin(FOG_ENABLE, 1),
            kelvin(FOG_COLOUR, 0x00FF_0000),
            kelvin(FOG_GEN_MODE, FOG_RADIAL),
            float(FOG_BASE, 2.0),
            float(FOG_SCALE, -0.25),
            float(0x048C, 2.0),
        ];
        let draw = [
            kelvin(SET_BEGIN_END, 1),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            kelvin(SET_BEGIN_END, 0),
        ];
        let unfogged: Vec<Command> = base.iter().chain(&draw).copied().collect();
        assert_eq!(primitives(&unfogged)[0].fog, None);
        let mut fogged = base;
        fogged.push(kelvin(FINAL_COMBINER0, 3 << 24));
        fogged.extend(draw);
        let p = &primitives(&fogged)[0];
        assert_eq!(p.fog, Some(0x00FF_0000));
        assert_eq!(p.vertices[0].fog, 0.5);
        assert!(
            p.combiner.is_none(),
            "the combiner program is set by the control register"
        );
    }

    #[test]
    fn the_fog_equations_take_the_bias_back_out() {
        // Linear fog from 0 to 100, as Direct3D sends it: 1 + 100 / 100 and -1 / 100.
        assert!((fog_factor(FOG_LINEAR, 2.0, -0.01, 50.0) - 0.5).abs() < 1e-6);
        assert_eq!(fog_factor(FOG_LINEAR, 2.0, -0.01, 200.0), 0.0);
        assert_eq!(fog_factor(FOG_LINEAR, 2.0, -0.01, f32::INFINITY), 1.0);
        assert_eq!(fog_factor(FOG_LINEAR_ABS, 0.0, -1.0, 0.5), 1.0);
        // EXP with density 1 / 16 at distance 1: 1.5 + 2^-1 - 1.5.
        assert!((fog_factor(FOG_EXP, 1.5, -0.0625, 1.0) - 0.5).abs() < 1e-6);
        // EXP2: 1.5 + 2^(-32 * 1 * 1/32) - 1.5.
        let scale = (1.0f32 / 32.0).sqrt();
        assert!((fog_factor(FOG_EXP2, 1.5, scale, 1.0) - 0.5).abs() < 1e-6);
        assert_eq!(fog_factor(FOG_LINEAR, f32::NAN, 0.0, 1.0), 1.0);
    }

    #[test]
    fn fixed_function_fog_distance_follows_the_generation_mode() {
        let mut state = State::new();
        // Eye = position moved by -10 in z; the fog plane is (0, 0, -1, 0): minus the depth.
        state.apply(0x04AC, (-10.0f32).to_bits());
        for (i, v) in [0.0f32, 0.0, -1.0, 0.0].into_iter().enumerate() {
            state.apply(0x09D0 + 4 * u32::try_from(i).unwrap(), v.to_bits());
        }
        let mut inputs = [[0.0, 0.0, 0.0, 1.0]; 16];
        inputs[4] = [0.0, 0.0, 0.0, 0.25];
        inputs[5] = [7.0, 0.0, 0.0, 1.0];
        let eye = [3.0, 4.0, -10.0, 1.0];
        assert_eq!(state.fog_distance(&inputs, eye), 10.0, "planar by default");
        let mode = |state: &mut State, m| state.apply(FOG_GEN_MODE, m);
        mode(&mut state, FOG_RADIAL);
        assert!((state.fog_distance(&inputs, eye) - 125f32.sqrt()).abs() < 1e-5);
        mode(&mut state, FOG_ABS_PLANAR);
        assert_eq!(state.fog_distance(&inputs, [0.0, 0.0, 10.0, 1.0]), 10.0);
        mode(&mut state, FOG_FROM_SPECULAR_ALPHA);
        assert_eq!(state.fog_distance(&inputs, eye), 0.25);
        mode(&mut state, FOG_FROM_COORDINATE);
        assert_eq!(state.fog_distance(&inputs, eye), 7.0);
        // Through a whole draw: the model-view matrix puts the vertex at depth 10.
        let mut commands = vec![
            kelvin(ARRAY_FORMAT, 3 << 4 | 2),
            kelvin(FOG_ENABLE, 1),
            kelvin(FINAL_COMBINER0, 3 << 24),
            float(FOG_BASE, 2.0),
            float(FOG_SCALE, -0.05),
            float(0x09D8, -1.0),
            float(0x04AC, -10.0),
        ];
        for k in [0u32, 5, 10, 15] {
            commands.push(float(0x0480 + 4 * k, 1.0));
        }
        commands.extend([
            kelvin(SET_BEGIN_END, 1),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            kelvin(SET_BEGIN_END, 0),
        ]);
        assert!((primitives(&commands)[0].vertices[0].fog - 0.5).abs() < 1e-6);
    }

    /// The vertex of the one point `commands` draw: the centre of its square.
    fn point(commands: &[Command]) -> Vertex {
        let p = &primitives(commands)[0];
        assert_eq!(p.kind, Kind::Quads);
        let (a, c) = (p.vertices[0], p.vertices[2]);
        Vertex {
            x: f32::midpoint(a.x, c.x),
            y: f32::midpoint(a.y, c.y),
            ..a
        }
    }

    /// A point at the origin with a position of three floats, after `state`.
    fn point_after(state: &[Command]) -> Vertex {
        let mut commands = vec![kelvin(ARRAY_FORMAT, 3 << 4 | 2)];
        commands.extend_from_slice(state);
        commands.extend([
            kelvin(SET_BEGIN_END, 1),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            float(INLINE_ARRAY, 0.0),
            kelvin(SET_BEGIN_END, 0),
        ]);
        point(&commands)
    }

    #[test]
    fn the_fixed_function_lights_the_vertex() {
        // Lighting on, light 0 infinite along the normal (0, 0, 1) given inline, red diffuse.
        let mut state = vec![kelvin(0x0314, 1), kelvin(0x03BC, 1)];
        state.extend(
            (0u32..)
                .zip([0.0f32, 0.0, 1.0])
                .map(|(i, v)| float(0x1034 + 4 * i, v)),
        );
        state.extend(
            (0u32..)
                .zip([1.0f32, 0.0, 0.0])
                .map(|(i, v)| float(0x100C + 4 * i, v)),
        );
        state.extend(
            (0u32..)
                .zip([0.0f32, 0.0, 1.0, 1.0])
                .map(|(i, v)| float(VERTEX_DATA4F + 32 + 4 * i, v)),
        );
        // The inverse model-view matrix, which takes the normal to the eye, is the identity.
        state.extend([0u32, 5, 10, 15].map(|k| float(0x0580 + 4 * k, 1.0)));
        let v = point_after(&state);
        assert_eq!(v.colour[..3], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn inline_values_feed_the_attributes_the_vertices_lack() {
        // Without a diffuse colour the fixed function draws white; an inline one replaces it.
        assert_eq!(point_after(&[]).colour, [1.0; 4]);
        let v = point_after(&[kelvin(VERTEX_DATA4UB + 12, 0xFF00_00FF)]);
        assert_eq!(v.colour, [1.0, 0.0, 0.0, 1.0]);
        // The two-float form: attribute 9's s and t, r and q 0 and 1.
        let v = point_after(&[
            float(VERTEX_DATA2F + 72, 0.25),
            float(VERTEX_DATA2F + 76, 0.75),
        ]);
        assert_eq!(v.texture[0], [0.25, 0.75, 0.0, 1.0]);
        // Four shorts, two words for each attribute: attribute 10 is (2, 4).
        let v = point_after(&[kelvin(VERTEX_DATA4S + 80, 0x0004_0002)]);
        assert_eq!(v.texture[1], [2.0, 4.0, 0.0, 1.0]);
    }

    #[test]
    fn a_draw_leaves_its_attributes_at_the_last_vertex() {
        let mut values = InlineValues::default();
        let layout = layout(|i| match i {
            0 => 3 << 4 | 2,
            3 => 4 << 4,
            _ => 2,
        });
        let words = [0, 0, 0, 0x8011_2233, 0, 0, 0, 0x4044_5566];
        values.carry(&layout, &words);
        // The packed colour is kept blue first.
        let b = |v: u8| f32::from(v) / 255.0;
        assert_eq!(values.values[3], [b(0x66), b(0x55), b(0x44), b(0x40)]);
        assert_eq!(values.values[0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(values.set, 1 | 1 << 3);
    }

    #[test]
    fn the_shadow_replays_set_and_carried_values() {
        let mut shadow = InlineShadow::default();
        for (method, value) in [
            (VERTEX_DATA4F + 16 * 9 + 4, 2.0f32.to_bits()),
            (ARRAY_FORMAT + 12, 4 << 4),
            (SET_BEGIN_END, 1),
            (INLINE_ARRAY, 0x0000_00FF),
            (INLINE_ARRAY, 0xFF00_0000),
            (SET_BEGIN_END, 0),
        ] {
            shadow.take(method, value);
        }
        let replay = shadow.replay();
        let words: Vec<(u32, f32)> = replay
            .iter()
            .map(|c| (c.method, f32::from_bits(c.value)))
            .collect();
        assert_eq!(
            words,
            [
                (VERTEX_DATA4F + 48, 0.0),
                (VERTEX_DATA4F + 52, 0.0),
                (VERTEX_DATA4F + 56, 0.0),
                (VERTEX_DATA4F + 60, 1.0),
                (VERTEX_DATA4F + 144, 0.0),
                (VERTEX_DATA4F + 148, 2.0),
                (VERTEX_DATA4F + 152, 0.0),
                (VERTEX_DATA4F + 156, 1.0),
            ]
        );
        // Replayed into a fresh decoder, they are the values.
        let mut state = State::new();
        for c in &replay {
            state.apply(c.method, c.value);
        }
        assert_eq!(state.inline.values[9], [0.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn texture_coordinates_are_generated() {
        // Stage 0: S eye linear on (1, 0, 0, 0.5); the vertex's own q is 2.
        let mut state = vec![
            kelvin(0x03C0, 0x2400),
            float(0x0840, 1.0),
            float(0x084C, 0.5),
        ];
        state.extend([0u32, 5, 10, 15].map(|k| float(0x0480 + 4 * k, 1.0)));
        state.extend(
            (0u32..)
                .zip([0.0f32, 3.0, 0.0, 2.0])
                .map(|(i, v)| float(VERTEX_DATA4F + 16 * 9 + 4 * i, v)),
        );
        // The coordinates go on as they are: the sampler divides by q.
        assert_eq!(point_after(&state).texture[0], [0.5, 3.0, 0.0, 2.0]);
    }

    #[test]
    fn a_vertex_program_places_the_vertex() {
        let mut commands = vec![
            kelvin(ARRAY_FORMAT, 3 << 4 | 2),
            kelvin(TRANSFORM_MODE, 2),
            kelvin(PROGRAM_LOAD, 0),
            kelvin(PROGRAM_START, 0),
        ];
        // MOV o0, v0 (final).
        let insn = [
            0u32,
            1 << 21 | 0b00_01_10_11,
            2 << 26 | 0b00_01_10_11 << 17 | 0b00_01_10_11 << 2,
            15 << 12 | 1 << 11 | 1,
        ];
        commands.extend((0u32..).zip(insn).map(|(i, w)| kelvin(0xB00 + 4 * i, w)));
        commands.extend([
            kelvin(SET_BEGIN_END, 1),
            float(INLINE_ARRAY, 7.0),
            float(INLINE_ARRAY, 8.0),
            float(INLINE_ARRAY, 0.25),
            kelvin(SET_BEGIN_END, 0),
        ]);
        let v = point(&commands);
        assert_eq!(
            (v.x, v.y, v.depth, v.w),
            (7.0, 8.0, 0.25 / 16_777_215.0, 1.0)
        );
    }

    #[test]
    fn a_programs_zero_or_infinite_w_is_taken_as_one() {
        // MOV oPos, v0 (final), with v0 carrying w.
        let insn = [
            0u32,
            1 << 21 | 0b00_01_10_11,
            2 << 26 | 0b00_01_10_11 << 17 | 0b00_01_10_11 << 2,
            15 << 12 | 1 << 11 | 1,
        ];
        for (w, expected) in [(0.0f32, 1.0), (f32::INFINITY, 1.0), (2.0, 2.0)] {
            let mut commands = vec![
                kelvin(ARRAY_FORMAT, 4 << 4 | 2),
                kelvin(TRANSFORM_MODE, 2),
                kelvin(PROGRAM_LOAD, 0),
            ];
            commands.extend(
                (0u32..)
                    .zip(insn)
                    .map(|(i, word)| kelvin(0xB00 + 4 * i, word)),
            );
            commands.extend([
                kelvin(SET_BEGIN_END, 1),
                float(INLINE_ARRAY, 100.0),
                float(INLINE_ARRAY, 50.0),
                float(INLINE_ARRAY, 0.5),
                float(INLINE_ARRAY, w),
                kelvin(SET_BEGIN_END, 0),
            ]);
            let v = point(&commands);
            assert_eq!((v.x, v.y), (100.0, 50.0));
            assert_eq!(v.w, expected);
        }
    }

    #[test]
    fn a_program_reads_the_viewport_the_title_set() {
        // MUL oPos, v0, c58 (the viewport scale), final.
        let insn = [
            0u32,
            2 << 21 | 58 << 13 | 0b00_01_10_11,
            2 << 26 | 0b00_01_10_11 << 17 | 3 << 11 | 0b00_01_10_11 << 2,
            15 << 12 | 1 << 11 | 1,
        ];
        let mut commands = vec![
            kelvin(ARRAY_FORMAT, 3 << 4 | 2),
            kelvin(TRANSFORM_MODE, 2),
            kelvin(PROGRAM_LOAD, 0),
            float(0x0AF0, 320.0),
            float(0x0AF4, -240.0),
            float(0x0AF8, 1.0),
            float(0x0AFC, 1.0),
        ];
        commands.extend((0u32..).zip(insn).map(|(i, w)| kelvin(0xB00 + 4 * i, w)));
        commands.extend([
            kelvin(SET_BEGIN_END, 1),
            float(INLINE_ARRAY, 0.5),
            float(INLINE_ARRAY, 0.5),
            float(INLINE_ARRAY, 0.25),
            kelvin(SET_BEGIN_END, 0),
        ]);
        let v = point(&commands);
        assert_eq!((v.x, v.y, v.depth), (160.0, -120.0, 0.25 / 16_777_215.0));
    }

    #[test]
    fn programmed_draws_keep_their_program_and_constants_and_their_place() {
        // MUL oPos, v0, c58 (the viewport scale), final; and the same as a MOV.
        let mul = [
            0u32,
            2 << 21 | 58 << 13 | 0b00_01_10_11,
            2 << 26 | 0b00_01_10_11 << 17 | 3 << 11 | 0b00_01_10_11 << 2,
            15 << 12 | 1 << 11 | 1,
        ];
        let program = |insn: [u32; 4]| {
            let mut commands = vec![kelvin(PROGRAM_LOAD, 0)];
            commands.extend((0u32..).zip(insn).map(|(i, w)| kelvin(0xB00 + 4 * i, w)));
            commands
        };
        let scale = |s: f32| (0..4).map(move |i| float(0x0AF0 + 4 * i, s));
        // Enough vertices that they run on several threads.
        let triangles = |commands: &mut Vec<Command>| {
            commands.push(kelvin(SET_BEGIN_END, 5));
            for i in 0..1500u16 {
                let x = f32::from(i);
                commands.extend([x, x, 0.5].map(|v| float(INLINE_ARRAY, v)));
            }
            commands.push(kelvin(SET_BEGIN_END, 0));
        };
        let mut commands = vec![kelvin(ARRAY_FORMAT, 3 << 4 | 2), kelvin(TRANSFORM_MODE, 2)];
        commands.extend(program(mul));
        commands.extend(scale(2.0));
        triangles(&mut commands);
        commands.extend(scale(3.0));
        triangles(&mut commands);
        // A fixed-function draw between them, then the program changed to a MOV.
        commands.push(kelvin(TRANSFORM_MODE, 0));
        triangles(&mut commands);
        commands.push(kelvin(TRANSFORM_MODE, 2));
        let mut mov = mul;
        mov[1] = 1 << 21 | 0b00_01_10_11;
        commands.extend(program(mov));
        triangles(&mut commands);
        let out = primitives(&commands);
        assert_eq!(out.len(), 4);
        let x = |p: usize, v: usize| out[p].vertices[v].x;
        assert_eq!((x(0, 1), x(0, 1499)), (2.0, 2998.0));
        assert_eq!((x(1, 1), x(1, 1499)), (3.0, 4497.0));
        assert_eq!((x(2, 1), x(2, 1499)), (1.0, 1499.0), "the fixed function");
        assert_eq!((x(3, 1), x(3, 1499)), (1.0, 1499.0));
    }

    /// The primitives of a draw of `begin` over vertices at `xs` (y = x), after `state`.
    fn drawn(begin: u32, xs: &[f32], state: &[Command]) -> Vec<Primitive> {
        let mut commands = vec![kelvin(ARRAY_FORMAT, 2 << 4 | 2)];
        commands.extend_from_slice(state);
        commands.push(kelvin(SET_BEGIN_END, begin));
        for &x in xs {
            commands.extend([float(INLINE_ARRAY, x), float(INLINE_ARRAY, x)]);
        }
        commands.push(kelvin(SET_BEGIN_END, 0));
        primitives(&commands)
    }

    fn xs(p: &Primitive) -> Vec<f32> {
        p.vertices.iter().map(|v| v.x).collect()
    }

    #[test]
    fn every_begin_kind_is_assembled() {
        let strip = &drawn(4, &[0.0, 1.0, 2.0], &[])[0];
        assert_eq!(
            (strip.kind, xs(strip)),
            (Kind::Lines, vec![0.0, 1.0, 1.0, 2.0])
        );
        let looped = &drawn(3, &[0.0, 1.0, 2.0], &[])[0];
        assert_eq!(xs(looped), [0.0, 1.0, 1.0, 2.0, 2.0, 0.0]);
        let quads = &drawn(9, &[0.0, 1.0, 2.0, 3.0, 4.0], &[])[0];
        assert_eq!(
            (quads.kind, xs(quads)),
            (Kind::TriangleStrip, vec![0.0, 1.0, 2.0, 3.0])
        );
        let polygon = &drawn(10, &[0.0, 1.0, 2.0, 3.0], &[])[0];
        assert_eq!(polygon.kind, Kind::TriangleFan);
        assert_eq!(drawn(11, &[0.0; 3], &[]).len(), 0, "not a kind");
    }

    #[test]
    fn points_are_squares_of_their_size() {
        let p = &drawn(1, &[10.0], &[kelvin(POINT_SIZE, 4 * 8)])[0];
        assert_eq!(p.kind, Kind::Quads);
        let corners: Vec<(f32, f32)> = p.vertices.iter().map(|v| (v.x, v.y)).collect();
        assert_eq!(
            corners,
            [(8.0, 8.0), (12.0, 8.0), (12.0, 12.0), (8.0, 12.0)]
        );
        // At least a pixel.
        let p = &drawn(1, &[10.0], &[kelvin(POINT_SIZE, 1)])[0];
        assert_eq!(p.vertices[1].x - p.vertices[0].x, 1.0);
    }

    #[test]
    fn point_parameters_attenuate_fixed_function_points() {
        let mut state = State::new();
        state.apply(POINT_PARAMS_ENABLE, 1);
        // a = 0, b = 0, c = 1 (1 / d), scale 8, bias 0, smallest 1.
        for (i, v) in [0.0f32, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0]
            .into_iter()
            .enumerate()
        {
            state.apply(POINT_PARAMS + 4 * u32::try_from(i).unwrap(), v.to_bits());
        }
        // At distance 2: (0.5 + 0) * 8 + 1 = 5, inside 1..9.
        assert_eq!(state.fixed_point_size([0.0, 0.0, -2.0, 1.0]), 5.0);
        // Near the eye it stops at scale + smallest.
        assert_eq!(state.fixed_point_size([0.0, 0.0, -0.1, 1.0]), 9.0);
        state.apply(POINT_PARAMS_ENABLE, 0);
        state.apply(POINT_SIZE, 3 * 8);
        assert_eq!(state.fixed_point_size([0.0; 4]), 3.0);
    }

    #[test]
    fn a_program_sizes_its_points_with_opts_when_the_parameters_are_on() {
        // MOV oPos, v0; MOV oPts, v0 (x is 6), final.
        let mov = |out: u32, last: u32| {
            [
                0u32,
                1 << 21 | 0b00_01_10_11,
                2 << 26 | 0b00_01_10_11 << 17 | 0b00_01_10_11 << 2,
                15 << 12 | 1 << 11 | out << 3 | last,
            ]
        };
        let mut state = vec![
            kelvin(TRANSFORM_MODE, 2),
            kelvin(PROGRAM_LOAD, 0),
            kelvin(POINT_PARAMS_ENABLE, 1),
        ];
        let words = mov(0, 0).into_iter().chain(mov(6, 1));
        state.extend((0u32..).zip(words).map(|(i, w)| kelvin(0xB00 + 4 * i, w)));
        let p = &drawn(1, &[6.0], &state)[0];
        assert_eq!(p.vertices[1].x - p.vertices[0].x, 6.0);
    }

    #[test]
    fn wide_lines_are_quads_along_them() {
        let mut commands = vec![
            kelvin(ARRAY_FORMAT, 2 << 4 | 2),
            kelvin(LINE_WIDTH, 4 * 8),
            kelvin(SET_BEGIN_END, 2),
        ];
        for (x, y) in [(0.0f32, 5.0f32), (10.0, 5.0)] {
            commands.extend([float(INLINE_ARRAY, x), float(INLINE_ARRAY, y)]);
        }
        commands.push(kelvin(SET_BEGIN_END, 0));
        let p = &primitives(&commands)[0];
        assert_eq!(p.kind, Kind::Quads);
        let corners: Vec<(f32, f32)> = p.vertices.iter().map(|v| (v.x, v.y)).collect();
        assert_eq!(corners, [(0.0, 7.0), (10.0, 7.0), (10.0, 3.0), (0.0, 3.0)]);
        assert!(
            p.vertices
                .iter()
                .all(|v| (v.depth + LINE_PULL).abs() < 1e-6)
        );
        // A line of no length draws nothing.
        assert_eq!(wide_line(p.vertices[0], p.vertices[0], 4.0), []);
    }

    #[test]
    fn bound_lists_each_texture_once_in_first_bound_order() {
        let tex = |offset| {
            [
                kelvin(STAGE_BASE, offset),
                kelvin(STAGE_BASE + 4, 6 << 8 | 1 << 16),
            ]
        };
        let mut commands: Vec<Command> = tex(0x1000).to_vec();
        commands.extend([kelvin(SET_BEGIN_END, 5), kelvin(SET_BEGIN_END, 0)]);
        commands.extend(tex(0x2000));
        commands.extend([kelvin(SET_BEGIN_END, 5), kelvin(SET_BEGIN_END, 0)]);
        commands.extend(tex(0x1000));
        commands.extend([kelvin(SET_BEGIN_END, 5), kelvin(SET_BEGIN_END, 0)]);
        let offsets: Vec<u32> = bound(&commands).iter().map(|t| t.offset).collect();
        assert_eq!(offsets, [0x1000, 0x2000]);
        // Stage 1 starts off; its control register turns it on.
        let mut stage1 = vec![
            kelvin(STAGE_BASE + 0x40, 0x3000),
            kelvin(STAGE_BASE + 0x44, 6 << 8 | 1 << 16),
        ];
        stage1.extend([kelvin(SET_BEGIN_END, 5), kelvin(SET_BEGIN_END, 0)]);
        assert!(bound(&stage1).iter().all(|t| t.offset != 0x3000));
        stage1.insert(0, kelvin(STAGE_BASE + 0x4C, 1 << 30));
        assert!(bound(&stage1).iter().any(|t| t.offset == 0x3000));
    }

    #[test]
    fn attribute_types() {
        assert_eq!(attribute(0, 4, &[0xFF00_FF00]), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(attribute(4, 2, &[0x0000_FF00]), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            attribute(5, 3, &[0xFFFF_0002, 0x0000_0005]),
            [2.0, -1.0, 5.0, 1.0]
        );
        assert_eq!(
            attribute(2, 2, &[1.5f32.to_bits(), 2.5f32.to_bits()]),
            [1.5, 2.5, 0.0, 1.0]
        );
        // Normalized shorts: 32767 is 1, -32768 stops at -1.
        assert_eq!(attribute(1, 2, &[0x8000_7FFF]), [1.0, -1.0, 0.0, 1.0]);
        // Packed normals: (0, 1, 0), and (-1, 0, -1).
        assert_eq!(attribute(6, 1, &[0x001F_F800]), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(attribute(6, 1, &[0x8040_0401]), [-1.0, 0.0, -1.0, 1.0]);
    }

    /// A triangle of three vertices at the origin after `state`, position only.
    fn triangle_after(state: &[Command]) -> Primitive {
        let mut commands = vec![kelvin(ARRAY_FORMAT, 3 << 4 | 2)];
        commands.extend_from_slice(state);
        commands.push(kelvin(SET_BEGIN_END, 5));
        commands.extend((0..9).map(|_| float(INLINE_ARRAY, 0.0)));
        commands.push(kelvin(SET_BEGIN_END, 0));
        primitives(&commands).remove(0)
    }

    /// Stage `stage` on, with a 2x2 A8R8G8B8 texture at `0x1000 * (stage + 1)`.
    fn bound_on(stage: u32) -> [Command; 3] {
        let base = STAGE_BASE + STAGE_STRIDE * stage;
        [
            kelvin(base, 0x1000 * (stage + 1)),
            kelvin(base + 4, 6 << 8 | 1 << 16 | 1 << 20 | 1 << 24),
            kelvin(base + 0x0C, 1 << 30),
        ]
    }

    #[test]
    fn stages_2_and_3_carry_their_textures_modes_and_the_texture_shader() {
        let mut state: Vec<Command> = (0..4).flat_map(bound_on).collect();
        // PROJECT2D, DOT_PRODUCT, DOT_ST and DEPENDENT_GB.
        state.push(kelvin(
            SHADER_STAGE_PROGRAM,
            1 | 0x11 << 5 | 9 << 10 | 0x10 << 15,
        ));
        state.push(kelvin(crate::gpu::texture::DOT_RGBMAPPING, 0x321));
        // Stage 3 reads stage 2's texel.
        state.push(kelvin(
            crate::gpu::texture::SHADER_OTHER_STAGE_INPUT,
            2 << 20,
        ));
        state.push(float(STAGE_BASE + STAGE_STRIDE + 0x38, 0.5));
        state.push(kelvin(SURFACE_COLOR_OFFSET, 0x0040_0000));
        let p = triangle_after(&state);
        let offsets = p.textures.map(|t| t.map(|t| t.offset));
        // Stage 1's dot product reads no texture of its own.
        assert_eq!(offsets, [Some(0x1000), None, Some(0x3000), Some(0x4000)]);
        assert_eq!(
            p.modes,
            [
                Mode::Sample,
                Mode::DotProduct,
                Mode::DotSt,
                Mode::DependentGb
            ]
        );
        assert_eq!((p.shader.mapping(3), p.shader.input(3)), (3, 2));
        assert_eq!(p.shader.bump_scale[1], 0.5);
        assert_eq!(p.surface, 0x0040_0000);
        // The frame's bound textures leave stage 1's out too.
        let mut commands = state.clone();
        commands.extend([kelvin(SET_BEGIN_END, 5), kelvin(SET_BEGIN_END, 0)]);
        assert_eq!(bound(&commands).len(), 3);
        // Until the program is written, stages 2 and 3 read their texture only while on.
        let unset = triangle_after(&bound_on(2));
        assert_eq!(
            unset.modes,
            [Mode::Sample, Mode::Sample, Mode::Sample, Mode::Off]
        );
        assert_eq!(unset.textures[2].map(|t| t.offset), Some(0x3000));
    }

    #[test]
    fn every_stage_takes_four_texture_coordinates() {
        // Position, then attributes 11 and 12 with four floats each.
        let mut commands = vec![
            kelvin(ARRAY_FORMAT, 3 << 4 | 2),
            kelvin(ARRAY_FORMAT + 44, 4 << 4 | 2),
            kelvin(ARRAY_FORMAT + 48, 4 << 4 | 2),
            kelvin(SET_BEGIN_END, 5),
        ];
        let words = [0.0f32, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for _ in 0..3 {
            commands.extend(words.map(|w| float(INLINE_ARRAY, w)));
        }
        commands.push(kelvin(SET_BEGIN_END, 0));
        let v = primitives(&commands)[0].vertices[0];
        assert_eq!(v.texture[0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(v.texture[2], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.texture[3], [5.0, 6.0, 7.0, 8.0]);
    }
}
