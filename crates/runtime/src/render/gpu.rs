//! The wgpu side of the renderer (spec 11.1, 11.3): pipelines, textures and samplers, and the
//! render targets, one for each colour buffer a frame shows or samples, sized as its surface,
//! each with a depth and stencil buffer.

use std::hash::{Hash, Hasher};

/// The renderer's maps, looked up for every primitive and batch of a frame: a fast hash, as
/// their keys are small and no one chooses them to collide.
type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<Fast>>;

/// A multiply-and-rotate hash of words.
#[derive(Debug, Default, Clone, Copy)]
struct Fast(u64);

impl Fast {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
}

impl Hasher for Fast {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        let mut last = [0; 8];
        last[..rest.len()].copy_from_slice(rest);
        self.add(u64::from_le_bytes(last));
    }

    fn write_u8(&mut self, n: u8) {
        self.add(u64::from(n));
    }

    fn write_u16(&mut self, n: u16) {
        self.add(u64::from(n));
    }

    fn write_u32(&mut self, n: u32) {
        self.add(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
}

use super::batch::{Batch, Blend, Clip, GpuVertex};
use super::combiner::Params;
use crate::gpu::Command;
use crate::gpu::draw::{self, Primitive};
use crate::gpu::pixel::{Cull, Stencil, Surface};
use crate::gpu::texture::{Shape, Texture};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// A depth buffer with a stencil, as the console's Z24S8.
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;
/// The largest side of a target.
const MAX_SIDE: u32 = 8192;
/// A texture unused for this many frames leaves the cache.
const KEEP_FRAMES: u64 = 120;
/// A texture leaves [`RESIDENT`] this many frames before it leaves the cache, so a frame sent
/// without its bytes finds it still there.
const RESIDENT_GRACE: u64 = 10;
/// A target, depth buffer, copy or bind group unused for this many frames is let go.
const KEEP_TARGETS: u64 = 30;
/// Textures kept spare, for new contents of their shape, after they leave the cache.
const SPARE_TEXTURES: usize = 32;

/// The content ids of the textures the renderer holds: the kernel sends a frame's textures
/// without their bytes when they are here.
pub static RESIDENT: std::sync::Mutex<std::collections::BTreeSet<u64>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

fn resident() -> std::sync::MutexGuard<'static, std::collections::BTreeSet<u64>> {
    RESIDENT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A bound texture, its content id, and its bytes from guest memory; the bytes are empty when
/// the renderer already holds the texture.
#[derive(Debug, Clone)]
pub struct TextureData {
    pub texture: Texture,
    pub id: u64,
    pub bytes: Vec<u8>,
    pub palette: Vec<u8>,
}

impl TextureData {
    /// A texture with its bytes, its content id worked out from them.
    #[must_use]
    pub fn new(texture: Texture, bytes: Vec<u8>, palette: Vec<u8>) -> Self {
        let seed = identity_seed(&texture);
        let id = crate::mem::fingerprint(&palette, crate::mem::fingerprint(&bytes, seed));
        Self {
            texture,
            id,
            bytes,
            palette,
        }
    }
}

/// The seed of a texture's content id: its identity, so the same bytes read another way are
/// another texture. A content id is then the fingerprint of the bytes and the palette from it.
#[must_use]
pub fn identity_seed(texture: &Texture) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    identity(texture).hash(&mut hasher);
    hasher.finish()
}

/// A texture's identity: the fields that say where and what it is (spec 10.9).
type Identity = (u32, u32, u32, u32, u32, u32, u32, u32, bool);

fn identity(t: &Texture) -> Identity {
    (
        t.offset, t.code, t.width, t.height, t.levels, t.pitch, t.palette, t.depth, t.cube,
    )
}

/// What a pipeline depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    depth: Option<(u32, bool)>,
    /// Source and destination factors and the equation.
    blend: Option<(u32, u32, u32)>,
    lines: bool,
    colour_writes: u8,
    cull: Option<Cull>,
    /// The stencil test, its reference left out: that is set per batch.
    stencil: Option<Stencil>,
    w_buffer: bool,
}

/// A sampler's state (spec 11.3), hashable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SamplerKey {
    address: u32,
    filter: u32,
    border: u32,
    control0: u32,
    levels: u32,
}

/// `0x200 + n` as a comparison.
fn comparison(function: u32) -> wgpu::CompareFunction {
    use wgpu::CompareFunction::{
        Always, Equal, Greater, GreaterEqual, Less, LessEqual, Never, NotEqual,
    };
    match function.wrapping_sub(0x200) {
        0 => Never,
        1 => Less,
        2 => Equal,
        3 => LessEqual,
        4 => Greater,
        5 => NotEqual,
        6 => GreaterEqual,
        _ => Always,
    }
}

/// A GL blend factor. wgpu has no constant-alpha factor: the constant alpha ones use the blend
/// constant, which [`blend_constant`] then makes the alpha replicated.
fn factor(value: u32) -> wgpu::BlendFactor {
    use wgpu::BlendFactor::{
        Constant, Dst, DstAlpha, One, OneMinusConstant, OneMinusDst, OneMinusDstAlpha, OneMinusSrc,
        OneMinusSrcAlpha, Src, SrcAlpha, SrcAlphaSaturated, Zero,
    };
    match value {
        0 => Zero,
        0x300 => Src,
        0x301 => OneMinusSrc,
        0x302 => SrcAlpha,
        0x303 => OneMinusSrcAlpha,
        0x304 => DstAlpha,
        0x305 => OneMinusDstAlpha,
        0x306 => Dst,
        0x307 => OneMinusDst,
        0x308 => SrcAlphaSaturated,
        0x8001 | 0x8003 => Constant,
        0x8002 | 0x8004 => OneMinusConstant,
        _ => One,
    }
}

/// A blend equation. The signed add and signed reverse subtract are drawn as the unsigned ones.
fn equation(value: u32) -> wgpu::BlendOperation {
    match value {
        0x800A => wgpu::BlendOperation::Subtract,
        0x800B | 0xF005 => wgpu::BlendOperation::ReverseSubtract,
        0x8007 => wgpu::BlendOperation::Min,
        0x8008 => wgpu::BlendOperation::Max,
        _ => wgpu::BlendOperation::Add,
    }
}

/// The blend constant a blend needs, if a factor uses it: the constant colour, or when only the
/// constant alpha factors are used, the alpha in every channel.
fn blend_constant(blend: &Blend) -> Option<wgpu::Color> {
    let factors = [blend.source, blend.destination];
    let colour = factors.iter().any(|f| matches!(f, 0x8001 | 0x8002));
    let alpha = factors.iter().any(|f| matches!(f, 0x8003 | 0x8004));
    let [r, g, b, a] = super::combiner::colour(blend.constant).map(f64::from);
    match (colour, alpha) {
        (false, false) => None,
        (false, true) => Some(wgpu::Color {
            r: a,
            g: a,
            b: a,
            a,
        }),
        (true, _) => Some(wgpu::Color { r, g, b, a }),
    }
}

/// A `SET_STENCIL_OP_*` value; any other keeps the stencil.
fn stencil_operation(op: u32) -> wgpu::StencilOperation {
    use wgpu::StencilOperation::{
        DecrementClamp, DecrementWrap, IncrementClamp, IncrementWrap, Invert, Keep, Replace, Zero,
    };
    match op {
        0 => Zero,
        0x1E01 => Replace,
        0x1E02 => IncrementClamp,
        0x1E03 => DecrementClamp,
        0x150A => Invert,
        0x8507 => IncrementWrap,
        0x8508 => DecrementWrap,
        _ => Keep,
    }
}

/// The size of the target a surface is drawn into: its own, within what the renderer draws.
fn target_size(surface: &Surface) -> (u32, u32) {
    let [width, height] = surface.size.map(|side| side.clamp(1, MAX_SIDE));
    (width, height)
}

/// A drawn picture of `width` by `height` pixels, RGBA8 row by row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// What the display shows of a drawing `width` by `height` of an anti-aliased surface with
/// `samples` across and down: the samples averaged down to the pixels the title addresses.
#[must_use]
pub fn downsample(pixels: Vec<u8>, width: u32, height: u32, samples: [u32; 2]) -> Image {
    let [sx, sy] = samples.map(|s| s.max(1));
    if sx == 1 && sy == 1 {
        return Image {
            width,
            height,
            rgba: pixels,
        };
    }
    let (out_width, out_height) = ((width / sx).max(1), (height / sy).max(1));
    let (width, height, sx, sy) = (width as usize, height as usize, sx as usize, sy as usize);
    let mut rgba = Vec::with_capacity(out_width as usize * out_height as usize * 4);
    for y in 0..out_height as usize {
        for x in 0..out_width as usize {
            let mut sum = [0usize; 4];
            for dy in 0..sy {
                for dx in 0..sx {
                    let row = (y * sy + dy).min(height - 1);
                    let at = (row * width + (x * sx + dx).min(width - 1)) * 4;
                    for (total, byte) in sum.iter_mut().zip(&pixels[at..at + 4]) {
                        *total += usize::from(*byte);
                    }
                }
            }
            let n = sx * sy;
            rgba.extend(sum.map(|total| u8::try_from((total + n / 2) / n).unwrap_or(u8::MAX)));
        }
    }
    Image {
        width: out_width,
        height: out_height,
        rgba,
    }
}

/// A frame's shown target as drawn: its size and anti-aliasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Drawn {
    width: u32,
    height: u32,
    samples: [u32; 2],
}

/// A render target, with the view drawn through, kept from frame to frame while its size holds.
#[derive(Debug, Clone)]
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    serial: u64,
}

impl Target {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = attachment(
            device,
            "target",
            FORMAT,
            (width, height),
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            serial: next_serial(),
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.texture.width(), self.texture.height())
    }

    /// The target as a texture a later pass samples.
    fn sampled(&self) -> Uploaded {
        Uploaded {
            texture: self.texture.clone(),
            view: self.view.clone(),
            shape: Shape::Flat,
            serial: self.serial,
        }
    }
}

/// A new serial for a texture view: a bind group made with a view is found again only by the
/// serial of that view.
fn next_serial() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// What a batch's texture bind group is made of: each stage's view by serial (0 for white) and
/// its sampler (`None` for the default).
type BindKey = [(u64, Option<SamplerKey>); 4];

/// A batch ready to draw: its pipeline, bind groups, vertices, and the state the pass sets.
struct Prepared {
    key: PipelineKey,
    bind: BindKey,
    /// The offset of its uniform block in the frame's uniform buffer.
    uniform: u32,
    /// Its vertices in the frame's vertex buffer.
    vertices: std::ops::Range<u32>,
    clip: Clip,
    viewport: (f32, f32),
    stencil_reference: Option<u32>,
    blend_constant: Option<wgpu::Color>,
    /// The texture the target as drawn so far is copied into just before this batch, for the
    /// stages that sample the target.
    feedback: Option<Uploaded>,
}

/// A pass of a frame: the target it draws into, its batches, and the textures that read the
/// target once it is drawn, each a copy of its top-left part (none when the texture is the
/// target itself).
struct Plan {
    target: Target,
    batches: Vec<Batch>,
    crops: Vec<Uploaded>,
}

/// What a render pass was last given, to give it again only what changes.
#[derive(Default)]
struct SetState {
    scissor: Option<[u32; 4]>,
    viewport: Option<(u32, u32)>,
    stencil_reference: Option<u32>,
    blend_constant: Option<[u64; 4]>,
    pipeline: Option<PipelineKey>,
    bind: Option<BindKey>,
    uniform: Option<u32>,
}

/// Fills the next uniform slot, or shares the one before when the parameters are the same.
struct Uniforms<'a> {
    bytes: &'a mut Vec<u8>,
    slot: usize,
    last: Option<Params>,
}

impl Uniforms<'_> {
    /// The offset of the slot that holds the uniform block of `params`.
    fn offset_of(&mut self, params: &Params) -> u32 {
        if self.last.as_ref() != Some(params) {
            let start = self.bytes.len().next_multiple_of(self.slot);
            self.bytes.resize(start, 0);
            self.bytes.extend_from_slice(&params.bytes());
            self.last = Some(*params);
        }
        let start = self.bytes.len() - super::combiner::PARAMS_SIZE;
        u32::try_from(start).unwrap_or(u32::MAX)
    }
}

/// The renderer.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    /// The textures and samplers (group 0) and the uniform block (group 1, at an offset).
    layout: wgpu::BindGroupLayout,
    uniform_layout: wgpu::BindGroupLayout,
    /// The distance between two uniform blocks in the uniform buffer.
    uniform_slot: usize,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    samplers: HashMap<SamplerKey, wgpu::Sampler>,
    /// Texture bind groups, with the frame they were last used in.
    binds: HashMap<BindKey, (wgpu::BindGroup, u64)>,
    /// The frame's vertices, kept for the next frame while it is big enough.
    vertices: Option<wgpu::Buffer>,
    /// The frame's uniform blocks and their bind group, kept the same way.
    uniforms: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    /// The bytes of the frame's uniform blocks, kept for their capacity.
    uniform_bytes: Vec<u8>,
    /// Uploaded textures by content id, with the frame they were last used in.
    textures: HashMap<u64, (Uploaded, u64)>,
    /// Textures that left the cache, to take new contents of their shape.
    spare: Vec<wgpu::Texture>,
    /// What this frame drew into the colour buffers it samples, by a content id of its own.
    drawn: HashMap<u64, Uploaded>,
    /// The targets of the colour buffers other than the shown one, by surface and size; the
    /// depth and stencil buffers by size; the copies of a target's part that textures read, by
    /// surface and size; the copies of a target as drawn so far, by size. Each with the frame
    /// it was last used in.
    off_screen: HashMap<(u32, u32, u32), (Target, u64)>,
    depths: HashMap<(u32, u32), (wgpu::TextureView, u64)>,
    crops: HashMap<(u32, u32, u32), (Uploaded, u64)>,
    feedbacks: HashMap<(u32, u32), (Uploaded, u64)>,
    /// A white texel, flat, 3D and as a cube map, for the bindings a stage does not use.
    white: wgpu::TextureView,
    white_volume: wgpu::TextureView,
    white_cube: wgpu::TextureView,
    default_sampler: wgpu::Sampler,
    /// The shown surface's target, kept while frames keep its size.
    target: Option<Target>,
    frame: u64,
    clamp_depth: bool,
    border: bool,
    /// Where frames are shown, for a renderer made [`Renderer::with_layer`].
    presenter: Option<super::present::Presenter>,
    /// The anti-aliasing of the frame the presenter's copy back is of.
    snapshot_samples: [u32; 2],
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("frame", &self.frame)
            .field("textures", &self.textures.len())
            .field("pipelines", &self.pipelines.len())
            .finish_non_exhaustive()
    }
}

impl Renderer {
    /// A renderer on the default adapter, or the reason there is none.
    ///
    /// # Errors
    ///
    /// When no adapter or device can be had.
    pub fn new() -> Result<Self, String> {
        // SAFETY: there is no layer.
        unsafe { Self::open(None) }
    }

    /// A renderer that also shows its frames on the Core Animation layer `layer` (a
    /// `CAMetalLayer`), with [`Renderer::present`].
    ///
    /// # Safety
    ///
    /// `layer` must be a valid `CAMetalLayer` that outlives the renderer.
    ///
    /// # Errors
    ///
    /// When no adapter or device can draw to the layer.
    pub unsafe fn with_layer(layer: *mut std::ffi::c_void) -> Result<Self, String> {
        // SAFETY: the caller vouches for the layer.
        unsafe { Self::open(Some(layer)) }
    }

    /// # Safety
    ///
    /// `layer`, if any, is a valid `CAMetalLayer` that outlives the renderer.
    unsafe fn open(layer: Option<*mut std::ffi::c_void>) -> Result<Self, String> {
        // A new renderer holds no textures yet.
        resident().clear();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = match layer {
            // SAFETY: the caller vouches for the layer.
            Some(layer) => Some(
                unsafe {
                    instance
                        .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(layer))
                }
                .map_err(|e| format!("cannot draw to the window: {e}"))?,
            ),
            None => None,
        };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: surface.as_ref(),
            ..wgpu::RequestAdapterOptions::default()
        }))
        .map_err(|e| format!("no graphics adapter: {e}"))?;
        let wanted =
            wgpu::Features::DEPTH_CLIP_CONTROL | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER;
        let features = adapter.features() & wanted;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("anyxbox"),
            required_features: features,
            ..wgpu::DeviceDescriptor::default()
        }))
        .map_err(|e| format!("no graphics device: {e}"))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anyxbox"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let layout = bind_group_layout(&device);
        let uniform_layout = uniform_layout(&device);
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1) as usize;
        let uniform_slot = super::combiner::PARAMS_SIZE.next_multiple_of(alignment);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anyxbox"),
            bind_group_layouts: &[Some(&layout), Some(&uniform_layout)],
            immediate_size: 0,
        });
        let white_level = |layers| crate::gpu::texture::Level {
            width: 1,
            height: 1,
            layers,
            rgba: vec![255; 4 * layers as usize],
        };
        let [white, white_volume, white_cube] =
            [(Shape::Flat, 1), (Shape::Volume, 1), (Shape::Cube, 6)].map(|(shape, layers)| {
                upload(
                    &device,
                    &queue,
                    &[white_level(layers)],
                    shape,
                    &mut Vec::new(),
                )
                .view
            });
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let presenter = match surface {
            Some(surface) => Some(super::present::Presenter::new(
                &device, &queue, surface, adapter,
            )?),
            None => None,
        };
        Ok(Self {
            presenter,
            target: None,
            device,
            queue,
            shader,
            layout,
            uniform_layout,
            uniform_slot,
            pipeline_layout,
            pipelines: HashMap::default(),
            samplers: HashMap::default(),
            binds: HashMap::default(),
            vertices: None,
            uniforms: None,
            uniform_bytes: Vec::new(),
            textures: HashMap::default(),
            spare: Vec::new(),
            drawn: HashMap::default(),
            off_screen: HashMap::default(),
            depths: HashMap::default(),
            crops: HashMap::default(),
            feedbacks: HashMap::default(),
            white,
            white_volume,
            white_cube,
            default_sampler,
            frame: 0,
            clamp_depth: features.contains(wgpu::Features::DEPTH_CLIP_CONTROL),
            border: features.contains(wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER),
            snapshot_samples: [1, 1],
        })
    }

    #[allow(clippy::too_many_lines)]
    fn pipeline(&mut self, key: PipelineKey) {
        if self.pipelines.contains_key(&key) {
            return;
        }
        let attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4];
        let blend = key.blend.map(|(src, dst, op)| {
            let operation = equation(op);
            // Min and max take no factors.
            let weighed = !matches!(
                operation,
                wgpu::BlendOperation::Min | wgpu::BlendOperation::Max
            );
            let weight = |f: u32| {
                if weighed {
                    factor(f)
                } else {
                    wgpu::BlendFactor::One
                }
            };
            let component = wgpu::BlendComponent {
                src_factor: weight(src),
                dst_factor: weight(dst),
                operation,
            };
            wgpu::BlendState {
                color: component,
                alpha: component,
            }
        });
        let (compare, write) = match key.depth {
            Some((function, write)) => (comparison(function), write),
            None => (wgpu::CompareFunction::Always, false),
        };
        let stencil = key
            .stencil
            .map_or_else(wgpu::StencilState::default, |stencil| {
                let face = wgpu::StencilFaceState {
                    compare: comparison(stencil.function),
                    fail_op: stencil_operation(stencil.fail),
                    depth_fail_op: stencil_operation(stencil.depth_fail),
                    pass_op: stencil_operation(stencil.pass),
                };
                wgpu::StencilState {
                    front: face,
                    back: face,
                    read_mask: u32::from(stencil.read_mask),
                    write_mask: u32::from(stencil.write_mask),
                }
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("anyxbox"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    topology: if key.lines {
                        wgpu::PrimitiveTopology::LineList
                    } else {
                        wgpu::PrimitiveTopology::TriangleList
                    },
                    // Clip space keeps the screen's handedness (y only flipped to point up), so
                    // a triangle that winds clockwise on the screen winds clockwise here too.
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: match key.cull {
                        _ if key.lines => None,
                        Some(Cull::Clockwise) => Some(wgpu::Face::Back),
                        Some(Cull::CounterClockwise) => Some(wgpu::Face::Front),
                        _ => None,
                    },
                    // Depth outside the range is clamped by the viewport, or dropped by the
                    // fragment shader: clamp, not clip.
                    unclipped_depth: self.clamp_depth,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(write),
                    depth_compare: Some(compare),
                    stencil,
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some(if key.w_buffer { "fs_w_buffer" } else { "fs" }),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend,
                        write_mask: wgpu::ColorWrites::from_bits_truncate(u32::from(
                            key.colour_writes,
                        )),
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.pipelines.insert(key, pipeline);
    }

    /// The sampler of a texture's stage registers (spec 11.3).
    fn sampler(&mut self, t: &Texture) -> &wgpu::Sampler {
        let key = sampler_key(t);
        let border_supported = self.border;
        self.samplers.entry(key).or_insert_with(|| {
            let border_colour = {
                let [b, g, r, a] = t.border.to_le_bytes();
                if a < 128 {
                    wgpu::SamplerBorderColor::TransparentBlack
                } else if u32::from(r) + u32::from(g) + u32::from(b) >= 384 {
                    wgpu::SamplerBorderColor::OpaqueWhite
                } else {
                    wgpu::SamplerBorderColor::OpaqueBlack
                }
            };
            let mut uses_border = false;
            let mut mode = |shift: u32| match (t.address >> shift) & 7 {
                2 => wgpu::AddressMode::MirrorRepeat,
                4 if border_supported => {
                    uses_border = true;
                    wgpu::AddressMode::ClampToBorder
                }
                3..=5 => wgpu::AddressMode::ClampToEdge,
                _ => wgpu::AddressMode::Repeat,
            };
            let (u, v, w) = (mode(0), mode(8), mode(16));
            let linear = |on: bool| {
                if on {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                }
            };
            let mag = linear((t.filter >> 24) & 0xF != 1);
            let (min, mip, mips) = match (t.filter >> 16) & 0xFF {
                1 => (
                    wgpu::FilterMode::Nearest,
                    wgpu::MipmapFilterMode::Nearest,
                    false,
                ),
                3 => (
                    wgpu::FilterMode::Nearest,
                    wgpu::MipmapFilterMode::Nearest,
                    true,
                ),
                4 => (
                    wgpu::FilterMode::Linear,
                    wgpu::MipmapFilterMode::Nearest,
                    true,
                ),
                5 => (
                    wgpu::FilterMode::Nearest,
                    wgpu::MipmapFilterMode::Linear,
                    true,
                ),
                6 => (
                    wgpu::FilterMode::Linear,
                    wgpu::MipmapFilterMode::Linear,
                    true,
                ),
                _ => (
                    wgpu::FilterMode::Linear,
                    wgpu::MipmapFilterMode::Nearest,
                    false,
                ),
            };
            let top = t.levels.saturating_sub(1) as f32;
            let (lod_min, lod_max) = if mips {
                let min_lod = ((t.control0 >> 18) & 0xFFF) as f32 / 256.0;
                let max_lod = ((t.control0 >> 6) & 0xFFF) as f32 / 256.0;
                let lo = min_lod.min(top);
                (lo, max_lod.min(top).max(lo))
            } else {
                (0.0, 0.0)
            };
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: None,
                address_mode_u: u,
                address_mode_v: v,
                address_mode_w: w,
                mag_filter: mag,
                min_filter: min,
                mipmap_filter: mip,
                lod_min_clamp: lod_min,
                lod_max_clamp: lod_max,
                compare: None,
                anisotropy_clamp: 1,
                border_color: uses_border.then_some(border_colour),
            })
        })
    }

    /// Uploads the frame's textures not yet cached, decoding them; returns their content ids by
    /// identity. A texture that leaves the cache is kept spare, for new contents of its shape.
    fn prepare_textures(&mut self, textures: &[TextureData]) -> HashMap<Identity, u64> {
        let mut ids = HashMap::default();
        let mut resident = resident();
        for data in textures {
            let id = data.id;
            let frame = self.frame;
            if let Some(entry) = self.textures.get_mut(&id) {
                entry.1 = frame;
            } else if let Some(levels) = data.texture.decode(&data.bytes, &data.palette) {
                let uploaded = upload(
                    &self.device,
                    &self.queue,
                    &levels,
                    data.texture.shape(),
                    &mut self.spare,
                );
                self.textures.insert(id, (uploaded, frame));
            } else {
                continue;
            }
            resident.insert(id);
            ids.insert(identity(&data.texture), id);
        }
        let now = self.frame;
        for (id, (_, last)) in &self.textures {
            if frame_age(now, *last) >= KEEP_FRAMES - RESIDENT_GRACE {
                resident.remove(id);
            }
        }
        let spare = &mut self.spare;
        self.textures.retain(|_, (uploaded, last)| {
            let keep = frame_age(now, *last) < KEEP_FRAMES;
            if !keep {
                spare.push(uploaded.texture.clone());
            }
            keep
        });
        // The oldest go first.
        let excess = self.spare.len().saturating_sub(SPARE_TEXTURES);
        self.spare.drain(..excess);
        ids
    }

    /// Lets go of the targets, copies and bind groups no frame has used for a while.
    fn evict(&mut self) {
        let now = self.frame;
        let fresh = |last: u64| frame_age(now, last) < KEEP_TARGETS;
        self.off_screen.retain(|_, (_, last)| fresh(*last));
        self.depths.retain(|_, (_, last)| fresh(*last));
        self.crops.retain(|_, (_, last)| fresh(*last));
        self.feedbacks.retain(|_, (_, last)| fresh(*last));
        self.binds.retain(|_, (_, last)| fresh(*last));
    }

    /// A batch's pipeline (made if new), bind group (made if new), its place in the frame's
    /// vertices after the `vertex_count` before it, its uniform block in `uniforms`, and the
    /// state its draw sets. `feedback` is the copy of the target the stages that sample it read.
    fn prepare(
        &mut self,
        batch: &Batch,
        feedback: Option<Uploaded>,
        vertex_count: &mut u32,
        uniforms: &mut Uniforms<'_>,
    ) -> Prepared {
        let state = &batch.state;
        let range = state.depth_range;
        let key = PipelineKey {
            depth: state.depth.map(|d| (d.function, d.write)),
            blend: state.blend.map(|b| (b.source, b.destination, b.equation)),
            lines: state.lines,
            colour_writes: state.colour_writes,
            cull: state.cull,
            stencil: state.stencil.map(|s| Stencil { reference: 0, ..s }),
            w_buffer: range.w_buffer.is_some(),
        };
        self.pipeline(key);
        // Each stage's texture as uploaded, if it has one, and its shape.
        let uploaded: [Option<Uploaded>; 4] = std::array::from_fn(|s| {
            if state.feedback[s] {
                return feedback.clone();
            }
            state.textures[s].and_then(|(_, id)| {
                self.drawn
                    .get(&id)
                    .or_else(|| self.textures.get(&id).map(|(u, _)| u))
                    .cloned()
            })
        });
        let shapes = uploaded
            .each_ref()
            .map(|u| u.as_ref().map_or(Shape::Flat, |u| u.shape));
        let params = Params {
            combiner: state.combiner,
            alpha_test: state.alpha_test,
            alpha_kill: state.textures.map(|t| t.is_some_and(|(t, _)| t.alpha_kill)),
            bias: state.textures.map(|t| {
                t.map_or(0.0, |(t, _)| {
                    // Bits 0..12: a signed bias in 1/256.
                    let raw = i32::try_from(t.filter & 0x1FFF).unwrap_or(0);
                    let signed = if raw & 0x1000 != 0 { raw - 0x2000 } else { raw };
                    signed as f32 / 256.0
                })
            }),
            modes: state.modes,
            shapes,
            scales: state.textures.map(|t| match t {
                Some((t, _)) if t.linear() => [1.0 / t.width as f32, 1.0 / t.height as f32],
                _ => [1.0, 1.0],
            }),
            shader: state.shader,
            colour_keys: std::array::from_fn(|s| {
                let (key, mask) = state.colour_keys[s];
                let mode = state.textures[s].map_or(0, |(t, _)| t.control0 & 3);
                (mode, key, mask)
            }),
            window: state.clip.window.map(|w| (w.exclusive, w.rects)),
            drop_outside: !range.clamp,
            w_buffer: range.w_buffer,
            depth_range: [range.min, range.max],
        };
        let uniform = uniforms.offset_of(&params);
        let first = *vertex_count;
        *vertex_count =
            first.saturating_add(u32::try_from(batch.vertices.len()).unwrap_or(u32::MAX));
        let vertices = first..*vertex_count;
        let bind = self.bind_group(state, &uploaded);
        Prepared {
            key,
            bind,
            uniform,
            vertices,
            clip: state.clip,
            viewport: super::batch::viewport(&range),
            stencil_reference: state.stencil.map(|s| u32::from(s.reference)),
            blend_constant: state.blend.as_ref().and_then(blend_constant),
            feedback,
        }
    }

    /// The key of a batch's texture bind group, made if new: each stage's texture in the binding
    /// of its shape (white in the other two) with its sampler.
    fn bind_group(
        &mut self,
        state: &super::batch::State,
        uploaded: &[Option<Uploaded>; 4],
    ) -> BindKey {
        let key: BindKey = std::array::from_fn(|s| {
            (
                uploaded[s].as_ref().map_or(0, |u| u.serial),
                state.textures[s].map(|(t, _)| sampler_key(&t)),
            )
        });
        let frame = self.frame;
        if let Some(entry) = self.binds.get_mut(&key) {
            entry.1 = frame;
            return key;
        }
        // Each stage's flat, 3D and cube-map bindings: its texture in the one of its shape.
        let views: [[wgpu::TextureView; 3]; 4] = uploaded.each_ref().map(|u| {
            let mut views = [
                self.white.clone(),
                self.white_volume.clone(),
                self.white_cube.clone(),
            ];
            if let Some(u) = u {
                let slot = match u.shape {
                    Shape::Flat => 0,
                    Shape::Volume => 1,
                    Shape::Cube => 2,
                };
                views[slot] = u.view.clone();
            }
            views
        });
        let samplers: [wgpu::Sampler; 4] = [0, 1, 2, 3].map(|s| match state.textures[s] {
            Some((t, _)) => self.sampler(&t).clone(),
            None => self.default_sampler.clone(),
        });
        let mut entries = Vec::with_capacity(16);
        for (stage, (views, sampler)) in (0u32..).zip(views.iter().zip(&samplers)) {
            entries.extend([
                wgpu::BindGroupEntry {
                    binding: 1 + 2 * stage,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2 + 2 * stage,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9 + stage,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 13 + stage,
                    resource: wgpu::BindingResource::TextureView(&views[2]),
                },
            ]);
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        self.binds.insert(key, (group, frame));
        key
    }

    /// Renders a frame's commands with its bound textures. Returns the image the display shows
    /// (an anti-aliased surface's samples averaged down), or `None` for a frame that draws
    /// nothing.
    pub fn render(&mut self, commands: &[Command], textures: &[TextureData]) -> Option<Image> {
        self.render_primitives(draw::primitives(commands), textures)
    }

    /// [`Renderer::render`] for a frame whose commands the GPU model already made into
    /// `primitives` (a `Vec` or a slice).
    pub fn render_primitives(
        &mut self,
        primitives: impl std::borrow::Borrow<[Primitive]>,
        textures: &[TextureData],
    ) -> Option<Image> {
        let (mut encoder, drawn) = self.encode(primitives.borrow(), textures)?;
        let target = &self.target.as_ref()?.texture;
        let copy = super::present::Snapshot::start(
            &self.device,
            &mut encoder,
            target,
            drawn.width,
            drawn.height,
        );
        self.submit(encoder);
        copy.map();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        Some(downsample(
            copy.pixels(),
            drawn.width,
            drawn.height,
            drawn.samples,
        ))
    }

    /// Renders a frame as [`Renderer::render`] does and shows it on the layer, scaled to `size`
    /// pixels, without waiting for the GPU. With `snapshot`, the frame is also copied back; the
    /// picture of a copy started earlier is returned once it has arrived. Returns `None` for a
    /// frame that draws nothing, or when the renderer has no layer.
    pub fn present(
        &mut self,
        commands: &[Command],
        textures: &[TextureData],
        size: (u32, u32),
        snapshot: bool,
    ) -> Option<Option<Image>> {
        self.present_primitives(draw::primitives(commands), textures, size, snapshot)
    }

    /// [`Renderer::present`] for a frame whose commands the GPU model already made into
    /// `primitives` (a `Vec` or a slice).
    pub fn present_primitives(
        &mut self,
        primitives: impl std::borrow::Borrow<[Primitive]>,
        textures: &[TextureData],
        size: (u32, u32),
        snapshot: bool,
    ) -> Option<Option<Image>> {
        self.presenter.as_ref()?;
        let arrived = self.take_snapshot();
        let (mut encoder, drawn) = self.encode(primitives.borrow(), textures)?;
        let (width, height) = (drawn.width, drawn.height);
        // The layer is given the target itself: the next frame's drawing into it is a later
        // submission, so the layer shows a whole frame either way.
        let shown = self.target.as_ref()?.texture.clone();
        let presenter = self.presenter.as_mut()?;
        let copying = (snapshot && presenter.snapshot.is_none()).then(|| {
            super::present::Snapshot::start(&self.device, &mut encoder, &shown, width, height)
        });
        presenter.submit(encoder);
        if size.0 != 0 && size.1 != 0 {
            presenter.offer(shown, size);
        }
        if let Some(copy) = copying {
            copy.map();
            presenter.snapshot = Some(copy);
            self.snapshot_samples = drawn.samples;
        }
        Some(arrived)
    }

    /// The picture of the frame copied back by an earlier [`Renderer::present`], if it has
    /// arrived.
    fn take_snapshot(&mut self) -> Option<Image> {
        let presenter = self.presenter.as_mut()?;
        presenter.snapshot.as_ref()?;
        let _ = self.device.poll(wgpu::PollType::Poll);
        if !presenter.snapshot.as_ref()?.ready() {
            return None;
        }
        let copy = presenter.snapshot.take()?;
        let (width, height) = (copy.width, copy.height);
        Some(downsample(
            copy.pixels(),
            width,
            height,
            self.snapshot_samples,
        ))
    }

    /// Whether the renderer shows its frames on a layer.
    #[must_use]
    pub fn has_layer(&self) -> bool {
        self.presenter.is_some()
    }

    /// Submits `encoder`, never while the presenting thread configures the surface.
    fn submit(&self, encoder: wgpu::CommandEncoder) {
        match &self.presenter {
            Some(presenter) => presenter.submit(encoder),
            None => {
                self.queue.submit([encoder.finish()]);
            }
        }
    }

    /// Records drawing a frame's primitives, every pass in one encoder; `None` for a frame that
    /// draws nothing.
    ///
    /// The frame shows the colour buffer its last primitive draws into, at that surface's size.
    /// What it draws into other colour buffers it then samples as linear textures (render to
    /// texture: the model writes no pixels to guest memory, so those textures' bytes are stale)
    /// is drawn first, each at its surface's size, and each such texture reads that drawing
    /// instead. Primitives drawn into other buffers that nothing samples are dropped. Targets,
    /// depth buffers and copies are kept from frame to frame, each cleared as it is drawn.
    fn encode(
        &mut self,
        primitives: &[Primitive],
        textures: &[TextureData],
    ) -> Option<(wgpu::CommandEncoder, Drawn)> {
        self.frame += 1;
        let mut ids = self.prepare_textures(textures);
        let shown = primitives.last()?.surface;
        self.evict();
        self.drawn.clear();
        let mut plans = Vec::new();
        for (surface, sampled) in off_screen(primitives, shown) {
            let on_it = || primitives.iter().filter(|p| p.surface == surface);
            // Drawn at the size the surface last had.
            let (width, height) = target_size(&on_it().next_back()?.pixel.target);
            let batches = super::batch::batches(on_it(), &mut |t| ids.get(&identity(t)).copied());
            let target = self.off_screen_target(surface, width, height);
            let mut crops: Vec<Uploaded> = Vec::new();
            for texture in sampled {
                // Ids of their own, counting down from the top, which content ids do not reach
                // in practice.
                let id = u64::MAX - self.drawn.len() as u64;
                let picture = self.picture_of(surface, &target, &texture);
                if picture.serial != target.serial
                    && crops.iter().all(|c| c.serial != picture.serial)
                {
                    crops.push(picture.clone());
                }
                self.drawn.insert(id, picture);
                ids.insert(identity(&texture), id);
            }
            plans.push(Plan {
                target,
                batches,
                crops,
            });
        }
        let on_screen = || primitives.iter().filter(|p| p.surface == shown);
        let surface = on_screen().next_back()?.pixel.target;
        let batches = super::batch::batches(on_screen(), &mut |t| ids.get(&identity(t)).copied());
        if batches.is_empty() {
            return None;
        }
        let (width, height) = target_size(&surface);
        let target = match self.target.take() {
            Some(target) if target.size() == (width, height) => target,
            _ => Target::new(&self.device, width, height),
        };
        self.target = Some(target.clone());
        plans.push(Plan {
            target,
            batches,
            crops: Vec::new(),
        });
        let prepared = self.prepare_frame(&plans);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for (plan, prepared) in plans.iter().zip(&prepared) {
            self.encode_pass(&mut encoder, &plan.target, prepared);
            for crop in &plan.crops {
                encoder.copy_texture_to_texture(
                    plan.target.texture.as_image_copy(),
                    crop.texture.as_image_copy(),
                    crop.texture.size(),
                );
            }
        }
        Some((
            encoder,
            Drawn {
                width,
                height,
                samples: surface.samples,
            },
        ))
    }

    /// Prepares every pass's batches, and writes the frame's vertices and uniform blocks to
    /// their buffers.
    fn prepare_frame(&mut self, plans: &[Plan]) -> Vec<Vec<Prepared>> {
        let mut vertex_count = 0;
        let mut uniform_bytes = std::mem::take(&mut self.uniform_bytes);
        uniform_bytes.clear();
        let mut uniforms = Uniforms {
            bytes: &mut uniform_bytes,
            slot: self.uniform_slot,
            last: None,
        };
        let mut prepared = Vec::with_capacity(plans.len());
        for plan in plans {
            self.depth(plan.target.size());
            let mut list = Vec::with_capacity(plan.batches.len());
            for batch in &plan.batches {
                let feedback = self.feedback_copy(batch, &plan.target);
                list.push(self.prepare(batch, feedback, &mut vertex_count, &mut uniforms));
            }
            prepared.push(list);
        }
        self.write_vertices(plans, vertex_count);
        self.write_uniforms(&uniform_bytes);
        self.uniform_bytes = uniform_bytes;
        prepared
    }

    /// Writes the frame's `count` vertices, every pass's batches in order, growing the vertex
    /// buffer when it is too small.
    fn write_vertices(&mut self, plans: &[Plan], count: u32) {
        const SIZE: usize = std::mem::size_of::<GpuVertex>();
        let Some(size) = wgpu::BufferSize::new(u64::from(count) * SIZE as u64) else {
            return;
        };
        if self
            .vertices
            .as_ref()
            .is_none_or(|buffer| buffer.size() < size.get())
        {
            self.vertices = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vertices"),
                size: size.get().next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let Some(buffer) = &self.vertices else {
            return;
        };
        let Some(mut view) = self.queue.write_buffer_with(buffer, 0, size) else {
            return;
        };
        let mut at = 0;
        for vertex in plans
            .iter()
            .flat_map(|plan| &plan.batches)
            .flat_map(|batch| &batch.vertices)
        {
            view.slice(at..at + SIZE)
                .copy_from_slice(&vertex.to_bytes());
            at += SIZE;
        }
    }

    /// Writes the frame's uniform blocks, growing the uniform buffer when it is too small.
    fn write_uniforms(&mut self, uniform_bytes: &[u8]) {
        let size = uniform_bytes.len().max(self.uniform_slot) as u64;
        if self
            .uniforms
            .as_ref()
            .is_none_or(|(buffer, _)| buffer.size() < size)
        {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniforms"),
                size: size.next_power_of_two(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("uniforms"),
                layout: &self.uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(super::combiner::PARAMS_SIZE as u64),
                    }),
                }],
            });
            self.uniforms = Some((buffer, group));
        }
        if let Some((buffer, _)) = &self.uniforms
            && !uniform_bytes.is_empty()
        {
            self.queue.write_buffer(buffer, 0, uniform_bytes);
        }
    }

    /// The target of colour buffer `surface` drawn at `width` by `height`, made if new.
    fn off_screen_target(&mut self, surface: u32, width: u32, height: u32) -> Target {
        let frame = self.frame;
        let device = &self.device;
        let entry = self
            .off_screen
            .entry((surface, width, height))
            .or_insert_with(|| (Target::new(device, width, height), frame));
        entry.1 = frame;
        entry.0.clone()
    }

    /// Makes sure there is a depth and stencil buffer of `size`.
    fn depth(&mut self, size: (u32, u32)) {
        let frame = self.frame;
        let device = &self.device;
        let entry = self.depths.entry(size).or_insert_with(|| {
            let depth = attachment(
                device,
                "depth",
                DEPTH,
                size,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            (
                depth.create_view(&wgpu::TextureViewDescriptor::default()),
                frame,
            )
        });
        entry.1 = frame;
    }

    /// The part of `target` texture `t` covers (its top-left corner, of its size within the
    /// target's): what a draw sampling that colour buffer reads. The target itself when that is
    /// all of it, else a copy kept for the surface and size, which the frame fills once the
    /// target is drawn.
    fn picture_of(&mut self, surface: u32, target: &Target, t: &Texture) -> Uploaded {
        let (target_width, target_height) = target.size();
        let size = (
            t.width.clamp(1, target_width),
            t.height.clamp(1, target_height),
        );
        if size == target.size() {
            return target.sampled();
        }
        let frame = self.frame;
        let device = &self.device;
        let entry = self
            .crops
            .entry((surface, size.0, size.1))
            .or_insert_with(|| (copy_texture(device, size), frame));
        entry.1 = frame;
        entry.0.clone()
    }

    /// The copy of `target` as drawn so far that a batch sampling it reads, of the size of the
    /// first texture that samples it; `None` for a batch that samples no target.
    fn feedback_copy(&mut self, batch: &Batch, target: &Target) -> Option<Uploaded> {
        let stage = batch.state.feedback.iter().position(|on| *on)?;
        let (t, _) = batch.state.textures[stage]?;
        let (target_width, target_height) = target.size();
        let size = (
            t.width.clamp(1, target_width),
            t.height.clamp(1, target_height),
        );
        let frame = self.frame;
        let device = &self.device;
        let entry = self
            .feedbacks
            .entry(size)
            .or_insert_with(|| (copy_texture(device, size), frame));
        entry.1 = frame;
        Some(entry.0.clone())
    }

    /// Records drawing the `prepared` batches into `target`, cleared first to opaque black with
    /// its depth at the far end and its stencil 0. A batch that samples the target as drawn so
    /// far splits the drawing: the target is copied for it to sample, and drawing goes on over
    /// what is there, depth and stencil included. State a batch sets that the one before it in
    /// the pass set already is not set again.
    #[allow(clippy::too_many_lines)]
    fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &Target,
        prepared: &[Prepared],
    ) {
        let (width, height) = target.size();
        let depth_view = &self.depths[&(width, height)].0;
        let mut start = 0;
        // At least one pass, which clears.
        loop {
            let end = prepared
                .iter()
                .enumerate()
                .skip(start + 1)
                .find(|(_, p)| p.feedback.is_some())
                .map_or(prepared.len(), |(i, _)| i);
            if let Some(copy) = prepared.get(start).and_then(|p| p.feedback.as_ref()) {
                encoder.copy_texture_to_texture(
                    target.texture.as_image_copy(),
                    copy.texture.as_image_copy(),
                    copy.texture.size(),
                );
            }
            let clear = start == 0;
            let load = |clear_to| {
                if clear {
                    wgpu::LoadOp::Clear(clear_to)
                } else {
                    wgpu::LoadOp::Load
                }
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: if clear {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: if clear {
                            wgpu::LoadOp::Clear(0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let (Some(vertices), Some((_, uniforms))) = (&self.vertices, &self.uniforms) {
                pass.set_vertex_buffer(0, vertices.slice(..));
                let mut set = SetState::default();
                for p in &prepared[start..end] {
                    // A scissor off the target leaves nothing to draw.
                    let Some(rect) = p.clip.rect_within(width, height) else {
                        continue;
                    };
                    if set.scissor.replace(rect) != Some(rect) {
                        let [x, y, w, h] = rect;
                        pass.set_scissor_rect(x, y, w, h);
                    }
                    let (near, far) = p.viewport;
                    let viewport = (near.to_bits(), far.to_bits());
                    if set.viewport.replace(viewport) != Some(viewport) {
                        pass.set_viewport(0.0, 0.0, width as f32, height as f32, near, far);
                    }
                    if let Some(reference) = p.stencil_reference
                        && set.stencil_reference.replace(reference) != Some(reference)
                    {
                        pass.set_stencil_reference(reference);
                    }
                    if let Some(constant) = p.blend_constant {
                        let bits =
                            [constant.r, constant.g, constant.b, constant.a].map(f64::to_bits);
                        if set.blend_constant.replace(bits) != Some(bits) {
                            pass.set_blend_constant(constant);
                        }
                    }
                    if set.pipeline.replace(p.key) != Some(p.key) {
                        pass.set_pipeline(&self.pipelines[&p.key]);
                    }
                    if set.bind.replace(p.bind) != Some(p.bind) {
                        pass.set_bind_group(0, &self.binds[&p.bind].0, &[]);
                    }
                    if set.uniform.replace(p.uniform) != Some(p.uniform) {
                        pass.set_bind_group(1, uniforms, &[p.uniform]);
                    }
                    pass.draw(p.vertices.clone(), 0..1);
                }
            }
            drop(pass);
            if end >= prepared.len() {
                break;
            }
            start = end;
        }
    }
}

/// The bind group layout of group 0: each stage's texture and sampler, then each stage's texture
/// again as a 3D texture and as a cube map.
fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_entry = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: None,
    };
    let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let mut entries = Vec::with_capacity(16);
    for stage in 0..4 {
        entries.push(texture_entry(1 + 2 * stage, wgpu::TextureViewDimension::D2));
        entries.push(sampler_entry(2 + 2 * stage));
    }
    for stage in 0..4 {
        entries.push(texture_entry(9 + stage, wgpu::TextureViewDimension::D3));
        entries.push(texture_entry(13 + stage, wgpu::TextureViewDimension::Cube));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("anyxbox"),
        entries: &entries,
    })
}

/// The bind group layout of group 1: the uniform block, at an offset in the frame's uniforms.
fn uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(super::combiner::PARAMS_SIZE as u64),
            },
            count: None,
        }],
    })
}

/// A texture's sampler state.
fn sampler_key(t: &Texture) -> SamplerKey {
    SamplerKey {
        address: t.address,
        filter: t.filter,
        border: t.border,
        control0: t.control0,
        levels: t.levels,
    }
}

/// A flat texture of `size` that a part of a target is copied into, to be sampled.
fn copy_texture(device: &wgpu::Device, size: (u32, u32)) -> Uploaded {
    let texture = attachment(
        device,
        "drawn",
        FORMAT,
        size,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    Uploaded {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        texture,
        shape: Shape::Flat,
        serial: next_serial(),
    }
}

/// The colour buffers other than `shown` that `primitives` draw into and also sample as linear
/// textures, in first-drawn order, each with the distinct textures that read it.
fn off_screen(primitives: &[Primitive], shown: u32) -> Vec<(u32, Vec<Texture>)> {
    let mut out: Vec<(u32, Vec<Texture>)> = Vec::new();
    let mut last = shown;
    for p in primitives {
        if p.surface != last && p.surface != shown && out.iter().all(|(s, _)| *s != p.surface) {
            out.push((p.surface, Vec::new()));
        }
        last = p.surface;
    }
    for t in primitives.iter().flat_map(|p| p.textures.iter().flatten()) {
        if !t.linear() {
            continue;
        }
        if let Some((_, sampled)) = out.iter_mut().find(|(s, _)| *s == t.offset)
            && !sampled.contains(t)
        {
            sampled.push(*t);
        }
    }
    out.retain(|(_, sampled)| !sampled.is_empty());
    out
}

/// A `width` by `height` render attachment.
fn attachment(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    (width, height): (u32, u32),
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// How many frames ago `last` was.
fn frame_age(now: u64, last: u64) -> u64 {
    now.saturating_sub(last)
}

/// An uploaded texture: its view, of the dimension its shape is sampled through, and the
/// view's serial.
#[derive(Debug, Clone)]
struct Uploaded {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    shape: Shape,
    serial: u64,
}

/// Uploads decoded levels as an RGBA8 texture with its mip chain: a 3D texture of its slices, a
/// cube map of its six faces, or flat (a level's layers past the first then left out). A texture
/// of `spare` of the same shape takes the levels when there is one.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    levels: &[crate::gpu::texture::Level],
    shape: Shape,
    spare: &mut Vec<wgpu::Texture>,
) -> Uploaded {
    let layers = match shape {
        Shape::Flat => 1,
        Shape::Volume => levels[0].layers,
        Shape::Cube => 6,
    };
    let descriptor = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: levels[0].width,
            height: levels[0].height,
            depth_or_array_layers: layers,
        },
        mip_level_count: u32::try_from(levels.len()).unwrap_or(1),
        sample_count: 1,
        dimension: if shape == Shape::Volume {
            wgpu::TextureDimension::D3
        } else {
            wgpu::TextureDimension::D2
        },
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let same = spare.iter().position(|t| {
        (t.size(), t.mip_level_count(), t.dimension())
            == (
                descriptor.size,
                descriptor.mip_level_count,
                descriptor.dimension,
            )
    });
    let texture = match same {
        Some(index) => spare.remove(index),
        None => device.create_texture(&descriptor),
    };
    for (mip, level) in levels.iter().enumerate() {
        let layers = match shape {
            Shape::Flat => 1,
            Shape::Volume => level.layers,
            Shape::Cube => 6,
        };
        let len = (level.width * level.height * layers * 4) as usize;
        let mut rgba = level.rgba.clone();
        rgba.resize(len, 0);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: u32::try_from(mip).unwrap_or(0),
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(level.width * 4),
                rows_per_image: Some(level.height),
            },
            wgpu::Extent3d {
                width: level.width,
                height: level.height,
                depth_or_array_layers: layers,
            },
        );
    }
    let dimension = match shape {
        Shape::Flat => wgpu::TextureViewDimension::D2,
        Shape::Volume => wgpu::TextureViewDimension::D3,
        Shape::Cube => wgpu::TextureViewDimension::Cube,
    };
    Uploaded {
        view: texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(dimension),
            ..wgpu::TextureViewDescriptor::default()
        }),
        texture,
        shape,
        serial: next_serial(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::draw::Combiner;
    use crate::render::combiner::{Fragment, evaluate};

    const KELVIN: u32 = 0x97;

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

    /// A renderer, or `None` (with a note) where the machine has no graphics adapter.
    fn renderer() -> Option<Renderer> {
        match Renderer::new() {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("skipped: {e}");
                None
            }
        }
    }

    /// Position (3 floats) and a packed diffuse colour.
    fn layout() -> Vec<Command> {
        vec![kelvin(0x1760, 3 << 4 | 2), kelvin(0x1760 + 12, 4 << 4)]
    }

    fn vertex(x: f32, y: f32, depth: f32, argb: u32) -> Vec<Command> {
        vec![
            float(0x1818, x),
            float(0x1818, y),
            float(0x1818, depth),
            kelvin(0x1818, argb),
        ]
    }

    /// A full-screen quad of one colour at `depth`.
    fn quad(argb: u32, depth: f32) -> Vec<Command> {
        let mut c = vec![kelvin(0x17FC, 8)];
        for (x, y) in [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)] {
            c.extend(vertex(x, y, depth, argb));
        }
        c.push(kelvin(0x17FC, 0));
        c
    }

    fn pixel(image: &Image, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * image.width + x) * 4) as usize;
        let p = &image.rgba[i..i + 4];
        [p[0], p[1], p[2], p[3]]
    }

    #[test]
    fn a_triangle_is_drawn_in_its_colour_on_black() {
        let Some(mut r) = renderer() else { return };
        let mut commands = layout();
        commands.push(kelvin(0x17FC, 5));
        commands.extend(vertex(100.0, 100.0, 0.5, 0xFFFF_0000));
        commands.extend(vertex(300.0, 100.0, 0.5, 0xFFFF_0000));
        commands.extend(vertex(100.0, 300.0, 0.5, 0xFFFF_0000));
        commands.push(kelvin(0x17FC, 0));
        let image = r.render(&commands, &[]).expect("a frame with vertices");
        assert_eq!((image.width, image.height), (640, 480));
        assert_eq!(image.rgba.len(), 640 * 480 * 4);
        assert_eq!(pixel(&image, 150, 150), [255, 0, 0, 255]);
        assert_eq!(pixel(&image, 400, 400), [0, 0, 0, 255]);
        assert!(r.render(&layout(), &[]).is_none(), "no vertices, no frame");
    }

    #[test]
    fn the_combiner_shader_matches_the_reference_evaluator() {
        let Some(mut r) = renderer() else { return };
        let diffuse = 0xCC33_6699u32;
        let programs = [
            // Final combiner only: diffuse alpha * diffuse + (1 - a) * constant... (C = final constant 0).
            Combiner {
                final0: (0x10 | 4) << 24 | 4 << 16 | 1 << 8,
                final1: 4 << 8,
                final_constants: [0xFF20_4080, 0],
                ..Combiner::default()
            },
            // One stage: spare0 = diffuse * constant 0, scaled by 2; final outputs spare0.
            Combiner {
                colour_inputs: [4 << 24 | 1 << 16, 0, 0, 0, 0, 0, 0, 0],
                colour_outputs: [12 << 4 | 2 << 15, 0, 0, 0, 0, 0, 0, 0],
                alpha_inputs: [(0x10 | 4) << 24 | 0x20 << 16, 0, 0, 0, 0, 0, 0, 0],
                alpha_outputs: [12 << 4, 0, 0, 0, 0, 0, 0, 0],
                constant0: [0xFF80_C040, 0, 0, 0, 0, 0, 0, 0],
                control: 1,
                final0: 12 << 24 | 0x20 << 16,
                final1: 12 << 8,
                ..Combiner::default()
            },
            // Two stages with a dot product and an inverted mapping.
            Combiner {
                colour_inputs: [
                    4 << 24 | 1 << 16,
                    (0x20 | 0xC) << 24 | 4 << 16,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                colour_outputs: [12 << 4 | 1 << 13, 13 << 4, 0, 0, 0, 0, 0, 0],
                constant0: [0xFF40_4040, 0, 0, 0, 0, 0, 0, 0],
                control: 2,
                final0: 13 << 24 | 0x20 << 16,
                final1: 4 << 8,
                ..Combiner::default()
            },
        ];
        for (n, program) in programs.iter().enumerate() {
            let mut commands = layout();
            let words = |base: u32, a: &[u32; 8]| {
                a.iter()
                    .enumerate()
                    .map(move |(i, &v)| kelvin(base + 4 * u32::try_from(i).unwrap_or(0), v))
                    .collect::<Vec<_>>()
            };
            commands.extend(words(0x0260, &program.alpha_inputs));
            commands.extend(words(0x0AA0, &program.alpha_outputs));
            commands.extend(words(0x0AC0, &program.colour_inputs));
            commands.extend(words(0x1E40, &program.colour_outputs));
            commands.extend(words(0x0A60, &program.constant0));
            commands.extend(words(0x0A80, &program.constant1));
            commands.extend([
                kelvin(0x0288, program.final0),
                kelvin(0x028C, program.final1),
                kelvin(0x1E20, program.final_constants[0]),
                kelvin(0x1E24, program.final_constants[1]),
                kelvin(0x1E60, program.control),
            ]);
            commands.extend(quad(diffuse, 0.5));
            let image = r.render(&commands, &[]).expect("a frame");
            let want = evaluate(
                program,
                &Fragment {
                    texture0: [1.0; 4],
                    texture1: [1.0; 4],
                    texture2: [0.0, 0.0, 0.0, 1.0],
                    texture3: [0.0, 0.0, 0.0, 1.0],
                    colour: crate::render::combiner::colour(diffuse),
                    fog: [0.0; 4],
                    specular: [0.0, 0.0, 0.0, 1.0],
                },
            );
            let got = pixel(&image, 320, 240);
            for k in 0..3 {
                let expected = (want[k] * 255.0).round();
                assert!(
                    (f32::from(got[k]) - expected).abs() <= 1.0,
                    "program {n} channel {k}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    #[test]
    fn alpha_test_blending_and_depth() {
        let Some(mut r) = renderer() else { return };
        // A grey quad, then a half-transparent white quad blended over it.
        let mut commands = layout();
        commands.extend(quad(0xFF80_8080, 0.5));
        commands.extend([
            kelvin(0x0304, 1),
            kelvin(0x0344, 0x302),
            kelvin(0x0348, 0x303),
        ]);
        commands.extend(quad(0x80FF_FFFF, 0.5));
        let image = r.render(&commands, &[]).unwrap();
        let p = pixel(&image, 10, 10);
        assert!((i32::from(p[0]) - 192).abs() <= 2, "blended: {p:?}");
        // The alpha test (GREATER than 0x90) discards the second quad entirely.
        let mut commands = layout();
        commands.extend(quad(0xFF80_8080, 0.5));
        commands.extend([
            kelvin(0x0300, 1),
            kelvin(0x033C, 0x204),
            kelvin(0x0340, 0x90),
        ]);
        commands.extend(quad(0x80FF_FFFF, 0.5));
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!(pixel(&image, 10, 10), [128, 128, 128, 255]);
        // Depth: a nearer quad first hides a farther one drawn after it.
        let mut commands = layout();
        commands.push(kelvin(0x030C, 1));
        commands.extend(quad(0xFF00_FF00, 0.25));
        commands.extend(quad(0xFFFF_0000, 0.75));
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!(pixel(&image, 10, 10), [0, 255, 0, 255]);
    }

    #[test]
    fn a_texture_is_sampled_and_cached_by_content() {
        let Some(mut r) = renderer() else { return };
        // A 2x2 A8R8G8B8 swizzled texture, all blue, nearest filtering.
        let stage = crate::gpu::texture::StageRegs {
            offset: 0x1000,
            format: 6 << 8 | 1 << 16 | 1 << 20 | 1 << 24,
            filter: 1 << 24 | 1 << 16,
            ..crate::gpu::texture::StageRegs::default()
        };
        let texture = stage.texture().unwrap();
        let data = TextureData::new(texture, [255u8, 0, 0, 255].repeat(4), Vec::new());
        let mut commands = layout();
        commands.extend([
            kelvin(0x1760 + 36, 2 << 4 | 2),
            kelvin(0x1B00, 0x1000),
            kelvin(0x1B04, stage.format),
            kelvin(0x1B14, stage.filter),
        ]);
        commands.push(kelvin(0x17FC, 8));
        for (x, y) in [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)] {
            commands.extend(vertex(x, y, 0.5, 0xFFFF_FFFF));
            commands.extend([float(0x1818, x / 640.0), float(0x1818, y / 480.0)]);
        }
        commands.push(kelvin(0x17FC, 0));
        let image = r.render(&commands, std::slice::from_ref(&data)).unwrap();
        assert_eq!(pixel(&image, 320, 240), [0, 0, 255, 255]);
        r.render(&commands, &[data]).unwrap();
        assert_eq!(r.textures.len(), 1, "the same content is uploaded once");
    }

    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const WHITE: [u8; 4] = [255; 4];

    /// A texture on `stage` with nearest filtering, from its format word and its texels' RGBA
    /// (stored as A8R8G8B8, or as the format's bytes for others): the commands that bind it, and
    /// its data.
    fn stage_texture(
        stage: u32,
        offset: u32,
        format: u32,
        rgba: &[[u8; 4]],
    ) -> (Vec<Command>, TextureData) {
        let regs = crate::gpu::texture::StageRegs {
            offset,
            format,
            filter: 1 << 24 | 1 << 16,
            control0: crate::gpu::texture::DEFAULT_CONTROL0 | 1 << 30,
            ..crate::gpu::texture::StageRegs::default()
        };
        let base = 0x1B00 + 0x40 * stage;
        let commands = vec![
            kelvin(base, regs.offset),
            kelvin(base + 4, regs.format),
            kelvin(base + 0x0C, regs.control0),
            kelvin(base + 0x14, regs.filter),
        ];
        let bytes = rgba.iter().flat_map(|&[r, g, b, a]| [b, g, r, a]).collect();
        (
            commands,
            TextureData::new(regs.texture().unwrap(), bytes, Vec::new()),
        )
    }

    /// A full-screen white quad whose four stages' coordinates are `coordinates` everywhere.
    fn stage_quad(coordinates: [[f32; 4]; 4]) -> Vec<Command> {
        let mut c = layout();
        for attribute in 9..13 {
            c.push(kelvin(0x1760 + 4 * attribute, 4 << 4 | 2));
        }
        c.push(kelvin(0x17FC, 8));
        for (x, y) in [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)] {
            c.extend(vertex(x, y, 0.5, 0xFFFF_FFFF));
            c.extend(coordinates.as_flattened().iter().map(|&f| float(0x1818, f)));
        }
        c.push(kelvin(0x17FC, 0));
        c
    }

    /// The final combiner showing register `register`, colour and alpha.
    fn showing(register: u32) -> Vec<Command> {
        vec![
            kelvin(0x0288, register << 24 | 0x20 << 16),
            kelvin(0x028C, register << 8),
            kelvin(0x1E60, 0),
        ]
    }

    fn near(got: [u8; 4], want: [u8; 4]) -> bool {
        got.iter().zip(want).all(|(g, w)| g.abs_diff(w) <= 2)
    }

    /// A 2x2 swizzled A8R8G8B8 texture's format word.
    const TWO_BY_TWO: u32 = 6 << 8 | 1 << 16 | 1 << 20 | 1 << 24;

    #[test]
    fn stages_2_and_3_reach_the_combiners_as_registers_10_and_11() {
        let Some(mut r) = renderer() else { return };
        let (two, red) = stage_texture(2, 0x1000, 6 << 8 | 1 << 16, &[RED]);
        let (three, green) = stage_texture(3, 0x2000, 6 << 8 | 1 << 16, &[GREEN]);
        let program = |modes: u32| kelvin(0x1E70, modes);
        for (register, modes, want) in [
            (0xA, 1 << 10 | 1 << 15, RED),
            (0xB, 1 << 10 | 1 << 15, GREEN),
            // A stage without a program gives black with alpha 1; PASS_THROUGH its coordinates.
            (0xA, 1 << 15, [0, 0, 0, 255]),
            (0xB, 4 << 15, [64, 128, 0, 255]),
        ] {
            let mut commands = two.clone();
            commands.extend(three.clone());
            commands.push(program(modes));
            commands.extend(showing(register));
            commands.extend(stage_quad([
                [0.5, 0.5, 0.0, 1.0],
                [0.5, 0.5, 0.0, 1.0],
                [0.5, 0.5, 0.0, 1.0],
                [0.25, 0.5, -1.0, 2.0],
            ]));
            let image = r.render(&commands, &[red.clone(), green.clone()]).unwrap();
            let got = pixel(&image, 320, 240);
            assert!(near(got, want), "register {register:#x}: {got:?}");
        }
    }

    #[test]
    fn a_cube_map_is_looked_up_toward_the_coordinates() {
        let Some(mut r) = renderer() else { return };
        // A8R8G8B8 cube map of 1x1 faces +X, -X, +Y, -Y, +Z, -Z, each 128 bytes apart.
        let faces = [
            RED,
            GREEN,
            BLUE,
            [255, 255, 0, 255],
            WHITE,
            [0, 255, 255, 255],
        ];
        let format = 6 << 8 | 1 << 16 | 2 << 4 | 1 << 2;
        let (mut commands, data) = stage_texture(0, 0x1000, format, &[[0; 4]]);
        let mut bytes = vec![0u8; 5 * 128 + 4];
        for (face, [r, g, b, a]) in faces.iter().enumerate() {
            bytes[face * 128..][..4].copy_from_slice(&[*b, *g, *r, *a]);
        }
        let data = TextureData::new(data.texture, bytes, Vec::new());
        commands.push(kelvin(0x1E70, 3));
        for (direction, want) in [
            ([1.0, 0.1, 0.2], RED),
            ([0.1, 1.0, -0.2], BLUE),
            ([0.2, 0.1, -1.0], [0, 255, 255, 255]),
        ] {
            let mut frame = commands.clone();
            let [s, t, rr] = direction;
            frame.extend(stage_quad([[s, t, rr, 1.0], [0.0; 4], [0.0; 4], [0.0; 4]]));
            let image = r.render(&frame, std::slice::from_ref(&data)).unwrap();
            assert_eq!(pixel(&image, 320, 240), want, "{direction:?}");
        }
    }

    #[test]
    fn a_3d_texture_is_read_at_r_over_q() {
        let Some(mut r) = renderer() else { return };
        // A8R8G8B8, 1x1x2: slice 0 red, slice 1 blue.
        let format = 1 << 28 | 6 << 8 | 1 << 16 | 3 << 4;
        let (mut commands, data) = stage_texture(0, 0x1000, format, &[RED, BLUE]);
        commands.push(kelvin(0x1E70, 2));
        for (rr, q, want) in [(0.25, 1.0, RED), (0.75, 1.0, BLUE), (1.5, 2.0, BLUE)] {
            let mut frame = commands.clone();
            frame.extend(stage_quad([
                [0.5 * q, 0.5 * q, rr, q],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
            ]));
            let image = r.render(&frame, std::slice::from_ref(&data)).unwrap();
            assert_eq!(pixel(&image, 320, 240), want, "r {rr} q {q}");
        }
    }

    #[test]
    fn bump_mapping_moves_the_next_stages_coordinates_by_the_matrix() {
        let Some(mut r) = renderer() else { return };
        // Stage 0's texel: luminance 1 in red, du = 1 in blue, dv = 0 in green.
        let (zero, offsets) = stage_texture(0, 0x1000, 6 << 8 | 1 << 16, &[[255, 0, 127, 255]]);
        // Stage 1: red on the left, green on the right.
        let (one, halves) = stage_texture(1, 0x2000, TWO_BY_TWO, &[RED, GREEN, RED, GREEN]);
        for (mode, m00, want) in [(6, 0.0, RED), (6, 0.5, GREEN), (7, 0.5, [0, 128, 0, 128])] {
            let mut commands = zero.clone();
            commands.extend(one.clone());
            commands.push(kelvin(0x1E70, 1 | mode << 5));
            commands.push(float(0x1B00 + 0x40 + 0x28, m00));
            commands.push(float(0x1B00 + 0x40 + 0x38, 0.5));
            commands.extend(showing(9));
            commands.extend(stage_quad([
                [0.5, 0.5, 0.0, 1.0],
                [0.25, 0.25, 0.0, 1.0],
                [0.0; 4],
                [0.0; 4],
            ]));
            let image = r
                .render(&commands, &[offsets.clone(), halves.clone()])
                .unwrap();
            let got = pixel(&image, 320, 240);
            assert!(near(got, want), "mode {mode} m00 {m00}: {got:?}");
        }
    }

    #[test]
    fn dependent_reads_and_dot_products_look_up_where_the_earlier_stages_say() {
        let Some(mut r) = renderer() else { return };
        let (one, quarters) = stage_texture(1, 0x2000, TWO_BY_TWO, &[RED, GREEN, BLUE, WHITE]);
        // Alpha and red point at (0.75, 0.25), green and blue at (0.25, 0.75).
        let (zero, input) = stage_texture(0, 0x1000, 6 << 8 | 1 << 16, &[[64, 64, 191, 191]]);
        for (mode, want) in [(0x0F, GREEN), (0x10, BLUE)] {
            let mut commands = zero.clone();
            commands.extend(one.clone());
            commands.push(kelvin(0x1E70, 1 | mode << 5));
            commands.extend(showing(9));
            commands.extend(stage_quad([[0.5, 0.5, 0.0, 1.0]; 4]));
            let image = r
                .render(&commands, &[input.clone(), quarters.clone()])
                .unwrap();
            assert_eq!(pixel(&image, 320, 240), want, "mode {mode:#x}");
        }
        // Stage 1's dot product of its coordinates with stage 0's red texel is 0.75, stage 2's
        // 0.25: stage 2 (DOT_ST) reads (0.75, 0.25).
        let (zero, red) = stage_texture(0, 0x1000, 6 << 8 | 1 << 16, &[RED]);
        let (two, quarters) = stage_texture(2, 0x3000, TWO_BY_TWO, &[RED, GREEN, BLUE, WHITE]);
        let mut commands = zero;
        commands.extend(two);
        commands.push(kelvin(0x1E70, 1 | 0x11 << 5 | 9 << 10));
        commands.extend(showing(0xA));
        commands.extend(stage_quad([
            [0.5, 0.5, 0.0, 1.0],
            [0.75, 0.5, 0.0, 1.0],
            [0.25, 0.5, 0.0, 1.0],
            [0.0; 4],
        ]));
        let image = r.render(&commands, &[red, quarters]).unwrap();
        assert_eq!(pixel(&image, 320, 240), GREEN);
    }

    #[test]
    fn an_off_screen_colour_buffer_is_drawn_first_and_sampled_as_drawn() {
        let Some(mut r) = renderer() else { return };
        // Draw red into the buffer at 0x0010_0000 (its guest bytes stay zero) ...
        let mut commands = layout();
        commands.push(kelvin(0x0210, 0x0010_0000));
        commands.extend(quad(0xFFFF_0000, 0.5));
        // ... then show the buffer at 0, textured with that one as a linear 640x480 texture
        // (coordinates in texels).
        let regs = crate::gpu::texture::StageRegs {
            offset: 0x0010_0000,
            format: 0x12 << 8,
            control1: 2560 << 16,
            rect: 0x280 << 16 | 0x1E0,
            filter: 1 << 24 | 1 << 16,
            ..crate::gpu::texture::StageRegs::default()
        };
        commands.extend([
            kelvin(0x0210, 0),
            kelvin(0x1B00, regs.offset),
            kelvin(0x1B04, regs.format),
            kelvin(0x1B10, regs.control1),
            kelvin(0x1B14, regs.filter),
            kelvin(0x1B1C, regs.rect),
        ]);
        commands.extend(stage_quad([[320.0, 240.0, 0.0, 1.0]; 4]));
        let stale = TextureData::new(regs.texture().unwrap(), vec![0; 2560 * 480], Vec::new());
        let image = r.render(&commands, &[stale]).unwrap();
        assert_eq!(pixel(&image, 100, 100), RED);
        // The one buffer drawn off the screen and sampled, read by one texture.
        assert_eq!(
            off_screen(&draw::primitives(&commands), 0)
                .iter()
                .map(|(surface, sampled)| (*surface, sampled.len()))
                .collect::<Vec<_>>(),
            [(0x0010_0000, 1)]
        );
        // Later frames draw the same into the targets, depth buffers and bind groups kept from
        // the first, each cleared again.
        let binds = r.binds.len();
        for _ in 0..3 {
            assert_eq!(r.render(&commands, &[]).unwrap(), image);
        }
        assert_eq!((r.off_screen.len(), r.binds.len()), (1, binds));
    }

    /// The largest depth of the default Z24 buffer: depths are over it.
    const Z24: f32 = 16_777_215.0;

    #[test]
    fn a_draw_that_samples_its_own_target_reads_it_as_drawn_and_the_colour_mask_holds() {
        let Some(mut r) = renderer() else { return };
        let mut commands = layout();
        commands.extend(quad(0xFFFF_0000, 0.5));
        // The buffer being drawn, sampled as a linear 640x480 texture whose bytes are stale.
        let regs = crate::gpu::texture::StageRegs {
            offset: 0,
            format: 0x12 << 8,
            control1: 2560 << 16,
            rect: 0x280 << 16 | 0x1E0,
            filter: 1 << 24 | 1 << 16,
            ..crate::gpu::texture::StageRegs::default()
        };
        commands.extend([
            kelvin(0x1B00, regs.offset),
            kelvin(0x1B04, regs.format),
            kelvin(0x1B10, regs.control1),
            kelvin(0x1B14, regs.filter),
            kelvin(0x1B1C, regs.rect),
        ]);
        commands.extend(stage_quad([[320.0, 240.0, 0.0, 1.0]; 4]));
        let stale = TextureData::new(regs.texture().unwrap(), vec![0; 2560 * 480], Vec::new());
        let image = r.render(&commands, &[stale]).unwrap();
        assert_eq!(
            pixel(&image, 100, 100),
            RED,
            "the target as drawn, not the stale bytes"
        );
        // White through a mask of blue alone keeps the red.
        let mut commands = layout();
        commands.extend(quad(0xFFFF_0000, 0.5));
        commands.push(kelvin(0x0358, 0x0000_0001));
        commands.extend(quad(0xFFFF_FFFF, 0.5));
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!(pixel(&image, 100, 100), [255, 0, 255, 255]);
    }

    #[test]
    fn a_stencil_clear_keeps_a_later_draw_to_its_rectangle() {
        let Some(mut r) = renderer() else { return };
        let mut commands = layout();
        commands.extend([
            // Stencil 0x80 over the left half.
            kelvin(0x1D8C, 0x80),
            kelvin(0x1D98, 319 << 16),
            kelvin(0x1D9C, 479 << 16),
            kelvin(0x1D94, 2),
            // Draw only where the stencil is 0x80.
            kelvin(0x032C, 1),
            kelvin(0x0364, 0x202),
            kelvin(0x0368, 0x80),
        ]);
        commands.extend(quad(0xFF00_FF00, 0.5));
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!(pixel(&image, 100, 100), GREEN);
        assert_eq!(pixel(&image, 500, 100), [0, 0, 0, 255]);
    }

    #[test]
    fn blend_equations_and_the_constant_colour() {
        let Some(mut r) = renderer() else { return };
        // Destination minus source: red less half red.
        let mut commands = layout();
        commands.extend(quad(0xFFFF_0000, 0.5));
        commands.extend([
            kelvin(0x0304, 1),
            kelvin(0x0344, 1),
            kelvin(0x0348, 1),
            kelvin(0x0350, 0x800B),
        ]);
        commands.extend(quad(0xFF80_0000, 0.5));
        let image = r.render(&commands, &[]).unwrap();
        assert!((i32::from(pixel(&image, 10, 10)[0]) - 127).abs() <= 1);
        // The constant colour as the source factor, the destination dropped.
        let mut commands = layout();
        commands.extend([
            kelvin(0x0304, 1),
            kelvin(0x0344, 0x8001),
            kelvin(0x0348, 0),
            kelvin(0x034C, 0xFF00_FF00),
        ]);
        commands.extend(quad(0xFFFF_FFFF, 0.5));
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!(pixel(&image, 10, 10), GREEN);
    }

    #[test]
    fn culling_drops_the_faces_that_wind_the_culled_way() {
        let Some(mut r) = renderer() else { return };
        // The full-screen quad winds clockwise on the screen.
        let culled = |front: u32| {
            let mut commands = layout();
            commands.extend([
                kelvin(0x0308, 1),
                kelvin(0x039C, 0x405),
                kelvin(0x03A0, front),
            ]);
            commands.extend(quad(0xFF00_FF00, 0.5));
            commands
        };
        let image = r.render(&culled(0x900), &[]).unwrap();
        assert_eq!(pixel(&image, 10, 10), GREEN, "a front face, kept");
        let image = r.render(&culled(0x901), &[]).unwrap();
        assert_eq!(
            pixel(&image, 10, 10),
            [0, 0, 0, 255],
            "a back face, dropped"
        );
    }

    #[test]
    fn an_anti_aliased_surface_is_drawn_at_its_samples_and_shown_averaged() {
        let Some(mut r) = renderer() else { return };
        // A 320x240 surface with 2x2 samples, its left half cleared red.
        let commands = vec![
            kelvin(0x0200, 320 << 16),
            kelvin(0x0204, 240 << 16),
            kelvin(0x0208, 0x2128),
            kelvin(0x1D90, 0xFFFF_0000),
            kelvin(0x1D98, 159 << 16),
            kelvin(0x1D9C, 239 << 16),
            kelvin(0x1D94, 0xF0),
        ];
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!((image.width, image.height), (320, 240));
        assert_eq!(pixel(&image, 159, 120), RED);
        assert_eq!(pixel(&image, 160, 120), [0, 0, 0, 255]);
        let mixed = downsample(
            [[255, 0, 0, 255], [0, 0, 0, 255]].repeat(2).concat(),
            2,
            2,
            [2, 2],
        );
        assert_eq!(mixed.rgba, [128, 0, 0, 255]);
    }

    #[test]
    fn the_surface_clip_and_the_window_clip_keep_draws_to_their_rectangles() {
        let Some(mut r) = renderer() else { return };
        let mut commands = layout();
        commands.extend([
            // The surface clip starts 100 pixels in.
            kelvin(0x0200, (540 << 16) + 100),
            // Drawn outside the window rectangle (110..119 across, 10..19 down), in the shader.
            kelvin(0x02B4, 1),
            kelvin(0x02C0, (119 << 16) + 110),
            kelvin(0x02E0, (19 << 16) + 10),
        ]);
        commands.extend(quad(0xFF00_FF00, 0.5));
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!(
            pixel(&image, 50, 50),
            [0, 0, 0, 255],
            "left of the surface clip"
        );
        assert_eq!(
            pixel(&image, 115, 15),
            [0, 0, 0, 255],
            "inside the window rectangle"
        );
        assert_eq!(pixel(&image, 300, 300), GREEN);
    }

    #[test]
    fn the_specular_colour_reaches_the_combiners_as_register_5() {
        let Some(mut r) = renderer() else { return };
        // Position, diffuse (attribute 3) and specular (attribute 4), packed colours.
        let mut commands = vec![
            kelvin(0x1760, 3 << 4 | 2),
            kelvin(0x1760 + 12, 4 << 4),
            kelvin(0x1760 + 16, 4 << 4),
            kelvin(0x03B8, 1),
        ];
        commands.extend(showing(5));
        commands.push(kelvin(0x17FC, 8));
        for (x, y) in [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)] {
            commands.extend(vertex(x, y, 0.5, 0xFFFF_0000));
            commands.push(kelvin(0x1818, 0xFF00_00FF));
        }
        commands.push(kelvin(0x17FC, 0));
        let image = r.render(&commands, &[]).unwrap();
        assert_eq!(pixel(&image, 10, 10), BLUE);
    }

    #[test]
    fn a_colour_key_drops_the_fragments_of_matching_texels() {
        let Some(mut r) = renderer() else { return };
        let (mut commands, data) = stage_texture(0, 0x1000, TWO_BY_TWO, &[BLUE; 4]);
        // Key mode 3 drops the fragment; the key is the texel, 0xAARRGGBB.
        commands.push(kelvin(
            0x1B0C,
            crate::gpu::texture::DEFAULT_CONTROL0 | 1 << 30 | 3,
        ));
        commands.push(kelvin(0x0AE0, 0xFF00_00FF));
        commands.extend(stage_quad([[0.5, 0.5, 0.0, 1.0]; 4]));
        let image = r.render(&commands, std::slice::from_ref(&data)).unwrap();
        assert_eq!(pixel(&image, 10, 10), [0, 0, 0, 255]);
        let (mut commands, data) = stage_texture(0, 0x1000, TWO_BY_TWO, &[BLUE; 4]);
        commands.push(kelvin(0x0AE0, 0xFF00_FF00));
        commands.extend(stage_quad([[0.5, 0.5, 0.0, 1.0]; 4]));
        let image = r.render(&commands, &[data]).unwrap();
        assert_eq!(pixel(&image, 10, 10), BLUE, "no key mode, no key");
    }

    #[test]
    fn depth_beyond_the_clip_range_is_dropped_or_clamped() {
        let Some(mut r) = renderer() else { return };
        let past = |control: u32| {
            let mut commands = layout();
            commands.extend([
                float(0x0398, 0.5 * Z24),
                kelvin(0x1D78, control),
                kelvin(0x030C, 1),
            ]);
            commands.extend(quad(0xFF00_FF00, 0.75 * Z24));
            commands
        };
        let image = r.render(&past(0), &[]).unwrap();
        assert_eq!(pixel(&image, 10, 10), [0, 0, 0, 255], "dropped");
        let image = r.render(&past(1 << 4), &[]).unwrap();
        assert_eq!(pixel(&image, 10, 10), GREEN, "clamped into the range");
    }
}
