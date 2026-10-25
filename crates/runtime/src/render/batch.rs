//! From primitives to batches (spec 11.2): clip-space vertices, primitive kinds as triangle and
//! line lists, and consecutive primitives with the same state joined into one batch.

use crate::gpu::draw::{Combiner, Depth, Kind, Primitive, Vertex};
use crate::gpu::pixel::{Cull, DepthRange, Stencil, Surface, WindowClip};
use crate::gpu::texture::{Mode, Shader, Texture};

/// A vertex as the shader reads it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GpuVertex {
    pub clip: [f32; 4],
    pub colour: [f32; 4],
    /// `(r, g, b, 1 - fog)` when fogged, else zero (no fog).
    pub fog: [f32; 4],
    /// The four stages' texture coordinates, `s, t, r, q`.
    pub texture: [[f32; 4]; 4],
    /// The specular colour: combiner register 5.
    pub specular: [f32; 4],
}

impl GpuVertex {
    /// The vertex's bytes, as the vertex buffer holds them.
    #[must_use]
    pub fn bytes(vertices: &[Self]) -> Vec<u8> {
        let mut out = Vec::with_capacity(std::mem::size_of_val(vertices));
        Self::append_bytes(vertices, &mut out);
        out
    }

    /// Appends the vertices' bytes, as the vertex buffer holds them, to `out`.
    pub fn append_bytes(vertices: &[Self], out: &mut Vec<u8>) {
        out.reserve(std::mem::size_of_val(vertices));
        for v in vertices {
            out.extend_from_slice(&v.to_bytes());
        }
    }

    /// The vertex's bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; std::mem::size_of::<Self>()] {
        let mut out = [0; std::mem::size_of::<Self>()];
        let floats = self
            .clip
            .iter()
            .chain(&self.colour)
            .chain(&self.fog)
            .chain(self.texture.as_flattened())
            .chain(&self.specular);
        for (bytes, f) in out.as_chunks_mut::<4>().0.iter_mut().zip(floats) {
            *bytes = f.to_le_bytes();
        }
        out
    }
}

/// How fragments blend: source and destination factors and the equation (GL enums), and the
/// constant colour (`0xAARRGGBB`) the constant factors use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Blend {
    pub source: u32,
    pub destination: u32,
    pub equation: u32,
    pub constant: u32,
}

/// Where on the target a batch may draw, in the target's pixels: inside a scissor (`None` for
/// the whole target), and inside or outside the window clip's rectangles, which the fragment
/// shader tests (`None` when they clip nothing the scissor does not).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Clip {
    pub scissor: Option<[u32; 4]>,
    pub window: Option<WindowClip>,
}

impl Clip {
    /// The scissor within a `width` by `height` target as `set_scissor_rect` takes it; `None`
    /// when nothing of the target is left.
    #[must_use]
    pub fn rect_within(&self, width: u32, height: u32) -> Option<[u32; 4]> {
        let [x, y, w, h] = self.scissor.unwrap_or([0, 0, width, height]);
        let (left, top) = (x.min(width), y.min(height));
        let right = x.saturating_add(w).min(width);
        let bottom = y.saturating_add(h).min(height);
        (right > left && bottom > top).then_some([left, top, right - left, bottom - top])
    }
}

/// Everything that must match for primitives to share a batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    /// The four stages' textures (with their content ids, filled by the caller), their modes,
    /// and what the modes read.
    pub textures: [Option<(Texture, u64)>; 4],
    pub modes: [Mode; 4],
    pub shader: Shader,
    pub depth: Option<Depth>,
    pub blend: Option<Blend>,
    pub combiner: Option<Combiner>,
    pub alpha_test: Option<(u32, u8)>,
    pub lines: bool,
    /// The colour channels written: bit 0 red, 1 green, 2 blue, 3 alpha.
    pub colour_writes: u8,
    pub stencil: Option<Stencil>,
    /// The triangles dropped by their winding (never `Cull::All`: those are not drawn).
    pub cull: Option<Cull>,
    pub clip: Clip,
    pub depth_range: DepthRange,
    /// Each stage's colour key and the texel bits it compares.
    pub colour_keys: [(u32, u32); 4],
    /// Which stages sample the target itself, as drawn so far: a title that samples the buffer
    /// it draws into. Such a batch is drawn alone.
    pub feedback: [bool; 4],
}

/// A batch: one state and its vertices, as a triangle or line list.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub state: State,
    pub vertices: Vec<GpuVertex>,
}

/// The screen position as the rasteriser takes it: in sixteenths of a pixel, truncated.
fn snapped(p: f32) -> f32 {
    (p * 16.0).trunc() / 16.0
}

/// The clip-space vertex of a screen vertex on a surface whose title coordinates span
/// `logical` pixels. (The shader brings linear textures' coordinates to 0..1.)
fn clip(v: &Vertex, logical: [f32; 2], fog: Option<u32>) -> GpuVertex {
    let w = if v.w == 0.0 { 1.0 } else { v.w };
    let fog = fog.map_or([0.0; 4], |colour| {
        let [r, g, b, _] = colour.to_le_bytes();
        [
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
            1.0 - v.fog,
        ]
    });
    let [half_width, half_height] = logical.map(|side| side / 2.0);
    GpuVertex {
        clip: [
            (snapped(v.x) / half_width - 1.0) * w,
            (1.0 - snapped(v.y) / half_height) * w,
            v.depth * w,
            w,
        ],
        colour: v.colour,
        fog,
        texture: v.texture,
        specular: v.specular,
    }
}
/// Brings a clip-space `z` from the depth range to the viewport's 0..1, the viewport mapping it
/// back into the range; w-buffered depth comes from the fragment shader, so `z` is only kept
/// inside the range.
fn place(range: &DepthRange, clip: &mut [f32; 4]) {
    if range.w_buffer.is_some() {
        clip[2] = 0.5 * clip[3];
        return;
    }
    let (min, max) = viewport(range);
    if (min, max) != (0.0, 1.0) {
        clip[2] = (clip[2] - min * clip[3]) / (max - min).max(f32::EPSILON);
    }
}

/// The part of 0..1 a batch's viewport maps depth to: its range, held inside the buffer.
#[must_use]
pub fn viewport(range: &DepthRange) -> (f32, f32) {
    if range.w_buffer.is_some() {
        return (0.0, 1.0);
    }
    let min = range.min.clamp(0.0, 1.0);
    (min, range.max.clamp(min, 1.0))
}

/// Where on its surface a primitive may draw: the surface clip as a scissor, narrowed by the
/// window clip when that is one rectangle drawn inside, else with the window clip for the
/// fragment shader to test. What covers the whole surface is left out.
#[must_use]
pub fn clip_of(target: &Surface) -> Clip {
    let [width, height] = target.size;
    let window = target.window;
    let first = window.rects[0];
    let one_rectangle = !window.exclusive && window.rects.iter().all(|r| *r == first);
    let mut scissor = target.scissor;
    if one_rectangle {
        let span = |from: u32, size: u32, min: u32, max: u32| {
            let start = from.max(min);
            (
                start,
                from.saturating_add(size).min(max).saturating_sub(start),
            )
        };
        let [x, y, w, h] = scissor;
        let ((x, w), (y, h)) = (
            span(x, w, first[0], first[2]),
            span(y, h, first[1], first[3]),
        );
        scissor = [x, y, w, h];
    }
    // An exclusive rectangle that is empty or off the surface takes nothing away.
    let takes_away = |[left, top, right, bottom]: [u32; 4]| {
        left < right && top < bottom && left < width && top < height
    };
    let tested =
        !one_rectangle && (!window.exclusive || window.rects.iter().any(|r| takes_away(*r)));
    let whole = scissor[0] == 0 && scissor[1] == 0 && scissor[2] >= width && scissor[3] >= height;
    Clip {
        scissor: (!whole).then_some(scissor),
        window: tested.then_some(window),
    }
}

/// The colour channels `SET_COLOR_MASK` (alpha bit 24, red 16, green 8, blue 0) lets through,
/// as bits 0 red, 1 green, 2 blue, 3 alpha.
#[must_use]
pub fn colour_writes(mask: u32) -> u8 {
    [16, 8, 0, 24]
        .iter()
        .enumerate()
        .filter(|(_, shift)| (mask >> **shift) & 1 != 0)
        .fold(0, |bits, (bit, _)| bits | 1 << bit)
}

/// The texel bits a colour key compares, `0xAARRGGBB`: formats without alpha leave it out.
fn colour_key_mask(texture: &Texture) -> u32 {
    match texture.code {
        // X1R5G5B5 and X8R8G8B8, swizzled and linear.
        0x03 | 0x07 | 0x1C | 0x1E => 0x00FF_FFFF,
        _ => 0xFFFF_FFFF,
    }
}

/// The indices of a primitive's vertices as a triangle list, or a line list for lines; none for
/// points.
#[must_use]
pub fn indices(kind: Kind, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    match kind {
        Kind::Points => {}
        Kind::Lines => out.extend(0..n - n % 2),
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
        Kind::TriangleFan => {
            for i in 1..n.saturating_sub(1) {
                out.extend([0, i, i + 1]);
            }
        }
        Kind::Quads => {
            for b in (0..n - n % 4).step_by(4) {
                out.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
            }
        }
    }
    out
}

/// Joins `primitives` into batches. `content_id` gives a texture's content id, or `None` when its
/// bytes are not available (it then samples white).
pub fn batches<'a>(
    primitives: impl IntoIterator<Item = &'a Primitive>,
    content_id: &mut dyn FnMut(&Texture) -> Option<u64>,
) -> Vec<Batch> {
    let mut out: Vec<Batch> = Vec::new();
    for p in primitives {
        let lines = p.kind == Kind::Lines;
        let cull = if matches!(p.kind, Kind::Lines | Kind::Points) {
            None
        } else {
            p.pixel.cull
        };
        if cull == Some(Cull::All) {
            continue;
        }
        let feedback = p
            .textures
            .map(|t| t.is_some_and(|t| t.offset == p.surface && t.linear()));
        let textures: [Option<(Texture, u64)>; 4] = std::array::from_fn(|s| {
            let t = p.textures[s]?;
            // The target stands in for the texture; its id is not looked up.
            if feedback[s] {
                Some((t, 0))
            } else {
                content_id(&t).map(|id| (t, id))
            }
        });
        let blend = p.blend.map(|(source, destination)| Blend {
            source,
            destination,
            equation: p.pixel.blend_equation,
            constant: p.pixel.blend_colour,
        });
        let state = State {
            textures,
            modes: p.modes,
            shader: p.shader,
            depth: p.depth,
            blend,
            combiner: p.combiner,
            alpha_test: p.alpha_test,
            lines,
            colour_writes: colour_writes(p.pixel.colour_mask),
            stencil: p.pixel.stencil,
            cull,
            clip: clip_of(&p.pixel.target),
            depth_range: p.pixel.depth_range,
            colour_keys: std::array::from_fn(|s| {
                textures[s].map_or((0, 0), |(t, _)| {
                    (p.pixel.colour_keys[s], colour_key_mask(&t))
                })
            }),
            feedback,
        };
        let order = indices(p.kind, p.vertices.len());
        if order.is_empty() {
            continue;
        }
        let alone = feedback.contains(&true);
        let joins = out.last().is_some_and(|last| {
            !alone && !last.state.feedback.contains(&true) && last.state == state
        });
        if !joins {
            out.push(Batch {
                state,
                vertices: Vec::with_capacity(order.len()),
            });
        }
        let Some(batch) = out.last_mut() else {
            continue;
        };
        let logical = p.pixel.target.logical_size();
        batch.vertices.extend(order.into_iter().map(|i| {
            let mut v = clip(&p.vertices[i], logical, p.fog);
            if lines {
                // Pulled toward the eye so outlines pass the depth test.
                v.clip[2] -= 0.001 * v.clip[3];
            }
            place(&state.depth_range, &mut v.clip);
            v
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(x: f32, y: f32) -> Vertex {
        Vertex {
            x,
            y,
            depth: 0.5,
            w: 2.0,
            colour: [1.0; 4],
            texture: [[0.0, 0.0, 0.0, 1.0]; 4],
            fog: 1.0,
            specular: [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn primitive(kind: Kind, n: usize) -> Primitive {
        Primitive {
            kind,
            vertices: (0..n).map(|i| vertex(i as f32, 0.0)).collect(),
            textures: [None; 4],
            modes: [Mode::Sample; 4],
            shader: Shader::default(),
            surface: 0,
            depth: None,
            blend: None,
            fog: None,
            combiner: None,
            alpha_test: None,
            point_size: 1.0,
            pixel: crate::gpu::pixel::Pixel::default(),
            clear: false,
        }
    }

    #[test]
    fn every_kind_becomes_a_list() {
        assert_eq!(indices(Kind::Triangles, 7), [0, 1, 2, 3, 4, 5]);
        assert_eq!(indices(Kind::TriangleStrip, 5), [0, 1, 2, 2, 1, 3, 2, 3, 4]);
        assert_eq!(indices(Kind::TriangleFan, 5), [0, 1, 2, 0, 2, 3, 0, 3, 4]);
        assert_eq!(
            indices(Kind::Quads, 8),
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        );
        assert_eq!(indices(Kind::Lines, 5), [0, 1, 2, 3]);
        assert_eq!(indices(Kind::Points, 5), Vec::<usize>::new());
        assert_eq!(indices(Kind::TriangleFan, 1), Vec::<usize>::new());
    }

    #[test]
    fn screen_coordinates_become_clip_space() {
        let screen = [640.0, 480.0];
        let v = clip(&vertex(480.0, 120.0), screen, None);
        assert_eq!(v.clip, [1.0, 1.0, 1.0, 2.0]);
        assert_eq!(v.fog, [0.0; 4]);
        let fogged = clip(
            &Vertex {
                fog: 0.25,
                ..vertex(0.0, 0.0)
            },
            screen,
            Some(0x00FF_0000),
        );
        assert_eq!(fogged.fog, [0.0, 0.0, 1.0, 0.75]);
    }

    #[test]
    fn consecutive_primitives_with_the_same_state_join_and_points_are_dropped() {
        let mut blended = primitive(Kind::Triangles, 3);
        blended.blend = Some((0x302, 0x303));
        let prims = [
            primitive(Kind::Triangles, 3),
            primitive(Kind::Quads, 4),
            blended,
            primitive(Kind::Points, 4),
            primitive(Kind::Lines, 2),
        ];
        let b = batches(&prims, &mut |_| None);
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].vertices.len(), 3 + 6);
        assert_eq!(
            b[1].state.blend,
            Some(Blend {
                source: 0x302,
                destination: 0x303,
                equation: 0x8006,
                constant: 0
            })
        );
        assert!(b[2].state.lines);
        // Lines are pulled toward the eye by 0.001 * w.
        assert!((b[2].vertices[0].clip[2] - (0.5 * 2.0 - 0.002)).abs() < 1e-6);
    }

    #[test]
    fn a_vertex_is_128_bytes() {
        assert_eq!(std::mem::size_of::<GpuVertex>(), 128);
        assert_eq!(GpuVertex::bytes(&[GpuVertex::default()]).len(), 128);
    }

    #[test]
    fn positions_take_the_surfaces_own_size_in_sixteenths_of_a_pixel() {
        // A 1280x720 surface: its far corner is clip space's.
        let v = clip(&vertex(1280.0, 720.0), [1280.0, 720.0], None);
        assert_eq!(v.clip[..2], [2.0, -2.0]);
        let v = clip(&vertex(320.03, 240.0), [640.0, 480.0], None);
        assert_eq!(v.clip[..2], [0.0, 0.0]);
    }

    #[test]
    fn the_colour_mask_becomes_the_channels_written() {
        assert_eq!(colour_writes(0x0101_0101), 0xF);
        assert_eq!(colour_writes(0x0100_0000), 0b1000);
        assert_eq!(colour_writes(0x0000_0001), 0b0100);
        assert_eq!(colour_writes(0x0001_0000), 0b0001);
    }

    #[test]
    fn the_window_clip_folds_into_the_scissor_when_one_rectangle_draws_inside() {
        let mut target = Surface::default();
        assert_eq!(clip_of(&target), Clip::default(), "open: nothing to clip");
        target.window.rects = [[0, 0, 640, 480]; 8];
        assert_eq!(clip_of(&target), Clip::default());
        target.scissor = [100, 0, 540, 480];
        target.window.rects = [[0, 40, 320, 480]; 8];
        let clip = clip_of(&target);
        assert_eq!(clip.scissor, Some([100, 40, 220, 440]));
        assert_eq!(clip.window, None);
        // Two rectangles, or one to draw outside of, are the shader's.
        target.window.rects[7] = [600, 0, 640, 10];
        assert!(clip_of(&target).window.is_some());
        target.window.rects = [[0; 4]; 8];
        target.window.exclusive = true;
        assert_eq!(
            clip_of(&target).window,
            None,
            "empty rectangles take nothing"
        );
        target.window.rects[2] = [10, 10, 20, 20];
        assert!(clip_of(&target).window.is_some_and(|w| w.exclusive));
        let scissor = |rect| Clip {
            scissor: Some(rect),
            window: None,
        };
        assert_eq!(
            scissor([600, 0, 100, 10]).rect_within(640, 480),
            Some([600, 0, 40, 10])
        );
        assert_eq!(scissor([700, 0, 10, 10]).rect_within(640, 480), None);
    }

    #[test]
    fn a_draw_that_samples_its_own_target_is_a_batch_of_its_own() {
        let stage = crate::gpu::texture::StageRegs {
            offset: 0x1000,
            format: 0x12 << 8,
            rect: (640 << 16) + 480,
            ..Default::default()
        };
        let mut sampling = primitive(Kind::Triangles, 3);
        sampling.textures[0] = stage.texture();
        sampling.surface = 0x1000;
        let mut elsewhere = sampling.clone();
        elsewhere.surface = 0x2000;
        let prims = [sampling.clone(), sampling, elsewhere.clone(), elsewhere];
        let b = batches(&prims, &mut |_| Some(7));
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].state.feedback, [true, false, false, false]);
        assert_eq!(b[0].state.textures[0].map(|(_, id)| id), Some(0));
        assert_eq!(b[2].state.feedback, [false; 4]);
        assert_eq!(b[2].vertices.len(), 6);
    }

    #[test]
    fn culled_faces_and_the_depth_range() {
        let mut both = primitive(Kind::Triangles, 3);
        both.pixel.cull = Some(Cull::All);
        let mut lines = primitive(Kind::Lines, 2);
        lines.pixel.cull = Some(Cull::Clockwise);
        let b = batches(&[both, lines], &mut |_| None);
        assert_eq!(b.len(), 1, "front and back culled: nothing drawn");
        assert_eq!(b[0].state.cull, None, "lines are never culled");
        let range = DepthRange {
            min: 0.25,
            max: 0.75,
            clamp: false,
            w_buffer: None,
        };
        // Depth 0.75 (z 1.5 at w 2) is the far end.
        let mut z = [0.0, 0.0, 1.5, 2.0];
        place(&range, &mut z);
        assert!((z[2] / z[3] - 1.0).abs() < 1e-6);
        assert_eq!(viewport(&range), (0.25, 0.75));
    }
}
