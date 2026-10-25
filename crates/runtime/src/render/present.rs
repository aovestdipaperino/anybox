//! Showing frames on the window's Core Animation layer (spec 11.6), from a thread of its own:
//! each is scaled to the layer's size by a full-screen triangle. Getting the layer's next image
//! waits for the display to give one back, so the renderer never does: it hands its newest frame
//! over, and the presenting thread shows the latest one it was given. Now and then a frame is
//! also copied back for the CPU, without waiting for it.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission.

/// Shows frames on a layer; see the module documentation.
#[derive(Debug)]
pub(super) struct Presenter {
    shared: std::sync::Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// A copy of a frame on its way back to the CPU.
    pub(super) snapshot: Option<Snapshot>,
}

/// What the presenting thread works with.
#[derive(Debug)]
struct Shared {
    surface: wgpu::Surface<'static>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    state: std::sync::Mutex<Handover>,
    wake: std::sync::Condvar,
    /// Held around every submission and around configuring the surface: configuring waits
    /// for the GPU to finish, and fails if another thread submits meanwhile.
    submitting: std::sync::Mutex<()>,
}

/// The frame waiting to be shown, and whether the presenting thread must stop.
#[derive(Debug, Default)]
struct Handover {
    next: Option<(wgpu::Texture, (u32, u32))>,
    stop: bool,
}

/// A frame being copied back: the buffer, its rows, and whether it can be read yet.
#[derive(Debug)]
pub(super) struct Snapshot {
    buffer: wgpu::Buffer,
    pub(super) width: u32,
    pub(super) height: u32,
    padded: u32,
    ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// The shader that scales a frame onto the layer.
const BLIT_SHADER: &str = r"
@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle that covers the target: uv (0,0) is the picture's top left.
@vertex
fn vs(@builtin(vertex_index) index: u32) -> Out {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Out;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(picture, linear_sampler, in.uv).rgb, 1.0);
}
";

impl Snapshot {
    /// Records copying `target` (`width` x `height`, RGBA8) into a buffer to read back.
    pub(super) fn start(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Self {
        let row = width * 4;
        let padded =
            row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("snapshot"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Self {
            buffer,
            width,
            height,
            padded,
            ready: std::sync::Arc::default(),
        }
    }

    /// Asks for the buffer to be mapped once the copy is done; [`Snapshot::ready`] says when.
    pub(super) fn map(&self) {
        let ready = std::sync::Arc::clone(&self.ready);
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                ready.store(result.is_ok(), std::sync::atomic::Ordering::Release);
            });
    }

    /// Whether the pixels have arrived.
    pub(super) fn ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The pixels, RGBA8 row by row, once [`Snapshot::ready`].
    pub(super) fn pixels(self) -> Vec<u8> {
        let row = (self.width * 4) as usize;
        let pixels = self.buffer.slice(..).get_mapped_range().map_or_else(
            |_| Vec::new(),
            |data| {
                let mut pixels = Vec::with_capacity(row * self.height as usize);
                for line in data.chunks(self.padded as usize).take(self.height as usize) {
                    pixels.extend_from_slice(&line[..row]);
                }
                pixels
            },
        );
        self.buffer.unmap();
        pixels
    }
}

impl Presenter {
    /// Builds the scaling pipeline and starts the presenting thread.
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: wgpu::Surface<'static>,
        adapter: wgpu::Adapter,
    ) -> Result<Self, String> {
        // The frames hold the title's colours as they are, so the layer must not encode them
        // again: a format without sRGB.
        let formats = surface.get_capabilities(&adapter).formats;
        let format = formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .or_else(|| formats.first().copied())
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let shared = std::sync::Arc::new(Shared {
            surface,
            adapter,
            device: device.clone(),
            queue: queue.clone(),
            format,
            pipeline,
            layout,
            sampler,
            state: std::sync::Mutex::default(),
            wake: std::sync::Condvar::new(),
            submitting: std::sync::Mutex::new(()),
        });
        let thread = std::thread::Builder::new()
            .name("anyxbox-present".to_owned())
            .spawn({
                let shared = std::sync::Arc::clone(&shared);
                move || shared.show_frames()
            })
            .map_err(|e| format!("cannot start presenting: {e}"))?;
        Ok(Self {
            shared,
            thread: Some(thread),
            snapshot: None,
        })
    }

    /// Submits `encoder`, never while the presenting thread configures the surface.
    pub(super) fn submit(&self, encoder: wgpu::CommandEncoder) {
        let _submitting = self
            .shared
            .submitting
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.shared.queue.submit([encoder.finish()]);
    }

    /// Hands `frame` to the presenting thread, to show at `size` pixels in place of any frame it
    /// has not shown yet.
    pub(super) fn offer(&self, frame: wgpu::Texture, size: (u32, u32)) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.next = Some((frame, size));
        self.shared.wake.notify_one();
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stop = true;
        self.shared.wake.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Shared {
    /// The presenting thread: shows each frame handed over, until told to stop.
    fn show_frames(&self) {
        let mut configured = None;
        loop {
            let (picture, size) = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                loop {
                    if state.stop {
                        return;
                    }
                    if let Some(next) = state.next.take() {
                        break next;
                    }
                    state = self
                        .wake
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            };
            if configured != Some(size) {
                let _submitting = self
                    .submitting
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                self.configure(size);
                configured = Some(size);
            }
            let frame = match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    // The layer changed under the surface: configure it again.
                    configured = None;
                    continue;
                }
                _ => continue,
            };
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.blit(&mut encoder, &picture, &frame.texture);
            self.queue.submit([encoder.finish()]);
            self.queue.present(frame);
        }
    }

    /// Sizes the layer's images to `size` pixels. Frames wait for the display's refresh, so
    /// none tears; only the presenting thread waits for it.
    fn configure(&self, size: (u32, u32)) {
        let caps = self.surface.get_capabilities(&self.adapter);
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto)
        };
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                width: size.0,
                height: size.1,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode,
                view_formats: vec![],
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );
    }

    /// Records drawing `picture` over the whole of `frame`.
    fn blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        picture: &wgpu::Texture,
        frame: &wgpu::Texture,
    ) {
        let source = picture.create_view(&wgpu::TextureViewDescriptor::default());
        let target = frame.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
