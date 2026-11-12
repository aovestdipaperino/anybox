//! The renderer (spec 11): a frame's primitives drawn with wgpu, each colour buffer at its
//! surface's size.
//!
//! Small integers (texture sizes, fixed-point register fields) become floats freely, and the
//! combiner code names its inputs A, B, C, D as the specification does.
#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

pub mod batch;
pub mod combiner;
pub mod dump;
pub mod gpu;
mod present;
pub mod stages;

pub use dump::dump_next;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::time::{Duration, Instant};

use crate::gpu::Command;
use crate::gpu::draw::{self, Primitive};
use gpu::{Image, Renderer, TextureData};

/// A finished frame on its way to the renderer: its commands and bound textures.
pub type FrameJob = (Vec<Command>, Vec<TextureData>);

/// A frame the GPU model has made into primitives, on its way to be drawn, with how long the
/// model waited for it and then took over it.
type Modelled = (Vec<Command>, Vec<TextureData>, Vec<Primitive>, Times);

/// How long a thread waited for a frame, and how long it then took over it.
#[derive(Debug, Clone, Copy, Default)]
struct Times {
    idle: Duration,
    busy: Duration,
}

impl std::ops::AddAssign for Times {
    fn add_assign(&mut self, other: Self) {
        self.idle += other.idle;
        self.busy += other.busy;
    }
}

/// For [`Rate`], from the kernel's side: the 3D frames the title finished, those dropped because
/// the model was still behind, and the time the thread holding the baton slept (held for the
/// next frame, or nothing to run, or a loop that only waits), when no guest code ran.
pub(crate) static TITLE_FRAMES: AtomicU64 = AtomicU64::new(0);
pub(crate) static DROPPED_FRAMES: AtomicU64 = AtomicU64::new(0);
pub(crate) static GUEST_ASLEEP_NS: AtomicU64 = AtomicU64::new(0);

/// With `ANYXBOX_FPS` set, a line on standard error every two seconds: the frames a second, and
/// for the model and the renderer the time each took over a frame and waited for one. The thread
/// that never waits is the one that holds the frame rate back.
#[derive(Debug)]
struct Rate {
    since: Instant,
    frames: u32,
    model: Times,
    render: Times,
}

impl Rate {
    fn from_env() -> Option<Self> {
        for count in [&TITLE_FRAMES, &DROPPED_FRAMES, &GUEST_ASLEEP_NS] {
            count.store(0, Ordering::Relaxed);
        }
        std::env::var_os("ANYXBOX_FPS").map(|_| Self {
            since: Instant::now(),
            frames: 0,
            model: Times::default(),
            render: Times::default(),
        })
    }

    fn frame(&mut self, model: Times, render: Times) {
        self.frames += 1;
        self.model += model;
        self.render += render;
        let elapsed = self.since.elapsed();
        if elapsed < Duration::from_secs(2) {
            return;
        }
        let each = |d: Duration| d.as_secs_f64() * 1000.0 / f64::from(self.frames);
        let seconds = elapsed.as_secs_f64();
        let title = TITLE_FRAMES.swap(0, Ordering::Relaxed);
        let dropped = DROPPED_FRAMES.swap(0, Ordering::Relaxed);
        let asleep = Duration::from_nanos(GUEST_ASLEEP_NS.swap(0, Ordering::Relaxed));
        // The guest's time a frame of the title's: what it ran, and what it slept.
        let per_title_frame = |d: Duration| d.as_secs_f64() * 1000.0 / (title.max(1) as f64);
        eprintln!(
            "anyxbox: {:.1} frames/s; title {:.1} frames/s ({dropped} dropped), guest {:.1} ms \
             a frame (asleep {:.1}); model {:.1} ms (waiting {:.1}), renderer {:.1} ms \
             (waiting {:.1})",
            f64::from(self.frames) / seconds,
            title as f64 / seconds,
            per_title_frame(elapsed.saturating_sub(asleep)),
            per_title_frame(asleep),
            each(self.model.busy),
            each(self.model.idle),
            each(self.render.busy),
            each(self.render.idle),
        );
        *self = Self {
            since: Instant::now(),
            frames: 0,
            model: Times::default(),
            render: Times::default(),
        };
    }
}

/// Starts the renderer thread when something takes its frames: the window, and `ANYXBOX_FRAMES`
/// (spec 12.5), which writes every `ANYXBOX_FRAMES_EVERY`th rendered frame (30) as
/// `frame_NNNNNN.ppm`, at most `ANYXBOX_FRAMES_LIMIT` (100). Returns where to send frames.
///
/// Frames go through two threads a frame apart: `anyxbox-model` makes a frame's commands into
/// primitives (running its vertex programs) while `anyxbox-render` draws the frame before it,
/// handed over one frame deep.
#[must_use]
pub fn start(window: bool) -> Option<SyncSender<FrameJob>> {
    let dir = std::env::var_os("ANYXBOX_FRAMES").map(PathBuf::from);
    if !window && dir.is_none() {
        return None;
    }
    let number = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let files = dir.map(|dir| {
        (
            dir,
            number("ANYXBOX_FRAMES_EVERY", 30).max(1),
            number("ANYXBOX_FRAMES_LIMIT", 100),
        )
    });
    let (sender, receiver) = std::sync::mpsc::sync_channel::<FrameJob>(2);
    let (modelled, to_draw) = std::sync::mpsc::sync_channel::<Modelled>(1);
    std::thread::Builder::new()
        .name("anyxbox-render".to_owned())
        .spawn(move || render_frames(&to_draw, window, files.as_ref()))
        .ok()?;
    std::thread::Builder::new()
        .name("anyxbox-model".to_owned())
        .spawn(move || {
            let mut waiting = Instant::now();
            for (commands, textures) in receiver {
                let start = Instant::now();
                let primitives = draw::primitives(&commands);
                let times = Times {
                    idle: start - waiting,
                    busy: start.elapsed(),
                };
                if modelled
                    .send((commands, textures, primitives, times))
                    .is_err()
                {
                    return;
                }
                waiting = Instant::now();
            }
        })
        .ok()?;
    Some(sender)
}

/// Frames between copies of a frame on the layer back to the CPU: half a second. The window
/// needs one for the picture the CRT switches off on.
const SNAPSHOT_EVERY: u64 = 30;

/// The renderer thread: renders each frame, shows it in the window, writes every `every`th.
/// Once the window has its GPU layer, frames go there instead of being copied back.
fn render_frames(receiver: &Receiver<Modelled>, window: bool, files: Option<&(PathBuf, u64, u64)>) {
    let mut renderer = match Renderer::new() {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("anyxbox: frames are not rendered: {error}");
            return;
        }
    };
    if let Some((dir, _, _)) = files {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut dumps = dump::Dumps::from_env();
    // Set when the layer could not be drawn to: frames are copied to the window instead.
    let mut no_layer = false;
    let mut since_snapshot = SNAPSHOT_EVERY;
    let (mut sequence, mut drawn, mut written) = (0u64, 0u64, 0u64);
    let mut rate = Rate::from_env();
    // The frame before: when its drawing started, how long the renderer waited for it, and the
    // model's times for it.
    let mut last: Option<(Instant, Duration, Times)> = None;
    loop {
        if let (Some(rate), Some((start, idle, model))) = (&mut rate, last.take()) {
            let busy = start.elapsed();
            rate.frame(model, Times { idle, busy });
        }
        let waiting = Instant::now();
        let Ok((commands, textures, primitives, model)) = receiver.recv() else {
            break;
        };
        last = Some((Instant::now(), waiting.elapsed(), model));
        sequence += 1;
        dumps.frame(sequence, &commands, &textures);
        // A frame of clears alone is drawn, so the target is cleared, but not shown: the title's
        // movie is on the picture beneath.
        let shows = crate::gpu::draw::shows_anything(&primitives);
        let write = files.is_some_and(|(_, every, limit)| {
            (drawn + 1).is_multiple_of(*every) && written < *limit
        });
        if window && !write && !no_layer {
            if !renderer.has_layer()
                && let Some(layer) = crate::host::layer::pointer()
            {
                // SAFETY: the window's layer lives as long as the program.
                match unsafe { Renderer::with_layer(layer) } {
                    Ok(on_layer) => renderer = on_layer,
                    Err(error) => {
                        eprintln!("anyxbox: frames are copied to the window: {error}");
                        no_layer = true;
                    }
                }
            }
            if renderer.has_layer() {
                if !shows {
                    // A size of nothing draws the frame without offering it to the layer.
                    let _ = renderer.present_primitives(primitives, &textures, (0, 0), false);
                    continue;
                }
                let snapshot = since_snapshot >= SNAPSHOT_EVERY;
                let size = crate::host::layer::size();
                if let Some(pixels) =
                    renderer.present_primitives(primitives, &textures, size, snapshot)
                {
                    since_snapshot = if snapshot { 0 } else { since_snapshot + 1 };
                    drawn += 1;
                    crate::host::milestone("3D frame");
                    crate::host::show_on_layer(pixels.as_ref().map(|image| {
                        (
                            image.rgba.as_slice(),
                            image.width as usize,
                            image.height as usize,
                        )
                    }));
                }
                continue;
            }
        }
        let Some(image) = renderer.render_primitives(primitives, &textures) else {
            continue;
        };
        drawn += 1;
        crate::host::milestone("3D frame");
        if window && shows {
            crate::host::show(&image.rgba, 4, image.width as usize, image.height as usize);
        }
        if let Some((dir, _, _)) = files
            && write
        {
            let path = dir.join(format!("frame_{drawn:06}.ppm"));
            if let Err(error) = std::fs::write(&path, ppm(&image)) {
                eprintln!("anyxbox: {}: {error}", path.display());
            }
            written += 1;
        }
    }
}

/// A binary PPM of an image.
fn ppm(image: &Image) -> Vec<u8> {
    let mut out = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
    for pixel in image.rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&pixel[..3]);
    }
    out
}
