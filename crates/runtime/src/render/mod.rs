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
use std::sync::mpsc::{Receiver, SyncSender};

use crate::gpu::Command;
use crate::gpu::draw::{self, Primitive};
use gpu::{Image, Renderer, TextureData};

/// A finished frame on its way to the renderer: its commands and bound textures.
pub type FrameJob = (Vec<Command>, Vec<TextureData>);

/// A frame the GPU model has made into primitives, on its way to be drawn.
type Modelled = (Vec<Command>, Vec<TextureData>, Vec<Primitive>);

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
            for (commands, textures) in receiver {
                let primitives = draw::primitives(&commands);
                if modelled.send((commands, textures, primitives)).is_err() {
                    return;
                }
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
    while let Ok((commands, textures, primitives)) = receiver.recv() {
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
