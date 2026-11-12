//! Dumping a frame (spec 12.5): F12 in the window, `ANYXBOX_DUMP_AT` (seconds after boot) or
//! `ANYXBOX_DUMP_FRAME` (frame numbers) writes a frame's GPU commands and the textures it binds,
//! for reading what a frame draws and rendering it again without the title.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};

use super::gpu::TextureData;
use crate::gpu::Command;

/// Set by F12 in the window: the next frame that draws anything is dumped.
static DUMP_NEXT_FRAME: AtomicBool = AtomicBool::new(false);
/// Set while `ANYXBOX_DUMP_FRAME` has frames still to dump.
static NUMBERED_PENDING: AtomicBool = AtomicBool::new(false);

/// Whether a dump may come soon, so frames must carry every texture's bytes.
pub fn wanted() -> bool {
    DUMP_NEXT_FRAME.load(Ordering::Relaxed) || NUMBERED_PENDING.load(Ordering::Relaxed)
}

/// Asks for the next frame that draws anything to be dumped.
pub fn dump_next() {
    DUMP_NEXT_FRAME.store(true, Ordering::Relaxed);
}

/// The frames still to dump: `ANYXBOX_DUMP_FRAME`, a comma-separated list of frame numbers.
#[derive(Debug)]
pub(super) struct Dumps {
    at: VecDeque<u64>,
}

impl Dumps {
    pub(super) fn from_env() -> Self {
        let mut at: Vec<u64> = std::env::var("ANYXBOX_DUMP_FRAME")
            .unwrap_or_default()
            .split(',')
            .filter_map(|text| text.trim().parse().ok())
            .collect();
        at.sort_unstable();
        NUMBERED_PENDING.store(!at.is_empty(), Ordering::Relaxed);
        Self { at: at.into() }
    }

    /// Dumps frame `sequence` if it was asked for: by number (the first frame at or after the
    /// one asked for, as timing varies from run to run) or by F12. Frames that draw little are
    /// passed over.
    pub(super) fn frame(&mut self, sequence: u64, commands: &[Command], textures: &[TextureData]) {
        let asked = self.at.front().is_some_and(|&at| sequence >= at);
        if commands.len() > 100 && (asked || DUMP_NEXT_FRAME.swap(false, Ordering::Relaxed)) {
            if asked {
                self.at.pop_front();
                NUMBERED_PENDING.store(!self.at.is_empty(), Ordering::Relaxed);
            }
            dump_commands(sequence, commands);
            dump_textures(sequence, commands, textures);
        }
    }
}

/// Writes frame `sequence`'s GPU commands to `frame_SEQUENCE.txt`, one `class method value`
/// line each, for reading what a frame draws.
fn dump_commands(sequence: u64, commands: &[Command]) {
    let mut text = String::new();
    for command in commands {
        let _ = writeln!(
            text,
            "{:#06x} {:#06x} {:#010x}",
            command.class, command.method, command.value
        );
    }
    let path = format!("frame_{sequence}.txt");
    match std::fs::write(&path, text) {
        Ok(()) => eprintln!("anyxbox: wrote {} commands to {path}", commands.len()),
        Err(error) => eprintln!("anyxbox: cannot write {path}: {error}"),
    }
}

/// A bound texture's file name: `OFFSET_fFORMAT_WIDTHxHEIGHT`, with `_pPALETTE` for an indexed
/// one.
pub(super) fn texture_name(texture: &crate::gpu::texture::Texture) -> String {
    let mut name = format!(
        "{:08x}_f{:02x}_{}x{}",
        texture.offset, texture.code, texture.width, texture.height
    );
    if texture.palette != 0 {
        let _ = write!(name, "_p{:08x}", texture.palette);
    }
    name
}

/// Writes the textures frame `sequence` binds to `frame_SEQUENCE_textures/`: each one read as
/// `NAME.bin` (the palette after the texels for an indexed one), the decoded ones also as `.ppm`,
/// and `index.txt` listing every texture bound, with those whose bytes were not read marked.
fn dump_textures(sequence: u64, commands: &[Command], textures: &[TextureData]) {
    let dir = std::path::PathBuf::from(format!("frame_{sequence}_textures"));
    if let Err(error) = std::fs::create_dir_all(&dir) {
        eprintln!("anyxbox: cannot create {}: {error}", dir.display());
        return;
    }
    let mut index = String::new();
    for bound in crate::gpu::draw::bound(commands) {
        let name = texture_name(&bound);
        let Some(data) = textures.iter().find(|data| data.texture == bound) else {
            let _ = writeln!(index, "{name} format not read");
            continue;
        };
        let mut bytes = data.bytes.clone();
        bytes.extend_from_slice(&data.palette);
        let _ = std::fs::write(dir.join(format!("{name}.bin")), &bytes);
        let decoded = bound.decode(&data.bytes, &data.palette);
        if let Some(level) = decoded.as_ref().and_then(|levels| levels.first()) {
            let mut ppm = format!("P6\n{} {}\n255\n", level.width, level.height).into_bytes();
            // The first layer: a 3D texture's first slice, a cube map's +X face.
            let layer = (level.width * level.height * 4) as usize;
            for texel in level.rgba[..layer.min(level.rgba.len())].as_chunks::<4>().0 {
                ppm.extend_from_slice(&texel[..3]);
            }
            let _ = std::fs::write(dir.join(format!("{name}.ppm")), ppm);
        }
        let state = if decoded.is_some() {
            "decoded"
        } else {
            "not decoded"
        };
        let _ = writeln!(index, "{name} {} bytes {state}", bytes.len());
    }
    match std::fs::write(dir.join("index.txt"), index) {
        Ok(()) => eprintln!("anyxbox: wrote the textures to {}", dir.display()),
        Err(error) => eprintln!("anyxbox: cannot write {}: {error}", dir.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame dumped with `ANYXBOX_DUMP_FRAME` or F12: its commands, and the textures dumped
    /// beside it.
    fn load(path: &str) -> (Vec<Command>, Vec<TextureData>) {
        let text = std::fs::read_to_string(path).unwrap();
        let word = |text: &str| u32::from_str_radix(text.trim_start_matches("0x"), 16).unwrap();
        let commands: Vec<Command> = text
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                Some(Command {
                    class: word(parts.next()?),
                    method: word(parts.next()?),
                    value: word(parts.next()?),
                })
            })
            .collect();
        // The commands say what each texture is; the files beside them hold its bytes.
        let dir = std::path::PathBuf::from(path.replace(".txt", "_textures"));
        let textures: Vec<TextureData> = crate::gpu::draw::bound(&commands)
            .into_iter()
            .filter_map(|texture| {
                let mut bytes =
                    std::fs::read(dir.join(format!("{}.bin", texture_name(&texture)))).ok()?;
                let palette = bytes.split_off((texture.byte_len() as usize).min(bytes.len()));
                Some(TextureData::new(texture, bytes, palette))
            })
            .collect();
        (commands, textures)
    }

    /// Renders a frame dumped with `ANYXBOX_DUMP_FRAME` or F12, with the textures dumped beside
    /// it, to `ANYXBOX_RENDER_DUMP` with `.ppm` appended: a way to look at a frame without
    /// replaying the title. Skips when the variable is unset.
    #[test]
    fn a_dumped_frame_renders() {
        let Ok(path) = std::env::var("ANYXBOX_RENDER_DUMP") else {
            return;
        };
        let (commands, textures) = load(&path);
        let Ok(mut renderer) = super::super::gpu::Renderer::new() else {
            return;
        };
        let pixels = renderer.render(&commands, &textures).unwrap();
        std::fs::write(format!("{path}.ppm"), super::super::ppm(&pixels)).unwrap();
    }

    /// A frame rendering benchmark on dumps, without running a title:
    /// `ANYXBOX_BENCH_DUMPS=a.txt,b.txt cargo test --release -p runtime --lib bench_dumped --
    /// --nocapture` renders each dump again and again (`ANYXBOX_BENCH_ROUNDS`, 20) and prints
    /// the median and best time per frame, the GPU model's share (decoding the commands into
    /// primitives), and the renderer's alone (drawing primitives decoded once). Each frame is read
    /// back, as the file path does. Skips when unset.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn bench_dumped_frames() {
        let Ok(dumps) = std::env::var("ANYXBOX_BENCH_DUMPS") else {
            return;
        };
        let rounds: usize = std::env::var("ANYXBOX_BENCH_ROUNDS")
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(20)
            .max(1);
        let timed = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1000.0
        };
        let median_and_best = |mut times: Vec<f64>| {
            times.sort_by(f64::total_cmp);
            (times[times.len() / 2], times[0])
        };
        for path in dumps.split(',').filter(|path| !path.is_empty()) {
            let (commands, textures) = load(path);
            let Ok(mut renderer) = super::super::gpu::Renderer::new() else {
                eprintln!("bench: no GPU here; skipping");
                return;
            };
            // The first frames build pipelines and upload textures.
            for _ in 0..3 {
                renderer.render(&commands, &textures);
            }
            let frame: Vec<f64> = (0..rounds)
                .map(|_| {
                    timed(&mut || {
                        std::hint::black_box(renderer.render(&commands, &textures));
                    })
                })
                .collect();
            let model: Vec<f64> = (0..rounds)
                .map(|_| {
                    timed(&mut || {
                        std::hint::black_box(crate::gpu::draw::primitives(&commands));
                    })
                })
                .collect();
            let deferring: Vec<f64> = (0..rounds)
                .map(|_| {
                    timed(&mut || {
                        std::hint::black_box(crate::gpu::draw::defer_only(&commands));
                    })
                })
                .collect();
            let (deferring, _) = median_and_best(deferring);
            let bound = crate::gpu::draw::bound(&commands);
            assert_eq!(bound, crate::gpu::draw::bound_by_replay(&commands));
            let mut binding = (Vec::new(), Vec::new());
            for _ in 0..rounds.min(10) {
                binding.0.push(timed(&mut || {
                    std::hint::black_box(crate::gpu::draw::bound(&commands));
                }));
                binding.1.push(timed(&mut || {
                    std::hint::black_box(crate::gpu::draw::bound_by_replay(&commands));
                }));
            }
            let ((binding, _), (replaying, _)) =
                (median_and_best(binding.0), median_and_best(binding.1));
            eprintln!(
                "bench {path}: the {} bound textures {binding:.2} ms (by replaying every command \
                 {replaying:.2} ms)",
                bound.len()
            );
            eprintln!("bench {path}: the commands without their programs {deferring:.1} ms");
            let primitives = crate::gpu::draw::primitives(&commands);
            // The frame drawn, as a hash of its pixels: a change to the renderer that should not
            // change the picture leaves it alone.
            let pixels = renderer
                .render_primitives(primitives.as_slice(), &textures)
                .map_or(0, |image| {
                    use std::hash::{Hash as _, Hasher as _};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    (image.width, image.height, image.rgba).hash(&mut hasher);
                    hasher.finish()
                });
            let batched = crate::render::batch::batches(&primitives, &mut |_| Some(0));
            let expanded: usize = batched.iter().map(|b| b.vertices.len()).sum();
            eprintln!(
                "bench {path}: {} primitives, {} source vertices; {} batches, {expanded} vertices \
                 drawn ({} MB); pixels {pixels:016x}",
                primitives.len(),
                primitives.iter().map(|p| p.vertices.len()).sum::<usize>(),
                batched.len(),
                (expanded * std::mem::size_of::<crate::render::batch::GpuVertex>()) >> 20,
            );
            let drawing: Vec<f64> = (0..rounds)
                .map(|_| {
                    timed(&mut || {
                        std::hint::black_box(
                            renderer.render_primitives(primitives.as_slice(), &textures),
                        );
                    })
                })
                .collect();
            let ((frame, best), (model, _), (drawing, _)) = (
                median_and_best(frame),
                median_and_best(model),
                median_and_best(drawing),
            );
            eprintln!(
                "bench {path}: {} commands; frame {frame:.1} ms median, {best:.1} ms best \
                 ({:.0} frames/s); of which the GPU model {model:.1} ms; the renderer alone \
                 {drawing:.1} ms",
                commands.len(),
                1000.0 / frame,
            );
            let (draws, vertices, ops) = crate::gpu::draw::program_work(&commands);
            let mut programs = Vec::new();
            crate::gpu::draw::with_program_runs(&commands, |run| {
                programs = (0..rounds)
                    .map(|_| {
                        timed(&mut || {
                            std::hint::black_box(run());
                        })
                    })
                    .collect();
            });
            let (programs, _) = median_and_best(programs);
            let mut interpreted = Vec::new();
            crate::gpu::draw::with_program_only(&commands, |run| {
                interpreted = (0..rounds)
                    .map(|_| {
                        timed(&mut || {
                            std::hint::black_box(run());
                        })
                    })
                    .collect();
            });
            let (interpreted, _) = median_and_best(interpreted);
            let mut alone = Vec::new();
            crate::gpu::draw::with_interpreter_only(&commands, |run| {
                alone = (0..rounds)
                    .map(|_| {
                        timed(&mut || {
                            std::hint::black_box(run());
                        })
                    })
                    .collect();
            });
            let (alone, _) = median_and_best(alone);
            eprintln!(
                "bench {path}: the interpreter alone {alone:.2} ms ({:.2} ns an instruction)",
                alone * 1e6 / ops.max(1) as f64
            );
            eprintln!(
                "bench {path}: {draws} programmed draws, {vertices} vertices, {ops} instructions \
                 run; the programmed draws on one thread {programs:.2} ms ({:.1} ns a vertex, \
                 {:.2} ns an instruction); the programs alone {interpreted:.2} ms ({:.2} ns an \
                 instruction); outputs {:016x}",
                programs * 1e6 / vertices.max(1) as f64,
                programs * 1e6 / ops.max(1) as f64,
                interpreted * 1e6 / ops.max(1) as f64,
                crate::gpu::draw::program_checksum(&commands),
            );
        }
    }
}
