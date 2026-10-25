//! A benchmark of the push-buffer intake on the title's side, without running a title:
//! `ANYXBOX_BENCH_DUMPS=a.txt,b.txt cargo test --release -p runtime --lib bench_intake --
//! --nocapture` turns each dumped frame back into a push buffer, with its array draws back as
//! `DRAW_ARRAYS` over vertex buffers in guest memory, pushes it through [`Gpu::write`] again and
//! again (`ANYXBOX_BENCH_ROUNDS`, 20), and prints the median and best time per frame. Each round
//! checks that the frame recorded is the frame dumped. Skips when unset.

use super::{
    CACHE1_DMA_GET, CACHE1_DMA_INSTANCE, CACHE1_DMA_PUSH, CACHE1_DMA_PUT, CACHE1_PULL0,
    CACHE1_PUSH0, Command, Gpu, PFIFO_RAMHT, PGRAPH_CTX_CONTROL, PGRAPH_FIFO, PRAMIN,
    guest_address,
};
use crate::mem::Memory;

const KELVIN: u32 = 0x97;
const SET_OBJECT: u32 = 0x000;
const NO_OPERATION: u32 = 0x100;
const ARRAY_OFFSET: u32 = 0x1720;
const ARRAY_FORMAT: u32 = 0x1760;
const DRAW_ARRAYS: u32 = 0x1810;
const INLINE_ARRAY: u32 = 0x1818;

/// Where the push buffer is: physical, and its DMA object's instance.
const PUSH_PHYSICAL: u32 = 0x0380_0000;
const PUSH_INSTANCE: u32 = 0x8000;
/// The 3D object's instance and handle.
const KELVIN_INSTANCE: u32 = 0x8100;
const KELVIN_HANDLE: u32 = 0xBEEF;
/// The most words between two kicks of `PUT`.
const KICK: usize = 16 * 1024;

/// A run of the push buffer and the vertex data its draws read, written before it is kicked.
struct Segment {
    writes: Vec<(u32, u32)>,
    put: u32,
}

/// A dumped frame as a push buffer.
struct Stream {
    words: Vec<u32>,
    segments: Vec<Segment>,
    /// The commands the frame should record.
    expected: Vec<Command>,
    /// The handles the methods `0x180..0x200` name, with the instance each stands for.
    handles: Vec<(u32, u32)>,
}

fn load(path: &str) -> Vec<Command> {
    let text = std::fs::read_to_string(path).unwrap();
    let word = |text: &str| u32::from_str_radix(text.trim_start_matches("0x"), 16).unwrap();
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            Some(Command {
                class: word(parts.next()?),
                method: word(parts.next()?),
                value: word(parts.next()?),
            })
        })
        .collect()
}

/// The words an attribute of `format` takes in a vertex.
fn attribute_words(format: u32) -> u32 {
    let (kind, size) = (format & 0xF, (format >> 4) & 0xF);
    match kind {
        _ if size == 0 => 0,
        0 | 4 | 6 => 1,
        1 | 5 => size.div_ceil(2),
        _ => size,
    }
}

/// The RAMHT hash of `handle` on channel 0 for a 4 KiB table.
fn ramht_hash(handle: u32) -> u32 {
    let mut hash = 0;
    let mut rest = handle;
    while rest != 0 {
        hash ^= rest & 0x7FF;
        rest >>= 11;
    }
    hash
}

impl Stream {
    #[allow(clippy::too_many_lines)]
    fn new(commands: &[Command]) -> Self {
        let commands: Vec<Command> = commands
            .iter()
            .copied()
            .filter(|c| {
                c.class == KELVIN
                    && c.method != SET_OBJECT
                    && !(c.method == NO_OPERATION && c.value != 0)
            })
            .collect();
        let mut handles: Vec<(u32, u32)> = Vec::new();
        let mut handle_of = |instance: u32| {
            if let Some(&(handle, _)) = handles.iter().find(|h| h.1 == instance) {
                return handle;
            }
            let mut handle = 0x5000_0000 | instance;
            while handle == KELVIN_HANDLE
                || ramht_hash(handle) == ramht_hash(KELVIN_HANDLE)
                || handles
                    .iter()
                    .any(|h| ramht_hash(h.0) == ramht_hash(handle))
            {
                handle += 0x1_0000;
            }
            handles.push((handle, instance));
            handle
        };
        let mut words = vec![1 << 18 | SET_OBJECT, KELVIN_HANDLE];
        let mut expected = Vec::new();
        let mut segments = Vec::new();
        let mut writes: Vec<(u32, u32)> = Vec::new();
        let mut written = std::collections::HashMap::new();
        let mut formats = [0u32; 16];
        let mut offsets = [0u32; 16];
        // The open run of methods: its first method, whether it does not increase, its values.
        let mut run: Option<(u32, bool, Vec<u32>)> = None;
        let flush = |run: &mut Option<(u32, bool, Vec<u32>)>, words: &mut Vec<u32>| {
            if let Some((method, same, values)) = run.take() {
                let count = u32::try_from(values.len()).unwrap();
                words.push(u32::from(same) << 30 | count << 18 | method);
                words.extend(values);
            }
        };
        let mut i = 0;
        while i < commands.len() {
            let Command { method, value, .. } = commands[i];
            i += 1;
            if (ARRAY_FORMAT..ARRAY_FORMAT + 64).contains(&method) {
                formats[((method - ARRAY_FORMAT) / 4) as usize] = value;
            }
            if (ARRAY_OFFSET..ARRAY_OFFSET + 64).contains(&method) {
                offsets[((method - ARRAY_OFFSET) / 4) as usize] = value;
            }
            if method == DRAW_ARRAYS {
                // The inline words after it are its vertices: they go back into memory.
                let per_vertex: u32 = formats.iter().map(|&f| attribute_words(f)).sum();
                let first = value & 0x00FF_FFFF;
                let count = (value >> 24) + 1;
                let len = (per_vertex * count) as usize;
                let inline = commands.get(i..i + len).unwrap_or_default();
                if inline.len() == len && inline.iter().all(|c| c.method == INLINE_ARRAY) {
                    let mut at = 0;
                    let mut draw = Vec::new();
                    for index in first..first + count {
                        for (format, offset) in formats.iter().zip(offsets) {
                            let stride = format >> 8;
                            let base = (offset & 0x7FFF_FFFF).wrapping_add(index * stride);
                            for w in 0..attribute_words(*format) {
                                let address = guest_address(base + 4 * w);
                                draw.push((address, inline[at].value));
                                at += 1;
                            }
                        }
                    }
                    // Vertex data the frame wrote over: what came before is kicked first.
                    let conflict = draw
                        .iter()
                        .any(|(a, v)| written.get(a).is_some_and(|old| old != v));
                    if conflict {
                        flush(&mut run, &mut words);
                        segments.push(Segment {
                            writes: std::mem::take(&mut writes),
                            put: 4 * u32::try_from(words.len()).unwrap(),
                        });
                        written.clear();
                    }
                    written.extend(draw.iter().copied());
                    writes.extend(draw);
                    expected.extend_from_slice(inline);
                    i += len;
                    flush(&mut run, &mut words);
                    words.extend([1 << 18 | DRAW_ARRAYS, value]);
                    continue;
                }
            }
            expected.push(commands[i - 1]);
            let value = if (0x180..0x200).contains(&method) {
                handle_of(value)
            } else {
                value
            };
            let extends = run.as_ref().is_some_and(|(first, same, values)| {
                let n = u32::try_from(values.len()).unwrap();
                n < 2047
                    && if *same {
                        method == *first
                    } else {
                        method == first + 4 * n && method != INLINE_ARRAY
                    }
            });
            if extends {
                run.as_mut().unwrap().2.push(value);
            } else {
                flush(&mut run, &mut words);
                run = Some((method, method == INLINE_ARRAY, vec![value]));
            }
            let open = words.len() + run.as_ref().map_or(0, |r| r.2.len() + 1);
            let kicked = segments.last().map_or(0, |s: &Segment| s.put as usize / 4);
            if open - kicked >= KICK {
                flush(&mut run, &mut words);
                segments.push(Segment {
                    writes: std::mem::take(&mut writes),
                    put: 4 * u32::try_from(words.len()).unwrap(),
                });
            }
        }
        flush(&mut run, &mut words);
        segments.push(Segment {
            writes,
            put: 4 * u32::try_from(words.len()).unwrap(),
        });
        Self {
            words,
            segments,
            expected,
            handles,
        }
    }
}

/// A GPU with the push buffer's DMA object, the 3D object and every handle in RAMHT, and the
/// FIFO running.
fn gpu(memory: &Memory, stream: &Stream) -> Gpu {
    let mut gpu = Gpu::new();
    let put = |gpu: &mut Gpu, offset: u32, value: u32| gpu.write(offset, 4, value, memory);
    // RAMHT at PRAMIN + 0x4000, clear of the dumped DMA objects.
    put(&mut gpu, PFIFO_RAMHT, 0x40);
    put(&mut gpu, PRAMIN + PUSH_INSTANCE, 0);
    put(&mut gpu, PRAMIN + PUSH_INSTANCE + 4, 0x00FF_FFFF);
    put(&mut gpu, PRAMIN + PUSH_INSTANCE + 8, PUSH_PHYSICAL);
    put(&mut gpu, PRAMIN + KELVIN_INSTANCE, KELVIN);
    let bind = |gpu: &mut Gpu, handle: u32, instance: u32, graphics: bool| {
        let at = PRAMIN + 0x4000 + ramht_hash(handle) * 8;
        put(gpu, at, handle);
        put(
            gpu,
            at + 4,
            1 << 31 | u32::from(graphics) << 16 | instance >> 4,
        );
    };
    bind(&mut gpu, KELVIN_HANDLE, KELVIN_INSTANCE, true);
    for &(handle, instance) in &stream.handles {
        bind(&mut gpu, handle, instance, false);
        // Every DMA object starts at 0 and reaches everything.
        put(&mut gpu, PRAMIN + instance, 0);
        put(&mut gpu, PRAMIN + instance + 4, 0xFFFF_FFFF);
        put(&mut gpu, PRAMIN + instance + 8, 0);
    }
    put(&mut gpu, PGRAPH_CTX_CONTROL, 1 << 16);
    put(&mut gpu, PGRAPH_FIFO, 1);
    put(&mut gpu, CACHE1_PULL0, 1);
    put(&mut gpu, CACHE1_DMA_INSTANCE, PUSH_INSTANCE >> 4);
    put(&mut gpu, CACHE1_DMA_PUSH, 1);
    put(&mut gpu, CACHE1_PUSH0, 1);
    gpu
}

#[test]
fn bench_intake() {
    let Ok(dumps) = std::env::var("ANYXBOX_BENCH_DUMPS") else {
        return;
    };
    let rounds: usize = std::env::var("ANYXBOX_BENCH_ROUNDS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(20)
        .max(1);
    let memory = Memory::new().unwrap();
    for path in dumps.split(',') {
        let stream = Stream::new(&load(path));
        let bytes: Vec<u8> = stream.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(memory.write(guest_address(PUSH_PHYSICAL), &bytes));
        let mut gpu = gpu(&memory, &stream);
        gpu.record_frames(true);
        let mut times = Vec::new();
        for _ in 0..rounds {
            // The pusher stops while the pointers go back to the start.
            gpu.write(CACHE1_DMA_PUSH, 4, 0, &memory);
            gpu.write(CACHE1_DMA_PUT, 4, 0, &memory);
            gpu.write(CACHE1_DMA_GET, 4, 0, &memory);
            gpu.write(CACHE1_DMA_PUSH, 4, 1, &memory);
            let mut spent = std::time::Duration::ZERO;
            for segment in &stream.segments {
                for &(address, value) in &segment.writes {
                    memory.write(address, &value.to_le_bytes());
                }
                let start = std::time::Instant::now();
                gpu.write(CACHE1_DMA_PUT, 4, segment.put, &memory);
                spent += start.elapsed();
            }
            assert_eq!(gpu.stall, None, "{}", gpu.fifo_state());
            let frames = gpu.take_frames();
            assert_eq!(
                frames.len(),
                1,
                "one frame a round: {} of {} bytes",
                gpu.fifo_state(),
                4 * stream.words.len()
            );
            let recorded = &frames[0].commands;
            let tail = &recorded[recorded.len() - stream.expected.len()..];
            if let Some(at) = tail.iter().zip(&stream.expected).position(|(a, b)| a != b) {
                panic!(
                    "command {at} recorded {:x?}, dumped {:x?}",
                    tail[at], stream.expected[at]
                );
            }
            times.push(spent);
        }
        times.sort();
        eprintln!(
            "{path}: {} words, {} kicks, {} commands: median {:.2} ms, best {:.2} ms a frame",
            stream.words.len(),
            stream.segments.len(),
            stream.expected.len(),
            times[times.len() / 2].as_secs_f64() * 1e3,
            times[0].as_secs_f64() * 1e3,
        );
    }
}
