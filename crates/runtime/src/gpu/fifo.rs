//! The push-buffer FIFO (spec 10.3, 10.4, 10.5): the pusher reads the push buffer and decodes its
//! commands; the puller hands each method to its object; the 3D methods are recorded.

use serde::{Deserialize, Serialize};

use super::{
    Bus, CACHE1_DMA_GET, CACHE1_DMA_INSTANCE, CACHE1_DMA_PUSH, CACHE1_DMA_PUT, CACHE1_DMA_STATE,
    CACHE1_DMA_SUBROUTINE, CACHE1_PULL0, CACHE1_PUSH0, CACHE1_PUSH1, Frame, Gpu,
    INTR_CONTEXT_SWITCH, INTR_ERROR, PFIFO_RAMHT, PGRAPH_COLOR_CLEAR_VALUE, PGRAPH_CTX_CACHE1,
    PGRAPH_CTX_CONTROL, PGRAPH_CTX_SWITCH1, PGRAPH_CTX_USER, PGRAPH_FIFO, PGRAPH_INTR,
    PGRAPH_NSOURCE, PGRAPH_TRAPPED_ADDR, PGRAPH_TRAPPED_DATA_LOW, PGRAPH_ZSTENCIL_CLEAR_VALUE,
    PRAMIN, PRAMIN_END, guest_address,
};

/// One recorded method: the object's class, the method, and its parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Command {
    pub class: u32,
    pub method: u32,
    pub value: u32,
}

/// The 3D engine.
pub const KELVIN: u32 = 0x97;

// Kelvin methods [F].
const SET_OBJECT: u32 = 0x000;
const NO_OPERATION: u32 = 0x100;
const FLIP_STALL: u32 = 0x130;
const CONTEXT_DMA_VERTEX_A: u32 = 0x19C;
const CONTEXT_DMA_VERTEX_B: u32 = 0x1A0;
const SET_CONTEXT_DMA_SEMAPHORE: u32 = 0x1A4;
const SET_CONTEXT_DMA_REPORT: u32 = 0x1A8;
const PROGRAM_WORDS: std::ops::Range<u32> = 0xB00..0xB80;
const CONSTANT_WORDS: std::ops::Range<u32> = 0xB80..0xC00;
/// The methods of the transform registers, which are constants too.
const TRANSFORM_WORDS: std::ops::Range<u32> = 0x440..0xB00;
const ARRAY_OFFSET: u32 = 0x1720;
const ARRAY_FORMAT: u32 = 0x1760;
const ARRAY_ELEMENT16: u32 = 0x1800;
const ARRAY_ELEMENT32: u32 = 0x1808;
const DRAW_ARRAYS: u32 = 0x1810;
const INLINE_ARRAY: u32 = 0x1818;
/// `CLEAR_REPORT_VALUE`, `SET_ZPASS_PIXEL_COUNT_ENABLE` and `GET_REPORT`: a visibility test
/// counts the pixels the draws cover while it is on, into a value the clear sets back to zero and
/// the report writes to memory.
const CLEAR_REPORT_VALUE: u32 = 0x17C8;
const SET_ZPASS_PIXEL_COUNT_ENABLE: u32 = 0x17CC;
const GET_REPORT: u32 = 0x17D0;
const SET_SEMAPHORE_OFFSET: u32 = 0x1D6C;
const BACK_END_WRITE_SEMAPHORE_RELEASE: u32 = 0x1D70;
const SET_ZSTENCIL_CLEAR_VALUE: u32 = 0x1D8C;
const SET_COLOR_CLEAR_VALUE: u32 = 0x1D90;
const PROGRAM_LOAD: u32 = 0x1E9C;
const PROGRAM_START: u32 = 0x1EA0;
const CONSTANT_LOAD: u32 = 0x1EA4;

/// The "action" methods of the 3D class (the spec's ranges are inclusive); every other method
/// below `0x2000` is state, of which the model keeps the latest value (spec 10.5).
const ACTIONS: [std::ops::Range<u32>; 7] = [
    0x100..0x134,
    0xB00..0xC00,
    0x1700..0x1760,
    0x17A0..0x1B00,
    0x1D6C..0x1D75,
    // `CLEAR_SURFACE` alone: the clear rectangle after it is state.
    0x1D94..0x1D95,
    0x1E98..0x1EA1,
];

/// Pusher errors (`DMA_STATE` bits 29..31).
const ERROR_CALL: u32 = 1;
const ERROR_RETURN: u32 = 3;
const ERROR_RESERVED: u32 = 4;
const ERROR_PROTECTION: u32 = 6;

/// A report is 16 bytes: a timestamp, the pixel count and a status word. Direct3D marks one
/// pending with this status, and reads it as incomplete until the GPU writes it as 0 (done).
const REPORT_PENDING: u32 = 0xFFFF_FFFF;
/// The timestamp written into every report.
const REPORT_TIMESTAMP: u64 = 0x0011_2233_4455_6677;
/// The kernel's contiguous memory below 16 MiB, where Direct3D's report pages belong (the
/// method's offset is 24 bits).
const LOW_CONTIGUOUS: std::ops::Range<u32> = 0x4000..0xC000;

/// The most push-buffer bytes read at a time.
const RUN: u32 = 1024;

/// Whether a 3D method is state.
fn is_state(method: u32) -> bool {
    method < 0x2000 && !ACTIONS.iter().any(|r| r.contains(&method))
}

impl Gpu {
    /// A DMA object's physical base and limit (spec 10.3): three words, flags, limit, frame. An
    /// address in it is `guest_address(base + offset)`: the page-0 mirror applies to the sum.
    pub(crate) fn dma_object(&self, instance: u32) -> Option<(u32, u32)> {
        let at = PRAMIN + instance;
        if at + 12 > PRAMIN_END {
            return None;
        }
        let (flags, limit, frame) = (self.reg(at), self.reg(at + 4), self.reg(at + 8));
        let base = ((frame & 0xFFFF_F000) | (flags >> 20)) & 0x07FF_FFFF;
        Some((base, limit))
    }

    /// Looks `handle` up in RAMHT for `channel` (spec 10.4): the context word of a valid entry.
    pub(crate) fn ramht(&self, handle: u32, channel: u32) -> Option<u32> {
        let ramht = self.reg(PFIFO_RAMHT);
        let size = 1u32 << (((ramht >> 16) & 3) + 12);
        let bits = size.trailing_zeros() - 1;
        let mut hash = 0;
        let mut rest = handle;
        while rest != 0 {
            hash ^= rest & ((1 << bits) - 1);
            rest >>= bits;
        }
        hash ^= (channel & 0x1F) << (bits - 4);
        let entry = PRAMIN + ((ramht & 0x1F0) << 8) + hash * 8;
        let context = self.reg(entry + 4);
        (self.reg(entry) == handle && context & 1 << 31 != 0).then_some(context)
    }

    fn channel(&self) -> u32 {
        self.reg(CACHE1_PUSH1) & 0x1F
    }

    /// Whether the pusher may run now.
    fn may_push(&self) -> bool {
        let push = self.reg(CACHE1_DMA_PUSH);
        self.reg(CACHE1_PUSH0) & 1 != 0
            && push & 1 != 0
            && push & 1 << 12 == 0
            && self.reg(PGRAPH_FIFO) & 1 != 0
            && !self.nop_wait
    }

    /// Stops the pusher with `error`.
    fn push_error(&mut self, error: u32) {
        let state = self.reg(CACHE1_DMA_STATE) & !(7 << 29);
        self.set(CACHE1_DMA_STATE, state | error << 29);
        self.set(CACHE1_DMA_PUSH, self.reg(CACHE1_DMA_PUSH) | 1 << 12);
    }

    /// Runs the pusher until `GET == PUT`, an error, or a method the puller cannot take yet.
    pub(crate) fn run_pusher(&mut self, bus: &dyn Bus) {
        if !self.may_push() || self.reg(CACHE1_DMA_GET) == self.reg(CACHE1_DMA_PUT) {
            return;
        }
        // The words read from memory are kept only for this run: the title writes more before
        // its next kick. The buffer's allocation is kept.
        let mut buffer = std::mem::take(&mut self.push_buffer);
        self.push_words(bus, &mut buffer);
        self.push_buffer = buffer;
    }

    /// [`Gpu::run_pusher`]'s loop, reading words into `buffer`.
    fn push_words(&mut self, bus: &dyn Bus, buffer: &mut Vec<u8>) {
        buffer.clear();
        let mut buffered_at = 0;
        // Nothing the pusher does changes its DMA object: it is looked up once a run.
        let Some((base, limit)) = self.dma_object((self.reg(CACHE1_DMA_INSTANCE) & 0xFFFF) << 4)
        else {
            self.push_error(ERROR_PROTECTION);
            return;
        };
        while self.may_push() {
            let (get, put) = (self.reg(CACHE1_DMA_GET), self.reg(CACHE1_DMA_PUT));
            if get == put {
                return;
            }
            if get > limit {
                self.push_error(ERROR_PROTECTION);
                return;
            }
            // Words come from guest memory in runs of up to 1 KiB.
            let have = get
                .checked_sub(buffered_at)
                .is_some_and(|o| o as usize + 4 <= buffer.len());
            if !have {
                let end = if put > get {
                    put
                } else {
                    limit.saturating_add(1)
                };
                let len = (end - get).clamp(4, RUN) & !3;
                buffer.clear();
                buffer.resize(len as usize, 0);
                if !bus.read(guest_address(base.wrapping_add(get)), buffer) {
                    buffer.truncate(4);
                    if !bus.read(guest_address(base.wrapping_add(get)), buffer) {
                        self.push_error(ERROR_PROTECTION);
                        return;
                    }
                }
                buffered_at = get;
            }
            let at = (get - buffered_at) as usize;
            let word =
                u32::from_le_bytes([buffer[at], buffer[at + 1], buffer[at + 2], buffer[at + 3]]);
            if !self.push_word(word, get, bus) {
                return;
            }
        }
    }

    /// Handles one push-buffer word at `get`. Returns false when the pusher must stop (an error,
    /// or a method the puller cannot take yet, which is retried later).
    fn push_word(&mut self, word: u32, get: u32, bus: &dyn Bus) -> bool {
        let state = self.reg(CACHE1_DMA_STATE);
        let count = (state >> 18) & 0x7FF;
        if count > 0 {
            let method = state & 0x1FFC;
            let subchannel = (state >> 13) & 7;
            if !self.pull(subchannel, method, word, bus) {
                return false;
            }
            let non_increasing = state & 1 != 0;
            let method = if non_increasing {
                method
            } else {
                (method + 4) & 0x1FFC
            };
            let state = (state & !(0x7FF << 18) & !0x1FFC) | (count - 1) << 18 | method;
            self.set(CACHE1_DMA_STATE, state);
            self.set(CACHE1_DMA_GET, get + 4);
            return true;
        }
        let subroutine = self.reg(CACHE1_DMA_SUBROUTINE);
        if word & 0xE000_0003 == 0x2000_0000 {
            // Old jump.
            self.set(CACHE1_DMA_GET, word & 0x1FFF_FFFF);
        } else if word & 3 == 1 {
            self.set(CACHE1_DMA_GET, word & !3);
        } else if word & 3 == 2 {
            if subroutine & 1 != 0 {
                self.push_error(ERROR_CALL);
                return false;
            }
            self.set(CACHE1_DMA_SUBROUTINE, (get + 4) | 1);
            self.set(CACHE1_DMA_GET, word & !3);
        } else if word == 0x0002_0000 {
            if subroutine & 1 == 0 {
                self.push_error(ERROR_RETURN);
                return false;
            }
            self.set(CACHE1_DMA_SUBROUTINE, subroutine & !1);
            self.set(CACHE1_DMA_GET, subroutine & !3);
        } else if word & 0xE003_0003 == 0 || word & 0xE003_0003 == 0x4000_0000 {
            let non_increasing = u32::from(word & 0xE003_0003 == 0x4000_0000);
            let method = word & 0x1FFC;
            let subchannel = (word >> 13) & 7;
            let count = (word >> 18) & 0x7FF;
            self.set(
                CACHE1_DMA_STATE,
                non_increasing | method | subchannel << 13 | count << 18,
            );
            self.set(CACHE1_DMA_GET, get + 4);
        } else {
            self.push_error(ERROR_RESERVED);
            return false;
        }
        true
    }

    /// Hands a method to the object on `subchannel`. Returns false when it cannot be taken yet.
    fn pull(&mut self, subchannel: u32, method: u32, value: u32, bus: &dyn Bus) -> bool {
        if self.nop_wait
            || self.context_wait
            || self.reg(PGRAPH_FIFO) & 1 == 0
            || self.reg(CACHE1_PULL0) & 1 == 0
        {
            return false;
        }
        let channel = self.channel();
        if method == SET_OBJECT {
            return self.set_object(subchannel, value, channel);
        }
        let mut value = value;
        if (0x180..0x200).contains(&method) {
            // A handle: the object's instance address takes its place.
            let Some(context) = self.ramht(value, channel) else {
                self.stall = Some(format!(
                    "handle {value:#x} (method {method:#x}) is not in RAMHT"
                ));
                return false;
            };
            value = (context & 0xFFFF) << 4;
        }
        self.stall = None;
        let class = self.subchannels[subchannel as usize].class;
        if class == KELVIN {
            self.kelvin(subchannel, method, value, bus);
        } else {
            self.record(class, method, value);
        }
        true
    }

    /// `SET_OBJECT`: binds a graphics object to a subchannel once the channel's context is loaded.
    fn set_object(&mut self, subchannel: u32, handle: u32, channel: u32) -> bool {
        let Some(context) = self.ramht(handle, channel).filter(|c| (c >> 16) & 3 == 1) else {
            self.stall = Some(format!(
                "object {handle:#x} is not a graphics object in RAMHT"
            ));
            return false;
        };
        let loaded = self.reg(PGRAPH_CTX_CONTROL) & 1 << 16 != 0
            && (self.reg(PGRAPH_CTX_USER) >> 24) & 0x1F == channel;
        if !loaded {
            let trapped = (self.reg(PGRAPH_TRAPPED_ADDR) & !(0x1F << 20)) | channel << 20;
            self.set(PGRAPH_TRAPPED_ADDR, trapped);
            self.set(PGRAPH_INTR, self.reg(PGRAPH_INTR) | INTR_CONTEXT_SWITCH);
            self.context_wait = true;
            return false;
        }
        let instance = (context & 0xFFFF) << 4;
        let mut words = [0; 5];
        for (i, word) in words.iter_mut().take(4).enumerate() {
            *word = self.reg(PRAMIN + instance + 4 * u32::try_from(i).unwrap_or(0));
        }
        words[4] = instance;
        for (i, &word) in words.iter().enumerate() {
            let i = u32::try_from(i).unwrap_or(0);
            self.set(PGRAPH_CTX_CACHE1 + i * 0x20 + subchannel * 4, word);
            self.set(PGRAPH_CTX_SWITCH1 + i * 4, word);
        }
        self.subchannels[subchannel as usize] = super::Subchannel {
            class: words[0] & 0xFF,
        };
        self.stall = None;
        self.record(words[0] & 0xFF, SET_OBJECT, handle);
        true
    }

    /// A method of the 3D engine.
    fn kelvin(&mut self, subchannel: u32, method: u32, value: u32, bus: &dyn Bus) {
        if method == INLINE_ARRAY {
            // Vertex words, most of a frame's commands, touch no register the model keeps and
            // are no state.
            self.vertex_word(value);
            return;
        }
        if matches!(method, CONTEXT_DMA_VERTEX_A | CONTEXT_DMA_VERTEX_B)
            || (ARRAY_OFFSET..ARRAY_FORMAT + 64).contains(&method)
        {
            self.arrays = None;
        }
        match method {
            NO_OPERATION if value != 0 => {
                // The XDK's software interrupt request: the FIFO waits until the guest clears it.
                let trapped =
                    (self.reg(PGRAPH_TRAPPED_ADDR) & 0x1F << 20) | subchannel << 16 | method;
                self.set(PGRAPH_TRAPPED_ADDR, trapped);
                self.set(PGRAPH_TRAPPED_DATA_LOW, value);
                self.set(PGRAPH_NSOURCE, 1);
                self.set(PGRAPH_INTR, self.reg(PGRAPH_INTR) | INTR_ERROR);
                self.nop_wait = true;
            }
            BACK_END_WRITE_SEMAPHORE_RELEASE => {
                let instance = self.state.get(SET_CONTEXT_DMA_SEMAPHORE).unwrap_or(0);
                let offset = self.state.get(SET_SEMAPHORE_OFFSET).unwrap_or(0);
                if let Some((base, limit)) = self.dma_object(instance)
                    && offset <= limit
                {
                    bus.write(
                        guest_address(base.wrapping_add(offset)),
                        &value.to_le_bytes(),
                    );
                }
            }
            // The engine keeps the clear values in PGRAPH registers, and Direct3D reads them back
            // as a mailbox: it queues these two methods with a callback address and its context,
            // raises the software-method interrupt, and its interrupt handler calls the address
            // it reads from the depth-stencil clear value.
            SET_ZSTENCIL_CLEAR_VALUE => self.set(PGRAPH_ZSTENCIL_CLEAR_VALUE, value),
            SET_COLOR_CLEAR_VALUE => self.set(PGRAPH_COLOR_CLEAR_VALUE, value),
            // The method's address names the word, and only the fourth moves to the next slot:
            // a non-incrementing push writes one word over and over.
            m if PROGRAM_WORDS.contains(&m) => {
                let word = (m - PROGRAM_WORDS.start) / 4 % 4;
                let slot = self.program.program_slot;
                self.program.instructions.set(slot, word as usize, value);
                if word == 3 {
                    self.program.program_slot = slot + 1;
                }
            }
            m if CONSTANT_WORDS.contains(&m) => {
                let word = (m - CONSTANT_WORDS.start) / 4 % 4;
                let slot = self.program.constant_slot;
                self.program.constants.set(slot, word as usize, value);
                if word == 3 {
                    self.program.constant_slot = slot + 1;
                }
            }
            PROGRAM_LOAD => self.program.program_slot = value,
            CONSTANT_LOAD => self.program.constant_slot = value,
            SET_ZPASS_PIXEL_COUNT_ENABLE => {
                self.count_probe();
                if value != 0 {
                    self.start_probe();
                }
            }
            CLEAR_REPORT_VALUE => {
                // Counts not reported yet go with the value; a test still on counts on.
                if self.probe.take().is_some() {
                    self.start_probe();
                }
                self.zpass_pixels = 0;
            }
            GET_REPORT => {
                if self.probe.is_some() {
                    self.count_probe();
                    self.start_probe();
                }
                self.write_report(value, bus);
            }
            // The transform registers share storage with the constants.
            m if TRANSFORM_WORDS.contains(&m) => {
                if let Some((row, component)) = super::program::transform_constant(m) {
                    let row = u32::try_from(row).unwrap_or(u32::MAX);
                    self.program.constants.set(row, component, value);
                }
            }
            _ => {}
        }
        let wanted = self.recording || self.probe.is_some();
        if wanted && matches!(method, ARRAY_ELEMENT16 | ARRAY_ELEMENT32 | DRAW_ARRAYS) {
            self.expand_arrays(method, value, bus);
        } else {
            self.inline.take(method, value);
            self.record(KELVIN, method, value);
            self.probe_command(method, value);
        }
        // The latest value of every method; the frame prefix takes only the state ones.
        self.state.insert(method, value);
        if method == FLIP_STALL {
            self.flips += 1;
            if self.recording {
                // The next frame starts with the state this one left, in a buffer as large as
                // this one, so the frame is not copied nor grown as it is recorded.
                let prefix = self.frame_prefix();
                let mut next = Vec::with_capacity((self.frame.len() + prefix.len()) * 9 / 8);
                next.extend(prefix);
                let commands = std::mem::replace(&mut self.frame, next);
                self.frames.push(Frame { commands });
            }
        }
    }

    /// Takes one vertex word: the inline shadow, the frame, the visibility test.
    #[inline]
    fn vertex_word(&mut self, word: u32) {
        self.inline.take_vertex_word(word);
        self.record(KELVIN, INLINE_ARRAY, word);
        self.probe_command(INLINE_ARRAY, word);
    }

    /// Starts a visibility test: its commands start with the state its draws depend on.
    fn start_probe(&mut self) {
        self.probe = Some(self.frame_prefix());
    }

    /// Adds a 3D command to the visibility test running, if any; a clear is not one of its
    /// draws.
    fn probe_command(&mut self, method: u32, value: u32) {
        if let Some(probe) = &mut self.probe
            && method != super::pixel::CLEAR_SURFACE
        {
            probe.push(Command {
                class: KELVIN,
                method,
                value,
            });
        }
    }

    /// Ends the visibility test running, if any, and adds the pixels its draws cover. They are
    /// counted here, on the title's thread, since the title may wait for the report before it
    /// draws on; the depth buffer is not at hand, so hidden pixels count too.
    fn count_probe(&mut self) {
        if let Some(probe) = self.probe.take() {
            let pixels = super::pixel::visible_pixels(&probe);
            self.zpass_pixels = self.zpass_pixels.saturating_add(pixels);
        }
    }

    /// Writes the report `GET_REPORT` asks for, done, with the pixels counted.
    ///
    /// Direct3D names a report by its physical address with the report type in bit 24, and
    /// wants its report pages below 16 MiB, where the address fits the method's 24-bit offset.
    /// A report in the kernel's low contiguous memory is at that offset in the report DMA
    /// object, once Direct3D marked it pending. Should its pages lie above 16 MiB instead, bit
    /// 24 may be the type or the address: the report is the reading that Direct3D marked
    /// pending, its timestamp zero (a fresh page) or the one written here before.
    fn write_report(&self, value: u32, bus: &dyn Bus) {
        let instance = self.state.get(SET_CONTEXT_DMA_REPORT).unwrap_or(0);
        let Some((base, limit)) = self.dma_object(instance) else {
            return;
        };
        let mut report = [0u8; 16];
        report[..8].copy_from_slice(&REPORT_TIMESTAMP.to_le_bytes());
        report[8..12].copy_from_slice(&self.zpass_pixels.to_le_bytes());
        let pending_at = |offset: u32, fresh: bool| {
            if offset.saturating_add(15) > limit {
                return None;
            }
            let at = guest_address(base.wrapping_add(offset));
            let mut found = [0u8; 16];
            if !bus.read(at, &mut found) {
                return None;
            }
            let [timestamp, status] = [&found[..8], &found[12..]];
            let timestamp = u64::from_le_bytes(timestamp.try_into().unwrap_or_default());
            let pending =
                u32::from_le_bytes(status.try_into().unwrap_or_default()) == REPORT_PENDING;
            (pending && (!fresh || timestamp == 0 || timestamp == REPORT_TIMESTAMP)).then_some(at)
        };
        let offset = value & 0x00FF_FFFF;
        let low = LOW_CONTIGUOUS
            .contains(&base.wrapping_add(offset))
            .then(|| pending_at(offset, false))
            .flatten();
        let at = low.or_else(|| {
            [value & 0x03FF_FFFF, value & 0x02FF_FFFF, offset]
                .into_iter()
                .find_map(|offset| pending_at(offset, true))
        });
        if let Some(at) = at {
            bus.write(at, &report);
        }
    }

    /// Appends a command to the frame being recorded.
    #[inline]
    fn record(&mut self, class: u32, method: u32, value: u32) {
        if self.recording {
            self.frame.push(Command {
                class,
                method,
                value,
            });
        }
    }

    /// The commands a frame starts with: the latest state in method order, the attributes'
    /// inline values, then the replay of the vertex program and constants, which leaves each
    /// load slot where the title's streaming left it.
    pub(crate) fn frame_prefix(&self) -> Vec<Command> {
        let command = |method, value| Command {
            class: KELVIN,
            method,
            value,
        };
        let mut prefix: Vec<Command> = self
            .state
            .iter()
            .filter(|&(method, _)| is_state(method))
            .map(|(method, value)| command(method, value))
            .collect();
        prefix.extend(self.inline.replay());
        let latest = |method| self.state.get(method).unwrap_or(0);
        prefix.push(command(PROGRAM_START, latest(PROGRAM_START)));
        let replay = |words: &super::Slots,
                      load,
                      slot,
                      window: std::ops::Range<u32>,
                      prefix: &mut Vec<Command>| {
            let Some(last) = words.last() else {
                return;
            };
            prefix.push(command(load, 0));
            for slot in 0..=last {
                let insn = words.get(slot).copied().unwrap_or_default();
                for (i, word) in insn.iter().enumerate() {
                    let n = slot * 4 + u32::try_from(i).unwrap_or(0);
                    prefix.push(command(window.start + 4 * (n % 32), *word));
                }
            }
            prefix.push(command(load, slot));
        };
        replay(
            &self.program.instructions,
            PROGRAM_LOAD,
            self.program.program_slot,
            PROGRAM_WORDS,
            &mut prefix,
        );
        replay(
            &self.program.constants,
            CONSTANT_LOAD,
            self.program.constant_slot,
            CONSTANT_WORDS,
            &mut prefix,
        );
        prefix
    }

    /// The enabled vertex arrays: where each starts, its stride and its words a vertex. Each DMA
    /// object is looked up once.
    fn arrays(&self) -> Arrays {
        let latest = |method| self.state.get(method).unwrap_or(0);
        let objects = [CONTEXT_DMA_VERTEX_A, CONTEXT_DMA_VERTEX_B]
            .map(|dma| self.dma_object(latest(dma)).map_or(0, |(base, _)| base));
        let mut arrays = Arrays::default();
        for i in 0..16 {
            let format = latest(ARRAY_FORMAT + 4 * i);
            let words = super::draw::attribute_words(format);
            if words == 0 {
                continue;
            }
            let offset = latest(ARRAY_OFFSET + 4 * i);
            let base = objects[usize::from(offset & 1 << 31 != 0)];
            arrays.enabled[arrays.len] = Array {
                start: base.wrapping_add(offset & 0x7FFF_FFFF),
                stride: format >> 8,
                words: words as u32,
            };
            arrays.len += 1;
        }
        arrays
    }

    /// Expands a vertex-array draw into `INLINE_ARRAY` words while the arrays are in memory: for
    /// each index, for each enabled attribute, its words at `start + index * stride`, 0 where
    /// memory is not mapped. A run of indices (`DRAW_ARRAYS`) reads each array's span at once.
    fn expand_arrays(&mut self, method: u32, value: u32, bus: &dyn Bus) {
        let arrays = match self.arrays {
            Some(arrays) => arrays,
            None => *self.arrays.insert(self.arrays()),
        };
        let (first, count) = match method {
            ARRAY_ELEMENT16 => (value & 0xFFFF, 2),
            ARRAY_ELEMENT32 => (value, 1),
            _ => (value & 0x00FF_FFFF, (value >> 24) + 1),
        };
        let index = |n: u32| {
            if method == ARRAY_ELEMENT16 && n == 1 {
                value >> 16
            } else {
                first.wrapping_add(n)
            }
        };
        let arrays = &arrays.enabled[..arrays.len];
        // The words go into a buffer kept from draw to draw, then to the shadow and the frame.
        let mut words = std::mem::take(&mut self.array_words);
        words.clear();
        let per_vertex: u32 = arrays.iter().map(|a| a.words).sum();
        words.reserve((per_vertex * count) as usize);
        if method == DRAW_ARRAYS && count > 1 {
            // Each array's span, read in one go into a buffer kept from draw to draw.
            let mut spans = std::mem::take(&mut self.array_spans);
            spans.clear();
            let mut starts = [usize::MAX; 16];
            for (slot, array) in starts.iter_mut().zip(arrays) {
                let len = (count - 1) as usize * array.stride as usize + 4 * array.words as usize;
                let at = array.start.wrapping_add(first.wrapping_mul(array.stride));
                let Some(address) = span_address(at, len).filter(|_| len <= SPAN_LIMIT) else {
                    continue;
                };
                let offset = spans.len();
                spans.resize(offset + len, 0);
                if bus.read(address, &mut spans[offset..]) {
                    *slot = offset;
                } else {
                    spans.truncate(offset);
                }
            }
            for n in 0..count {
                for (array, &offset) in arrays.iter().zip(&starts) {
                    if offset == usize::MAX {
                        vertex_words(bus, array, index(n), &mut words);
                        continue;
                    }
                    let at = offset + (n * array.stride) as usize;
                    words.extend(
                        spans[at..at + 4 * array.words as usize]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|w| u32::from_le_bytes(*w)),
                    );
                }
            }
            self.array_spans = spans;
        } else {
            for n in 0..count {
                for array in arrays {
                    vertex_words(bus, array, index(n), &mut words);
                }
            }
        }
        self.inline.take_vertex_words(&words);
        let command = |&value: &u32| Command {
            class: KELVIN,
            method: INLINE_ARRAY,
            value,
        };
        if self.recording {
            self.frame.extend(words.iter().map(command));
        }
        if let Some(probe) = &mut self.probe {
            probe.extend(words.iter().map(command));
        }
        self.array_words = words;
    }
}

/// Appends the words of vertex `index` of `array` to `words`: read in one go, or a word at a time
/// where that fails, 0 for each word not mapped.
fn vertex_words(bus: &dyn Bus, array: &Array, index: u32, words: &mut Vec<u32>) {
    let at = array.start.wrapping_add(index.wrapping_mul(array.stride));
    let mut bytes = [0u8; 16];
    let bytes = &mut bytes[..4 * array.words as usize];
    if span_address(at, bytes.len()).is_some_and(|address| bus.read(address, bytes)) {
        words.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w)),
        );
        return;
    }
    for w in 0..array.words {
        let mut word = [0u8; 4];
        let read = bus.read(guest_address(at.wrapping_add(4 * w)), &mut word);
        words.push(if read { u32::from_le_bytes(word) } else { 0 });
    }
}

/// The guest address of the `len` bytes the GPU reaches at `physical`, when they are one run there
/// as they are word by word.
fn span_address(physical: u32, len: usize) -> Option<u32> {
    let last = physical.checked_add(u32::try_from(len.checked_sub(1)?).ok()?)?;
    let address = guest_address(physical);
    (guest_address(last).wrapping_sub(address) == last - physical).then_some(address)
}

/// The longest span of one vertex array a draw reads in one go.
const SPAN_LIMIT: usize = 1 << 20;

/// An enabled vertex array.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct Array {
    start: u32,
    stride: u32,
    words: u32,
}

/// The enabled vertex arrays, in attribute order.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub(crate) struct Arrays {
    enabled: [Array; 16],
    len: usize,
}

#[cfg(test)]
mod tests {
    use super::super::tests::FlatBus;
    use super::*;

    #[test]
    fn the_clear_rectangle_is_state_and_the_clear_an_action() {
        assert!(!is_state(0x1D94));
        assert!(is_state(0x1D98) && is_state(0x1D9C));
    }

    /// A GPU with a push buffer at 0x100000 (DMA object at instance 0x100) and a Kelvin object
    /// bound in RAMHT, its context loaded, the FIFO running.
    fn setup(bus: &FlatBus) -> Gpu {
        let mut gpu = Gpu::new();
        // RAMHT at PRAMIN + 0, 4 KiB.
        gpu.write(PFIFO_RAMHT, 4, 0, bus);
        // The push buffer DMA object: base 0x100000, limit 0xFFFF.
        gpu.write(PRAMIN + 0x100, 4, 0, bus);
        gpu.write(PRAMIN + 0x104, 4, 0xFFFF, bus);
        gpu.write(PRAMIN + 0x108, 4, 0x10_0000, bus);
        // A Kelvin object at instance 0x200 with handle 0xBEEF.
        gpu.write(PRAMIN + 0x200, 4, KELVIN, bus);
        let hash = {
            let mut h = 0;
            let mut v = 0xBEEFu32;
            while v != 0 {
                h ^= v & 0x7FF;
                v >>= 11;
            }
            h
        };
        gpu.write(PRAMIN + hash * 8, 4, 0xBEEF, bus);
        gpu.write(PRAMIN + hash * 8 + 4, 4, 1 << 31 | 1 << 16 | 0x20, bus);
        // A semaphore DMA object at instance 0x300: base 0x200000, handle 0xCAFE.
        gpu.write(PRAMIN + 0x300, 4, 0, bus);
        gpu.write(PRAMIN + 0x304, 4, 0xFFF, bus);
        gpu.write(PRAMIN + 0x308, 4, 0x20_0000, bus);
        let hash = (0xCAFEu32 & 0x7FF) ^ (0xCAFE >> 11);
        gpu.write(PRAMIN + hash * 8, 4, 0xCAFE, bus);
        gpu.write(PRAMIN + hash * 8 + 4, 4, 1 << 31 | 0x30, bus);
        gpu.write(PGRAPH_CTX_CONTROL, 4, 1 << 16, bus);
        gpu.write(PGRAPH_FIFO, 4, 1, bus);
        gpu.write(CACHE1_PULL0, 4, 1, bus);
        gpu.write(CACHE1_DMA_INSTANCE, 4, 0x10, bus);
        gpu.write(CACHE1_DMA_PUSH, 4, 1, bus);
        gpu.write(CACHE1_PUSH0, 4, 1, bus);
        gpu
    }

    /// Writes push-buffer words at 0x100000 + `at` and moves PUT past them.
    fn push(gpu: &mut Gpu, bus: &FlatBus, at: u32, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            bus.put32(0x10_0000 + at + 4 * u32::try_from(i).unwrap(), *w);
        }
        gpu.write(
            CACHE1_DMA_PUT,
            4,
            at + 4 * u32::try_from(words.len()).unwrap(),
            bus,
        );
    }

    /// An increasing-method header.
    fn methods(subchannel: u32, method: u32, count: u32) -> u32 {
        count << 18 | subchannel << 13 | method
    }

    /// Pushes `words` after binding the 3D object and the report DMA object (0xCAFE: base
    /// 0x200000, limit 0xFFF).
    fn with_reports(gpu: &mut Gpu, bus: &FlatBus, words: &[u32]) {
        let mut all = vec![
            methods(0, SET_OBJECT, 1),
            0xBEEF,
            methods(0, SET_CONTEXT_DMA_REPORT, 1),
            0xCAFE,
            // Attribute 0: three floats.
            methods(0, ARRAY_FORMAT, 1),
            0x32,
        ];
        all.extend_from_slice(words);
        push(gpu, bus, 0, &all);
    }

    /// The words drawing the square from `(x, y)` with sides `size`, in pixels.
    fn square(x: f32, y: f32, size: f32) -> Vec<u32> {
        let mut words = vec![
            methods(0, 0x17FC, 1),
            8,
            0x4000_0000 | methods(0, INLINE_ARRAY, 12),
        ];
        for (cx, cy) in [(x, y), (x + size, y), (x + size, y + size), (x, y + size)] {
            words.extend([cx.to_bits(), cy.to_bits(), 0]);
        }
        words.extend([methods(0, 0x17FC, 1), 0]);
        words
    }

    /// Marks the report at 0x200100 pending, asks for it with the type in bit 24, and returns
    /// its pixel count, checking that it is done.
    fn report(gpu: &mut Gpu, bus: &FlatBus, at: u32) -> u32 {
        for i in 0..3 {
            bus.put32(0x20_0100 + 4 * i, 0);
        }
        bus.put32(0x20_010C, REPORT_PENDING);
        let words = [methods(0, GET_REPORT, 1), 0x0100_0100];
        for (i, w) in words.iter().enumerate() {
            bus.put32(0x10_0000 + at + 4 * u32::try_from(i).unwrap(), *w);
        }
        gpu.write(CACHE1_DMA_PUT, 4, at + 8, bus);
        assert_eq!(bus.get32(0x20_010C), 0, "done");
        assert_eq!(
            [bus.get32(0x20_0100), bus.get32(0x20_0104)],
            [0x4455_6677, 0x0011_2233]
        );
        bus.get32(0x20_0108)
    }

    #[test]
    fn a_visibility_test_reports_the_pixels_its_draws_cover() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        let mut words = square(0.0, 0.0, 50.0);
        words.extend([methods(0, SET_ZPASS_PIXEL_COUNT_ENABLE, 1), 1]);
        words.extend(square(10.0, 10.0, 10.0));
        words.extend([methods(0, SET_ZPASS_PIXEL_COUNT_ENABLE, 1), 0]);
        with_reports(&mut gpu, &bus, &words);
        let mut at = u32::try_from(words.len() + 6).unwrap() * 4;
        assert_eq!(
            report(&mut gpu, &bus, at),
            100,
            "only the draws while counting"
        );
        at += 8;
        // Reported while counting; the count adds up until it is cleared.
        let mut more = vec![methods(0, SET_ZPASS_PIXEL_COUNT_ENABLE, 1), 1];
        more.extend(square(0.0, 0.0, 4.0));
        for (i, w) in more.iter().enumerate() {
            bus.put32(0x10_0000 + at + 4 * u32::try_from(i).unwrap(), *w);
        }
        at += u32::try_from(more.len()).unwrap() * 4;
        gpu.write(CACHE1_DMA_PUT, 4, at, &bus);
        assert_eq!(report(&mut gpu, &bus, at), 116);
        at += 8;
        for (i, w) in [methods(0, CLEAR_REPORT_VALUE, 1), 0].iter().enumerate() {
            bus.put32(0x10_0000 + at + 4 * u32::try_from(i).unwrap(), *w);
        }
        at += 8;
        gpu.write(CACHE1_DMA_PUT, 4, at, &bus);
        assert_eq!(report(&mut gpu, &bus, at), 0);
    }

    #[test]
    fn a_report_in_low_memory_is_at_the_methods_offset() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        // A report DMA object over all of memory at instance 0x500, handle 0xD00D.
        gpu.write(PRAMIN + 0x500, 4, 0, &bus);
        gpu.write(PRAMIN + 0x504, 4, 0x03FF_FFFF, &bus);
        gpu.write(PRAMIN + 0x508, 4, 0, &bus);
        let hash = (0xD00Du32 & 0x7FF) ^ (0xD00D >> 11);
        gpu.write(PRAMIN + hash * 8, 4, 0xD00D, &bus);
        gpu.write(PRAMIN + hash * 8 + 4, 4, 1 << 31 | 0x50, &bus);
        // Written before, so its timestamp is not a fresh one; the heap reading of bit 24 is
        // also marked pending.
        for i in 0..3 {
            bus.put32(0x5010 + 4 * i, 0xABAB_ABAB);
        }
        bus.put32(0x501C, REPORT_PENDING);
        bus.put32(0x0100_501C, REPORT_PENDING);
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, SET_CONTEXT_DMA_REPORT, 1),
                0xD00D,
                methods(0, GET_REPORT, 1),
                0x0100_5010,
            ],
        );
        assert_eq!(bus.get32(0x501C), 0, "done");
        assert_eq!(bus.get32(0x5018), 0, "nothing counted");
        assert_eq!(bus.get32(0x5010), 0x4455_6677);
        assert_eq!(
            bus.get32(0x0100_501C),
            REPORT_PENDING,
            "the heap reading is left"
        );
    }

    #[test]
    fn ramht_hashes_the_handle_and_the_channel() {
        let bus = FlatBus::new();
        let gpu = setup(&bus);
        assert_eq!(gpu.ramht(0xBEEF, 0).map(|c| (c & 0xFFFF) << 4), Some(0x200));
        assert_eq!(gpu.ramht(0xBEEF, 1), None);
        assert_eq!(gpu.ramht(0x1234, 0), None);
    }

    #[test]
    fn the_semaphore_release_is_the_fence_direct3d_waits_on() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, SET_CONTEXT_DMA_SEMAPHORE, 1),
                0xCAFE,
                methods(0, SET_SEMAPHORE_OFFSET, 2),
                0x10,
                0x1234_5678,
            ],
        );
        assert_eq!(gpu.reg(CACHE1_DMA_GET), 28);
        assert_eq!(bus.get32(0x20_0010), 0x1234_5678);
        assert_eq!(gpu.reg(PGRAPH_CTX_CACHE1), KELVIN, "bound on subchannel 0");
    }

    #[test]
    fn the_clear_values_are_readable_in_their_pgraph_registers() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, SET_ZSTENCIL_CLEAR_VALUE, 2),
                0x0013_B000,
                0x8123_4567,
            ],
        );
        assert_eq!(gpu.read(PGRAPH_ZSTENCIL_CLEAR_VALUE, 4), 0x0013_B000);
        assert_eq!(gpu.read(PGRAPH_COLOR_CLEAR_VALUE, 4), 0x8123_4567);
    }

    #[test]
    fn jumps_calls_and_returns() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        // 0x00: call 0x40; 0x04: jump 0x10. 0x40: SET_OBJECT, then return to 0x04.
        bus.put32(0x10_0000, 0x40 | 2);
        bus.put32(0x10_0004, 0x10 | 1);
        bus.put32(0x10_0040, methods(0, SET_OBJECT, 1));
        bus.put32(0x10_0044, 0xBEEF);
        bus.put32(0x10_0048, 0x0002_0000);
        gpu.write(CACHE1_DMA_PUT, 4, 0x10, &bus);
        assert_eq!(gpu.reg(CACHE1_DMA_GET), 0x10);
        assert_eq!(gpu.reg(CACHE1_DMA_SUBROUTINE) & 1, 0);
        assert_eq!(gpu.subchannels[0].class, KELVIN);
    }

    #[test]
    fn a_return_outside_a_subroutine_stops_the_pusher() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        push(&mut gpu, &bus, 0, &[0x0002_0000]);
        assert_ne!(gpu.reg(CACHE1_DMA_PUSH) & 1 << 12, 0);
        assert_eq!(gpu.reg(CACHE1_DMA_STATE) >> 29, ERROR_RETURN);
    }

    #[test]
    fn a_nop_with_a_parameter_waits_until_the_guest_clears_the_error() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, NO_OPERATION, 1),
                7,
                methods(0, NO_OPERATION, 1),
                0,
            ],
        );
        assert!(gpu.nop_wait);
        assert_eq!(
            gpu.reg(CACHE1_DMA_GET),
            16,
            "the FIFO waits after the request"
        );
        assert_eq!(gpu.reg(PGRAPH_TRAPPED_DATA_LOW), 7);
        assert_eq!(gpu.reg(PGRAPH_TRAPPED_ADDR) & 0xFFFF, NO_OPERATION);
        assert_ne!(gpu.reg(PGRAPH_INTR) & INTR_ERROR, 0);
        gpu.write(PGRAPH_INTR, 4, INTR_ERROR, &bus);
        assert!(!gpu.nop_wait);
        assert_eq!(gpu.reg(CACHE1_DMA_GET), 24);
    }

    #[test]
    fn binding_an_object_without_its_context_asks_for_a_switch() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        gpu.write(PGRAPH_CTX_CONTROL, 4, 0, &bus);
        push(&mut gpu, &bus, 0, &[methods(0, SET_OBJECT, 1), 0xBEEF]);
        assert!(gpu.context_wait);
        assert_ne!(gpu.reg(PGRAPH_INTR) & INTR_CONTEXT_SWITCH, 0);
        gpu.write(PGRAPH_CTX_CONTROL, 4, 1 << 16, &bus);
        gpu.write(PGRAPH_INTR, 4, INTR_CONTEXT_SWITCH, &bus);
        assert_eq!(gpu.reg(CACHE1_DMA_GET), 8);
        assert_eq!(gpu.subchannels[0].class, KELVIN);
    }

    #[test]
    fn frames_start_with_the_state_and_end_at_the_flip() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        gpu.record_frames(true);
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, 0x30C, 1),
                1, // depth test on (state)
                methods(0, 0x17FC, 1),
                5, // begin triangles (action)
                methods(0, FLIP_STALL, 1),
                0,
            ],
        );
        let frames = gpu.take_frames();
        assert_eq!(frames.len(), 1);
        let methods: Vec<u32> = frames[0].commands.iter().map(|c| c.method).collect();
        assert_eq!(
            methods,
            [PROGRAM_START, SET_OBJECT, 0x30C, 0x17FC, FLIP_STALL]
        );
        assert_eq!(gpu.flips, 1);
        // The next frame starts with the latest state.
        assert!(gpu.frame.iter().any(|c| c.method == 0x30C && c.value == 1));
    }

    #[test]
    fn a_vertex_array_draw_is_expanded_into_inline_words() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        gpu.record_frames(true);
        // Attribute 0: three floats, stride 12, at 0x300000 in DMA object 0x300's space: use a
        // vertex DMA object with base 0.
        gpu.write(PRAMIN + 0x400, 4, 0, &bus);
        gpu.write(PRAMIN + 0x404, 4, 0xFFFF_FFFF, &bus);
        gpu.write(PRAMIN + 0x408, 4, 0, &bus);
        let hash = (0xD00Du32 & 0x7FF) ^ (0xD00D >> 11);
        gpu.write(PRAMIN + hash * 8, 4, 0xD00D, &bus);
        gpu.write(PRAMIN + hash * 8 + 4, 4, 1 << 31 | 0x40, &bus);
        for (i, v) in [1u32, 2, 3, 4, 5, 6].iter().enumerate() {
            bus.put32(0x30_0000 + 4 * u32::try_from(i).unwrap(), *v);
        }
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, CONTEXT_DMA_VERTEX_A, 1),
                0xD00D,
                methods(0, ARRAY_OFFSET, 1),
                0x30_0000,
                methods(0, ARRAY_FORMAT, 1),
                12 << 8 | 3 << 4 | 2,
                methods(0, DRAW_ARRAYS, 1),
                1 << 24,
            ],
        );
        let inline: Vec<u32> = gpu
            .frame
            .iter()
            .filter(|c| c.method == INLINE_ARRAY)
            .map(|c| c.value)
            .collect();
        assert_eq!(inline, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn a_non_incrementing_push_writes_one_word_over_and_over() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        let mut words = vec![
            methods(0, SET_OBJECT, 1),
            0xBEEF,
            methods(0, CONSTANT_LOAD, 1),
            3,
            0x4000_0000 | methods(0, 0xB80, 8),
        ];
        words.extend(1..=8);
        push(&mut gpu, &bus, 0, &words);
        assert_eq!(gpu.program.constants.get(3), Some(&[8, 0, 0, 0]));
        assert_eq!(gpu.program.constants.len(), 1);
        assert_eq!(gpu.program.constant_slot, 3, "the slot stays put");
    }

    #[test]
    fn the_transform_registers_are_replayed_as_constants() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, 0xAF0, 1), // viewport scale x: row 58
                0x4000_0000,
            ],
        );
        assert_eq!(gpu.program.constants.get(58), Some(&[0x4000_0000, 0, 0, 0]));
        let prefix = gpu.frame_prefix();
        let at = 58 * 4;
        let constants: Vec<u32> = prefix
            .iter()
            .filter(|c| CONSTANT_WORDS.contains(&c.method))
            .map(|c| c.value)
            .collect();
        assert_eq!(constants.len(), 59 * 4);
        assert_eq!(constants[at], 0x4000_0000);
    }

    #[test]
    fn inline_values_are_replayed_at_each_frame_start() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        // SET_VERTEX_DATA4F_M of attribute 3: (1, 0.5, 0.25, 1).
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, 0x1A30, 4),
                1.0f32.to_bits(),
                0.5f32.to_bits(),
                0.25f32.to_bits(),
                1.0f32.to_bits(),
            ],
        );
        let prefix = gpu.frame_prefix();
        let replayed: Vec<(u32, u32)> = prefix
            .iter()
            .filter(|c| (0x1A00..0x1B00).contains(&c.method))
            .map(|c| (c.method, c.value))
            .collect();
        assert_eq!(
            replayed,
            [
                (0x1A30, 1.0f32.to_bits()),
                (0x1A34, 0.5f32.to_bits()),
                (0x1A38, 0.25f32.to_bits()),
                (0x1A3C, 1.0f32.to_bits()),
            ]
        );
    }

    #[test]
    fn the_vertex_program_is_replayed_at_each_frame_start() {
        let bus = FlatBus::new();
        let mut gpu = setup(&bus);
        push(
            &mut gpu,
            &bus,
            0,
            &[
                methods(0, SET_OBJECT, 1),
                0xBEEF,
                methods(0, PROGRAM_LOAD, 1),
                1,
                methods(0, 0xB00, 4),
                10,
                11,
                12,
                13,
            ],
        );
        // The load slot is replayed where streaming left it, one past the instruction.
        let prefix = gpu.frame_prefix();
        let words: Vec<(u32, u32)> = prefix
            .iter()
            .filter(|c| c.method != 0x0)
            .map(|c| (c.method, c.value))
            .collect();
        assert!(
            words.ends_with(&[
                (PROGRAM_LOAD, 0),
                (0xB00, 0),
                (0xB04, 0),
                (0xB08, 0),
                (0xB0C, 0),
                (0xB10, 10),
                (0xB14, 11),
                (0xB18, 12),
                (0xB1C, 13),
                (PROGRAM_LOAD, 2),
            ]),
            "{words:x?}"
        );
    }
}
