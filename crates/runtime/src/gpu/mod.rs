//! The GPU model (spec 10): a register-level model of the NV2A that answers the registers
//! Direct3D reads, runs the push-buffer FIFO, performs the interrupt handshakes, writes
//! semaphores, shows the video overlay, and records every 3D method for the renderer. It does not
//! rasterise.
//!
//! Truncating casts are deliberate: registers are 32-bit words of bit fields.
#![allow(clippy::cast_possible_truncation)]

pub mod draw;
mod fifo;
pub mod fixed;
#[cfg(test)]
mod intake_bench;
pub mod overlay;
pub mod pixel;
pub mod program;
pub mod texture;

pub use fifo::Command;
pub use overlay::Picture;

use serde::{Deserialize, Serialize};

/// Guest memory as the GPU sees it.
pub trait Bus {
    /// Reads `out.len()` bytes at guest `address`; false when any is unmapped.
    fn read(&self, address: u32, out: &mut [u8]) -> bool;
    /// Writes `data` at guest `address`; false when any byte is unmapped.
    fn write(&self, address: u32, data: &[u8]) -> bool;
}

impl Bus for crate::mem::Memory {
    fn read(&self, address: u32, out: &mut [u8]) -> bool {
        crate::mem::Memory::read(self, address, out)
    }

    fn write(&self, address: u32, data: &[u8]) -> bool {
        crate::mem::Memory::write(self, address, data)
    }
}

/// The guest address the GPU reaches at physical address `physical` (spec 6.2): page 0 is read
/// through its mirror at `0x80000000`, and the RAM the kernel hands out (the heap and the low
/// contiguous memory) through the kernel's view of it, `0x80000000 + physical`, where the heap
/// lives; everything else (the title's image) at the same address.
#[must_use]
pub fn guest_address(physical: u32) -> u32 {
    use crate::mem::{HEAP_PHYSICAL, KERNEL_ALIAS, LOW_PHYSICAL};
    let physical = physical & 0x07FF_FFFF;
    if physical < 0x1000 || HEAP_PHYSICAL.contains(&physical) || LOW_PHYSICAL.contains(&physical) {
        KERNEL_ALIAS + physical
    } else {
        physical
    }
}

/// The register window's size in bytes.
pub const WINDOW: u32 = 0x0100_0000;

// Registers (offsets in the window) [F: envytools].
pub(crate) const PMC_BOOT_0: u32 = 0x00_0000;
pub(crate) const PMC_INTR_0: u32 = 0x00_0100;
pub(crate) const PMC_INTR_EN_0: u32 = 0x00_0140;
pub(crate) const PBUS_INTR_0: u32 = 0x00_1100;
pub(crate) const PBUS_INTR_EN_0: u32 = 0x00_1140;
pub(crate) const PFIFO_INTR_0: u32 = 0x00_2100;
pub(crate) const PFIFO_INTR_EN_0: u32 = 0x00_2140;
pub(crate) const PFIFO_RAMHT: u32 = 0x00_2210;
pub(crate) const PFIFO_RUNOUT_STATUS: u32 = 0x00_2400;
pub(crate) const CACHE1_PUSH0: u32 = 0x00_3200;
pub(crate) const CACHE1_PUSH1: u32 = 0x00_3204;
pub(crate) const CACHE1_STATUS: u32 = 0x00_3214;
pub(crate) const CACHE1_DMA_PUSH: u32 = 0x00_3220;
pub(crate) const CACHE1_DMA_STATE: u32 = 0x00_3228;
pub(crate) const CACHE1_DMA_INSTANCE: u32 = 0x00_322C;
pub(crate) const CACHE1_DMA_CTL: u32 = 0x00_3230;
pub(crate) const CACHE1_DMA_PUT: u32 = 0x00_3240;
pub(crate) const CACHE1_DMA_GET: u32 = 0x00_3244;
pub(crate) const CACHE1_REF: u32 = 0x00_3248;
pub(crate) const CACHE1_DMA_SUBROUTINE: u32 = 0x00_324C;
pub(crate) const CACHE1_PULL0: u32 = 0x00_3250;
pub(crate) const PVIDEO_INTR_0: u32 = 0x00_8100;
pub(crate) const PVIDEO_INTR_EN_0: u32 = 0x00_8140;
pub(crate) const PVIDEO_BUFFER: u32 = 0x00_8700;
pub(crate) const PVIDEO_STOP: u32 = 0x00_8704;
pub(crate) const PTIMER_INTR_0: u32 = 0x00_9100;
pub(crate) const PTIMER_INTR_EN_0: u32 = 0x00_9140;
pub(crate) const PTIMER_TIME_0: u32 = 0x00_9400;
pub(crate) const PFB_WBC: u32 = 0x10_0410;
pub(crate) const PGRAPH_INTR: u32 = 0x40_0100;
pub(crate) const PGRAPH_NSOURCE: u32 = 0x40_0108;
pub(crate) const PGRAPH_INTR_EN: u32 = 0x40_0140;
pub(crate) const PGRAPH_CTX_CONTROL: u32 = 0x40_0144;
pub(crate) const PGRAPH_CTX_USER: u32 = 0x40_0148;
pub(crate) const PGRAPH_CTX_SWITCH1: u32 = 0x40_014C;
pub(crate) const PGRAPH_CTX_CACHE1: u32 = 0x40_0160;
pub(crate) const PGRAPH_TRAPPED_ADDR: u32 = 0x40_0704;
pub(crate) const PGRAPH_TRAPPED_DATA_LOW: u32 = 0x40_0708;
pub(crate) const PGRAPH_FIFO: u32 = 0x40_0720;
/// The clear values, which Direct3D also uses as a mailbox (see `SET_COLOR_CLEAR_VALUE`).
pub(crate) const PGRAPH_COLOR_CLEAR_VALUE: u32 = 0x40_186C;
pub(crate) const PGRAPH_ZSTENCIL_CLEAR_VALUE: u32 = 0x40_1A88;
pub(crate) const PCRTC_INTR_0: u32 = 0x60_0100;
pub(crate) const PCRTC_INTR_EN_0: u32 = 0x60_0140;
pub(crate) const PRAMIN: u32 = 0x70_0000;
pub(crate) const PRAMIN_END: u32 = 0x80_0000;
pub(crate) const USER: u32 = 0x80_0000;
const USER_END: u32 = USER + 32 * 0x1_0000;

/// `PGRAPH_INTR` bits: a context switch is needed, and the error the NOP request raises.
pub(crate) const INTR_CONTEXT_SWITCH: u32 = 1 << 12;
pub(crate) const INTR_ERROR: u32 = 1 << 20;

/// Each engine's interrupt status and enable registers, and its bit in `PMC_INTR_0`.
const ENGINES: [(u32, u32, u32); 6] = [
    (PFIFO_INTR_0, PFIFO_INTR_EN_0, 8),
    (PGRAPH_INTR, PGRAPH_INTR_EN, 12),
    (PVIDEO_INTR_0, PVIDEO_INTR_EN_0, 16),
    (PTIMER_INTR_0, PTIMER_INTR_EN_0, 20),
    (PCRTC_INTR_0, PCRTC_INTR_EN_0, 24),
    (PBUS_INTR_0, PBUS_INTR_EN_0, 28),
];

/// A finished 3D frame: its commands, which start with the state that applies (spec 10.5).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Frame {
    pub commands: Vec<Command>,
}

/// The object bound to a subchannel.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub(crate) struct Subchannel {
    pub class: u32,
}

/// The latest value of every 3D method, by method: a flat table, as the FIFO stores one on
/// nearly every word it takes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Methods {
    #[serde(with = "crate::hibernate::serde_util::boxed_array")]
    values: Box<[u32; 2048]>,
    /// The methods set, a bit each.
    set: [u64; 32],
}

impl Default for Methods {
    fn default() -> Self {
        Self {
            values: Box::new([0; 2048]),
            set: [0; 32],
        }
    }
}

impl Methods {
    /// A method's slot: methods are words below `0x2000`.
    const fn slot(method: u32) -> usize {
        ((method >> 2) & 0x7FF) as usize
    }

    /// The latest value of `method`, if it was ever set.
    #[must_use]
    pub(crate) fn get(&self, method: u32) -> Option<u32> {
        let slot = Self::slot(method);
        (self.set[slot / 64] & 1 << (slot % 64) != 0).then(|| self.values[slot])
    }

    /// Sets `method`'s latest value.
    #[inline]
    pub(crate) fn insert(&mut self, method: u32, value: u32) {
        let slot = Self::slot(method);
        self.values[slot] = value;
        self.set[slot / 64] |= 1 << (slot % 64);
    }

    /// Every method set, in method order, with its value.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..2048u32)
            .filter(|&slot| self.set[slot as usize / 64] & 1 << (slot % 64) != 0)
            .map(|slot| (slot << 2, self.values[slot as usize]))
    }
}

/// Four-word slots by number, those set and those not: a flat table, as the FIFO sets one word
/// of them at a time.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct Slots {
    words: Vec<Option<[u32; 4]>>,
}

impl Slots {
    /// The most slots kept: far more than the engine's 136 instructions and 192 constants; words
    /// for slots past it are dropped.
    const MOST: u32 = 1 << 16;

    /// The words of slot `slot`, if set.
    #[must_use]
    pub(crate) fn get(&self, slot: u32) -> Option<&[u32; 4]> {
        self.words.get(slot as usize)?.as_ref()
    }

    /// Sets word `word` (0..4) of slot `slot`; the others of a new slot are 0.
    #[inline]
    pub(crate) fn set(&mut self, slot: u32, word: usize, value: u32) {
        if slot >= Self::MOST {
            return;
        }
        let slot = slot as usize;
        if slot >= self.words.len() {
            self.words.resize(slot + 1, None);
        }
        self.words[slot].get_or_insert([0; 4])[word] = value;
    }

    /// The highest slot set.
    #[must_use]
    pub(crate) fn last(&self) -> Option<u32> {
        let at = self.words.iter().rposition(Option::is_some)?;
        u32::try_from(at).ok()
    }

    /// How many slots are set.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.words.iter().filter(|w| w.is_some()).count()
    }
}

/// The vertex program and constants the title streamed, kept for replay at each frame start.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ProgramShadow {
    /// Instruction words by slot.
    pub instructions: Slots,
    /// Constant words by slot.
    pub constants: Slots,
    /// The slot the next streamed program and constant words go to. A word's place in its slot
    /// comes from the method's address, and only the fourth word moves to the next slot.
    pub program_slot: u32,
    pub constant_slot: u32,
}

/// The GPU.
#[derive(Serialize, Deserialize)]
pub struct Gpu {
    /// The register window, PRAMIN included, as words.
    regs: Vec<u32>,
    /// The FIFO waits until the guest clears the NOP error, or a context switch.
    pub(crate) nop_wait: bool,
    pub(crate) context_wait: bool,
    pub(crate) subchannels: [Subchannel; 8],
    /// The latest value of every 3D method (the frame prefix uses the state ones).
    pub(crate) state: Methods,
    pub(crate) program: ProgramShadow,
    /// The attributes' inline values, kept for replay at each frame start.
    pub(crate) inline: draw::InlineShadow,
    /// Frames ended (`FLIP_STALL`).
    pub flips: u64,
    /// Whether frames are recorded, and the commands of the frame being recorded, which start
    /// with its state prefix.
    pub(crate) recording: bool,
    pub(crate) frame: Vec<Command>,
    /// Finished frames and pictures not yet taken.
    pub(crate) frames: Vec<Frame>,
    pictures: Vec<Picture>,
    /// Why the FIFO is stalled, for diagnostics.
    pub stall: Option<String>,
    /// While a visibility test counts, the state it started with and the commands since.
    pub(crate) probe: Option<Vec<Command>>,
    /// The pixels the visibility tests counted since `CLEAR_REPORT_VALUE`.
    pub(crate) zpass_pixels: u32,
    /// The enabled vertex arrays as the array draws read them, until a register they come from
    /// changes.
    pub(crate) arrays: Option<fifo::Arrays>,
    /// The push-buffer words read from guest memory, kept from run to run.
    pub(crate) push_buffer: Vec<u8>,
    /// The vertex arrays an array draw read, and the words it made of them, kept from draw to
    /// draw.
    pub(crate) array_spans: Vec<u8>,
    pub(crate) array_words: Vec<u32>,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu")
            .field("flips", &self.flips)
            .field("nop_wait", &self.nop_wait)
            .field("context_wait", &self.context_wait)
            .field("stall", &self.stall)
            .finish_non_exhaustive()
    }
}

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Gpu {
    /// A GPU after reset: every register 0 but the FIFO's low-mark status.
    #[must_use]
    pub fn new() -> Self {
        let mut gpu = Self {
            regs: vec![0; (WINDOW / 4) as usize],
            nop_wait: false,
            context_wait: false,
            subchannels: [Subchannel::default(); 8],
            state: Methods::default(),
            program: ProgramShadow::default(),
            inline: draw::InlineShadow::default(),
            flips: 0,
            recording: false,
            frame: Vec::new(),
            arrays: None,
            push_buffer: Vec::new(),
            array_spans: Vec::new(),
            array_words: Vec::new(),
            frames: Vec::new(),
            pictures: Vec::new(),
            stall: None,
            probe: None,
            zpass_pixels: 0,
        };
        gpu.set(CACHE1_STATUS, 0x10);
        gpu
    }

    /// A register's stored value.
    #[must_use]
    pub(crate) fn reg(&self, offset: u32) -> u32 {
        self.regs.get((offset / 4) as usize).copied().unwrap_or(0)
    }

    pub(crate) fn set(&mut self, offset: u32, value: u32) {
        if let Some(word) = self.regs.get_mut((offset / 4) as usize) {
            *word = value;
        }
    }

    /// `PMC_INTR_0`: one bit per engine with a pending enabled interrupt.
    #[must_use]
    pub fn pending(&self) -> u32 {
        ENGINES
            .iter()
            .filter(|(status, enable, _)| self.reg(*status) & self.reg(*enable) != 0)
            .fold(0, |bits, (_, _, bit)| bits | 1 << bit)
    }

    /// Whether the hardware interrupt line is asserted.
    #[must_use]
    pub fn line(&self) -> bool {
        self.reg(PMC_INTR_EN_0) & 1 != 0 && self.pending() != 0
    }

    /// The word a USER-window offset aliases, for the current channel.
    fn user_alias(offset: u32) -> Option<u32> {
        if !(USER..USER_END).contains(&offset) {
            return None;
        }
        match offset & 0xFFFF {
            0x40 => Some(CACHE1_DMA_PUT),
            0x44 => Some(CACHE1_DMA_GET),
            0x48 => Some(CACHE1_REF),
            _ => None,
        }
    }

    /// Reads the word at `offset` with its side effects.
    fn read_word(&mut self, offset: u32) -> u32 {
        let offset = Self::user_alias(offset).unwrap_or(offset);
        match offset {
            PMC_BOOT_0 => 0x02A0_00A1,
            PMC_INTR_0 => self.pending(),
            PFIFO_RUNOUT_STATUS => 0x10,
            CACHE1_DMA_CTL => self.reg(offset) | 1 << 31,
            PFB_WBC => self.reg(offset) & !(1 << 16),
            PTIMER_TIME_0 => {
                let time = self.reg(offset).wrapping_add(0x400);
                self.set(offset, time);
                time
            }
            _ => self.reg(offset),
        }
    }

    /// A guest load of `size` bytes (1, 2 or 4) at `offset` in the window.
    pub fn read(&mut self, offset: u32, size: u32) -> u32 {
        let word = self.read_word(offset & !3);
        let shift = 8 * (offset & 3);
        match size {
            1 => (word >> shift) & 0xFF,
            2 => (word >> shift) & 0xFFFF,
            _ => word,
        }
    }

    /// A guest store of `size` bytes at `offset`: merges into the word, applies its effect, and
    /// kicks the pusher.
    pub fn write(&mut self, offset: u32, size: u32, value: u32, bus: &dyn Bus) {
        let aligned = offset & !3;
        let target = Self::user_alias(aligned).unwrap_or(aligned);
        let shift = 8 * (offset & 3);
        let mask: u32 = match size {
            1 => 0xFF << shift,
            2 => 0xFFFF << shift,
            _ => u32::MAX,
        };
        let value = (value << shift) & mask;
        let merged = (self.reg(target) & !mask) | value;
        match target {
            PMC_INTR_0 => {}
            PBUS_INTR_0 | PFIFO_INTR_0 | PVIDEO_INTR_0 | PTIMER_INTR_0 | PCRTC_INTR_0 => {
                self.set(target, self.reg(target) & !value);
            }
            PGRAPH_INTR => {
                self.set(target, self.reg(target) & !value);
                if value & INTR_ERROR != 0 {
                    self.nop_wait = false;
                }
                if value & INTR_CONTEXT_SWITCH != 0 {
                    self.context_wait = false;
                }
            }
            PVIDEO_BUFFER => {
                for (bit, buffer) in [(1, 0), (1 << 4, 1)] {
                    if value & bit != 0 {
                        if let Some(picture) = overlay::picture(self, buffer, bus) {
                            self.pictures.push(picture);
                        }
                        self.set(target, self.reg(target) | bit);
                    }
                }
            }
            PVIDEO_STOP => {
                self.set(target, merged);
                if merged != 0 {
                    self.set(PVIDEO_BUFFER, self.reg(PVIDEO_BUFFER) & !0x11);
                }
            }
            _ => {
                // A DMA object the vertex arrays name may have changed.
                if (PRAMIN..PRAMIN_END).contains(&target) {
                    self.arrays = None;
                }
                self.set(target, merged);
            }
        }
        self.run_pusher(bus);
    }

    /// A vertical blank (spec 10.7): sets the CRTC's interrupt, frees overlay buffers in use.
    /// Returns whether the interrupt line is now asserted.
    pub fn vblank(&mut self) -> bool {
        self.set(PCRTC_INTR_0, self.reg(PCRTC_INTR_0) | 1);
        let in_use = self.reg(PVIDEO_BUFFER) & 0x11;
        if in_use != 0 {
            self.set(PVIDEO_BUFFER, self.reg(PVIDEO_BUFFER) & !0x11);
            self.set(PVIDEO_INTR_0, self.reg(PVIDEO_INTR_0) | in_use);
        }
        self.line()
    }

    /// The FIFO's state, for `ANYXBOX_TRACE_FPS`.
    #[must_use]
    pub fn fifo_state(&self) -> String {
        format!(
            "GET {:#x} PUT {:#x} push {:#x} state {:#x}{}{}{}",
            self.reg(CACHE1_DMA_GET),
            self.reg(CACHE1_DMA_PUT),
            self.reg(CACHE1_DMA_PUSH),
            self.reg(CACHE1_DMA_STATE),
            if self.nop_wait { ", NOP wait" } else { "" },
            if self.context_wait {
                ", context wait"
            } else {
                ""
            },
            self.stall
                .as_ref()
                .map(|s| format!(", stalled: {s}"))
                .unwrap_or_default(),
        )
    }

    /// Starts or stops recording frames (a frame sink is installed).
    pub fn record_frames(&mut self, on: bool) {
        self.recording = on;
        self.frame.clear();
        if on {
            self.frame = self.frame_prefix();
        }
    }

    /// The finished frames since the last call.
    pub fn take_frames(&mut self) -> Vec<Frame> {
        std::mem::take(&mut self.frames)
    }

    /// The overlay pictures submitted since the last call.
    pub fn take_pictures(&mut self) -> Vec<Picture> {
        std::mem::take(&mut self.pictures)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A flat guest memory for tests: 64 MiB from address 0, and the page-0 mirror.
    pub struct FlatBus(pub RefCell<Vec<u8>>);

    impl FlatBus {
        pub fn new() -> Self {
            Self(RefCell::new(vec![0; 0x0400_0000]))
        }

        fn index(address: u32) -> usize {
            (if address >= 0x8000_0000 {
                address - 0x8000_0000
            } else {
                address
            }) as usize
        }

        pub fn put32(&self, address: u32, value: u32) {
            let i = Self::index(address);
            self.0.borrow_mut()[i..i + 4].copy_from_slice(&value.to_le_bytes());
        }

        pub fn get32(&self, address: u32) -> u32 {
            let i = Self::index(address);
            u32::from_le_bytes(self.0.borrow()[i..i + 4].try_into().unwrap())
        }
    }

    impl Bus for FlatBus {
        fn read(&self, address: u32, out: &mut [u8]) -> bool {
            let i = Self::index(address);
            let memory = self.0.borrow();
            match memory.get(i..i + out.len()) {
                Some(bytes) => {
                    out.copy_from_slice(bytes);
                    true
                }
                None => false,
            }
        }

        fn write(&self, address: u32, data: &[u8]) -> bool {
            let i = Self::index(address);
            let mut memory = self.0.borrow_mut();
            match memory.get_mut(i..i + data.len()) {
                Some(bytes) => {
                    bytes.copy_from_slice(data);
                    true
                }
                None => false,
            }
        }
    }

    #[test]
    fn identity_and_fixed_registers() {
        let mut gpu = Gpu::new();
        assert_eq!(gpu.read(PMC_BOOT_0, 4), 0x02A0_00A1);
        assert_eq!(gpu.read(PFIFO_RUNOUT_STATUS, 4), 0x10);
        assert_eq!(gpu.read(CACHE1_STATUS, 4), 0x10);
        assert_eq!(gpu.read(CACHE1_DMA_CTL, 4), 1 << 31);
        assert_eq!(gpu.read(PMC_BOOT_0 + 1, 1), 0x00);
        assert_eq!(gpu.read(PMC_BOOT_0 + 2, 2), 0x02A0);
    }

    #[test]
    fn the_timer_advances_on_every_read() {
        let mut gpu = Gpu::new();
        assert_eq!(gpu.read(PTIMER_TIME_0, 4), 0x400);
        assert_eq!(gpu.read(PTIMER_TIME_0, 4), 0x800);
    }

    #[test]
    fn byte_writes_merge_and_a_flush_completes_at_once() {
        let bus = FlatBus::new();
        let mut gpu = Gpu::new();
        gpu.write(PRAMIN, 4, 0x1122_3344, &bus);
        gpu.write(PRAMIN + 1, 1, 0xAA, &bus);
        assert_eq!(gpu.read(PRAMIN, 4), 0x1122_AA44);
        gpu.write(PFB_WBC, 4, 1 << 16, &bus);
        assert_eq!(gpu.read(PFB_WBC, 4), 0);
    }

    #[test]
    fn interrupts_are_write_one_to_clear_and_assert_the_line_when_enabled() {
        let bus = FlatBus::new();
        let mut gpu = Gpu::new();
        assert!(!gpu.vblank(), "nothing is enabled");
        assert_eq!(gpu.read(PCRTC_INTR_0, 4), 1);
        gpu.write(PCRTC_INTR_EN_0, 4, 1, &bus);
        assert_eq!(gpu.read(PMC_INTR_0, 4), 1 << 24);
        assert!(!gpu.line());
        gpu.write(PMC_INTR_EN_0, 4, 1, &bus);
        assert!(gpu.line());
        gpu.write(PCRTC_INTR_0, 4, 1, &bus);
        assert_eq!(gpu.read(PMC_INTR_0, 4), 0);
        assert!(!gpu.line());
    }

    #[test]
    fn a_blank_frees_the_overlay_buffers() {
        let bus = FlatBus::new();
        let mut gpu = Gpu::new();
        gpu.write(PVIDEO_BUFFER, 4, 0x10, &bus);
        assert_eq!(gpu.read(PVIDEO_BUFFER, 4), 0x10);
        gpu.vblank();
        assert_eq!(gpu.read(PVIDEO_BUFFER, 4), 0);
        assert_eq!(gpu.read(PVIDEO_INTR_0, 4), 0x10);
    }

    #[test]
    fn the_user_window_aliases_the_fifo_pointers() {
        let bus = FlatBus::new();
        let mut gpu = Gpu::new();
        gpu.write(USER + 0x40, 4, 0x1234, &bus);
        assert_eq!(gpu.read(CACHE1_DMA_PUT, 4), 0x1234);
        assert_eq!(gpu.read(USER + 0x40, 4), 0x1234);
    }
}

#[cfg(test)]
mod bus_tests {
    use super::guest_address;

    #[test]
    fn the_gpu_reaches_the_heap_and_low_memory_through_the_kernel_view() {
        assert_eq!(guest_address(0x0100_0000), 0x8100_0000);
        assert_eq!(guest_address(0x03FF_FFFF), 0x83FF_FFFF);
        assert_eq!(
            guest_address(0xF123_4560),
            0x8123_4560,
            "the GPU view, masked"
        );
        assert_eq!(guest_address(0x0000_4000), 0x8000_4000);
        assert_eq!(guest_address(0x0000_BFFF), 0x8000_BFFF);
        assert_eq!(guest_address(0x0000_0010), 0x8000_0010, "page 0's mirror");
        // The title's image is where it is.
        assert_eq!(guest_address(0x0000_C000), 0x0000_C000);
        assert_eq!(guest_address(0x0002_0000), 0x0002_0000);
        assert_eq!(guest_address(0x0400_0000), 0x0400_0000);
    }
}
