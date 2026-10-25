//! What a snapshot keeps of the kernel and of guest memory.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::gpu::Gpu;
use crate::kernel::fs::FileSystem;
use crate::kernel::heap::Heap;
use crate::kernel::hooks::Gamepad;
use crate::kernel::object::Objects;
use crate::kernel::sound::Sound;
use crate::kernel::thread::Threads;
use crate::kernel::time::Clock;
use crate::kernel::{Dpc, Kernel};
use crate::mem::{HEAP_BASE, HEAP_END, HOST_PAGE, LOW_BASE, LOW_END, Memory};

/// The guest state of the kernel: everything but guest memory and what belongs to this process
/// (the disc, where saves go, the renderer, the hook plan, tracing).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct KernelState {
    pub heap: Heap,
    pub low_heap: Heap,
    pub objects: Objects,
    pub threads: Threads,
    pub current: u32,
    pub clock: Clock,
    pub dpcs: HashMap<u32, Dpc>,
    pub fs: FileSystem,
    pub eeprom: Vec<u8>,
    pub cache_pages: Option<u32>,
    pub tls_template: Vec<u8>,
    pub tls_zero_fill: u32,
    pub unstarted: Vec<u32>,
    pub calls: u64,
    pub av_saved_data: u32,
    pub display_mode: [u32; 6],
    pub reported: HashSet<u32>,
    pub interrupts: Vec<u32>,
    pub section_waiters: HashMap<u32, VecDeque<u32>>,
    pub hooks: SavedHooks,
    pub sound: Sound,
    pub gpu: Gpu,
    pub next_vblank: u64,
    pub sync_irps: HashMap<u32, (u32, u32)>,
}

/// The hooks' state that changes as the title runs: the vtable slots made since boot (by class
/// name) and the gamepad. The plan itself is made again at boot.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedHooks {
    pub slots: Vec<(String, u32)>,
    pub vtables: Vec<(String, u32)>,
    pub gamepad: Gamepad,
}

/// A class name for the hooks' tables, which keep static names. A restore makes a few, once.
fn class_name(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

impl KernelState {
    /// Lends the guest state of `kernel` to a snapshot, leaving placeholders until
    /// [`put_back`](Self::put_back).
    pub fn take(kernel: &mut Kernel) -> Self {
        use std::mem::take;
        Self {
            heap: take(&mut kernel.heap),
            low_heap: take(&mut kernel.low_heap),
            objects: take(&mut kernel.objects),
            threads: take(&mut kernel.threads),
            current: kernel.current,
            clock: take(&mut kernel.clock),
            dpcs: take(&mut kernel.dpcs),
            fs: take(&mut kernel.fs),
            eeprom: kernel.eeprom.to_vec(),
            cache_pages: kernel.cache_pages,
            tls_template: take(&mut kernel.tls_template),
            tls_zero_fill: kernel.tls_zero_fill,
            unstarted: take(&mut kernel.unstarted),
            calls: kernel.calls,
            av_saved_data: kernel.av_saved_data,
            display_mode: kernel.display_mode,
            reported: take(&mut kernel.reported),
            interrupts: take(&mut kernel.interrupts),
            section_waiters: take(&mut kernel.section_waiters),
            hooks: SavedHooks {
                slots: kernel
                    .hooks
                    .slots
                    .iter()
                    .map(|&(class, slot)| (class.to_owned(), slot))
                    .collect(),
                vtables: kernel
                    .hooks
                    .vtables
                    .iter()
                    .map(|(&class, &vtable)| (class.to_owned(), vtable))
                    .collect(),
                gamepad: take(&mut kernel.hooks.gamepad),
            },
            sound: take(&mut kernel.sound),
            gpu: take(&mut kernel.gpu),
            next_vblank: kernel.next_vblank,
            sync_irps: take(&mut kernel.sync_irps),
        }
    }

    /// Gives the state back to the kernel it was taken from.
    pub fn put_back(self, kernel: &mut Kernel) {
        kernel.heap = self.heap;
        kernel.low_heap = self.low_heap;
        kernel.objects = self.objects;
        kernel.threads = self.threads;
        kernel.clock = self.clock;
        kernel.dpcs = self.dpcs;
        kernel.fs = self.fs;
        kernel.tls_template = self.tls_template;
        kernel.unstarted = self.unstarted;
        kernel.reported = self.reported;
        kernel.interrupts = self.interrupts;
        kernel.section_waiters = self.section_waiters;
        kernel.hooks.gamepad = self.hooks.gamepad;
        kernel.sound = self.sound;
        kernel.gpu = self.gpu;
        kernel.sync_irps = self.sync_irps;
    }

    /// Puts the saved state into a kernel just booted for it, which keeps its memory, disc, save
    /// file, renderer, hook plan and tracing.
    pub fn apply(self, kernel: &mut Kernel) {
        kernel.heap.adopt(self.heap);
        kernel.low_heap.adopt(self.low_heap);
        kernel.objects = self.objects;
        kernel.threads = self.threads;
        kernel.current = self.current;
        kernel.clock.resume_from(&self.clock);
        kernel.dpcs = self.dpcs;
        kernel.fs.adopt(self.fs);
        if let Ok(eeprom) = self.eeprom.try_into() {
            kernel.eeprom = eeprom;
        }
        kernel.cache_pages = self.cache_pages;
        kernel.tls_template = self.tls_template;
        kernel.tls_zero_fill = self.tls_zero_fill;
        kernel.unstarted = self.unstarted;
        kernel.calls = self.calls;
        kernel.av_saved_data = self.av_saved_data;
        kernel.display_mode = self.display_mode;
        kernel.reported = self.reported;
        kernel.interrupts = self.interrupts;
        kernel.section_waiters = self.section_waiters;
        kernel.hooks.slots = self
            .hooks
            .slots
            .into_iter()
            .map(|(class, slot)| (class_name(class), slot))
            .collect();
        kernel.hooks.vtables = self
            .hooks
            .vtables
            .into_iter()
            .map(|(class, vtable)| (class_name(class), vtable))
            .collect();
        kernel.hooks.gamepad = self.hooks.gamepad;
        kernel.sound = self.sound;
        let recording = kernel.gpu.recording;
        kernel.gpu = self.gpu;
        kernel.gpu.recording = recording;
        kernel.next_vblank = self.next_vblank;
        kernel.sync_irps = self.sync_irps;
    }
}

/// Guest memory as a snapshot keeps it: the committed pages outside the shared RAM, then the
/// shared RAM (the heap, then the low contiguous memory) once.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MemoryImage {
    pub pages: Vec<(u32, Vec<u8>)>,
    pub shared: Vec<u8>,
}

impl MemoryImage {
    /// Reads guest memory. The title is frozen, so nothing writes it meanwhile.
    #[must_use]
    pub fn read(memory: &Memory) -> Self {
        let pages = memory
            .committed_pages()
            .into_iter()
            .map(|address| {
                let mut bytes = vec![0; HOST_PAGE];
                memory.read(address, &mut bytes);
                (address, bytes)
            })
            .collect();
        let mut shared = vec![0; (HEAP_END - HEAP_BASE + LOW_END - LOW_BASE) as usize];
        let (heap, low) = shared.split_at_mut((HEAP_END - HEAP_BASE) as usize);
        memory.read(HEAP_BASE, heap);
        memory.read(LOW_BASE, low);
        Self { pages, shared }
    }

    /// Writes the image into `memory`, committing its pages.
    pub fn load(&self, memory: &Memory) {
        for (address, bytes) in &self.pages {
            memory.commit(*address);
            memory.write(*address, bytes);
        }
        let split = ((HEAP_END - HEAP_BASE) as usize).min(self.shared.len());
        let (heap, low) = self.shared.split_at(split);
        memory.write(HEAP_BASE, heap);
        memory.write(LOW_BASE, low);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::testing;

    /// A kernel with some state of each kind a title makes.
    fn busy_kernel() -> Kernel {
        let mut k = testing::kernel();
        testing::threads(&mut k, 2);
        k.heap.alloc(0x100);
        k.gpu.flips = 3;
        k.clock.pass(5_000_000);
        k.hooks.slots.push(("IDirectSound", 4));
        k.hooks.vtables.insert("IDirectSound", 0x8100_0000);
        k.hooks.gamepad.packet = 9;
        k.eeprom[7] = 0x5A;
        k
    }

    #[test]
    fn the_kernel_state_round_trips_into_a_fresh_kernel() {
        let mut k = busy_kernel();
        let state = KernelState::take(&mut k);
        let bytes = postcard::to_stdvec(&state).unwrap();
        let back: KernelState = postcard::from_bytes(&bytes).unwrap();
        let mut fresh = testing::kernel();
        let base = fresh.memory.heap();
        back.apply(&mut fresh);
        assert_eq!(fresh.threads.live(), 3);
        assert_eq!(fresh.gpu.flips, 3);
        assert_eq!(
            fresh.clock.now(),
            crate::kernel::time::BOOT_CLOCK + 5_000_000
        );
        assert_eq!(fresh.hooks.slots, [("IDirectSound", 4)]);
        assert_eq!(fresh.hooks.vtables.get("IDirectSound"), Some(&0x8100_0000));
        assert_eq!(fresh.hooks.gamepad.packet, 9);
        assert_eq!(fresh.eeprom[7], 0x5A);
        assert_eq!(fresh.memory.heap(), base, "the fresh kernel's memory stays");
        // The heap goes on from where it was: the next block is where it would have been.
        let mut original = busy_kernel();
        assert_eq!(fresh.heap.alloc(0x40), original.heap.alloc(0x40));
    }

    #[test]
    fn state_taken_for_encoding_goes_back() {
        let mut k = busy_kernel();
        let state = KernelState::take(&mut k);
        assert_eq!(k.gpu.flips, 0);
        state.put_back(&mut k);
        assert_eq!((k.gpu.flips, k.threads.live()), (3, 3));
        assert_eq!(k.hooks.gamepad.packet, 9);
    }

    #[test]
    fn guest_memory_round_trips_page_by_page_and_through_the_shared_ram() {
        use crate::mem::{HEAP_BASE, Memory};
        let memory = Memory::new().unwrap();
        memory.commit(0x0001_0000);
        memory.write(0x0001_0010, b"title image");
        memory.write(HEAP_BASE + 0x20, b"heap block");
        let image = MemoryImage::read(&memory);
        assert!(
            image
                .pages
                .iter()
                .any(|(address, _)| *address == 0x0001_0000)
        );
        assert!(
            image
                .pages
                .iter()
                .all(|(address, _)| *address < HEAP_BASE || *address >= crate::mem::HEAP_END),
            "the heap goes in the shared RAM, not page by page"
        );
        let other = Memory::new().unwrap();
        image.load(&other);
        let mut out = [0u8; 11];
        assert!(other.read(0x0001_0010, &mut out));
        assert_eq!(&out, b"title image");
        let mut out = [0u8; 10];
        assert!(other.read(HEAP_BASE + 0x20, &mut out));
        assert_eq!(&out, b"heap block");
        // The other views of the shared RAM see it too.
        assert!(other.read(
            0x1000_0000 + crate::mem::physical(HEAP_BASE + 0x20),
            &mut out
        ));
        assert_eq!(&out, b"heap block");
    }
}
