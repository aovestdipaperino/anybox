//! The kernel: state, guest memory access, and call dispatch. Spec 8.
//!
//! Truncating casts are deliberate throughout: guest registers are 32 bits, a 64-bit result is
//! split into EDX:EAX, and byte arguments are the low byte of a dword, as on x86.
#![allow(clippy::cast_possible_truncation)]

pub mod fs;
pub mod handlers;
pub mod heap;
pub mod hooks;
pub mod mixer;
pub mod object;
pub mod ordinals;
pub mod sound;
pub mod spawn;
pub mod status;
pub mod sync;
pub mod thread;
pub mod time;

use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet, VecDeque};

use crate::mem::Memory;
use heap::Heap;
use object::Objects;
use ordinals::Convention;
use thread::Threads;

/// The trap addresses (spec 5.5, 6.2).
pub const TRAP_BASE: u32 = 0x8001_0000;
pub const HOOK_BASE: u32 = 0x8001_0800;
pub const THREAD_RETURN: u32 = 0x8001_0FFC;

/// A guest access the runtime made to uncommitted memory (spec 6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault(pub u32);

/// The result of a guest memory access by the kernel.
pub type KResult<T> = Result<T, Fault>;

/// A DPC as `KeInitializeDpc` recorded it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Dpc {
    pub routine: u32,
    pub context: u32,
    /// Whether a thread is running it now.
    pub queued: bool,
}

/// How much `ANYXBOX_TRACE` prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trace {
    Off,
    /// Every call but the critical-section ones.
    On,
    All,
}

/// The kernel's state, behind one lock.
#[derive(Debug)]
pub struct Kernel {
    pub memory: Memory,
    pub heap: Heap,
    /// The low contiguous memory, under 16 MiB physical, for `MmAllocateContiguousMemoryEx`
    /// callers whose highest acceptable address is under the heap.
    pub low_heap: Heap,
    pub objects: Objects,
    pub threads: Threads,
    /// The running guest thread.
    pub current: u32,
    pub clock: time::Clock,
    pub dpcs: HashMap<u32, Dpc>,
    pub fs: fs::FileSystem,
    /// The EEPROM image (spec 8.12).
    pub eeprom: [u8; 256],
    /// The file system cache's size in pages, once `FscSetCacheSize` set it.
    pub cache_pages: Option<u32>,
    /// The TLS template and zero-fill size, for new threads (spec 7.5).
    pub tls_template: Vec<u8>,
    pub tls_zero_fill: u32,
    /// The raw bytes of every section by virtual address, for `XeLoadSection`.
    pub sections: Vec<(u32, Vec<u8>)>,
    /// Guest threads created whose host threads have not been started yet.
    pub unstarted: Vec<u32>,
    /// Kernel calls made, for the preemption rule (spec 7.4).
    pub calls: u64,
    /// `AvGetSavedDataAddress`.
    pub av_saved_data: u32,
    /// The last `AvSetDisplayMode`: base, step, mode, format, pitch, frame buffer (spec 8.12).
    pub display_mode: [u32; 6],
    pub trace: Trace,
    /// The untested exports already reported.
    pub reported: HashSet<u32>,
    /// Connected interrupt objects (`KINTERRUPT` addresses).
    pub interrupts: Vec<u32>,
    /// The threads waiting for each contended critical section, oldest first.
    pub section_waiters: HashMap<u32, VecDeque<u32>>,
    /// The replaced XDK functions and the vtable slots after them.
    pub hooks: hooks::Hooks,
    /// `DirectSound` voices and streams.
    pub sound: sound::Sound,
    /// The GPU model, and the clock tick of the next vertical blank.
    pub gpu: crate::gpu::Gpu,
    pub next_vblank: u64,
    /// `ANYXBOX_TRACE_FPS`: the clock tick of the last report and the flips then.
    pub fps_report: Option<(u64, u64)>,
    /// Where finished 3D frames go, when something renders them.
    pub frame_sink: Option<std::sync::mpsc::SyncSender<crate::render::FrameJob>>,
    /// IRPs sent by `IoSynchronous*Request` and not yet completed: where the bytes transferred
    /// go, and the thread waiting for them.
    pub sync_irps: HashMap<u32, (u32, u32)>,
}

impl Kernel {
    /// Reads `out.len()` bytes at `address`.
    ///
    /// # Errors
    ///
    /// A fault when any byte is uncommitted.
    pub fn read_into(&self, address: u32, out: &mut [u8]) -> KResult<()> {
        if out.is_empty() || self.memory.read(address, out) {
            Ok(())
        } else {
            Err(Fault(address))
        }
    }

    /// Reads `len` bytes at `address`.
    ///
    /// # Errors
    ///
    /// A fault when any byte is uncommitted.
    pub fn read(&self, address: u32, len: u32) -> KResult<Vec<u8>> {
        let mut out = vec![0; len as usize];
        self.read_into(address, &mut out)?;
        Ok(out)
    }

    /// Writes `data` at `address`.
    ///
    /// # Errors
    ///
    /// A fault when any byte is uncommitted.
    pub fn write(&self, address: u32, data: &[u8]) -> KResult<()> {
        if data.is_empty() || self.memory.write(address, data) {
            Ok(())
        } else {
            Err(Fault(address))
        }
    }

    /// # Errors
    ///
    /// A fault when the byte is uncommitted.
    pub fn r8(&self, address: u32) -> KResult<u8> {
        let mut b = [0; 1];
        self.read_into(address, &mut b)?;
        Ok(b[0])
    }

    /// # Errors
    ///
    /// A fault when a byte is uncommitted.
    pub fn r16(&self, address: u32) -> KResult<u16> {
        let mut b = [0; 2];
        self.read_into(address, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    /// # Errors
    ///
    /// A fault when a byte is uncommitted.
    pub fn r32(&self, address: u32) -> KResult<u32> {
        let mut b = [0; 4];
        self.read_into(address, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// # Errors
    ///
    /// A fault when a byte is uncommitted.
    pub fn r64(&self, address: u32) -> KResult<u64> {
        let mut b = [0; 8];
        self.read_into(address, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// # Errors
    ///
    /// A fault when the byte is uncommitted.
    pub fn w8(&self, address: u32, value: u8) -> KResult<()> {
        self.write(address, &[value])
    }

    /// # Errors
    ///
    /// A fault when a byte is uncommitted.
    pub fn w16(&self, address: u32, value: u16) -> KResult<()> {
        self.write(address, &value.to_le_bytes())
    }

    /// # Errors
    ///
    /// A fault when a byte is uncommitted.
    pub fn w32(&self, address: u32, value: u32) -> KResult<()> {
        self.write(address, &value.to_le_bytes())
    }

    /// # Errors
    ///
    /// A fault when a byte is uncommitted.
    pub fn w64(&self, address: u32, value: u64) -> KResult<()> {
        self.write(address, &value.to_le_bytes())
    }

    /// Writes a dword when `address` is not null (an optional out parameter).
    ///
    /// # Errors
    ///
    /// A fault when a byte is uncommitted.
    pub fn w32_opt(&self, address: u32, value: u32) -> KResult<()> {
        if address == 0 {
            Ok(())
        } else {
            self.w32(address, value)
        }
    }

    /// Reads a NUL-terminated 8-bit string (at most 64 KiB).
    ///
    /// # Errors
    ///
    /// A fault when a byte before the NUL is uncommitted.
    pub fn c_string(&self, address: u32) -> KResult<Vec<u8>> {
        let mut out = Vec::new();
        for i in 0..0x1_0000 {
            let b = self.r8(address.wrapping_add(i))?;
            if b == 0 {
                break;
            }
            out.push(b);
        }
        Ok(out)
    }

    /// Reads the text of an `ANSI_STRING` at `address` (spec 8.15): Latin-1.
    ///
    /// # Errors
    ///
    /// A fault when the structure or its text is uncommitted.
    pub fn ansi_string(&self, address: u32) -> KResult<String> {
        let len = self.r16(address)?;
        let buffer = self.r32(address + 4)?;
        let bytes = self.read(buffer, u32::from(len))?;
        Ok(bytes.iter().map(|&b| char::from(b)).collect())
    }

    /// The KTHREAD of the running thread, as `fs:[0x28]` reads it.
    #[must_use]
    pub fn current_kthread(&self) -> u32 {
        self.threads.get(self.current).map_or(0, |t| t.kthread)
    }

    /// Prints `untested kernel call NAME (ordinal N) with [args]` once per export.
    fn report_untested(&mut self, ordinal: u32, name: &str, args: &[u32]) {
        if handlers::is_untested(name) && self.reported.insert(ordinal) {
            eprintln!("untested kernel call {name} (ordinal {ordinal}) with {args:x?}");
        }
    }
}

/// One kernel call's arguments.
#[derive(Debug)]
pub struct Call {
    pub args: Vec<u32>,
    /// ESP at the call: it points at the return address.
    pub esp: u32,
}

impl Call {
    /// Argument `i`, or 0 past the end.
    #[must_use]
    pub fn arg(&self, i: usize) -> u32 {
        self.args.get(i).copied().unwrap_or(0)
    }

    /// Arguments `i` (low) and `i + 1` (high) as a 64-bit value.
    #[must_use]
    pub fn arg64(&self, i: usize) -> u64 {
        u64::from(self.arg(i)) | (u64::from(self.arg(i + 1)) << 32)
    }
}

/// What a handler returns (spec 8.1).
#[derive(Debug, PartialEq, Eq)]
pub enum CallResult {
    /// A 32-bit value.
    Value(u32),
    /// A 64-bit value: EDX:EAX = high:low.
    Value64(u32, u32),
    /// The thread now waits; the call resumes with the value it is woken with.
    Block,
    /// The call completes, but other ready threads run first; it returns 0.
    Yield,
    /// The thread ends.
    Exit,
    /// The run ends.
    Halt,
    /// The run stops, printing the reason.
    Unsupported(String),
}

/// A kernel call handler.
pub type Handler = fn(&mut Kernel, &Call) -> KResult<CallResult>;

/// What the ABI layer must do after a native call.
#[derive(Debug, PartialEq, Eq)]
pub enum Dispatch {
    /// Completed: set EAX, EDX and ESP; `wide` for a 64-bit result.
    Done {
        eax: u32,
        edx: u32,
        esp: u32,
        wide: bool,
    },
    /// The thread waits: reschedule, then complete with the resume value in EAX.
    Block { edx: u32, esp: u32 },
    /// Completed with EAX 0 after other ready threads ran.
    Yield { edx: u32, esp: u32 },
    /// The calling thread ends.
    ThreadEnd,
    /// The run ends normally.
    Halt,
    /// The call cannot complete; the message has been printed.
    Fail,
}

/// Reads a call's arguments from the guest stack (spec 8.1): stdcall `arg_bytes / 4` dwords;
/// fastcall ECX, EDX and then the stack; cdecl six dwords where an unmapped slot reads as 0.
fn read_args(
    kernel: &Kernel,
    esp: u32,
    ecx: u32,
    edx: u32,
    convention: Convention,
    bytes: u32,
) -> Vec<u32> {
    let stack = |i: u32| kernel.r32(esp.wrapping_add(4 + 4 * i)).unwrap_or(0);
    match convention {
        Convention::Stdcall => (0..bytes / 4).map(stack).collect(),
        Convention::Fastcall => {
            let mut args = vec![ecx, edx];
            args.extend((0..bytes.saturating_sub(8) / 4).map(stack));
            args.truncate((bytes / 4) as usize);
            args
        }
        Convention::Cdecl => (0..6).map(stack).collect(),
    }
}

/// The critical-section calls `ANYXBOX_TRACE=all` adds.
fn is_critical_section_call(name: &str) -> bool {
    matches!(
        name,
        "RtlEnterCriticalSection" | "RtlLeaveCriticalSection" | "RtlTryEnterCriticalSection"
    )
}

/// Runs the kernel call at trap address `target` (spec 8.1). XDK hooks go to `hooks`.
pub fn dispatch(kernel: &mut Kernel, target: u32, esp: u32, ecx: u32, edx: u32) -> Dispatch {
    kernel.calls += 1;
    kernel.service_time();
    if target == TRAP_BASE || target == THREAD_RETURN {
        return Dispatch::ThreadEnd;
    }
    let Some(resolved) = resolve(kernel, target, esp, ecx, edx) else {
        return Dispatch::Fail;
    };
    let call = Call {
        args: resolved.args,
        esp,
    };
    let result = (resolved.handler)(kernel, &call).unwrap_or_else(|Fault(address)| {
        CallResult::Unsupported(format!("bad pointer {address:#010x}"))
    });
    let traced = match kernel.trace {
        Trace::Off => false,
        Trace::On => !is_critical_section_call(resolved.name),
        Trace::All => true,
    };
    if traced {
        let args: Vec<String> = call.args.iter().map(|a| format!("{a:#x}")).collect();
        let shown = match &result {
            CallResult::Value(v) => format!("{v:#x}"),
            CallResult::Value64(low, high) => {
                format!("{:#x}", u64::from(*low) | (u64::from(*high) << 32))
            }
            CallResult::Block => "<switch>".to_owned(),
            other => format!("{other:?}"),
        };
        eprintln!(
            "{} {}({}) = {shown}",
            resolved.label,
            resolved.name,
            args.join(", ")
        );
    }
    let esp = esp + 4 + resolved.pops;
    match result {
        CallResult::Value(eax) => Dispatch::Done {
            eax,
            edx,
            esp,
            wide: false,
        },
        CallResult::Value64(eax, high) => Dispatch::Done {
            eax,
            edx: high,
            esp,
            wide: true,
        },
        CallResult::Block => Dispatch::Block { edx, esp },
        CallResult::Yield => Dispatch::Yield { edx, esp },
        CallResult::Exit => Dispatch::ThreadEnd,
        CallResult::Halt => Dispatch::Halt,
        CallResult::Unsupported(reason) => {
            eprintln!("{}: {reason}", resolved.name);
            Dispatch::Fail
        }
    }
}

/// A trap resolved to what runs: the trace label (`@ORD`, or `xdk`), the name, the handler,
/// the arguments, and the bytes the callee pops.
struct Resolved {
    label: String,
    name: &'static str,
    handler: Handler,
    args: Vec<u32>,
    pops: u32,
}

/// Resolves a kernel ordinal or a hook trap (spec 8.1, 5.6, 9.2). Prints why and returns `None`
/// when nothing can run there.
fn resolve(kernel: &mut Kernel, target: u32, esp: u32, ecx: u32, edx: u32) -> Option<Resolved> {
    if target >= HOOK_BASE {
        let index = ((target - HOOK_BASE) / 4) as usize;
        let (name, handler, pops) = match kernel.hooks.target(index) {
            hooks::Target::Planned(None, pops) => ("(stub)", handlers::nothing as Handler, pops),
            hooks::Target::Planned(Some(name), pops) => {
                let Some(handler) = hooks::handler(name) else {
                    eprintln!("an XDK function the runtime does not replace: {name}");
                    return None;
                };
                (name, handler, pops)
            }
            hooks::Target::Slot(class, slot) => {
                let Some((name, handler, pops)) = hooks::slot(class, slot) else {
                    eprintln!("an XDK method the runtime does not replace: {class}::slot{slot}");
                    return None;
                };
                (name, handler, pops)
            }
            hooks::Target::Unknown => {
                eprintln!("a call to trap {target:#010x}, which no hook or method has");
                return None;
            }
        };
        let pops = u32::from(pops);
        let args = read_args(kernel, esp, ecx, edx, Convention::Stdcall, pops);
        return Some(Resolved {
            label: "xdk".to_owned(),
            name,
            handler,
            args,
            pops,
        });
    }
    let ordinal = (target - TRAP_BASE) / 4;
    let Some(export) = ordinals::export(ordinal) else {
        eprintln!("a call to kernel ordinal {ordinal}, which the kernel does not export");
        return None;
    };
    let args = read_args(kernel, esp, ecx, edx, export.convention, export.arg_bytes);
    let Some(handler) = handlers::lookup(export.name) else {
        eprintln!(
            "kernel call {} (ordinal {ordinal}) has no handler; arguments {args:x?}",
            export.name
        );
        return None;
    };
    kernel.report_untested(ordinal, export.name, &args);
    Some(Resolved {
        label: format!("@{ordinal}"),
        name: export.name,
        handler,
        args,
        pops: export.callee_pops(),
    })
}

#[cfg(test)]
pub mod testing {
    //! A kernel with no title, for unit tests: guest memory, the heap, a KPCR, and the boot
    //! thread as thread 1 (current).

    use super::object::{Body, Objects};
    use super::thread::{State, Thread, Threads};
    use super::{Kernel, Trace};
    use crate::boot::{BOOT_KTHREAD, KPCR, KTHREAD_BLOCKS, KTHREAD_SIZE, KTHREAD_SLOTS};
    use crate::mem::{HEAP_BASE, HEAP_END, LOW_BASE, LOW_END, Memory};
    use std::collections::{HashMap, HashSet};

    /// # Panics
    ///
    /// Panics when the guest space cannot be reserved.
    #[must_use]
    pub fn kernel() -> Kernel {
        let memory = Memory::new().expect("the guest space");
        memory.commit_range(KPCR, 0x1000);
        memory.commit_range(KTHREAD_BLOCKS, KTHREAD_SLOTS * KTHREAD_SIZE);
        let heap = super::heap::Heap::new(memory.heap(), HEAP_BASE, HEAP_END - HEAP_BASE);
        let low_heap = super::heap::Heap::new(memory.low(), LOW_BASE, LOW_END - LOW_BASE);
        let mut objects = Objects::new();
        let object = objects.add_at(BOOT_KTHREAD, Body::Thread { tid: 1 });
        let boot = Thread {
            id: 1,
            state: State::Ready,
            suspend_count: 0,
            priority: 8,
            base_priority: 8,
            stack_block: 0,
            stack_base: 0,
            stack_limit: 0,
            kthread: BOOT_KTHREAD,
            slot: None,
            tls: 0,
            exception_list: u32::MAX,
            resume: 0,
            wait: None,
            object,
            dpc: 0,
            follow_up: None,
            park: super::thread::Park::Start,
            start: (0, 0),
        };
        Kernel {
            memory,
            heap,
            low_heap,
            objects,
            threads: Threads::new(boot, KTHREAD_SLOTS),
            current: 1,
            clock: super::time::Clock::new(),
            dpcs: HashMap::new(),
            fs: super::fs::FileSystem::new(),
            eeprom: super::handlers::default_eeprom(),
            cache_pages: None,
            tls_template: Vec::new(),
            tls_zero_fill: 0,
            sections: Vec::new(),
            unstarted: Vec::new(),
            calls: 0,
            av_saved_data: 0,
            display_mode: [0; 6],
            trace: Trace::Off,
            reported: HashSet::new(),
            interrupts: Vec::new(),
            section_waiters: HashMap::new(),
            hooks: super::hooks::Hooks::default(),
            sound: super::sound::Sound::default(),
            gpu: crate::gpu::Gpu::new(),
            next_vblank: super::time::BOOT_CLOCK + super::time::vblank_period(),
            fps_report: None,
            frame_sink: None,
            sync_irps: HashMap::new(),
        }
    }

    /// Creates `n` more ready threads and returns their ids.
    ///
    /// # Panics
    ///
    /// Panics when a thread cannot be made.
    pub fn threads(k: &mut Kernel, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| {
                k.create_thread(0x1000, &[], super::spawn::Spawn::default())
                    .expect("a thread")
            })
            .collect()
    }

    /// Makes `tid` the running thread (without the KPCR switch's side effects mattering).
    pub fn run_as(k: &mut Kernel, tid: u32) {
        k.switch_to(tid);
    }
}
