//! Creating, switching and ending guest threads; DPCs and routines run on threads of their own
//! (spec 7.5, 8.5).

use super::object::Body;
use super::thread::{FollowUp, Park, State, Thread};
use super::{Kernel, THREAD_RETURN};
use crate::boot::{KPCR, KTHREAD_BLOCKS, KTHREAD_SIZE};

/// The default stack of a new thread.
pub const DEFAULT_STACK: u32 = 64 * 1024;
/// The stack of a DPC or APC thread.
pub const SMALL_STACK: u32 = 16 * 1024;
/// The FPU save area at the top of every stack; the TIB stack base sits below it.
pub const FPU_AREA: u32 = 0x210;
/// More live threads than this stops the process (spec 7.1).
pub const MAX_THREADS: usize = 256;

/// How a new thread is made.
#[derive(Debug, Clone, Copy, Default)]
pub struct Spawn {
    /// The requested stack size; 0 is the default.
    pub stack: u32,
    /// The TLS size the creator asked for, if it asked for TLS.
    pub tls: Option<u32>,
    pub suspended: bool,
    /// The DPC this thread runs, or 0.
    pub dpc: u32,
}

impl Kernel {
    /// Creates a guest thread that runs `routine` with the stdcall `args`. Its host thread is
    /// started by the ABI layer. Returns the thread id, or `None` without a stack or a slot.
    ///
    /// # Panics
    ///
    /// Stops the process when more than 256 guest threads are live (a thread leak must not take
    /// the host down).
    pub fn create_thread(&mut self, routine: u32, args: &[u32], spawn: Spawn) -> Option<u32> {
        if self.threads.live() >= MAX_THREADS {
            eprintln!("more than {MAX_THREADS} live guest threads: stopping");
            std::process::exit(1);
        }
        let default = if spawn.dpc == 0 {
            DEFAULT_STACK
        } else {
            SMALL_STACK
        };
        let size = if spawn.stack == 0 {
            default
        } else {
            spawn.stack
        }
        .next_multiple_of(0x1000);
        let stack = self.heap.alloc_aligned(size, 0x1000);
        if stack == 0 {
            return None;
        }
        let Some(slot) = self.threads.take_slot() else {
            self.heap.free(stack);
            return None;
        };
        let kthread = KTHREAD_BLOCKS + slot * KTHREAD_SIZE;
        let stack_base = stack + size - FPU_AREA;
        // The TLS block sits directly below the stack base: its first dword points to the next
        // one, then the template, then zeros.
        let tls = spawn.tls.map_or(0, |tls_size| {
            let template = u32::try_from(self.tls_template.len()).unwrap_or(0);
            let len = tls_size
                .next_multiple_of(4)
                .max(template.next_multiple_of(4) + 4);
            let block = stack_base - len;
            let _ = self.w32(block, block + 4);
            let _ = self.write(block + 4, &self.tls_template.clone());
            block
        });
        let _ = self.write(kthread, &[0; KTHREAD_SIZE as usize]);
        let _ = self.w32(kthread + 0x28, tls);
        // The initial stack: the thread-return trap, then the routine's arguments.
        let top = if tls == 0 { stack_base } else { tls };
        let esp = top - 4 * (u32::try_from(args.len()).unwrap_or(0) + 1);
        let _ = self.w32(esp, THREAD_RETURN);
        for (i, &arg) in args.iter().enumerate() {
            let _ = self.w32(esp + 4 + 4 * u32::try_from(i).unwrap_or(0), arg);
        }
        let id = self.threads.take_id();
        let object = self.objects.add_at(kthread, Body::Thread { tid: id });
        self.threads.insert(Thread {
            id,
            state: State::Ready,
            suspend_count: u32::from(spawn.suspended),
            priority: 8,
            base_priority: 8,
            stack_block: stack,
            stack_base,
            stack_limit: stack,
            kthread,
            slot: Some(slot),
            tls,
            exception_list: u32::MAX,
            resume: 0,
            wait: None,
            object,
            dpc: spawn.dpc,
            follow_up: None,
            park: Park::Start,
            start: (routine, esp),
        });
        self.unstarted.push(id);
        Some(id)
    }

    /// Runs `routine(args)` on a thread of its own with a small stack.
    pub fn run_routine(&mut self, routine: u32, args: &[u32]) -> Option<u32> {
        self.create_thread(
            routine,
            args,
            Spawn {
                stack: SMALL_STACK,
                ..Spawn::default()
            },
        )
    }

    /// Queues a DPC: runs `routine(dpc, context, a1, a2)` on a new thread marked as a DPC thread.
    /// False if the DPC is unknown, already queued, or no thread could be made.
    pub fn queue_dpc(&mut self, dpc: u32, a1: u32, a2: u32) -> bool {
        let Some(&record) = self.dpcs.get(&dpc) else {
            return false;
        };
        if record.queued {
            return false;
        }
        let spawn = Spawn {
            stack: SMALL_STACK,
            dpc,
            ..Spawn::default()
        };
        if self
            .create_thread(record.routine, &[dpc, record.context, a1, a2], spawn)
            .is_none()
        {
            return false;
        }
        if let Some(record) = self.dpcs.get_mut(&dpc) {
            record.queued = true;
        }
        true
    }

    /// Ends thread `tid`: frees its stack and slot, clears its DPC's queued flag, signals its
    /// object, then applies its follow-up. A second end does nothing.
    pub fn end_thread(&mut self, tid: u32, result: Option<u32>) {
        let Some(thread) = self.threads.get_mut(tid) else {
            return;
        };
        if thread.state == State::Done {
            return;
        }
        thread.state = State::Done;
        thread.wait = None;
        let (stack, slot, dpc, follow_up) = (
            thread.stack_block,
            thread.slot,
            thread.dpc,
            thread.follow_up,
        );
        if stack != 0 {
            self.heap.free(stack);
        }
        if let Some(slot) = slot {
            self.threads.free_slot(slot);
        }
        if let Some(record) = self.dpcs.get_mut(&dpc) {
            record.queued = false;
        }
        match follow_up {
            Some(FollowUp::WakeWithResult { tid, low_byte }) => {
                let value = match result {
                    Some(eax) if low_byte => eax & 0xFF,
                    Some(eax) => eax,
                    None => 1,
                };
                self.wake(tid, value);
            }
            Some(FollowUp::IrpCompletion { irp }) => {
                let _ = super::handlers::irp::resume_completion(self, irp, result);
            }
            None => {}
        }
        self.wake_satisfiable();
    }

    /// Makes `next` the running thread: saves the outgoing thread's exception list from the KPCR
    /// and loads the incoming one's exception list, stack base, stack limit and KTHREAD.
    pub fn switch_to(&mut self, next: u32) {
        if next == self.current {
            return;
        }
        let outgoing = self.r32(KPCR).unwrap_or(u32::MAX);
        if let Some(thread) = self.threads.get_mut(self.current) {
            thread.exception_list = outgoing;
        }
        if let Some(thread) = self.threads.get(next) {
            let _ = self.w32(KPCR, thread.exception_list);
            let _ = self.w32(KPCR + 0x04, thread.stack_base);
            let _ = self.w32(KPCR + 0x08, thread.stack_limit);
            let _ = self.w32(KPCR + 0x28, thread.kthread);
        }
        self.current = next;
    }
}
