//! Asynchronous procedure calls: `KAPC` objects, APCs queued to a thread, and the I/O completion
//! APC dispatcher.
//!
//! Copied from the author's original anybox-tk code (its `threads.rs` and `ntobjects.rs`), with
//! the author's permission, and adapted to this runtime's kernel.
//!
//! This runtime does not deliver APCs to the thread they are queued to: an APC routine runs at
//! once on a thread of its own, as DPCs do.

use super::nt::thread_of_handle;
use super::value;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `KAPC` offsets.
const APC_TYPE: u32 = 0x00;
const APC_MODE: u32 = 0x02;
const APC_INSERTED: u32 = 0x03;
const APC_THREAD: u32 = 0x04;
const APC_KERNEL_ROUTINE: u32 = 0x10;
const APC_RUNDOWN_ROUTINE: u32 = 0x14;
const APC_NORMAL_ROUTINE: u32 = 0x18;
const APC_NORMAL_CONTEXT: u32 = 0x1C;
const APC_ARGUMENT1: u32 = 0x20;
const APC_ARGUMENT2: u32 = 0x24;
/// The `KAPC` size, and its type.
const APC_SIZE: usize = 0x28;
const APC_OBJECT: u16 = 0x12;

/// `KeInitializeApc(Apc, Thread, KernelRoutine, RundownRoutine, NormalRoutine, ApcMode,
/// NormalContext)`.
pub fn initialize_apc(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (apc, normal) = (c.arg(0), c.arg(4));
    k.write(apc, &[0; APC_SIZE])?;
    k.w16(apc + APC_TYPE, APC_OBJECT)?;
    // Without a normal routine the APC is a kernel one, without a context.
    let (mode, context) = if normal == 0 {
        (0, 0)
    } else {
        (c.arg(5) as u8, c.arg(6))
    };
    k.w8(apc + APC_MODE, mode)?;
    k.w32(apc + APC_THREAD, c.arg(1))?;
    k.w32(apc + APC_KERNEL_ROUTINE, c.arg(2))?;
    k.w32(apc + APC_RUNDOWN_ROUTINE, c.arg(3))?;
    k.w32(apc + APC_NORMAL_ROUTINE, normal)?;
    k.w32(apc + APC_NORMAL_CONTEXT, context)?;
    value(0)
}

/// `KeInsertQueueApc(Apc, SystemArgument1, SystemArgument2, Increment)`: runs the APC's kernel
/// routine, which may change what the normal routine gets, or, for an APC without one, its
/// normal routine. Returns whether the APC was queued: not when it already is.
pub fn insert_queue_apc(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let apc = c.arg(0);
    if k.r8(apc + APC_INSERTED)? != 0 {
        return value(0);
    }
    k.w32(apc + APC_ARGUMENT1, c.arg(1))?;
    k.w32(apc + APC_ARGUMENT2, c.arg(2))?;
    let kernel = k.r32(apc + APC_KERNEL_ROUTINE)?;
    let normal = k.r32(apc + APC_NORMAL_ROUTINE)?;
    let started = if kernel != 0 {
        // KernelRoutine(Apc, &NormalRoutine, &NormalContext, &SystemArgument1,
        // &SystemArgument2): the fields in the APC itself.
        k.run_routine(
            kernel,
            &[
                apc,
                apc + APC_NORMAL_ROUTINE,
                apc + APC_NORMAL_CONTEXT,
                apc + APC_ARGUMENT1,
                apc + APC_ARGUMENT2,
            ],
        )
    } else if normal != 0 {
        let context = k.r32(apc + APC_NORMAL_CONTEXT)?;
        k.run_routine(normal, &[context, c.arg(1), c.arg(2)])
    } else {
        None
    };
    k.w8(apc + APC_INSERTED, u8::from(started.is_some()))?;
    value(u32::from(started.is_some()))
}

/// `NtQueueApcThread(ThreadHandle, ApcRoutine, ApcArgument1, ApcArgument2, ApcArgument3)`.
pub fn queue_apc_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if thread_of_handle(k, c.arg(0)).is_none() {
        return value(status::INVALID_HANDLE);
    }
    let started = k.run_routine(c.arg(1), &[c.arg(2), c.arg(3), c.arg(4)]);
    value(if started.is_some() {
        status::SUCCESS
    } else {
        status::NO_MEMORY
    })
}

/// `NtUserIoApcDispatcher(ApcContext, IoStatusBlock, Reserved)`: calls the completion routine
/// in `ApcContext` with the error, the bytes transferred and the status block (the
/// `OVERLAPPED`), and returns when it has run.
pub fn user_io_apc_dispatcher(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (routine, iosb) = (c.arg(0), c.arg(1));
    let result = k.r32(iosb)?;
    let (error, transferred) = if result & 0x8000_0000 == 0 {
        (0, k.r32(iosb + 4)?)
    } else {
        (status::to_dos_error(result), 0)
    };
    let Some(tid) = k.run_routine(routine, &[error, transferred, iosb]) else {
        return value(0);
    };
    let Some(object) = k.threads.get(tid).map(|t| t.object) else {
        return value(0);
    };
    k.wait_for(vec![object], false, 0)
}
