//! NT system services on objects by handle: queries of mutants, semaphores and timers; timers,
//! I/O completion ports, object directories and symbolic link objects; signal-and-wait.
//!
//! Copied from the author's original anybox-tk code (its `ntobjects.rs` and `io.rs`), with the
//! author's permission, and adapted to this runtime's kernel.

use std::collections::{HashMap, VecDeque};

use super::ke::{release_mutant_object, release_semaphore_object};
use super::nt::{attributes_name, create_named};
use super::value;
use crate::kernel::object::{Body, ObjId, Timer, WaitOn};
use crate::kernel::time::TICKS_PER_MS;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `NtQueryMutant(MutantHandle, *MutantInformation)`: the count (1 when free), whether the caller
/// owns it, and whether it was abandoned.
pub fn query_mutant(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(Body::Mutant { owner, count }) = k
        .object_of_handle(c.arg(0))
        .map(|id| &k.objects.get(id).body)
    else {
        return value(status::INVALID_HANDLE);
    };
    let (owned, signal) = (*owner == k.current, 1u32.wrapping_sub(*count));
    let info = c.arg(1);
    k.w32(info, signal)?;
    k.w8(info + 4, u8::from(owned))?;
    k.w8(info + 5, 0)?;
    value(status::SUCCESS)
}

/// `NtQuerySemaphore(SemaphoreHandle, *SemaphoreInformation)`: the count and the limit.
pub fn query_semaphore(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(&Body::Semaphore { count, limit }) = k
        .object_of_handle(c.arg(0))
        .map(|id| &k.objects.get(id).body)
    else {
        return value(status::INVALID_HANDLE);
    };
    k.w32(c.arg(1), count.cast_unsigned())?;
    k.w32(c.arg(1) + 4, limit.cast_unsigned())?;
    value(status::SUCCESS)
}

/// `NtCreateTimer(*TimerHandle, ObjectAttributes, TimerType)`: a notification timer (0) stays
/// signaled, a synchronization timer (1) releases one waiter.
pub fn create_timer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let timer = Timer {
        sync: c.arg(2) == 1,
        ..Timer::default()
    };
    let same = |b: &Body| matches!(b, Body::Timer(_));
    create_named(k, c.arg(0), c.arg(1), same, Body::Timer(timer))
}

/// The timer behind `handle`.
fn timer_of(k: &mut Kernel, handle: u32) -> Option<(ObjId, &mut Timer)> {
    let id = k.object_of_handle(handle)?;
    match &mut k.objects.get_mut(id).body {
        Body::Timer(timer) => Some((id, timer)),
        _ => None,
    }
}

/// `NtSetTimerEx(TimerHandle, *DueTime, TimerApcRoutine, ApcMode, TimerContext, ResumeTimer,
/// Period, *PreviousState)`: arms the timer, re-armed every `Period` milliseconds, with the APC
/// routine run at each expiry.
pub fn set_timer_ex(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let due = k.clock.due(k.r64(c.arg(1))?.cast_signed());
    let (routine, context, period, previous_out) = (c.arg(2), c.arg(4), c.arg(6), c.arg(7));
    let Some((id, timer)) = timer_of(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let previous = timer.signaled;
    timer.due = Some(due);
    timer.signaled = false;
    timer.period = u64::from(period) * TICKS_PER_MS;
    timer.apc = (routine != 0).then_some((routine, context));
    k.mirror(id);
    if previous_out != 0 {
        k.w8(previous_out, u8::from(previous))?;
    }
    value(status::SUCCESS)
}

/// `NtCancelTimer(TimerHandle, *CurrentState)`.
pub fn cancel_timer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let state_out = c.arg(1);
    let Some((_, timer)) = timer_of(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    timer.due = None;
    let signaled = timer.signaled;
    if state_out != 0 {
        k.w8(state_out, u8::from(signaled))?;
    }
    value(status::SUCCESS)
}

/// `NtQueryTimer(TimerHandle, *TimerInformation)`: the time left until it expires, and whether
/// it is signaled.
pub fn query_timer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (now, info) = (k.clock.now(), c.arg(1));
    let Some((_, timer)) = timer_of(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let (left, signaled) = (
        timer.due.map_or(0, |due| due.saturating_sub(now)),
        timer.signaled,
    );
    k.w64(info, left)?;
    k.w8(info + 8, u8::from(signaled))?;
    value(status::SUCCESS)
}

/// `NtCreateIoCompletion(*IoCompletionHandle, DesiredAccess, ObjectAttributes, Count)`.
pub fn create_io_completion(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let same = |b: &Body| matches!(b, Body::IoCompletion { .. });
    let port = Body::IoCompletion {
        packets: VecDeque::new(),
        receivers: HashMap::new(),
    };
    create_named(k, c.arg(0), c.arg(2), same, port)
}

/// Writes a packet (key, context, status, information) where a thread taking one wants it: key,
/// context, then the status block.
fn deliver(k: &Kernel, [key, context, iosb]: [u32; 3], packet: [u32; 4]) -> KResult<()> {
    k.w32(key, packet[0])?;
    k.w32(context, packet[1])?;
    k.w32(iosb, packet[2])?;
    k.w32(iosb + 4, packet[3])
}

/// Queues `packet` on completion port `id`, or hands it to the oldest waiting thread; returns
/// the status.
///
/// # Errors
///
/// A fault when the receiving thread's buffers cannot be written.
pub(super) fn set_io_completion_object(
    k: &mut Kernel,
    id: ObjId,
    packet: [u32; 4],
) -> KResult<u32> {
    for tid in k.takers(id) {
        let receiver = match &mut k.objects.get_mut(id).body {
            Body::IoCompletion { receivers, .. } => receivers.remove(&tid),
            _ => return Ok(status::OBJECT_TYPE_MISMATCH),
        };
        if let Some(receiver) = receiver {
            deliver(k, receiver, packet)?;
            k.wake(tid, status::SUCCESS);
            return Ok(status::SUCCESS);
        }
    }
    match &mut k.objects.get_mut(id).body {
        Body::IoCompletion { packets, .. } => packets.push_back(packet),
        _ => return Ok(status::OBJECT_TYPE_MISMATCH),
    }
    k.wake_satisfiable();
    Ok(status::SUCCESS)
}

/// `NtSetIoCompletion(IoCompletionHandle, KeyContext, ApcContext, IoStatus,
/// IoStatusInformation)`: queues a packet, or hands it to the oldest waiting thread.
pub fn set_io_completion(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_of_handle(c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let packet = [c.arg(1), c.arg(2), c.arg(3), c.arg(4)];
    value(set_io_completion_object(k, id, packet)?)
}

/// `NtRemoveIoCompletion(IoCompletionHandle, *KeyContext, *ApcContext, *IoStatusBlock,
/// *Timeout)`: takes the oldest packet, waiting for one if there is none.
pub fn remove_io_completion(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_of_handle(c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let to = [c.arg(1), c.arg(2), c.arg(3)];
    let packet = match &mut k.objects.get_mut(id).body {
        Body::IoCompletion { packets, .. } => packets.pop_front(),
        _ => return value(status::OBJECT_TYPE_MISMATCH),
    };
    if let Some(packet) = packet {
        deliver(k, to, packet)?;
        return value(status::SUCCESS);
    }
    let deadline = k.deadline(c.arg(4))?;
    if deadline.is_some_and(|d| d <= k.clock.now()) {
        return value(status::TIMEOUT);
    }
    let me = k.current;
    if let Body::IoCompletion { receivers, .. } = &mut k.objects.get_mut(id).body {
        receivers.insert(me, to);
    }
    k.block(WaitOn::Take(id), deadline);
    Ok(CallResult::Block)
}

/// `NtQueryIoCompletion(IoCompletionHandle, *IoCompletionInformation)`: the packets waiting.
pub fn query_io_completion(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let depth = match k
        .object_of_handle(c.arg(0))
        .map(|id| &k.objects.get(id).body)
    {
        Some(Body::IoCompletion { packets, .. }) => u32::try_from(packets.len()).unwrap_or(0),
        _ => return value(status::INVALID_HANDLE),
    };
    k.w32(c.arg(1), depth)?;
    value(status::SUCCESS)
}

/// The directories every console has, which open without being created.
const STANDARD_DIRECTORIES: [&str; 3] = ["\\", "\\??", "\\device"];

/// `NtCreateDirectoryObject(*DirectoryHandle, ObjectAttributes)`.
pub fn create_directory_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let name = attributes_name(k, c.arg(1))?.unwrap_or_default();
    let same = |b: &Body| matches!(b, Body::Directory(_));
    create_named(k, c.arg(0), c.arg(1), same, Body::Directory(name))
}

/// `NtOpenDirectoryObject(*DirectoryHandle, ObjectAttributes)`: a directory created earlier, or
/// one of the standard ones.
pub fn open_directory_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(name) = attributes_name(k, c.arg(1))? else {
        return value(status::INVALID_PARAMETER);
    };
    let id = match k.objects.named(&name) {
        Some(id) => id,
        None if STANDARD_DIRECTORIES.contains(&name.to_ascii_lowercase().as_str()) => {
            let id = k.objects.add(Body::Directory(name.clone()));
            k.objects.set_name(id, &name);
            id
        }
        None => return value(status::OBJECT_NAME_NOT_FOUND),
    };
    let handle = k.objects.open(id);
    k.w32(c.arg(0), handle)?;
    value(status::SUCCESS)
}

/// The names in directory `dir`: drive letters in `\??`, devices in `\Device`, and objects
/// named inside it.
fn directory_entries(k: &Kernel, dir: &str) -> Vec<String> {
    let mut names = k.objects.names_in(dir);
    let lower = dir.to_ascii_lowercase();
    if lower == "\\??" {
        names.extend(k.fs.link_names().filter_map(|name| {
            let (parent, last) = name.rsplit_once('\\')?;
            (parent == "\\??").then(|| last.to_owned())
        }));
    }
    if lower == "\\device" {
        names.extend(["CdRom0".to_owned(), "Harddisk0".to_owned()]);
    }
    names.sort_by_key(|name| name.to_ascii_lowercase());
    names.dedup_by_key(|name| name.to_ascii_lowercase());
    names
}

/// `NtQueryDirectoryObject(DirectoryHandle, Buffer, Length, RestartScan, *Context,
/// *ReturnedLength)`: one entry per call, from where `Context` says the last call stopped. An
/// entry is the name as an 8-bit counted string, then the name's text.
pub fn query_directory_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (buffer, length, restart, context, returned) =
        (c.arg(1), c.arg(2), c.arg(3) & 0xFF != 0, c.arg(4), c.arg(5));
    let Some(Body::Directory(name)) = k
        .object_of_handle(c.arg(0))
        .map(|id| &k.objects.get(id).body)
    else {
        return value(status::INVALID_HANDLE);
    };
    let entries = directory_entries(k, name);
    let index = if restart { 0 } else { k.r32(context)? };
    let Some(entry) = entries.get(index as usize) else {
        return value(status::NO_MORE_ENTRIES);
    };
    let text = entry.as_bytes();
    let len = u16::try_from(text.len()).unwrap_or(u16::MAX);
    let needed = 8 + u32::from(len) + 1;
    k.w32_opt(returned, needed)?;
    if length < needed {
        return value(status::BUFFER_TOO_SMALL);
    }
    k.w16(buffer, len)?;
    k.w16(buffer + 2, len + 1)?;
    k.w32(buffer + 4, buffer + 8)?;
    k.write(buffer + 8, &text[..usize::from(len)])?;
    k.w8(buffer + 8 + u32::from(len), 0)?;
    k.w32(context, index + 1)?;
    value(status::SUCCESS)
}

/// `NtOpenSymbolicLinkObject(*LinkHandle, ObjectAttributes)`.
pub fn open_symbolic_link_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(name) = attributes_name(k, c.arg(1))? else {
        return value(status::INVALID_PARAMETER);
    };
    let Some(target) = k.fs.link_target(&name).map(str::to_owned) else {
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    let id = k.objects.add(Body::SymbolicLink(target));
    let handle = k.objects.open(id);
    k.w32(c.arg(0), handle)?;
    value(status::SUCCESS)
}

/// `NtQuerySymbolicLinkObject(LinkHandle, *LinkTarget, *ReturnedLength)`: the target into an
/// `ANSI_STRING`.
pub fn query_symbolic_link_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(Body::SymbolicLink(target)) =
        k.objects.handle(c.arg(0)).map(|id| &k.objects.get(id).body)
    else {
        return value(status::INVALID_HANDLE);
    };
    let bytes: Vec<u8> = target.chars().map(|ch| ch as u8).collect();
    let (out, length_out) = (c.arg(1), c.arg(2));
    let capacity = u32::from(k.r16(out + 2)?);
    let buffer = k.r32(out + 4)?;
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    k.w32_opt(length_out, len)?;
    if len > capacity {
        return value(status::BUFFER_TOO_SMALL);
    }
    k.write(buffer, &bytes)?;
    k.w16(out, len as u16)?;
    value(status::SUCCESS)
}

/// `NtSignalAndWaitForSingleObjectEx(SignalHandle, WaitHandle, WaitMode, Alertable, *Timeout)`:
/// sets an event, releases a semaphore once or a mutant, then waits.
pub fn signal_and_wait_for_single_object_ex(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (Some(signal), Some(wait)) = (k.object_of_handle(c.arg(0)), k.object_of_handle(c.arg(1)))
    else {
        return value(status::INVALID_HANDLE);
    };
    match k.objects.get(signal).body {
        Body::Event { .. } => {
            k.set_event(signal);
        }
        Body::Semaphore { .. } => {
            if release_semaphore_object(k, signal, 1).is_none() {
                return value(status::SEMAPHORE_LIMIT_EXCEEDED);
            }
        }
        Body::Mutant { .. } => {
            if let Err(code) = release_mutant_object(k, signal) {
                return value(code);
            }
        }
        _ => return value(status::OBJECT_TYPE_MISMATCH),
    }
    k.wait_for(vec![wait], false, c.arg(4))
}
