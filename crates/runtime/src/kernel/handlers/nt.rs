//! NT object services: handles, events, waits, semaphores, mutants, threads, system time (spec
//! 8.3, 8.7).

use super::ke::{release_mutant_object, release_semaphore_object, resume, suspend};
use super::value;
use crate::kernel::object::{Body, ObjId};
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// The full path an `OBJECT_ATTRIBUTES` names (`+0` root directory handle, `+4` `ANSI_STRING*`),
/// joined to the root file's path when relative; `None` without a name.
///
/// # Errors
///
/// A fault when the structure or the name cannot be read.
pub fn attributes_name(k: &Kernel, attributes: u32) -> KResult<Option<String>> {
    if attributes == 0 {
        return Ok(None);
    }
    let name_ptr = k.r32(attributes + 4)?;
    if name_ptr == 0 {
        return Ok(None);
    }
    let name = k.ansi_string(name_ptr)?;
    let root = k.r32(attributes)?;
    if root == 0 {
        return Ok(Some(name));
    }
    let base = k
        .objects
        .handle(root)
        .and_then(|id| match &k.objects.get(id).body {
            Body::File(file) => Some(file.path.clone()),
            Body::Directory(path) => Some(path.clone()),
            _ => None,
        });
    Ok(Some(match base {
        Some(base) => format!("{}\\{name}", base.trim_end_matches('\\')),
        None => name,
    }))
}

/// Creates an object, or opens the existing one of the same kind with `OBJECT_NAME_EXISTS`.
/// Writes the handle.
pub(super) fn create_named(
    k: &mut Kernel,
    handle_out: u32,
    attributes: u32,
    same_kind: fn(&Body) -> bool,
    body: Body,
) -> KResult<CallResult> {
    let name = attributes_name(k, attributes)?;
    if let Some(name) = &name
        && let Some(existing) = k.objects.named(name)
    {
        if !same_kind(&k.objects.get(existing).body) {
            return value(status::OBJECT_TYPE_MISMATCH);
        }
        let handle = k.objects.open(existing);
        k.w32(handle_out, handle)?;
        return value(status::OBJECT_NAME_EXISTS);
    }
    let id = k.objects.add(body);
    if let Some(name) = &name {
        k.objects.set_name(id, name);
    }
    let handle = k.objects.open(id);
    k.w32(handle_out, handle)?;
    value(status::SUCCESS)
}

/// `NtClose(h)`: persists the saves if they changed.
pub fn close(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    match k.objects.close(c.arg(0)) {
        Some(id) => {
            if matches!(k.objects.get(id).body, Body::File(_)) {
                k.fs.persist();
            }
            value(status::SUCCESS)
        }
        None => value(status::INVALID_HANDLE),
    }
}

/// `NtCreateEvent(*h, attr, type, state)`: the name is ignored.
pub fn create_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let id = k.objects.add(Body::Event {
        sync: c.arg(2) == 1,
        signaled: c.arg(3) != 0,
    });
    let handle = k.objects.open(id);
    k.w32(c.arg(0), handle)?;
    value(status::SUCCESS)
}

/// The event a handle names.
fn event(k: &Kernel, handle: u32) -> Option<ObjId> {
    k.object_of_handle(handle)
        .filter(|&id| matches!(k.objects.get(id).body, Body::Event { .. }))
}

/// `NtSetEvent(h, *prev)`.
pub fn set_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = event(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let previous = k.set_event(id);
    k.w32_opt(c.arg(1), previous)?;
    value(status::SUCCESS)
}

pub fn clear_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = event(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    k.reset_event(id);
    value(status::SUCCESS)
}

/// `NtPulseEvent(h, *prev)`.
pub fn pulse_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = event(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let previous = k.set_event(id);
    k.reset_event(id);
    k.w32_opt(c.arg(1), previous)?;
    value(status::SUCCESS)
}

/// `NtQueryEvent(h, *info)`: type and state.
pub fn query_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = event(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let Body::Event { sync, signaled } = k.objects.get(id).body else {
        return value(status::INVALID_HANDLE);
    };
    k.w32(c.arg(1), u32::from(sync))?;
    k.w32(c.arg(1) + 4, u32::from(signaled))?;
    value(status::SUCCESS)
}

/// Waits on the objects of `handles`.
fn wait_handles(k: &mut Kernel, handles: &[u32], all: bool, timeout: u32) -> KResult<CallResult> {
    let mut ids = Vec::with_capacity(handles.len());
    for &handle in handles {
        let Some(id) = k.object_of_handle(handle) else {
            return value(status::INVALID_HANDLE);
        };
        ids.push(id);
    }
    k.wait_for(ids, all, timeout)
}

/// `NtWaitForSingleObject(h, alertable, *timeout)`.
pub fn wait_for_single_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    wait_handles(k, &[c.arg(0)], false, c.arg(2))
}

/// `NtWaitForSingleObjectEx(h, mode, alertable, *timeout)`.
pub fn wait_for_single_object_ex(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    wait_handles(k, &[c.arg(0)], false, c.arg(3))
}

/// `NtWaitForMultipleObjectsEx(n, h[], type, mode, alertable, *timeout)`; type 0 is wait-all.
pub fn wait_for_multiple_objects_ex(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mut handles = Vec::new();
    for i in 0..c.arg(0) {
        handles.push(k.r32(c.arg(1) + 4 * i)?);
    }
    wait_handles(k, &handles, c.arg(2) == 0, c.arg(5))
}

/// `NtCreateSemaphore(*h, attr, count, max)`.
pub fn create_semaphore(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (count, limit) = (c.arg(2).cast_signed(), c.arg(3).cast_signed());
    if limit <= 0 || count < 0 || count > limit {
        return value(status::INVALID_PARAMETER);
    }
    let same = |b: &Body| matches!(b, Body::Semaphore { .. });
    create_named(
        k,
        c.arg(0),
        c.arg(1),
        same,
        Body::Semaphore { count, limit },
    )
}

/// `NtReleaseSemaphore(h, n, *prev)`.
pub fn release_semaphore(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_of_handle(c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let n = c.arg(1).cast_signed();
    if n <= 0 {
        return value(status::INVALID_PARAMETER);
    }
    match release_semaphore_object(k, id, n) {
        Some(previous) => {
            k.w32_opt(c.arg(2), previous)?;
            value(status::SUCCESS)
        }
        None => value(status::SEMAPHORE_LIMIT_EXCEEDED),
    }
}

/// `NtCreateMutant(*h, attr, owned)`.
pub fn create_mutant(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (owner, count) = if c.arg(2) != 0 {
        (k.current, 1)
    } else {
        (0, 0)
    };
    let same = |b: &Body| matches!(b, Body::Mutant { .. });
    create_named(k, c.arg(0), c.arg(1), same, Body::Mutant { owner, count })
}

/// `NtReleaseMutant(h, *prev)`.
pub fn release_mutant(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_of_handle(c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    match release_mutant_object(k, id) {
        Ok(previous) => {
            k.w32_opt(c.arg(1), previous)?;
            value(status::SUCCESS)
        }
        Err(code) => value(code),
    }
}

/// `NtDuplicateObject(src, *dst, options)`: option 1 closes the source.
pub fn duplicate_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_of_handle(c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let handle = k.objects.open(id);
    k.w32(c.arg(1), handle)?;
    if c.arg(2) & 1 != 0 {
        k.objects.close(c.arg(0));
    }
    value(status::SUCCESS)
}

pub fn yield_execution(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    Ok(CallResult::Yield)
}

/// The thread a handle names.
pub(super) fn thread_of_handle(k: &Kernel, handle: u32) -> Option<u32> {
    let id = k.object_of_handle(handle)?;
    match k.objects.get(id).body {
        Body::Thread { tid } => Some(tid),
        _ => None,
    }
}

/// `NtSuspendThread(h, *prev)`.
pub fn suspend_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(tid) = thread_of_handle(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let previous = k.threads.get(tid).map_or(0, |t| t.suspend_count);
    k.w32_opt(c.arg(1), previous)?;
    let result = suspend(k, tid);
    if result == CallResult::Block {
        if let Some(thread) = k.threads.get_mut(tid) {
            thread.resume = status::SUCCESS;
        }
        return Ok(result);
    }
    value(status::SUCCESS)
}

/// `NtResumeThread(h, *prev)`.
pub fn resume_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(tid) = thread_of_handle(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let previous = resume(k, tid);
    k.w32_opt(c.arg(1), previous)?;
    value(status::SUCCESS)
}

/// `NtSetSystemTime(*new, *prev)`: moves only the offset of system time.
pub fn set_system_time(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if c.arg(0) == 0 {
        return value(status::INVALID_PARAMETER);
    }
    let new = k.r64(c.arg(0))?;
    let high = (new >> 32) as u32;
    if high == 0 || high > 0x2000_0000 {
        return value(status::INVALID_PARAMETER);
    }
    if c.arg(1) != 0 {
        k.w64(c.arg(1), k.clock.system_time())?;
    }
    k.clock.offset = new.cast_signed() - k.clock.now().cast_signed();
    value(status::SUCCESS)
}

/// `ObReferenceObjectByHandle(h, type, *obj)`: the object's guest pointer, a 16-byte zeroed block
/// made on demand for an object with none.
pub fn reference_object_by_handle(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_of_handle(c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let Some(address) = object_pointer(k, id) else {
        return value(status::NO_MEMORY);
    };
    k.w32(c.arg(2), address)?;
    value(status::SUCCESS)
}

/// An object's guest pointer, binding a 16-byte zeroed block made on demand to an object with
/// none; `None` when out of memory.
pub(super) fn object_pointer(k: &mut Kernel, id: ObjId) -> Option<u32> {
    let address = k.objects.get(id).address;
    if address != 0 {
        return Some(address);
    }
    let address = k.heap.alloc(16);
    if address == 0 {
        return None;
    }
    k.objects.bind(address, id);
    Some(address)
}

pub fn dereference_object(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(1)
}
