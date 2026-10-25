//! Kernel exports: events, timers, DPCs, interrupts, waits, mutants, semaphores, IRQL, time and
//! threads (spec 8.5, 8.6, 8.8).

use super::value;
use crate::boot::KPCR;
use crate::kernel::object::{Body, ObjId, Timer, WaitOn};
use crate::kernel::spawn::Spawn;
use crate::kernel::thread::FollowUp;
use crate::kernel::time::BOOT_CLOCK;
use crate::kernel::{Call, CallResult, Dpc, KResult, Kernel, status};

/// The performance counter's frequency.
const PERFORMANCE_FREQUENCY: u64 = 3_375_000;
/// The IRQL byte in the KPCR.
const IRQL: u32 = KPCR + 0x24;

/// Writes a dispatcher header: type, size in dwords, signal state, and an empty wait list.
fn write_header(k: &Kernel, at: u32, kind: u8, size: u8, signal: u32) -> KResult<()> {
    k.write(at, &[kind, 0, size, 0])?;
    k.w32(at + 4, signal)?;
    k.w32(at + 8, at + 8)?;
    k.w32(at + 12, at + 8)
}

/// `KeInitializeEvent(e, type, state)`.
pub fn initialize_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (at, sync, signaled) = (c.arg(0), c.arg(1) == 1, c.arg(2) != 0);
    write_header(k, at, u8::from(sync), 4, u32::from(signaled))?;
    k.objects.add_at(at, Body::Event { sync, signaled });
    value(0)
}

pub fn set_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_at(c.arg(0))? else {
        return value(0);
    };
    value(k.set_event(id))
}

pub fn reset_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_at(c.arg(0))? else {
        return value(0);
    };
    value(k.reset_event(id))
}

/// Sets, wakes who it can, clears; returns the previous state.
pub fn pulse_event(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_at(c.arg(0))? else {
        return value(0);
    };
    let previous = k.set_event(id);
    k.reset_event(id);
    value(previous)
}

pub fn set_event_boost_priority(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if let Some(id) = k.object_at(c.arg(0))? {
        k.set_event(id);
    }
    k.w32_opt(c.arg(1), 0)?;
    value(0)
}

/// `KeWaitForSingleObject(o, reason, mode, alertable, *timeout)`.
pub fn wait_for_single_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_at(c.arg(0))? else {
        return value(status::INVALID_PARAMETER);
    };
    k.wait_for(vec![id], false, c.arg(4))
}

/// `KeWaitForMultipleObjects(n, objs[], type, reason, mode, alertable, *timeout, blocks)`;
/// type 0 is wait-all.
pub fn wait_for_multiple_objects(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mut ids = Vec::new();
    for i in 0..c.arg(0) {
        let address = k.r32(c.arg(1) + 4 * i)?;
        let Some(id) = k.object_at(address)? else {
            return value(status::INVALID_PARAMETER);
        };
        ids.push(id);
    }
    k.wait_for(ids, c.arg(2) == 0, c.arg(6))
}

/// Waits for the clock alone until `deadline`; a deadline already passed yields.
fn delay_until(k: &mut Kernel, deadline: u64) -> CallResult {
    if deadline <= k.clock.now() {
        return CallResult::Yield;
    }
    k.block(WaitOn::Delay, Some(deadline));
    CallResult::Block
}

/// `KeDelayExecutionThread(mode, alertable, *interval)`.
pub fn delay_execution_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let interval = k.r64(c.arg(2))?.cast_signed();
    let deadline = k.clock.due(interval);
    Ok(delay_until(k, deadline))
}

/// `KeStallExecutionProcessor(us)`: others run meanwhile.
pub fn stall_execution_processor(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if c.arg(0) == 0 {
        return value(0);
    }
    let deadline = k.clock.now() + u64::from(c.arg(0)) * 10;
    Ok(delay_until(k, deadline))
}

/// The timer bound at `address`, made on first use.
fn timer_at(k: &mut Kernel, address: u32) -> ObjId {
    match k.objects.at(address) {
        Some(id) if matches!(k.objects.get(id).body, Body::Timer(_)) => id,
        _ => k.objects.add_at(address, Body::Timer(Timer::default())),
    }
}

/// `KeInitializeTimerEx(t, type)`: type 1 is a synchronization timer.
pub fn initialize_timer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let timer = Timer {
        sync: c.arg(1) == 1,
        ..Timer::default()
    };
    k.objects.add_at(c.arg(0), Body::Timer(timer));
    value(0)
}

/// Arms a timer at a guest due time; returns whether it was armed.
fn arm(k: &mut Kernel, address: u32, due: i64, period_ms: u32, dpc: u32) -> u32 {
    let due = k.clock.due(due);
    let id = timer_at(k, address);
    let Body::Timer(timer) = &mut k.objects.get_mut(id).body else {
        return 0;
    };
    let was = timer.due.is_some();
    timer.due = Some(due);
    timer.period = u64::from(period_ms) * crate::kernel::time::TICKS_PER_MS;
    timer.dpc = dpc;
    timer.signaled = false;
    k.mirror(id);
    u32::from(was)
}

/// `KeSetTimer(t, due_low, due_high, dpc)`.
pub fn set_timer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(arm(k, c.arg(0), c.arg64(1).cast_signed(), 0, c.arg(3)))
}

/// `KeSetTimerEx(t, due_low, due_high, period_ms, dpc)`.
pub fn set_timer_ex(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(arm(
        k,
        c.arg(0),
        c.arg64(1).cast_signed(),
        c.arg(3),
        c.arg(4),
    ))
}

pub fn cancel_timer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let id = timer_at(k, c.arg(0));
    let Body::Timer(timer) = &mut k.objects.get_mut(id).body else {
        return value(0);
    };
    value(u32::from(timer.due.take().is_some()))
}

/// `KeInitializeDpc(d, routine, context)`.
pub fn initialize_dpc(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.dpcs.insert(
        c.arg(0),
        Dpc {
            routine: c.arg(1),
            context: c.arg(2),
            queued: false,
        },
    );
    value(0)
}

pub fn insert_queue_dpc(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(u32::from(k.queue_dpc(c.arg(0), c.arg(1), c.arg(2))))
}

/// A queued DPC is already running.
pub fn remove_queue_dpc(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(0)
}

pub fn is_executing_dpc(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(u32::from(
        k.threads.get(k.current).is_some_and(|t| t.dpc != 0),
    ))
}

/// `KeInitializeInterrupt(k, routine, ctx, vector, irql, mode, share)`: writes the `KINTERRUPT`.
pub fn initialize_interrupt(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(0);
    k.w32(at, c.arg(1))?;
    k.w32(at + 4, c.arg(2))?;
    k.w32(at + 8, c.arg(3))?;
    k.w32(at + 0xC, c.arg(4))?;
    k.write(at + 0x10, &[0, c.arg(6) as u8, c.arg(5) as u8])?;
    value(0)
}

/// Sets the connected byte and registers the interrupt; returns whether it was newly connected.
pub fn connect_interrupt(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(0);
    if k.r8(at + 0x10)? != 0 {
        return value(0);
    }
    k.w8(at + 0x10, 1)?;
    k.interrupts.push(at);
    // Its service routine runs as a DPC `routine(k, ctx, 0, 0)` when the GPU line is asserted.
    let (routine, context) = (k.r32(at)?, k.r32(at + 4)?);
    k.dpcs.insert(
        at,
        Dpc {
            routine,
            context,
            queued: false,
        },
    );
    value(1)
}

pub fn disconnect_interrupt(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(0);
    if k.r8(at + 0x10)? == 0 {
        return value(0);
    }
    k.w8(at + 0x10, 0)?;
    k.interrupts.retain(|&i| i != at);
    k.dpcs.remove(&at);
    value(1)
}

/// `KeInitializeMutant(m, owned)`.
pub fn initialize_mutant(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (at, initially_owned) = (c.arg(0), c.arg(1) != 0);
    let (owner, count) = if initially_owned {
        (k.current, 1)
    } else {
        (0, 0)
    };
    write_header(k, at, 2, 8, 1 - count)?;
    k.w32(at + 0x10, at + 0x10)?;
    k.w32(at + 0x14, at + 0x10)?;
    k.w32(
        at + 0x18,
        if initially_owned {
            k.current_kthread()
        } else {
            0
        },
    )?;
    k.w32(at + 0x1C, 0)?;
    k.objects.add_at(at, Body::Mutant { owner, count });
    value(0)
}

/// Releases a mutant the caller owns; at count 0 it becomes free and the next owner (a woken
/// waiter) is written to `+0x18`. Returns the previous signal state.
///
/// # Errors
///
/// `MUTANT_NOT_OWNED` for a caller that does not own it, `OBJECT_TYPE_MISMATCH` for another
/// kind of object.
pub fn release_mutant_object(k: &mut Kernel, id: ObjId) -> Result<u32, u32> {
    let me = k.current;
    let Body::Mutant { owner, count } = &mut k.objects.get_mut(id).body else {
        return Err(status::OBJECT_TYPE_MISMATCH);
    };
    if *owner != me {
        return Err(status::MUTANT_NOT_OWNED);
    }
    let previous = 1u32.wrapping_sub(*count);
    *count -= 1;
    if *count == 0 {
        *owner = 0;
        k.wake_satisfiable();
    }
    k.mirror(id);
    let address = k.objects.get(id).address;
    if address != 0 {
        let owner = match k.objects.get(id).body {
            Body::Mutant { owner, .. } => owner,
            _ => 0,
        };
        let kthread = k.threads.get(owner).map_or(0, |t| t.kthread);
        k.w32(address + 0x18, kthread)
            .map_err(|_| status::INVALID_PARAMETER)?;
    }
    Ok(previous)
}

/// `KeReleaseMutant(m, incr, abandoned, wait)`.
pub fn release_mutant(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_at(c.arg(0))? else {
        return Ok(CallResult::Unsupported(format!(
            "no mutant at {:#010x}",
            c.arg(0)
        )));
    };
    match release_mutant_object(k, id) {
        Ok(previous) => value(previous),
        Err(_) => Ok(CallResult::Unsupported(
            "the caller does not own the mutant".to_owned(),
        )),
    }
}

/// `KeInitializeSemaphore(s, count, limit)`.
pub fn initialize_semaphore(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (at, count, limit) = (c.arg(0), c.arg(1), c.arg(2));
    write_header(k, at, 5, 5, count)?;
    k.w32(at + 0x10, limit)?;
    k.objects.add_at(
        at,
        Body::Semaphore {
            count: count.cast_signed(),
            limit: limit.cast_signed(),
        },
    );
    value(0)
}

/// Adds `adjustment` to a semaphore and wakes who it can; returns the previous count, or `None`
/// past the limit.
pub fn release_semaphore_object(k: &mut Kernel, id: ObjId, adjustment: i32) -> Option<u32> {
    let Body::Semaphore { count, limit } = &mut k.objects.get_mut(id).body else {
        return None;
    };
    let previous = *count;
    let new = count.checked_add(adjustment).filter(|n| n <= limit)?;
    *count = new;
    k.mirror(id);
    k.wake_satisfiable();
    Some(previous.cast_unsigned())
}

/// `KeReleaseSemaphore(s, incr, adjustment, wait)`.
pub fn release_semaphore(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.object_at(c.arg(0))? else {
        return Ok(CallResult::Unsupported(format!(
            "no semaphore at {:#010x}",
            c.arg(0)
        )));
    };
    match release_semaphore_object(k, id, c.arg(2).cast_signed()) {
        Some(previous) => value(previous),
        None => Ok(CallResult::Unsupported(
            "the semaphore's limit is exceeded".to_owned(),
        )),
    }
}

/// Fastcall `KfRaiseIrql(new)`: returns the old IRQL.
pub fn raise_irql(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let old = k.r8(IRQL)?;
    k.w8(IRQL, c.arg(0) as u8)?;
    value(u32::from(old))
}

/// Fastcall `KfLowerIrql(new)`.
pub fn lower_irql(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w8(IRQL, c.arg(0) as u8)?;
    value(0)
}

pub fn raise_irql_to_dpc_level(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    let old = k.r8(IRQL)?;
    k.w8(IRQL, 2)?;
    value(u32::from(old))
}

pub fn raise_irql_to_synch_level(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    let old = k.r8(IRQL)?;
    k.w8(IRQL, old.max(28))?;
    value(u32::from(old))
}

pub fn get_current_irql(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(u32::from(k.r8(IRQL)?))
}

/// `KeQuerySystemTime(*t)`.
pub fn query_system_time(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.w64(c.arg(0), k.clock.system_time())?;
    value(0)
}

fn value64(v: u64) -> KResult<CallResult> {
    Ok(CallResult::Value64(v as u32, (v >> 32) as u32))
}

pub fn query_interrupt_time(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value64(k.clock.since_boot())
}

/// Time since boot at 3375000 Hz.
pub fn query_performance_counter(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    let ticks = u128::from(k.clock.now() - BOOT_CLOCK);
    value64(
        u64::try_from(ticks * u128::from(PERFORMANCE_FREQUENCY) / 10_000_000).unwrap_or(u64::MAX),
    )
}

pub fn query_performance_frequency(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value64(PERFORMANCE_FREQUENCY)
}

pub fn get_current_thread(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(k.current_kthread())
}

/// The thread id whose KTHREAD is at `kthread`.
fn thread_at(k: &Kernel, kthread: u32) -> Option<u32> {
    k.threads.by_kthread(kthread).map(|t| t.id)
}

/// `KeSetBasePriorityThread(kthread, increment)`: the base priority relative to the process's
/// (8), which the thread then runs at; returns the old increment.
pub fn set_base_priority_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(thread) = thread_at(k, c.arg(0)).and_then(|id| k.threads.get_mut(id)) else {
        return value(0);
    };
    let old = thread.base_priority - 8;
    thread.base_priority = 8 + c.arg(1).cast_signed();
    thread.priority = thread.base_priority;
    value(old.cast_unsigned())
}

pub fn query_base_priority_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let base = thread_at(k, c.arg(0))
        .and_then(|id| k.threads.get(id))
        .map_or(8, |t| t.base_priority);
    value((base - 8).cast_unsigned())
}

/// `KeSetPriorityThread(kthread, priority)`: returns the old priority.
pub fn set_priority_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(thread) = thread_at(k, c.arg(0)).and_then(|id| k.threads.get_mut(id)) else {
        return value(8);
    };
    let old = thread.priority;
    thread.priority = c.arg(1).cast_signed();
    value(old.cast_unsigned())
}

pub fn bug_check(_: &mut Kernel, c: &Call) -> KResult<CallResult> {
    Ok(CallResult::Unsupported(format!(
        "bug check {:#x} with {:#x}, {:#x}, {:#x}, {:#x}",
        c.arg(0),
        c.arg(1),
        c.arg(2),
        c.arg(3),
        c.arg(4)
    )))
}

/// `KeSynchronizeExecution(k, routine, ctx)`: runs `routine(ctx)` on a new thread and resumes
/// the caller with the routine's AL.
pub fn synchronize_execution(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let me = k.current;
    let Some(tid) = k.run_routine(c.arg(1), &[c.arg(2)]) else {
        return value(1);
    };
    if let Some(thread) = k.threads.get_mut(tid) {
        thread.follow_up = Some(FollowUp::WakeWithResult {
            tid: me,
            low_byte: true,
        });
    }
    k.block(WaitOn::Explicit, None);
    Ok(CallResult::Block)
}

/// Adds to a thread's suspend count; returns the previous count. A thread that suspends itself
/// stops until resumed.
pub fn suspend(k: &mut Kernel, tid: u32) -> CallResult {
    let me = k.current;
    let Some(thread) = k.threads.get_mut(tid) else {
        return CallResult::Value(0);
    };
    let previous = thread.suspend_count;
    thread.suspend_count += 1;
    if tid == me {
        thread.resume = previous;
        return CallResult::Block;
    }
    CallResult::Value(previous)
}

/// Takes one from a thread's suspend count; returns the previous count.
pub fn resume(k: &mut Kernel, tid: u32) -> u32 {
    let Some(thread) = k.threads.get_mut(tid) else {
        return 0;
    };
    let previous = thread.suspend_count;
    thread.suspend_count = previous.saturating_sub(1);
    previous
}

pub fn suspend_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(tid) = thread_at(k, c.arg(0)) else {
        return value(0);
    };
    Ok(suspend(k, tid))
}

pub fn resume_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(tid) = thread_at(k, c.arg(0)) else {
        return value(0);
    };
    value(resume(k, tid))
}

/// Creates a thread and writes its handle and id.
fn create(
    k: &mut Kernel,
    handle: u32,
    id_out: u32,
    routine: u32,
    args: &[u32],
    spawn: Spawn,
) -> KResult<CallResult> {
    let Some(tid) = k.create_thread(routine, args, spawn) else {
        return value(status::NO_MEMORY);
    };
    let object = k.threads.get(tid).map_or(0, |t| t.object);
    if handle != 0 {
        let h = k.objects.open(object);
        k.w32(handle, h)?;
    }
    k.w32_opt(id_out, tid)?;
    value(status::SUCCESS)
}

/// `PsCreateSystemThreadEx(*h, extra, stack, tls, *id, ctx1, ctx2, suspended, debug, routine)`:
/// runs `routine(ctx1, ctx2)`.
pub fn create_system_thread_ex(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let spawn = Spawn {
        stack: c.arg(2),
        tls: (c.arg(3) != 0).then_some(c.arg(3)),
        suspended: c.arg(7) != 0,
        dpc: 0,
    };
    create(
        k,
        c.arg(0),
        c.arg(4),
        c.arg(9),
        &[c.arg(5), c.arg(6)],
        spawn,
    )
}

/// `PsCreateSystemThread(*h, *id, routine, ctx, debug)`: default stack, no TLS.
pub fn create_system_thread(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    create(
        k,
        c.arg(0),
        c.arg(1),
        c.arg(2),
        &[c.arg(3)],
        Spawn::default(),
    )
}

pub fn terminate_system_thread(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    Ok(CallResult::Exit)
}

/// `PsQueryStatistics(*s)`: live threads and open handles.
pub fn query_statistics(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(0);
    if k.r32(at)? < 12 {
        return value(status::INVALID_PARAMETER);
    }
    k.w32(at + 4, u32::try_from(k.threads.live()).unwrap_or(0))?;
    k.w32(at + 8, u32::try_from(k.objects.handle_count()).unwrap_or(0))?;
    value(status::SUCCESS)
}
