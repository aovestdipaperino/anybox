//! Waits and signalling (spec 8.3): which objects are acquirable, what acquiring does, and the
//! order waiters are served in.

use super::object::{Body, ObjId, Wait, WaitOn};
use super::status;
use super::thread::State;
use super::{CallResult, KResult, Kernel};

/// `WAIT_0`: the first object of a wait.
pub const WAIT_0: u32 = 0;

/// The dispatcher header's signal state is at `+4` (spec 8.5).
const SIGNAL_STATE: u32 = 4;

impl Kernel {
    /// Whether thread `tid` could acquire object `id` now.
    #[must_use]
    pub fn acquirable(&self, id: ObjId, tid: u32) -> bool {
        match &self.objects.get(id).body {
            Body::Thread { tid: of } => {
                self.threads.get(*of).is_none_or(|t| t.state == State::Done)
            }
            Body::Event { signaled, .. } => *signaled,
            Body::Timer(timer) => timer.signaled,
            Body::Mutant { owner, .. } => *owner == 0 || *owner == tid,
            Body::Semaphore { count, .. } => *count > 0,
            Body::IoCompletion { packets, .. } => !packets.is_empty(),
            // A file signals when its I/O is done, and all I/O is synchronous.
            Body::File(_) => true,
            Body::CriticalSection
            | Body::SymbolicLink(_)
            | Body::Device { .. }
            | Body::Directory(_)
            | Body::Queue { .. }
            | Body::Other => false,
        }
    }

    /// Acquires object `id` for `tid`: clears a synchronization event or timer, makes a mutant
    /// the caller's, takes one from a semaphore.
    pub fn acquire(&mut self, id: ObjId, tid: u32) {
        match &mut self.objects.get_mut(id).body {
            Body::Event {
                sync: true,
                signaled,
            } => *signaled = false,
            Body::Timer(timer) if timer.sync => timer.signaled = false,
            Body::Mutant { owner, count } => {
                *owner = tid;
                *count += 1;
            }
            Body::Semaphore { count, .. } => *count -= 1,
            _ => {}
        }
        self.mirror(id);
    }

    /// Writes an object's signal state into its guest dispatcher header, which titles may read.
    pub fn mirror(&self, id: ObjId) {
        let object = self.objects.get(id);
        if object.address == 0 {
            return;
        }
        let state = match &object.body {
            Body::Event { signaled, .. } => u32::from(*signaled),
            Body::Timer(timer) => u32::from(timer.signaled),
            Body::Mutant { count, .. } => 1u32.wrapping_sub(*count),
            Body::Semaphore { count, .. } => count.cast_unsigned(),
            _ => return,
        };
        let _ = self.w32(object.address + SIGNAL_STATE, state);
    }

    /// Tries to satisfy a wait for `tid` now. Returns the resume value when it is satisfied
    /// (having acquired what it waited for).
    fn try_satisfy(&mut self, on: &WaitOn, tid: u32) -> Option<u32> {
        let WaitOn::Objects { ids, all } = on else {
            return None;
        };
        if *all {
            if ids.iter().all(|&id| self.acquirable(id, tid)) {
                for &id in ids {
                    self.acquire(id, tid);
                }
                return Some(WAIT_0);
            }
            return None;
        }
        let index = ids.iter().position(|&id| self.acquirable(id, tid))?;
        self.acquire(ids[index], tid);
        Some(WAIT_0 + u32::try_from(index).unwrap_or(0))
    }

    /// Starts a wait of the current thread on `ids` (spec 8.3). `timeout` is the guest
    /// `LARGE_INTEGER*`: null waits forever, negative is relative, positive absolute; zero or a
    /// deadline already passed returns `TIMEOUT` at once after trying to acquire.
    ///
    /// # Errors
    ///
    /// A fault when the timeout cannot be read.
    pub fn wait_for(&mut self, ids: Vec<ObjId>, all: bool, timeout: u32) -> KResult<CallResult> {
        let deadline = self.deadline(timeout)?;
        let on = WaitOn::Objects { ids, all };
        let tid = self.current;
        if let Some(value) = self.try_satisfy(&on, tid) {
            return Ok(CallResult::Value(value));
        }
        if deadline.is_some_and(|d| d <= self.clock.now()) {
            return Ok(CallResult::Value(status::TIMEOUT));
        }
        self.block(on, deadline);
        Ok(CallResult::Block)
    }

    /// Converts a guest timeout pointer to a deadline in clock ticks.
    ///
    /// # Errors
    ///
    /// A fault when the timeout cannot be read.
    pub fn deadline(&self, timeout: u32) -> KResult<Option<u64>> {
        if timeout == 0 {
            return Ok(None);
        }
        let value = self.r64(timeout)?.cast_signed();
        Ok(Some(self.clock.due(value)))
    }

    /// Marks the current thread waiting.
    pub fn block(&mut self, on: WaitOn, deadline: Option<u64>) {
        let seq = self.threads.next_wait_sequence();
        let me = self.current;
        if let Some(thread) = self.threads.get_mut(me) {
            thread.state = State::Waiting;
            thread.wait = Some(Wait { on, deadline, seq });
        }
    }

    /// Wakes a waiting thread with `value`.
    pub fn wake(&mut self, tid: u32, value: u32) {
        if let Some(thread) = self.threads.get_mut(tid)
            && thread.state == State::Waiting
        {
            thread.state = State::Ready;
            thread.wait = None;
            thread.resume = value;
        }
    }

    /// Serves waiters oldest first, each as long as what it waits for is acquirable. Call after
    /// anything is signalled.
    pub fn wake_satisfiable(&mut self) {
        let waits_on_objects = |t: &super::thread::Thread| {
            t.state == State::Waiting
                && t.wait
                    .as_ref()
                    .is_some_and(|w| matches!(w.on, WaitOn::Objects { .. }))
        };
        // Time is serviced on every kernel contact: most of the time no one waits on objects.
        if !self.threads.iter().any(waits_on_objects) {
            return;
        }
        let mut waiting: Vec<(u64, u32)> = self
            .threads
            .iter()
            .filter(|t| waits_on_objects(t))
            .filter_map(|t| t.wait.as_ref().map(|w| (w.seq, t.id)))
            .collect();
        waiting.sort_unstable();
        for (_, tid) in waiting {
            // The wait is taken out while it is tried, and put back unless it is satisfied.
            let Some(wait) = self.threads.get_mut(tid).and_then(|t| t.wait.take()) else {
                continue;
            };
            match self.try_satisfy(&wait.on, tid) {
                Some(value) => {
                    if let Some(thread) = self.threads.get_mut(tid) {
                        thread.wait = Some(wait);
                    }
                    self.wake(tid, value);
                }
                None => {
                    if let Some(thread) = self.threads.get_mut(tid) {
                        thread.wait = Some(wait);
                    }
                }
            }
        }
    }

    /// The threads waiting to be handed an entry of queue or completion port `id`, oldest first.
    #[must_use]
    pub fn takers(&self, id: ObjId) -> Vec<u32> {
        let mut takers: Vec<(u64, u32)> = self
            .threads
            .iter()
            .filter(|t| t.state == State::Waiting)
            .filter_map(|t| {
                let wait = t.wait.as_ref()?;
                (wait.on == WaitOn::Take(id)).then_some((wait.seq, t.id))
            })
            .collect();
        takers.sort_unstable();
        takers.into_iter().map(|(_, tid)| tid).collect()
    }

    /// Wakes every thread whose wait deadline has passed: object waits with `TIMEOUT`, delays
    /// and kernel waits with `SUCCESS` and `TIMEOUT`.
    pub fn wake_expired(&mut self) {
        let now = self.clock.now();
        let expired_by = |t: &super::thread::Thread| {
            t.state == State::Waiting
                && t.wait
                    .as_ref()
                    .and_then(|w| w.deadline)
                    .is_some_and(|d| d <= now)
        };
        if !self.threads.iter().any(expired_by) {
            return;
        }
        let expired: Vec<(u32, u32)> = self
            .threads
            .iter()
            .filter(|t| t.state == State::Waiting)
            .filter_map(|t| {
                let wait = t.wait.as_ref()?;
                let deadline = wait.deadline?;
                (deadline <= now).then_some((
                    t.id,
                    if wait.on == WaitOn::Delay {
                        status::SUCCESS
                    } else {
                        status::TIMEOUT
                    },
                ))
            })
            .collect();
        for (tid, value) in expired {
            self.wake(tid, value);
        }
    }

    /// Whether `address` holds an event's dispatcher header: byte `+0` (the type) is 0 or 1 and
    /// byte `+2` (the size in dwords) is 4.
    fn event_header_at(&self, address: u32) -> bool {
        matches!(
            (self.r8(address), self.r8(address + 2)),
            (Ok(kind), Ok(4)) if kind <= 1
        )
    }

    /// Takes the signal state the guest left in the header of the event `id` at `address`.
    /// Titles change it in place without the kernel (Direct3D's wait for the vertical blank
    /// resets its event by writing zero there, an inline `KeResetEvent`); the kernel writes its
    /// own changes to the header ([`mirror`](Self::mirror)), so the header is the event's state
    /// while it still holds an event header.
    fn take_guest_event_state(&mut self, id: ObjId, address: u32) {
        if !matches!(self.objects.get(id).body, Body::Event { .. })
            || !self.event_header_at(address)
        {
            return;
        }
        let Ok(state) = self.r32(address + SIGNAL_STATE) else {
            return;
        };
        if let Body::Event { signaled, .. } = &mut self.objects.get_mut(id).body {
            *signaled = state != 0;
        }
    }

    /// The object a guest dispatcher pointer names, adopting an event a title built in guest
    /// memory without the kernel (spec 8.3): byte `+0` is 0 or 1 and byte `+2` is 4. A known
    /// event first takes the signal state the guest left in its header.
    ///
    /// # Errors
    ///
    /// A fault when the header cannot be read.
    pub fn object_at(&mut self, address: u32) -> KResult<Option<ObjId>> {
        if let Some(id) = self.objects.at(address) {
            self.take_guest_event_state(id, address);
            return Ok(Some(id));
        }
        if let Some(thread) = self.threads.by_kthread(address) {
            return Ok(Some(thread.object));
        }
        if address == 0 || !self.memory.is_committed(address) {
            return Ok(None);
        }
        let kind = self.r8(address)?;
        if self.event_header_at(address) {
            let signaled = self.r32(address + SIGNAL_STATE)? != 0;
            let id = self.objects.add_at(
                address,
                Body::Event {
                    sync: kind == 1,
                    signaled,
                },
            );
            return Ok(Some(id));
        }
        Ok(None)
    }

    /// The object a handle names; the current-thread pseudo-handle names the current thread.
    #[must_use]
    pub fn object_of_handle(&self, handle: u32) -> Option<ObjId> {
        if handle == super::object::CURRENT_THREAD {
            return self.threads.get(self.current).map(|t| t.object);
        }
        self.objects.handle(handle)
    }

    /// Sets an event, wakes who it can, and returns the previous state.
    pub fn set_event(&mut self, id: ObjId) -> u32 {
        let previous = match &mut self.objects.get_mut(id).body {
            Body::Event { signaled, .. } => std::mem::replace(signaled, true),
            _ => return 0,
        };
        self.mirror(id);
        self.wake_satisfiable();
        u32::from(previous)
    }

    /// Clears an event and returns the previous state.
    pub fn reset_event(&mut self, id: ObjId) -> u32 {
        let previous = match &mut self.objects.get_mut(id).body {
            Body::Event { signaled, .. } => std::mem::replace(signaled, false),
            _ => return 0,
        };
        self.mirror(id);
        u32::from(previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::object::Timer;
    use crate::kernel::testing::{kernel, run_as, threads};
    use crate::kernel::{Call, Dpc};

    /// A guest `LARGE_INTEGER` holding `value`, for timeouts.
    fn large_integer(k: &Kernel, value: i64) -> u32 {
        let at = crate::mem::HEAP_BASE + 0x2F_0000;
        k.w64(at, value.cast_unsigned()).unwrap();
        at
    }

    fn event(k: &mut Kernel, sync: bool) -> ObjId {
        k.objects.add(Body::Event {
            sync,
            signaled: false,
        })
    }

    #[test]
    fn a_wait_on_a_signalled_object_returns_at_once() {
        let mut k = kernel();
        let e = event(&mut k, false);
        k.set_event(e);
        assert_eq!(k.wait_for(vec![e], false, 0), Ok(CallResult::Value(WAIT_0)));
        // A notification event stays signalled.
        assert_eq!(k.wait_for(vec![e], false, 0), Ok(CallResult::Value(WAIT_0)));
    }

    #[test]
    fn a_synchronization_event_wakes_one_waiter_oldest_first() {
        let mut k = kernel();
        let ids = threads(&mut k, 2);
        let e = event(&mut k, true);
        run_as(&mut k, ids[1]);
        assert_eq!(k.wait_for(vec![e], false, 0), Ok(CallResult::Block));
        run_as(&mut k, ids[0]);
        assert_eq!(k.wait_for(vec![e], false, 0), Ok(CallResult::Block));
        run_as(&mut k, 1);
        k.set_event(e);
        assert_eq!(
            k.threads.get(ids[1]).unwrap().state,
            State::Ready,
            "the oldest waiter"
        );
        assert_eq!(k.threads.get(ids[0]).unwrap().state, State::Waiting);
        assert!(!k.acquirable(e, 1), "the wake consumed the signal");
        k.set_event(e);
        assert_eq!(k.threads.get(ids[0]).unwrap().state, State::Ready);
    }

    #[test]
    fn a_notification_event_wakes_every_waiter() {
        let mut k = kernel();
        let ids = threads(&mut k, 2);
        let e = event(&mut k, false);
        for &t in &ids {
            run_as(&mut k, t);
            assert_eq!(k.wait_for(vec![e], false, 0), Ok(CallResult::Block));
        }
        run_as(&mut k, 1);
        k.set_event(e);
        assert!(
            ids.iter()
                .all(|&t| k.threads.get(t).unwrap().state == State::Ready)
        );
    }

    #[test]
    fn wait_any_resumes_with_the_index_and_wait_all_needs_every_object() {
        let mut k = kernel();
        let ids = threads(&mut k, 2);
        let (a, b) = (event(&mut k, true), event(&mut k, true));
        run_as(&mut k, ids[0]);
        assert_eq!(k.wait_for(vec![a, b], false, 0), Ok(CallResult::Block));
        run_as(&mut k, ids[1]);
        assert_eq!(k.wait_for(vec![a, b], true, 0), Ok(CallResult::Block));
        run_as(&mut k, 1);
        k.set_event(b);
        let any = k.threads.get(ids[0]).unwrap();
        assert_eq!((any.state, any.resume), (State::Ready, WAIT_0 + 1));
        k.set_event(a);
        assert_eq!(
            k.threads.get(ids[1]).unwrap().state,
            State::Waiting,
            "b was taken"
        );
        k.set_event(b);
        let all = k.threads.get(ids[1]).unwrap();
        assert_eq!((all.state, all.resume), (State::Ready, WAIT_0));
        assert!(
            !k.acquirable(a, 1) && !k.acquirable(b, 1),
            "wait-all acquired both"
        );
    }

    #[test]
    fn timeouts() {
        let mut k = kernel();
        let e = event(&mut k, false);
        let zero = large_integer(&k, 0);
        assert_eq!(
            k.wait_for(vec![e], false, zero),
            Ok(CallResult::Value(status::TIMEOUT))
        );
        let ten_ms = large_integer(&k, -100_000);
        assert_eq!(k.wait_for(vec![e], false, ten_ms), Ok(CallResult::Block));
        k.clock.pass(99_999);
        k.service_time();
        assert_eq!(k.threads.get(1).unwrap().state, State::Waiting);
        k.clock.pass(1);
        k.service_time();
        let me = k.threads.get(1).unwrap();
        assert_eq!((me.state, me.resume), (State::Ready, status::TIMEOUT));
    }

    #[test]
    fn a_mutant_is_recursive_for_its_owner_and_free_after_the_last_release() {
        let mut k = kernel();
        let ids = threads(&mut k, 1);
        let m = k.objects.add(Body::Mutant { owner: 0, count: 0 });
        assert_eq!(k.wait_for(vec![m], false, 0), Ok(CallResult::Value(WAIT_0)));
        assert_eq!(k.wait_for(vec![m], false, 0), Ok(CallResult::Value(WAIT_0)));
        run_as(&mut k, ids[0]);
        assert_eq!(k.wait_for(vec![m], false, 0), Ok(CallResult::Block));
        run_as(&mut k, 1);
        let release = crate::kernel::handlers::release_mutant_object;
        assert_eq!(release(&mut k, m), Ok(1u32.wrapping_sub(2)));
        assert_eq!(k.threads.get(ids[0]).unwrap().state, State::Waiting);
        assert_eq!(release(&mut k, m), Ok(0));
        assert_eq!(k.threads.get(ids[0]).unwrap().state, State::Ready);
        assert!(
            matches!(k.objects.get(m).body, Body::Mutant { owner, count: 1 } if owner == ids[0])
        );
        assert_eq!(release(&mut k, m), Err(status::MUTANT_NOT_OWNED));
    }

    #[test]
    fn a_periodic_timer_catches_up_by_whole_periods_and_queues_its_dpc_once() {
        let mut k = kernel();
        k.dpcs.insert(
            0x2000,
            Dpc {
                routine: 0x3000,
                context: 7,
                queued: false,
            },
        );
        let now = k.clock.now();
        let t = k.objects.add(Body::Timer(Timer {
            due: Some(now + 100),
            period: 100,
            dpc: 0x2000,
            ..Timer::default()
        }));
        k.clock.pass(350);
        k.service_time();
        let Body::Timer(timer) = &k.objects.get(t).body else {
            unreachable!()
        };
        assert!(timer.signaled);
        assert_eq!(timer.due, Some(now + 400));
        assert!(k.dpcs[&0x2000].queued);
        assert_eq!(k.unstarted.len(), 1, "one DPC thread");
        let dpc_thread = k.unstarted[0];
        k.clock.pass(100);
        k.service_time();
        assert_eq!(k.unstarted.len(), 1, "a queued DPC is not queued again");
        k.end_thread(dpc_thread, None);
        assert!(!k.dpcs[&0x2000].queued);
    }

    #[test]
    fn a_contended_critical_section_is_handed_to_the_oldest_waiter() {
        let mut k = kernel();
        let ids = threads(&mut k, 2);
        let cs = crate::mem::HEAP_BASE + 0x2E_0000;
        let call = |cs| Call {
            args: vec![cs],
            esp: 0,
        };
        let enter = crate::kernel::handlers::lookup("RtlEnterCriticalSection").unwrap();
        let leave = crate::kernel::handlers::lookup("RtlLeaveCriticalSection").unwrap();
        let init = crate::kernel::handlers::lookup("RtlInitializeCriticalSection").unwrap();
        init(&mut k, &call(cs)).unwrap();
        assert_eq!(enter(&mut k, &call(cs)), Ok(CallResult::Value(0)));
        for &t in &ids {
            run_as(&mut k, t);
            assert_eq!(enter(&mut k, &call(cs)), Ok(CallResult::Block));
        }
        assert_eq!(k.r32(cs + 0x10), Ok(2), "the owner and two waiters");
        run_as(&mut k, 1);
        assert_eq!(leave(&mut k, &call(cs)), Ok(CallResult::Value(0)));
        let first = k.threads.get(ids[0]).unwrap();
        assert_eq!(first.state, State::Ready);
        assert_eq!(k.r32(cs + 0x18), Ok(first.kthread));
        assert_eq!(k.r32(cs + 0x14), Ok(1));
        assert_eq!(k.r32(cs + 0x10), Ok(1));
        assert_eq!(k.threads.get(ids[1]).unwrap().state, State::Waiting);
    }

    /// Calls the handler of export `name` with `args`.
    fn export(k: &mut Kernel, name: &str, args: &[u32]) -> CallResult {
        let handler = crate::kernel::handlers::lookup(name).unwrap();
        handler(
            k,
            &Call {
                args: args.to_vec(),
                esp: 0,
            },
        )
        .unwrap()
    }

    #[test]
    fn an_event_the_guest_resets_in_its_header_is_unsignalled_for_the_next_wait() {
        let mut k = kernel();
        let e = crate::mem::HEAP_BASE + 0x2E_1000;
        let no_wait = large_integer(&k, 0);
        export(&mut k, "KeInitializeEvent", &[e, 0, 1]); // notification, signalled
        let wait = [e, 0, 0, 0, no_wait];
        assert_eq!(
            export(&mut k, "KeWaitForSingleObject", &wait),
            CallResult::Value(WAIT_0)
        );
        // What Direct3D's wait for the vertical blank does first: zero the signal state.
        k.w32(e + 4, 0).unwrap();
        assert_eq!(
            export(&mut k, "KeWaitForSingleObject", &wait),
            CallResult::Value(status::TIMEOUT)
        );
    }

    #[test]
    fn ke_set_event_shows_in_the_header_and_reports_the_state_the_guest_left() {
        let mut k = kernel();
        let e = crate::mem::HEAP_BASE + 0x2E_1000;
        export(&mut k, "KeInitializeEvent", &[e, 0, 0]);
        assert_eq!(
            export(&mut k, "KeSetEvent", &[e, 1, 0]),
            CallResult::Value(0)
        );
        assert_eq!(k.r32(e + 4), Ok(1), "the header shows the set");
        k.w32(e + 4, 0).unwrap();
        assert_eq!(
            export(&mut k, "KeSetEvent", &[e, 1, 0]),
            CallResult::Value(0),
            "the reset in place was seen"
        );
    }

    #[test]
    fn a_wait_that_takes_an_auto_reset_signal_clears_the_header() {
        let mut k = kernel();
        let e = crate::mem::HEAP_BASE + 0x2E_1000;
        let no_wait = large_integer(&k, 0);
        export(&mut k, "KeInitializeEvent", &[e, 1, 1]); // synchronization, signalled
        let wait = [e, 0, 0, 0, no_wait];
        assert_eq!(
            export(&mut k, "KeWaitForSingleObject", &wait),
            CallResult::Value(WAIT_0)
        );
        assert_eq!(k.r32(e + 4), Ok(0), "the signal was taken");
        assert_eq!(
            export(&mut k, "KeWaitForSingleObject", &wait),
            CallResult::Value(status::TIMEOUT)
        );
    }

    #[test]
    fn a_header_that_is_no_event_header_leaves_the_state_alone() {
        let mut k = kernel();
        // An object the kernel gave a zeroed block for, as ObReferenceObjectByHandle does.
        let at = crate::mem::HEAP_BASE + 0x2E_2000;
        let id = k.objects.add_at(
            at,
            Body::Event {
                sync: false,
                signaled: true,
            },
        );
        assert_eq!(k.object_at(at), Ok(Some(id)));
        assert!(k.acquirable(id, 1));
    }
}
