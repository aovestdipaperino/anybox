//! Guest threads, the baton and the thread table. Spec 7.1, 7.5, 7.6.

use serde::{Deserialize, Serialize};

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use super::object::{ObjId, Wait};

/// A thread's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    /// May run (when its suspend count is 0).
    Ready,
    /// Blocked on a wait.
    Waiting,
    /// Finished.
    Done,
}

/// What happens when a thread ends, besides signalling its object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FollowUp {
    /// Wake thread `tid`, which waits for this one, with the routine's result: its AL for
    /// `KeSynchronizeExecution`, or EAX.
    WakeWithResult { tid: u32, low_byte: bool },
    /// Carry on completing the IRP at `irp`, whose completion routine this thread ran.
    IrpCompletion { irp: u32 },
}

/// Where a guest thread last gave up the CPU, which says whether and how a hibernation can
/// resume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Park {
    /// Not started: it starts at its start routine.
    #[default]
    Start,
    /// In a kernel call that completed or blocked: it resumes with this ESP and EDX, and EAX
    /// from `resume` unless `eax` gives it.
    Native {
        esp: u32,
        edx: u32,
        eax: Option<u32>,
    },
    /// Before a kernel call to `target`, which it makes when resumed.
    Call { target: u32 },
    /// At a loop's poll.
    Poll,
    /// Mid-instruction: polling a GPU register, or held for frame pacing.
    Busy,
}

impl Park {
    /// Whether a hibernation can resume the thread where it is.
    #[must_use]
    pub fn resumable(self) -> bool {
        self != Self::Busy
    }
}

/// One guest thread.
#[derive(Debug, Serialize, Deserialize)]
pub struct Thread {
    /// 1-based; the boot thread is 1.
    pub id: u32,
    pub state: State,
    pub suspend_count: u32,
    pub priority: i32,
    pub base_priority: i32,
    /// The heap block holding the stack, or 0 (the boot stack is not on the heap).
    pub stack_block: u32,
    /// The TIB stack base, as KPCR `+0x04` holds it.
    pub stack_base: u32,
    /// The stack limit, as KPCR `+0x08` holds it.
    pub stack_limit: u32,
    /// The KTHREAD block's guest address.
    pub kthread: u32,
    /// The KTHREAD slot, or `None` for the boot thread's fixed block.
    pub slot: Option<u32>,
    /// The TLS block, or 0.
    pub tls: u32,
    /// The exception list head, kept while the thread is switched out (KPCR `+0x00`).
    pub exception_list: u32,
    /// The value its pending kernel call resumes with.
    pub resume: u32,
    pub wait: Option<Wait>,
    /// Its thread object.
    pub object: ObjId,
    /// The DPC it runs, or 0.
    pub dpc: u32,
    pub follow_up: Option<FollowUp>,
    /// The start routine and initial ESP, for its host thread.
    pub start: (u32, u32),
    /// Where it last gave up the CPU.
    pub park: Park,
}

/// The priority a DPC thread is scheduled at: above every thread's (31 is the highest), as a
/// DPC on the console runs before any thread does.
pub const DPC_PRIORITY: i32 = 32;

impl Thread {
    /// Whether the scheduler may pick it.
    #[must_use]
    pub fn runnable(&self) -> bool {
        self.state == State::Ready && self.suspend_count == 0
    }

    /// The priority the scheduler gives it: its own, or [`DPC_PRIORITY`] while it runs a DPC.
    #[must_use]
    pub fn scheduling_priority(&self) -> i32 {
        if self.dpc == 0 {
            self.priority
        } else {
            DPC_PRIORITY
        }
    }
}

/// The baton: names the one guest thread that may run and the value its pending kernel call
/// resumes with. It is separate from the kernel lock, so no thread holds the kernel lock while
/// waiting for its turn.
pub static BATON: Baton = Baton::new();

/// The boot epoch: a reset advances it, and every guest thread of an older epoch leaves (spec
/// 7.8).
pub static EPOCH: AtomicU64 = AtomicU64::new(0);

/// The guest thread holding the baton, readable without a lock (the fault handler reports it).
pub static RUNNING: AtomicU32 = AtomicU32::new(1);

/// The baton.
#[derive(Debug)]
pub struct Baton {
    lock: Mutex<(u32, u32)>,
    cond: Condvar,
}

impl Baton {
    const fn new() -> Self {
        Self {
            lock: Mutex::new((1, 0)),
            cond: Condvar::new(),
        }
    }

    /// Hands the baton to thread `t`, which resumes with `resume`.
    ///
    /// # Panics
    ///
    /// Panics if the baton's lock is poisoned.
    pub fn handoff(&self, t: u32, resume: u32) {
        let mut inner = self.lock.lock().unwrap();
        *inner = (t, resume);
        RUNNING.store(t, Ordering::Relaxed);
        self.cond.notify_all();
    }

    /// Parks until the baton names `me`, then returns the resume value; `None` once the boot
    /// epoch is no longer `epoch` (a reset: the thread must leave).
    ///
    /// # Panics
    ///
    /// Panics if the baton's lock is poisoned.
    pub fn wait_until(&self, me: u32, epoch: u64) -> Option<u32> {
        let mut inner = self.lock.lock().unwrap();
        while inner.0 != me {
            if EPOCH.load(Ordering::SeqCst) != epoch {
                return None;
            }
            inner = self.cond.wait(inner).unwrap();
        }
        (EPOCH.load(Ordering::SeqCst) == epoch).then_some(inner.1)
    }

    /// Wakes every parked thread, to find out about a reset.
    ///
    /// # Panics
    ///
    /// Panics if the baton's lock is poisoned.
    pub fn wake_all(&self) {
        let _inner = self.lock.lock().unwrap();
        self.cond.notify_all();
    }
}

/// The thread table.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Threads {
    /// Live threads, and the current one even when done.
    table: Vec<Thread>,
    next_id: u32,
    wait_sequence: u64,
    /// Which KTHREAD slots are taken.
    slots: Vec<bool>,
}

impl Threads {
    /// A live thread besides `me` that cannot be resumed where it is parked: while freezing for a
    /// hibernation it runs next, on to a point that can.
    #[must_use]
    pub fn busy_besides(&self, me: u32) -> Option<u32> {
        self.table
            .iter()
            .find(|t| {
                t.id != me && t.state != State::Done && t.park == Park::Busy && t.suspend_count == 0
            })
            .map(|t| t.id)
    }

    /// Whether a live thread besides `me` is suspended where it cannot be resumed: running it to
    /// a resumable point would break the suspension, so a freeze has to give up.
    #[must_use]
    pub fn busy_suspended_besides(&self, me: u32) -> bool {
        self.table.iter().any(|t| {
            t.id != me && t.state != State::Done && t.park == Park::Busy && t.suspend_count > 0
        })
    }

    /// A table holding the boot thread.
    #[must_use]
    pub fn new(boot: Thread, slots: u32) -> Self {
        Self {
            table: vec![boot],
            next_id: 2,
            wait_sequence: 0,
            slots: vec![false; slots as usize],
        }
    }

    /// The next thread id.
    pub fn take_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// A free KTHREAD slot, now taken.
    pub fn take_slot(&mut self) -> Option<u32> {
        let slot = self.slots.iter().position(|taken| !taken)?;
        self.slots[slot] = true;
        u32::try_from(slot).ok()
    }

    /// Releases a KTHREAD slot.
    pub fn free_slot(&mut self, slot: u32) {
        if let Some(taken) = self.slots.get_mut(slot as usize) {
            *taken = false;
        }
    }

    pub fn insert(&mut self, thread: Thread) {
        self.table.push(thread);
    }

    #[must_use]
    pub fn get(&self, id: u32) -> Option<&Thread> {
        self.table.iter().find(|t| t.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Thread> {
        self.table.iter_mut().find(|t| t.id == id)
    }

    /// The thread whose KTHREAD block is at `kthread`.
    #[must_use]
    pub fn by_kthread(&self, kthread: u32) -> Option<&Thread> {
        self.table
            .iter()
            .find(|t| t.kthread == kthread && t.state != State::Done)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Thread> {
        self.table.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Thread> {
        self.table.iter_mut()
    }

    /// The next wait sequence number.
    pub fn next_wait_sequence(&mut self) -> u64 {
        self.wait_sequence += 1;
        self.wait_sequence
    }

    /// The runnable thread to run next: the one of the highest priority and, among those, the
    /// first after `current`, so threads of equal priority take turns; `current` itself comes
    /// last.
    #[must_use]
    pub fn next_runnable(&self, current: u32) -> Option<u32> {
        let count = self.table.len();
        let start = self.table.iter().position(|t| t.id == current).unwrap_or(0);
        let mut best: Option<&Thread> = None;
        for thread in (1..=count).map(|step| &self.table[(start + step) % count]) {
            if thread.runnable()
                && best.is_none_or(|b| thread.scheduling_priority() > b.scheduling_priority())
            {
                best = Some(thread);
            }
        }
        best.map(|t| t.id)
    }

    /// As [`next_runnable`](Self::next_runnable), but any other runnable thread, whatever its
    /// priority, comes before `current`: for a thread that is held (frame pacing) rather than
    /// giving way.
    #[must_use]
    pub fn next_runnable_besides(&self, current: u32) -> Option<u32> {
        let count = self.table.len();
        let start = self.table.iter().position(|t| t.id == current).unwrap_or(0);
        let mut best: Option<&Thread> = None;
        for thread in (1..count).map(|step| &self.table[(start + step) % count]) {
            if thread.runnable()
                && best.is_none_or(|b| thread.scheduling_priority() > b.scheduling_priority())
            {
                best = Some(thread);
            }
        }
        best.or_else(|| self.get(current).filter(|t| t.runnable()))
            .map(|t| t.id)
    }

    /// Whether another thread is runnable with a priority above `current`'s at least by
    /// `above` (0: equal or higher; 1: higher).
    fn other_runnable_from(&self, current: u32, above: i32) -> bool {
        let floor = self
            .get(current)
            .map_or(i32::MIN, |t| t.scheduling_priority().saturating_add(above));
        self.table
            .iter()
            .any(|t| t.id != current && t.runnable() && t.scheduling_priority() >= floor)
    }

    /// Whether another thread is runnable that takes the CPU from `current` when its time slice
    /// ends: one of equal or higher priority (a lower one waits, as on the console).
    #[must_use]
    pub fn ready_at_slice_end(&self, current: u32) -> bool {
        self.other_runnable_from(current, 0)
    }

    /// Whether another thread is runnable with a higher priority than `current`, which must then
    /// give way at once: the other was resumed or woken, or `current`'s priority dropped.
    #[must_use]
    pub fn preempted(&self, current: u32) -> bool {
        self.other_runnable_from(current, 1)
    }

    /// Whether any other thread is runnable, whatever its priority.
    #[must_use]
    pub fn other_runnable(&self, current: u32) -> bool {
        self.table.iter().any(|t| t.id != current && t.runnable())
    }

    /// Drops finished threads, except `keep`.
    pub fn prune(&mut self, keep: u32) {
        self.table
            .retain(|t| t.state != State::Done || t.id == keep);
    }

    /// Whether every thread has finished.
    #[must_use]
    pub fn all_done(&self) -> bool {
        self.table.iter().all(|t| t.state == State::Done)
    }

    /// The number of live threads.
    #[must_use]
    pub fn live(&self) -> usize {
        self.table.iter().filter(|t| t.state != State::Done).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thread(id: u32, state: State) -> Thread {
        Thread {
            id,
            state,
            suspend_count: 0,
            priority: 8,
            base_priority: 8,
            stack_block: 0,
            stack_base: 0,
            stack_limit: 0,
            kthread: 0x8006_0000 + id * 0x400,
            slot: None,
            tls: 0,
            exception_list: u32::MAX,
            resume: 0,
            wait: None,
            object: 0,
            dpc: 0,
            follow_up: None,
            park: Park::Start,
            start: (0, 0),
        }
    }

    #[test]
    fn only_a_thread_mid_instruction_cannot_be_resumed() {
        assert!(Park::Start.resumable());
        assert!(Park::Poll.resumable());
        assert!(
            Park::Call {
                target: 0x8001_0000
            }
            .resumable()
        );
        assert!(
            Park::Native {
                esp: 0,
                edx: 0,
                eax: None
            }
            .resumable()
        );
        assert!(!Park::Busy.resumable());
    }

    #[test]
    fn a_busy_thread_other_than_the_caller_runs_next_while_freezing() {
        let mut threads = table(8, &[8, 8]);
        assert_eq!(threads.busy_besides(1), None);
        threads.get_mut(1).unwrap().park = Park::Busy;
        assert_eq!(threads.busy_besides(1), None, "not the caller itself");
        threads.get_mut(3).unwrap().park = Park::Busy;
        assert_eq!(threads.busy_besides(1), Some(3));
        threads.get_mut(3).unwrap().state = State::Done;
        assert_eq!(threads.busy_besides(1), None, "not a finished one");
    }

    #[test]
    fn a_suspended_busy_thread_is_not_run_while_freezing() {
        let mut threads = table(8, &[8, 8]);
        threads.get_mut(2).unwrap().park = Park::Busy;
        threads.get_mut(2).unwrap().suspend_count = 1;
        assert_eq!(threads.busy_besides(1), None, "not a suspended one");
        assert!(threads.busy_suspended_besides(1));
        assert!(!threads.busy_suspended_besides(2), "not the caller itself");
        threads.get_mut(3).unwrap().park = Park::Busy;
        assert_eq!(
            threads.busy_besides(1),
            Some(3),
            "a runnable one still runs"
        );
        threads.get_mut(2).unwrap().suspend_count = 0;
        assert!(!threads.busy_suspended_besides(1), "resumed again");
        threads.get_mut(2).unwrap().suspend_count = 1;
        threads.get_mut(2).unwrap().state = State::Done;
        assert!(!threads.busy_suspended_besides(1), "not a finished one");
    }

    #[test]
    fn the_scheduler_is_round_robin_and_the_current_thread_comes_last() {
        let mut threads = Threads::new(thread(1, State::Ready), 4);
        threads.insert(thread(2, State::Waiting));
        threads.insert(thread(3, State::Ready));
        assert_eq!(threads.next_runnable(1), Some(3));
        assert_eq!(threads.next_runnable(3), Some(1));
        threads.get_mut(1).unwrap().state = State::Waiting;
        assert_eq!(threads.next_runnable(3), Some(3));
        threads.get_mut(3).unwrap().suspend_count = 1;
        assert_eq!(threads.next_runnable(3), None);
    }

    /// A table whose boot thread (1, current) has priority `boot`, and a ready thread of each
    /// priority in `others`, ids from 2.
    fn table(boot: i32, others: &[i32]) -> Threads {
        let mut first = thread(1, State::Ready);
        first.priority = boot;
        let mut threads = Threads::new(first, 4);
        for (id, &priority) in (2u32..).zip(others) {
            let mut t = thread(id, State::Ready);
            t.priority = priority;
            threads.insert(t);
        }
        threads
    }

    #[test]
    fn the_highest_priority_runnable_thread_runs_next() {
        let threads = table(8, &[8, 24, 16]);
        assert_eq!(threads.next_runnable(1), Some(3), "24 beats 16 and 8");
        let threads = table(24, &[8, 16]);
        assert_eq!(
            threads.next_runnable(1),
            Some(1),
            "the current one keeps the CPU"
        );
    }

    #[test]
    fn threads_of_equal_priority_take_turns() {
        let threads = table(8, &[8, 8]);
        assert_eq!(threads.next_runnable(1), Some(2));
        assert_eq!(threads.next_runnable(2), Some(3));
        assert_eq!(threads.next_runnable(3), Some(1));
    }

    #[test]
    fn only_a_higher_priority_thread_preempts_and_a_lower_one_waits_for_the_slice() {
        let threads = table(8, &[8, 4]);
        assert!(!threads.preempted(1), "an equal and a lower thread wait");
        assert!(
            threads.ready_at_slice_end(1),
            "the equal one takes the slice's end"
        );
        let threads = table(8, &[4]);
        assert!(
            !threads.ready_at_slice_end(1),
            "a lower thread never takes the CPU"
        );
        assert!(threads.other_runnable(1));
        let mut threads = table(8, &[4, 9]);
        assert!(threads.preempted(1));
        threads.get_mut(3).unwrap().suspend_count = 1;
        assert!(!threads.preempted(1), "a suspended thread is not ready");
    }

    #[test]
    fn a_dpc_thread_runs_before_every_other_thread() {
        let mut threads = table(8, &[31, 4]);
        threads.get_mut(3).unwrap().dpc = 0x1234;
        assert_eq!(threads.next_runnable(1), Some(3));
        assert!(threads.preempted(1));
        assert!(threads.preempted(2), "even a real-time thread's");
    }

    #[test]
    fn a_held_thread_lets_a_lower_one_run() {
        let threads = table(16, &[8]);
        assert_eq!(threads.next_runnable(1), Some(1));
        assert_eq!(threads.next_runnable_besides(1), Some(2));
        let threads = table(16, &[]);
        assert_eq!(
            threads.next_runnable_besides(1),
            Some(1),
            "alone, it runs on"
        );
    }

    #[test]
    fn slots_are_reused() {
        let mut threads = Threads::new(thread(1, State::Ready), 2);
        assert_eq!(threads.take_slot(), Some(0));
        assert_eq!(threads.take_slot(), Some(1));
        assert_eq!(threads.take_slot(), None);
        threads.free_slot(0);
        assert_eq!(threads.take_slot(), Some(0));
    }
}
