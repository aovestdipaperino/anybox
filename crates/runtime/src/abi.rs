//! The C ABI between the lifted code and the runtime (spec 5), and the scheduler that hands the
//! baton between guest threads (spec 7.1, 7.4).

use std::cell::Cell;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::boot::KPCR;
use crate::kernel::thread::{BATON, EPOCH, Park, State};
use crate::kernel::{Dispatch, Kernel};

/// The result of a native call, as the glue reads it (spec 5.2).
#[derive(Debug, Default)]
#[repr(C)]
pub struct Result {
    pub state: u32,
    pub eax: u32,
    pub esp: u32,
    pub edx: u32,
}

/// The kernel, once booted, behind one lock.
static KERNEL: Mutex<Option<Kernel>> = Mutex::new(None);

/// Every 256th kernel call may switch threads (spec 7.4).
const CALLS_PER_SLICE: u64 = 256;
/// Reads in a row of one GPU register that find it unchanged before the polling thread may give
/// way.
const MMIO_READS_PER_SLICE: u64 = 64;
/// The default length of a time slice, in microseconds (`ANYXBOX_SLICE_US` overrides it).
const SLICE_US: u64 = 250;
/// The longest the scheduler sleeps before looking again.
const IDLE_STEP: Duration = Duration::from_millis(2);
/// A guest thread's host stack.
const HOST_STACK: usize = 64 * 1024 * 1024;

thread_local! {
    /// The guest thread this host thread runs; the process's main thread runs the boot thread.
    static ME: Cell<u32> = const { Cell::new(1) };
    /// The boot epoch this host thread's guest thread belongs to.
    static MY_EPOCH: Cell<u64> = const { Cell::new(0) };
    /// Whether this host thread is counted in `LIVE`.
    static COUNTED: Cell<bool> = const { Cell::new(false) };
}

/// Host threads that may still touch guest state: a reset waits for none (spec 7.8).
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// The title's executable, which stays valid for the life of the process, for a reset.
static XBE: OnceLock<&'static [u8]> = OnceLock::new();

unsafe extern "C" {
    /// Defined by the glue: runs a new guest thread's start routine (spec 5.3).
    fn anyxbox_thread_body(start: u32, esp: u32);
    /// Defined by the glue: runs the title's entry point over new guest memory (spec 5.3).
    fn anyxbox_entry_body(base: *mut u8, esp: u32, fs_base: u32);
    /// Defined by the glue: resumes guest thread `tid` from its frames, with FS at `fs_base`.
    fn anyxbox_resume_body(tid: u32, fs_base: u32);
}

/// This host thread no longer touches guest state: it leaves the live count, once, and the
/// runtime forgets where its guest CPU was (before the count drops, so a reset waiting on the
/// count never sees a layout of a thread from before it).
fn uncount() {
    if COUNTED.replace(false) {
        crate::hibernate::cpu::forget(ME.get());
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Whether a reset has happened since this host thread's guest thread was started.
fn reset_since_start() -> bool {
    EPOCH.load(Ordering::SeqCst) != MY_EPOCH.get()
}

/// Ends this host thread because of a reset. The caller holds no lock.
fn leave() -> ! {
    uncount();
    unsafe { libc::pthread_exit(std::ptr::null_mut()) }
}

/// Leaves when a reset has happened since this thread started (every contact with the runtime
/// checks, spec 7.8).
fn leave_if_reset() {
    if COUNTED.get() && reset_since_start() {
        leave();
    }
}

/// Starts a counted host thread with a 64 MiB stack running `body(arg)`.
fn spawn_guest_thread(
    body: extern "C" fn(*mut libc::c_void) -> *mut libc::c_void,
    arg: *mut libc::c_void,
) -> bool {
    LIVE.fetch_add(1, Ordering::SeqCst);
    unsafe {
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        libc::pthread_attr_init(&raw mut attr);
        libc::pthread_attr_setstacksize(&raw mut attr, HOST_STACK);
        libc::pthread_attr_setdetachstate(&raw mut attr, libc::PTHREAD_CREATE_DETACHED);
        let mut handle: libc::pthread_t = std::mem::zeroed();
        let created = libc::pthread_create(&raw mut handle, &raw const attr, body, arg);
        libc::pthread_attr_destroy(&raw mut attr);
        if created != 0 {
            LIVE.fetch_sub(1, Ordering::SeqCst);
        }
        created == 0
    }
}

/// Runs `f` on the kernel under its lock; `None` before boot.
pub(crate) fn with_kernel<R>(f: impl FnOnce(&mut Kernel) -> R) -> Option<R> {
    lock().as_mut().map(f)
}

/// Locks the kernel, recovering it if a thread panicked while holding it.
fn lock() -> MutexGuard<'static, Option<Kernel>> {
    KERNEL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The start of a guest thread's host thread: waits for its first turn, then runs the routine.
extern "C" fn host_thread(arg: *mut libc::c_void) -> *mut libc::c_void {
    let (tid, start, esp, epoch) = *unsafe { Box::from_raw(arg.cast::<(u32, u32, u32, u64)>()) };
    ME.set(tid);
    MY_EPOCH.set(epoch);
    COUNTED.set(true);
    if BATON.wait_until(tid, epoch).is_none() {
        leave();
    }
    unsafe { anyxbox_thread_body(start, esp) };
    uncount();
    std::ptr::null_mut()
}

/// Starts host threads for guest threads created since the last call.
fn start_host_threads(kernel: &mut Kernel) {
    for tid in std::mem::take(&mut kernel.unstarted) {
        let Some(thread) = kernel.threads.get(tid) else {
            continue;
        };
        let epoch = EPOCH.load(Ordering::SeqCst);
        let arg = Box::into_raw(Box::new((tid, thread.start.0, thread.start.1, epoch)));
        if !spawn_guest_thread(host_thread, arg.cast()) {
            eprintln!("anyxbox: could not start a host thread for guest thread {tid}");
            std::process::exit(1);
        }
    }
}

/// The guest thread the calling host thread runs.
pub(crate) fn me() -> u32 {
    ME.get()
}

/// While freezing for a hibernation, parks `me` where it can be resumed (`park`) and gives the
/// baton to a thread that cannot be resumed where it is, so it runs on to a point that can; with
/// none left, the title is frozen and nobody runs. Returns, locked again, when `me`'s turn comes
/// back after the hibernation is called off.
fn park_for_hibernation(
    mut guard: MutexGuard<'static, Option<Kernel>>,
    me: u32,
    park: Park,
) -> MutexGuard<'static, Option<Kernel>> {
    let Some(kernel) = guard.as_mut() else {
        return guard;
    };
    if let Some(thread) = kernel.threads.get_mut(me) {
        thread.park = park;
    }
    if let Some(next) = kernel.threads.busy_besides(me) {
        kernel.switch_to(next);
        let resume = kernel.threads.get(next).map_or(0, |t| t.resume);
        BATON.handoff(next, resume);
    } else if kernel.threads.busy_suspended_besides(me) {
        // A suspended thread is mid-instruction and must not run: call the hibernation off.
        crate::hibernate::give_up("a suspended thread is mid-instruction; try again".to_owned());
        return guard;
    } else if crate::hibernate::mark_frozen() {
        BATON.handoff(0, 0);
    } else {
        // The hibernation was called off meanwhile: carry on.
        return guard;
    }
    drop(guard);
    if BATON.wait_until(me, MY_EPOCH.get()).is_none() {
        leave();
    }
    lock()
}

/// After a hibernation is called off with the title frozen, gives the baton back to the thread
/// that froze it.
pub(crate) fn hand_back() {
    if let Some((current, resume)) =
        with_kernel(|k| (k.current, k.threads.get(k.current).map_or(0, |t| t.resume)))
    {
        BATON.handoff(current, resume);
    }
}

/// Gives the baton to the next thread that may run and, unless `me` has finished, waits until it
/// comes back; returns `me`'s resume value. With nothing to run, services time without the lock
/// in steps of at most 2 ms; exits the process when every thread has finished (0) or nothing can
/// ever wake the waiting ones (1).
fn reschedule(guard: MutexGuard<'static, Option<Kernel>>, me: u32) -> u32 {
    reschedule_with(guard, me, false)
}

/// As [`reschedule`]; with `held`, any other runnable thread, whatever its priority, runs before
/// `me` (a thread held for frame pacing is not giving way to its equals but waiting).
fn reschedule_with(mut guard: MutexGuard<'static, Option<Kernel>>, me: u32, held: bool) -> u32 {
    loop {
        let Some(kernel) = guard.as_mut().filter(|_| !reset_since_start()) else {
            drop(guard);
            leave();
        };
        if crate::hibernate::freezing()
            && let Some(park) = kernel
                .threads
                .get(me)
                .filter(|t| t.state != State::Done && t.park.resumable())
                .map(|t| t.park)
        {
            guard = park_for_hibernation(guard, me, park);
            continue;
        }
        kernel.service_time();
        start_host_threads(kernel);
        let current = kernel.current;
        let next = if held {
            kernel.threads.next_runnable_besides(current)
        } else {
            kernel.threads.next_runnable(current)
        };
        if let Some(next) = next {
            if next == me {
                kernel.switch_to(me);
                return kernel.threads.get(me).map_or(0, |t| t.resume);
            }
            kernel.switch_to(next);
            kernel.threads.prune(next);
            let resume = kernel.threads.get(next).map_or(0, |t| t.resume);
            if kernel.trace != crate::kernel::Trace::Off {
                eprintln!("anyxbox: baton to thread {next} from {me}");
            }
            BATON.handoff(next, resume);
            let finished = kernel
                .threads
                .get(me)
                .is_none_or(|t| t.state == State::Done);
            drop(guard);
            if finished {
                return 0;
            }
            return BATON
                .wait_until(me, MY_EPOCH.get())
                .unwrap_or_else(|| leave());
        }
        if kernel.threads.all_done() {
            kernel.fs.persist();
            crate::hibernate::discard();
            std::process::exit(0);
        }
        let Some(wake) = kernel.next_wake() else {
            eprintln!("every thread is waiting and nothing will wake them");
            std::process::exit(1);
        };
        let ticks = wake.saturating_sub(kernel.clock.now());
        let wait = Duration::from_nanos(ticks.saturating_mul(100)).min(IDLE_STEP);
        drop(guard);
        std::thread::sleep(wait);
        guard = lock();
    }
}

/// Boots the title from the embedded XBE (spec 5.1, 6.3, 6.4). Installs the host fault handler,
/// resumes the title from its snapshot if it has one, else boots it, writes
/// `start = {esp, entry, fs_base, restored}`, starts the host-side threads, returns the guest base
/// `G`, or null on failure after printing why. After a resume `start` is `{0, 0, 0, 1}`, and the
/// resumed threads run on their own host threads. The XBE bytes stay valid for the life of the
/// process.
///
/// # Safety
///
/// `xbe` must point to `len` readable bytes for the life of the process; `start` must point to
/// four writable `u32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_boot(xbe: *const u8, len: usize, start: *mut u32) -> *mut u8 {
    let bytes: &'static [u8] = if xbe.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(xbe, len) }
    };
    let _ = XBE.set(bytes);
    crate::fault::install();
    if let Some(base) = crate::hibernate::restore::try_restore(bytes) {
        if !start.is_null() {
            unsafe {
                for word in 0..4 {
                    start.add(word).write(u32::from(word == 3));
                }
            }
        }
        start_preempt_timer();
        crate::host::start_timers();
        return base;
    }
    let Some((base, esp, entry)) = boot_title(bytes) else {
        return std::ptr::null_mut();
    };
    if !start.is_null() {
        unsafe {
            start.write(esp);
            start.add(1).write(entry);
            start.add(2).write(KPCR);
            start.add(3).write(0);
        }
    }
    // The process's main thread runs the boot thread.
    LIVE.fetch_add(1, Ordering::SeqCst);
    COUNTED.set(true);
    start_preempt_timer();
    crate::host::start_timers();
    base
}

/// Starts the timer that raises the preempt flag at the end of every time slice.
fn start_preempt_timer() {
    let slice = Duration::from_micros(slice_us());
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(slice);
            anyxbox_preempt_flag.store(1, Ordering::Relaxed);
        }
    });
}

/// The thread that was current when the title froze, which runs the scheduler first after a
/// restore; 0 once it has.
static RESTORED_CURRENT: AtomicU32 = AtomicU32::new(0);

/// The host thread of a resumed guest thread.
extern "C" fn resume_thread(arg: *mut libc::c_void) -> *mut libc::c_void {
    let (tid, epoch) = *unsafe { Box::from_raw(arg.cast::<(u32, u64)>()) };
    ME.set(tid);
    MY_EPOCH.set(epoch);
    COUNTED.set(true);
    unsafe { anyxbox_resume_body(tid, KPCR) };
    uncount();
    std::ptr::null_mut()
}

/// Installs a kernel restored from a snapshot, and starts a host thread for each guest thread it
/// had running (`cpus`); the threads not yet started start as new ones do. Nobody runs until the
/// thread that was current resumes and schedules.
pub(crate) fn install_restored(mut kernel: Kernel, cpus: Vec<crate::hibernate::cpu::GuestCpu>) {
    let epoch = EPOCH.load(Ordering::SeqCst);
    let starting: Vec<u32> = kernel
        .threads
        .iter()
        .filter(|t| t.state != State::Done && t.park == Park::Start)
        .map(|t| t.id)
        .filter(|id| !kernel.unstarted.contains(id))
        .collect();
    kernel.unstarted.extend(starting);
    let resumed: Vec<u32> = cpus.iter().map(|cpu| cpu.tid).collect();
    crate::hibernate::cpu::set_resumes(cpus);
    let Some(first) = first_schedule(kernel.current, &resumed) else {
        eprintln!("anyxbox: the snapshot has no thread to resume");
        crate::hibernate::discard();
        std::process::exit(1);
    };
    if first != kernel.current {
        // Nobody would ever schedule: the snapshot is suspect, so the next launch boots afresh,
        // but this one carries on with the first resumed thread scheduling.
        eprintln!(
            "anyxbox: the snapshot's current thread {} is not among those it resumes",
            kernel.current
        );
        crate::hibernate::discard();
    }
    RESTORED_CURRENT.store(first, Ordering::SeqCst);
    BATON.handoff(0, 0);
    *lock() = Some(kernel);
    for tid in resumed {
        let arg = Box::into_raw(Box::new((tid, epoch)));
        if !spawn_guest_thread(resume_thread, arg.cast()) {
            eprintln!("anyxbox: could not start a host thread to resume guest thread {tid}");
            crate::hibernate::discard();
            std::process::exit(1);
        }
    }
}

/// The resumed thread that schedules first: the one that was current, else the first resumed;
/// `None` when no thread is resumed.
fn first_schedule(current: u32, resumed: &[u32]) -> Option<u32> {
    if resumed.contains(&current) {
        Some(current)
    } else {
        resumed.first().copied()
    }
}

/// Called by a resumed thread at its innermost frame: waits for its turn (the thread that was
/// current schedules first), then says how to finish the contact it was parked in.
///
/// # Safety
///
/// `out` must point to a writable `anyxbox_resume`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_resume_wait(out: *mut crate::hibernate::cpu::Resume) {
    let me = ME.get();
    let resume = if RESTORED_CURRENT
        .compare_exchange(me, 0, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        // A resumed title with a picture to fade in stays frozen until the fade has finished.
        crate::hibernate::wait_until_released();
        let mut guard = lock();
        if let Some(kernel) = guard.as_mut() {
            crate::hibernate::start_resumed(kernel);
        }
        reschedule(guard, me)
    } else {
        BATON
            .wait_until(me, MY_EPOCH.get())
            .unwrap_or_else(|| leave())
    };
    let park = with_kernel(|k| k.threads.get(me).map(|t| t.park))
        .flatten()
        .unwrap_or(Park::Poll);
    unsafe { out.write(crate::hibernate::cpu::Resume::of(park, resume)) };
}

/// Boots the title over new guest memory and installs its kernel. Returns the guest base, the
/// boot thread's ESP, and the entry point; `None` after printing why it failed.
fn boot_title(bytes: &[u8]) -> Option<(*mut u8, u32, u32)> {
    let parsed = match xbe::Xbe::parse(bytes.to_vec()) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("anyxbox_boot: {error:?}");
            return None;
        }
    };
    crate::hibernate::set_title(parsed.certificate().title_id());
    let Some(memory) = crate::mem::Memory::new() else {
        eprintln!("anyxbox_boot: could not reserve the guest space");
        return None;
    };
    let base = memory.base();
    crate::mem::set_guest_base(base);
    crate::fault::set_guest_base(base);
    let kernel = match crate::boot::boot(memory, &parsed) {
        Ok(kernel) => kernel,
        Err(error) => {
            eprintln!("anyxbox: cannot boot the title: {error}");
            return None;
        }
    };
    // The boot thread starts on its TLS block when it has one, else at the TIB stack base (spec
    // 6.3 step 7): the TLS block sits right below the stack base, so starting above it would let
    // the first pushes overwrite the thread's TLS.
    let (entry, esp) = kernel.threads.get(1).map_or((0, 0), |t| t.start);
    *lock() = Some(kernel);
    Some((base, esp, entry))
}

/// The host thread that runs the entry point again after a reset.
extern "C" fn entry_thread(arg: *mut libc::c_void) -> *mut libc::c_void {
    let (base, esp, epoch) = *unsafe { Box::from_raw(arg.cast::<(usize, u32, u64)>()) };
    ME.set(1);
    MY_EPOCH.set(epoch);
    COUNTED.set(true);
    unsafe { anyxbox_entry_body(base as *mut u8, esp, KPCR) };
    uncount();
    std::ptr::null_mut()
}

/// The title's executable, once booted.
pub(crate) fn xbe() -> Option<&'static [u8]> {
    XBE.get().copied()
}

/// Resets the console (spec 7.8): the title stops (see [`stop_title`]) and boots again with its
/// entry point on a new host thread.
pub(crate) fn reset_console() {
    let epoch = stop_title();
    let Some((base, esp, _)) = XBE.get().and_then(|bytes| boot_title(bytes)) else {
        eprintln!("anyxbox: the reset could not boot the title again");
        std::process::exit(1);
    };
    let arg = Box::into_raw(Box::new((base as usize, esp, epoch)));
    if !spawn_guest_thread(entry_thread, arg.cast()) {
        eprintln!("anyxbox: could not start the entry point after the reset");
        std::process::exit(1);
    }
}

/// Takes the console back to `snapshot`, a perma snapshot of the running title: the title stops
/// as for a reset, and resumes from the snapshot instead of booting. Returns the picture the
/// window should fade in, with the title held until it has.
pub(crate) fn rewind_console(
    snapshot: crate::hibernate::file::Snapshot,
) -> Option<crate::hibernate::file::Picture> {
    stop_title();
    match crate::hibernate::restore::rewind(snapshot) {
        Ok(picture) => picture,
        Err(error) => {
            eprintln!("anyxbox: cannot go back to the perma snapshot: {error}");
            std::process::exit(1);
        }
    }
}

/// Stops the title for a reset or a rewind (spec 7.8): every guest thread leaves at its next
/// contact with the runtime (parked ones are woken to find out); after up to 5 s for them, the
/// saves are kept and the kernel and guest memory dropped. If threads do not leave in time, the
/// program runs itself again instead. Returns the new boot epoch.
fn stop_title() -> u64 {
    let epoch = EPOCH.fetch_add(1, Ordering::SeqCst) + 1;
    let deadline = Instant::now() + Duration::from_secs(5);
    while LIVE.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        BATON.wake_all();
        std::thread::sleep(Duration::from_millis(1));
    }
    let old = lock().take();
    if let Some(mut kernel) = old {
        kernel.fs.persist();
        if LIVE.load(Ordering::SeqCst) > 0 {
            eprintln!("anyxbox: guest threads did not stop for the reset; starting again");
            restart_process();
        }
        drop(kernel);
        crate::hibernate::cpu::forget_all();
    }
    BATON.handoff(1, 0);
    PACE_PENDING.store(0, Ordering::Relaxed);
    *NEXT_FRAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    epoch
}

/// Runs the program again with the same arguments, replacing this process.
fn restart_process() -> ! {
    use std::os::unix::process::CommandExt;
    let error = std::env::current_exe().map(|exe| {
        std::process::Command::new(exe)
            .args(std::env::args_os().skip(1))
            .exec()
    });
    eprintln!("anyxbox: could not start again: {error:?}");
    std::process::exit(1);
}

/// Runs the kernel call or XDK replacement at trap address `target` (spec 5.1).
///
/// # Safety
///
/// `out` must point to a writable `anyxbox_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_native_call(
    target: u32,
    esp: u32,
    ecx: u32,
    edx: u32,
    out: *mut Result,
) {
    leave_if_reset();
    let me = ME.get();
    let mut result = Result::default();
    let mut guard = lock();
    if crate::hibernate::freezing() && guard.is_some() {
        guard = park_for_hibernation(guard, me, Park::Call { target });
    }
    let Some(kernel) = guard.as_mut() else {
        eprintln!("anyxbox_native_call before boot");
        unsafe { out.write(result) };
        return;
    };
    let dispatched = crate::kernel::dispatch(kernel, target, esp, ecx, edx);
    start_host_threads(kernel);
    match dispatched {
        Dispatch::Done {
            eax,
            edx,
            esp,
            wide,
        } => {
            result = Result {
                state: 1,
                eax,
                esp,
                edx,
            };
            // Every 256th call returning a 32-bit value lets another ready thread of equal or
            // higher priority run; a call that made a higher-priority thread ready (a resume, a
            // wake, a priority change) gives way to it at once.
            if !wide
                && ((kernel.calls.is_multiple_of(CALLS_PER_SLICE)
                    && kernel.threads.ready_at_slice_end(me))
                    || kernel.threads.preempted(me))
            {
                if let Some(thread) = kernel.threads.get_mut(me) {
                    thread.resume = eax;
                    thread.park = Park::Native {
                        esp,
                        edx,
                        eax: None,
                    };
                }
                result.eax = reschedule(guard, me);
            }
        }
        Dispatch::Block { edx, esp } => {
            if let Some(thread) = kernel.threads.get_mut(me) {
                thread.park = Park::Native {
                    esp,
                    edx,
                    eax: None,
                };
            }
            let eax = reschedule(guard, me);
            result = Result {
                state: 1,
                eax,
                esp,
                edx,
            };
        }
        Dispatch::Yield { edx, esp } => {
            if let Some(thread) = kernel.threads.get_mut(me) {
                thread.resume = 0;
                thread.park = Park::Native {
                    esp,
                    edx,
                    eax: Some(0),
                };
            }
            reschedule(guard, me);
            result = Result {
                state: 1,
                eax: 0,
                esp,
                edx,
            };
        }
        Dispatch::ThreadEnd => result.state = 3,
        Dispatch::Halt => {
            kernel.fs.persist();
            crate::hibernate::discard();
            std::process::exit(0);
        }
        Dispatch::Fail => result.state = 0,
    }
    unsafe { out.write(result) };
}

/// Ends the running guest thread and hands the baton on. A second end does nothing.
fn end_thread(result: Option<u32>) {
    leave_if_reset();
    let me = ME.get();
    let mut guard = lock();
    let Some(kernel) = guard.as_mut() else {
        return;
    };
    if kernel
        .threads
        .get(me)
        .is_none_or(|t| t.state == State::Done)
    {
        return;
    }
    kernel.end_thread(me, result);
    reschedule(guard, me);
    // The thread runs no more guest code.
    uncount();
}

/// The running guest thread has finished (spec 5.1).
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_thread_end() {
    end_thread(None);
}

/// As `anyxbox_thread_end`, and the start routine returned `eax` (spec 5.1).
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_thread_end_with(eax: u32) {
    end_thread(Some(eax));
}

/// Runs the window on the process's main thread until quit, then exits the process. Without a
/// window, parks forever (spec 5.1).
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_main_loop() {
    crate::host::main_loop();
}

/// `ANYXBOX_TRACE_MMIO`: GPU register accesses still to print (the first 2 million).
static MMIO_TRACE_LEFT: AtomicU64 = AtomicU64::new(u64::MAX);

/// Prints a GPU register access while `ANYXBOX_TRACE_MMIO` has some left.
fn trace_mmio(kind: char, addr: u32, size: u32, value: u32) {
    let mut left = MMIO_TRACE_LEFT.load(Ordering::Relaxed);
    if left == u64::MAX {
        left = if std::env::var_os("ANYXBOX_TRACE_MMIO").is_some() {
            2_000_000
        } else {
            0
        };
        MMIO_TRACE_LEFT.store(left, Ordering::Relaxed);
    }
    if left > 0 {
        MMIO_TRACE_LEFT.store(left - 1, Ordering::Relaxed);
        eprintln!(
            "mmio {kind} {addr:#010x}/{size} {value:#010x} thread {}",
            ME.get()
        );
    }
}

/// Reads in a row of one GPU register that found the same value, for the preemption rule.
static MMIO_READS: AtomicU64 = AtomicU64::new(0);
/// That register and value: the address in the high half.
static MMIO_LAST_READ: AtomicU64 = AtomicU64::new(0);

/// A guest load of `size` bytes in `0xFD000000..0xFE000000` (spec 5.1). Services time first.
/// Titles poll GPU registers in loops without kernel calls, so a thread that has read one
/// register 64 times in a row and found it unchanged lets another ready thread run. Reads that
/// find what they wait for at once (Direct3D flushes a cache and checks it many times a frame)
/// do not give way: that only handed the title's turn away.
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_mmio_read(addr: u32, size: u32) -> u32 {
    leave_if_reset();
    let me = ME.get();
    let mut guard = lock();
    let mut value = 0;
    if let Some(kernel) = guard.as_mut() {
        kernel.service_time();
        value = kernel
            .gpu
            .read(addr.wrapping_sub(crate::mem::MMIO_BASE), size);
        trace_mmio('R', addr, size, value);
        start_host_threads(kernel);
        if !crate::hibernate::freezing()
            && ((polled_unchanged(addr, value) && kernel.threads.ready_at_slice_end(me))
                || kernel.threads.preempted(me))
        {
            MMIO_READS.store(0, Ordering::Relaxed);
            if let Some(thread) = kernel.threads.get_mut(me) {
                thread.park = Park::Busy;
            }
            reschedule(guard, me);
        }
    }
    value
}

/// Counts a read of `addr` that found `value`; whether the reads of one register in a row that
/// found it unchanged have reached [`MMIO_READS_PER_SLICE`].
fn polled_unchanged(addr: u32, value: u32) -> bool {
    let read = u64::from(addr) << 32 | u64::from(value);
    let repeats = if MMIO_LAST_READ.swap(read, Ordering::Relaxed) == read {
        MMIO_READS.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        MMIO_READS.store(0, Ordering::Relaxed);
        0
    };
    repeats >= MMIO_READS_PER_SLICE
}

/// The length of a time slice, in microseconds: `ANYXBOX_SLICE_US`, 250 by default. A thread
/// that spins on memory another thread changes (a title waiting on its vertical-blank count)
/// holds the turn until its slice ends, so the slice bounds how late that other thread runs.
fn slice_us() -> u64 {
    static US: OnceLock<u64> = OnceLock::new();
    *US.get_or_init(|| {
        std::env::var("ANYXBOX_SLICE_US")
            .ok()
            .and_then(|text| text.parse().ok())
            .filter(|&us| us > 0)
            .unwrap_or(SLICE_US)
    })
}

/// Whether frames are held to the refresh rate: unless `ANYXBOX_UNPACED` is set, read once.
fn paced() -> bool {
    static PACED: OnceLock<bool> = OnceLock::new();
    *PACED.get_or_init(|| std::env::var_os("ANYXBOX_UNPACED").is_none())
}

/// A guest store in `0xFD000000..0xFE000000` (spec 5.1): the GPU model takes it, which may run
/// the FIFO and raise an interrupt.
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_mmio_write(addr: u32, size: u32, value: u32) {
    leave_if_reset();
    let mut guard = lock();
    let Some(kernel) = guard.as_mut() else {
        return;
    };
    let flips = kernel.gpu.flips;
    trace_mmio('W', addr, size, value);
    let offset = addr.wrapping_sub(crate::mem::MMIO_BASE);
    kernel.gpu.write(offset, size, value, &kernel.memory);
    kernel.service_interrupts();
    start_host_threads(kernel);
    let me = ME.get();
    let in_dpc = kernel.threads.get(me).is_some_and(|t| t.dpc != 0);
    // A frame that ends on a DPC thread (Direct3D's interrupt routine resuming the FIFO) is not
    // held there: the hold goes to the next GPU write of a thread that is not running a DPC,
    // which is the title's own push-buffer kick (spec 7.7, matching the reference's behaviour).
    let frames = kernel.gpu.flips - flips;
    if frames > 0 {
        crate::host::record::apply_due(kernel, kernel.gpu.flips);
    }
    if in_dpc {
        PACE_PENDING.fetch_add(frames, Ordering::Relaxed);
        return;
    }
    let holds = frames + PACE_PENDING.swap(0, Ordering::Relaxed);
    if holds > 0 && paced() && !crate::hibernate::freezing() {
        // One refresh interval per frame ended since the last hold.
        for _ in 1..holds {
            advance_frame_deadline();
        }
        pace(guard);
    }
}

/// Frames that ended on DPC threads whose holds are still to be applied.
static PACE_PENDING: AtomicU64 = AtomicU64::new(0);

/// The deadline of the next frame (host time), for pacing.
static NEXT_FRAME: Mutex<Option<Instant>> = Mutex::new(None);

/// The deadline of the frame that just ended, moving the next one a refresh interval past the
/// later of it and now (a late frame does not wait).
fn advance_frame_deadline() -> Instant {
    let interval = Duration::from_nanos(if crate::kernel::time::refresh_is_50() {
        20_000_000
    } else {
        16_666_667
    });
    let now = Instant::now();
    let mut next = NEXT_FRAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let deadline = next.unwrap_or(now);
    *next = Some(deadline.max(now) + interval);
    deadline
}

/// Holds the thread that ended a frame until the next refresh interval after the previous frame
/// (spec 7.7), servicing time and letting ready threads run meanwhile, sleeping at most 1 ms at a
/// time. A late frame does not wait; the next deadline is one interval after the later of the old
/// deadline and now.
fn pace(mut guard: MutexGuard<'static, Option<Kernel>>) {
    if crate::hibernate::freezing() {
        return;
    }
    let deadline = advance_frame_deadline();
    let me = ME.get();
    while Instant::now() < deadline {
        let Some(kernel) = guard.as_mut() else {
            return;
        };
        kernel.service_time();
        start_host_threads(kernel);
        // The held thread waits rather than gives way: a thread of any priority may run.
        if kernel.threads.other_runnable(me) {
            if let Some(thread) = kernel.threads.get_mut(me) {
                thread.resume = 0;
                thread.park = Park::Busy;
            }
            reschedule_with(guard, me, true);
        } else {
            drop(guard);
            std::thread::sleep(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
        guard = lock();
        if reset_since_start() {
            drop(guard);
            leave();
        }
        if crate::hibernate::freezing() {
            return;
        }
    }
}

/// Set at the end of every time slice ([`slice_us`]) by a timer thread; lifted code polls it on
/// loop back-edges (spec 5.1).
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static anyxbox_preempt_flag: AtomicI32 = AtomicI32::new(0);

/// Called by lifted code when the flag is set: clears it, services time, and lets another ready
/// thread of equal or higher priority run (spec 7.4).
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_preempt() {
    anyxbox_preempt_flag.store(0, Ordering::Relaxed);
    leave_if_reset();
    let me = ME.get();
    let mut guard = lock();
    if crate::hibernate::freezing() && guard.is_some() {
        drop(park_for_hibernation(guard, me, Park::Poll));
        return;
    }
    let Some(kernel) = guard.as_mut() else {
        return;
    };
    kernel.service_time();
    start_host_threads(kernel);
    if kernel.threads.ready_at_slice_end(me) {
        if let Some(thread) = kernel.threads.get_mut(me) {
            thread.resume = 0;
            thread.park = Park::Poll;
        }
        reschedule(guard, me);
    }
}

/// Called by the lifted code when it traps: with `ANYXBOX_DUMP=ADDR:LEN,...` (hex), prints the
/// guest memory it names, what the title left in its data as it stopped.
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_dump_requested() {
    use std::fmt::Write as _;
    let Ok(spec) = std::env::var("ANYXBOX_DUMP") else {
        return;
    };
    // A trap can come while another thread holds the kernel; then nothing is printed.
    let Ok(guard) = KERNEL.try_lock() else {
        return;
    };
    let Some(kernel) = guard.as_ref() else {
        return;
    };
    for (address, len) in dump_ranges(&spec) {
        for row in (0..len.min(0x1_0000)).step_by(16) {
            let at = address.wrapping_add(row);
            let Ok(bytes) = kernel.read(at, 16.min(len - row)) else {
                eprintln!("anyxbox: dump {at:08x}: not mapped");
                break;
            };
            let mut line = format!("anyxbox: dump {at:08x}:");
            for byte in bytes {
                let _ = write!(line, " {byte:02x}");
            }
            eprintln!("{line}");
        }
    }
}

/// The `(address, length)` pairs of `ADDR:LEN,ADDR:LEN` (hex, `0x` optional).
fn dump_ranges(spec: &str) -> Vec<(u32, u32)> {
    spec.split(',')
        .filter_map(|range| {
            let (address, len) = range.split_once(':')?;
            let hex =
                |text: &str| u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok();
            Some((hex(address)?, hex(len)?))
        })
        .collect()
}

/// The trace function, for debug builds of lifted code only (spec 5.1).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn anyxbox_trace(
    at: u32,
    eax: u32,
    ecx: u32,
    edx: u32,
    ebx: u32,
    esp: u32,
    ebp: u32,
    esi: u32,
    edi: u32,
) {
    let _ = (at, eax, ecx, edx, ebx, esp, ebp, esi, edi);
}

#[cfg(test)]
mod tests {
    use super::{MMIO_READS_PER_SLICE, dump_ranges, first_schedule, polled_unchanged};

    #[test]
    fn the_current_thread_schedules_first_after_a_restore_else_the_first_resumed() {
        assert_eq!(first_schedule(4, &[1, 4]), Some(4));
        assert_eq!(first_schedule(9, &[1, 4]), Some(1));
        assert_eq!(first_schedule(9, &[]), None);
    }

    #[test]
    fn dump_ranges_are_hex_pairs_and_bad_ones_are_skipped() {
        assert_eq!(
            dump_ranges("0x1000:20, 2000:4,bad,3000"),
            [(0x1000, 0x20), (0x2000, 4)]
        );
    }

    #[test]
    fn only_a_register_read_unchanged_many_times_in_a_row_gives_way() {
        // Reads that keep finding something new never give way.
        for n in 0..200u32 {
            assert!(!polled_unchanged(0xFD40_0700, n));
        }
        // Polling one register that stays the same gives way after 64 repeats.
        assert!(
            !polled_unchanged(0xFD40_0700, 7),
            "the first read of the value"
        );
        for _ in 1..MMIO_READS_PER_SLICE {
            assert!(!polled_unchanged(0xFD40_0700, 7));
        }
        assert!(polled_unchanged(0xFD40_0700, 7));
        // Another register starts the count again.
        assert!(!polled_unchanged(0xFD40_0704, 7));
    }
}
