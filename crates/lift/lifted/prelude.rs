// The Rust every lifted unit starts with: the guest state and the helpers.
//
// A unit is a `no_std` crate compiled to an object on its own and linked with the runtime, which
// carries its own copy of the standard library: so nothing here may need one. Nothing here may
// panic either (no indexing with a variable, no `/` or `%`, no `unwrap`), because a panic would
// name symbols of the standard library the object does not have.
//
// Every lifted function takes the guest CPU of its host thread as its argument: each guest thread
// runs on a host thread of its own, and the pointer is passed down instead of looked up.

#[allow(unused_imports)]
use core::ptr;

#[allow(unused_imports)]
use ext::*;

/// What the lifted code calls and reads that another unit, the glue or the runtime defines. A unit
/// that defines one of them itself shadows the import.
#[allow(dead_code, non_upper_case_globals)]
mod ext {
    use super::{Cpu, Timespec};
    unsafe extern "C" {
        /// The guest memory's base.
        pub static mut G: *mut u8;
        /// The guest's FS segment base, the KPCR.
        pub static mut FS_BASE: u32;
        /// Raised by the runtime every millisecond: loops poll it on their backward branches.
        pub static anyxbox_preempt_flag: i32;
        /// The time-stamp counter's offset from host time, which a resume sets.
        pub static anyxbox_tsc_offset: i64;
        pub fn anyxbox_preempt();
        pub fn anyxbox_mmio_read(a: u32, size: u32) -> u32;
        pub fn anyxbox_mmio_write(a: u32, size: u32, v: u32);
        pub fn anyxbox_trace(
            at: u32,
            eax: u32,
            ecx: u32,
            edx: u32,
            ebx: u32,
            esp: u32,
            ebp: u32,
            esi: u32,
            edi: u32,
        );
        /// Defined by the program that links the lifted code: prints what it knows of the guest.
        pub fn lift_trap_hook(c: *mut Cpu);
        /// A kernel call or a missing function: the glue completes it.
        pub fn lift_native(c: *mut Cpu, target: u32, at: u32);
        /// The innermost frame of a resumed thread: finishes the contact it was parked in.
        pub fn lift_resume_innermost(c: *mut Cpu);
        /// The dispatcher's, which maps a guest address to its function.
        pub fn rt_call(c: *mut Cpu, target: u32, at: u32);
        pub fn lift_resume_call(c: *mut Cpu);
        pub fn lift_resume_root(c: *mut Cpu);

        pub fn abort() -> !;
        pub fn write(fd: i32, buf: *const u8, len: usize) -> isize;
        pub fn calloc(count: usize, size: usize) -> *mut u8;
        pub fn free(p: *mut u8);
        pub fn mmap(
            at: *mut u8,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut u8;
        pub fn munmap(at: *mut u8, len: usize) -> i32;
        pub fn mprotect(at: *mut u8, len: usize, prot: i32) -> i32;
        pub fn clock_gettime(clock: u32, ts: *mut Timespec) -> i32;
        pub fn floor(x: f64) -> f64;
        pub fn ceil(x: f64) -> f64;
        pub fn trunc(x: f64) -> f64;
        pub fn nearbyint(x: f64) -> f64;
        pub fn fmod(x: f64, y: f64) -> f64;
        pub fn remquo(x: f64, y: f64, quotient: *mut i32) -> f64;
        pub fn ldexp(x: f64, e: i32) -> f64;
        pub fn frexp(x: f64, e: *mut i32) -> f64;
        pub fn sin(x: f64) -> f64;
        pub fn cos(x: f64) -> f64;
        pub fn tan(x: f64) -> f64;
        pub fn atan2(y: f64, x: f64) -> f64;
        pub fn sqrt(x: f64) -> f64;
        pub fn sqrtf(x: f32) -> f32;
        pub fn log2(x: f64) -> f64;
        pub fn expm1(x: f64) -> f64;
    }
}

#[repr(C)]
pub struct Timespec {
    pub sec: i64,
    pub nsec: i64,
}

/// SSE: one of the eight XMM registers.
#[repr(C)]
#[derive(Clone, Copy)]
pub union Xmm {
    pub f: [f32; 4],
    pub u: [u32; 4],
    pub q: [u64; 2],
}

/// Where each host frame of a guest thread is: the lifted function running in it, and the site it
/// was last at (a call, a call that returns as soon as its callee does, or a loop's poll). A
/// hibernated thread is resumed by calling down these frames again.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Frame {
    pub function: u32,
    pub kind: u32,
    pub site: u32,
}

const K_CALL: u32 = 0;
const K_RET: u32 = 1;
const K_POLL: u32 = 2;

/// The guest CPU of one host thread. The runtime saves and restores its bytes, so its layout is
/// part of the snapshot format.
#[repr(C)]
pub struct Cpu {
    pub eax: u32,
    pub ecx: u32,
    pub edx: u32,
    pub ebx: u32,
    pub esp: u32,
    pub ebp: u32,
    pub esi: u32,
    pub edi: u32,
    pub cf: u8,
    pub zf: u8,
    pub sf: u8,
    pub of: u8,
    pub pf: u8,
    pub df: u8,
    pub fpu_top: u32,
    pub fpu_cc: u32,
    /// The x87 control word and MXCSR, each kept as its difference from the value the console's
    /// kernel starts a thread with (0x027F: 53-bit precision, every exception masked; 0x1F80), so
    /// a zeroed CPU is a new thread's.
    pub fpu_cwx: u32,
    pub mxcsr_x: u32,
    /// The x87 register stack holds doubles: 64-bit precision where the title keeps 80-bit
    /// extended values, which matches for values that fit in a double.
    pub fpu: [f64; 8],
    pub xmm: [Xmm; 8],
    pub mm: [u64; 8],
    /// The frame records and the depth of the current frame; while resuming, the number of frames
    /// to resume down.
    pub frames: *mut Frame,
    pub depth: u32,
    pub resuming: u32,
    pub resume_depth: u32,
}

/// A `setjmp` context: the buffer, the address `longjmp` returns to, and the frame and site that
/// hold it. The runtime saves the first four words.
#[repr(C)]
pub struct Jmp {
    pub buf: u32,
    pub resume: u32,
    pub depth: u32,
    pub site: u32,
    pub active: u32,
}

/// A context as a hibernation saved it, which a resumed thread registers again.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JmpSaved {
    pub buf: u32,
    pub resume: u32,
    pub depth: u32,
    pub site: u32,
}

const JMP_ROOM: u32 = 256;

/// A host thread's guest state: the CPU first, then the `setjmp` contexts. `jumping` is the
/// context a `longjmp` is on its way to: every lifted function returns until the one that holds it.
#[repr(C)]
pub struct Thread {
    pub cpu: Cpu,
    pub jumping: *mut Jmp,
    pub jmp_top: u32,
    pub jmp_stack: [*mut Jmp; JMP_ROOM as usize],
    pub restore_count: u32,
    pub restore: [JmpSaved; JMP_ROOM as usize],
}

#[inline(always)]
fn thread(c: *mut Cpu) -> *mut Thread {
    c.cast()
}

/// A lifted function. It returns the function it tail-calls, which its caller runs next, so a
/// loop through several functions keeps one host frame.
pub type Lifted = unsafe extern "C" fn(*mut Cpu) -> Next;

#[repr(transparent)]
pub struct Next(pub Option<Lifted>);

const END: Next = Next(None);

#[inline(always)]
fn to(f: Lifted) -> Next {
    Next(Some(f))
}

/// Runs `f` and the functions it tail-calls.
#[inline]
unsafe fn run(c: *mut Cpu, f: Lifted) {
    unsafe {
        let mut next = f(c);
        while let Some(f) = next.0 {
            next = f(c);
        }
    }
}

// Writes for traps, without the formatting machinery of `core`.
unsafe fn put(text: &[u8]) {
    unsafe {
        write(2, text.as_ptr(), text.len());
    }
}

unsafe fn put_hex(v: u32) {
    let mut text = *b"0x00000000";
    let mut shift = 28u32;
    for digit in text.iter_mut().skip(2) {
        *digit = b"0123456789abcdef"
            .get(((v >> shift) & 15) as usize)
            .copied()
            .unwrap_or(b'?');
        shift = shift.wrapping_sub(4);
    }
    unsafe { put(&text) }
}

/// Stops the program at `at` with `what`, after the glue prints what it knows of the guest.
#[cold]
#[inline(never)]
unsafe fn lift_trap(c: *mut Cpu, at: u32, what: &[u8]) -> ! {
    unsafe {
        put(b"lifted code trapped at ");
        put_hex(at);
        put(b": ");
        put(what);
        put(b"\n");
        lift_trap_hook(c);
        abort()
    }
}

/// Room for this many frames, with a guard page after them, so a runaway depth faults.
const FRAMES: usize = 131_072;
const FRAMES_LEN: usize = FRAMES * core::mem::size_of::<Frame>();
const FRAMES_GUARD: usize = 16384;

/// A new host thread's guest state, zeroed, with its frame records.
#[allow(dead_code)]
unsafe fn lift_thread_new() -> *mut Cpu {
    unsafe {
        let t = calloc(1, core::mem::size_of::<Thread>()).cast::<Thread>();
        // PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON.
        let at = mmap(ptr::null_mut(), FRAMES_LEN + FRAMES_GUARD, 3, 0x1002, -1, 0);
        if t.is_null() || at as isize == -1 {
            put(b"no memory for the guest thread\n");
            abort();
        }
        mprotect(at.add(FRAMES_LEN), FRAMES_GUARD, 0);
        (*t).cpu.frames = at.cast();
        t.cast()
    }
}

/// Gives back what `lift_thread_new` made.
#[allow(dead_code)]
unsafe fn lift_thread_free(c: *mut Cpu) {
    unsafe {
        if c.is_null() {
            return;
        }
        munmap((*c).frames.cast(), FRAMES_LEN + FRAMES_GUARD);
        free(c.cast());
    }
}

#[inline(always)]
unsafe fn frame(c: *mut Cpu) -> *mut Frame {
    unsafe { (*c).frames.add((*c).depth as usize) }
}

#[inline(always)]
unsafe fn enter(c: *mut Cpu, function: u32) {
    unsafe { (*frame(c)).function = function }
}

#[inline(always)]
unsafe fn site(c: *mut Cpu, s: u32, kind: u32) {
    unsafe {
        let f = frame(c);
        (*f).site = s;
        (*f).kind = kind;
    }
}

/// While resuming: the site of this frame, as a `match` key.
#[inline(always)]
unsafe fn resume_key(c: *mut Cpu) -> u64 {
    unsafe {
        let f = frame(c);
        (u64::from((*f).site) << 2) | u64::from((*f).kind)
    }
}

#[inline(always)]
unsafe fn jumping(c: *mut Cpu) -> bool {
    unsafe { !(*thread(c)).jumping.is_null() }
}

/// A call to lifted function `f` from site `s`: true when a `longjmp` is passing through.
#[inline(always)]
unsafe fn call(c: *mut Cpu, s: u32, kind: u32, f: Lifted) -> bool {
    unsafe {
        site(c, s, kind);
        (*c).depth = (*c).depth.wrapping_add(1);
        run(c, f);
        (*c).depth = (*c).depth.wrapping_sub(1);
        jumping(c)
    }
}

/// A call through the dispatcher to guest address `target`.
#[inline(always)]
unsafe fn call_rt(c: *mut Cpu, s: u32, kind: u32, target: u32, at: u32) -> bool {
    unsafe {
        site(c, s, kind);
        (*c).depth = (*c).depth.wrapping_add(1);
        rt_call(c, target, at);
        (*c).depth = (*c).depth.wrapping_sub(1);
        jumping(c)
    }
}

/// Resuming a frame's call: the frame below resumes at its own site.
#[inline(always)]
unsafe fn resume_call(c: *mut Cpu) -> bool {
    unsafe {
        lift_resume_call(c);
        jumping(c)
    }
}

/// Resuming at a poll, the innermost frame.
#[inline(always)]
unsafe fn resume_poll(c: *mut Cpu) -> bool {
    unsafe {
        lift_resume_innermost(c);
        jumping(c)
    }
}

/// Time slices: the runtime raises the flag every millisecond, and loops check it on their
/// backward branches, so a thread that spins without kernel calls still lets time pass and others
/// run. A poll that calls the runtime records its site first: a thread hibernated there resumes at
/// it. True when the thread must leave its frames.
#[inline(always)]
unsafe fn poll(c: *mut Cpu, s: u32) -> bool {
    unsafe {
        if ptr::read_volatile(&raw const anyxbox_preempt_flag) != 0 {
            site(c, s, K_POLL);
            anyxbox_preempt();
            return jumping(c);
        }
        false
    }
}

#[inline(always)]
#[allow(unused_variables)]
unsafe fn trace(c: *mut Cpu, at: u32) {
    #[cfg(lift_trace)]
    unsafe {
        anyxbox_trace(
            at,
            (*c).eax,
            (*c).ecx,
            (*c).edx,
            (*c).ebx,
            (*c).esp,
            (*c).ebp,
            (*c).esi,
            (*c).edi,
        );
    }
}

/// The title's `setjmp` and `longjmp` (MSVC's). A lifted function that calls `setjmp` holds a
/// context for each such call, registered on the thread's stack of contexts while the function
/// runs. `longjmp` ends with a jump to the saved return address in the middle of that function,
/// which the dispatcher turns into a jump to the context: every function returns until the one
/// that holds it, which goes on after its `setjmp` call.
pub struct Jmps<const N: usize> {
    c: *mut Cpu,
    j: [Jmp; N],
}

impl<const N: usize> Jmps<N> {
    #[inline(always)]
    fn new(c: *mut Cpu) -> Self {
        Self {
            c,
            j: [const {
                Jmp {
                    buf: 0,
                    resume: 0,
                    depth: 0,
                    site: 0,
                    active: 0,
                }
            }; N],
        }
    }

    #[inline(always)]
    fn at(&mut self, i: usize) -> *mut Jmp {
        // SAFETY: `i` is a site of the function, below `N`.
        unsafe { self.j.as_mut_ptr().add(i) }
    }

    /// The index of the context a `longjmp` is on its way to, if it is one of these: the jump
    /// ends here, and the contexts registered after it are gone.
    #[inline(never)]
    unsafe fn land(&mut self) -> Option<usize> {
        unsafe {
            let t = thread(self.c);
            let target = (*t).jumping;
            let index = (0..N).find(|&i| self.at(i) == target)?;
            (*t).jumping = ptr::null_mut();
            let mut top = (*t).jmp_top;
            while top > 0 {
                let j = *(*t).jmp_stack.as_ptr().add(top as usize - 1);
                if j == target {
                    break;
                }
                (*j).active = 0;
                top -= 1;
            }
            (*t).jmp_top = top;
            Some(index)
        }
    }
}

impl<const N: usize> Drop for Jmps<N> {
    fn drop(&mut self) {
        for i in 0..N {
            unsafe { jmp_exit(self.c, self.at(i)) }
        }
    }
}

unsafe fn jmp_register(c: *mut Cpu, j: *mut Jmp, buf: u32, resume: u32, s: u32) {
    unsafe {
        (*j).buf = buf;
        (*j).resume = resume;
        (*j).depth = (*c).depth;
        (*j).site = s;
        if (*j).active == 0 {
            let t = thread(c);
            if (*t).jmp_top >= JMP_ROOM {
                lift_trap(c, resume, b"setjmp nested too deeply");
            }
            *(*t).jmp_stack.as_mut_ptr().add((*t).jmp_top as usize) = j;
            (*t).jmp_top += 1;
            (*j).active = 1;
        }
    }
}

/// While resuming a frame: registers the context of `setjmp` site `s` again, if the hibernation
/// saved one for it.
unsafe fn jmp_reregister(c: *mut Cpu, j: *mut Jmp, s: u32) {
    unsafe {
        let t = thread(c);
        for i in 0..(*t).restore_count.min(JMP_ROOM) {
            let saved = *(*t).restore.as_ptr().add(i as usize);
            if saved.depth == (*c).depth && saved.site == s {
                jmp_register(c, j, saved.buf, saved.resume, s);
                return;
            }
        }
    }
}

unsafe fn jmp_exit(c: *mut Cpu, j: *mut Jmp) {
    unsafe {
        if (*j).active == 0 {
            return;
        }
        let t = thread(c);
        let stack = (*t).jmp_stack.as_mut_ptr();
        let top = (*t).jmp_top as usize;
        if let Some(i) = (0..top).rev().find(|&i| *stack.add(i) == j) {
            ptr::copy(stack.add(i + 1), stack.add(i), top - i - 1);
            (*t).jmp_top -= 1;
        }
        (*j).active = 0;
    }
}

/// Starts a jump to the `setjmp` that saved the address `target` (and, first choice, the buffer in
/// EDX, which is what `longjmp` jumps through), if there is one: true when it found one.
#[allow(dead_code)]
unsafe fn lift_longjmp(c: *mut Cpu, target: u32) -> bool {
    unsafe {
        let t = thread(c);
        for exact in [true, false] {
            for i in (0..(*t).jmp_top.min(JMP_ROOM) as usize).rev() {
                let j = *(*t).jmp_stack.as_ptr().add(i);
                if (*j).resume == target && (!exact || (*j).buf == (*c).edx) {
                    (*t).jumping = j;
                    return true;
                }
            }
        }
        false
    }
}

// Guest memory. The GPU's registers (0xFD000000-0xFDFFFFFF) are a device, not memory: the runtime
// serves them.
#[inline(always)]
fn is_mmio(a: u32) -> bool {
    (a >> 24) == 0xFD
}

#[cold]
unsafe fn mmio_read(a: u32, size: u32) -> u32 {
    unsafe { anyxbox_mmio_read(a, size) }
}

#[cold]
unsafe fn mmio_write(a: u32, size: u32, v: u32) {
    unsafe { anyxbox_mmio_write(a, size, v) }
}

#[inline(always)]
unsafe fn host(a: u32) -> *mut u8 {
    unsafe { G.add(a as usize) }
}

#[inline(always)]
unsafe fn rd8(a: u32) -> u32 {
    unsafe {
        if is_mmio(a) {
            return mmio_read(a, 1) & 0xFF;
        }
        u32::from(*host(a))
    }
}

#[inline(always)]
unsafe fn rd16(a: u32) -> u32 {
    unsafe {
        if is_mmio(a) {
            return mmio_read(a, 2) & 0xFFFF;
        }
        u32::from(ptr::read_unaligned(host(a).cast::<u16>()))
    }
}

#[inline(always)]
unsafe fn rd32(a: u32) -> u32 {
    unsafe {
        if is_mmio(a) {
            return mmio_read(a, 4);
        }
        ptr::read_unaligned(host(a).cast::<u32>())
    }
}

#[inline(always)]
unsafe fn wr8(a: u32, v: u32) {
    unsafe {
        if is_mmio(a) {
            return mmio_write(a, 1, v & 0xFF);
        }
        *host(a) = v as u8;
    }
}

#[inline(always)]
unsafe fn wr16(a: u32, v: u32) {
    unsafe {
        if is_mmio(a) {
            return mmio_write(a, 2, v & 0xFFFF);
        }
        ptr::write_unaligned(host(a).cast::<u16>(), v as u16);
    }
}

#[inline(always)]
unsafe fn wr32(a: u32, v: u32) {
    unsafe {
        if is_mmio(a) {
            return mmio_write(a, 4, v);
        }
        ptr::write_unaligned(host(a).cast::<u32>(), v);
    }
}

#[inline(always)]
unsafe fn ldx128(a: u32) -> Xmm {
    unsafe { ptr::read_unaligned(host(a).cast::<Xmm>()) }
}

#[inline(always)]
unsafe fn stx128(a: u32, v: Xmm) {
    unsafe { ptr::write_unaligned(host(a).cast::<Xmm>(), v) }
}

#[inline(always)]
unsafe fn ldf32x(a: u32) -> f32 {
    unsafe { ptr::read_unaligned(host(a).cast::<f32>()) }
}

#[inline(always)]
unsafe fn stf32x(a: u32, f: f32) {
    unsafe { ptr::write_unaligned(host(a).cast::<f32>(), f) }
}

#[inline(always)]
unsafe fn ldq64(a: u32) -> u64 {
    unsafe { ptr::read_unaligned(host(a).cast::<u64>()) }
}

#[inline(always)]
unsafe fn stq64(a: u32, v: u64) {
    unsafe { ptr::write_unaligned(host(a).cast::<u64>(), v) }
}

#[inline(always)]
unsafe fn push32(c: *mut Cpu, v: u32) {
    unsafe {
        (*c).esp = (*c).esp.wrapping_sub(4);
        wr32((*c).esp, v);
    }
}

#[inline(always)]
unsafe fn pop32(c: *mut Cpu) -> u32 {
    unsafe {
        let v = rd32((*c).esp);
        (*c).esp = (*c).esp.wrapping_add(4);
        v
    }
}

/// A string step: forwards, or backwards when the direction flag is set.
#[inline(always)]
unsafe fn dstep(c: *mut Cpu, n: u32) -> u32 {
    unsafe { if (*c).df != 0 { n.wrapping_neg() } else { n } }
}

/// The time-stamp counter: host time at the Xbox CPU's 733 MHz, plus the runtime's offset, which a
/// resume sets so the counter goes on from where the hibernated title left it.
unsafe fn lift_rdtsc() -> u64 {
    unsafe {
        let mut ts = Timespec { sec: 0, nsec: 0 };
        // CLOCK_MONOTONIC.
        clock_gettime(6, &raw mut ts);
        let nanos = (ts.sec as u64)
            .wrapping_mul(1_000_000_000)
            .wrapping_add(ts.nsec as u64);
        let host = nanos.wrapping_mul(733) / 1000;
        host.wrapping_add(ptr::read_volatile(&raw const anyxbox_tsc_offset) as u64)
    }
}

// The integer flags.
#[inline(always)]
fn mask_of(bits: u32) -> u32 {
    if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits).wrapping_sub(1)
    }
}

#[inline(always)]
fn sign_of(bits: u32) -> u32 {
    1u32 << (bits - 1)
}

#[inline(always)]
fn b(v: bool) -> u8 {
    u8::from(v)
}

#[inline(always)]
unsafe fn set_zs(c: *mut Cpu, r: u32, bits: u32) {
    unsafe {
        (*c).zf = b(r & mask_of(bits) == 0);
        (*c).sf = b(r & sign_of(bits) != 0);
        (*c).pf = b((r & 0xFF).count_ones() & 1 == 0);
    }
}

unsafe fn alu_add(c: *mut Cpu, a: u32, b2: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let (a, b2) = (a & m, b2 & m);
        let wide = u64::from(a) + u64::from(b2);
        let r = (wide as u32) & m;
        (*c).cf = ((wide >> bits) & 1) as u8;
        (*c).of = b(((a ^ r) & (b2 ^ r)) & sign_of(bits) != 0);
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_sub(c: *mut Cpu, a: u32, b2: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let (a, b2) = (a & m, b2 & m);
        let r = a.wrapping_sub(b2) & m;
        (*c).cf = b(a < b2);
        (*c).of = b(((a ^ b2) & (a ^ r)) & sign_of(bits) != 0);
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_logic(c: *mut Cpu, r: u32, bits: u32) -> u32 {
    unsafe {
        let r = r & mask_of(bits);
        (*c).cf = 0;
        (*c).of = 0;
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_inc(c: *mut Cpu, a: u32, bits: u32) -> u32 {
    unsafe {
        let cf = (*c).cf;
        let r = alu_add(c, a, 1, bits);
        (*c).cf = cf;
        r
    }
}

unsafe fn alu_dec(c: *mut Cpu, a: u32, bits: u32) -> u32 {
    unsafe {
        let cf = (*c).cf;
        let r = alu_sub(c, a, 1, bits);
        (*c).cf = cf;
        r
    }
}

unsafe fn alu_neg(c: *mut Cpu, a: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let r = alu_sub(c, 0, a, bits);
        (*c).cf = b(a & m != 0);
        r & m
    }
}

unsafe fn alu_adc(c: *mut Cpu, a: u32, b2: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let (a, b2) = (a & m, b2 & m);
        let wide = u64::from(a) + u64::from(b2) + u64::from((*c).cf);
        let r = (wide as u32) & m;
        (*c).cf = ((wide >> bits) & 1) as u8;
        (*c).of = b(((a ^ r) & (b2 ^ r)) & sign_of(bits) != 0);
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_sbb(c: *mut Cpu, a: u32, b2: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let (a, b2) = (a & m, b2 & m);
        let borrow = u32::from((*c).cf);
        let r = a.wrapping_sub(b2).wrapping_sub(borrow) & m;
        (*c).cf = b(u64::from(a) < u64::from(b2) + u64::from(borrow));
        (*c).of = b(((a ^ b2) & (a ^ r)) & sign_of(bits) != 0);
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_bsf(c: *mut Cpu, src: u32, dst: u32) -> u32 {
    unsafe {
        (*c).zf = b(src == 0);
        if src == 0 { dst } else { src.trailing_zeros() }
    }
}

unsafe fn alu_bsr(c: *mut Cpu, src: u32, dst: u32) -> u32 {
    unsafe {
        (*c).zf = b(src == 0);
        if src == 0 {
            dst
        } else {
            31 - src.leading_zeros()
        }
    }
}

unsafe fn alu_rol(c: *mut Cpu, a: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        let a = a & m;
        if count == 0 {
            return a;
        }
        let n = count.checked_rem(bits).unwrap_or(0);
        let r = if n != 0 {
            ((a << n) | (a >> (bits - n))) & m
        } else {
            a
        };
        (*c).cf = (r & 1) as u8;
        if count == 1 {
            (*c).of = (((r >> (bits - 1)) & 1) as u8) ^ (*c).cf;
        }
        r
    }
}

unsafe fn alu_ror(c: *mut Cpu, a: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        let a = a & m;
        if count == 0 {
            return a;
        }
        let n = count.checked_rem(bits).unwrap_or(0);
        let r = if n != 0 {
            ((a >> n) | (a << (bits - n))) & m
        } else {
            a
        };
        (*c).cf = ((r >> (bits - 1)) & 1) as u8;
        if count == 1 {
            (*c).of = (((r >> (bits - 1)) ^ (r >> (bits - 2))) & 1) as u8;
        }
        r
    }
}

unsafe fn alu_shld(c: *mut Cpu, dst: u32, src: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        let (dst, src) = (dst & m, src & m);
        if count == 0 || count > bits {
            return dst;
        }
        (*c).cf = ((dst >> (bits - count)) & 1) as u8;
        let low = if count == bits {
            0
        } else {
            src >> (bits - count)
        };
        let r = (dst.checked_shl(count).unwrap_or(0) | low) & m;
        if count == 1 {
            (*c).of = (((r >> (bits - 1)) & 1) as u8) ^ (*c).cf;
        }
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_shrd(c: *mut Cpu, dst: u32, src: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        let (dst, src) = (dst & m, src & m);
        if count == 0 || count > bits {
            return dst;
        }
        (*c).cf = ((dst >> (count - 1)) & 1) as u8;
        let high = if count == bits {
            0
        } else {
            src << (bits - count)
        };
        let r = (dst.checked_shr(count).unwrap_or(0) | high) & m;
        if count == 1 {
            (*c).of = (((r >> (bits - 1)) ^ (dst >> (bits - 1))) & 1) as u8;
        }
        set_zs(c, r, bits);
        r
    }
}

unsafe fn eflags_get(c: *mut Cpu) -> u32 {
    unsafe {
        2 | u32::from((*c).cf)
            | (u32::from((*c).pf) << 2)
            | (u32::from((*c).zf) << 6)
            | (u32::from((*c).sf) << 7)
            | 0x200
            | (u32::from((*c).df) << 10)
            | (u32::from((*c).of) << 11)
    }
}

unsafe fn eflags_set(c: *mut Cpu, f: u32) {
    unsafe {
        (*c).cf = (f & 1) as u8;
        (*c).pf = ((f >> 2) & 1) as u8;
        (*c).zf = ((f >> 6) & 1) as u8;
        (*c).sf = ((f >> 7) & 1) as u8;
        (*c).df = ((f >> 10) & 1) as u8;
        (*c).of = ((f >> 11) & 1) as u8;
    }
}

unsafe fn alu_rcl(c: *mut Cpu, a: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        let mut a = a & m;
        for _ in 0..count {
            let top = ((a >> (bits - 1)) & 1) as u8;
            a = ((a << 1) | u32::from((*c).cf)) & m;
            (*c).cf = top;
        }
        if count == 1 {
            (*c).of = (((a >> (bits - 1)) & 1) as u8) ^ (*c).cf;
        }
        a
    }
}

unsafe fn alu_rcr(c: *mut Cpu, a: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        let mut a = a & m;
        if count == 1 {
            (*c).of = (((a >> (bits - 1)) & 1) as u8) ^ (*c).cf;
        }
        for _ in 0..count {
            let low = (a & 1) as u8;
            a = (a >> 1) | (u32::from((*c).cf) << (bits - 1));
            (*c).cf = low;
        }
        a & m
    }
}

unsafe fn alu_shl(c: *mut Cpu, a: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        if count == 0 {
            return a & m;
        }
        let a = a & m;
        (*c).cf = if count <= bits {
            ((a >> (bits - count)) & 1) as u8
        } else {
            0
        };
        let r = if count >= bits { 0 } else { (a << count) & m };
        (*c).of = if count == 1 {
            b(r & sign_of(bits) != 0) ^ (*c).cf
        } else {
            0
        };
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_shr(c: *mut Cpu, a: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        if count == 0 {
            return a & m;
        }
        let a = a & m;
        (*c).cf = if count <= bits {
            ((a >> (count - 1)) & 1) as u8
        } else {
            0
        };
        let r = if count >= bits { 0 } else { a >> count };
        (*c).of = if count == 1 {
            b(a & sign_of(bits) != 0)
        } else {
            0
        };
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_sar(c: *mut Cpu, a: u32, count: u32, bits: u32) -> u32 {
    unsafe {
        let m = mask_of(bits);
        let count = count & 31;
        if count == 0 {
            return a & m;
        }
        let s = ((a << (32 - bits)) as i32) >> (32 - bits);
        (*c).cf = if count <= bits {
            (((s as u32) >> (count - 1)) & 1) as u8
        } else {
            b(s < 0)
        };
        let r = if count >= bits {
            if s < 0 { -1 } else { 0 }
        } else {
            s >> count
        };
        (*c).of = 0;
        set_zs(c, r as u32, bits);
        (r as u32) & m
    }
}

/// The low `bits` of `v`, sign-extended.
#[inline(always)]
fn sx(v: u32, bits: u32) -> i32 {
    ((v << (32 - bits)) as i32) >> (32 - bits)
}

unsafe fn alu_imul(c: *mut Cpu, a: u32, b2: u32, bits: u32) -> u32 {
    unsafe {
        let wide = i64::from(sx(a, bits)) * i64::from(sx(b2, bits));
        let r = (wide as u32) & mask_of(bits);
        let back = i64::from(sx(r, bits));
        (*c).cf = b(back != wide);
        (*c).of = (*c).cf;
        set_zs(c, r, bits);
        r
    }
}

unsafe fn alu_mul(c: *mut Cpu, a: u32) {
    unsafe {
        let p = u64::from((*c).eax) * u64::from(a);
        (*c).eax = p as u32;
        (*c).edx = (p >> 32) as u32;
        (*c).cf = b((*c).edx != 0);
        (*c).of = (*c).cf;
    }
}

unsafe fn alu_imul1(c: *mut Cpu, a: u32) {
    unsafe {
        let p = i64::from((*c).eax as i32) * i64::from(a as i32);
        (*c).eax = p as u32;
        (*c).edx = ((p as u64) >> 32) as u32;
        (*c).cf = b(i64::from((*c).eax as i32) != p);
        (*c).of = (*c).cf;
    }
}

unsafe fn alu_div(c: *mut Cpu, a: u32, at: u32) {
    unsafe {
        if a == 0 {
            lift_trap(c, at, b"divide by zero");
        }
        let n = (u64::from((*c).edx) << 32) | u64::from((*c).eax);
        let d = u64::from(a);
        (*c).eax = n.checked_div(d).unwrap_or(0) as u32;
        (*c).edx = n.checked_rem(d).unwrap_or(0) as u32;
    }
}

unsafe fn alu_idiv(c: *mut Cpu, a: u32, at: u32) {
    unsafe {
        let d = i64::from(a as i32);
        if d == 0 {
            lift_trap(c, at, b"divide by zero");
        }
        let n = ((u64::from((*c).edx) << 32) | u64::from((*c).eax)) as i64;
        // Only the most negative dividend over -1 overflows; it wraps, as the hardware's #DE is
        // not modelled.
        (*c).eax = n.checked_div(d).unwrap_or(n) as u32;
        (*c).edx = n.checked_rem(d).unwrap_or(0) as u32;
    }
}

// The 8- and 16-bit forms of mul, imul, div and idiv: AL or AX times the operand, or AX or DX:AX
// divided by it. A quotient that does not fit, or a zero divisor, is the console's #DE.
unsafe fn alu_mul8(c: *mut Cpu, a: u32) {
    unsafe {
        let p = ((*c).eax & 0xFF) * (a & 0xFF);
        (*c).eax = ((*c).eax & !0xFFFF) | p;
        (*c).cf = b((p >> 8) != 0);
        (*c).of = (*c).cf;
    }
}

unsafe fn alu_mul16(c: *mut Cpu, a: u32) {
    unsafe {
        let p = ((*c).eax & 0xFFFF) * (a & 0xFFFF);
        (*c).eax = ((*c).eax & !0xFFFF) | (p & 0xFFFF);
        (*c).edx = ((*c).edx & !0xFFFF) | (p >> 16);
        (*c).cf = b((p >> 16) != 0);
        (*c).of = (*c).cf;
    }
}

unsafe fn alu_imul1_8(c: *mut Cpu, a: u32) {
    unsafe {
        let p = sx((*c).eax, 8) * sx(a, 8);
        (*c).eax = ((*c).eax & !0xFFFF) | ((p as u32) & 0xFFFF);
        (*c).cf = b(sx(p as u32, 8) != p);
        (*c).of = (*c).cf;
    }
}

unsafe fn alu_imul1_16(c: *mut Cpu, a: u32) {
    unsafe {
        let p = sx((*c).eax, 16) * sx(a, 16);
        (*c).eax = ((*c).eax & !0xFFFF) | ((p as u32) & 0xFFFF);
        (*c).edx = ((*c).edx & !0xFFFF) | (((p as u32) >> 16) & 0xFFFF);
        (*c).cf = b(sx(p as u32, 16) != p);
        (*c).of = (*c).cf;
    }
}

unsafe fn alu_div8(c: *mut Cpu, a: u32, at: u32) {
    unsafe {
        let (d, n) = (a & 0xFF, (*c).eax & 0xFFFF);
        let (Some(q), Some(r)) = (n.checked_div(d), n.checked_rem(d)) else {
            lift_trap(c, at, b"divide by zero");
        };
        if q > 0xFF {
            lift_trap(c, at, b"divide overflow");
        }
        (*c).eax = ((*c).eax & !0xFFFF) | q | (r << 8);
    }
}

unsafe fn alu_div16(c: *mut Cpu, a: u32, at: u32) {
    unsafe {
        let d = a & 0xFFFF;
        let n = (((*c).edx & 0xFFFF) << 16) | ((*c).eax & 0xFFFF);
        let (Some(q), Some(r)) = (n.checked_div(d), n.checked_rem(d)) else {
            lift_trap(c, at, b"divide by zero");
        };
        if q > 0xFFFF {
            lift_trap(c, at, b"divide overflow");
        }
        (*c).eax = ((*c).eax & !0xFFFF) | q;
        (*c).edx = ((*c).edx & !0xFFFF) | r;
    }
}

unsafe fn alu_idiv8(c: *mut Cpu, a: u32, at: u32) {
    unsafe {
        let (d, n) = (sx(a, 8), sx((*c).eax, 16));
        let (Some(q), Some(r)) = (n.checked_div(d), n.checked_rem(d)) else {
            lift_trap(c, at, b"divide by zero");
        };
        if !(-128..=127).contains(&q) {
            lift_trap(c, at, b"divide overflow");
        }
        (*c).eax = ((*c).eax & !0xFFFF) | ((q as u32) & 0xFF) | (((r as u32) & 0xFF) << 8);
    }
}

unsafe fn alu_idiv16(c: *mut Cpu, a: u32, at: u32) {
    unsafe {
        let d = sx(a, 16);
        let n = ((((*c).edx & 0xFFFF) << 16) | ((*c).eax & 0xFFFF)) as i32;
        let (Some(q), Some(r)) = (n.checked_div(d), n.checked_rem(d)) else {
            lift_trap(c, at, b"divide by zero");
        };
        if !(-32768..=32767).contains(&q) {
            lift_trap(c, at, b"divide overflow");
        }
        (*c).eax = ((*c).eax & !0xFFFF) | ((q as u32) & 0xFFFF);
        (*c).edx = ((*c).edx & !0xFFFF) | ((r as u32) & 0xFFFF);
    }
}

#[inline(always)]
unsafe fn cond(c: *mut Cpu, code: u32) -> bool {
    unsafe {
        let (cf, zf, sf) = ((*c).cf != 0, (*c).zf != 0, (*c).sf != 0);
        let (of, pf) = ((*c).of != 0, (*c).pf != 0);
        match code {
            0 => of,
            1 => !of,
            2 => cf,
            3 => !cf,
            4 => zf,
            5 => !zf,
            6 => cf || zf,
            7 => !cf && !zf,
            8 => sf,
            9 => !sf,
            10 => pf,
            11 => !pf,
            12 => sf != of,
            13 => sf == of,
            14 => zf || sf != of,
            _ => !zf && sf == of,
        }
    }
}

// x87 floating point.

/// `ST(i)`.
#[inline(always)]
unsafe fn st(c: *mut Cpu, i: u32) -> *mut f64 {
    unsafe {
        (&raw mut (*c).fpu)
            .cast::<f64>()
            .add(((*c).fpu_top.wrapping_add(i) & 7) as usize)
    }
}

#[inline(always)]
unsafe fn fpush(c: *mut Cpu, v: f64) {
    unsafe {
        (*c).fpu_top = ((*c).fpu_top + 7) & 7;
        *st(c, 0) = v;
    }
}

#[inline(always)]
unsafe fn fpop(c: *mut Cpu) {
    unsafe { (*c).fpu_top = ((*c).fpu_top + 1) & 7 }
}

#[inline(always)]
unsafe fn fpu_cw(c: *mut Cpu) -> u32 {
    unsafe { (*c).fpu_cwx ^ 0x027F }
}

#[inline(always)]
unsafe fn mxcsr_get(c: *mut Cpu) -> u32 {
    unsafe { (*c).mxcsr_x ^ 0x1F80 }
}

/// The result of an x87 operation that produced a NaN from operands `a` and `b`: a NaN operand
/// made quiet (the quiet one beats a signaling one, the larger significand beats the smaller), or
/// the negative "real indefinite" when neither operand was a NaN.
fn fnan2(a: f64, b2: f64) -> f64 {
    const QUIET: u64 = 1 << 51;
    const FRAC: u64 = (1 << 52) - 1;
    let (ua, ub) = (a.to_bits(), b2.to_bits());
    let (an, bn) = (a.is_nan(), b2.is_nan());
    let u = if an && bn {
        let (signaling_a, signaling_b) = (ua & QUIET == 0, ub & QUIET == 0);
        if signaling_a != signaling_b {
            if signaling_a { ub } else { ua }
        } else if ua & FRAC == ub & FRAC {
            if ua >> 63 != 0 { ub } else { ua }
        } else if ua & FRAC > ub & FRAC {
            ua
        } else {
            ub
        }
    } else if an || bn {
        if an { ua } else { ub }
    } else {
        0xFFF8_0000_0000_0000
    };
    f64::from_bits(u | QUIET)
}

/// Rounds to the control word's precision: 24 bits when it says single (the exponent range stays
/// wide, so values outside a float's normal range are left alone); doubles are the stack's own.
#[inline(always)]
unsafe fn fprec(c: *mut Cpu, v: f64) -> f64 {
    unsafe {
        if (*c).fpu_cwx & 0x300 != 0x200 {
            return v;
        }
        let m = v.abs();
        if (1.175_494_35e-38..=3.402_823_47e38).contains(&m) {
            return f64::from(v as f32);
        }
        v
    }
}

/// The result of an arithmetic operation on `a` and `b`, as the x87 delivers it.
#[inline(always)]
unsafe fn fres(c: *mut Cpu, r: f64, a: f64, b2: f64) -> f64 {
    unsafe {
        if r.is_nan() {
            fnan2(a, b2)
        } else {
            fprec(c, r)
        }
    }
}

/// Rounds to an integer as the control word's rounding mode says.
unsafe fn fround_cw(c: *mut Cpu, v: f64) -> f64 {
    unsafe {
        match (fpu_cw(c) >> 10) & 3 {
            0 => nearbyint(v),
            1 => floor(v),
            2 => ceil(v),
            _ => trunc(v),
        }
    }
}

/// An integral value as a signed integer of `bits` bits; out of range or a NaN gives the "integer
/// indefinite", the most negative value, as the x87 and SSE conversions do.
fn fint(r: f64, bits: u32) -> i64 {
    let lim = (1u64 << (bits - 1)) as f64;
    if !(r >= -lim && r < lim) {
        return (u64::MAX << (bits - 1)) as i64;
    }
    r as i64
}

unsafe fn cvt_ss2si(c: *mut Cpu, v: f32, truncate: bool) -> u32 {
    unsafe {
        let v = f64::from(v);
        let r = if truncate {
            trunc(v)
        } else {
            match (mxcsr_get(c) >> 13) & 3 {
                0 => nearbyint(v),
                1 => floor(v),
                2 => ceil(v),
                _ => trunc(v),
            }
        };
        fint(r, 32) as i32 as u32
    }
}

/// The result of a single-precision SSE operation that produced a NaN from `a` and `b`: the first
/// operand if it is a NaN, else the second, made quiet; the negative default NaN otherwise.
fn snan2(a: f32, b2: f32) -> f32 {
    let u = if a.is_nan() {
        a.to_bits()
    } else if b2.is_nan() {
        b2.to_bits()
    } else {
        0xFFC0_0000
    };
    f32::from_bits(u | 0x0040_0000)
}

#[inline(always)]
fn sres(r: f32, a: f32, b2: f32) -> f32 {
    if r.is_nan() { snan2(a, b2) } else { r }
}

unsafe fn fxam(c: *mut Cpu, v: f64) {
    unsafe {
        let class = if v.is_nan() {
            0x0100
        } else if v.is_infinite() {
            0x0500
        } else if v == 0.0 {
            0x4000
        } else {
            0x0400
        };
        (*c).fpu_cc = class | if v.is_sign_negative() { 0x0200 } else { 0 };
    }
}

/// FPREM and FPREM1: the remainder, with the quotient's low three bits in C0, C3 and C1, and C2
/// clear (the reduction is always complete).
unsafe fn fprem_op(c: *mut Cpu, nearest: bool) {
    unsafe {
        let (a, b2) = (*st(c, 0), *st(c, 1));
        if a.is_nan() || b2.is_nan() || a.is_infinite() || b2 == 0.0 {
            *st(c, 0) = fres(c, f64::NAN, a, b2);
            (*c).fpu_cc = 0;
            return;
        }
        let (r, q) = if nearest {
            let mut qi = 0i32;
            let r = remquo(a, b2, &raw mut qi);
            (r, qi.unsigned_abs() & 7)
        } else {
            let r = fmod(a, b2);
            let n = ((a - r) / b2).abs();
            (
                r,
                if n < 9.2e18 {
                    ((n as u64) & 7) as u32
                } else {
                    0
                },
            )
        };
        *st(c, 0) = r;
        (*c).fpu_cc = (if q & 4 != 0 { 0x0100 } else { 0 })
            | (if q & 2 != 0 { 0x4000 } else { 0 })
            | (if q & 1 != 0 { 0x0200 } else { 0 });
    }
}

unsafe fn fscale_op(x: f64, s: f64) -> f64 {
    unsafe {
        if x.is_nan() || s.is_nan() {
            return fnan2(x, s);
        }
        if s.is_infinite() {
            return if s > 0.0 {
                if x == 0.0 {
                    fnan2(x, x)
                } else {
                    x * f64::INFINITY
                }
            } else if x.is_infinite() {
                fnan2(x, x)
            } else {
                x * 0.0
            };
        }
        let s = if s > 100_000.0 {
            100_000.0
        } else if s < -100_000.0 {
            -100_000.0
        } else {
            s
        };
        ldexp(x, trunc(s) as i32)
    }
}

/// The sine, cosine and tangent are defined below 2^63 in magnitude; beyond it C2 is set and the
/// stack is left alone. Returns whether the argument is in range.
unsafe fn ftrig_ok(c: *mut Cpu, v: f64) -> bool {
    unsafe {
        if v.abs() >= 9.223_372_036_854_775_808e18 {
            (*c).fpu_cc |= 0x0400;
            return false;
        }
        (*c).fpu_cc &= !0x0400;
        true
    }
}

unsafe fn ldf32(a: u32) -> f64 {
    unsafe { f64::from(f32::from_bits(rd32(a))) }
}

unsafe fn ldf64(a: u32) -> f64 {
    unsafe { f64::from_bits(u64::from(rd32(a)) | (u64::from(rd32(a.wrapping_add(4))) << 32)) }
}

unsafe fn stf32(a: u32, v: f64) {
    unsafe { wr32(a, (v as f32).to_bits()) }
}

unsafe fn stf64(a: u32, v: f64) {
    unsafe {
        let bits = v.to_bits();
        wr32(a, bits as u32);
        wr32(a.wrapping_add(4), (bits >> 32) as u32);
    }
}

unsafe fn ld80(a: u32) -> f64 {
    unsafe {
        let m = u64::from(rd32(a)) | (u64::from(rd32(a.wrapping_add(4))) << 32);
        let se = rd16(a.wrapping_add(8));
        let negative = (se >> 15) & 1 != 0;
        let e = (se & 0x7FFF) as i32;
        if e == 0 && m == 0 {
            return if negative { -0.0 } else { 0.0 };
        }
        if e == 0x7FFF {
            if m << 1 == 0 {
                return if negative {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            let u = (u64::from(negative) << 63)
                | (0x7FF << 52)
                | (1 << 51)
                | ((m >> 11) & ((1 << 51) - 1));
            return f64::from_bits(u);
        }
        let v = ldexp(m as f64, e - 16383 - 63);
        if negative { -v } else { v }
    }
}

unsafe fn st80(a: u32, v: f64) {
    unsafe {
        let sign = u32::from(v.is_sign_negative());
        let magnitude = v.abs();
        let (mut m, mut e) = (0u64, 0u32);
        if magnitude == f64::INFINITY {
            m = 1 << 63;
            e = 0x7FFF;
        } else if magnitude != 0.0 && !magnitude.is_nan() {
            let mut exp2 = 0i32;
            let f = frexp(magnitude, &raw mut exp2);
            m = ldexp(f, 64) as u64;
            e = (exp2 - 1 + 16383) as u32;
        } else if magnitude.is_nan() {
            m = (3 << 62) | ((v.to_bits() & ((1 << 51) - 1)) << 11);
            e = 0x7FFF;
        }
        wr32(a, m as u32);
        wr32(a.wrapping_add(4), (m >> 32) as u32);
        wr16(a.wrapping_add(8), (sign << 15) | e);
    }
}

unsafe fn fcompare(c: *mut Cpu, a: f64, b2: f64) {
    unsafe {
        (*c).fpu_cc = if a.is_nan() || b2.is_nan() {
            0x4500
        } else if a < b2 {
            0x0100
        } else if a == b2 {
            0x4000
        } else {
            0
        };
    }
}

unsafe fn fcompare_flags(c: *mut Cpu, a: f64, b2: f64) {
    unsafe {
        fcompare(c, a, b2);
        (*c).zf = b((*c).fpu_cc & 0x4000 != 0);
        (*c).cf = b((*c).fpu_cc & 0x0100 != 0);
        (*c).sf = 0;
        (*c).of = 0;
        (*c).pf = b((*c).fpu_cc & 0x4500 == 0x4500);
    }
}

unsafe fn fstatus(c: *mut Cpu) -> u32 {
    unsafe { ((*c).fpu_top << 11) | (*c).fpu_cc }
}

/// `fnstenv` and `fnsave`'s 28-byte environment, in the 32-bit layout: the control, status and
/// tag words, the tags all valid as the stack keeps none, and no last instruction or operand.
unsafe fn fstenv_op(c: *mut Cpu, a: u32) {
    unsafe {
        wr32(a, 0xFFFF_0000 | fpu_cw(c));
        wr32(a.wrapping_add(4), 0xFFFF_0000 | fstatus(c));
        wr32(a.wrapping_add(8), 0xFFFF_0000);
        let mut i = 12u32;
        while i < 28 {
            wr32(a.wrapping_add(i), 0);
            i += 4;
        }
    }
}

unsafe fn fldenv_op(c: *mut Cpu, a: u32) {
    unsafe {
        let sw = rd16(a.wrapping_add(4));
        (*c).fpu_cwx = rd16(a) ^ 0x027F;
        (*c).fpu_top = (sw >> 11) & 7;
        (*c).fpu_cc = sw & !0x3800;
    }
}

/// `fnsave` stores the environment and `ST(0)` to `ST(7)`, then initializes as `fninit` does;
/// `frstor` loads them back.
unsafe fn fsave_op(c: *mut Cpu, a: u32) {
    unsafe {
        fstenv_op(c, a);
        let mut i = 0u32;
        while i < 8 {
            st80(a.wrapping_add(28 + 10 * i), *st(c, i));
            i += 1;
        }
        (*c).fpu_top = 0;
        (*c).fpu_cc = 0;
        (*c).fpu_cwx = 0x0100;
    }
}

unsafe fn frstor_op(c: *mut Cpu, a: u32) {
    unsafe {
        fldenv_op(c, a);
        let mut i = 0u32;
        while i < 8 {
            *st(c, i) = ld80(a.wrapping_add(28 + 10 * i));
            i += 1;
        }
    }
}

// SSE and MMX registers.

/// XMM register `i`.
#[inline(always)]
unsafe fn xmm(c: *mut Cpu, i: usize) -> *mut Xmm {
    unsafe { (&raw mut (*c).xmm).cast::<Xmm>().add(i & 7) }
}

/// MMX register `i`.
#[inline(always)]
unsafe fn mm(c: *mut Cpu, i: usize) -> *mut u64 {
    unsafe { (&raw mut (*c).mm).cast::<u64>().add(i & 7) }
}

const XMM_ZERO: Xmm = Xmm { u: [0; 4] };

/// `a` and `b` as lanes of `w` bits: `f` takes each pair, zero-extended, and gives the result
/// lane, which is cut to `w` bits.
#[inline(always)]
fn lanes(a: u64, b2: u64, w: u32, f: impl Fn(u64, u64) -> u64) -> u64 {
    let m = u64::MAX >> (64 - w);
    let mut r = 0u64;
    let mut s = 0u32;
    while s < 64 {
        r |= (f((a >> s) & m, (b2 >> s) & m) & m) << s;
        s += w;
    }
    r
}

/// A lane of `w` bits, sign-extended.
#[inline(always)]
fn lane_sx(v: u64, w: u32) -> i64 {
    ((v << (64 - w)) as i64) >> (64 - w)
}

#[inline(always)]
fn clamp_lane(v: i64, lo: i64, hi: i64) -> u64 {
    (if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }) as u64
}

fn mmx_paddb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, |p, q| p + q)
}
fn mmx_paddw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| p + q)
}
fn mmx_paddd(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 32, |p, q| p + q)
}
fn mmx_paddq(a: u64, b2: u64) -> u64 {
    a.wrapping_add(b2)
}
fn mmx_paddsb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, |p, q| {
        clamp_lane(lane_sx(p, 8) + lane_sx(q, 8), -128, 127)
    })
}
fn mmx_paddsw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| {
        clamp_lane(lane_sx(p, 16) + lane_sx(q, 16), -32768, 32767)
    })
}
fn mmx_paddusb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, |p, q| (p + q).min(255))
}
fn mmx_paddusw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| (p + q).min(65535))
}
fn mmx_psubb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, u64::wrapping_sub)
}
fn mmx_psubw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, u64::wrapping_sub)
}
fn mmx_psubd(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 32, u64::wrapping_sub)
}
fn mmx_psubq(a: u64, b2: u64) -> u64 {
    a.wrapping_sub(b2)
}
fn mmx_psubsb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, |p, q| {
        clamp_lane(lane_sx(p, 8) - lane_sx(q, 8), -128, 127)
    })
}
fn mmx_psubsw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| {
        clamp_lane(lane_sx(p, 16) - lane_sx(q, 16), -32768, 32767)
    })
}
fn mmx_psubusb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, u64::saturating_sub)
}
fn mmx_psubusw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, u64::saturating_sub)
}
fn mmx_pmullw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| (lane_sx(p, 16) * lane_sx(q, 16)) as u64)
}
fn mmx_pmulhw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| {
        ((lane_sx(p, 16) * lane_sx(q, 16)) >> 16) as u64
    })
}
fn mmx_pmulhuw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| (p * q) >> 16)
}
fn mmx_pavgb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, |p, q| (p + q + 1) >> 1)
}
fn mmx_pavgw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| (p + q + 1) >> 1)
}
fn mmx_pminub(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, u64::min)
}
fn mmx_pmaxub(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, u64::max)
}
fn mmx_pminsw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| {
        if lane_sx(p, 16) < lane_sx(q, 16) {
            p
        } else {
            q
        }
    })
}
fn mmx_pmaxsw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| {
        if lane_sx(p, 16) > lane_sx(q, 16) {
            p
        } else {
            q
        }
    })
}
fn mmx_pcmpeqb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, |p, q| if p == q { u64::MAX } else { 0 })
}
fn mmx_pcmpeqw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| if p == q { u64::MAX } else { 0 })
}
fn mmx_pcmpeqd(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 32, |p, q| if p == q { u64::MAX } else { 0 })
}
fn mmx_pcmpgtb(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 8, |p, q| {
        if lane_sx(p, 8) > lane_sx(q, 8) {
            u64::MAX
        } else {
            0
        }
    })
}
fn mmx_pcmpgtw(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 16, |p, q| {
        if lane_sx(p, 16) > lane_sx(q, 16) {
            u64::MAX
        } else {
            0
        }
    })
}
fn mmx_pcmpgtd(a: u64, b2: u64) -> u64 {
    lanes(a, b2, 32, |p, q| {
        if lane_sx(p, 32) > lane_sx(q, 32) {
            u64::MAX
        } else {
            0
        }
    })
}
fn mmx_pand(a: u64, b2: u64) -> u64 {
    a & b2
}
fn mmx_pandn(a: u64, b2: u64) -> u64 {
    !a & b2
}
fn mmx_por(a: u64, b2: u64) -> u64 {
    a | b2
}
fn mmx_pxor(a: u64, b2: u64) -> u64 {
    a ^ b2
}

/// The word at lane `i` of `v`, sign-extended.
#[inline(always)]
fn word(v: u64, i: u32) -> i64 {
    lane_sx((v >> (16 * i)) & 0xFFFF, 16)
}

fn mmx_pmaddwd(a: u64, b2: u64) -> u64 {
    let low = word(a, 0) * word(b2, 0) + word(a, 1) * word(b2, 1);
    let high = word(a, 2) * word(b2, 2) + word(a, 3) * word(b2, 3);
    u64::from(low as u32) | (u64::from(high as u32) << 32)
}

fn mmx_psadbw(a: u64, b2: u64) -> u64 {
    let mut sum = 0u64;
    for i in 0..8 {
        let (p, q) = ((a >> (8 * i)) & 0xFF, (b2 >> (8 * i)) & 0xFF);
        sum += p.abs_diff(q);
    }
    sum
}

/// Interleaves the low (`half` 0) or high (`half` 1) lanes of `a` and `b` of `w` bits.
#[inline(always)]
fn unpack(a: u64, b2: u64, w: u32, half: u32) -> u64 {
    let m = (1u64 << w).wrapping_sub(1);
    let n = 32 / w;
    let mut r = 0u64;
    for i in 0..n {
        let s = (half * n + i) * w;
        r |= ((a >> s) & m) << (2 * i * w);
        r |= ((b2 >> s) & m) << ((2 * i + 1) * w);
    }
    r
}

fn mmx_punpcklbw(a: u64, b2: u64) -> u64 {
    unpack(a, b2, 8, 0)
}
fn mmx_punpcklwd(a: u64, b2: u64) -> u64 {
    unpack(a, b2, 16, 0)
}
fn mmx_punpckldq(a: u64, b2: u64) -> u64 {
    unpack(a, b2, 32, 0)
}
fn mmx_punpckhbw(a: u64, b2: u64) -> u64 {
    unpack(a, b2, 8, 1)
}
fn mmx_punpckhwd(a: u64, b2: u64) -> u64 {
    unpack(a, b2, 16, 1)
}
fn mmx_punpckhdq(a: u64, b2: u64) -> u64 {
    unpack(a, b2, 32, 1)
}

fn mmx_packuswb(a: u64, b2: u64) -> u64 {
    let mut r = 0u64;
    for i in 0..4 {
        r |= clamp_lane(word(a, i), 0, 255) << (8 * i);
        r |= clamp_lane(word(b2, i), 0, 255) << (8 * (i + 4));
    }
    r
}

fn mmx_packsswb(a: u64, b2: u64) -> u64 {
    let mut r = 0u64;
    for i in 0..4 {
        r |= (clamp_lane(word(a, i), -128, 127) & 0xFF) << (8 * i);
        r |= (clamp_lane(word(b2, i), -128, 127) & 0xFF) << (8 * (i + 4));
    }
    r
}

fn mmx_packssdw(a: u64, b2: u64) -> u64 {
    let mut r = 0u64;
    for i in 0..2 {
        let (p, q) = (lane_sx(a >> (32 * i), 32), lane_sx(b2 >> (32 * i), 32));
        r |= (clamp_lane(p, -32768, 32767) & 0xFFFF) << (16 * i);
        r |= (clamp_lane(q, -32768, 32767) & 0xFFFF) << (16 * (i + 2));
    }
    r
}

/// A packed shift of lanes of `w` bits: left (`op` 0), logical right (1) or arithmetic right (2).
#[inline(always)]
fn mmx_shift(a: u64, count: u64, w: u32, op: u32) -> u64 {
    lanes(a, 0, w, |p, _| {
        if count >= u64::from(w) {
            if op == 2 {
                (lane_sx(p, w) >> (w - 1)) as u64
            } else {
                0
            }
        } else {
            let n = count as u32;
            match op {
                0 => p << n,
                1 => p >> n,
                _ => (lane_sx(p, w) >> n) as u64,
            }
        }
    })
}

fn mmx_psllw(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 16, 0)
}
fn mmx_pslld(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 32, 0)
}
fn mmx_psllq(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 64, 0)
}
fn mmx_psrlw(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 16, 1)
}
fn mmx_psrld(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 32, 1)
}
fn mmx_psrlq(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 64, 1)
}
fn mmx_psraw(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 16, 2)
}
fn mmx_psrad(a: u64, n: u64) -> u64 {
    mmx_shift(a, n, 32, 2)
}

fn mmx_pshufw(a: u64, order: u32) -> u64 {
    let mut r = 0u64;
    for i in 0..4 {
        let from = (order >> (2 * i)) & 3;
        r |= ((a >> (16 * from)) & 0xFFFF) << (16 * i);
    }
    r
}

/// pmovmskb: the sign bit of each byte of an MMX value.
fn mmx_pmovmskb(v: u64) -> u32 {
    let mut m = 0u32;
    for i in 0..8 {
        m |= (((v >> (8 * i + 7)) & 1) as u32) << i;
    }
    m
}
