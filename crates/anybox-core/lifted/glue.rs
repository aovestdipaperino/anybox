// The glue the lifted code and the runtime meet through: the kernel-call hook, the guest thread
// bodies, and `main`. It is a unit of the lifted program, after the prelude, the globals, and the
// constants of the title it is built for: `ENTRY`, `ENTER_SECTION`, `LEAVE_SECTION` and
// `APP_NAME`.

#[repr(C)]
struct AnyxboxResult {
    state: u32,
    eax: u32,
    esp: u32,
    edx: u32,
}

/// Where a host thread's guest CPU is, so a hibernation can save it and a restore fill it.
#[repr(C)]
struct CpuLayout {
    cpu: *mut Cpu,
    cpu_size: u32,
    depth: *mut u32,
    frames: *mut Frame,
    jumps: *mut *mut Jmp,
    jump_top: *mut u32,
    restore: *mut JmpSaved,
    restore_count: *mut u32,
}

#[repr(C)]
struct AnyxboxResume {
    kind: u32,
    eax: u32,
    esp: u32,
    edx: u32,
    target: u32,
}

unsafe extern "C" {
    fn anyxbox_boot(xbe: *const u8, len: usize, start: *mut u32) -> *mut u8;
    /// The lifted code's digest, from the digest unit.
    static anyxbox_code_digest: [u8; 20];
    fn anyxbox_set_code_digest(digest: *const u8);
    /// The title's executable, from the boot unit.
    static anyxbox_xbe: [u8; 0];
    static anyxbox_xbe_end: [u8; 0];
    fn anyxbox_native_call(target: u32, esp: u32, ecx: u32, edx: u32, out: *mut AnyxboxResult);
    fn anyxbox_thread_end();
    fn anyxbox_thread_end_with(result: u32);
    fn anyxbox_main_loop();
    /// The guest base the runtime set.
    fn anyxbox_guest_base() -> *mut u8;
    fn anyxbox_register_cpu(layout: *const CpuLayout);
    fn anyxbox_resume_load() -> u32;
    fn anyxbox_resume_wait(out: *mut AnyxboxResume);
    /// The runtime prints the guest memory `ANYXBOX_DUMP` names.
    fn anyxbox_dump_requested();
    /// The runtime deletes a snapshot that could not be resumed.
    fn anyxbox_resume_failed();

    fn pthread_exit(value: *mut u8) -> !;
    fn pthread_key_create(key: *mut usize, destructor: unsafe extern "C" fn(*mut u8)) -> i32;
    fn pthread_setspecific(key: usize, value: *const u8) -> i32;
    fn setenv(name: *const u8, value: *const u8, overwrite: i32) -> i32;
}

/// The key whose destructor gives back each host thread's guest state when the thread ends: a
/// title that makes a guest thread per DPC would otherwise leave 1.5 MiB behind for each. The
/// destructor runs after the thread's last guest code, on `pthread_exit` and on return alike, and
/// touches only the state it is handed. `main` makes the key before any guest thread starts.
static mut THREAD_KEY: usize = 0;

unsafe extern "C" fn thread_release(c: *mut u8) {
    unsafe { lift_thread_free(c.cast()) }
}

/// Every host thread that runs a guest thread starts here: its guest state, and the runtime told
/// where its guest CPU is.
unsafe fn guest_thread_start() -> *mut Cpu {
    unsafe {
        let c = lift_thread_new();
        pthread_setspecific(THREAD_KEY, c.cast());
        let t = c.cast::<Thread>();
        let layout = CpuLayout {
            cpu: c,
            cpu_size: core::mem::size_of::<Cpu>() as u32,
            depth: &raw mut (*c).depth,
            frames: (*c).frames,
            jumps: (*t).jmp_stack.as_mut_ptr(),
            jump_top: &raw mut (*t).jmp_top,
            restore: (*t).restore.as_mut_ptr(),
            restore_count: &raw mut (*t).restore_count,
        };
        anyxbox_register_cpu(&raw const layout);
        c
    }
}

/// The uncontended cases of RtlEnterCriticalSection and RtlLeaveCriticalSection, which titles call
/// millions of times, done here as the emulator's guest stubs do them instead of in the kernel.
/// `LockCount` is at +0x10, `RecursionCount` at +0x14, `OwningThread` at +0x18. Returns whether
/// the call is complete.
unsafe fn fast_section(c: *mut Cpu, target: u32) -> bool {
    unsafe {
        let section = rd32((*c).esp.wrapping_add(4));
        if target == ENTER_SECTION {
            let me = rd32(FS_BASE.wrapping_add(0x28));
            let owner = rd32(section + 0x18);
            if owner == 0 {
                wr32(section + 0x18, me);
                wr32(section + 0x14, 1);
                wr32(section + 0x10, 0);
            } else if owner == me {
                wr32(section + 0x14, rd32(section + 0x14).wrapping_add(1));
                wr32(section + 0x10, rd32(section + 0x10).wrapping_add(1));
            } else {
                return false;
            }
        } else {
            let recursion = rd32(section + 0x14);
            if recursion > 1 {
                wr32(section + 0x14, recursion - 1);
                wr32(section + 0x10, rd32(section + 0x10).wrapping_sub(1));
            } else if rd32(section + 0x10) == 0 {
                wr32(section + 0x18, 0);
                wr32(section + 0x14, 0);
                wr32(section + 0x10, 0xFFFF_FFFF);
            } else {
                return false;
            }
        }
        (*c).eax = 0;
        // The return address and the argument: stdcall.
        (*c).esp = (*c).esp.wrapping_add(8);
        true
    }
}

/// The return addresses along the guest's frame-pointer chain: where the title was when the
/// runtime could not go on. Functions built without frame pointers end the walk early.
unsafe fn guest_backtrace(c: *mut Cpu) {
    unsafe {
        put(b"guest registers:");
        for (name, value) in [
            (&b" eax="[..], (*c).eax),
            (b" ecx=", (*c).ecx),
            (b" edx=", (*c).edx),
            (b" ebx=", (*c).ebx),
            (b" esp=", (*c).esp),
            (b" ebp=", (*c).ebp),
            (b" esi=", (*c).esi),
            (b" edi=", (*c).edi),
        ] {
            put(name);
            put_hex(value);
        }
        put(b"\nguest backtrace:");
        let mut ebp = (*c).ebp;
        // Only frames within a megabyte above the stack pointer: EBP is a general register too.
        for _ in 0..24 {
            if ebp < (*c).esp || ebp - (*c).esp >= 0x10_0000 {
                break;
            }
            put(b" ");
            put_hex(rd32(ebp.wrapping_add(4)));
            let next = rd32(ebp);
            if next <= ebp {
                break;
            }
            ebp = next;
        }
        put(b"\n");
    }
}

/// What `lift_trap` prints besides its message: the registers, the frame-pointer chain, and the
/// memory the user asked for. A trap while resuming down the frames means the snapshot cannot
/// resume, so it goes, and the next launch boots afresh.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_trap_hook(c: *mut Cpu) {
    unsafe {
        if (*c).resuming != 0 {
            anyxbox_resume_failed();
        }
        guest_backtrace(c);
        anyxbox_dump_requested();
    }
}

/// A kernel call or a missing function: the runtime completes the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_native(c: *mut Cpu, target: u32, at: u32) {
    unsafe {
        if (target == ENTER_SECTION || target == LEAVE_SECTION) && fast_section(c, target) {
            return;
        }
        let mut r = AnyxboxResult {
            state: 0,
            eax: 0,
            esp: 0,
            edx: 0,
        };
        anyxbox_native_call(target, (*c).esp, (*c).ecx, (*c).edx, &raw mut r);
        if r.state == 0 {
            lift_trap(c, at, b"a kernel call the native runtime cannot complete");
        }
        if r.state == 3 {
            // The calling guest thread finished: it runs no more guest code.
            anyxbox_thread_end();
            pthread_exit(core::ptr::null_mut());
        }
        (*c).esp = r.esp;
        (*c).eax = r.eax;
        // The high half of a 64-bit result, or EDX unchanged.
        (*c).edx = r.edx;
    }
}

/// The innermost frame of a resumed thread: once its turn comes, it finishes the contact it was
/// parked in, a kernel call's result, the kernel call itself, or a poll, and runs on.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_resume_innermost(c: *mut Cpu) {
    unsafe {
        let mut r = AnyxboxResume {
            kind: 0,
            eax: 0,
            esp: 0,
            edx: 0,
            target: 0,
        };
        anyxbox_resume_wait(&raw mut r);
        (*c).resuming = 0;
        if r.kind == 0 {
            (*c).esp = r.esp;
            (*c).eax = r.eax;
            (*c).edx = r.edx;
        } else if r.kind == 1 {
            lift_native(c, r.target, 0);
        }
    }
}

/// A guest thread resumed from a snapshot, on its own host thread: its CPU and frames come from
/// the snapshot, and it calls down its frames again. It ends as the thread it was would have.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_resume_body(tid: u32, fs_base: u32) {
    unsafe {
        // `main` sets G and FS_BASE only when it boots the title itself, and only once
        // `anyxbox_boot` returns, but resumed threads start inside it: each takes the guest base
        // and the KPCR from the runtime, which set them before starting them.
        G = anyxbox_guest_base();
        FS_BASE = fs_base;
        let c = guest_thread_start();
        let frames = (*c).frames;
        let count = anyxbox_resume_load();
        (*c).frames = frames;
        (*c).resume_depth = count;
        (*c).depth = 0;
        (*c).resuming = 1;
        lift_resume_root(c);
        if tid == 1 {
            anyxbox_thread_end();
        } else {
            anyxbox_thread_end_with((*c).eax);
        }
        pthread_exit(core::ptr::null_mut());
    }
}

/// The title's entry point after a reset, on its own host thread: the guest memory is new.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_entry_body(base: *mut u8, esp: u32, fs_base: u32) {
    unsafe {
        let c = guest_thread_start();
        G = base;
        (*c).esp = esp;
        (*c).eax = 0;
        FS_BASE = fs_base;
        // The entry return trap.
        push32(c, 0x8001_0000);
        run(c, ENTRY);
        anyxbox_thread_end();
        pthread_exit(core::ptr::null_mut());
    }
}

/// The routine of a guest thread created by the kernel, on its own host thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_thread_body(start: u32, esp: u32) {
    unsafe {
        let c = guest_thread_start();
        (*c).esp = esp;
        (*c).eax = 0;
        // THREAD_RETURN: the routine returns here.
        rt_call(c, start, 0x8001_0FFC);
        // Its result: a completion routine's verdict.
        anyxbox_thread_end_with((*c).eax);
        pthread_exit(core::ptr::null_mut());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        pthread_key_create(&raw mut THREAD_KEY, thread_release);
        // The window's title.
        setenv(b"ANYXBOX_NAME\0".as_ptr(), APP_NAME.as_ptr(), 0);
        let mut start = [0u32; 4];
        // Snapshots belong to this lifted code, whatever runtime it is linked with.
        anyxbox_set_code_digest((&raw const anyxbox_code_digest).cast::<u8>());
        let xbe = (&raw const anyxbox_xbe).cast::<u8>();
        let len = (&raw const anyxbox_xbe_end as usize) - (xbe as usize);
        G = anyxbox_boot(xbe, len, start.as_mut_ptr());
        if G.is_null() {
            return 1;
        }
        if start[3] != 0 {
            // Resumed from a snapshot: the guest threads run on host threads of their own.
            anyxbox_main_loop();
            return 0;
        }
        let c = guest_thread_start();
        (*c).esp = start[0];
        FS_BASE = start[2];
        // The entry return trap: where the entry point returns to.
        push32(c, 0x8001_0000);
        run(c, ENTRY);
        anyxbox_thread_end();
        // The boot thread is done; the main thread runs the window, which macOS requires.
        anyxbox_main_loop();
        0
    }
}
