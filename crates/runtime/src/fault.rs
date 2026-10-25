//! The host fault handler: reports a fault in lifted code with the guest address, the guest
//! thread id, the faulting host symbol and its caller, the KPCR's current thread and its TLS
//! pointer, then exits with `128 + signal`. Spec 6.1.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::boot::KPCR;

/// The guest base, read by the handler.
static GUEST_BASE: AtomicUsize = AtomicUsize::new(0);

/// Installs the SIGSEGV and SIGBUS handlers.
pub fn install() {
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handle as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&raw mut action.sa_mask);
        libc::sigaction(libc::SIGSEGV, &raw const action, std::ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &raw const action, std::ptr::null_mut());
    }
}

/// Sets the guest base for the handler.
pub fn set_guest_base(base: *mut u8) {
    GUEST_BASE.store(base as usize, Ordering::Relaxed);
}

/// A fixed buffer the report is formatted into, so the handler does not allocate.
struct Report {
    bytes: [u8; 512],
    len: usize,
}

impl std::fmt::Write for Report {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let n = s.len().min(self.bytes.len() - self.len);
        self.bytes[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

/// Writes the name of the host symbol containing `pc`, as `dladdr` finds it, and the offset.
fn write_symbol(report: &mut Report, pc: usize) {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let found = unsafe { libc::dladdr(pc as *const libc::c_void, &raw mut info) } != 0;
    if found && !info.dli_sname.is_null() {
        let name = unsafe { std::ffi::CStr::from_ptr(info.dli_sname) };
        let offset = pc.wrapping_sub(info.dli_saddr as usize);
        let _ = write!(report, "{}+{offset:#x}", name.to_string_lossy());
    } else {
        let _ = write!(report, "{pc:#x}");
    }
}

/// Reads a guest dword the handler may touch: only from the KPCR and KTHREAD pages, which are
/// committed at boot.
fn read_guest(base: usize, address: u32) -> u32 {
    unsafe { std::ptr::read_unaligned((base + address as usize) as *const u32) }
}

extern "C" fn handle(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let fault = unsafe { (*info).si_addr() } as usize;
    let base = GUEST_BASE.load(Ordering::Relaxed);
    // The runtime is built for arm64 macOS only, where `usize` is 64 bits.
    #[allow(clippy::cast_possible_truncation)]
    let (pc, lr) = unsafe {
        let uc = ctx.cast::<libc::ucontext_t>();
        (*uc)
            .uc_mcontext
            .as_ref()
            .map_or((0, 0), |mc| (mc.__ss.__pc as usize, mc.__ss.__lr as usize))
    };
    let mut report = Report {
        bytes: [0; 512],
        len: 0,
    };
    let _ = write!(report, "guest fault: signal {sig}, ");
    if base != 0 && fault >= base && fault - base < 1 << 32 {
        let _ = write!(report, "guest address {:#010x}", fault - base);
    } else {
        let _ = write!(report, "host address {fault:#x} (outside the guest space)");
    }
    let _ = write!(
        report,
        ", guest thread {}, in ",
        crate::kernel::thread::RUNNING.load(Ordering::Relaxed)
    );
    write_symbol(&mut report, pc);
    let _ = write!(report, " called from ");
    write_symbol(&mut report, lr);
    if base != 0 {
        let kthread = read_guest(base, KPCR + 0x28);
        let tls = if kthread == 0 {
            0
        } else {
            read_guest(base, kthread + 0x28)
        };
        let _ = write!(report, ", KTHREAD {kthread:#010x}, TLS {tls:#010x}");
    }
    let _ = writeln!(report);
    unsafe {
        libc::write(2, report.bytes.as_ptr().cast(), report.len);
        libc::_exit(128 + sig);
    }
}
