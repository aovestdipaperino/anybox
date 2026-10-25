//! The guest CPUs of the host threads that run guest threads, as the glue tells the runtime
//! where they are: read to save a frozen thread, and filled to resume one.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};

use crate::kernel::thread::Park;

/// One frame record, laid out as the lifted code's `lift_frame` (C) or `Frame` (Rust).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frame {
    /// The lifted function running in the host frame.
    pub function: u32,
    /// `LIFT_K_CALL`, `LIFT_K_RET` or `LIFT_K_POLL`.
    pub kind: u32,
    /// The site it is at.
    pub site: u32,
}

/// A `setjmp` context's guest side, laid out as the first words of `lift_jmp` and as
/// `lift_jmp_saved` (`Jmp` and `JmpSaved` in lifted Rust).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Jump {
    pub buf: u32,
    pub resume: u32,
    pub depth: u32,
    pub site: u32,
}

/// A guest thread's CPU, frame records and `setjmp` contexts, as a snapshot keeps them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestCpu {
    pub tid: u32,
    /// `lift_cpu`'s bytes (`Cpu` in lifted Rust, with the same layout).
    pub cpu: Vec<u8>,
    /// The frames to resume down, outermost first.
    pub frames: Vec<Frame>,
    pub jumps: Vec<Jump>,
}

/// `LIFT_K_POLL`.
pub const KIND_POLL: u32 = 2;
/// The room in `lift_jmp_restore`.
const RESTORE_ROOM: usize = 256;
/// The room in a thread's frame records (`LIFT_FRAMES`).
const FRAME_ROOM: usize = 131_072;

/// Where a host thread's guest CPU is (`anyxbox_cpu_layout` in the C glue, `CpuLayout` in the
/// Rust glue): all in state that lives as long as the thread, its thread-local storage in C and a
/// block the glue allocates for it in Rust.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub cpu: *mut u8,
    pub cpu_size: u32,
    pub depth: *mut u32,
    pub frames: *mut Frame,
    pub jumps: *mut *mut Jump,
    pub jump_top: *mut u32,
    pub restore: *mut Jump,
    pub restore_count: *mut u32,
}

// A layout is read only while its thread is parked for good (the title is frozen) or by the
// thread itself.
unsafe impl Send for Layout {}

/// The layouts by guest thread, of the host threads still running guest code.
static LAYOUTS: LazyLock<Mutex<HashMap<u32, Layout>>> = LazyLock::new(Mutex::default);
/// The CPUs of the threads a restore is resuming, until each takes its own.
static RESUMES: Mutex<Vec<GuestCpu>> = Mutex::new(Vec::new());

/// Records where guest thread `tid`'s CPU is.
pub(crate) fn register(tid: u32, layout: Layout) {
    LAYOUTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(tid, layout);
}

/// Forgets guest thread `tid`'s layout: its host thread runs no more guest code, and its
/// thread-local storage goes with it.
pub(crate) fn forget(tid: u32) {
    LAYOUTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&tid);
}

/// Forgets every layout: after a reset, the threads re-register.
pub(crate) fn forget_all() {
    LAYOUTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// Where guest thread `tid`'s CPU is, if it said.
fn layout(tid: u32) -> Option<Layout> {
    LAYOUTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&tid)
        .copied()
}

/// Reads parked thread `tid`'s CPU. At a poll its own frame is the innermost; elsewhere the
/// innermost frame is the one that made the kernel call.
///
/// # Errors
///
/// A message when the thread never registered, is in a kernel call no lifted frame made, or has
/// more frames or `setjmp` contexts than a resume has room for (a snapshot that could not resume
/// is not written; the title carries on).
pub(crate) fn read(tid: u32, park: Park) -> Result<GuestCpu, String> {
    let layout = layout(tid).ok_or_else(|| format!("thread {tid} never said where its CPU is"))?;
    // SAFETY: the layout points into the live thread-local storage of a thread that is parked
    // and stays so while it is read.
    unsafe {
        let depth = *layout.depth;
        let count = if park == Park::Poll {
            depth.checked_add(1)
        } else {
            Some(depth)
        }
        .map_or(usize::MAX, |count| count as usize);
        if count == 0 {
            return Err(format!(
                "thread {tid} is in a kernel call outside the lifted code"
            ));
        }
        if count > FRAME_ROOM {
            return Err(format!("thread {tid} is {count} frames deep"));
        }
        let top = (*layout.jump_top) as usize;
        if top > RESTORE_ROOM {
            return Err(format!("thread {tid} holds {top} setjmp contexts"));
        }
        let mut jumps = Vec::with_capacity(top);
        for i in 0..top {
            let jump = *layout.jumps.add(i);
            if jump.is_null() {
                return Err(format!("thread {tid} has no setjmp context {i}"));
            }
            jumps.push(*jump);
        }
        let cpu = std::slice::from_raw_parts(layout.cpu, layout.cpu_size as usize).to_vec();
        let frames = std::slice::from_raw_parts(layout.frames, count).to_vec();
        Ok(GuestCpu {
            tid,
            cpu,
            frames,
            jumps,
        })
    }
}

/// Fills the CPU at `layout` from `saved`; returns the number of frames to resume down.
///
/// # Errors
///
/// A message when the snapshot's CPU does not fit this build's.
///
/// # Safety
///
/// `layout` must point to the calling thread's own guest CPU.
pub(crate) unsafe fn load(layout: &Layout, saved: &GuestCpu) -> Result<u32, String> {
    if saved.cpu.len() != layout.cpu_size as usize {
        return Err(format!(
            "a CPU of {} bytes, not {}",
            saved.cpu.len(),
            layout.cpu_size
        ));
    }
    if saved.jumps.len() > RESTORE_ROOM || saved.frames.len() > FRAME_ROOM {
        return Err("more frames or contexts than there is room for".to_owned());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(saved.cpu.as_ptr(), layout.cpu, saved.cpu.len());
        std::ptr::copy_nonoverlapping(saved.frames.as_ptr(), layout.frames, saved.frames.len());
        std::ptr::copy_nonoverlapping(saved.jumps.as_ptr(), layout.restore, saved.jumps.len());
        *layout.restore_count = u32::try_from(saved.jumps.len()).unwrap_or(0);
    }
    u32::try_from(saved.frames.len()).map_err(|error| error.to_string())
}

/// Hands the restored threads' CPUs over, for each to take its own.
pub(crate) fn set_resumes(cpus: Vec<GuestCpu>) {
    *RESUMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = cpus;
}

/// Takes guest thread `tid`'s CPU from those a restore is resuming.
fn take_resume(tid: u32) -> Option<GuestCpu> {
    let mut resumes = RESUMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let at = resumes.iter().position(|cpu| cpu.tid == tid)?;
    Some(resumes.swap_remove(at))
}

/// How a resumed thread finishes the contact it was parked in (`anyxbox_resume` in the glue).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Resume {
    /// 0: a kernel call's result (`eax`, `esp`, `edx`); 1: make the kernel call to `target`; 2:
    /// a poll, nothing to finish.
    pub kind: u32,
    pub eax: u32,
    pub esp: u32,
    pub edx: u32,
    pub target: u32,
}

impl Resume {
    /// How a thread parked at `park` finishes, woken with `resume`.
    #[must_use]
    pub fn of(park: Park, resume: u32) -> Self {
        match park {
            Park::Native { esp, edx, eax } => Self {
                kind: 0,
                eax: eax.unwrap_or(resume),
                esp,
                edx,
                target: 0,
            },
            Park::Call { target } => Self {
                kind: 1,
                target,
                ..Self::default()
            },
            Park::Poll | Park::Start | Park::Busy => Self {
                kind: 2,
                ..Self::default()
            },
        }
    }
}

/// Called by the glue as each host thread that runs a guest thread starts: where its guest CPU
/// is.
///
/// # Safety
///
/// `layout` must point to a layout of the calling thread's own thread-local storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_register_cpu(layout: *const Layout) {
    if let Some(layout) = unsafe { layout.as_ref() } {
        register(crate::abi::me(), *layout);
    }
}

/// Called by the glue on a resumed host thread, after it registered: fills its CPU, frames and
/// `setjmp` contexts from the snapshot, and returns the number of frames to resume down.
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_resume_load() -> u32 {
    let me = crate::abi::me();
    let (Some(layout), Some(saved)) = (layout(me), take_resume(me)) else {
        eprintln!("anyxbox: thread {me} has nothing to resume");
        super::discard();
        std::process::exit(1);
    };
    // SAFETY: the calling thread registered `layout` for itself.
    match unsafe { load(&layout, &saved) } {
        Ok(count) => count,
        Err(error) => {
            eprintln!("anyxbox: cannot resume thread {me}: {error}");
            super::discard();
            std::process::exit(1);
        }
    }
}

/// Called by the glue when the lifted code traps while a thread is resuming down its frames (a
/// site the function does not have, a function that is not lifted): the snapshot cannot resume,
/// so it is deleted and the next launch boots afresh.
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_resume_failed() {
    eprintln!("anyxbox: the snapshot could not be resumed");
    super::discard();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::thread::Park;

    /// A host thread's guest CPU, frames and contexts, owned by the test.
    struct Fake {
        cpu: Vec<u8>,
        depth: u32,
        frames: Vec<Frame>,
        jump: Jump,
        jumps: Vec<*mut Jump>,
        jump_top: u32,
        restore: Vec<Jump>,
        restore_count: u32,
    }

    impl Fake {
        // Boxed so it stays put: `jumps[0]` points into it.
        #[allow(clippy::unnecessary_box_returns)]
        fn new() -> Box<Self> {
            let mut fake = Box::new(Self {
                cpu: vec![0; 64],
                depth: 0,
                frames: vec![Frame::default(); 8],
                jump: Jump::default(),
                jumps: vec![std::ptr::null_mut(); 4],
                jump_top: 0,
                restore: vec![Jump::default(); RESTORE_ROOM],
                restore_count: 0,
            });
            fake.jumps[0] = &raw mut fake.jump;
            fake
        }

        fn layout(&mut self) -> Layout {
            Layout {
                cpu: self.cpu.as_mut_ptr(),
                cpu_size: 64,
                depth: &raw mut self.depth,
                frames: self.frames.as_mut_ptr(),
                jumps: self.jumps.as_mut_ptr(),
                jump_top: &raw mut self.jump_top,
                restore: self.restore.as_mut_ptr(),
                restore_count: &raw mut self.restore_count,
            }
        }
    }

    #[test]
    fn a_parked_thread_is_read_and_loaded_into_another() {
        let mut parked = Fake::new();
        parked.cpu[0] = 0xAB;
        parked.depth = 2;
        parked.frames[0] = Frame {
            function: 0x1000,
            kind: 0,
            site: 0x1005,
        };
        parked.frames[1] = Frame {
            function: 0x1010,
            kind: 0,
            site: 0x1016,
        };
        parked.jump = Jump {
            buf: 0x2000,
            resume: 0x1090,
            depth: 0,
            site: 0,
        };
        parked.jump_top = 1;
        register(77, parked.layout());
        let native = Park::Native {
            esp: 0,
            edx: 0,
            eax: None,
        };
        let saved = read(77, native).unwrap();
        assert_eq!(saved.frames.len(), 2, "a kernel call: the depth's frames");
        assert_eq!(saved.jumps, [parked.jump]);
        parked.depth = 1;
        parked.frames[1].kind = KIND_POLL;
        assert_eq!(
            read(77, Park::Poll).unwrap().frames.len(),
            2,
            "a poll: its own frame too"
        );

        let mut fresh = Fake::new();
        let layout = fresh.layout();
        let count = unsafe { load(&layout, &saved) }.unwrap();
        assert_eq!(count, 2);
        assert_eq!(fresh.cpu[0], 0xAB);
        assert_eq!(fresh.frames[1].site, 0x1016);
        assert_eq!((fresh.restore_count, fresh.restore[0]), (1, parked.jump));
    }

    #[test]
    fn a_thread_in_a_kernel_call_outside_lifted_code_cannot_be_saved() {
        let mut parked = Fake::new();
        register(78, parked.layout());
        let error = read(
            78,
            Park::Call {
                target: 0x8001_0000,
            },
        )
        .unwrap_err();
        assert!(error.contains("outside the lifted code"), "{error}");
        assert!(
            read(79, Park::Poll).is_err(),
            "a thread that never registered"
        );
    }

    #[test]
    fn a_thread_that_could_not_resume_is_not_saved() {
        let mut parked = Fake::new();
        parked.depth = u32::MAX;
        register(81, parked.layout());
        let error = read(81, Park::Poll).unwrap_err();
        assert!(error.contains("frames deep"), "{error}");
        let native = Park::Native {
            esp: 0,
            edx: 0,
            eax: None,
        };
        parked.depth = u32::try_from(FRAME_ROOM).unwrap() + 1;
        assert!(read(81, native).unwrap_err().contains("frames deep"));
        parked.depth = 1;
        parked.jump_top = u32::try_from(RESTORE_ROOM).unwrap() + 1;
        assert!(read(81, native).unwrap_err().contains("setjmp contexts"));
        parked.jump_top = 2;
        let error = read(81, native).unwrap_err();
        assert!(error.contains("no setjmp context 1"), "{error}");
        parked.jump_top = 1;
        assert!(read(81, native).is_ok());
        forget(81);
    }

    #[test]
    fn a_thread_that_leaves_guest_code_is_forgotten() {
        let mut parked = Fake::new();
        register(80, parked.layout());
        assert!(layout(80).is_some());
        forget(80);
        assert!(layout(80).is_none(), "no layout is left behind");
        assert!(read(80, Park::Poll).is_err());
    }

    #[test]
    fn a_resumed_thread_finishes_where_it_was_parked() {
        let native = Resume::of(
            Park::Native {
                esp: 0x10,
                edx: 0x20,
                eax: None,
            },
            5,
        );
        assert_eq!(
            (native.kind, native.eax, native.esp, native.edx),
            (0, 5, 0x10, 0x20)
        );
        let yielded = Resume::of(
            Park::Native {
                esp: 0x10,
                edx: 0x20,
                eax: Some(0),
            },
            5,
        );
        assert_eq!(yielded.eax, 0);
        let call = Resume::of(
            Park::Call {
                target: 0x8001_0004,
            },
            5,
        );
        assert_eq!((call.kind, call.target), (1, 0x8001_0004));
        assert_eq!(Resume::of(Park::Poll, 5).kind, 2);
    }
}
