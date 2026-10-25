// The dispatcher's code after its table, `LIFT_TABLE`: the search, and the calls into lifted
// code by guest address.

static LIFT_CACHE: [core::sync::atomic::AtomicU64; 4096] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 4096];

/// A relaxed atomic load of a cache slot: one `ldr`, as `AtomicU64::load` compiles to, without the
/// check of the ordering that an unoptimized build keeps and that names `core`'s panics.
#[inline(always)]
unsafe fn lift_cache_load(slot: *const u64) -> u64 {
    let value: u64;
    unsafe {
        core::arch::asm!(
            "ldr {value}, [{slot}]",
            slot = in(reg) slot,
            value = out(reg) value,
            options(nostack, readonly, preserves_flags)
        );
    }
    value
}

/// A relaxed atomic store to a cache slot: one `str`.
#[inline(always)]
unsafe fn lift_cache_store(slot: *mut u64, value: u64) {
    unsafe {
        core::arch::asm!(
            "str {value}, [{slot}]",
            slot = in(reg) slot,
            value = in(reg) value,
            options(nostack, preserves_flags)
        );
    }
}

/// The function lifted at `target`, if there is one.
unsafe fn lift_find(target: u32) -> Option<Lifted> {
    unsafe {
        let slot = LIFT_CACHE
            .get_unchecked(((target ^ (target >> 12)) & 4095) as usize)
            .as_ptr();
        let seen = lift_cache_load(slot);
        if (seen >> 32) as u32 == target && seen as u32 != 0 {
            return Some(LIFT_TABLE.get_unchecked(seen as u32 as usize - 1).1);
        }
        let (mut low, mut high) = (0usize, LIFT_TABLE.len());
        while low < high {
            let mid = low + (high - low) / 2;
            if LIFT_TABLE.get_unchecked(mid).0 < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low < LIFT_TABLE.len() && LIFT_TABLE.get_unchecked(low).0 == target {
            lift_cache_store(slot, (u64::from(target) << 32) | (low as u64 + 1));
            return Some(LIFT_TABLE.get_unchecked(low).1);
        }
        None
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_call(c: *mut Cpu, target: u32, at: u32) {
    unsafe {
        if let Some(function) = lift_find(target) {
            run(c, function);
            return;
        }
        // A `longjmp` lands inside a function: that is not an entry, and the context knows it.
        if lift_longjmp(c, target) {
            return;
        }
        // Anything else is a kernel call or a missing function: the native hook decides.
        lift_native(c, target, at);
    }
}

/// Resuming a frame's call: calls the function in the next frame down, which resumes at its own
/// site, or, below the last frame, finishes the parked contact with the runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_resume_call(c: *mut Cpu) {
    unsafe {
        let depth = (*c).depth + 1;
        (*c).depth = depth;
        if depth < (*c).resume_depth {
            let at = (*(*c).frames.add(depth as usize)).function;
            match lift_find(at) {
                Some(function) => run(c, function),
                None => lift_trap(c, at, b"resuming a function that is not lifted"),
            }
        } else {
            lift_resume_innermost(c);
        }
        (*c).depth = depth - 1;
    }
}

/// Resumes a thread from its outermost frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_resume_root(c: *mut Cpu) {
    unsafe {
        (*c).depth = 0;
        let at = (*(*c).frames).function;
        match lift_find(at) {
            Some(function) => run(c, function),
            None => lift_trap(c, at, b"resuming a function that is not lifted"),
        }
    }
}
