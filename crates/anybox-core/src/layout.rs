//! The guest addresses the lifted program and the runtime agree on.
//!
//! A call to a kernel function, or to a statically linked library function the runtime replaces,
//! is compiled as a call to a trap address, which the runtime recognises. These are the runtime's
//! own constants: change them only together.

/// First trap address; kernel ordinal `n` traps at `TRAP_BASE + 4 * n`.
const TRAP_BASE: u32 = 0x8001_0000;
/// First trap address of the hooks that replace statically linked library functions.
const HOOK_TRAP_BASE: u32 = TRAP_BASE + 0x800;
/// Return address of every thread start routine; trapping here ends the thread.
const THREAD_RETURN: u32 = TRAP_BASE + 0xFFC;

/// How many library hooks fit before the thread-return trap.
pub const MAX_HOOKS: u32 = (THREAD_RETURN - HOOK_TRAP_BASE) / 4;

/// The trap address bound to a kernel function ordinal.
#[must_use]
pub fn kernel_function_address(ordinal: u32) -> u32 {
    TRAP_BASE + ordinal * 4
}

/// The trap address of the library hook with this index.
#[must_use]
pub fn hook_trap_address(index: u32) -> u32 {
    HOOK_TRAP_BASE + index * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trap_addresses_match_the_runtime() {
        assert_eq!(kernel_function_address(0), 0x8001_0000);
        assert_eq!(kernel_function_address(277), 0x8001_0000 + 277 * 4);
        assert_eq!(hook_trap_address(0), 0x8001_0800);
        assert_eq!(MAX_HOOKS, (0xFFC - 0x800) / 4);
    }

    #[test]
    fn hooks_stay_below_the_thread_return() {
        assert!(hook_trap_address(MAX_HOOKS - 1) < THREAD_RETURN);
    }
}
