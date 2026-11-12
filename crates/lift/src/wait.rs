//! Loops that only wait: a title waiting for its vertical-blank count, a flag another thread
//! sets, a GPU register. Such a loop writes nothing and computes each turn from memory alone, so
//! it can end only when something else changes memory, and its turns until then do nothing. The
//! lifter marks its backward branch so the runtime can let others run, or sleep, instead.

use std::collections::BTreeSet;

use iced_x86::{FlowControl, InstructionInfoFactory, Mnemonic, OpAccess, Register};

use super::{Discovered, addr};

impl Discovered {
    /// Whether the backward branch at `branch` to `target` closes a loop that only waits: its
    /// instructions run straight from `target` to `branch`, read memory but write none, leave
    /// its other exits only forward or out of the loop, and each turn sets every register and
    /// flag it changes before reading it, so a turn depends only on memory and on registers the
    /// loop never changes. Turns over memory that does not change all go the same way.
    pub(crate) fn only_waits(&self, target: u32, branch: u32) -> bool {
        let turn: Vec<_> = self.insts.range(target..=branch).collect();
        let straight = turn.first().is_some_and(|(at, _)| **at == target)
            && turn.last().is_some_and(|(at, _)| **at == branch)
            && turn
                .windows(2)
                .all(|pair| addr(pair[0].1.next_ip()) == *pair[1].0);
        if !straight {
            return false;
        }
        let mut factory = InstructionInfoFactory::new();
        // The full registers the loop writes anywhere: a turn must set each before reading it.
        let mut changed: BTreeSet<Register> = BTreeSet::new();
        for (_, instruction) in &turn {
            for used in factory.info(instruction).used_registers() {
                if matches!(used.access(), OpAccess::Write | OpAccess::ReadWrite) {
                    changed.insert(used.register().full_register32());
                }
            }
        }
        let mut set: BTreeSet<Register> = BTreeSet::new();
        let mut flags_set = 0u32;
        for &(&at, instruction) in &turn {
            let last = at == branch;
            let allowed = match instruction.flow_control() {
                FlowControl::Next => matches!(
                    instruction.mnemonic(),
                    Mnemonic::Cmp
                        | Mnemonic::Test
                        | Mnemonic::Bt
                        | Mnemonic::Mov
                        | Mnemonic::Movzx
                        | Mnemonic::Movsx
                        | Mnemonic::And
                        | Mnemonic::Or
                        | Mnemonic::Xor
                        | Mnemonic::Add
                        | Mnemonic::Sub
                        | Mnemonic::Nop
                        | Mnemonic::Pause
                ),
                // Another exit leaves the loop; only the last branch closes it.
                FlowControl::ConditionalBranch if instruction.is_jcc_short_or_near() => {
                    let to = addr(instruction.near_branch_target());
                    if last {
                        to == target
                    } else {
                        !(target..=branch).contains(&to)
                    }
                }
                FlowControl::UnconditionalBranch if last => {
                    instruction.mnemonic() == Mnemonic::Jmp
                        && addr(instruction.near_branch_target()) == target
                }
                _ => false,
            };
            if !allowed || instruction.has_lock_prefix() {
                return false;
            }
            if instruction.rflags_read() & !flags_set != 0 {
                return false;
            }
            flags_set |= instruction.rflags_modified();
            let info = factory.info(instruction);
            if info.used_memory().iter().any(|memory| {
                !matches!(
                    memory.access(),
                    OpAccess::Read | OpAccess::CondRead | OpAccess::NoMemAccess
                )
            }) {
                return false;
            }
            for used in info.used_registers() {
                let register = used.register();
                let full = register.full_register32();
                let reads = matches!(
                    used.access(),
                    OpAccess::Read | OpAccess::CondRead | OpAccess::ReadWrite
                );
                let writes = match used.access() {
                    OpAccess::Write | OpAccess::ReadWrite => true,
                    OpAccess::Read
                    | OpAccess::CondRead
                    | OpAccess::NoMemAccess
                    | OpAccess::None => false,
                    // A write that may not happen leaves the old value: neither set nor not.
                    _ => return false,
                };
                // A part of a register written keeps the rest: a read of the old value.
                let partial = writes && register != full;
                if (reads || partial) && changed.contains(&full) && !set.contains(&full) {
                    return false;
                }
                if writes {
                    if full == Register::ESP || register.is_segment_register() {
                        return false;
                    }
                    set.insert(full);
                }
            }
        }
        true
    }
}
