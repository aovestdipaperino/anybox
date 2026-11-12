//! A census of a title's code as the lifter sees it: which instructions it holds, how many of the
//! flags they compute any later instruction can read, and which sequences recur. It measures where
//! the lifted code spends its effort, before any change to the emitters.

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{CpuidFeature, FlowControl, Instruction, Mnemonic, OpKind, RflagsBits};

use super::{Discovered, Image, Root, addr, falls_through, walk};

/// The flags the lifted code keeps, each as computed by the `alu_*` helpers.
const FLAGS: [(u32, &str); 5] = [
    (RflagsBits::CF, "CF"),
    (RflagsBits::ZF, "ZF"),
    (RflagsBits::SF, "SF"),
    (RflagsBits::OF, "OF"),
    (RflagsBits::PF, "PF"),
];
const KEPT: u32 =
    RflagsBits::CF | RflagsBits::ZF | RflagsBits::SF | RflagsBits::OF | RflagsBits::PF;

/// What the census counted over every lifted function.
#[derive(Debug, Default)]
pub struct Census {
    /// Functions lifted, and functions that failed to lift.
    pub functions: usize,
    pub failed: usize,
    /// Instructions in the lifted functions.
    pub instructions: u64,
    /// Instructions by mnemonic.
    pub mnemonics: BTreeMap<Mnemonic, u64>,
    /// Pairs of instructions where the first falls through to the second, by mnemonic.
    pub pairs: BTreeMap<(Mnemonic, Mnemonic), u64>,
    /// Instructions that write at least one kept flag.
    pub flag_writers: u64,
    /// Of those, the ones none of whose kept flags any later instruction can read.
    pub flag_writers_dead: u64,
    /// For each kept flag, in the order of `FLAGS`: the writes, and the writes no later
    /// instruction can read. Flags are taken as dead across a call and at a return, as the
    /// calling convention leaves them; anywhere control leaves the function otherwise they are
    /// taken as read.
    pub flag_writes: [u64; 5],
    pub flag_dead: [u64; 5],
    /// The same dead writes with flags taken as read across calls and returns: what liveness
    /// gains without trusting the calling convention.
    pub flag_dead_strict: [u64; 5],
    /// String instructions by form, `rep movsd` and the like.
    pub strings: BTreeMap<String, u64>,
    /// Instructions of the x87, MMX and SSE units.
    pub x87: u64,
    pub mmx: u64,
    pub sse: u64,
    /// Instructions that read or write memory, each a check for the GPU's registers.
    pub memory: u64,
    /// Calls by a direct address and through a register or memory.
    pub direct_calls: u64,
    pub indirect_calls: u64,
    /// Backward branches, each a preemption poll.
    pub backward_branches: u64,
    /// Indirect jumps through a table.
    pub jump_tables: u64,
    /// Places where a body runs into another function's entry and tail-calls it: a function cut
    /// in two.
    pub splits: u64,
}

/// Counts the functions reachable from `roots`, as [`crate::lift_program_with_progress`] would
/// lift them with the same `weak` roots.
#[must_use]
pub fn census(
    image: &Image,
    roots: &[u32],
    weak: &BTreeSet<u32>,
    hooks: &BTreeMap<u32, u32>,
) -> Census {
    let mut census = Census::default();
    let walked = walk(image, roots, weak, hooks, &mut |_, _| {}, &mut |_, root| {
        if let Root::Lifted(found) = root {
            census.count(&found);
        }
    });
    census.functions = walked.lifted.len();
    census.failed = walked.failures.len();
    census
}

impl Census {
    fn count(&mut self, found: &Discovered) {
        let trusting = liveness(found, true);
        let strict = liveness(found, false);
        let body = found.body();
        for (&at, instruction) in &found.insts {
            let mnemonic = instruction.mnemonic();
            self.instructions += 1;
            if falls_through(instruction) && !body.contains(&addr(instruction.next_ip())) {
                self.splits += 1;
            }
            *self.mnemonics.entry(mnemonic).or_default() += 1;
            if instruction.flow_control() == FlowControl::Next {
                let next = addr(instruction.next_ip());
                if let Some(following) = found.insts.get(&next) {
                    *self
                        .pairs
                        .entry((mnemonic, following.mnemonic()))
                        .or_default() += 1;
                }
            }
            let written = instruction.rflags_modified() & KEPT;
            if written != 0 && !is_call(instruction) {
                self.flag_writers += 1;
                let live = trusting.get(&at).copied().unwrap_or(KEPT);
                let live_strict = strict.get(&at).copied().unwrap_or(KEPT);
                if written & live == 0 {
                    self.flag_writers_dead += 1;
                }
                for (index, &(flag, _)) in FLAGS.iter().enumerate() {
                    if written & flag != 0 {
                        self.flag_writes[index] += 1;
                        if live & flag == 0 {
                            self.flag_dead[index] += 1;
                        }
                        if live_strict & flag == 0 {
                            self.flag_dead_strict[index] += 1;
                        }
                    }
                }
            }
            if instruction.is_string_instruction() {
                let prefix = if instruction.has_rep_prefix() || instruction.has_repe_prefix() {
                    "rep "
                } else if instruction.has_repne_prefix() {
                    "repne "
                } else {
                    ""
                };
                let form = format!("{prefix}{mnemonic:?}").to_lowercase();
                *self.strings.entry(form).or_default() += 1;
            }
            let features = instruction.cpuid_features();
            if features.contains(&CpuidFeature::FPU) {
                self.x87 += 1;
            } else if features.contains(&CpuidFeature::MMX) {
                self.mmx += 1;
            } else if features.iter().any(|feature| {
                matches!(
                    feature,
                    CpuidFeature::SSE | CpuidFeature::SSE2 | CpuidFeature::SSE3
                )
            }) {
                self.sse += 1;
            }
            let reads_memory = (0..instruction.op_count())
                .any(|index| instruction.op_kind(index) == OpKind::Memory);
            if reads_memory && !matches!(mnemonic, Mnemonic::Lea | Mnemonic::Nop) {
                self.memory += 1;
            }
            match instruction.flow_control() {
                FlowControl::Call => self.direct_calls += 1,
                FlowControl::IndirectCall => self.indirect_calls += 1,
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch
                    if addr(instruction.near_branch_target()) <= at =>
                {
                    self.backward_branches += 1;
                }
                _ => {}
            }
        }
        self.jump_tables += found.tables.len() as u64;
    }

    /// The census as lines of text, the most common forms first.
    #[must_use]
    pub fn report(&self) -> Vec<String> {
        let n = self.instructions;
        let mut lines = vec![
            format!(
                "{} functions lifted, {} failed; {n} instructions; {} fall-through splits",
                self.functions, self.failed, self.splits
            ),
            format!(
                "memory operands {} ({}), direct calls {}, indirect calls {}, backward branches {}, jump tables {}",
                self.memory,
                percent(self.memory, n),
                self.direct_calls,
                self.indirect_calls,
                self.backward_branches,
                self.jump_tables
            ),
            format!(
                "x87 {} ({}), MMX {} ({}), SSE {} ({})",
                self.x87,
                percent(self.x87, n),
                self.mmx,
                percent(self.mmx, n),
                self.sse,
                percent(self.sse, n)
            ),
            String::new(),
            format!(
                "flag writers {} ({} of instructions); all kept flags dead in {} ({})",
                self.flag_writers,
                percent(self.flag_writers, n),
                self.flag_writers_dead,
                percent(self.flag_writers_dead, self.flag_writers)
            ),
            "flag  writes      dead (calling convention)   dead (strict)".to_owned(),
        ];
        for (index, &(_, name)) in FLAGS.iter().enumerate() {
            let writes = self.flag_writes[index];
            lines.push(format!(
                "{name:<4}  {writes:>9}  {:>9} {:>7}               {:>9} {:>7}",
                self.flag_dead[index],
                percent(self.flag_dead[index], writes),
                self.flag_dead_strict[index],
                percent(self.flag_dead_strict[index], writes)
            ));
        }
        lines.push(String::new());
        lines.push("string instructions:".to_owned());
        for (form, count) in top(
            self.strings
                .iter()
                .map(|(form, &count)| (form.clone(), count)),
            20,
        ) {
            lines.push(format!("  {count:>8}  {form}"));
        }
        lines.push(String::new());
        lines.push("mnemonics:".to_owned());
        for (mnemonic, count) in top(
            self.mnemonics
                .iter()
                .map(|(mnemonic, &count)| (format!("{mnemonic:?}").to_lowercase(), count)),
            40,
        ) {
            lines.push(format!("  {count:>8} {:>7}  {mnemonic}", percent(count, n)));
        }
        lines.push(String::new());
        lines.push("pairs (first falls through to second):".to_owned());
        for (pair, count) in top(
            self.pairs.iter().map(|((first, second), &count)| {
                (format!("{first:?} ; {second:?}").to_lowercase(), count)
            }),
            40,
        ) {
            lines.push(format!("  {count:>8}  {pair}"));
        }
        lines
    }
}

/// The `limit` largest counts, the largest first.
fn top(counts: impl Iterator<Item = (String, u64)>, limit: usize) -> Vec<(String, u64)> {
    let mut counts: Vec<(String, u64)> = counts.collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts.truncate(limit);
    counts
}

/// `part` of `whole` as a percentage with one decimal.
fn percent(part: u64, whole: u64) -> String {
    if whole == 0 {
        return "-".to_owned();
    }
    let tenths = part * 1000 / whole;
    format!("{}.{}%", tenths / 10, tenths % 10)
}

fn is_call(instruction: &Instruction) -> bool {
    matches!(
        instruction.flow_control(),
        FlowControl::Call | FlowControl::IndirectCall
    )
}

/// The kept flags each instruction of `found` writes for certain. A shift or rotate by `CL`
/// leaves the flags alone when the count is zero, and a repeated string instruction when `ECX`
/// is, so neither is a certain write; a shift by an immediate count of zero writes nothing.
fn certain_writes(instruction: &Instruction) -> u32 {
    let written = instruction.rflags_modified() & KEPT;
    let shift = matches!(
        instruction.mnemonic(),
        Mnemonic::Shl
            | Mnemonic::Sal
            | Mnemonic::Shr
            | Mnemonic::Sar
            | Mnemonic::Rol
            | Mnemonic::Ror
            | Mnemonic::Rcl
            | Mnemonic::Rcr
            | Mnemonic::Shld
            | Mnemonic::Shrd
    );
    if shift {
        let count = instruction.op_count() - 1;
        return match instruction.op_kind(count) {
            OpKind::Register => 0,
            OpKind::Immediate8 if instruction.immediate8().trailing_zeros() >= 5 => 0,
            _ => written,
        };
    }
    let repeated = instruction.has_rep_prefix()
        || instruction.has_repe_prefix()
        || instruction.has_repne_prefix();
    if instruction.is_string_instruction() && repeated {
        return 0;
    }
    written
}

/// The kept flags live after each instruction of `found`: those some path from it reads before
/// writing. With `trusting`, a call or a return ends every flag's life, as the calling
/// convention has it; else both are taken as reading every flag. Control that leaves the function
/// by any other way (a tail call, a hooked place, a trap) is taken as reading every flag.
fn liveness(found: &Discovered, trusting: bool) -> BTreeMap<u32, u32> {
    let mut live_in: BTreeMap<u32, u32> = found.insts.keys().map(|&at| (at, 0)).collect();
    let mut live_out: BTreeMap<u32, u32> = BTreeMap::new();
    let entry_of = |live_in: &BTreeMap<u32, u32>, target: u32| -> u32 {
        live_in.get(&target).copied().unwrap_or(KEPT)
    };
    loop {
        let mut changed = false;
        for (&at, instruction) in found.insts.iter().rev() {
            let next = addr(instruction.next_ip());
            let out = match instruction.flow_control() {
                FlowControl::Return => {
                    if trusting {
                        0
                    } else {
                        KEPT
                    }
                }
                FlowControl::Interrupt | FlowControl::Exception => KEPT,
                FlowControl::UnconditionalBranch => {
                    entry_of(&live_in, addr(instruction.near_branch_target()))
                }
                FlowControl::ConditionalBranch => {
                    entry_of(&live_in, addr(instruction.near_branch_target()))
                        | entry_of(&live_in, next)
                }
                FlowControl::IndirectBranch => match found.tables.get(&at) {
                    Some(cases) => cases
                        .iter()
                        .fold(0, |live, &(_, target)| live | entry_of(&live_in, target)),
                    None => KEPT,
                },
                _ if falls_through(instruction) => entry_of(&live_in, next),
                _ => KEPT,
            };
            let (reads, kills) = if is_call(instruction) {
                if trusting { (0, KEPT) } else { (KEPT, 0) }
            } else {
                (
                    instruction.rflags_read() & KEPT,
                    certain_writes(instruction),
                )
            };
            let into = reads | (out & !kills);
            if live_in.insert(at, into) != Some(into) {
                changed = true;
            }
            live_out.insert(at, out);
        }
        if !changed {
            return live_out;
        }
    }
}
