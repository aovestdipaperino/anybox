//! Lifts 32-bit x86 functions to C or to Rust.
//!
//! Each function becomes a function in the host language that works on the guest's registers and
//! flags as a structure, and on guest memory through a base pointer. It is compiled for the host
//! (arm64 on Apple silicon), so the translated code runs natively. Arithmetic goes through helpers
//! that compute the flags, so the flag logic is written once.
//!
//! Two emitters share the discovery here. [`c`] writes C for clang, and is the default: the items
//! at the top of the crate are its. [`rust`] writes `no_std` Rust for `rustc`, so a build needs no
//! C compiler.
//!
//! Supported: integer moves, arithmetic, logic, shifts, multiplication and division, stack
//! operations, direct calls and jumps, conditional jumps and moves, `setcc`, `leave` and `ret`.
//! Anything else stops the translation with the address and the mnemonic, so an unsupported
//! instruction is never silently wrong.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};

/// The bytes of a title's image, by section, so jump tables in data can be read at build time.
#[derive(Debug, Clone, Default)]
pub struct Image {
    sections: Vec<(u32, Vec<u8>)>,
}

impl Image {
    /// An image made of one section.
    #[must_use]
    pub fn single(base: u32, bytes: Vec<u8>) -> Self {
        Self {
            sections: vec![(base, bytes)],
        }
    }

    /// The base address of every section.
    pub fn bases(&self) -> impl Iterator<Item = u32> + '_ {
        self.sections.iter().map(|(base, _)| *base)
    }

    /// Adds a section at `base`.
    pub fn push(&mut self, base: u32, bytes: Vec<u8>) {
        self.sections.push((base, bytes));
    }

    /// The bytes from `address` to the end of its section, if it is in one.
    #[must_use]
    pub fn at(&self, address: u32) -> Option<&[u8]> {
        self.sections.iter().find_map(|(base, bytes)| {
            let offset = address.checked_sub(*base)? as usize;
            // An address at the very end of a section is the start of the next one: an empty
            // remainder must not hide it.
            (offset < bytes.len()).then(|| &bytes[offset..])
        })
    }

    /// The little-endian word at `address`.
    #[must_use]
    pub fn read_u32(&self, address: u32) -> Option<u32> {
        let bytes = self.at(address)?;
        Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
    }

    /// Whether `address` is inside one of the sections that hold code.
    #[must_use]
    pub fn contains(&self, address: u32) -> bool {
        self.at(address).is_some()
    }
}

/// Why a function could not be lifted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftError {
    /// Guest address of the instruction that stopped the translation.
    pub address: u32,
    /// What is not supported there.
    pub message: String,
}

impl std::fmt::Display for LiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#010x}: {}", self.address, self.message)
    }
}

impl std::error::Error for LiftError {}

pub mod c;
pub mod rust;

pub use c::{PRELUDE, Program, lift, lift_program, lift_program_with_progress};

/// What discovery found of one function: its instructions, and the places in it that are not its
/// own code.
pub(crate) struct Discovered {
    pub(crate) insts: BTreeMap<u32, Instruction>,
    /// The functions it calls directly, which are lifted too.
    pub(crate) callees: Vec<u32>,
    /// Addresses the lifter cannot translate, with why. They trap if the code runs them.
    pub(crate) faults: BTreeMap<u32, String>,
    /// Hooked addresses the body reaches other than by calling them: the kernel's replacement
    /// runs there, as the patched jump does in the emulator.
    pub(crate) hooked: BTreeMap<u32, u32>,
    /// The jump tables, by the address of their jump.
    pub(crate) tables: BTreeMap<u32, Vec<(u32, u32)>>,
}

impl Discovered {
    /// The addresses of the function's own code: its instructions, hooked places and faults.
    pub(crate) fn body(&self) -> BTreeSet<u32> {
        let mut body: BTreeSet<u32> = self.insts.keys().copied().collect();
        body.extend(self.faults.keys().copied());
        body.extend(self.hooked.keys().copied());
        body
    }

    /// The places the lifter could not translate, each a trap if it runs.
    pub(crate) fn traps(&self) -> Vec<LiftError> {
        self.faults
            .iter()
            .map(|(&address, message)| unsupported(address, message.clone()))
            .collect()
    }
}

/// Discovers the function at `entry`. Discovery follows fall-through and every branch target, so
/// jumps out of the function's own range are lifted inline and stay correct, up to the other
/// `entries`, which are functions of their own and are tail-called.
///
/// # Errors
///
/// The function's first instruction cannot be translated.
pub(crate) fn discover(
    image: &Image,
    entry: u32,
    entries: &BTreeSet<u32>,
    hooks: &BTreeMap<u32, u32>,
) -> Result<Discovered, LiftError> {
    let mut insts: BTreeMap<u32, Instruction> = BTreeMap::new();
    let mut callees = Vec::new();
    // Addresses the lifter cannot translate, with why. They trap if the code runs them.
    let mut faults: BTreeMap<u32, String> = BTreeMap::new();
    // Hooked addresses the body reaches other than by calling them: the kernel's replacement
    // runs there, as the patched jump does in the emulator.
    let mut hooked: BTreeMap<u32, u32> = BTreeMap::new();
    let mut tables: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
    let mut work = VecDeque::from([entry]);
    while let Some(start) = work.pop_front() {
        let mut at = start;
        loop {
            if insts.contains_key(&at) || hooked.contains_key(&at) {
                break;
            }
            if at != entry
                && let Some(&trap) = hooks.get(&at)
            {
                hooked.insert(at, trap);
                break;
            }
            if at != entry && entries.contains(&at) {
                break;
            }
            let instruction = match decode_at(image, at) {
                Ok(instruction) => instruction,
                Err(error) if at != entry => {
                    faults.insert(at, error.message);
                    break;
                }
                Err(error) => return Err(error),
            };
            let flow = instruction.flow_control();
            let next = addr(instruction.next_ip());
            let direct = instruction.op0_kind() == OpKind::NearBranch32
                || instruction.op0_kind() == OpKind::NearBranch64;
            let stop = match flow {
                FlowControl::Next | FlowControl::IndirectCall => false,
                FlowControl::Call => {
                    if direct {
                        callees.push(addr(instruction.near_branch_target()));
                    }
                    false
                }
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch => {
                    if !direct {
                        faults.insert(at, "indirect jump".to_owned());
                        break;
                    }
                    let target = addr(instruction.near_branch_target());
                    work.push_back(target);
                    flow == FlowControl::UnconditionalBranch
                }
                // An interrupt or exception is a stop at run time, like the emulator's.
                FlowControl::Return | FlowControl::Interrupt | FlowControl::Exception => true,
                FlowControl::IndirectBranch => {
                    if let Some(entries) = jump_table(image, at, &instruction) {
                        for &(_, target) in &entries {
                            work.push_back(target);
                        }
                        tables.insert(at, entries);
                    } else if !is_tail_call(&instruction) {
                        faults.insert(
                            at,
                            format!("indirect jump without a table: {instruction:?}"),
                        );
                        break;
                    }
                    true
                }
                FlowControl::XbeginXabortXend => {
                    faults.insert(at, "transactional memory".to_owned());
                    break;
                }
            };
            insts.insert(at, instruction);
            if stop {
                break;
            }
            at = next;
        }
    }

    if let Some(message) = faults.get(&entry) {
        return Err(unsupported(entry, message.clone()));
    }
    if let Some(message) = faults.get(&entry) {
        return Err(unsupported(entry, message.clone()));
    }
    Ok(Discovered {
        insts,
        callees,
        faults,
        hooked,
        tables,
    })
}

/// What an emitter makes a function of.
pub(crate) enum Root {
    /// A function in the hooks: a call to its trap.
    Hooked(u32),
    /// A function that discovery found.
    Lifted(Discovered),
    /// A function that cannot be lifted: a stub that traps with the reason.
    Failed(LiftError),
}

/// What [`walk`] found: the functions that lifted (the hooked ones among them) and those that
/// failed.
pub(crate) struct Walked {
    pub(crate) lifted: BTreeSet<u32>,
    pub(crate) failures: Vec<LiftError>,
}

/// Walks the functions reachable from `roots`, each root and every function it calls directly,
/// and hands each to `emit`. `progress` hears after each function how many are done and how many
/// are known so far; the second grows as lifted functions name callees not seen before.
pub(crate) fn walk(
    image: &Image,
    roots: &[u32],
    hooks: &BTreeMap<u32, u32>,
    progress: &mut dyn FnMut(usize, usize),
    emit: &mut dyn FnMut(u32, Root),
) -> Walked {
    let entries: BTreeSet<u32> = roots.iter().copied().collect();
    let mut lifted = BTreeSet::new();
    let mut failed = BTreeSet::new();
    let mut failures = Vec::new();
    let mut pending: VecDeque<u32> = roots.iter().copied().collect();
    let mut known: BTreeSet<u32> = roots.iter().copied().collect();
    while let Some(function) = pending.pop_front() {
        if lifted.contains(&function) || failed.contains(&function) {
            continue;
        }
        // A hooked function never runs its own code: it is the kernel's replacement, at a trap.
        if let Some(&trap) = hooks.get(&function) {
            lifted.insert(function);
            emit(function, Root::Hooked(trap));
            continue;
        }
        match discover(image, function, &entries, hooks) {
            Ok(found) => {
                lifted.insert(function);
                known.extend(found.callees.iter().copied());
                pending.extend(found.callees.iter().copied());
                emit(function, Root::Lifted(found));
            }
            Err(error) => {
                failed.insert(function);
                failures.push(error.clone());
                emit(function, Root::Failed(error));
            }
        }
        progress(lifted.len() + failed.len(), known.len());
    }
    Walked { lifted, failures }
}

/// Decodes the instruction at guest address `at`.
fn decode_at(image: &Image, at: u32) -> Result<Instruction, LiftError> {
    let bytes = image.at(at).ok_or_else(|| LiftError {
        address: at,
        message: "address outside the image".into(),
    })?;
    let mut decoder = Decoder::with_ip(32, bytes, u64::from(at), DecoderOptions::NONE);
    let instruction = decoder.decode();
    if instruction.is_invalid() {
        return Err(LiftError {
            address: at,
            message: "invalid instruction".into(),
        });
    }
    Ok(instruction)
}

/// The cases of a `jmp [table + index * 4]` whose table is in the image: each index value the
/// jump can take and the code it goes to.
///
/// A bound before the jump (`cmp index, N`, `and index, M`, or a byte map) gives the exact set of
/// indexes. Without one, the indexes are a window around zero, which covers a table that the
/// code reaches with a biased index (`sub index, K` before a borrow test). Index values outside
/// the cases trap at run time. Returns `None` for any other indirect jump, or when no index leads
/// to code.
fn jump_table(image: &Image, at: u32, instruction: &Instruction) -> Option<Vec<(u32, u32)>> {
    if instruction.op_kind(0) != OpKind::Memory
        || instruction.memory_base() != Register::None
        || instruction.memory_index() == Register::None
        || instruction.memory_index_scale() != 4
    {
        return None;
    }
    let index = register_number(instruction.memory_index())?;
    let table = addr(instruction.memory_displacement64());
    let indexes: Vec<u32> = match bound_before(image, at, index) {
        Some(bound) => (0..=bound).collect(),
        None => match byte_map_count(image, at, index) {
            Some(count) => (0..count).collect(),
            None => (0..2 * UNBOUNDED_WINDOW)
                .map(|i| i.wrapping_sub(UNBOUNDED_WINDOW))
                .collect(),
        },
    };
    let cases: Vec<(u32, u32)> = indexes
        .into_iter()
        .filter_map(|value| {
            let entry = image.read_u32(table.wrapping_add(value.wrapping_mul(4)))?;
            image.contains(entry).then_some((value, entry))
        })
        .collect();
    (!cases.is_empty()).then_some(cases)
}

/// The number in a register's `ModRM` field.
fn register_number(register: Register) -> Option<u8> {
    match register {
        Register::EAX => Some(0),
        Register::ECX => Some(1),
        Register::EDX => Some(2),
        Register::EBX => Some(3),
        Register::EBP => Some(5),
        Register::ESI => Some(6),
        Register::EDI => Some(7),
        _ => None,
    }
}

/// The largest index a jump through `reg` can take, from the instruction that bounds it before
/// `at`: `cmp reg, N` (the index is at most N) or `and reg, M` (at most M).
///
/// A compiler may schedule stores and pushes between the bound and the jump (one title puts 22 bytes
/// there), so the bound is looked for up to [`BOUND_SEARCH`] bytes back. Past the first 16, a
/// candidate counts only when decoding from it lands on `at` through straight-line code that
/// leaves `reg` alone.
fn bound_before(image: &Image, at: u32, reg: u8) -> Option<u32> {
    let compare = 0xF8 | reg;
    let mask = 0xE0 | reg;
    for back in 3..=BOUND_SEARCH {
        let Some(start) = at.checked_sub(back) else {
            continue;
        };
        let Some(bytes) = image.at(start) else {
            continue;
        };
        let bound = match bytes {
            [0x83, m, n, ..] if *m == compare || *m == mask => u32::from(*n),
            [0x81, m, n0, n1, n2, n3, ..] if *m == compare || *m == mask => {
                u32::from_le_bytes([*n0, *n1, *n2, *n3])
            }
            _ => continue,
        };
        if back <= 16 || reaches_untouched(image, start, at, reg) {
            return Some(bound);
        }
    }
    None
}

/// How far back from a jump table's jump its bound is looked for.
const BOUND_SEARCH: u32 = 48;

/// Whether the instructions decoded from `start` reach `at` exactly, through code that falls
/// through or branches conditionally, with no instruction after the first writing `reg`.
fn reaches_untouched(image: &Image, start: u32, at: u32, reg: u8) -> bool {
    let Ok(first) = decode_at(image, start) else {
        return false;
    };
    let mut pc = addr(first.next_ip());
    while pc < at {
        let Ok(instruction) = decode_at(image, pc) else {
            return false;
        };
        let straight = matches!(
            instruction.flow_control(),
            FlowControl::Next | FlowControl::ConditionalBranch
        );
        if !straight || writes_register(&instruction, reg) {
            return false;
        }
        pc = addr(instruction.next_ip());
    }
    pc == at
}

/// Whether `instruction` may write the 32-bit register numbered `reg`, or a part of it.
fn writes_register(instruction: &Instruction, reg: u8) -> bool {
    if matches!(
        instruction.mnemonic(),
        Mnemonic::Cmp | Mnemonic::Test | Mnemonic::Push | Mnemonic::Bt
    ) {
        return false;
    }
    // Instructions that write a register they do not name as an operand.
    if matches!(
        instruction.mnemonic(),
        Mnemonic::Cdq
            | Mnemonic::Cpuid
            | Mnemonic::Rdtsc
            | Mnemonic::Mul
            | Mnemonic::Div
            | Mnemonic::Imul
            | Mnemonic::Idiv
            | Mnemonic::Loop
            | Mnemonic::Loope
            | Mnemonic::Loopne
    ) || instruction.has_rep_prefix()
        || instruction.has_repe_prefix()
        || instruction.has_repne_prefix()
    {
        return true;
    }
    (0..instruction.op_count()).any(|operand| {
        instruction.op_kind(operand) == OpKind::Register
            && (operand == 0 || matches!(instruction.mnemonic(), Mnemonic::Xchg | Mnemonic::Xadd))
            && register_number(instruction.op_register(operand).full_register32()) == Some(reg)
    })
}

/// The entry count of a switch that indexes its jump table through a byte map: a
/// `movzx index, byte [bounded + map]` just before `at`, bounded by a `cmp bounded, N`. The count
/// is the largest map entry for indexes 0 to N, plus one.
fn byte_map_count(image: &Image, at: u32, index: u8) -> Option<u32> {
    for back in 4..=16u32 {
        let Some(start) = at.checked_sub(back) else {
            continue;
        };
        let Some(bytes) = image.at(start) else {
            continue;
        };
        // movzx r32, byte ptr [r32 + disp32]: 0F B6, ModRM with mod 10 and rm as the bounded register.
        let [0x0F, 0xB6, modrm, d0, d1, d2, d3, ..] = bytes else {
            continue;
        };
        if (modrm >> 6) != 0b10 || ((modrm >> 3) & 7) != index {
            continue;
        }
        let bounded = modrm & 7;
        let map = u32::from_le_bytes([*d0, *d1, *d2, *d3]);
        let bound = bound_before(image, start, bounded)?;
        let mut largest = 0u32;
        for i in 0..=bound {
            let value = u32::from(*image.at(map.checked_add(i)?)?.first()?);
            largest = largest.max(value);
        }
        return Some(largest + 1);
    }
    None
}

/// How far either side of zero an unbounded jump table's index is taken to range.
const UNBOUNDED_WINDOW: u32 = 16;

/// The start of MSVC's `_setjmp3`: `mov edx, [esp+4]` then it stores EBP, EBX, EDI and ESI into the
/// buffer `edx` points to.
const SETJMP_PROLOGUE: [u8; 15] = [
    0x8B, 0x54, 0x24, 0x04, 0x89, 0x2A, 0x89, 0x5A, 0x04, 0x89, 0x7A, 0x08, 0x89, 0x72, 0x0C,
];

/// The addresses of the title's `setjmp` functions, found by their code.
fn find_setjmps(image: &Image) -> BTreeSet<u32> {
    let mut found = BTreeSet::new();
    for (base, bytes) in &image.sections {
        for (offset, window) in bytes.windows(SETJMP_PROLOGUE.len()).enumerate() {
            if window == SETJMP_PROLOGUE {
                found.insert(base + u32::try_from(offset).unwrap_or(0));
            }
        }
    }
    found
}

/// Whether `instruction` is a direct call to one of the title's `setjmp` functions.
fn is_setjmp_call(instruction: &Instruction, setjmps: &BTreeSet<u32>) -> bool {
    instruction.flow_control() == FlowControl::Call
        && matches!(
            instruction.op0_kind(),
            OpKind::NearBranch32 | OpKind::NearBranch64
        )
        && setjmps.contains(&addr(instruction.near_branch_target()))
}

/// Whether control goes on to the next instruction after `instruction`, at least sometimes.
fn falls_through(instruction: &Instruction) -> bool {
    !matches!(
        instruction.flow_control(),
        FlowControl::Return
            | FlowControl::UnconditionalBranch
            | FlowControl::IndirectBranch
            | FlowControl::Interrupt
            | FlowControl::Exception
    )
}

/// The low 32 bits of a decoded value. Immediates and displacements arrive sign-extended to 64
/// bits, and the guest uses only the low half.
fn addr(value: u64) -> u32 {
    u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX)
}

fn unsupported(address: u32, what: impl Into<String>) -> LiftError {
    LiftError {
        address,
        message: what.into(),
    }
}

/// Whether an indirect jump is a tail call through memory with no index: an import's slot, or a
/// function pointer in a structure. A computed jump (with an index) is never taken for one.
fn is_tail_call(instruction: &Instruction) -> bool {
    // `jmp [slot]` through an import's slot, or `jmp reg` through a function pointer in a register:
    // the callee returns to our caller. A jump through an indexed table is a switch, not this.
    match instruction.op_kind(0) {
        OpKind::Memory => instruction.memory_index() == Register::None,
        OpKind::Register => true,
        _ => false,
    }
}

/// The condition code of a `jcc`, `setcc` or `cmovcc` mnemonic.
fn condition_code(mnemonic: Mnemonic) -> Option<i32> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    Some(match mnemonic {
        Jp | Setp | Cmovp => 10,
        Jnp | Setnp | Cmovnp => 11,
        Jo | Seto | Cmovo => 0,
        Jno | Setno | Cmovno => 1,
        Jb | Setb | Cmovb => 2,
        Jae | Setae | Cmovae => 3,
        Je | Sete | Cmove => 4,
        Jne | Setne | Cmovne => 5,
        Jbe | Setbe | Cmovbe => 6,
        Ja | Seta | Cmova => 7,
        Js | Sets | Cmovs => 8,
        Jns | Setns | Cmovns => 9,
        Jl | Setl | Cmovl => 12,
        Jge | Setge | Cmovge => 13,
        Jle | Setle | Cmovle => 14,
        Jg | Setg | Cmovg => 15,
        _ => return None,
    })
}

#[cfg(test)]
mod image_tests {
    use super::Image;

    #[test]
    fn an_address_at_the_end_of_one_section_is_in_the_next() {
        let mut image = Image::single(0x1000, vec![1, 2, 3, 4]);
        image.push(0x1004, vec![9, 8]);
        assert_eq!(image.at(0x1003), Some(&[4u8][..]));
        assert_eq!(
            image.at(0x1004),
            Some(&[9u8, 8][..]),
            "not the empty end of the first"
        );
        assert_eq!(image.at(0x1006), None);
        assert_eq!(image.at(0xfff), None);
    }
}
