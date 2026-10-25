//! The Rust emitter: each function becomes a `no_std` Rust function, compiled to an object by
//! `rustc` and linked with the runtime. [`PRELUDE`] starts every unit.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use iced_x86::{FlowControl, Instruction, Mnemonic, OpKind, Register};

use super::{
    Discovered, Image, LiftError, Root, addr, condition_code, falls_through, find_setjmps,
    is_setjmp_call, is_tail_call, unsupported, walk,
};

mod mmx;
mod sse;
mod x87;

/// The Rust every lifted unit includes: the guest state and the helpers. It has no crate
/// attributes, so a unit puts its own first ([`UNIT_ATTRIBUTES`]), and a test can include it in a
/// program of its own.
pub const PRELUDE: &str = include_str!("../lifted/prelude.rs");

/// The definitions of the globals the lifted code shares. Exactly one unit of a program has them;
/// the others reach them through the prelude's declarations.
pub const GLOBALS: &str = "
/// The guest memory's base.
#[unsafe(no_mangle)]
pub static mut G: *mut u8 = core::ptr::null_mut();
/// The guest's FS segment base, the KPCR.
#[unsafe(no_mangle)]
pub static mut FS_BASE: u32 = 0;
";

/// The attributes a unit compiled on its own starts with.
pub const UNIT_ATTRIBUTES: &str = "#![no_std]\n";

/// The declarations of the lifted functions at `functions`, for a unit that calls them and does
/// not define them.
#[must_use]
pub fn declarations(functions: impl IntoIterator<Item = u32>) -> String {
    let mut rust = String::from(
        "\n#[allow(unused_imports)]\nuse lifted_functions::*;\n#[allow(dead_code)]\n\
         mod lifted_functions {\n    use super::{Cpu, Next};\n    unsafe extern \"C\" {\n",
    );
    for function in functions {
        writeln!(
            rust,
            "        pub fn fn_{function:08x}(c: *mut Cpu) -> Next;"
        )
        .unwrap();
    }
    rust.push_str("    }\n}\n");
    rust
}

/// One function's Rust, and the functions it names.
#[derive(Debug)]
struct Function {
    address: u32,
    rust: String,
    calls: BTreeSet<u32>,
}

/// A lifted program, in pieces: one function per lifted address, and the call dispatch.
#[derive(Debug, Default)]
pub struct Program {
    functions: Vec<Function>,
    /// The addresses of the functions that lifted, which the dispatcher finds.
    found: BTreeSet<u32>,
    /// The number of functions that lifted.
    pub lifted: usize,
    /// Places inside lifted functions that trap if they run, because the lifter could not
    /// translate them.
    pub traps: Vec<LiftError>,
    /// The functions that could not be lifted. Each has a stub that traps with the reason when it
    /// runs.
    pub failures: Vec<LiftError>,
}

impl Program {
    /// The Rust source of each function.
    #[must_use]
    pub fn bodies(&self) -> Vec<&str> {
        self.functions.iter().map(|f| f.rust.as_str()).collect()
    }

    /// The functions as `parts` units of about the same size, each a crate of its own that
    /// declares the functions it calls in other units.
    #[must_use]
    pub fn units(&self, parts: usize) -> Vec<String> {
        let total: usize = self.functions.iter().map(|f| f.rust.len()).sum();
        let target = total / parts.max(1) + 1;
        let mut units = Vec::new();
        let mut current: Vec<&Function> = Vec::new();
        let mut size = 0;
        for function in &self.functions {
            current.push(function);
            size += function.rust.len();
            if size >= target {
                units.push(unit(&std::mem::take(&mut current)));
                size = 0;
            }
        }
        if !current.is_empty() {
            units.push(unit(&current));
        }
        units
    }

    /// The unit of the dispatcher, which maps a guest address to its function.
    #[must_use]
    pub fn dispatch_unit(&self) -> String {
        format!(
            "{UNIT_ATTRIBUTES}{PRELUDE}{}{}",
            declarations(self.found.iter().copied()),
            dispatcher(&self.found)
        )
    }

    /// The whole program as one crate's source, globals included and without crate attributes, so
    /// a test can make it part of a program of its own.
    #[must_use]
    pub fn source(&self) -> String {
        let mut rust = format!("{PRELUDE}{GLOBALS}");
        for function in &self.functions {
            rust.push_str(&function.rust);
        }
        rust.push_str(&dispatcher(&self.found));
        rust
    }
}

/// A unit of `functions`, with the declarations of the functions they call in other units.
fn unit(functions: &[&Function]) -> String {
    let defined: BTreeSet<u32> = functions.iter().map(|f| f.address).collect();
    let called: BTreeSet<u32> = functions
        .iter()
        .flat_map(|f| f.calls.iter().copied())
        .filter(|address| !defined.contains(address))
        .collect();
    let mut rust = format!("{UNIT_ATTRIBUTES}{PRELUDE}{}", declarations(called));
    for function in functions {
        rust.push_str(&function.rust);
    }
    rust
}

/// Lifts the functions reachable from `roots`: each root and every function it calls directly.
///
/// Each root is a function of its own: a body that reaches another root, by a jump or by running
/// into it, tail-calls that root's function instead of holding a copy of its code.
///
/// A function the lifter cannot translate does not stop the others. It gets a stub that traps
/// with the reason, and it is listed in [`Program::failures`]. A function in `hooks` (function
/// address to trap address) is compiled as a call to its trap. With `trace`, every instruction
/// reports its address and registers when the unit is built with `--cfg lift_trace`.
#[must_use]
pub fn lift_program(
    image: &Image,
    roots: &[u32],
    hooks: &BTreeMap<u32, u32>,
    trace: bool,
) -> Program {
    lift_program_with_progress(image, roots, hooks, trace, &mut |_, _| {})
}

/// [`lift_program`], calling `progress` after each function with how many are done and how many
/// are known so far. The second grows as lifted functions name callees not seen before.
#[must_use]
pub fn lift_program_with_progress(
    image: &Image,
    roots: &[u32],
    hooks: &BTreeMap<u32, u32>,
    trace: bool,
    progress: &mut dyn FnMut(usize, usize),
) -> Program {
    let setjmps = find_setjmps(image);
    let mut functions = Vec::new();
    let mut traps = Vec::new();
    let walked = walk(image, roots, hooks, progress, &mut |function, root| {
        let (rust, calls) = match root {
            Root::Hooked(trap) => (
                format!(
                    "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn fn_{function:08x}(c: *mut Cpu) -> Next {{\n    \
                     unsafe {{\n        enter(c, {function:#010x});\n        \
                     if (*c).resuming != 0 {{\n            resume_call(c);\n            return END;\n        }}\n        \
                     call_rt(c, {function:#010x}, K_RET, {trap:#010x}, {function:#010x});\n        END\n    }}\n}}\n"
                ),
                BTreeSet::new(),
            ),
            Root::Lifted(found) => {
                let body = lift_function(function, &found, &setjmps, trace);
                traps.extend(body.traps);
                (body.rust, body.calls)
            }
            Root::Failed(error) => {
                let trap = trap_statement(function, &error.message);
                (
                    format!(
                        "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn fn_{function:08x}(c: *mut Cpu) -> Next {{\n    \
                         unsafe {{ {trap} }}\n}}\n"
                    ),
                    BTreeSet::new(),
                )
            }
        };
        functions.push(Function {
            address: function,
            rust,
            calls,
        });
    });
    Program {
        functions,
        lifted: walked.lifted.len(),
        found: walked.lifted,
        traps,
        failures: walked.failures,
    }
}

/// The dispatcher's search and calls, which follow its table.
const DISPATCH: &str = include_str!("../lifted/dispatch.rs");

/// The Rust of `rt_call`, which maps a guest address to its function, and of the calls that
/// resume a hibernated thread down its frames.
///
/// The functions are a sorted table searched by bisection, with a cache of the addresses called
/// last. The cache keeps each slot in one atomic word (the address above, the table index plus one
/// below), so threads need no lock: a slot is either a whole entry or one that does not match.
fn dispatcher(lifted: &BTreeSet<u32>) -> String {
    let mut rust = format!(
        "\n#[repr(C)]\nstruct LiftEntry(u32, Lifted);\n\n\
         static LIFT_TABLE: [LiftEntry; {}] = [\n",
        lifted.len()
    );
    for function in lifted {
        writeln!(rust, "    LiftEntry({function:#010x}, fn_{function:08x}),").unwrap();
    }
    rust.push_str("];\n");
    rust.push_str(DISPATCH);
    rust
}

/// Lifts the functions reachable from `entry`, and returns their Rust with the prelude first.
///
/// # Errors
///
/// Returns the first function the lifter cannot translate.
pub fn lift(image: &Image, entry: u32) -> Result<String, LiftError> {
    let program = lift_program(image, &[entry], &BTreeMap::new(), false);
    match program.failures.first() {
        Some(error) => Err(error.clone()),
        None => Ok(program.source()),
    }
}

struct Body {
    rust: String,
    /// Every function its code names, by a call or a tail call.
    calls: BTreeSet<u32>,
    /// Places in the function the lifter could not translate; each traps if it runs.
    traps: Vec<LiftError>,
}

/// A Rust statement that stops the program at `address` with `message`.
fn trap_statement(address: u32, message: &str) -> String {
    let mut literal = String::new();
    for byte in message.bytes() {
        match byte {
            b'"' | b'\\' => write!(literal, "\\x{byte:02x}").unwrap(),
            0x20..=0x7E => literal.push(char::from(byte)),
            _ => write!(literal, "\\x{byte:02x}").unwrap(),
        }
    }
    format!("lift_trap(c, {address:#010x}, b\"{literal}\");")
}

/// The addresses a lifted function's blocks start at: wherever something other than the
/// instruction before jumps to, a call returns to (a resume or a `longjmp` goes on there), or the
/// instruction before does not run into.
fn block_starts(
    entry: u32,
    insts: &BTreeMap<u32, Instruction>,
    tables: &BTreeMap<u32, Vec<(u32, u32)>>,
    body: &BTreeSet<u32>,
) -> BTreeSet<u32> {
    // The function starts at its entry, which need not be the lowest address.
    let mut starts = BTreeSet::from([entry]);
    let mut expected = None;
    for (&address, instruction) in insts {
        if expected != Some(address) {
            starts.insert(address);
        }
        let next = addr(instruction.next_ip());
        let direct = matches!(
            instruction.op0_kind(),
            OpKind::NearBranch32 | OpKind::NearBranch64
        );
        match instruction.flow_control() {
            FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch if direct => {
                starts.insert(addr(instruction.near_branch_target()));
            }
            FlowControl::Call | FlowControl::IndirectCall => {
                starts.insert(next);
            }
            _ => {}
        }
        if let Some(cases) = tables.get(&address) {
            starts.extend(cases.iter().map(|&(_, target)| target));
        }
        expected = falls_through(instruction).then_some(next);
    }
    starts.retain(|address| body.contains(address));
    starts
}

/// Emits one function that discovery found.
///
/// The Rust is a loop over a `match` on the guest address, one arm per block of straight-line
/// code in address order: a jump within the function sets the address and goes round the loop.
// Emission keeps several tables of its own; splitting it would pass them all around.
#[allow(clippy::too_many_lines)]
fn lift_function(entry: u32, found: &Discovered, setjmps: &BTreeSet<u32>, trace: bool) -> Body {
    let Discovered {
        insts,
        faults,
        hooked,
        tables,
        ..
    } = found;
    let mut traps = found.traps();
    let body_set = found.body();
    // The `setjmp` calls in address order: context `i` belongs to the `i`th, and a `longjmp` to it
    // goes on after it.
    let setjmp_calls: Vec<u32> = insts
        .iter()
        .filter(|(_, instruction)| is_setjmp_call(instruction, setjmps))
        .map(|(&address, _)| address)
        .collect();
    let mut emitter = Emitter {
        body: &body_set,
        landings: setjmp_calls
            .iter()
            .map(|address| addr(insts[address].next_ip()))
            .collect(),
        calls: BTreeSet::new(),
        sites: BTreeMap::new(),
        chunks: Vec::new(),
        chunk: 0,
    };
    let starts = block_starts(entry, insts, tables, &body_set);
    // Every block's address, the hooked places' and the faults' too: each is an arm. A function of
    // more than `CHUNK_ARMS` blocks is cut into chunks of that many, each a function of its own.
    let mut arm_starts: Vec<u32> = starts
        .iter()
        .copied()
        .filter(|address| insts.contains_key(address))
        .chain(hooked.keys().copied())
        .chain(faults.keys().copied())
        .collect();
    arm_starts.sort_unstable();
    if arm_starts.len() > CHUNK_ARMS {
        emitter.chunks = arm_starts.iter().copied().step_by(CHUNK_ARMS).collect();
    }
    let mut arms = vec![String::new(); emitter.chunks.len().max(1)];
    let addresses: Vec<u32> = insts.keys().copied().collect();
    let mut open = false;
    for (index, &address) in addresses.iter().enumerate() {
        let instruction = insts[&address];
        if !open {
            emitter.chunk = emitter.chunk_of(address);
            writeln!(arms[emitter.chunk], "                {address:#010x} => {{").unwrap();
            open = true;
        }
        let arms = &mut arms[emitter.chunk];
        if trace {
            writeln!(arms, "                    trace(c, {address:#010x});").unwrap();
        }
        let setjmp_site = setjmp_calls.iter().position(|&call| call == address);
        let statement = match emit(
            &mut emitter,
            &instruction,
            address,
            tables.get(&address),
            setjmp_site,
        ) {
            Ok(statement) => statement,
            Err(error) => {
                let statement = trap_statement(address, &error.message);
                traps.push(error);
                statement
            }
        };
        if !statement.is_empty() {
            writeln!(arms, "                    {statement}").unwrap();
        }
        if falls_through(&instruction) {
            let next = addr(instruction.next_ip());
            if addresses.get(index + 1) != Some(&next) || starts.contains(&next) {
                // A jump to the next block, or it runs into another function's entry.
                let jump = emitter.goto(next, address, Flow::Body);
                writeln!(arms, "                    {jump}\n                }}").unwrap();
                open = false;
            }
        } else {
            writeln!(arms, "                }}").unwrap();
            open = false;
        }
    }
    for (&address, &trap) in hooked {
        emitter.sites.insert((address, Site::Ret), 0);
        writeln!(
            arms[emitter.chunk_of(address)],
            "                {address:#010x} => {{\n                    \
             call_rt(c, {address:#010x}, K_RET, {trap:#010x}, {address:#010x});\n                    \
             return END;\n                }}"
        )
        .unwrap();
    }
    for (&address, message) in faults {
        let trap = trap_statement(address, message);
        writeln!(
            arms[emitter.chunk_of(address)],
            "                {address:#010x} => {{ {trap} }}"
        )
        .unwrap();
    }
    let resume = resume_prologue(&mut emitter, entry);
    let rust = function_rust(entry, &resume, &arms, &emitter.chunks, setjmp_calls.len());
    Body {
        rust,
        calls: emitter.calls,
        traps,
    }
}

/// The most blocks a lifted function holds before it is cut into chunks. The Rust compiler's
/// type checking and LLVM's optimization both take time that grows faster than a function's size,
/// so a function of tens of thousands of blocks would take longer to build than all the rest.
const CHUNK_ARMS: usize = 256;

/// The `match` arm that ends every block loop: an address with no block traps.
const NO_BLOCK: &str =
    "                _ => lift_trap(c, pc, b\"no lifted code at this address\"),\n";

/// The Rust of lifted function `entry`, from its resume prologue and its arms, one string per
/// chunk. With one chunk, the function loops over its blocks itself. With several, each chunk is
/// a function of its own that loops over its blocks and, for a jump to a block of another chunk,
/// returns with the address in `goto`; the lifted function calls the chunk that holds the address.
fn function_rust(
    entry: u32,
    resume: &str,
    arms: &[String],
    chunks: &[u32],
    contexts: usize,
) -> String {
    let declare_contexts = if contexts == 0 {
        String::new()
    } else {
        format!("        let mut lj = Jmps::<{contexts}>::new(c);\n")
    };
    let head = format!(
        "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn fn_{entry:08x}(c: *mut Cpu) -> Next {{\n    unsafe {{\n        \
         enter(c, {entry:#010x});\n{declare_contexts}        let mut pc: u32 = {entry:#010x};\n{resume}"
    );
    if chunks.is_empty() {
        return format!(
            "{head}        'l: loop {{\n            match pc {{\n{}{NO_BLOCK}            }}\n        }}\n    }}\n}}\n",
            arms.concat()
        );
    }
    let (parameter, argument) = if contexts == 0 {
        (String::new(), "")
    } else {
        (format!(", lj: &mut Jmps<{contexts}>"), ", &mut lj")
    };
    let mut rust = format!(
        "{head}        loop {{\n            let mut goto = u32::MAX;\n            let next = match pc {{\n"
    );
    let mut chunk_functions = String::new();
    for (index, (&start, arms)) in chunks.iter().zip(arms).enumerate() {
        let low = if index == 0 { 0 } else { start };
        let high = chunks.get(index + 1).map_or(u32::MAX - 1, |next| next - 1);
        writeln!(
            rust,
            "                {low:#010x}..={high:#010x} => fn_{entry:08x}_{index}(c, pc, &mut goto{argument}),"
        )
        .unwrap();
        write!(
            chunk_functions,
            "#[inline(never)]\nunsafe fn fn_{entry:08x}_{index}(c: *mut Cpu, mut pc: u32, goto: &mut u32{parameter}) -> Next {{\n    \
             unsafe {{\n        'l: loop {{\n            match pc {{\n{arms}{NO_BLOCK}            }}\n        }}\n    }}\n}}\n"
        )
        .unwrap();
    }
    rust.push_str(
        "                _ => lift_trap(c, pc, b\"no lifted code at this address\"),\n            };\n            \
         if goto == u32::MAX {\n                return next;\n            }\n            pc = goto;\n        }\n    }\n}\n",
    );
    rust.push_str(&chunk_functions);
    rust
}

/// The kind of site a host frame can be resumed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Site {
    /// A call: resumes the callee, then goes on after the call.
    Call,
    /// A call whose caller returns as soon as it does.
    Ret,
    /// A loop's poll: goes on after it.
    Poll,
}

impl Site {
    /// The prelude's `K_*` value.
    fn kind(self) -> u64 {
        match self {
            Self::Call => 0,
            Self::Ret => 1,
            Self::Poll => 2,
        }
    }
}

/// Where a jump goes from: the body's loop, or the resume prologue before it.
#[derive(Debug, Clone, Copy)]
enum Flow {
    Body,
    Resume,
}

impl Flow {
    /// The statement that goes on at the address just set.
    fn keyword(self) -> &'static str {
        match self {
            Self::Body => "continue 'l;",
            Self::Resume => "break 'r;",
        }
    }
}

/// What emitting one function needs besides the instruction: where its code is, how it leaves for
/// another function, and what it records on the way.
struct Emitter<'a> {
    /// The addresses of the function's own code: its instructions, hooked places and faults.
    body: &'a BTreeSet<u32>,
    /// For each `setjmp` context, the address a `longjmp` to it goes on at. A function that holds
    /// contexts calls other functions instead of tail-calling them: its frame must outlive the
    /// call.
    landings: Vec<u32>,
    /// The functions it names.
    calls: BTreeSet<u32>,
    /// The sites a resume can re-enter the function at, each with the address it goes on at.
    sites: BTreeMap<(u32, Site), u32>,
    /// The first address of each chunk, when the function is cut into chunks; empty when not.
    chunks: Vec<u32>,
    /// The chunk of the block being emitted.
    chunk: usize,
}

impl Emitter<'_> {
    fn holds_contexts(&self) -> bool {
        !self.landings.is_empty()
    }

    /// The chunk that holds the block at `address`.
    fn chunk_of(&self, address: u32) -> usize {
        self.chunks
            .partition_point(|&start| start <= address)
            .saturating_sub(1)
    }

    /// Goes on at `target`, a block of the function: within the chunk, a jump round its loop;
    /// to another chunk, a return with the address in `goto`.
    fn jump(&self, target: u32, flow: Flow) -> String {
        match flow {
            Flow::Body if !self.chunks.is_empty() && self.chunk_of(target) != self.chunk => {
                format!("{{ *goto = {target:#010x}; return END; }}")
            }
            _ => format!("{{ pc = {target:#010x}; {} }}", flow.keyword()),
        }
    }

    /// Goes on at `target`: a jump within the function, else a tail call from `site`.
    fn goto(&mut self, target: u32, site: u32, flow: Flow) -> String {
        if self.body.contains(&target) {
            return self.jump(target, flow);
        }
        self.calls.insert(target);
        if self.holds_contexts() {
            // A call whose caller returns after it: a site to resume at.
            self.sites.insert((site, Site::Ret), 0);
            let unwind = self.unwind(flow);
            format!(
                "{{ if call(c, {site:#010x}, K_RET, fn_{target:08x}) {{ {unwind} }} return END; }}"
            )
        } else {
            format!("{{ return to(fn_{target:08x}); }}")
        }
    }

    /// What the function does when a `longjmp` passes through one of its calls: it returns,
    /// unless the jump is to one of its own contexts, and then it goes on after that `setjmp`.
    fn unwind(&self, flow: Flow) -> String {
        if !self.holds_contexts() {
            return "return END;".to_owned();
        }
        let mut arms = String::new();
        for (index, &landing) in self.landings.iter().enumerate() {
            if self.body.contains(&landing) {
                write!(arms, " Some({index}) => {}", self.jump(landing, flow)).unwrap();
            } else {
                write!(
                    arms,
                    " Some({index}) => lift_trap(c, {landing:#010x}, b\"a longjmp lands outside its function\"),"
                )
                .unwrap();
            }
        }
        format!("match lj.land() {{{arms} _ => return END, }}")
    }

    /// A poll on a backward branch at `address` to `target`, which a resume goes on from.
    fn poll(&mut self, address: u32, target: u32) -> String {
        self.sites.insert((address, Site::Poll), target);
        format!("if poll(c, {address:#010x}) {{ return END; }} ")
    }
}

/// The prologue that resumes a function at a site: it registers the function's `setjmp` contexts
/// again, then calls down to, or goes on from, the site its frame record names.
fn resume_prologue(emitter: &mut Emitter, entry: u32) -> String {
    let mut rust = String::from("        if (*c).resuming != 0 {\n");
    for index in 0..emitter.landings.len() {
        writeln!(
            rust,
            "            jmp_reregister(c, lj.at({index}), {index});"
        )
        .unwrap();
    }
    rust.push_str("            'r: {\n                match resume_key(c) {\n");
    let sites: Vec<((u32, Site), u32)> = emitter.sites.iter().map(|(&k, &v)| (k, v)).collect();
    for ((address, site), next) in sites {
        let action = match site {
            Site::Call => {
                let unwind = emitter.unwind(Flow::Resume);
                let go = emitter.goto(next, address, Flow::Resume);
                format!("if resume_call(c) {{ {unwind} }} {go}")
            }
            Site::Ret => "resume_call(c); return END;".to_owned(),
            Site::Poll => {
                let go = emitter.goto(next, address, Flow::Resume);
                format!("if resume_poll(c) {{ return END; }} {go}")
            }
        };
        let key = (u64::from(address) << 2) | site.kind();
        writeln!(rust, "                    {key:#x} => {{ {action} }}").unwrap();
    }
    writeln!(
        rust,
        "                    _ => lift_trap(c, {entry:#010x}, b\"resuming at a site this function does not have\"),\n                \
         }}\n            }}\n        }}"
    )
    .unwrap();
    rust
}

/// The guest register a name refers to: the field holding it, and how it sits in that field.
fn register_slot(register: Register) -> Option<(&'static str, u32, u32)> {
    Some(match register {
        Register::EAX => ("eax", 0, 32),
        Register::AX => ("eax", 0, 16),
        Register::AL => ("eax", 0, 8),
        Register::AH => ("eax", 8, 8),
        Register::ECX => ("ecx", 0, 32),
        Register::CX => ("ecx", 0, 16),
        Register::CL => ("ecx", 0, 8),
        Register::CH => ("ecx", 8, 8),
        Register::EDX => ("edx", 0, 32),
        Register::DX => ("edx", 0, 16),
        Register::DL => ("edx", 0, 8),
        Register::DH => ("edx", 8, 8),
        Register::EBX => ("ebx", 0, 32),
        Register::BX => ("ebx", 0, 16),
        Register::BL => ("ebx", 0, 8),
        Register::BH => ("ebx", 8, 8),
        Register::ESP => ("esp", 0, 32),
        Register::SP => ("esp", 0, 16),
        Register::EBP => ("ebp", 0, 32),
        Register::BP => ("ebp", 0, 16),
        Register::ESI => ("esi", 0, 32),
        Register::SI => ("esi", 0, 16),
        Register::EDI => ("edi", 0, 32),
        Register::DI => ("edi", 0, 16),
        _ => return None,
    })
}

/// The Rust expression reading register `register`, zero-extended to 32 bits.
fn read_register(address: u32, register: Register) -> Result<String, LiftError> {
    let (name, shift, bits) = register_slot(register)
        .ok_or_else(|| unsupported(address, format!("register {register:?}")))?;
    Ok(match bits {
        32 => format!("(*c).{name}"),
        16 => format!("((*c).{name} & 0xFFFF)"),
        _ => format!("(((*c).{name} >> {shift}) & 0xFF)"),
    })
}

/// The Rust statement writing `value`, a `u32`, to register `register`.
fn write_register(address: u32, register: Register, value: &str) -> Result<String, LiftError> {
    let (name, shift, bits) = register_slot(register)
        .ok_or_else(|| unsupported(address, format!("register {register:?}")))?;
    let field = format!("(*c).{name}");
    Ok(match (bits, shift) {
        (32, _) => format!("{field} = {value};"),
        (16, _) => format!("{field} = ({field} & 0xFFFF0000) | (({value}) & 0xFFFF);"),
        (_, 0) => format!("{field} = ({field} & 0xFFFFFF00) | (({value}) & 0xFF);"),
        _ => format!("{field} = ({field} & 0xFFFF00FF) | ((({value}) & 0xFF) << 8);"),
    })
}

fn register_bits(address: u32, register: Register) -> Result<u32, LiftError> {
    register_slot(register)
        .map(|(_, _, bits)| bits)
        .ok_or_else(|| unsupported(address, format!("register {register:?}")))
}

/// The bit width of operand `index`, from its register or its memory size.
fn operand_bits(address: u32, instruction: &Instruction, index: u32) -> Result<u32, LiftError> {
    match instruction.op_kind(index) {
        OpKind::Register => register_bits(address, instruction.op_register(index)),
        OpKind::Memory => {
            Ok(u32::try_from(instruction.memory_size().size()).unwrap_or(u32::MAX) * 8)
        }
        _ => Err(unsupported(address, "operand width of an immediate")),
    }
}

/// The Rust expression computing the effective address of the memory operand of `instruction`.
fn effective_address(instruction: &Instruction) -> String {
    let mut parts = Vec::new();
    let base = instruction.memory_base();
    if base != Register::None {
        parts.push(read_register(0, base).unwrap_or_else(|_| "0u32".into()));
    }
    let index = instruction.memory_index();
    if index != Register::None {
        let scale = instruction.memory_index_scale();
        let reg = read_register(0, index).unwrap_or_else(|_| "0u32".into());
        parts.push(if scale == 1 {
            reg
        } else {
            format!("({reg}).wrapping_mul({scale})")
        });
    }
    let displacement = addr(instruction.memory_displacement64());
    if displacement != 0 || parts.is_empty() {
        parts.push(format!("{displacement:#x}u32"));
    }
    if instruction.segment_prefix() == Register::FS {
        parts.push("FS_BASE".into());
    }
    let mut parts = parts.into_iter();
    let mut ea = parts.next().unwrap_or_default();
    for part in parts {
        ea = format!("{ea}.wrapping_add({part})");
    }
    ea
}

/// The mask of the low `bits` of a value, as a Rust literal.
fn mask_literal(bits: u32) -> String {
    if bits >= 32 {
        "0xFFFFFFFFu32".to_owned()
    } else {
        format!("{:#x}u32", (1u64 << bits) - 1)
    }
}

/// The Rust expression reading operand `index`, masked to `bits`, as a `u32`. An immediate is
/// taken at `bits`.
fn read_operand(
    address: u32,
    instruction: &Instruction,
    index: u32,
    bits: u32,
) -> Result<String, LiftError> {
    Ok(match instruction.op_kind(index) {
        OpKind::Register => {
            let value = read_register(address, instruction.op_register(index))?;
            if bits >= 32 {
                value
            } else {
                format!("({value} & {})", mask_literal(bits))
            }
        }
        OpKind::Memory => {
            let ea = effective_address(instruction);
            match bits {
                8 => format!("rd8({ea})"),
                16 => format!("rd16({ea})"),
                _ => format!("rd32({ea})"),
            }
        }
        OpKind::Immediate8
        | OpKind::Immediate16
        | OpKind::Immediate32
        | OpKind::Immediate8to32
        | OpKind::Immediate8to16
        | OpKind::Immediate32to64 => {
            let mask = if bits >= 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            format!("{:#x}u32", addr(instruction.immediate(index)) & mask)
        }
        other => return Err(unsupported(address, format!("operand kind {other:?}"))),
    })
}

/// The Rust statement writing `value`, a `u32`, to operand `index` at `bits`.
fn write_operand(
    address: u32,
    instruction: &Instruction,
    index: u32,
    bits: u32,
    value: &str,
) -> Result<String, LiftError> {
    match instruction.op_kind(index) {
        OpKind::Register => write_register(address, instruction.op_register(index), value),
        OpKind::Memory => {
            let ea = effective_address(instruction);
            Ok(match bits {
                8 => format!("wr8({ea}, {value});"),
                16 => format!("wr16({ea}, {value});"),
                _ => format!("wr32({ea}, {value});"),
            })
        }
        other => Err(unsupported(address, format!("cannot write to {other:?}"))),
    }
}

/// The Rust expression sign-extending the low `bits` of `value`.
fn sign_extend(value: &str, bits: u32) -> String {
    format!(
        "(((({value}) << {shift}) as i32 >> {shift}) as u32)",
        shift = 32 - bits
    )
}

/// Counts ECX down once.
const ECX_DOWN: &str = "(*c).ecx = (*c).ecx.wrapping_sub(1);";

/// A string instruction, once or under `rep`.
fn string_loop(instruction: &Instruction, body: &str, advance: &str) -> String {
    if instruction.has_rep_prefix()
        || instruction.has_repe_prefix()
        || instruction.has_repne_prefix()
    {
        format!("while (*c).ecx != 0 {{ {body} {advance} {ECX_DOWN} }}")
    } else {
        format!("{body} {advance}")
    }
}

/// A comparing string instruction, once or under `repe` or `repne`.
fn compare_loop(instruction: &Instruction, body: &str, advance: &str) -> String {
    if instruction.has_repe_prefix() {
        format!(
            "while (*c).ecx != 0 {{ {body} {advance} {ECX_DOWN} if (*c).zf == 0 {{ break; }} }}"
        )
    } else if instruction.has_repne_prefix() {
        format!(
            "while (*c).ecx != 0 {{ {body} {advance} {ECX_DOWN} if (*c).zf != 0 {{ break; }} }}"
        )
    } else {
        format!("{body} {advance}")
    }
}

/// A step of a string register by `step` bytes, as the direction flag says.
fn advance(register: &str, step: u32) -> String {
    format!("(*c).{register} = (*c).{register}.wrapping_add(dstep(c, {step}));")
}

/// The Rust statements for one instruction.
///
/// A jump to an address outside the function's own code is a tail call.
// One arm per instruction form: the table is long because the instruction set is.
#[allow(clippy::too_many_lines)]
fn emit(
    e: &mut Emitter,
    instruction: &Instruction,
    address: u32,
    table: Option<&Vec<(u32, u32)>>,
    setjmp_site: Option<usize>,
) -> Result<String, LiftError> {
    #[allow(clippy::enum_glob_use)]
    use Mnemonic::*;
    if let Some(statement) = x87::emit(instruction, address)? {
        return Ok(statement);
    }
    if let Some(statement) = mmx::emit(instruction, address)? {
        return Ok(statement);
    }
    if let Some(statement) = sse::emit(instruction, address)? {
        return Ok(statement);
    }
    let next = addr(instruction.next_ip());
    let size = if instruction.op_count() > 0 && instruction.op_kind(0) != OpKind::NearBranch32 {
        operand_bits(address, instruction, 0).unwrap_or(32)
    } else {
        32
    };
    let mnemonic = instruction.mnemonic();
    if let Some(code) = condition_code(mnemonic) {
        return Ok(match mnemonic {
            m if format!("{m:?}").starts_with('J') => {
                let target = addr(instruction.near_branch_target());
                let poll = if target <= address {
                    e.poll(address, target)
                } else {
                    String::new()
                };
                let jump = e.goto(target, address, Flow::Body);
                format!("if cond(c, {code}) {{ {poll}{jump} }}")
            }
            m if format!("{m:?}").starts_with("Set") => write_operand(
                address,
                instruction,
                0,
                8,
                &format!("u32::from(cond(c, {code}))"),
            )?,
            _ => {
                let value = read_operand(address, instruction, 1, size)?;
                let dst = write_operand(address, instruction, 0, size, &value)?;
                format!("if cond(c, {code}) {{ {dst} }}")
            }
        });
    }
    Ok(match mnemonic {
        // No effect natively: interrupts are not modelled; port writes reach no device, as in the
        // emulator; caches, prefetches, fences and spin hints change nothing one host image sees.
        Nop | Cli | Sti | Out | Wbinvd | Invd | Invlpg | Clflush | Prefetchnta | Prefetcht0
        | Prefetcht1 | Prefetcht2 | Prefetchw | Sfence | Lfence | Mfence | Pause | Emms => {
            String::new()
        }
        // Port reads give zero, as in the emulator.
        In => write_operand(address, instruction, 0, size, "0u32")?,
        Cld => "(*c).df = 0;".to_string(),
        Clc => "(*c).cf = 0;".to_string(),
        Stc => "(*c).cf = 1;".to_string(),
        Cmc => "(*c).cf ^= 1;".to_string(),
        Xlatb => "(*c).eax = ((*c).eax & 0xFFFFFF00) | rd8((*c).ebx.wrapping_add((*c).eax & 0xFF));"
            .to_string(),
        Enter if instruction.immediate8_2nd() == 0 => format!(
            "push32(c, (*c).ebp); (*c).ebp = (*c).esp; (*c).esp = (*c).esp.wrapping_sub({});",
            instruction.immediate16()
        ),
        Loop | Loope | Loopne | Jecxz => {
            let target = addr(instruction.near_branch_target());
            let (count, test) = match mnemonic {
                Loop => (ECX_DOWN, "(*c).ecx != 0"),
                Loope => (ECX_DOWN, "(*c).ecx != 0 && (*c).zf != 0"),
                Loopne => (ECX_DOWN, "(*c).ecx != 0 && (*c).zf == 0"),
                _ => ("", "(*c).ecx == 0"),
            };
            let poll = if target <= address {
                e.poll(address, target)
            } else {
                String::new()
            };
            let jump = e.goto(target, address, Flow::Body);
            format!("{count} if {test} {{ {poll}{jump} }}")
        }
        Rdtsc => {
            "{ let t: u64 = lift_rdtsc(); (*c).eax = t as u32; (*c).edx = (t >> 32) as u32; }"
                .to_string()
        }
        Std => "(*c).df = 1;".to_string(),
        Stosb | Stosw | Stosd => {
            let (accumulator, store, step) = match mnemonic {
                Stosb => (Register::AL, "wr8", 1),
                Stosw => (Register::AX, "wr16", 2),
                _ => (Register::EAX, "wr32", 4),
            };
            let value = read_register(address, accumulator)?;
            string_loop(
                instruction,
                &format!("{store}((*c).edi, {value});"),
                &advance("edi", step),
            )
        }
        Cmpsb | Cmpsw | Cmpsd => {
            let (bits, load, step) = match mnemonic {
                Cmpsb => (8, "rd8", 1),
                Cmpsw => (16, "rd16", 2),
                _ => (32, "rd32", 4),
            };
            let body = format!("alu_sub(c, {load}((*c).esi), {load}((*c).edi), {bits});");
            let step = format!("{} {}", advance("esi", step), advance("edi", step));
            compare_loop(instruction, &body, &step)
        }
        Movsb | Movsw | Movsd => {
            let (load, store, step) = match mnemonic {
                Movsb => ("rd8", "wr8", 1),
                Movsw => ("rd16", "wr16", 2),
                _ => ("rd32", "wr32", 4),
            };
            string_loop(
                instruction,
                &format!("{store}((*c).edi, {load}((*c).esi));"),
                &format!("{} {}", advance("esi", step), advance("edi", step)),
            )
        }
        Int3 | Int | Ud2 | Hlt => trap_statement(address, "interrupt or halt"),
        // `movnti` is a store with a cache hint.
        Mov | Movnti => {
            let value = read_operand(address, instruction, 1, size)?;
            write_operand(address, instruction, 0, size, &value)?
        }
        Movzx => {
            let src_bits = operand_bits(address, instruction, 1)?;
            let value = read_operand(address, instruction, 1, src_bits)?;
            write_operand(address, instruction, 0, size, &value)?
        }
        Movsx => {
            let src_bits = operand_bits(address, instruction, 1)?;
            let value = sign_extend(&read_operand(address, instruction, 1, src_bits)?, src_bits);
            write_operand(address, instruction, 0, size, &value)?
        }
        Lea => {
            let value = effective_address(instruction);
            write_operand(address, instruction, 0, size, &value)?
        }
        Xchg => {
            let a = read_operand(address, instruction, 0, size)?;
            let b = read_operand(address, instruction, 1, size)?;
            let w0 = write_operand(address, instruction, 0, size, "t_b")?;
            let w1 = write_operand(address, instruction, 1, size, "t_a")?;
            format!("{{ let t_a: u32 = {a}; let t_b: u32 = {b}; {w0} {w1} }}")
        }
        Push => {
            let value = read_operand(address, instruction, 0, 32)?;
            format!("push32(c, {value});")
        }
        Pop => write_operand(address, instruction, 0, 32, "pop32(c)")?,
        Bsf | Bsr => {
            let src = read_operand(address, instruction, 1, size)?;
            let dst = read_operand(address, instruction, 0, size)?;
            let helper = if mnemonic == Bsf {
                "alu_bsf"
            } else {
                "alu_bsr"
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}(c, {src}, {dst})"),
            )?
        }
        Adc | Sbb => {
            let a = read_operand(address, instruction, 0, size)?;
            let b = read_operand(address, instruction, 1, size)?;
            let helper = if mnemonic == Adc {
                "alu_adc"
            } else {
                "alu_sbb"
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}(c, {a}, {b}, {size})"),
            )?
        }
        Add | Sub | Cmp | And | Or | Xor | Test => {
            let a = read_operand(address, instruction, 0, size)?;
            let b = read_operand(address, instruction, 1, size)?;
            let (expr, store) = match mnemonic {
                Add => (format!("alu_add(c, {a}, {b}, {size})"), true),
                Sub => (format!("alu_sub(c, {a}, {b}, {size})"), true),
                Cmp => (format!("alu_sub(c, {a}, {b}, {size})"), false),
                And => (format!("alu_logic(c, {a} & {b}, {size})"), true),
                Or => (format!("alu_logic(c, {a} | {b}, {size})"), true),
                Xor => (format!("alu_logic(c, {a} ^ {b}, {size})"), true),
                _ => (format!("alu_logic(c, {a} & {b}, {size})"), false),
            };
            if store {
                write_operand(address, instruction, 0, size, &expr)?
            } else {
                format!("{expr};")
            }
        }
        Inc | Dec | Neg | Not => {
            let a = read_operand(address, instruction, 0, size)?;
            let expr = match mnemonic {
                Inc => format!("alu_inc(c, {a}, {size})"),
                Dec => format!("alu_dec(c, {a}, {size})"),
                Neg => format!("alu_neg(c, {a}, {size})"),
                _ => format!("(!{a} & {})", mask_literal(size)),
            };
            write_operand(address, instruction, 0, size, &expr)?
        }
        Rol | Ror => {
            let a = read_operand(address, instruction, 0, size)?;
            let count = read_operand(address, instruction, 1, 8)?;
            let helper = if mnemonic == Rol { "alu_rol" } else { "alu_ror" };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}(c, {a}, {count}, {size})"),
            )?
        }
        Shld | Shrd => {
            let dst = read_operand(address, instruction, 0, size)?;
            let src = read_operand(address, instruction, 1, size)?;
            let count = read_operand(address, instruction, 2, 8)?;
            let helper = if mnemonic == Shld { "alu_shld" } else { "alu_shrd" };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}(c, {dst}, {src}, {count}, {size})"),
            )?
        }
        Cwde => "(*c).eax = (*c).eax as i16 as i32 as u32;".to_string(),
        Cbw => "(*c).eax = ((*c).eax & 0xFFFF0000) | u32::from((*c).eax as i8 as i16 as u16);"
            .to_string(),
        Cwd => "(*c).edx = ((*c).edx & 0xFFFF0000) | (if ((*c).eax as i16) < 0 { 0xFFFF } else { 0 });"
            .to_string(),
        Bswap => {
            let value = read_register(address, instruction.op0_register())?;
            write_register(
                address,
                instruction.op0_register(),
                &format!("{value}.swap_bytes()"),
            )?
        }
        Pushfd => "push32(c, eflags_get(c));".to_string(),
        Pushad => "{ let t_esp: u32 = (*c).esp; push32(c, (*c).eax); push32(c, (*c).ecx); push32(c, (*c).edx); \
                   push32(c, (*c).ebx); push32(c, t_esp); push32(c, (*c).ebp); push32(c, (*c).esi); push32(c, (*c).edi); }"
            .to_string(),
        Popad => "{ (*c).edi = pop32(c); (*c).esi = pop32(c); (*c).ebp = pop32(c); pop32(c); (*c).ebx = pop32(c); \
                  (*c).edx = pop32(c); (*c).ecx = pop32(c); (*c).eax = pop32(c); }"
            .to_string(),
        Popfd => "eflags_set(c, pop32(c));".to_string(),
        Lahf => "(*c).eax = ((*c).eax & 0xFFFF00FF) | (((u32::from((*c).sf) << 7) | (u32::from((*c).zf) << 6) \
                 | (u32::from((*c).pf) << 2) | 2 | u32::from((*c).cf)) << 8);"
            .to_string(),
        Xadd => {
            let dst = read_operand(address, instruction, 0, size)?;
            let src = read_operand(address, instruction, 1, size)?;
            let write_src = write_operand(address, instruction, 1, size, "t_a")?;
            let write_dst = write_operand(address, instruction, 0, size, "t_r")?;
            format!(
                "{{ let t_a: u32 = {dst}; let t_b: u32 = {src}; let t_r: u32 = alu_add(c, t_a, t_b, {size}); \
                 {write_src} {write_dst} }}"
            )
        }
        Cmpxchg => {
            let accumulator = match size {
                8 => Register::AL,
                16 => Register::AX,
                _ => Register::EAX,
            };
            let acc = read_register(address, accumulator)?;
            let dst = read_operand(address, instruction, 0, size)?;
            let src = read_operand(address, instruction, 1, size)?;
            let store = write_operand(address, instruction, 0, size, &src)?;
            let load = write_register(address, accumulator, "t_d")?;
            format!(
                "{{ let t_d: u32 = {dst}; alu_sub(c, {acc}, t_d, {size}); if (*c).zf != 0 {{ {store} }} else {{ {load} }} }}"
            )
        }
        Scasb | Scasw | Scasd => {
            let (bits, accumulator, load, step) = match mnemonic {
                Scasb => (8, Register::AL, "rd8", 1),
                Scasw => (16, Register::AX, "rd16", 2),
                _ => (32, Register::EAX, "rd32", 4),
            };
            let acc = read_register(address, accumulator)?;
            let body = format!("alu_sub(c, {acc}, {load}((*c).edi), {bits});");
            compare_loop(instruction, &body, &advance("edi", step))
        }
        Lodsb | Lodsw | Lodsd => {
            let (accumulator, load, step) = match mnemonic {
                Lodsb => (Register::AL, "rd8((*c).esi)", 1),
                Lodsw => (Register::AX, "rd16((*c).esi)", 2),
                _ => (Register::EAX, "rd32((*c).esi)", 4),
            };
            let store = write_register(address, accumulator, load)?;
            string_loop(instruction, &store, &advance("esi", step))
        }
        Rcl | Rcr => {
            let a = read_operand(address, instruction, 0, size)?;
            let count = read_operand(address, instruction, 1, 8)?;
            let helper = if mnemonic == Rcl {
                "alu_rcl"
            } else {
                "alu_rcr"
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}(c, {a}, {count}, {size})"),
            )?
        }
        Shl | Sal | Shr | Sar => {
            let a = read_operand(address, instruction, 0, size)?;
            let count = read_operand(address, instruction, 1, 8)?;
            let helper = match mnemonic {
                Shl | Sal => "alu_shl",
                Shr => "alu_shr",
                _ => "alu_sar",
            };
            write_operand(
                address,
                instruction,
                0,
                size,
                &format!("{helper}(c, {a}, {count}, {size})"),
            )?
        }
        Imul => match instruction.op_count() {
            1 => {
                let a = read_operand(address, instruction, 0, size)?;
                let helper = match size {
                    8 => "alu_imul1_8",
                    16 => "alu_imul1_16",
                    _ => "alu_imul1",
                };
                format!("{helper}(c, {a});")
            }
            2 => {
                let a = read_operand(address, instruction, 0, size)?;
                let b = read_operand(address, instruction, 1, size)?;
                write_operand(
                    address,
                    instruction,
                    0,
                    size,
                    &format!("alu_imul(c, {a}, {b}, {size})"),
                )?
            }
            3 => {
                let b = read_operand(address, instruction, 1, size)?;
                let third = read_operand(address, instruction, 2, size)?;
                write_operand(
                    address,
                    instruction,
                    0,
                    size,
                    &format!("alu_imul(c, {b}, {third}, {size})"),
                )?
            }
            _ => return Err(unsupported(address, "imul with this operand form")),
        },
        Mul | Div | Idiv => {
            let a = read_operand(address, instruction, 0, size)?;
            let suffix = match size {
                8 => "8",
                16 => "16",
                _ => "",
            };
            match mnemonic {
                Mul => format!("alu_mul{suffix}(c, {a});"),
                Div => format!("alu_div{suffix}(c, {a}, {address:#010x});"),
                _ => format!("alu_idiv{suffix}(c, {a}, {address:#010x});"),
            }
        }
        Cdq => "(*c).edx = (((*c).eax as i32) >> 31) as u32;".to_string(),
        Leave => "(*c).esp = (*c).ebp; (*c).ebp = pop32(c);".to_string(),
        Ret => {
            let extra = if instruction.op_count() == 1 {
                addr(instruction.immediate(0))
            } else {
                0
            };
            format!(
                "(*c).esp = (*c).esp.wrapping_add({}); return END;",
                extra.wrapping_add(4)
            )
        }
        Call => match instruction.op0_kind() {
            OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = addr(instruction.near_branch_target());
                e.calls.insert(target);
                e.sites.insert((address, Site::Call), next);
                let unwind = e.unwind(Flow::Body);
                let call = format!(
                    "if call(c, {address:#010x}, K_CALL, fn_{target:08x}) {{ {unwind} }}"
                );
                match setjmp_site {
                    // The first argument, the buffer, is above the return address just pushed. A
                    // `longjmp` to it goes on after the call, with the registers and EAX already
                    // set by the title's own code.
                    Some(index) => format!(
                        "push32(c, {next:#010x}); \
                         jmp_register(c, lj.at({index}), rd32((*c).esp.wrapping_add(4)), {next:#010x}, {index}); {call}"
                    ),
                    None => format!("push32(c, {next:#010x}); {call}"),
                }
            }
            _ => {
                let target = read_operand(address, instruction, 0, 32)?;
                e.sites.insert((address, Site::Call), next);
                let unwind = e.unwind(Flow::Body);
                format!(
                    "{{ let t_target: u32 = {target}; push32(c, {next:#010x}); \
                     if call_rt(c, {address:#010x}, K_CALL, t_target, {address:#010x}) {{ {unwind} }} }}"
                )
            }
        },
        Jmp if instruction.op0_kind() != OpKind::NearBranch32
            && instruction.op0_kind() != OpKind::NearBranch64
            && table.is_none()
            && is_tail_call(instruction) =>
        {
            // A jump through an import's slot: the callee returns to our caller, so we return after it.
            let target = read_operand(address, instruction, 0, 32)?;
            e.sites.insert((address, Site::Ret), 0);
            format!(
                "{{ let t_target: u32 = {target}; call_rt(c, {address:#010x}, K_RET, t_target, {address:#010x}); return END; }}"
            )
        }
        Jmp if instruction.op0_kind() != OpKind::NearBranch32
            && instruction.op0_kind() != OpKind::NearBranch64 =>
        {
            let entries =
                table.ok_or_else(|| unsupported(address, "indirect jump without a table"))?;
            let index = read_register(address, instruction.memory_index())?;
            let mut cases = String::new();
            for &(value, target) in entries {
                let jump = e.goto(target, address, Flow::Body);
                write!(cases, " {value}u32 => {jump}").unwrap();
            }
            let default = trap_statement(address, "jump table index out of range");
            format!("match {index} {{{cases} _ => {{ {default} }} }}")
        }
        Jmp => {
            let target = addr(instruction.near_branch_target());
            let poll = if target <= address {
                e.poll(address, target)
            } else {
                String::new()
            };
            let jump = e.goto(target, address, Flow::Body);
            format!("{poll}{jump}")
        }
        other => return Err(unsupported(address, format!("instruction {other:?}"))),
    })
}
