#![allow(clippy::too_many_lines)] // each test spells out its Rust harness

//! Lifts a few hand-assembled functions to Rust and checks it: its shape, that it compiles to
//! objects that need no standard library, and that it runs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use lift::Image;
use lift::rust::{UNIT_ATTRIBUTES, lift_program};

const LEAF: u32 = 0x1000;
const CALLER: u32 = 0x1010;
const HOOKED: u32 = 0x1020;
const TRAP: u32 = 0x8001_0800;
const JMP_SETJMP: u32 = 0x1000;
const JMP_LONGJMP: u32 = 0x1040;
const JMP_CALLER: u32 = 0x1080;
const JMP_BUFFER: u32 = 0x2000;

/// `mov eax, 1 ; ret`, then `call LEAF ; ret`, then a function the runtime replaces, in `len`
/// bytes.
fn image_bytes(len: usize) -> Vec<u8> {
    let mut bytes = vec![0x90u8; len];
    bytes[..6].copy_from_slice(&[0xB8, 1, 0, 0, 0, 0xC3]);
    let rel = LEAF.wrapping_sub(CALLER + 5);
    bytes[0x10] = 0xE8;
    bytes[0x11..0x15].copy_from_slice(&rel.to_le_bytes());
    bytes[0x15] = 0xC3;
    bytes[0x20..0x26].copy_from_slice(&[0xB8, 9, 0, 0, 0, 0xC3]);
    bytes
}

fn image() -> Image {
    Image::single(LEAF, image_bytes(0x30))
}

/// A directory of its own for test `name`, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("anybox-lift-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The Rust compiler cargo runs with, else the one on the `PATH`.
fn rustc() -> Command {
    Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
}

/// The flags every build of lifted code takes, as the title build passes them.
const FLAGS: [&str; 8] = [
    "--edition=2024",
    "-Cpanic=abort",
    "-Coverflow-checks=off",
    "-Cdebug-assertions=off",
    "--cap-lints=allow",
    "-Ccodegen-units=1",
    "--crate-type=lib",
    "--emit=obj",
];

/// Compiles the unit at `source` to an object at `opt`, and returns the object's path.
fn compile_unit(source: &Path, opt: &str) -> PathBuf {
    let object = source.with_extension("o");
    let output = rustc()
        .args(FLAGS)
        .arg(format!("-Copt-level={opt}"))
        .arg("--crate-name")
        .arg(source.file_stem().unwrap())
        .arg("-o")
        .arg(&object)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    object
}

/// The symbols `object` needs from elsewhere.
fn undefined_symbols(object: &Path) -> Vec<String> {
    use object::{Object, ObjectSymbol};
    let bytes = std::fs::read(object).unwrap();
    let file = object::File::parse(&*bytes).unwrap();
    file.symbols()
        .filter(ObjectSymbol::is_undefined)
        .filter_map(|symbol| symbol.name().ok().map(str::to_owned))
        .collect()
}

/// Builds `harness` with `program` (as one crate that defines the globals) at `opt`, runs it, and
/// returns its exit status and messages.
fn build_and_run(
    name: &str,
    program: &lift::rust::Program,
    harness: &str,
    opt: &str,
) -> (Option<i32>, String) {
    run_source(name, &format!("{harness}\n{}", program.source()), opt)
}

/// Builds the program `source` at `opt`, runs it, and returns its exit status and messages.
fn run_source(name: &str, source: &str, opt: &str) -> (Option<i32>, String) {
    let dir = Scratch::new(name);
    let main = dir.path("main.rs");
    std::fs::write(&main, source).unwrap();
    let exe = dir.path("test");
    let build = rustc()
        .args(&FLAGS[..5])
        .arg(format!("-Copt-level={opt}"))
        .arg("-o")
        .arg(&exe)
        .arg(&main)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    (
        run.status.code(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
    )
}

/// What every harness defines in place of the runtime, the native hook excepted.
const RUNTIME_STUBS: &str = r#"
#[unsafe(no_mangle)]
pub static mut anyxbox_preempt_flag: i32 = 0;
#[unsafe(no_mangle)]
pub static mut anyxbox_tsc_offset: i64 = 0;
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_mmio_read(_a: u32, _size: u32) -> u32 { 0 }
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_mmio_write(_a: u32, _size: u32, _v: u32) {}
#[unsafe(no_mangle)]
pub extern "C" fn lift_trap_hook(_c: *mut Cpu) {}
"#;

/// The stubs of a harness that runs lifted code without resuming it: a kernel call exits with 3.
const STUBS: &str = r#"
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_preempt() {}
#[unsafe(no_mangle)]
pub extern "C" fn lift_resume_innermost(_c: *mut Cpu) {}
#[unsafe(no_mangle)]
pub extern "C" fn lift_native(_c: *mut Cpu, target: u32, _at: u32) {
    eprintln!("native {target:#x}");
    std::process::exit(3);
}
/// A guest thread with a megabyte of guest memory, its stack at 0xF000.
unsafe fn setup() -> *mut Cpu {
    unsafe {
        G = Box::leak(vec![0u8; 1 << 20].into_boxed_slice()).as_mut_ptr();
        let c = lift_thread_new();
        (*c).esp = 0xF000;
        c
    }
}
"#;

/// A harness of the runtime stubs, the plain stubs, and `main`.
fn harness(main: &str) -> String {
    format!("{RUNTIME_STUBS}{STUBS}{main}")
}

#[test]
fn a_leaf_and_its_caller_lift_and_the_callee_is_found_from_the_caller() {
    let program = lift_program(&image(), &[CALLER], &BTreeMap::new(), false);
    assert_eq!(program.lifted, 2, "the caller and what it calls");
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let source = program.source();
    assert!(
        source.contains("pub unsafe extern \"C\" fn fn_00001010(c: *mut Cpu) -> Next"),
        "{source}"
    );
    assert!(
        source.contains("pub unsafe extern \"C\" fn fn_00001000(c: *mut Cpu) -> Next"),
        "{source}"
    );
}

#[test]
fn a_hooked_function_is_a_call_to_its_trap_and_its_own_code_is_ignored() {
    let hooks = BTreeMap::from([(HOOKED, TRAP)]);
    let program = lift_program(&image(), &[HOOKED], &hooks, false);
    let source = program.source();
    assert!(
        source.contains("call_rt(c, 0x00001020, K_RET, 0x80010800, 0x00001020);"),
        "{source}"
    );
}

#[test]
fn an_address_outside_the_image_does_not_stop_the_others() {
    let program = lift_program(&image(), &[LEAF, 0x9000], &BTreeMap::new(), false);
    assert_eq!(program.lifted, 1);
    assert_eq!(program.failures.len(), 1);
}

/// Each unit, and the dispatcher's, compiles on its own to an object that names no symbol of the
/// standard library, unoptimized and optimized: the program links them with a runtime that
/// carries its own copy, which may not be this compiler's.
#[test]
fn the_units_compile_to_objects_that_need_no_standard_library() {
    let hooks = BTreeMap::from([(HOOKED, TRAP)]);
    let mut program_bytes = image_bytes(0x200);
    // A function that uses a little of everything the prelude has: division, the x87, SSE, MMX,
    // string instructions and a jump table.
    let mixed: &[u8] = &[
        0xF7, 0xF1, // div ecx
        0xF7, 0xF9, // idiv ecx
        0xF6, 0xF3, // div bl
        0xF6, 0xFB, // idiv bl
        0xD9, 0xE8, // fld1
        0xD9, 0xF8, // fprem
        0xD9, 0xFE, // fsin
        0xDB, 0x3D, 0x00, 0x30, 0x00, 0x00, // fstp tword [0x3000]
        0xDB, 0x2D, 0x00, 0x30, 0x00, 0x00, // fld tword [0x3000]
        0xDF, 0x3D, 0x10, 0x30, 0x00, 0x00, // fistp qword [0x3010]
        0x0F, 0x58, 0xC1, // addps xmm0, xmm1
        0x0F, 0x2D, 0xC0, // cvtps2pi mm0, xmm0
        0x0F, 0xFC, 0xC1, // paddb mm0, mm1
        0x0F, 0xD8, 0xC1, // psubusb mm0, mm1
        0x0F, 0xF5, 0xC1, // pmaddwd mm0, mm1
        0x0F, 0x63, 0xC1, // packsswb mm0, mm1
        0x0F, 0x71, 0xE0, 0x03, // psraw mm0, 3
        0xF3, 0xA5, // rep movsd
        0xF3, 0xA6, // repe cmpsb
        0xD1, 0xD0, // rcl eax, 1
        0x0F, 0xA5, 0xD0, // shld eax, edx, cl
        0x0F, 0x31, // rdtsc
        0xC3,
    ];
    program_bytes[0x100..0x100 + mixed.len()].copy_from_slice(mixed);
    let program = lift_program(
        &Image::single(LEAF, program_bytes),
        &[CALLER, HOOKED, 0x1100],
        &hooks,
        false,
    );
    assert!(
        program.traps.is_empty() && program.failures.is_empty(),
        "{:?} {:?}",
        program.traps,
        program.failures
    );
    let dir = Scratch::new("objects");
    let mut sources: Vec<String> = program.units(2);
    sources.push(program.dispatch_unit());
    for opt in ["0", "1"] {
        for (index, source) in sources.iter().enumerate() {
            let path = dir.path(&format!("unit{index}.rs"));
            std::fs::write(&path, source).unwrap();
            let object = compile_unit(&path, opt);
            let rust: Vec<String> = undefined_symbols(&object)
                .into_iter()
                .filter(|name| name.starts_with("__ZN") || name.starts_with("__R"))
                .collect();
            assert!(rust.is_empty(), "unit {index} at -O{opt} needs {rust:?}");
        }
    }
}

#[test]
fn the_dispatcher_finds_each_function_and_sends_anything_else_to_the_native_hook() {
    let hooks = BTreeMap::from([(HOOKED, TRAP)]);
    let program = lift_program(&image(), &[CALLER, HOOKED], &hooks, false);
    let dispatcher = program
        .dispatch_unit()
        .strip_prefix(UNIT_ATTRIBUTES)
        .unwrap()
        .to_owned();
    // The functions record that they ran. Address 1 shares the cache slot of 0x1000 and is not a
    // function; 0 is not one either (an empty cache slot must not look like a hit).
    let main = format!(
        r#"{RUNTIME_STUBS}{}
#[unsafe(no_mangle)]
pub extern "C" fn anyxbox_preempt() {{}}
#[unsafe(no_mangle)]
pub extern "C" fn lift_resume_innermost(_c: *mut Cpu) {{}}
static mut LAST: i32 = 0;
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fn_00001000(_c: *mut Cpu) -> Next {{ unsafe {{ LAST = 1 }}; END }}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fn_00001010(_c: *mut Cpu) -> Next {{ unsafe {{ LAST = 2 }}; END }}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fn_00001020(_c: *mut Cpu) -> Next {{ unsafe {{ LAST = 3 }}; END }}
#[unsafe(no_mangle)]
pub extern "C" fn lift_native(_c: *mut Cpu, _target: u32, _at: u32) {{ unsafe {{ LAST = -1 }} }}
fn main() {{
    unsafe {{
        let c = lift_thread_new();
        let addresses = [0x1000, 0x1010, 0x1020, 0x1000, 1, 0x1000, 0x0fff, 0x1001, 0x2000, 0, 0xffff_ffff, 0x1020];
        let wanted = [1, 2, 3, 1, -1, 1, -1, -1, -1, -1, -1, 3];
        for (i, (&address, &want)) in addresses.iter().zip(&wanted).enumerate() {{
            LAST = 99;
            rt_call(c, address, 0);
            if LAST != want {{
                std::process::exit(10 + i as i32);
            }}
        }}
    }}
}}
{dispatcher}"#,
        lift::rust::GLOBALS
    );
    let (status, _) = run_source("dispatch", &main, "1");
    assert_eq!(
        status,
        Some(0),
        "case {:?} dispatched wrongly",
        status.map(|c| c - 10)
    );
}

/// A caller and its callee leave a frame record each: the function that owns the host frame and
/// the call it was at; the depth is back to 0 once both return.
#[test]
fn calls_leave_frame_records_and_the_depth_returns_to_zero() {
    let program = lift_program(&image(), &[CALLER], &BTreeMap::new(), false);
    let source = program.source();
    assert!(
        source.contains(
            "push32(c, 0x00001015); if call(c, 0x00001010, K_CALL, fn_00001000) { return END; }"
        ),
        "{source}"
    );
    assert!(source.contains("enter(c, 0x00001010);"), "{source}");
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        run(c, fn_00001010);
        if (*c).eax != 1 { std::process::exit(4); }
        if (*c).depth != 0 { std::process::exit(5); }
        let f = (*c).frames;
        if (*f).function != 0x1010 || (*f).site != 0x1010 || (*f).kind != K_CALL { std::process::exit(6); }
        if (*f.add(1)).function != 0x1000 { std::process::exit(7); }
    }
}
",
    );
    let (status, stderr) = build_and_run("frames", &program, &main, "1");
    assert_eq!(status, Some(0), "{stderr}");
}

/// MSVC's `_setjmp3`: saves EBP, EBX, EDI, ESI, ESP and the return address in the buffer, and
/// returns 0.
const SETJMP: [u8; 27] = [
    0x8B, 0x54, 0x24, 0x04, 0x89, 0x2A, 0x89, 0x5A, 0x04, 0x89, 0x7A, 0x08, 0x89, 0x72, 0x0C, 0x89,
    0x62, 0x10, 0x8B, 0x04, 0x24, 0x89, 0x42, 0x14, 0x31, 0xC0, 0xC3,
];

/// `longjmp(buffer, value)`: restores, pops the return address, sets EAX, and jumps to the saved
/// one.
const LONGJMP: [u8; 28] = [
    0x8B, 0x54, 0x24, 0x04, 0x8B, 0x44, 0x24, 0x08, 0x8B, 0x2A, 0x8B, 0x5A, 0x04, 0x8B, 0x7A, 0x08,
    0x8B, 0x72, 0x0C, 0x8B, 0x62, 0x10, 0x83, 0xC4, 0x04, 0xFF, 0x62, 0x14,
];

/// MSVC's `setjmp` and `longjmp` as the title has them, and a caller that `setjmp`s, sees zero,
/// then `longjmp`s with 7: after the jump, `eax` must be 7 and the caller must be past its call.
#[test]
fn a_longjmp_lands_after_its_setjmp_with_the_value() {
    let mut bytes = vec![0x90u8; 0x100];
    bytes[..SETJMP.len()].copy_from_slice(&SETJMP);
    bytes[0x40..0x40 + LONGJMP.len()].copy_from_slice(&LONGJMP);
    let mut caller = vec![0x68];
    caller.extend_from_slice(&JMP_BUFFER.to_le_bytes()); // push buffer
    caller.extend(call_bytes(JMP_CALLER + 5, JMP_SETJMP));
    caller.extend([0x83, 0xC4, 0x04]); // add esp, 4
    caller.extend([0x85, 0xC0]); // test eax, eax
    caller.extend([0x75, 0x0C]); // jnz done
    caller.extend([0x6A, 0x07]); // push 7
    caller.push(0x68);
    caller.extend_from_slice(&JMP_BUFFER.to_le_bytes()); // push buffer
    let at = JMP_CALLER + u32::try_from(caller.len()).unwrap();
    caller.extend(call_bytes(at, JMP_LONGJMP));
    caller.push(0xC3); // done: ret
    bytes[0x80..0x80 + caller.len()].copy_from_slice(&caller);
    let program = lift_program(
        &Image::single(JMP_SETJMP, bytes),
        &[JMP_CALLER],
        &BTreeMap::new(),
        false,
    );
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        run(c, fn_00001080);
        if (*c).eax != 7 { std::process::exit(4); }
        if (*c.cast::<Thread>()).jmp_top != 0 { std::process::exit(5); }
        if (*c).depth != 0 { std::process::exit(6); }
    }
}
",
    );
    for opt in ["0", "1"] {
        let (status, stderr) = build_and_run("longjmp", &program, &main, opt);
        assert_eq!(status, Some(0), "-O{opt}: {stderr}");
    }
}

/// Two roots, the first running into the second, which loops back to the first: each body holds
/// only its own code and reaches the other by a tail call, so ten million turns of the loop keep
/// one host frame even unoptimized.
#[test]
fn a_body_that_reaches_another_root_tail_calls_it() {
    const FIRST: u32 = 0x1000;
    const SECOND: u32 = 0x1005;
    let bytes = vec![
        0xB8, 1, 0, 0, 0,    // mov eax, 1
        0x49, // second: dec ecx
        0x75, 0xF8, // jnz first
        0xC3, // ret
    ];
    let program = lift_program(
        &Image::single(FIRST, bytes),
        &[FIRST, SECOND],
        &BTreeMap::new(),
        false,
    );
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let [first, second] = program.bodies()[..] else {
        panic!("{:?}", program.bodies());
    };
    assert!(first.contains("{ return to(fn_00001005); }"), "{first}");
    assert!(!first.contains("0x00001005 =>"), "{first}");
    assert!(
        second.contains("if poll(c, 0x00001006) { return END; } { return to(fn_00001000); }"),
        "{second}"
    );
    assert!(!second.contains("0x00001000 =>"), "{second}");
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        (*c).ecx = 10_000_000;
        run(c, fn_00001000);
        if (*c).ecx != 0 || (*c).eax != 1 { std::process::exit(4); }
        if (*c).esp != 0xF004 { std::process::exit(5); }
    }
}
",
    );
    let (status, stderr) = build_and_run("tail", &program, &main, "0");
    assert_eq!(status, Some(0), "{stderr}");
}

/// A function of 600 blocks (`inc eax ; jmp next` each) inside a loop that runs three times: it
/// is cut into chunks, and the jumps between them, forwards and back to the loop's start, still
/// go where the code says.
#[test]
fn a_function_too_large_for_one_chunk_runs_across_its_chunks() {
    const BLOCKS: usize = 600;
    let mut bytes = vec![0x90u8; 8 + BLOCKS * 3 + 16];
    bytes[..5].copy_from_slice(&[0xB9, 3, 0, 0, 0]); // mov ecx, 3
    let top = 5;
    for n in 0..BLOCKS {
        let at = top + n * 3;
        bytes[at..at + 3].copy_from_slice(&[0x40, 0xEB, 0x00]); // inc eax ; jmp next
    }
    let end = top + BLOCKS * 3;
    bytes[end] = 0x49; // dec ecx
    // jnz top (rel32), then ret
    let rel = i32::try_from(top).unwrap() - i32::try_from(end + 7).unwrap();
    bytes[end + 1..end + 3].copy_from_slice(&[0x0F, 0x85]);
    bytes[end + 3..end + 7].copy_from_slice(&rel.to_le_bytes());
    bytes[end + 7] = 0xC3;
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &[0x1000],
        &BTreeMap::new(),
        false,
    );
    assert!(program.traps.is_empty() && program.failures.is_empty());
    let source = program.source();
    assert!(source.contains("unsafe fn fn_00001000_2("), "not chunked");
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        run(c, fn_00001000);
        if (*c).eax != 1800 { std::process::exit(4); }
        if (*c).esp != 0xF004 { std::process::exit(5); }
    }
}
",
    );
    for opt in ["0", "1"] {
        let (status, stderr) = build_and_run("chunks", &program, &main, opt);
        assert_eq!(status, Some(0), "-O{opt}: {stderr}");
    }
}

/// A jump table indexed by EBP (`cmp ebp, 3 ; ja out ; jmp [ebp*4 + table]`), which a compiler
/// picks when the other registers are busy: it is a table, not an unknown indirect jump.
#[test]
fn a_jump_table_indexed_by_ebp_is_found() {
    const TABLE: u32 = 0x1100;
    let mut bytes = vec![0x90u8; 0x200];
    // cmp ebp, 3 ; ja +0x0D (the default) ; jmp [ebp*4 + TABLE]
    bytes[..5].copy_from_slice(&[0x83, 0xFD, 0x03, 0x77, 0x0D]);
    bytes[5] = 0xFF;
    bytes[6] = 0x24;
    bytes[7] = 0xAD;
    bytes[8..12].copy_from_slice(&TABLE.to_le_bytes());
    // The default, at 0x100C: ret. Four cases of `mov eax, n ; ret`, at 0x1020 and up.
    bytes[0x0C] = 0xC3;
    bytes[0x0D] = 0xC3;
    for n in 0..4usize {
        let at = 0x20 + n * 8;
        bytes[at..at + 6].copy_from_slice(&[0xB8, u8::try_from(n).unwrap(), 0, 0, 0, 0xC3]);
        let entry = 0x1000 + u32::try_from(at).unwrap();
        bytes[0x100 + n * 4..0x104 + n * 4].copy_from_slice(&entry.to_le_bytes());
    }
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &[0x1000],
        &BTreeMap::new(),
        false,
    );
    assert!(
        program.traps.is_empty() && program.failures.is_empty(),
        "{:?} {:?}",
        program.traps,
        program.failures
    );
    assert!(program.source().contains("match (*c).ebp {"));
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        for case in 0..4u32 {
            (*c).ebp = case;
            (*c).eax = 77;
            run(c, fn_00001000);
            if (*c).eax != case { std::process::exit(10 + case as i32); }
        }
    }
}
",
    );
    let (status, stderr) = build_and_run("table", &program, &main, "1");
    assert_eq!(status, Some(0), "{stderr}");
}

/// A switch whose bound sits 22 bytes before its jump, with stores and pushes scheduled in
/// between (as one title has: `cmp ecx, 0x1d ; mov [m], edx ; mov [m], eax ; ja out ; push esi ; push edi ;
/// jmp [ecx*4 + table]`): every one of its 30 cases is found.
#[test]
fn a_bound_far_before_the_jump_still_bounds_the_table() {
    const TABLE: u32 = 0x1200;
    let mut bytes = vec![0x90u8; 0x400];
    let code: [u8; 22] = [
        0x83, 0xF9, 0x1D, // cmp ecx, 0x1d
        0x89, 0x15, 0x00, 0x13, 0x00, 0x00, // mov [0x1300], edx
        0xA3, 0x04, 0x13, 0x00, 0x00, // mov [0x1304], eax
        0x0F, 0x87, 0x00, 0x00, 0x00, 0x00, // ja out (patched below)
        0x56, 0x57, // push esi ; push edi
    ];
    bytes[..22].copy_from_slice(&code);
    // jmp [ecx*4 + TABLE]
    bytes[22..25].copy_from_slice(&[0xFF, 0x24, 0x8D]);
    bytes[25..29].copy_from_slice(&TABLE.to_le_bytes());
    // The default at 0x1020: ret. The ja's displacement is from 0x1014.
    bytes[0x20] = 0xC3;
    bytes[16..20].copy_from_slice(&(0x1020u32 - 0x1014).to_le_bytes());
    // Thirty cases of `mov eax, n ; pop edi ; pop esi ; ret`, at 0x1040 and up.
    for n in 0..30usize {
        let at = 0x40 + n * 8;
        bytes[at..at + 8].copy_from_slice(&[
            0xB8,
            u8::try_from(n).unwrap(),
            0,
            0,
            0,
            0x5F,
            0x5E,
            0xC3,
        ]);
        let entry = 0x1000 + u32::try_from(at).unwrap();
        bytes[0x200 + n * 4..0x204 + n * 4].copy_from_slice(&entry.to_le_bytes());
    }
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &[0x1000],
        &BTreeMap::new(),
        false,
    );
    let source = program.source();
    assert!(source.contains(" 29u32 =>"), "{source}");
    assert!(!source.contains(" 30u32 =>"));
}

/// `jmp eax` through a function pointer is a tail call: the callee returns to our caller.
#[test]
fn a_jump_through_a_register_is_a_tail_call() {
    // mov eax, [esp+4] ; jmp eax
    let mut bytes = vec![0x90u8; 0x40];
    bytes[..6].copy_from_slice(&[0x8B, 0x44, 0x24, 0x04, 0xFF, 0xE0]);
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &[0x1000],
        &BTreeMap::new(),
        false,
    );
    assert!(program.traps.is_empty() && program.failures.is_empty());
    let source = program.source();
    assert!(
        source.contains("call_rt(c, 0x00001004, K_RET, t_target, 0x00001004); return END;"),
        "{source}"
    );
}

/// `cvtps2pi`, `cvttps2pi` and `cvtpi2ps`, run: 2.5 and -1.5 round to 2 and -2, truncate to 2 and -1,
/// and the integers convert back to floats.
#[test]
fn the_packed_float_and_integer_conversions_run() {
    // cvtps2pi mm0, xmm0 ; cvttps2pi mm1, xmm0 ; cvtpi2ps xmm2, mm1 ; ret
    let mut bytes = vec![0x90u8; 0x40];
    bytes[..10].copy_from_slice(&[0x0F, 0x2D, 0xC0, 0x0F, 0x2C, 0xC8, 0x0F, 0x2A, 0xD1, 0xC3]);
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &[0x1000],
        &BTreeMap::new(),
        false,
    );
    assert!(
        program.traps.is_empty() && program.failures.is_empty(),
        "{:?} {:?}",
        program.traps,
        program.failures
    );
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        (*xmm(c, 0)).f[0] = 2.5;
        (*xmm(c, 0)).f[1] = -1.5;
        run(c, fn_00001000);
        if *mm(c, 0) != (2u64 | (0xFFFF_FFFEu64 << 32)) { std::process::exit(4); }
        if *mm(c, 1) != (2u64 | (0xFFFF_FFFFu64 << 32)) { std::process::exit(5); }
        if (*xmm(c, 2)).f[0] != 2.0 || (*xmm(c, 2)).f[1] != -1.0 { std::process::exit(6); }
    }
}
",
    );
    let (status, stderr) = build_and_run("cvt", &program, &main, "1");
    assert_eq!(status, Some(0), "{stderr}");
}

/// The 8- and 16-bit `mul`, `imul`, `div` and `idiv`, and the MMX `pextrw`, `pinsrw` and
/// `pmovmskb`, each run on a value checked by hand.
#[test]
fn the_small_integer_forms_and_the_mmx_extensions_run() {
    // One function every 0x10 bytes: the instruction, then ret.
    let code: [&[u8]; 11] = [
        &[0xF6, 0xE3],             // mul bl
        &[0x66, 0xF7, 0xE3],       // mul bx
        &[0xF6, 0xF3],             // div bl
        &[0xF6, 0xFB],             // idiv bl
        &[0x66, 0xF7, 0xF3],       // div bx
        &[0x66, 0xF7, 0xFB],       // idiv bx
        &[0xF6, 0xEB],             // imul bl
        &[0x66, 0xF7, 0xEB],       // imul bx
        &[0x0F, 0xC5, 0xC0, 0x02], // pextrw eax, mm0, 2
        &[0x0F, 0xC4, 0xC8, 0x01], // pinsrw mm1, eax, 1
        &[0x0F, 0xD7, 0xC0],       // pmovmskb eax, mm0
    ];
    let mut bytes = vec![0x90u8; 0x100];
    for (n, instruction) in code.iter().enumerate() {
        let at = n * 0x10;
        bytes[at..at + instruction.len()].copy_from_slice(instruction);
        bytes[at + instruction.len()] = 0xC3;
    }
    let roots: Vec<u32> = (0..11).map(|n| 0x1000 + n * 0x10).collect();
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &roots,
        &BTreeMap::new(),
        false,
    );
    assert!(
        program.traps.is_empty() && program.failures.is_empty(),
        "{:?} {:?}",
        program.traps,
        program.failures
    );
    let main = harness(
        r"
fn check(code: i32, what: bool) {
    if !what { std::process::exit(code); }
}
fn main() {
    unsafe {
        let c = setup();
        (*c).eax = 0xAABB00C8; (*c).ebx = 3; run(c, fn_00001000);                       // mul bl: 200 * 3
        check(10, (*c).eax == 0xAABB0258 && (*c).cf != 0);
        (*c).eax = 0x11118000; (*c).edx = 0xFFFF1111; (*c).ebx = 4; run(c, fn_00001010); // mul bx: 0x8000 * 4
        check(11, (*c).eax == 0x11110000 && (*c).edx == 0xFFFF0002);
        (*c).eax = 0x11110257; (*c).ebx = 10; run(c, fn_00001020);                      // div bl: 599 / 10
        check(12, (*c).eax == 0x1111093B);
        (*c).eax = 0x1111FDA8; (*c).ebx = 7; run(c, fn_00001030);                       // idiv bl: -600 / 7
        check(13, (*c).eax == 0x1111FBAB);
        (*c).eax = 0; (*c).edx = 1; (*c).ebx = 3; run(c, fn_00001040);                  // div bx: 0x10000 / 3
        check(14, ((*c).eax & 0xFFFF) == 0x5555 && ((*c).edx & 0xFFFF) == 1);
        (*c).eax = 0x00007960; (*c).edx = 0x0000FFFE; (*c).ebx = 7; run(c, fn_00001050); // idiv bx: -100000 / 7
        check(15, ((*c).eax & 0xFFFF) == 0xC833 && ((*c).edx & 0xFFFF) == 0xFFFB);
        (*c).eax = 0xAABB00FD; (*c).ebx = 5; run(c, fn_00001060);                       // imul bl: -3 * 5
        check(16, ((*c).eax & 0xFFFF) == 0xFFF1);
        (*c).eax = 0x0000FED4; (*c).ebx = 200; run(c, fn_00001070);                     // imul bx: -300 * 200
        check(17, ((*c).eax & 0xFFFF) == 0x15A0 && ((*c).edx & 0xFFFF) == 0xFFFF);
        *mm(c, 0) = 0x1111222233334444; run(c, fn_00001080);                            // pextrw eax, mm0, 2
        check(18, (*c).eax == 0x2222);
        (*c).eax = 0xDEADBEEF; *mm(c, 1) = 0; run(c, fn_00001090);                      // pinsrw mm1, eax, 1
        check(19, *mm(c, 1) == 0x00000000BEEF0000);
        *mm(c, 0) = 0x8000FF01807F0080; run(c, fn_000010a0);                            // pmovmskb eax, mm0
        check(20, (*c).eax == 0xA9);
    }
}
",
    );
    let (status, stderr) = build_and_run("int", &program, &main, "1");
    assert_eq!(status, Some(0), "failed check {status:?}: {stderr}");
}

/// Without `fninit`, a thread's x87 and SSE start as the console's kernel starts them: control
/// word 0x027F (53-bit precision, every exception masked) and MXCSR 0x1F80; `fninit` gives
/// 0x037F. The precision field still rounds to single only when it says so.
#[test]
fn a_thread_starts_with_the_consoles_floating_point_state() {
    let mut bytes = vec![0x90u8; 0x80];
    let state: &[u8] = &[
        0xD9, 0x3D, 0x00, 0x20, 0x00, 0x00, // fnstcw [0x2000]
        0x0F, 0xAE, 0x1D, 0x04, 0x20, 0x00, 0x00, // stmxcsr [0x2004]
        0xDB, 0xE3, // fninit
        0xD9, 0x3D, 0x02, 0x20, 0x00, 0x00, // fnstcw [0x2002]
        0xC3,
    ];
    let divide: &[u8] = &[
        0xD9, 0x2D, 0x08, 0x20, 0x00, 0x00, // fldcw [0x2008]
        0xDD, 0x05, 0x10, 0x20, 0x00, 0x00, // fld qword [0x2010]
        0xDC, 0x35, 0x18, 0x20, 0x00, 0x00, // fdiv qword [0x2018]
        0xDD, 0x1D, 0x20, 0x20, 0x00, 0x00, // fstp qword [0x2020]
        0xC3,
    ];
    bytes[..state.len()].copy_from_slice(state);
    bytes[0x40..0x40 + divide.len()].copy_from_slice(divide);
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &[0x1000, 0x1040],
        &BTreeMap::new(),
        false,
    );
    assert!(
        program.traps.is_empty() && program.failures.is_empty(),
        "{:?} {:?}",
        program.traps,
        program.failures
    );
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        run(c, fn_00001000);
        if rd16(0x2000) != 0x027F { std::process::exit(4); }
        if rd32(0x2004) != 0x1F80 { std::process::exit(5); }
        if rd16(0x2002) != 0x037F { std::process::exit(6); }
        stf64(0x2010, 1.0);
        stf64(0x2018, 3.0);
        for (i, word) in [0x007Fu32, 0x027F, 0x037F].into_iter().enumerate() {
            wr16(0x2008, word);
            run(c, fn_00001040);
            let expected = if i == 0 { f64::from((1.0f64 / 3.0) as f32) } else { 1.0 / 3.0 };
            if ldf64(0x2020) != expected { std::process::exit(10 + i as i32); }
        }
    }
}
",
    );
    let (status, stderr) = build_and_run("fpu", &program, &main, "1");
    assert_eq!(status, Some(0), "failed check {status:?}: {stderr}");
}

/// `fnsave` stores the control and status words and the stack, ST(0) first as 80-bit values,
/// then initializes the unit; `frstor` brings all of it back.
#[test]
fn fnsave_and_frstor_keep_the_whole_x87_state() {
    let code: &[u8] = &[
        0xD9, 0xE8, // fld1
        0xD9, 0xEB, // fldpi
        0xDD, 0x35, 0x00, 0x30, 0x00, 0x00, // fnsave [0x3000]
        0xD9, 0x3D, 0x30, 0x20, 0x00, 0x00, // fnstcw [0x2030]
        0xD9, 0xEE, // fldz
        0xDD, 0x25, 0x00, 0x30, 0x00, 0x00, // frstor [0x3000]
        0xDD, 0x1D, 0x20, 0x20, 0x00, 0x00, // fstp qword [0x2020]
        0xDD, 0x1D, 0x28, 0x20, 0x00, 0x00, // fstp qword [0x2028]
        0xD9, 0x3D, 0x32, 0x20, 0x00, 0x00, // fnstcw [0x2032]
        0xC3,
    ];
    let mut bytes = vec![0x90u8; 0x40];
    bytes[..code.len()].copy_from_slice(code);
    let program = lift_program(
        &Image::single(0x1000, bytes),
        &[0x1000],
        &BTreeMap::new(),
        false,
    );
    assert!(
        program.traps.is_empty() && program.failures.is_empty(),
        "{:?} {:?}",
        program.traps,
        program.failures
    );
    let main = harness(
        r"
fn main() {
    unsafe {
        let c = setup();
        run(c, fn_00001000);
        if rd16(0x3000) != 0x027F { std::process::exit(4); }
        if rd16(0x3004) != 0x3000 { std::process::exit(5); }
        if rd16(0x3024) != 0x4000 || rd16(0x302E) != 0x3FFF { std::process::exit(6); }
        if rd16(0x2030) != 0x037F || rd16(0x2032) != 0x027F { std::process::exit(7); }
        if ldf64(0x2020) != core::f64::consts::PI || ldf64(0x2028) != 1.0 { std::process::exit(8); }
    }
}
",
    );
    let (status, stderr) = build_and_run("fsave", &program, &main, "1");
    assert_eq!(status, Some(0), "failed check {status:?}: {stderr}");
}

/// What a resume test's `main` needs besides its own function: stubs for the runtime, and a
/// freeze and a resume as the runtime does them, inside one process. `freeze` saves the CPU, the
/// frames, the `setjmp` contexts and guest memory, then leaves the thread's host frames as a
/// `longjmp` to no context would; `resume` loads them into fresh state and calls down the frames
/// again.
const RESUME_HARNESS: &str = r#"
use std::sync::Mutex;

struct Saved {
    cpu: Vec<u8>,
    frames: Vec<Frame>,
    jumps: Vec<JmpSaved>,
    memory: Vec<u8>,
}

static SAVED: Mutex<Option<Saved>> = Mutex::new(None);
static mut CPU: *mut Cpu = core::ptr::null_mut();
static mut FREEZING: bool = false;
/// The polls still to pass before the one that freezes; the flag stays raised until then.
static mut POLLS_LEFT: u32 = 0;
/// The kind of the innermost frame the freeze saved.
static mut INNERMOST: u32 = 0;
const MEMORY: usize = 1 << 20;

unsafe fn freeze(c: *mut Cpu, count: u32) {
    unsafe {
        FREEZING = false;
        let t = c.cast::<Thread>();
        let frames = std::slice::from_raw_parts((*c).frames, count as usize).to_vec();
        INNERMOST = frames.last().map_or(0, |f| f.kind);
        let jumps = (0..(*t).jmp_top as usize)
            .map(|i| {
                let j = *(*t).jmp_stack.as_ptr().add(i);
                JmpSaved { buf: (*j).buf, resume: (*j).resume, depth: (*j).depth, site: (*j).site }
            })
            .collect();
        *SAVED.lock().unwrap() = Some(Saved {
            cpu: std::slice::from_raw_parts(c.cast::<u8>(), size_of::<Cpu>()).to_vec(),
            frames,
            jumps,
            memory: std::slice::from_raw_parts(G, MEMORY).to_vec(),
        });
        // Leave every host frame: a jump to a context no frame holds.
        (*t).jumping = core::ptr::NonNull::dangling().as_ptr();
    }
}

unsafe fn resume(c: *mut Cpu) {
    unsafe {
        let saved = SAVED.lock().unwrap().take().unwrap();
        let t = c.cast::<Thread>();
        let frames = (*c).frames;
        std::ptr::copy_nonoverlapping(saved.memory.as_ptr(), G, MEMORY);
        std::ptr::copy_nonoverlapping(saved.cpu.as_ptr(), c.cast::<u8>(), size_of::<Cpu>());
        (*c).frames = frames;
        std::ptr::write_bytes(frames, 0, 64);
        std::ptr::copy_nonoverlapping(saved.frames.as_ptr(), frames, saved.frames.len());
        for (i, jump) in saved.jumps.iter().enumerate() {
            *(*t).restore.as_mut_ptr().add(i) = *jump;
        }
        (*t).restore_count = saved.jumps.len() as u32;
        (*t).jmp_top = 0;
        (*c).resume_depth = saved.frames.len() as u32;
        (*c).depth = 0;
        (*c).resuming = 1;
        lift_resume_root(c);
    }
}

/// The kernel call: returns 100 and pops its return address; while freezing, it parks instead.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_native(c: *mut Cpu, _target: u32, _at: u32) {
    unsafe {
        if FREEZING {
            freeze(c, (*c).depth);
            return;
        }
        (*c).eax = 100;
        (*c).esp += 4;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn anyxbox_preempt() {
    unsafe {
        if FREEZING {
            POLLS_LEFT -= 1;
            if POLLS_LEFT > 0 {
                return;
            }
        }
        anyxbox_preempt_flag = 0;
        if FREEZING {
            freeze(CPU, (*CPU).depth + 1);
        }
    }
}

/// The innermost frame: a parked kernel call completes now; a poll just goes on.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_resume_innermost(c: *mut Cpu) {
    unsafe {
        (*c).resuming = 0;
        if INNERMOST != K_POLL {
            (*c).eax = 100;
            (*c).esp += 4;
        }
    }
}

/// Runs `root` through, then again frozen and resumed: at its first kernel call, or with `poll`
/// at that poll (1 the first); both runs must end with the same EAX and ESP, and the resumed one
/// with no depth left.
unsafe fn both_runs(root: Lifted, poll: u32) -> i32 {
    unsafe {
        G = Box::leak(vec![0u8; MEMORY].into_boxed_slice()).as_mut_ptr();
        let c = lift_thread_new();
        CPU = c;
        (*c).esp = 0xF000;
        run(c, root);
        let (eax, esp) = ((*c).eax, (*c).esp);
        if (*c).depth != 0 {
            return 2;
        }
        let frames = (*c).frames;
        std::ptr::write_bytes(c.cast::<u8>(), 0, size_of::<Thread>());
        (*c).frames = frames;
        std::ptr::write_bytes(G, 0, MEMORY);
        (*c).esp = 0xF000;
        FREEZING = true;
        POLLS_LEFT = poll;
        anyxbox_preempt_flag = i32::from(poll != 0);
        run(c, root);
        let t = c.cast::<Thread>();
        if (*t).jumping.is_null() {
            return 3;
        }
        (*t).jumping = core::ptr::null_mut();
        if (*c).depth != 0 || (*t).jmp_top != 0 {
            return 7;
        }
        resume(c);
        if (*c).eax != eax || (*c).esp != esp {
            eprintln!("eax {:x}, not {eax:x}; esp {:x}, not {esp:x}", (*c).eax, (*c).esp);
            return 4;
        }
        if (*c).depth != 0 || (*c).resuming != 0 || (*t).jmp_top != 0 {
            return 5;
        }
        0
    }
}
"#;

/// A resume test's program: the runtime stubs, the resume harness and `main`.
fn resume_harness(main: &str) -> String {
    format!("{RUNTIME_STUBS}{RESUME_HARNESS}{main}")
}

/// A call rel32 from `from` to `to`.
fn call_bytes(from: u32, to: u32) -> Vec<u8> {
    let mut call = vec![0xE8];
    call.extend_from_slice(&to.wrapping_sub(from + 5).to_le_bytes());
    call
}

/// Frozen in a kernel call two lifted calls deep, the thread resumes and finishes as if it never
/// stopped.
#[test]
fn a_thread_frozen_in_a_kernel_call_resumes_down_its_calls() {
    const ROOT: u32 = 0x1000;
    const MIDDLE: u32 = 0x1010;
    const NATIVE: u32 = 0x1030;
    let mut bytes = vec![0xC3u8; 0x40];
    let mut root = vec![0xB9, 3, 0, 0, 0]; // mov ecx, 3
    root.extend(call_bytes(ROOT + 5, MIDDLE));
    root.extend([0x01, 0xC8, 0xC3]); // add eax, ecx ; ret
    bytes[..root.len()].copy_from_slice(&root);
    let mut middle = vec![0x53, 0xBB, 5, 0, 0, 0]; // push ebx ; mov ebx, 5
    middle.extend(call_bytes(MIDDLE + 6, NATIVE));
    middle.extend([0x01, 0xD8, 0x5B, 0xC3]); // add eax, ebx ; pop ebx ; ret
    bytes[0x10..0x10 + middle.len()].copy_from_slice(&middle);
    let hooks = BTreeMap::from([(NATIVE, TRAP)]);
    let program = lift_program(&Image::single(ROOT, bytes), &[ROOT], &hooks, false);
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let main = resume_harness(
        "fn main() { unsafe { let mut r = both_runs(fn_00001000, 0); if r == 0 && (*CPU).eax != 108 { r = 6; } std::process::exit(r); } }\n",
    );
    let (status, stderr) = build_and_run("resume-call", &program, &main, "1");
    assert_eq!(status, Some(0), "{stderr}");
}

/// Frozen at a loop's poll halfway through, the thread resumes at that poll and finishes the
/// loop. A resume that started the function over would count the memory word up twice and run the
/// loop again from the start: EAX would be 1002, not 1001.
#[test]
fn a_thread_frozen_at_a_poll_resumes_there() {
    const ROOT: u32 = 0x1000;
    let bytes = vec![
        0xFF, 0x05, 0x00, 0x20, 0, 0, // inc dword [0x2000]
        0xB9, 0xE8, 0x03, 0, 0, // mov ecx, 1000
        0x31, 0xC0, // xor eax, eax
        0x40, // loop: inc eax
        0x49, // dec ecx
        0x75, 0xFC, // jnz loop
        0x03, 0x05, 0x00, 0x20, 0, 0,    // add eax, [0x2000]
        0xC3, // ret
    ];
    let program = lift_program(
        &Image::single(ROOT, bytes),
        &[ROOT],
        &BTreeMap::new(),
        false,
    );
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let main = resume_harness(
        "fn main() { unsafe { let mut r = both_runs(fn_00001000, 500); if r == 0 && ((*CPU).eax != 1001 || rd32(0x2000) != 1) { r = 6; } std::process::exit(r); } }\n",
    );
    let (status, stderr) = build_and_run("resume-poll", &program, &main, "1");
    assert_eq!(status, Some(0), "{stderr}");
}

/// Frozen in a kernel call after a `setjmp`, the thread registers the context again as it
/// resumes, so the `longjmp` after the call still lands.
#[test]
fn a_setjmp_context_survives_a_resume() {
    const NATIVE: u32 = 0x10C0;
    let mut bytes = vec![0x90u8; 0x100];
    bytes[..SETJMP.len()].copy_from_slice(&SETJMP);
    bytes[0x40..0x40 + LONGJMP.len()].copy_from_slice(&LONGJMP);
    bytes[0xC0] = 0xC3;
    let mut caller = vec![0x68];
    caller.extend_from_slice(&JMP_BUFFER.to_le_bytes()); // push buffer
    caller.extend(call_bytes(JMP_CALLER + 5, JMP_SETJMP));
    caller.extend([0x83, 0xC4, 0x04]); // add esp, 4
    caller.extend([0x85, 0xC0]); // test eax, eax
    caller.extend([0x75, 0x11]); // jnz done
    let at = JMP_CALLER + u32::try_from(caller.len()).unwrap();
    caller.extend(call_bytes(at, NATIVE)); // the kernel call it freezes in
    caller.extend([0x6A, 0x07]); // push 7
    caller.push(0x68);
    caller.extend_from_slice(&JMP_BUFFER.to_le_bytes()); // push buffer
    let at = JMP_CALLER + u32::try_from(caller.len()).unwrap();
    caller.extend(call_bytes(at, JMP_LONGJMP));
    caller.push(0xC3); // done: ret
    bytes[0x80..0x80 + caller.len()].copy_from_slice(&caller);
    let hooks = BTreeMap::from([(NATIVE, TRAP)]);
    let program = lift_program(
        &Image::single(JMP_SETJMP, bytes),
        &[JMP_CALLER],
        &hooks,
        false,
    );
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let main = resume_harness(
        "fn main() { unsafe { let mut r = both_runs(fn_00001080, 0); if r == 0 && (*CPU).eax != 7 { r = 6; } std::process::exit(r); } }\n",
    );
    let (status, stderr) = build_and_run("resume-setjmp", &program, &main, "1");
    assert_eq!(status, Some(0), "{stderr}");
}
