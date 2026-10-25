#![allow(clippy::too_many_lines)] // each test spells out its C harness

//! Lifts a few hand-assembled functions and checks the C: its shape, and that a C compiler accepts
//! it (when one is installed).

use std::collections::BTreeMap;
use std::process::Command;

use lift::{Image, lift_program};

const LEAF: u32 = 0x1000;
const CALLER: u32 = 0x1010;
const HOOKED: u32 = 0x1020;
const TRAP: u32 = 0x8001_0800;
const JMP_SETJMP: u32 = 0x1000;
const JMP_LONGJMP: u32 = 0x1040;
const JMP_CALLER: u32 = 0x1080;
const JMP_BUFFER: u32 = 0x2000;

/// `mov eax, 1 ; ret`, then `call LEAF ; ret`, then a function the runtime replaces.
fn image() -> Image {
    let mut bytes = vec![0x90u8; 0x30];
    bytes[..6].copy_from_slice(&[0xB8, 1, 0, 0, 0, 0xC3]);
    let rel = LEAF.wrapping_sub(CALLER + 5);
    bytes[0x10] = 0xE8;
    bytes[0x11..0x15].copy_from_slice(&rel.to_le_bytes());
    bytes[0x15] = 0xC3;
    bytes[0x20..0x26].copy_from_slice(&[0xB8, 9, 0, 0, 0, 0xC3]);
    Image::single(LEAF, bytes)
}

#[test]
fn a_leaf_and_its_caller_lift_and_the_callee_is_found_from_the_caller() {
    let program = lift_program(&image(), &[CALLER], &BTreeMap::new(), false);
    assert_eq!(program.lifted, 2, "the caller and what it calls");
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let source = program.source();
    assert!(source.contains("void fn_00001010(void)"), "{source}");
    assert!(source.contains("void fn_00001000(void)"), "{source}");
}

#[test]
fn a_hooked_function_is_a_call_to_its_trap_and_its_own_code_is_ignored() {
    let hooks = BTreeMap::from([(HOOKED, TRAP)]);
    let program = lift_program(&image(), &[HOOKED], &hooks, false);
    let source = program.source();
    assert!(
        source.contains("rt_call(0x80010800u, 0x00001020u)"),
        "{source}"
    );
}

#[test]
fn an_address_outside_the_image_does_not_stop_the_others() {
    let program = lift_program(&image(), &[LEAF, 0x9000], &BTreeMap::new(), false);
    assert_eq!(program.lifted, 1);
    assert_eq!(program.failures.len(), 1);
}

#[test]
fn the_lifted_c_compiles() {
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
    let hooks = BTreeMap::from([(HOOKED, TRAP)]);
    let program = lift_program(&image(), &[CALLER, HOOKED], &hooks, false);
    let dir = std::env::temp_dir().join(format!("anybox-lift-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("lifted.c");
    std::fs::write(&file, program.source()).unwrap();
    let output = Command::new(clang)
        .args(["-fsyntax-only", "-w"])
        .arg(&file)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn the_dispatcher_finds_each_function_and_sends_anything_else_to_the_native_hook() {
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
    let hooks = BTreeMap::from([(HOOKED, TRAP)]);
    let program = lift_program(&image(), &[CALLER, HOOKED], &hooks, false);
    let dir = std::env::temp_dir().join(format!("anybox-lift-dispatch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("dispatch.c"),
        format!("#include \"lift.h\"\n{}", program.dispatch()),
    )
    .unwrap();
    // The functions record that they ran. Address 1 shares the cache slot of 0x1000 and is not a
    // function; 0 is not one either (an empty cache slot must not look like a hit).
    std::fs::write(
        dir.join("main.c"),
        r#"#define LIFT_DEFINE_GLOBALS
#include "lift.h"
static int last;
void fn_00001000(void) { last = 1; }
void fn_00001010(void) { last = 2; }
void fn_00001020(void) { last = 3; }
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { last = -1; }
int main(void) {
    lift_frames_init();
    uint32_t addresses[] = {0x1000, 0x1010, 0x1020, 0x1000, 1, 0x1000, 0x0fff, 0x1001, 0x2000, 0, 0xffffffffu, 0x1020};
    int wanted[] = {1, 2, 3, 1, -1, 1, -1, -1, -1, -1, -1, 3};
    for (int i = 0; i < 12; i++) {
        last = 99;
        rt_call(addresses[i], 0);
        if (last != wanted[i]) return 10 + i;
    }
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("dispatch.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).status().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        run.code(),
        Some(0),
        "case {:?} dispatched wrongly",
        run.code().map(|c| c - 10)
    );
}

/// A caller and its callee leave a frame record each: the function that owns the host frame and
/// the call it was at; the depth is back to 0 once both return.
#[test]
fn calls_leave_frame_records_and_the_depth_returns_to_zero() {
    let program = lift_program(&image(), &[CALLER], &BTreeMap::new(), false);
    let source = program.source();
    assert!(
        source.contains("LIFT_IN(0x00001010u, LIFT_K_CALL, fn_00001000()); C_00001010:;"),
        "{source}"
    );
    assert!(source.contains("LIFT_ENTER(0x00001010u);"), "{source}");
    let main = r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
void anyxbox_preempt(void) {}
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { exit(3); }
int main(void) {
    lift_frames_init();
    G = calloc(1, 1 << 20);
    R_ESP = 0xF000;
    fn_00001010();
    if (R_EAX != 1) return 4;
    if (lift_cpu.depth != 0) return 5;
    lift_frame *f = lift_cpu.frames;
    if (f[0].fn != 0x1010 || f[0].site != 0x1010 || f[0].kind != LIFT_K_CALL) return 6;
    if (f[1].fn != 0x1000) return 7;
    return 0;
}
"#;
    let Some((status, stderr)) = build_and_run("frames", &program, main) else {
        eprintln!("no clang; skipping");
        return;
    };
    assert_eq!(status, Some(0), "{stderr}");
}

/// Builds `program` (as one unit that defines the globals) with `main_c` at `-O1`, runs it, and
/// returns its exit status and messages; `None` without clang.
fn build_and_run(
    name: &str,
    program: &lift::Program,
    main_c: &str,
) -> Option<(Option<i32>, String)> {
    let clang = which("clang").ok()?;
    let dir = std::env::temp_dir().join(format!("anybox-lift-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("lifted.c"),
        format!(
            "#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{}{}",
            program.bodies().concat(),
            program.dispatch()
        ),
    )
    .unwrap();
    std::fs::write(dir.join("main.c"), main_c).unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-O1", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("lifted.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    Some((
        run.status.code(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
    ))
}

/// MSVC's `setjmp` and `longjmp` as the title has them, and a caller that `setjmp`s, sees zero,
/// then `longjmp`s with 7: after the jump, `eax` must be 7 and the caller must be past its call.
#[test]
fn a_longjmp_lands_after_its_setjmp_with_the_value() {
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
    let mut bytes = vec![0x90u8; 0x100];
    // _setjmp3: save EBP, EBX, EDI, ESI, ESP and the return address in the buffer; return 0.
    let setjmp = [
        0x8B, 0x54, 0x24, 0x04, 0x89, 0x2A, 0x89, 0x5A, 0x04, 0x89, 0x7A, 0x08, 0x89, 0x72, 0x0C,
        0x89, 0x62, 0x10, 0x8B, 0x04, 0x24, 0x89, 0x42, 0x14, 0x31, 0xC0, 0xC3,
    ];
    // longjmp(buffer, value): restore, pop the return address, set EAX, jump to the saved one.
    let longjmp = [
        0x8B, 0x54, 0x24, 0x04, 0x8B, 0x44, 0x24, 0x08, 0x8B, 0x2A, 0x8B, 0x5A, 0x04, 0x8B, 0x7A,
        0x08, 0x8B, 0x72, 0x0C, 0x8B, 0x62, 0x10, 0x83, 0xC4, 0x04, 0xFF, 0x62, 0x14,
    ];
    bytes[..setjmp.len()].copy_from_slice(&setjmp);
    bytes[0x40..0x40 + longjmp.len()].copy_from_slice(&longjmp);
    let mut caller = vec![0x68];
    caller.extend_from_slice(&JMP_BUFFER.to_le_bytes()); // push buffer
    let call = |from: u32, to: u32| {
        let mut call = vec![0xE8];
        call.extend_from_slice(&to.wrapping_sub(from + 5).to_le_bytes());
        call
    };
    caller.extend(call(JMP_CALLER + 5, JMP_SETJMP));
    caller.extend([0x83, 0xC4, 0x04]); // add esp, 4
    caller.extend([0x85, 0xC0]); // test eax, eax
    caller.extend([0x75, 0x0C]); // jnz done
    caller.extend([0x6A, 0x07]); // push 7
    caller.push(0x68);
    caller.extend_from_slice(&JMP_BUFFER.to_le_bytes()); // push buffer
    let at = JMP_CALLER + u32::try_from(caller.len()).unwrap();
    caller.extend(call(at, JMP_LONGJMP));
    caller.push(0xC3); // done: ret
    bytes[0x80..0x80 + caller.len()].copy_from_slice(&caller);
    let program = lift_program(
        &Image::single(JMP_SETJMP, bytes),
        &[JMP_CALLER],
        &BTreeMap::new(),
        false,
    );
    assert!(program.failures.is_empty() && program.traps.is_empty());
    let dir = std::env::temp_dir().join(format!("anybox-lift-longjmp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("lifted.c"),
        format!(
            "#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{}{}",
            program.bodies().concat(),
            program.dispatch()
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("main.c"),
        r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
void anyxbox_preempt(void) {}
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { fprintf(stderr, "native %x\n", target); exit(3); }
int main(void) {
    lift_frames_init();
    G = calloc(1, 1 << 20);
    R_ESP = 0xF000;
    fn_00001080();
    if (R_EAX != 7) return 4;
    if (lift_jmp_top != 0) return 5;
    if (lift_cpu.depth != 0) return 6;
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-O1", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("lifted.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        run.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
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
    let [first, second] = program.bodies() else {
        panic!("{:?}", program.bodies());
    };
    assert!(
        first.contains("LIFT_TAIL(fn_00001005, 0x00001000u);"),
        "{first}"
    );
    assert!(!first.contains("L_00001005"), "{first}");
    assert!(
        second
            .contains("LIFT_POLL(0x00001006u); P_00001006:; LIFT_TAIL(fn_00001000, 0x00001006u);"),
        "{second}"
    );
    assert!(!second.contains("L_00001000"), "{second}");
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
    let dir = std::env::temp_dir().join(format!("anybox-lift-tail-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("lifted.c"),
        format!(
            "#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{}{}",
            program.bodies().concat(),
            program.dispatch()
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("main.c"),
        r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
void anyxbox_preempt(void) {}
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { exit(3); }
int main(void) {
    lift_frames_init();
    G = calloc(1, 1 << 20);
    R_ESP = 0xF000;
    R_ECX = 10000000;
    fn_00001000();
    if (R_ECX != 0 || R_EAX != 1) return 4;
    if (R_ESP != 0xF004) return 5;
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-O0", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("lifted.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        run.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
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
    assert!(program.source().contains("switch (R_EBP)"));
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
    assert!(source.contains("case 29u:"), "{source}");
    assert!(!source.contains("case 30u:"));
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
        source.contains("LIFT_IN(0x00001004u, LIFT_K_RET, rt_call(t_target, 4100u)); return;"),
        "{source}"
    );
}

/// `cvtps2pi`, `cvttps2pi` and `cvtpi2ps`, run: 2.5 and -1.5 round to 2 and -2, truncate to 2 and -1,
/// and the integers convert back to floats.
#[test]
fn the_packed_float_and_integer_conversions_run() {
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
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
    let dir = std::env::temp_dir().join(format!("anybox-lift-cvt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("lifted.c"),
        format!(
            "#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{}{}",
            program.bodies().concat(),
            program.dispatch()
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("main.c"),
        r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
void anyxbox_preempt(void) {}
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { exit(3); }
int main(void) {
    lift_frames_init();
    XMM[0].f[0] = 2.5f; XMM[0].f[1] = -1.5f;
    fn_00001000();
    if (MM[0] != (2ull | (0xFFFFFFFEull << 32))) return 4;
    if (MM[1] != (2ull | (0xFFFFFFFFull << 32))) return 5;
    if (XMM[2].f[0] != 2.0f || XMM[2].f[1] != -1.0f) return 6;
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-O1", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("lifted.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).status().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(run.code(), Some(0));
}

/// The 8- and 16-bit `mul`, `imul`, `div` and `idiv`, and the MMX `pextrw`, `pinsrw` and
/// `pmovmskb`, each run on a value checked by hand.
#[test]
fn the_small_integer_forms_and_the_mmx_extensions_run() {
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
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
    let dir = std::env::temp_dir().join(format!("anybox-lift-int-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("lifted.c"),
        format!(
            "#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{}{}",
            program.bodies().concat(),
            program.dispatch()
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("main.c"),
        r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
void anyxbox_preempt(void) {}
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { exit(3); }
#define CHECK(code, what) if (!(what)) return code
int main(void) {
    lift_frames_init();
    R_EAX = 0xAABB00C8u; R_EBX = 3; fn_00001000();                      /* mul bl: 200 * 3 */
    CHECK(10, R_EAX == 0xAABB0258u && F_CF);
    R_EAX = 0x11118000u; R_EDX = 0xFFFF1111u; R_EBX = 4; fn_00001010(); /* mul bx: 0x8000 * 4 */
    CHECK(11, R_EAX == 0x11110000u && R_EDX == 0xFFFF0002u);
    R_EAX = 0x11110257u; R_EBX = 10; fn_00001020();                     /* div bl: 599 / 10 */
    CHECK(12, R_EAX == 0x1111093Bu);
    R_EAX = 0x1111FDA8u; R_EBX = 7; fn_00001030();                      /* idiv bl: -600 / 7 */
    CHECK(13, R_EAX == 0x1111FBABu);
    R_EAX = 0x00000000u; R_EDX = 0x00010000u >> 16; R_EBX = 3; fn_00001040(); /* div bx: 0x10000 / 3 */
    CHECK(14, (R_EAX & 0xFFFF) == 0x5555u && (R_EDX & 0xFFFF) == 1u);
    R_EAX = 0x00007960u; R_EDX = 0x0000FFFEu; R_EBX = 7; fn_00001050();  /* idiv bx: -100000 / 7 */
    CHECK(15, (R_EAX & 0xFFFF) == 0xC833u && (R_EDX & 0xFFFF) == 0xFFFBu);
    R_EAX = 0xAABB00FDu; R_EBX = 5; fn_00001060();                      /* imul bl: -3 * 5 */
    CHECK(16, (R_EAX & 0xFFFF) == 0xFFF1u);
    R_EAX = 0x0000FED4u; R_EBX = 200; fn_00001070();                    /* imul bx: -300 * 200 */
    CHECK(17, (R_EAX & 0xFFFF) == 0x15A0u && (R_EDX & 0xFFFF) == 0xFFFFu);
    MM[0] = 0x1111222233334444ull; fn_00001080();                       /* pextrw eax, mm0, 2 */
    CHECK(18, R_EAX == 0x2222u);
    R_EAX = 0xDEADBEEFu; MM[1] = 0; fn_00001090();                      /* pinsrw mm1, eax, 1 */
    CHECK(19, MM[1] == 0x00000000BEEF0000ull);
    MM[0] = 0x8000FF01807F0080ull; fn_000010a0();                       /* pmovmskb eax, mm0 */
    CHECK(20, R_EAX == 0xA9u);
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-O1", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("lifted.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).status().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(run.code(), Some(0), "failed check {:?}", run.code());
}

/// Without `fninit`, a thread's x87 and SSE start as the console's kernel starts them: control
/// word 0x027F (53-bit precision, every exception masked) and MXCSR 0x1F80; `fninit` gives
/// 0x037F. The precision field still rounds to single only when it says so.
#[test]
fn a_thread_starts_with_the_consoles_floating_point_state() {
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
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
    let dir = std::env::temp_dir().join(format!("anybox-lift-fpu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("lifted.c"),
        format!(
            "#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{}{}",
            program.bodies().concat(),
            program.dispatch()
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("main.c"),
        r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
void anyxbox_preempt(void) {}
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { exit(3); }
static uint32_t word(uint32_t at, int len) { uint32_t v = 0; memcpy(&v, (uint8_t *)G + at, len); return v; }
int main(void) {
    lift_frames_init();
    G = calloc(1, 1 << 20);
    R_ESP = 0xF000;
    fn_00001000();
    if (word(0x2000, 2) != 0x027F) return 4;
    if (word(0x2004, 4) != 0x1F80) return 5;
    if (word(0x2002, 2) != 0x037F) return 6;
    double one = 1.0, three = 3.0, r;
    memcpy((uint8_t *)G + 0x2010, &one, 8);
    memcpy((uint8_t *)G + 0x2018, &three, 8);
    const uint16_t words[3] = {0x007F, 0x027F, 0x037F};
    for (int i = 0; i < 3; i++) {
        memcpy((uint8_t *)G + 0x2008, &words[i], 2);
        fn_00001040();
        memcpy(&r, (uint8_t *)G + 0x2020, 8);
        double expected = i == 0 ? (double)(float)(1.0 / 3.0) : 1.0 / 3.0;
        if (r != expected) return 10 + i;
    }
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-O1", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("lifted.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).status().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(run.code(), Some(0), "failed check {:?}", run.code());
}

/// `fnsave` stores the control and status words and the stack, ST(0) first as 80-bit values,
/// then initializes the unit; `frstor` brings all of it back.
#[test]
fn fnsave_and_frstor_keep_the_whole_x87_state() {
    let Ok(clang) = which("clang") else {
        eprintln!("no clang; skipping");
        return;
    };
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
    let dir = std::env::temp_dir().join(format!("anybox-lift-fsave-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lift.h"), program.header()).unwrap();
    std::fs::write(
        dir.join("lifted.c"),
        format!(
            "#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{}{}",
            program.bodies().concat(),
            program.dispatch()
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("main.c"),
        r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
void anyxbox_preempt(void) {}
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
void lift_resume_innermost(void) {}
void lift_native(uint32_t target, uint32_t at) { exit(3); }
static uint32_t word(uint32_t at, int len) { uint32_t v = 0; memcpy(&v, (uint8_t *)G + at, len); return v; }
static double real(uint32_t at) { double v; memcpy(&v, (uint8_t *)G + at, 8); return v; }
int main(void) {
    lift_frames_init();
    G = calloc(1, 1 << 20);
    R_ESP = 0xF000;
    fn_00001000();
    if (word(0x3000, 2) != 0x027F) return 4;
    if (word(0x3004, 2) != 0x3000) return 5;
    if (word(0x3024, 2) != 0x4000 || word(0x302E, 2) != 0x3FFF) return 6;
    if (word(0x2030, 2) != 0x037F || word(0x2032, 2) != 0x027F) return 7;
    if (real(0x2020) != 3.14159265358979323846 || real(0x2028) != 1.0) return 8;
    return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("test");
    let build = Command::new(clang)
        .args(["-w", "-O1", "-I"])
        .arg(&dir)
        .arg("-o")
        .arg(&exe)
        .arg(dir.join("main.c"))
        .arg(dir.join("lifted.c"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).status().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(run.code(), Some(0), "failed check {:?}", run.code());
}

/// The path of `program` if it is on the `PATH`.
fn which(program: &str) -> Result<std::path::PathBuf, ()> {
    let path = std::env::var_os("PATH").ok_or(())?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
        .ok_or(())
}

/// What a resume test's `main` needs besides its own function: stubs for the runtime, and a
/// freeze and a resume as the runtime does them, inside one process. `freeze` saves the CPU, the
/// frames, the `setjmp` contexts and guest memory, then leaves the thread's host frames;
/// `resume` loads them into fresh state and calls down the frames again.
const RESUME_HARNESS: &str = r#"#include "lift.h"
volatile int anyxbox_preempt_flag;
volatile int64_t anyxbox_tsc_offset;
uint32_t anyxbox_mmio_read(uint32_t a, uint32_t size) { return 0; }
void anyxbox_mmio_write(uint32_t a, uint32_t size, uint32_t v) {}
void lift_trap_hook(void) {}
static lift_cpu_t saved;
static lift_frame saved_frames[64];
static lift_jmp_saved saved_jumps[16];
static uint32_t saved_count, saved_jump_count;
static uint8_t *saved_memory;
static sigjmp_buf frozen;
static int freezing;
static void freeze(uint32_t count) {
    freezing = 0;
    saved = lift_cpu;
    saved_count = count;
    memcpy(saved_frames, lift_cpu.frames, count * sizeof(lift_frame));
    saved_jump_count = lift_jmp_top;
    for (uint32_t i = 0; i < lift_jmp_top; i++) memcpy(&saved_jumps[i], lift_jmp_stack[i], sizeof(lift_jmp_saved));
    memcpy(saved_memory, G, 1 << 20);
    siglongjmp(frozen, 1);
}
static void resume(void) {
    lift_frame *frames = lift_cpu.frames;
    memcpy(G, saved_memory, 1 << 20);
    lift_cpu = saved;
    lift_cpu.frames = frames;
    memset(frames, 0, 64 * sizeof(lift_frame));
    memcpy(frames, saved_frames, saved_count * sizeof(lift_frame));
    memcpy(lift_jmp_restore, saved_jumps, sizeof saved_jumps);
    lift_jmp_restore_count = saved_jump_count;
    lift_jmp_top = 0;
    lift_cpu.resume_depth = saved_count;
    lift_cpu.depth = 0;
    lift_cpu.resuming = 1;
    lift_resume_root();
}
/* The kernel call: returns 100 and pops its return address; while freezing, it parks instead. */
void lift_native(uint32_t target, uint32_t at) {
    if (freezing) freeze(lift_cpu.depth);
    R_EAX = 100;
    R_ESP += 4;
}
/* The polls still to pass before the one that freezes; the flag stays raised until then. */
static uint32_t polls_left;
void anyxbox_preempt(void) {
    if (freezing && --polls_left > 0) return;
    anyxbox_preempt_flag = 0;
    if (freezing) freeze(lift_cpu.depth + 1);
}
/* The innermost frame: a parked kernel call completes now; a poll just goes on. */
void lift_resume_innermost(void) {
    lift_cpu.resuming = 0;
    if (saved_frames[saved_count - 1].kind != LIFT_K_POLL) {
        R_EAX = 100;
        R_ESP += 4;
    }
}
/* Runs ROOT through, then again frozen and resumed: at its first kernel call, or with POLL at
   that poll (1 the first); both runs must end with the same EAX and ESP, and the resumed one with
   no depth left. */
static int both_runs(void (*root)(void), uint32_t poll) {
    lift_frames_init();
    G = calloc(1, 1 << 20);
    saved_memory = calloc(1, 1 << 20);
    R_ESP = 0xF000;
    root();
    uint32_t eax = R_EAX, esp = R_ESP;
    if (lift_cpu.depth != 0) return 2;
    lift_frame *frames = lift_cpu.frames;
    memset(&lift_cpu, 0, sizeof lift_cpu);
    lift_cpu.frames = frames;
    memset(G, 0, 1 << 20);
    R_ESP = 0xF000;
    freezing = 1;
    polls_left = poll;
    anyxbox_preempt_flag = poll != 0;
    if (sigsetjmp(frozen, 0) == 0) {
        root();
        return 3;
    }
    resume();
    if (R_EAX != eax || R_ESP != esp) {
        fprintf(stderr, "eax %x, not %x; esp %x, not %x\n", R_EAX, eax, R_ESP, esp);
        return 4;
    }
    if (lift_cpu.depth != 0 || lift_cpu.resuming || lift_jmp_top != 0) return 5;
    return 0;
}
"#;

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
    let main = format!(
        "{RESUME_HARNESS}int main(void) {{ int r = both_runs(fn_00001000, 0); if (!r && R_EAX != 108) r = 6; return r; }}\n"
    );
    let Some((status, stderr)) = build_and_run("resume-call", &program, &main) else {
        eprintln!("no clang; skipping");
        return;
    };
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
    let main = format!(
        "{RESUME_HARNESS}int main(void) {{ int r = both_runs(fn_00001000, 500); if (!r && (R_EAX != 1001 || rd32(0x2000) != 1)) r = 6; return r; }}\n"
    );
    let Some((status, stderr)) = build_and_run("resume-poll", &program, &main) else {
        eprintln!("no clang; skipping");
        return;
    };
    assert_eq!(status, Some(0), "{stderr}");
}

/// Frozen in a kernel call after a `setjmp`, the thread registers the context again as it
/// resumes, so the `longjmp` after the call still lands.
#[test]
fn a_setjmp_context_survives_a_resume() {
    const NATIVE: u32 = 0x10C0;
    let mut bytes = vec![0x90u8; 0x100];
    let setjmp = [
        0x8B, 0x54, 0x24, 0x04, 0x89, 0x2A, 0x89, 0x5A, 0x04, 0x89, 0x7A, 0x08, 0x89, 0x72, 0x0C,
        0x89, 0x62, 0x10, 0x8B, 0x04, 0x24, 0x89, 0x42, 0x14, 0x31, 0xC0, 0xC3,
    ];
    let longjmp = [
        0x8B, 0x54, 0x24, 0x04, 0x8B, 0x44, 0x24, 0x08, 0x8B, 0x2A, 0x8B, 0x5A, 0x04, 0x8B, 0x7A,
        0x08, 0x8B, 0x72, 0x0C, 0x8B, 0x62, 0x10, 0x83, 0xC4, 0x04, 0xFF, 0x62, 0x14,
    ];
    bytes[..setjmp.len()].copy_from_slice(&setjmp);
    bytes[0x40..0x40 + longjmp.len()].copy_from_slice(&longjmp);
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
    let main = format!(
        "{RESUME_HARNESS}int main(void) {{ int r = both_runs(fn_00001080, 0); if (!r && R_EAX != 7) r = 6; return r; }}\n"
    );
    let Some((status, stderr)) = build_and_run("resume-setjmp", &program, &main) else {
        eprintln!("no clang; skipping");
        return;
    };
    assert_eq!(status, Some(0), "{stderr}");
}
