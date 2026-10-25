//! The C the lifted program is built with: the glue to the runtime, the translation units, and
//! the compile and link.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, PoisonError};

use crate::discover::read_default_xbe;
use crate::progress::{Phase, Reporter, Tracker, clock, note};

/// The glue the lifted code and the runtime meet through: the kernel-call hook, the guest thread
/// bodies, and `main`.
pub(crate) const GLUE: &str = r#"
#include <pthread.h>

typedef struct { uint32_t state, eax, esp, edx; } anyxbox_result;
uint8_t *anyxbox_boot(const uint8_t *xbe, size_t xbe_len, uint32_t *start);
extern const uint8_t anyxbox_xbe[], anyxbox_xbe_end[]; /* the title's executable, from boot.c */
void anyxbox_native_call(uint32_t target, uint32_t esp, uint32_t ecx, uint32_t edx,
                         anyxbox_result *out);
void anyxbox_thread_end(void);
void anyxbox_thread_end_with(uint32_t result);
void anyxbox_main_loop(void);
uint8_t *anyxbox_guest_base(void); /* the guest base the runtime set */
extern const uint8_t anyxbox_code_digest[20]; /* the lifted code's digest, from digest.c */
void anyxbox_set_code_digest(const uint8_t *digest);

/* Where a host thread's guest CPU is, so a hibernation can save it and a restore fill it. */
typedef struct {
    void *cpu; uint32_t cpu_size; uint32_t *depth; lift_frame *frames;
    lift_jmp **jumps; uint32_t *jump_top; lift_jmp_saved *restore; uint32_t *restore_count;
} anyxbox_cpu_layout;
void anyxbox_register_cpu(const anyxbox_cpu_layout *layout);
uint32_t anyxbox_resume_load(void);
typedef struct { uint32_t kind, eax, esp, edx, target; } anyxbox_resume;
void anyxbox_resume_wait(anyxbox_resume *out);

/* Each host thread's frame records, given back when the thread ends: a title that makes a guest
   thread per DPC would otherwise leave 1.5 MiB behind for each. The key's destructor runs after
   the thread's last guest code, on `pthread_exit` and on return alike, and touches only the
   records it is handed. */
static pthread_key_t frames_key;
static pthread_once_t frames_once = PTHREAD_ONCE_INIT;
static void frames_release(void *frames) { lift_frames_free((lift_frame *)frames); }
static void frames_key_create(void) { pthread_key_create(&frames_key, frames_release); }

/* Every host thread that runs a guest thread starts here: its frame records, and the runtime
   told where its guest CPU is. */
static void guest_thread_start(void) {
    lift_frames_init();
    pthread_once(&frames_once, frames_key_create);
    pthread_setspecific(frames_key, lift_cpu.frames);
    anyxbox_cpu_layout layout = {
        &lift_cpu, sizeof lift_cpu, &lift_cpu.depth, lift_cpu.frames,
        lift_jmp_stack, &lift_jmp_top, lift_jmp_restore, &lift_jmp_restore_count,
    };
    anyxbox_register_cpu(&layout);
}

/* The uncontended cases of RtlEnterCriticalSection and RtlLeaveCriticalSection, which titles call
   millions of times, done here as the emulator's guest stubs do them instead of in the kernel.
   `LockCount` is at +0x10, `RecursionCount` at +0x14, `OwningThread` at +0x18. Returns whether
   the call is complete. */
static int fast_section(uint32_t target) {
    uint32_t section = rd32(R_ESP + 4);
    if (target == ENTER_SECTION) {
        uint32_t me = rd32(FS_BASE + 0x28), owner = rd32(section + 0x18);
        if (owner == 0) {
            wr32(section + 0x18, me);
            wr32(section + 0x14, 1);
            wr32(section + 0x10, 0);
        } else if (owner == me) {
            wr32(section + 0x14, rd32(section + 0x14) + 1);
            wr32(section + 0x10, rd32(section + 0x10) + 1);
        } else {
            return 0;
        }
    } else {
        uint32_t recursion = rd32(section + 0x14);
        if (recursion > 1) {
            wr32(section + 0x14, recursion - 1);
            wr32(section + 0x10, rd32(section + 0x10) - 1);
        } else if (rd32(section + 0x10) == 0) {
            wr32(section + 0x18, 0);
            wr32(section + 0x14, 0);
            wr32(section + 0x10, 0xFFFFFFFFu);
        } else {
            return 0;
        }
    }
    R_EAX = 0;
    R_ESP += 8; /* the return address and the argument: stdcall */
    return 1;
}

/* The return addresses along the guest's frame-pointer chain: where the title was when the
   runtime could not go on. Functions built without frame pointers end the walk early. */
static void guest_backtrace(void) {
    uint32_t ebp = R_EBP;
    fprintf(stderr, "guest registers: eax=%#x ecx=%#x edx=%#x ebx=%#x esp=%#x ebp=%#x esi=%#x edi=%#x\n",
            R_EAX, R_ECX, R_EDX, R_EBX, R_ESP, R_EBP, R_ESI, R_EDI);
    fprintf(stderr, "guest backtrace:");
    /* Only frames within a megabyte above the stack pointer: EBP is a general register too. */
    for (int i = 0; i < 24 && ebp >= R_ESP && ebp - R_ESP < 0x100000u; i++) {
        fprintf(stderr, " %#x", rd32(ebp + 4));
        uint32_t next = rd32(ebp);
        if (next <= ebp) break;
        ebp = next;
    }
    fprintf(stderr, "\n");
}

/* The runtime prints the guest memory `ANYXBOX_DUMP` names. */
void anyxbox_dump_requested(void);
/* The runtime deletes a snapshot that could not be resumed. */
void anyxbox_resume_failed(void);

/* What lift_trap prints besides its message: the registers, the frame-pointer chain, and the
   memory the user asked for. A trap while resuming down the frames means the snapshot cannot
   resume, so it goes, and the next launch boots afresh. */
void lift_trap_hook(void) {
    if (lift_cpu.resuming) anyxbox_resume_failed();
    guest_backtrace();
    anyxbox_dump_requested();
}

/* A kernel call or a missing function: the runtime completes the call. */
void lift_native(uint32_t target, uint32_t at) {
    if ((target == ENTER_SECTION || target == LEAVE_SECTION) && fast_section(target)) {
        return;
    }
    anyxbox_result r = {0, 0, 0, 0};
    anyxbox_native_call(target, R_ESP, R_ECX, R_EDX, &r);
    if (r.state == 0) {
        lift_trap(at, "a kernel call the native runtime cannot complete");
    }
    if (r.state == 3) {
        /* The calling guest thread finished: it runs no more guest code. */
        anyxbox_thread_end();
        pthread_exit(NULL);
    }
    R_ESP = r.esp;
    R_EAX = r.eax;
    R_EDX = r.edx; /* the high half of a 64-bit result, or EDX unchanged */
}

/* The innermost frame of a resumed thread: once its turn comes, it finishes the contact it was
   parked in, a kernel call's result, the kernel call itself, or a poll, and runs on. */
void lift_resume_innermost(void) {
    anyxbox_resume r;
    anyxbox_resume_wait(&r);
    lift_cpu.resuming = 0;
    if (r.kind == 0) {
        R_ESP = r.esp;
        R_EAX = r.eax;
        R_EDX = r.edx;
    } else if (r.kind == 1) {
        lift_native(r.target, 0);
    }
}

/* A guest thread resumed from a snapshot, on its own host thread: its CPU and frames come from
   the snapshot, and it calls down its frames again. It ends as the thread it was would have. */
void anyxbox_resume_body(uint32_t tid, uint32_t fs_base) {
    /* `main` sets G and FS_BASE only when it boots the title itself, and only once
       `anyxbox_boot` returns, but resumed threads start inside it: each takes the guest base and
       the KPCR from the runtime, which set them before starting them. */
    G = anyxbox_guest_base();
    FS_BASE = fs_base;
    guest_thread_start();
    lift_frame *frames = lift_cpu.frames;
    uint32_t count = anyxbox_resume_load();
    lift_cpu.frames = frames;
    lift_cpu.resume_depth = count;
    lift_cpu.depth = 0;
    lift_cpu.resuming = 1;
    lift_resume_root();
    if (tid == 1) anyxbox_thread_end(); else anyxbox_thread_end_with(R_EAX);
    pthread_exit(NULL);
}

/* The title's entry point after a reset, on its own host thread: the guest memory is new. */
void anyxbox_entry_body(uint8_t *base, uint32_t esp, uint32_t fs_base) {
    guest_thread_start();
    G = base;
    R_ESP = esp;
    R_EAX = 0;
    FS_BASE = fs_base;
    push32(0x80010000u); /* the entry return trap */
    fn_ENTRY();
    anyxbox_thread_end();
    pthread_exit(NULL);
}

/* The routine of a guest thread created by the kernel, on its own host thread. */
void anyxbox_thread_body(uint32_t start, uint32_t esp) {
    guest_thread_start();
    R_ESP = esp;
    R_EAX = 0;
    rt_call(start, 0x80010FFCu); /* THREAD_RETURN: the routine returns here */
    anyxbox_thread_end_with(R_EAX); /* its result: a completion routine's verdict */
    pthread_exit(NULL);
}

int main(void) {
    setenv("ANYXBOX_NAME", APP_NAME, 0); /* the window's title */
    uint32_t start[4] = {0, 0, 0, 0};
    /* Snapshots belong to this lifted code, whatever runtime it is linked with. */
    anyxbox_set_code_digest(anyxbox_code_digest);
    G = anyxbox_boot(anyxbox_xbe, (size_t)(anyxbox_xbe_end - anyxbox_xbe), start);
    if (!G) return 1;
    if (start[3]) {
        /* Resumed from a snapshot: the guest threads run on host threads of their own. */
        anyxbox_main_loop();
        return 0;
    }
    guest_thread_start();
    R_ESP = start[0];
    FS_BASE = start[2];
    push32(0x80010000u); /* the entry return trap: where the entry point returns to */
    fn_ENTRY();
    anyxbox_thread_end();
    /* The boot thread is done; the main thread runs the window, which macOS requires. */
    anyxbox_main_loop();
    return 0;
}
"#;

/// Ordinal of `RtlEnterCriticalSection`.
pub(crate) const RTL_ENTER_CRITICAL_SECTION: u32 = 277;
/// Ordinal of `RtlLeaveCriticalSection`.
pub(crate) const RTL_LEAVE_CRITICAL_SECTION: u32 = 294;

/// `text` as a C string literal.
pub(crate) fn c_string(text: &str) -> String {
    let mut literal = String::from('"');
    for byte in text.bytes() {
        match byte {
            b'"' | b'\\' => {
                literal.push('\\');
                literal.push(char::from(byte));
            }
            0x20..=0x7E => literal.push(char::from(byte)),
            _ => {
                use std::fmt::Write as _;
                let _ = write!(literal, "\\{byte:03o}");
            }
        }
    }
    literal.push('"');
    literal
}

/// Writes the header and the translation units to `out`, and returns the units' paths: the glue
/// (which defines the globals), the dispatcher, and the functions split into balanced parts.
pub(crate) fn write_units(
    out: &Path,
    program: &lift::Program,
    entry: u32,
    name: &str,
) -> Result<Vec<PathBuf>, String> {
    let write = |name: &str, text: String| -> Result<PathBuf, String> {
        let path = out.join(name);
        // A file that already holds this text is left alone, so its object file stays current.
        if std::fs::read(&path).is_ok_and(|have| have == text.as_bytes()) {
            return Ok(path);
        }
        std::fs::write(&path, text).map_err(|error| error.to_string())?;
        Ok(path)
    };
    write("lift.h", program.header())?;
    let trap = |ordinal| format!("{:#010x}u", crate::layout::kernel_function_address(ordinal));
    let glue = GLUE
        .replace("fn_ENTRY", &format!("fn_{entry:08x}"))
        .replace("ENTER_SECTION", &trap(RTL_ENTER_CRITICAL_SECTION))
        .replace("LEAVE_SECTION", &trap(RTL_LEAVE_CRITICAL_SECTION))
        .replace("APP_NAME", &c_string(name));
    let mut units = vec![
        write(
            "glue.c",
            format!("#define LIFT_DEFINE_GLOBALS\n#include \"lift.h\"\n{glue}"),
        )?,
        write(
            "dispatch.c",
            format!("#include \"lift.h\"\n{}", program.dispatch()),
        )?,
    ];
    let parts = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .max(2);
    let total: usize = program.bodies().iter().map(String::len).sum();
    let target = total / parts + 1;
    let mut current = String::from("#include \"lift.h\"\n");
    let mut index = 0;
    for body in program.bodies() {
        current.push_str(body);
        if current.len() >= target {
            units.push(write(
                &format!("part{index}.c"),
                std::mem::take(&mut current),
            )?);
            index += 1;
            current = String::from("#include \"lift.h\"\n");
        }
    }
    if current.len() > "#include \"lift.h\"\n".len() {
        units.push(write(&format!("part{index}.c"), current)?);
    }
    Ok(units)
}

/// Whether `object` was built after `unit` and the header last changed, so it need not be built
/// again. A header that is missing counts as old.
fn object_is_current(object: &Path, unit: &Path, header: &Path) -> bool {
    let modified = |path: &Path| std::fs::metadata(path).and_then(|m| m.modified()).ok();
    match (modified(object), modified(unit), modified(header)) {
        (Some(object), Some(unit), Some(header)) => object >= unit && object >= header,
        _ => false,
    }
}

/// Runs `clang`, for `what`. Its messages are kept and returned when it fails, so a front end with
/// no terminal shows them too.
fn run_clang(command: &mut Command, what: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("clang: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let messages = String::from_utf8_lossy(&output.stderr);
    Err(format!("{what} failed\n{}", messages.trim_end()))
}

/// Compiles each C file to an object, all at once, except those already built. Returns the object
/// paths.
pub(crate) fn compile_in_parallel(
    units: &[PathBuf],
    flags: &[&str],
    reporter: &dyn Reporter,
) -> Result<Vec<PathBuf>, String> {
    let include = |unit: &Path| unit.parent().unwrap_or_else(|| Path::new(".")).to_owned();
    // Progress is counted in bytes of C, so a large unit counts for more than a small one. Units
    // already built are left out, or they would make the time left look shorter.
    let stale: Vec<Option<u64>> = units
        .iter()
        .map(|unit| {
            let object = unit.with_extension("o");
            (!object_is_current(&object, unit, &include(unit).join("lift.h")))
                .then(|| std::fs::metadata(unit).map_or(0, |metadata| metadata.len()))
        })
        .collect();
    let total: u64 = stale.iter().flatten().sum();
    let mut started = Tracker::start(Phase::Compiling, reporter);
    started.update(0, total);
    let compiling = Mutex::new((started, 0u64));
    let results: Vec<Result<PathBuf, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = units
            .iter()
            .zip(&stale)
            .map(|(unit, &stale)| {
                let compiling = &compiling;
                scope.spawn(move || {
                    let object = unit.with_extension("o");
                    let Some(size) = stale else {
                        return Ok(object);
                    };
                    run_clang(
                        Command::new("clang")
                            .args(flags)
                            .arg("-I")
                            .arg(include(unit))
                            .arg("-o")
                            .arg(&object)
                            .arg(unit),
                        &format!("compiling {}", unit.display()),
                    )?;
                    let mut compiling = compiling.lock().unwrap_or_else(PoisonError::into_inner);
                    compiling.1 += size;
                    let done = compiling.1;
                    compiling.0.update(done, total);
                    Ok(object)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("a compile thread panicked".to_owned()))
            })
            .collect()
    });
    let (compiling, _) = compiling
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    let took = compiling.finish();
    let objects = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    if total > 0 {
        note(
            reporter,
            format!("compiled {} MB of C in {}", total >> 20, clock(took)),
        );
    }
    Ok(objects)
}

/// The C unit that embeds the file at `xbe`, an absolute path, as `anyxbox_xbe`, with
/// `anyxbox_xbe_end` just past it. Done in assembly because a C array of megabytes compiles slowly.
pub(crate) fn embed_source(xbe: &Path) -> String {
    let path = xbe
        .display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    let assembly = format!(
        ".section __DATA,__const\n.globl _anyxbox_xbe\n_anyxbox_xbe:\n.incbin \"{path}\"\n\
         .globl _anyxbox_xbe_end\n_anyxbox_xbe_end:\n.text\n"
    );
    format!("__asm__({});\n", c_string(&assembly))
}

/// Writes the title's executable to `out/boot.xbe` and the unit that embeds it to `out/boot.c`,
/// and returns that unit.
pub(crate) fn write_boot_image(iso: &Path, out: &Path) -> Result<PathBuf, String> {
    let xbe = out.join("boot.xbe");
    std::fs::write(&xbe, read_default_xbe(iso)?).map_err(|error| error.to_string())?;
    let absolute = xbe.canonicalize().map_err(|error| error.to_string())?;
    let unit = out.join("boot.c");
    std::fs::write(&unit, embed_source(&absolute)).map_err(|error| error.to_string())?;
    Ok(unit)
}

/// Writes the unit that holds `digest`, the lifted code's (see `crate::code_digest`), to `out`.
pub(crate) fn write_digest_unit(out: &Path, digest: &[u8; 20]) -> Result<PathBuf, String> {
    let bytes: Vec<String> = digest.iter().map(|byte| format!("{byte:#04x}")).collect();
    let unit = out.join("digest.c");
    let text = format!(
        "#include <stdint.h>\nconst uint8_t anyxbox_code_digest[20] = {{{}}};\n",
        bytes.join(", ")
    );
    if !std::fs::read(&unit).is_ok_and(|have| have == text.as_bytes()) {
        std::fs::write(&unit, text).map_err(|error| error.to_string())?;
    }
    Ok(unit)
}

/// Removes the objects in `work` if they were built with other compiler flags than `flags`, and
/// records `flags` for the next build.
fn forget_objects_built_differently(work: &Path, flags: &[&str]) {
    let stamp = work.join("objects.flags");
    let now = flags.join(" ");
    if std::fs::read_to_string(&stamp).is_ok_and(|have| have == now) {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(work) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "o") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    let _ = std::fs::write(&stamp, now);
}

/// The frameworks the runtime links against.
pub(crate) const FRAMEWORKS: [&str; 13] = [
    "Security",
    "Metal",
    "QuartzCore",
    "Foundation",
    "CoreGraphics",
    "IOKit",
    "AppKit",
    "CoreFoundation",
    "MetalKit",
    "Carbon",
    "CoreAudio",
    "AudioToolbox",
    "AudioUnit",
];

/// Compiles `units` for arm64 and links them with the runtime `library` into `work/Game`.
pub(crate) fn compile_and_link(
    units: &[PathBuf],
    work: &Path,
    library: &Path,
    optimization: &str,
    tracing: bool,
    reporter: &dyn Reporter,
) -> Result<PathBuf, String> {
    let mut flags = vec![
        "-arch",
        "arm64",
        if tracing { "-O0" } else { optimization },
        "-w",
        "-c",
    ];
    if tracing {
        flags.push("-DLIFT_TRACE_ON");
    }
    forget_objects_built_differently(work, &flags);
    let objects = compile_in_parallel(units, &flags, reporter)?;
    let game = work.join("Game");
    let mut link = Command::new("clang");
    link.args(["-arch", "arm64", "-o"])
        .arg(&game)
        .args(&objects)
        .arg(library)
        .args(["-lc++", "-liconv", "-lobjc"]);
    for framework in FRAMEWORKS {
        link.args(["-framework", framework]);
    }
    let linking = Tracker::start(Phase::Linking, reporter);
    run_clang(&mut link, "linking the title")?;
    let took = linking.finish();
    note(reporter, format!("linked in {}", clock(took)));
    Ok(game)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// The glue and the units of a one-function program compile with the lifted header. The glue
    /// is held to the pointer and integer conversions, so a layout the runtime reads that drifts
    /// from the header's types fails here.
    #[test]
    fn the_glue_compiles_with_the_lifted_header() {
        if Command::new("clang").arg("--version").output().is_err() {
            eprintln!("no clang; skipping");
            return;
        }
        let image = lift::Image::single(0x1000, vec![0xC3]);
        let program = lift::lift_program(&image, &[0x1000], &BTreeMap::new(), false);
        let dir = std::env::temp_dir().join(format!("anybox-glue-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let units = write_units(&dir, &program, 0x1000, "Test").unwrap();
        for unit in &units {
            let strict: &[&str] = if unit.ends_with("glue.c") {
                &[
                    "-Werror=incompatible-pointer-types",
                    "-Werror=int-conversion",
                ]
            } else {
                &["-w"]
            };
            let output = Command::new("clang")
                .arg("-fsyntax-only")
                .args(strict)
                .arg("-I")
                .arg(&dir)
                .arg(unit)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}: {}",
                unit.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_name_becomes_a_c_string() {
        assert_eq!(c_string("Dragon's Lair 3D"), r#""Dragon's Lair 3D""#);
        assert_eq!(c_string(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(c_string("é"), r#""\303\251""#);
    }

    #[test]
    fn the_embedded_file_is_named_in_assembly_inside_a_c_string() {
        let source = embed_source(Path::new("/tmp/a b/boot.xbe"));
        assert!(source.starts_with("__asm__(\""), "{source}");
        assert!(
            source.contains(r#".incbin \"/tmp/a b/boot.xbe\""#),
            "{source}"
        );
        assert!(source.contains("_anyxbox_xbe_end"), "{source}");
        // A quote in the path is escaped for the assembler, then again for C.
        let source = embed_source(Path::new("/tmp/q\"x/boot.xbe"));
        assert!(source.contains(r#"q\\\"x"#), "{source}");
    }
}
