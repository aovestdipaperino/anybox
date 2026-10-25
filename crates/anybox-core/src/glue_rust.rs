//! The Rust a lifted program is built with, when a build asks for Rust (`Request::rust`): the glue to the runtime, the units, and the
//! compile and link.
//!
//! Each unit is a `no_std` crate that `rustc` compiles to an object on its own, all at once. The
//! objects are linked with the runtime by the system linker, `ld`, against the macOS SDK.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, PoisonError};

use crate::discover::read_default_xbe;
use crate::glue::{FRAMEWORKS, RTL_ENTER_CRITICAL_SECTION, RTL_LEAVE_CRITICAL_SECTION};
use crate::progress::{Phase, Reporter, Tracker, clock, note};

/// The glue the lifted code and the runtime meet through: the kernel-call hook, the guest thread
/// bodies, and `main`.
pub(crate) const GLUE: &str = include_str!("../lifted/glue.rs");

/// `text` as a Rust byte string literal ending in a NUL, for C.
pub(crate) fn c_bytes(text: &str) -> String {
    let mut literal = String::from("b\"");
    for byte in text.bytes() {
        match byte {
            b'"' | b'\\' => {
                literal.push('\\');
                literal.push(char::from(byte));
            }
            0x20..=0x7E => literal.push(char::from(byte)),
            _ => {
                use std::fmt::Write as _;
                let _ = write!(literal, "\\x{byte:02x}");
            }
        }
    }
    literal.push_str("\\0\"");
    literal
}

/// The glue's unit for a title whose entry point is `entry` and whose name is `name`.
pub(crate) fn glue_unit(entry: u32, name: &str) -> String {
    let trap = |ordinal| crate::layout::kernel_function_address(ordinal);
    format!(
        "{}{}{}{}\nconst ENTRY: Lifted = fn_{entry:08x};\n\
         const ENTER_SECTION: u32 = {:#010x};\nconst LEAVE_SECTION: u32 = {:#010x};\n\
         const APP_NAME: &[u8] = {};\n{GLUE}",
        lift::rust::UNIT_ATTRIBUTES,
        lift::rust::PRELUDE,
        lift::rust::GLOBALS,
        lift::rust::declarations([entry]),
        trap(RTL_ENTER_CRITICAL_SECTION),
        trap(RTL_LEAVE_CRITICAL_SECTION),
        c_bytes(name),
    )
}

/// Writes the units to `out`, and returns their paths: the glue (which defines the globals), the
/// dispatcher, and the functions split into balanced parts.
pub(crate) fn write_units(
    out: &Path,
    program: &lift::rust::Program,
    entry: u32,
    name: &str,
) -> Result<Vec<PathBuf>, String> {
    let parts = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .max(2);
    let mut units = vec![
        write_unit(out, "glue.rs", &glue_unit(entry, name))?,
        write_unit(out, "dispatch.rs", &program.dispatch_unit())?,
    ];
    for (index, unit) in program.units(parts).iter().enumerate() {
        units.push(write_unit(out, &format!("part{index}.rs"), unit)?);
    }
    Ok(units)
}

/// Writes `text` to `out/name`, unless the file already holds it, so its object stays current.
fn write_unit(out: &Path, name: &str, text: &str) -> Result<PathBuf, String> {
    let path = out.join(name);
    if std::fs::read(&path).is_ok_and(|have| have == text.as_bytes()) {
        return Ok(path);
    }
    std::fs::write(&path, text).map_err(|error| error.to_string())?;
    Ok(path)
}

/// Whether `object` was built after `unit`, so it need not be built again.
fn object_is_current(object: &Path, unit: &Path) -> bool {
    let modified = |path: &Path| std::fs::metadata(path).and_then(|m| m.modified()).ok();
    match (modified(object), modified(unit)) {
        (Some(object), Some(unit)) => object >= unit,
        _ => false,
    }
}

/// Runs `command`, the tool `tool`, for `what`. Its messages are kept and returned when it fails,
/// so a front end with no terminal shows them too.
fn run_tool(command: &mut Command, tool: &str, what: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("{tool}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let messages = String::from_utf8_lossy(&output.stderr);
    Err(format!("{what} failed\n{}", messages.trim_end()))
}

/// The Rust compiler: `$RUSTC`, else `rustc` on the `PATH`, else the one rustup installs in
/// `$CARGO_HOME/bin` or `~/.cargo/bin` (an app started from the Finder has a short `PATH`).
pub(crate) fn find_rustc() -> Option<PathBuf> {
    if let Some(rustc) = std::env::var_os("RUSTC") {
        return Some(rustc.into());
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".cargo")));
    std::env::split_paths(&path)
        .chain(cargo_home.map(|home| home.join("bin")))
        .map(|dir| dir.join("rustc"))
        .find(|candidate| candidate.is_file())
}

/// [`find_rustc`], or why there is none and how to install it.
fn rustc() -> Result<PathBuf, String> {
    find_rustc().ok_or_else(|| {
        crate::toolchain::check(true)
            .err()
            .unwrap_or_else(|| "no Rust compiler".to_owned())
    })
}

/// The flags every unit is compiled with: a `no_std` library crate, compiled whole to one object
/// for arm64, that never panics and wraps on overflow, as the guest's arithmetic does.
const UNIT_FLAGS: [&str; 10] = [
    "--edition=2024",
    "--crate-type=lib",
    "--emit=obj",
    "--target=aarch64-apple-darwin",
    "-Cpanic=abort",
    "-Coverflow-checks=off",
    "-Cdebug-assertions=off",
    "-Ccodegen-units=1",
    "--cap-lints=allow",
    "-Cstrip=none",
];

/// Compiles each unit to an object, all at once, except those already built. Returns the object
/// paths.
pub(crate) fn compile_in_parallel(
    units: &[PathBuf],
    flags: &[String],
    reporter: &dyn Reporter,
) -> Result<Vec<PathBuf>, String> {
    let rustc = rustc()?;
    // Progress is counted in bytes of Rust, so a large unit counts for more than a small one.
    // Units already built are left out, or they would make the time left look shorter.
    let stale: Vec<Option<u64>> = units
        .iter()
        .map(|unit| {
            (!object_is_current(&unit.with_extension("o"), unit))
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
                let (compiling, rustc) = (&compiling, &rustc);
                scope.spawn(move || {
                    let object = unit.with_extension("o");
                    let Some(size) = stale else {
                        return Ok(object);
                    };
                    let crate_name = unit
                        .file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    run_tool(
                        Command::new(rustc)
                            .args(UNIT_FLAGS)
                            .args(flags)
                            .args(["--crate-name", &crate_name])
                            .arg("-o")
                            .arg(&object)
                            .arg(unit),
                        "rustc",
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
            format!("compiled {} MB of Rust in {}", total >> 20, clock(took)),
        );
    }
    Ok(objects)
}

/// The unit that embeds the file at `xbe`, an absolute path, as `anyxbox_xbe`, with
/// `anyxbox_xbe_end` just past it. Done in assembly because an array of megabytes compiles
/// slowly.
pub(crate) fn embed_source(xbe: &Path) -> String {
    // The path is quoted for the assembler, and the braces doubled for `global_asm!`'s template.
    let path = xbe
        .display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('{', "{{")
        .replace('}', "}}");
    let incbin = format!(".incbin \"{path}\"");
    format!(
        "{}core::arch::global_asm!(\n    \".section __DATA,__const\",\n    \".globl _anyxbox_xbe\",\n    \
         \"_anyxbox_xbe:\",\n    {incbin:?},\n    \".globl _anyxbox_xbe_end\",\n    \"_anyxbox_xbe_end:\",\n    \
         \".text\",\n);\n",
        lift::rust::UNIT_ATTRIBUTES
    )
}

/// Writes the title's executable to `out/boot.xbe` and the unit that embeds it to `out/boot.rs`,
/// and returns that unit.
pub(crate) fn write_boot_image(iso: &Path, out: &Path) -> Result<PathBuf, String> {
    let xbe = out.join("boot.xbe");
    std::fs::write(&xbe, read_default_xbe(iso)?).map_err(|error| error.to_string())?;
    let absolute = xbe.canonicalize().map_err(|error| error.to_string())?;
    write_unit(out, "boot.rs", &embed_source(&absolute))
}

/// Writes the unit that holds `digest`, the lifted code's (see `crate::code_digest`), to `out`.
pub(crate) fn write_digest_unit(out: &Path, digest: &[u8; 20]) -> Result<PathBuf, String> {
    let bytes: Vec<String> = digest.iter().map(|byte| format!("{byte:#04x}")).collect();
    write_unit(
        out,
        "digest.rs",
        &format!(
            "{}#[unsafe(no_mangle)]\npub static anyxbox_code_digest: [u8; 20] = [{}];\n",
            lift::rust::UNIT_ATTRIBUTES,
            bytes.join(", ")
        ),
    )
}

/// Removes the objects in `work` if they were built with other compiler flags than `flags`, and
/// records `flags` for the next build.
fn forget_objects_built_differently(work: &Path, flags: &[String]) {
    let stamp = work.join("objects.flags");
    let now = UNIT_FLAGS
        .iter()
        .copied()
        .chain(flags.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ");
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

/// The oldest macOS the title runs on: `$MACOSX_DEPLOYMENT_TARGET`, as `rustc` reads it, else
/// `rustc`'s own default for arm64.
fn deployment_target() -> String {
    std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_owned())
}

/// What `xcrun` says for the macOS SDK: its path, or its version with `--show-sdk-version`.
fn sdk(question: &str) -> Result<String, String> {
    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", question])
        .output()
        .map_err(|error| format!("xcrun: {error} (are the Xcode command line tools installed?)"))?;
    let answer = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || answer.is_empty() {
        return Err(format!(
            "xcrun {question} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim_end()
        ));
    }
    Ok(answer)
}

/// Compiles `units` for arm64 and links them with the runtime `library` into `work/Game`.
/// `optimization` is the optimization level, `0` to `3`.
pub(crate) fn compile_and_link(
    units: &[PathBuf],
    work: &Path,
    library: &Path,
    optimization: &str,
    tracing: bool,
    reporter: &dyn Reporter,
) -> Result<PathBuf, String> {
    let level = if tracing { "0" } else { optimization };
    let mut flags = vec![format!("-Copt-level={level}")];
    if tracing {
        flags.extend(["--cfg".to_owned(), "lift_trace".to_owned()]);
    }
    forget_objects_built_differently(work, &flags);
    let objects = compile_in_parallel(units, &flags, reporter)?;
    let game = work.join("Game");
    let linking = Tracker::start(Phase::Linking, reporter);
    let mut link = Command::new("ld");
    link.args(["-arch", "arm64", "-platform_version", "macos"])
        .arg(deployment_target())
        .arg(sdk("--show-sdk-version")?)
        .arg("-syslibroot")
        .arg(sdk("--show-sdk-path")?)
        .arg("-o")
        .arg(&game)
        .args(&objects)
        .arg(library)
        .args(["-lc++", "-liconv", "-lobjc", "-lSystem"]);
    for framework in FRAMEWORKS {
        link.args(["-framework", framework]);
    }
    run_tool(&mut link, "ld", "linking the title")?;
    let took = linking.finish();
    note(reporter, format!("linked in {}", clock(took)));
    Ok(game)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// The glue and the units of a one-function program compile with the prelude, each to an
    /// object of its own, as a build compiles them.
    #[test]
    fn the_glue_and_the_units_compile() {
        let image = lift::Image::single(0x1000, vec![0xC3]);
        let program = lift::rust::lift_program(&image, &[0x1000], &BTreeMap::new(), false);
        let dir = std::env::temp_dir().join(format!("anybox-rust-glue-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut units = write_units(&dir, &program, 0x1000, "Test \"Title\"").unwrap();
        let xbe = dir.join("boot.xbe");
        std::fs::write(&xbe, [1u8, 2, 3]).unwrap();
        units.push(write_unit(&dir, "boot.rs", &embed_source(&xbe)).unwrap());
        units.push(write_digest_unit(&dir, &[7; 20]).unwrap());
        let result = compile_in_parallel(&units, &["-Copt-level=0".to_owned()], &|_| {});
        std::fs::remove_dir_all(&dir).unwrap();
        result.unwrap();
    }

    /// A one-function program links with the runtime into a program, by `ld` alone, when
    /// `$ANYBOX_RUNTIME` names a runtime directory.
    #[test]
    fn the_units_link_with_the_runtime() {
        let Some(runtime) = std::env::var_os("ANYBOX_RUNTIME").map(PathBuf::from) else {
            eprintln!("no $ANYBOX_RUNTIME; skipping");
            return;
        };
        let image = lift::Image::single(0x1000, vec![0xC3]);
        let program = lift::rust::lift_program(&image, &[0x1000], &BTreeMap::new(), false);
        let dir = std::env::temp_dir().join(format!("anybox-rust-link-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut units = write_units(&dir, &program, 0x1000, "Test").unwrap();
        let xbe = dir.join("boot.xbe");
        std::fs::write(&xbe, [1u8, 2, 3]).unwrap();
        units.push(write_unit(&dir, "boot.rs", &embed_source(&xbe)).unwrap());
        units.push(write_digest_unit(&dir, &[7; 20]).unwrap());
        let result = compile_and_link(
            &units,
            &dir,
            &runtime.join("libruntime.a"),
            "1",
            false,
            &|_| {},
        );
        let game = dir.join("Game");
        let linked = game.is_file();
        std::fs::remove_dir_all(&dir).unwrap();
        result.unwrap();
        assert!(linked);
    }

    #[test]
    fn a_name_becomes_a_byte_string_for_c() {
        assert_eq!(c_bytes("Dragon's Lair 3D"), r#"b"Dragon's Lair 3D\0""#);
        assert_eq!(c_bytes(r#"a"b\c"#), r#"b"a\"b\\c\0""#);
        assert_eq!(c_bytes("é"), r#"b"\xc3\xa9\0""#);
    }

    #[test]
    fn the_embedded_file_is_named_in_assembly_inside_a_rust_string() {
        let source = embed_source(Path::new("/tmp/a b/boot.xbe"));
        assert!(source.contains("core::arch::global_asm!("), "{source}");
        assert!(
            source.contains(r#"".incbin \"/tmp/a b/boot.xbe\"""#),
            "{source}"
        );
        assert!(source.contains("_anyxbox_xbe_end"), "{source}");
        // A quote in the path is escaped for the assembler, then again for Rust; a brace is
        // doubled for the template.
        let source = embed_source(Path::new("/tmp/q\"x{1}/boot.xbe"));
        assert!(source.contains(r#"q\\\"x{{1}}"#), "{source}");
    }
}
