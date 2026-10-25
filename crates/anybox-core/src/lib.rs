//! Turns an original Xbox disc image into a macOS app inside a disk image.
//!
//! The title's executable is lifted to C and compiled for arm64 with clang, or, when the request
//! asks for it ([`Request::rust`]), lifted to Rust and compiled with `rustc` and the system linker,
//! with no C compiler. Either way it is linked with a prebuilt runtime (`libruntime.a` and the
//! signature list `signatures.txt`, found in the runtime directory). The disc's other files are
//! written to a compressed pack the app carries in its resources, so the disc image is needed
//! here, not to run the title.
//!
//! The runtime is not part of this program. It is a separate library that implements the
//! console's kernel, graphics and audio against the symbols the glue declares (`glue.rs` for C,
//! `lifted/glue.rs` for Rust).
//!
//! This crate does the work; the command line (`anybox-cli`) and the window (`anybox-gui`) are
//! front ends over [`build`], which tells them how it goes through a [`progress::Reporter`].

mod bundle;
mod discover;
mod discpack;
mod glue;
mod glue_rust;
mod hooks;
mod icon;
mod layout;
pub mod progress;
pub mod toolchain;

use std::path::{Path, PathBuf};

use progress::{Phase, Reporter, Tracker, clock, note};

/// The zstd level the pack is compressed at unless a build asks for another.
pub const DEFAULT_PACK_LEVEL: i32 = pack::DEFAULT_LEVEL;

/// The optimization levels a build compiles the lifted code at; the first is the default.
pub const OPT_LEVELS: [&str; 3] = ["1", "0", "2"];

/// What to build, and how.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Request {
    /// The disc image (XDVDFS `.iso`).
    pub iso: PathBuf,
    /// An image file or an `http(s)` link for the app's icon; empty for the icon an earlier
    /// build left in the work directory, else the title's own image.
    pub icon: String,
    /// The title's name, shown by macOS.
    pub name: String,
    /// The disk image to write.
    pub dmg: PathBuf,
    /// The directory with `libruntime.a` and `signatures.txt`.
    pub runtime: PathBuf,
    /// Where to keep the build files (the pack is reused while the disc is the same); `None`
    /// for a temporary directory, removed at the end.
    pub work: Option<PathBuf>,
    /// The zstd level of the pack; `None` for [`DEFAULT_PACK_LEVEL`].
    pub pack_level: Option<i32>,
    /// The optimization level of the lifted code, one of [`OPT_LEVELS`]; `None` for the first.
    pub opt: Option<String>,
    /// Lift the title to Rust and build it with `rustc`, instead of to C for clang.
    pub rust: bool,
}

/// The runtime directory: `option`, else `$ANYBOX_RUNTIME`, else `runtime` beside the program,
/// else the app bundle's `Resources/runtime` when the program is in `Contents/MacOS`. The
/// program's path is resolved first, so a link to it on the PATH finds the same runtime.
#[must_use]
pub fn find_runtime(option: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = option {
        return Some(dir.to_owned());
    }
    if let Some(dir) = std::env::var_os("ANYBOX_RUNTIME") {
        return Some(dir.into());
    }
    let exe = std::env::current_exe().ok()?;
    let exe = exe.canonicalize().unwrap_or(exe);
    runtime_beside(exe.parent()?)
}

/// The runtime a program in `dir` uses: `dir/runtime`, else `dir/../Resources/runtime`.
fn runtime_beside(dir: &Path) -> Option<PathBuf> {
    [dir.join("runtime"), dir.join("../Resources/runtime")]
        .into_iter()
        .find(|dir| dir.is_dir())
}

/// The runtime's library and signature list in `runtime`, once both are there.
fn runtime_files(runtime: &Path) -> Result<(PathBuf, PathBuf), String> {
    let library = runtime.join("libruntime.a");
    let signatures = runtime.join("signatures.txt");
    for file in [&library, &signatures] {
        if !file.is_file() {
            return Err(format!("{} is missing", file.display()));
        }
    }
    Ok((library, signatures))
}

/// Builds what `request` asks for, telling `reporter` how it goes.
///
/// # Errors
///
/// Returns a message for the user when an input is missing or a step fails.
pub fn build(request: &Request, reporter: &dyn Reporter) -> Result<(), String> {
    let (library, signatures) = runtime_files(&request.runtime)?;
    if let Some(opt) = &request.opt
        && !OPT_LEVELS.contains(&opt.as_str())
    {
        return Err(format!("{opt:?} is not 0, 1 or 2"));
    }
    toolchain::check(request.rust)?;
    let temporary = request.work.is_none();
    let work = request
        .work
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("anybox-{}", std::process::id())));
    std::fs::create_dir_all(&work).map_err(|error| format!("{}: {error}", work.display()))?;
    let built = build_in(request, &work, &library, &signatures, reporter);
    if temporary {
        let _ = std::fs::remove_dir_all(&work);
    }
    built
}

/// The hooks of `xbe`, from the signature list in `signatures`. `ANYBOX_STUB` names the library
/// sections to stub function by function, `DSOUND` by default.
fn plan_hooks(xbe: &xbe::Xbe, signatures: &Path) -> Result<hooks::Plan, String> {
    let text = std::fs::read_to_string(signatures)
        .map_err(|error| format!("{}: {error}", signatures.display()))?;
    let entries: &'static [xdk::Entry] = Box::leak(
        xdk::parse_owned(text)
            .map_err(|error| error.to_string())?
            .into_boxed_slice(),
    );
    let stubbed = std::env::var("ANYBOX_STUB").unwrap_or_else(|_| "DSOUND".to_owned());
    let stubbed: Vec<&str> = stubbed.split(',').filter(|name| !name.is_empty()).collect();
    hooks::plan(&hooks::sections(xbe), entries, &stubbed)
}

/// The library functions the runtime replaces in the title on `iso`: `function trap name`, one a
/// line, with a stubbed library entry point named `(stub)`; then the signatures that matched more
/// than one function and so were not used.
///
/// # Errors
///
/// Returns a message when the disc, its executable or the signature list cannot be read.
pub fn plan(iso: &Path, runtime: &Path) -> Result<(Vec<String>, Vec<String>), String> {
    let xbe =
        xbe::Xbe::parse(discover::read_default_xbe(iso)?).map_err(|error| format!("{error:?}"))?;
    let plan = plan_hooks(&xbe, &runtime.join("signatures.txt"))?;
    let hooks = plan
        .hooks
        .iter()
        .zip(&plan.names)
        .map(|((function, trap), name)| {
            format!("{function:#010x} {trap:#010x} {}", name.unwrap_or("(stub)"))
        })
        .collect();
    let ambiguous = plan
        .ambiguous
        .iter()
        .map(|(name, matches)| format!("{name} is ambiguous ({matches} matches); not hooked"))
        .collect();
    Ok((hooks, ambiguous))
}

/// Lifts the function at `address` alone, with every function it calls, and returns a report:
/// which ways of finding functions find it, then each place in it the lifter could not translate
/// and why, then the functions that failed outright.
///
/// # Errors
///
/// Returns a message when the disc or its executable cannot be read.
pub fn explain(iso: &Path, address: u32) -> Result<Vec<String>, String> {
    let xbe =
        xbe::Xbe::parse(discover::read_default_xbe(iso)?).map_err(|error| format!("{error:?}"))?;
    let (image, code, entry) = discover::load_image(&xbe);
    let finds = [
        ("entry point", vec![entry]),
        ("frame prologue", discover::prologue_starts(&image, &code)),
        (
            "after a return or padding",
            discover::boundary_starts(&image, &code),
        ),
        ("named by code", discover::code_addresses(&image, &code)),
        ("named by data", discover::data_addresses(&image, &code)),
    ];
    let mut lines = Vec::new();
    for (how, roots) in &finds {
        let found = if roots.contains(&address) {
            "yes"
        } else {
            "no"
        };
        lines.push(format!("  found by {how}: {found}"));
    }
    let program = lift::lift_program(
        &image,
        &[address],
        &std::collections::BTreeMap::new(),
        false,
    );
    lines.push(format!(
        "{address:#010x}: {} functions lifted, {} failed, {} places trap",
        program.lifted,
        program.failures.len(),
        program.traps.len()
    ));
    for error in program.failures.iter().take(20) {
        lines.push(format!("  failed {error}"));
    }
    for error in program.traps.iter().take(20) {
        lines.push(format!("  traps  {error}"));
    }
    Ok(lines)
}

/// A lifted program, in the language the build asked for.
enum Lifted {
    C(lift::Program),
    Rust(lift::rust::Program),
}

impl Lifted {
    /// Lifts the functions reachable from `roots`, to Rust when `rust` says so, else to C.
    fn lift(
        rust: bool,
        image: &lift::Image,
        roots: &[u32],
        hooks: &std::collections::BTreeMap<u32, u32>,
        tracing: bool,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Self {
        if rust {
            Self::Rust(lift::rust::lift_program_with_progress(
                image, roots, hooks, tracing, progress,
            ))
        } else {
            Self::C(lift::lift_program_with_progress(
                image, roots, hooks, tracing, progress,
            ))
        }
    }

    /// How many functions lifted, the places that trap, and the functions that did not lift.
    fn outcome(&self) -> (usize, &[lift::LiftError], &[lift::LiftError]) {
        match self {
            Self::C(program) => (program.lifted, &program.traps, &program.failures),
            Self::Rust(program) => (program.lifted, &program.traps, &program.failures),
        }
    }

    /// Writes the unit that holds the digest of `units` (see [`code_digest`]), to be compiled at
    /// `optimization` with `tracing` or not, to `work`, and returns its path.
    fn write_digest_unit(
        &self,
        work: &Path,
        units: &[PathBuf],
        optimization: &str,
        tracing: bool,
    ) -> Result<PathBuf, String> {
        let back_end = match self {
            Self::C(_) => "c",
            Self::Rust(_) => "rust",
        };
        let digest = code_digest(work, units, &[back_end, optimization, &tracing.to_string()])?;
        match self {
            Self::C(_) => glue::write_digest_unit(work, &digest),
            Self::Rust(_) => glue_rust::write_digest_unit(work, &digest),
        }
    }

    /// Writes the program's units and the boot image's to `work`, and returns their paths.
    fn write_units(
        &self,
        iso: &Path,
        work: &Path,
        entry: u32,
        name: &str,
    ) -> Result<Vec<PathBuf>, String> {
        Ok(match self {
            Self::C(program) => {
                let mut units = glue::write_units(work, program, entry, name)?;
                units.push(glue::write_boot_image(iso, work)?);
                units
            }
            Self::Rust(program) => {
                let mut units = glue_rust::write_units(work, program, entry, name)?;
                units.push(glue_rust::write_boot_image(iso, work)?);
                units
            }
        })
    }

    /// Compiles `units` at `optimization` (one of [`OPT_LEVELS`]) and links them with the runtime
    /// `library` into `work/Game`.
    fn compile_and_link(
        &self,
        units: &[PathBuf],
        work: &Path,
        library: &Path,
        optimization: &str,
        tracing: bool,
        reporter: &dyn Reporter,
    ) -> Result<PathBuf, String> {
        match self {
            Self::C(_) => {
                let optimization = format!("-O{optimization}");
                glue::compile_and_link(units, work, library, &optimization, tracing, reporter)
            }
            Self::Rust(_) => {
                glue_rust::compile_and_link(units, work, library, optimization, tracing, reporter)
            }
        }
    }
}

/// The digest of the lifted code a build links: the SHA-1 of `units`, the C header beside them,
/// the title's executable (`boot.xbe`) and `settings`. A snapshot belongs to it, so a build that
/// only changes the runtime still resumes the snapshots of the one before.
fn code_digest(work: &Path, units: &[PathBuf], settings: &[&str]) -> Result<[u8; 20], String> {
    use sha1::{Digest, Sha1};
    let mut sha = Sha1::new();
    let (header, xbe) = (work.join("lift.h"), work.join("boot.xbe"));
    let files = units
        .iter()
        .chain(header.is_file().then_some(&header))
        .chain(std::iter::once(&xbe));
    for file in files {
        let bytes = std::fs::read(file).map_err(|error| format!("{}: {error}", file.display()))?;
        sha.update((bytes.len() as u64).to_le_bytes());
        sha.update(&bytes);
    }
    for setting in settings {
        sha.update((setting.len() as u64).to_le_bytes());
        sha.update(setting.as_bytes());
    }
    Ok(sha.finalize().into())
}

/// Everything from the disc to the disk image, in `work`.
fn build_in(
    request: &Request,
    work: &Path,
    library: &Path,
    signatures: &Path,
    reporter: &dyn Reporter,
) -> Result<(), String> {
    let (iso, name, dmg) = (request.iso.as_path(), request.name.as_str(), &request.dmg);
    let xbe =
        xbe::Xbe::parse(discover::read_default_xbe(iso)?).map_err(|error| format!("{error:?}"))?;
    // The icon first, so a bad link fails before the minute of lifting. Without one, the icon
    // an earlier build left in the work directory is kept, else the title's own image is used.
    let icon_png = work.join("icon.png");
    if !request.icon.is_empty() {
        icon::write_icon(&request.icon, &icon_png, reporter)?;
    } else if icon_png.is_file() {
        note(reporter, format!("icon: keeping {}", icon_png.display()));
    } else {
        let title_image = xbe
            .sections()
            .iter()
            .find(|section| section.name() == "$$XTIMAGE")
            .map(|section| xbe.section_data(section));
        icon::write_default_icon(title_image, &icon_png, reporter)?;
    }
    let (image, code, entry) = discover::load_image(&xbe);
    let mut roots = vec![entry];
    roots.extend(discover::prologue_starts(&image, &code));
    roots.extend(discover::boundary_starts(&image, &code));
    roots.extend(discover::code_addresses(&image, &code));
    roots.extend(discover::data_addresses(&image, &code));
    // ANYBOX_TRACE_INSNS at build time builds the instruction trace into the title.
    let tracing = std::env::var_os("ANYBOX_TRACE_INSNS").is_some();
    let verbose = std::env::var_os("ANYBOX_VERBOSE").is_some();
    let plan = plan_hooks(&xbe, signatures)?;
    note(
        reporter,
        format!(
            "{} functions are replaced by runtime hooks",
            plan.hooks.len()
        ),
    );
    if verbose {
        for (name, matches) in &plan.ambiguous {
            note(
                reporter,
                format!("  ambiguous signature {name}: {matches} matches"),
            );
        }
        for (function, trap) in &plan.hooks {
            note(reporter, format!("  hook {function:#010x} -> {trap:#010x}"));
        }
    }
    let hooks: std::collections::BTreeMap<u32, u32> = plan.hooks.into_iter().collect();
    roots.extend(hooks.keys().copied());
    roots.sort_unstable();
    roots.dedup();
    let mut lifting = Tracker::start(Phase::Lifting, reporter);
    let program = Lifted::lift(
        request.rust,
        &image,
        &roots,
        &hooks,
        tracing,
        &mut |done, known| {
            lifting.update(done as u64, known as u64);
        },
    );
    let took = lifting.finish();
    let (lifted, traps, failures) = program.outcome();
    note(
        reporter,
        format!(
            "lifted {lifted} functions in {}; {} did not lift and will trap when called; {} places inside lifted functions trap if run",
            clock(took),
            failures.len(),
            traps.len()
        ),
    );
    if verbose {
        discover::report_traps(traps, failures, reporter);
    }
    let mut units = program.write_units(iso, work, entry, name)?;
    let optimization = request.opt.as_deref().unwrap_or(OPT_LEVELS[0]);
    units.push(program.write_digest_unit(work, &units, optimization, tracing)?);
    let level = request.pack_level.unwrap_or(DEFAULT_PACK_LEVEL);
    discpack::write_pack(iso, work, level, reporter)?;
    let game = program.compile_and_link(&units, work, library, optimization, tracing, reporter)?;
    let packaging = Tracker::start(Phase::DiskImage, reporter);
    let stage = work.join("stage");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(|error| error.to_string())?;
    bundle::build_app(
        &bundle::Parts {
            name,
            game: &game,
            pack: &work.join("disc.pack"),
            icon: &icon_png,
        },
        &stage,
    )?;
    bundle::build_dmg(name, &stage, dmg)?;
    let took = packaging.finish();
    note(
        reporter,
        format!("wrote {} in {}", dmg.display(), clock(took)),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_in_an_app_bundle_finds_the_runtime_in_its_resources() {
        let app = std::env::temp_dir().join(format!("anybox-find-runtime-{}", std::process::id()));
        let macos = app.join("Contents/MacOS");
        let resources = app.join("Contents/Resources/runtime");
        std::fs::create_dir_all(&macos).unwrap();
        assert_eq!(runtime_beside(&macos), None);
        std::fs::create_dir_all(&resources).unwrap();
        assert_eq!(
            runtime_beside(&macos).map(|dir| dir.canonicalize().unwrap()),
            Some(resources.canonicalize().unwrap())
        );
        std::fs::create_dir_all(macos.join("runtime")).unwrap();
        assert_eq!(runtime_beside(&macos), Some(macos.join("runtime")));
        std::fs::remove_dir_all(&app).unwrap();
    }
}
