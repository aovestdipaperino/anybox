//! `anybox ISO ICON NAME DMG`: turns an original Xbox disc image into a macOS app inside a disk
//! image, from the command line. The work is `anybox-core`'s; this parses the arguments and draws
//! the build's progress on standard error.

mod terminal;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
usage: anybox [--runtime DIR] [--work DIR] [--pack-level N] [--opt 0|1|2] [--rust] ISO ICON NAME DMG
       anybox [--runtime DIR] --plan ISO
       anybox --explain ADDRESS ISO

  ISO     the disc image (XDVDFS .iso)
  ICON    an image file or an http(s) link, for the app's icon; empty for the icon an
          earlier build left in the work directory, else the title's own image
  NAME    the title's name, shown by macOS
  DMG     the disk image to write

  --runtime DIR     the directory with libruntime.a and signatures.txt; else $ANYBOX_RUNTIME,
                    else `runtime` beside this program
  --work DIR        keep the build files here (the pack is reused while the disc is the same);
                    else a temporary directory, removed at the end
  --pack-level N    the zstd level of the pack (a lower one is quicker to write)
  --opt LEVEL       the optimization level of the lifted code, 0, 1 (the default) or 2; 0 builds
                    a large title faster and runs slower
  --rust            lift the title to Rust and build it with rustc, which needs no C compiler;
                    else it is lifted to C and built with clang
  --plan            print the library functions the runtime will replace, and stop
  --explain ADDRESS lift the function at the hex ADDRESS alone, and print what it could not
                    translate, with each place's reason";

/// The command line.
#[derive(Debug, Default, PartialEq)]
struct Options {
    runtime: Option<PathBuf>,
    work: Option<PathBuf>,
    pack_level: Option<i32>,
    opt: Option<String>,
    plan: bool,
    rust: bool,
    explain: Option<u32>,
    positional: Vec<String>,
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut options = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--explain" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} needs a value\n{USAGE}"))?;
                    options.explain = Some(
                        u32::from_str_radix(value.trim_start_matches("0x"), 16)
                            .map_err(|_| format!("{value:?} is not a hex address\n{USAGE}"))?,
                    );
                }
                "--opt" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} needs a value\n{USAGE}"))?;
                    if !anybox_core::OPT_LEVELS.contains(&value.as_str()) {
                        return Err(format!("{value:?} is not 0, 1 or 2\n{USAGE}"));
                    }
                    options.opt = Some(value);
                }
                "--runtime" | "--work" | "--pack-level" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} needs a value\n{USAGE}"))?;
                    match arg.as_str() {
                        "--runtime" => options.runtime = Some(value.into()),
                        "--work" => options.work = Some(value.into()),
                        _ => {
                            options.pack_level = Some(
                                value
                                    .parse()
                                    .map_err(|_| format!("{value:?} is not a number\n{USAGE}"))?,
                            );
                        }
                    }
                }
                "--plan" => options.plan = true,
                "--rust" => options.rust = true,
                "--help" | "-h" => return Err(USAGE.to_owned()),
                _ if arg.starts_with("--") => return Err(format!("unknown option {arg}\n{USAGE}")),
                _ => options.positional.push(arg),
            }
        }
        Ok(options)
    }
}

/// The runtime directory, or the usage when none is found.
fn runtime_dir(option: Option<&Path>) -> Result<PathBuf, String> {
    anybox_core::find_runtime(option)
        .ok_or_else(|| format!("no runtime: pass --runtime DIR\n{USAGE}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("anybox: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let options = Options::parse(std::env::args().skip(1))?;
    if let Some(address) = options.explain {
        let [iso] = options.positional.as_slice() else {
            return Err(USAGE.to_owned());
        };
        for line in anybox_core::explain(Path::new(iso), address)? {
            println!("{line}");
        }
        return Ok(());
    }
    if options.plan {
        let [iso] = options.positional.as_slice() else {
            return Err(USAGE.to_owned());
        };
        let runtime = runtime_dir(options.runtime.as_deref())?;
        let (hooks, ambiguous) = anybox_core::plan(Path::new(iso), &runtime)?;
        for line in hooks {
            println!("{line}");
        }
        for line in ambiguous {
            eprintln!("anybox: {line}");
        }
        return Ok(());
    }
    let [iso, icon, name, dmg] = options.positional.as_slice() else {
        return Err(USAGE.to_owned());
    };
    let request = anybox_core::Request {
        iso: iso.into(),
        icon: icon.clone(),
        name: name.clone(),
        dmg: dmg.into(),
        runtime: runtime_dir(options.runtime.as_deref())?,
        work: options.work,
        pack_level: options.pack_level,
        opt: options.opt,
        rust: options.rust,
    };
    let terminal = terminal::Terminal::new();
    let built = anybox_core::build(&request, &terminal);
    terminal.clear();
    built
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        Options::parse(args.iter().map(|arg| (*arg).to_owned()))
    }

    #[test]
    fn the_four_paths_and_the_options_go_in_any_order() {
        let options = parse(&[
            "--runtime",
            "rt",
            "a.iso",
            "i.png",
            "--pack-level",
            "3",
            "Dragon's Lair 3D",
            "out.dmg",
            "--work",
            "w",
        ])
        .unwrap();
        assert_eq!(
            options.positional,
            ["a.iso", "i.png", "Dragon's Lair 3D", "out.dmg"]
        );
        assert_eq!(options.runtime, Some("rt".into()));
        assert_eq!(options.work, Some("w".into()));
        assert_eq!(options.pack_level, Some(3));
        assert!(!options.rust, "C unless asked");
        assert!(parse(&["--rust", "a.iso"]).unwrap().rust);
    }

    #[test]
    fn bad_options_are_refused() {
        assert!(parse(&["--runtime"]).is_err());
        assert!(parse(&["--pack-level", "fast"]).is_err());
        assert!(parse(&["--opt", "3"]).is_err());
        assert!(parse(&["--nope", "x"]).is_err());
        assert!(parse(&["--help"]).is_err());
    }
}
