//! Whether the tools a build compiles and links with are installed, checked before the build
//! starts, with how to install what is missing.
//!
//! A build in C needs clang, and a build in Rust needs `rustc`; both need Xcode's command line
//! tools, for clang itself or for the linker and the macOS SDK.

use std::path::Path;
use std::process::Command;

/// How to install Xcode's command line tools.
const INSTALL_TOOLS: &str =
    "install them by running `xcode-select --install` in Terminal, then build again";

/// How to install Rust.
const INSTALL_RUST: &str = "install it with rustup, by running \
     `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` in Terminal (see \
     https://rustup.rs), then build again";

/// Checks that the tools for a build in Rust (`rust`) or in C are installed.
///
/// # Errors
///
/// What is missing, and how to install it.
pub fn check(rust: bool) -> Result<(), String> {
    check_with(
        rust,
        "xcode-select",
        "clang",
        crate::glue_rust::find_rustc().as_deref(),
    )
}

/// [`check`], with the programs it runs named, so a test can name ones that are not there.
fn check_with(
    rust: bool,
    xcode_select: &str,
    clang: &str,
    rustc: Option<&Path>,
) -> Result<(), String> {
    // Without the command line tools, `/usr/bin/clang` and `/usr/bin/ld` are stubs that offer to
    // install them, so they are checked first, through `xcode-select`.
    if run(Command::new(xcode_select).arg("-p")).is_err() {
        let needs = if rust {
            "the linker and the macOS SDK"
        } else {
            "clang"
        };
        return Err(format!(
            "Xcode's command line tools are not installed, and the build needs them for {needs}: \
             {INSTALL_TOOLS}."
        ));
    }
    if rust {
        let Some(rustc) = rustc else {
            return Err(format!(
                "Rust is not installed, and a Rust build needs its compiler, rustc: {INSTALL_RUST}."
            ));
        };
        if let Err(why) = run(Command::new(rustc).arg("--version")) {
            return Err(format!(
                "the Rust compiler at {} does not run ({why}). If rustup has no toolchain yet, \
                 run `rustup default stable` in Terminal; else {INSTALL_RUST}.",
                rustc.display()
            ));
        }
    } else if let Err(why) = run(Command::new(clang).arg("--version")) {
        return Err(format!(
            "clang does not run ({why}). It comes with Xcode's command line tools: \
             {INSTALL_TOOLS}. The command line can also build without clang, in Rust: \
             `anybox --rust`."
        ));
    }
    Ok(())
}

/// Runs `command`, and says why when it cannot start or fails.
fn run(command: &mut Command) -> Result<(), String> {
    let output = command.output().map_err(|error| error.to_string())?;
    if output.status.success() {
        return Ok(());
    }
    let messages = String::from_utf8_lossy(&output.stderr);
    Err(messages
        .lines()
        .next()
        .unwrap_or("it failed")
        .trim()
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A program that is nowhere.
    const MISSING: &str = "anybox-no-such-program";

    #[test]
    fn missing_command_line_tools_say_how_to_install_them() {
        for rust in [false, true] {
            let error = check_with(rust, MISSING, "clang", None).unwrap_err();
            assert!(error.contains("xcode-select --install"), "{error}");
        }
    }

    #[test]
    fn a_missing_clang_says_how_to_install_it_and_how_to_do_without() {
        let error = check_with(false, "true", MISSING, None).unwrap_err();
        assert!(error.contains("xcode-select --install"), "{error}");
        assert!(error.contains("anybox --rust"), "{error}");
    }

    #[test]
    fn a_missing_rust_says_how_to_install_it() {
        let error = check_with(true, "true", "clang", None).unwrap_err();
        assert!(error.contains("https://sh.rustup.rs"), "{error}");
        let error = check_with(true, "true", "clang", Some(Path::new(MISSING))).unwrap_err();
        assert!(error.contains("rustup default stable"), "{error}");
    }

    #[test]
    fn a_c_build_does_not_need_rust() {
        // `true` stands in for clang: it runs and succeeds.
        assert_eq!(check_with(false, "true", "true", None), Ok(()));
    }
}
