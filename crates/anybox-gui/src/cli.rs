//! "Install Command Line Tool": puts `anybox` on the PATH as a link to the command-line program
//! inside this app, the way VS Code installs `code`. A link rather than a copy, so replacing the
//! app updates the command, and the command finds the runtime of the same build.
//!
//! The link goes in `/usr/local/bin`, asking macOS for an administrator's password when that
//! folder is not writable, and in `~/.local/bin` when the password is declined.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use gpui_kit::{App, PromptLevel, Window};

/// The command's name on the PATH, and the program's name beside the window's.
const NAME: &str = "anybox";

/// The folder for everyone, on the PATH of every macOS shell.
const SYSTEM_DIR: &str = "/usr/local/bin";

/// How long the login shell has to print its PATH.
const SHELL_TIMEOUT: Duration = Duration::from_secs(5);

/// Where `anybox` is on this Mac, as far as this app is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum State {
    /// Neither place holds an `anybox`.
    Absent,
    /// A link at this path leads to this app's program.
    Ours(PathBuf),
    /// Something else is at this path: a file, or a link to another program.
    Other(PathBuf),
}

/// The command-line program beside the window's, if it is there.
pub(crate) fn program() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe = exe.canonicalize().unwrap_or(exe);
    Some(exe.parent()?.join(NAME)).filter(|program| program.is_file())
}

/// The two places the command may be linked, in the order they are tried.
fn places() -> Vec<PathBuf> {
    let mut places = vec![Path::new(SYSTEM_DIR).join(NAME)];
    if let Some(home) = std::env::var_os("HOME") {
        places.push(PathBuf::from(home).join(".local/bin").join(NAME));
    }
    places
}

/// Whether `place` is a link that leads to `program`.
fn leads_to(place: &Path, program: &Path) -> bool {
    place.is_symlink()
        && match (place.canonicalize(), program.canonicalize()) {
            (Ok(place), Ok(program)) => place == program,
            _ => false,
        }
}

/// Where `program` is linked among `places`, or what stands in the way.
fn state_in(program: &Path, places: &[PathBuf]) -> State {
    if let Some(place) = places.iter().find(|place| leads_to(place, program)) {
        return State::Ours(place.clone());
    }
    places
        .iter()
        .find(|place| place.symlink_metadata().is_ok())
        .map_or(State::Absent, |place| State::Other(place.clone()))
}

/// The command's state on this Mac; [`State::Absent`] when the app has no program to link.
pub(crate) fn state() -> State {
    program().map_or(State::Absent, |program| state_in(&program, &places()))
}

/// `text` as one word for `/bin/sh`.
fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// `text` as an `AppleScript` string literal.
fn applescript_string(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', r"\\").replace('"', "\\\""))
}

/// Runs `script` in `/bin/sh` as root, after macOS asks for an administrator's password.
/// `Ok(false)` when the password was declined.
fn run_as_admin(script: &str) -> Result<bool, String> {
    let output = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(format!(
            "do shell script {} with administrator privileges",
            applescript_string(script)
        ))
        .output()
        .map_err(|error| format!("cannot run osascript: {error}"))?;
    if output.status.success() {
        return Ok(true);
    }
    let error = String::from_utf8_lossy(&output.stderr);
    // -128 is AppleScript's "User canceled".
    if error.contains("(-128)") {
        return Ok(false);
    }
    Err(error.trim().to_owned())
}

/// Links `place` to `program`, replacing what is there.
fn link(program: &Path, place: &Path) -> std::io::Result<()> {
    if let Some(dir) = place.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if place.symlink_metadata().is_ok() {
        std::fs::remove_file(place)?;
    }
    std::os::unix::fs::symlink(program, place)
}

/// The PATH a new Terminal window gets, as the user's login shell prints it.
fn login_path() -> Option<String> {
    let shell = std::env::var_os("SHELL").unwrap_or_else(|| "/bin/zsh".into());
    let mut child = Command::new(shell)
        .args(["-ilc", r#"printf '\n%s' "$PATH""#])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started = Instant::now();
    while child.try_wait().ok()?.is_none() {
        if started.elapsed() > SHELL_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let output = child.wait_with_output().ok()?;
    // Start-up files may print first; the PATH is the last line.
    let text = String::from_utf8_lossy(&output.stdout);
    text.rsplit('\n').next().map(str::to_owned)
}

/// Whether `dir` is one of the folders in `path`.
fn on_path(dir: &Path, path: &str) -> bool {
    std::env::split_paths(path).any(|entry| entry == dir)
}

/// What an install or uninstall did, to tell the user.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// Linked at this path; `false` when its folder isn't on the login shell's PATH.
    Installed(PathBuf, bool),
    /// These links were removed.
    Removed(Vec<PathBuf>),
}

/// Links the command, in [`SYSTEM_DIR`] if possible, else in `~/.local/bin`.
fn install(program: &Path) -> Result<Outcome, String> {
    let places = places();
    let system = &places[0];
    if link(program, system).is_ok() {
        return Ok(Outcome::Installed(system.clone(), true));
    }
    let script = format!(
        "mkdir -p {dir} && rm -f {place} && ln -s {program} {place}",
        dir = shell_quote(SYSTEM_DIR),
        place = shell_quote(&system.to_string_lossy()),
        program = shell_quote(&program.to_string_lossy()),
    );
    // A declined password, or a failure as root, falls back to the user's own folder.
    if run_as_admin(&script) == Ok(true) && leads_to(system, program) {
        return Ok(Outcome::Installed(system.clone(), true));
    }
    let Some(user) = places.get(1) else {
        return Err("no home folder to install in".to_owned());
    };
    link(program, user).map_err(|error| format!("cannot link {}: {error}", user.display()))?;
    let dir = user.parent().unwrap_or(user);
    let found = login_path().is_some_and(|path| on_path(dir, &path));
    Ok(Outcome::Installed(user.clone(), found))
}

/// Removes every link to `program`, asking for a password where the folder needs one.
fn uninstall(program: &Path) -> Result<Outcome, String> {
    let mut removed = Vec::new();
    for place in places()
        .into_iter()
        .filter(|place| leads_to(place, program))
    {
        if std::fs::remove_file(&place).is_err() {
            let script = format!("rm -f {}", shell_quote(&place.to_string_lossy()));
            if !run_as_admin(&script)? {
                break;
            }
        }
        removed.push(place);
    }
    Ok(Outcome::Removed(removed))
}

/// The message and detail an outcome is told with.
fn describe(outcome: &Outcome) -> (String, String) {
    match outcome {
        Outcome::Installed(place, true) => (
            "The anybox command is installed".to_owned(),
            format!(
                "{} links to the anybox program inside this app. Open a new Terminal window and run `anybox --help`.",
                place.display()
            ),
        ),
        Outcome::Installed(place, false) => {
            let dir = place.parent().unwrap_or(place);
            (
                "The anybox command is installed, but not on your PATH".to_owned(),
                format!(
                    "{} links to the anybox program inside this app, but {} isn't on your PATH. Add this line to ~/.zprofile, then open a new Terminal window:\n\nexport PATH=\"{}:$PATH\"",
                    place.display(),
                    dir.display(),
                    dir.display()
                ),
            )
        }
        Outcome::Removed(places) if places.is_empty() => (
            "The anybox command was not removed".to_owned(),
            "Removing it needs an administrator's password.".to_owned(),
        ),
        Outcome::Removed(places) => (
            "The anybox command is uninstalled".to_owned(),
            format!(
                "Removed {}.",
                places
                    .iter()
                    .map(|place| place.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" and ")
            ),
        ),
    }
}

/// The menu item's label for the command's current state.
pub(crate) fn menu_label() -> &'static str {
    match state() {
        State::Ours(_) => "Uninstall Command Line Tool",
        State::Absent | State::Other(_) => "Install Command Line Tool…",
    }
}

/// Installs the command, or uninstalls it when it is installed, telling the user in `window`.
/// `done` runs afterwards, to relabel the menu.
pub(crate) fn toggle(window: &mut Window, cx: &mut App, done: fn(&mut App)) {
    window
        .spawn(cx, async move |cx| {
            let Some(program) = program() else {
                let _ = cx
                    .prompt(
                        PromptLevel::Critical,
                        "This app has no anybox command to install",
                        Some("The anybox program should be beside the window's in Contents/MacOS."),
                        &["OK"],
                    )
                    .await;
                return;
            };
            let state = state_in(&program, &places());
            if let State::Other(place) = &state {
                let answer = cx
                    .prompt(
                        PromptLevel::Warning,
                        &format!("{} already exists", place.display()),
                        Some("It isn't a link to this app's anybox program. Replace it?"),
                        &["Replace", "Cancel"],
                    )
                    .await;
                if answer != Ok(0) {
                    return;
                }
            }
            let outcome = cx
                .background_executor()
                .spawn(async move {
                    match state {
                        State::Ours(_) => uninstall(&program),
                        State::Absent | State::Other(_) => install(&program),
                    }
                })
                .await;
            let _ = cx.update(|_, cx| done(cx));
            let (level, message, detail) = match &outcome {
                Ok(outcome @ (Outcome::Installed(_, true) | Outcome::Removed(_))) => {
                    let (message, detail) = describe(outcome);
                    (PromptLevel::Info, message, detail)
                }
                Ok(outcome) => {
                    let (message, detail) = describe(outcome);
                    (PromptLevel::Warning, message, detail)
                }
                Err(error) => (
                    PromptLevel::Critical,
                    "The anybox command could not be changed".to_owned(),
                    error.clone(),
                ),
            };
            let _ = cx.prompt(level, &message, Some(&detail), &["OK"]).await;
        })
        .detach();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder for one test, removed when it ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("anybox-cli-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_link_to_this_program_is_ours_and_anything_else_is_in_the_way() {
        let scratch = Scratch::new("state");
        let program = scratch.0.join("Anybox.app/Contents/MacOS/anybox");
        std::fs::create_dir_all(program.parent().unwrap()).unwrap();
        std::fs::write(&program, "").unwrap();
        let other = scratch.0.join("other");
        std::fs::write(&other, "").unwrap();
        let places = [
            scratch.0.join("system/anybox"),
            scratch.0.join("user/anybox"),
        ];

        assert_eq!(state_in(&program, &places), State::Absent);

        link(&other, &places[0]).unwrap();
        assert_eq!(state_in(&program, &places), State::Other(places[0].clone()));

        link(&program, &places[1]).unwrap();
        assert_eq!(state_in(&program, &places), State::Ours(places[1].clone()));

        link(&program, &places[0]).unwrap();
        assert_eq!(state_in(&program, &places), State::Ours(places[0].clone()));
    }

    #[test]
    fn a_copy_of_the_program_is_not_ours() {
        let scratch = Scratch::new("copy");
        let program = scratch.0.join("anybox");
        std::fs::write(&program, "").unwrap();
        let place = scratch.0.join("bin/anybox");
        std::fs::create_dir_all(place.parent().unwrap()).unwrap();
        std::fs::copy(&program, &place).unwrap();
        assert!(!leads_to(&place, &program));
    }

    #[test]
    fn paths_with_quotes_survive_the_shell_and_applescript() {
        assert_eq!(shell_quote("/Apps/A b"), "'/Apps/A b'");
        assert_eq!(shell_quote("/it's"), r"'/it'\''s'");
        assert_eq!(
            applescript_string(r#"say "hi" \ bye"#),
            r#""say \"hi\" \\ bye""#
        );
    }

    #[test]
    fn a_folder_is_on_the_path_only_as_a_whole_entry() {
        let path = "/usr/bin:/Users/me/.local/bin:/bin";
        assert!(on_path(Path::new("/Users/me/.local/bin"), path));
        assert!(!on_path(Path::new("/Users/me/.local"), path));
    }
}
