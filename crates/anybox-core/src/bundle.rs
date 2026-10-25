//! The macOS app and the disk image that carries it.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The bundle identifier's prefix; the title's slug follows it.
const IDENTIFIER_PREFIX: &str = "net.enzolombardi.anybox";

/// What the app is made of.
#[derive(Debug)]
pub struct Parts<'a> {
    /// The title's name, shown by macOS.
    pub name: &'a str,
    /// The linked game executable.
    pub game: &'a Path,
    /// The disc's files, which the app reads from `Contents/Resources`.
    pub pack: &'a Path,
    /// The icon, a PNG.
    pub icon: &'a Path,
}

/// The bundle identifier's last part: `name` in lower case without apostrophes, with a dash for
/// each run of anything else that is not a letter or a digit.
#[must_use]
pub fn slug(name: &str) -> String {
    let mut slug = String::new();
    let mut dash = false;
    for c in name.chars().filter(|c| *c != '\'') {
        if c.is_ascii_alphanumeric() {
            if dash && !slug.is_empty() {
                slug.push('-');
            }
            dash = false;
            slug.push(c.to_ascii_lowercase());
        } else {
            dash = true;
        }
    }
    slug
}

/// `text` with the characters XML reserves written as entities.
#[must_use]
pub fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The app's `Info.plist`.
#[must_use]
pub fn info_plist(name: &str, version: &str) -> String {
    let name_xml = xml_escape(name);
    let slug = slug(name);
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>{name_xml}</string>
  <key>CFBundleDisplayName</key><string>{name_xml}</string>
  <key>CFBundleIdentifier</key><string>{IDENTIFIER_PREFIX}.{slug}</string>
  <key>CFBundleExecutable</key><string>anybox</string>
  <key>CFBundleIconFile</key><string>anybox</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>{version}</string>
  <key>CFBundleVersion</key><string>{version}</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"#
    )
}

/// The file name of the app for `name`: a slash cannot be in one.
fn app_file_name(name: &str) -> String {
    format!("{}.app", name.replace(['/', ':'], "-"))
}

fn run(command: &mut Command, what: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("{what}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let messages = String::from_utf8_lossy(&output.stderr);
        Err(format!("{what} failed\n{}", messages.trim_end()))
    }
}

/// Links `from` at `to`, or copies it when they are on different volumes: the pack is gigabytes.
fn link_or_copy(from: &Path, to: &Path) -> Result<(), String> {
    if std::fs::hard_link(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to)
        .map(|_| ())
        .map_err(|error| format!("copying {}: {error}", from.display()))
}

/// Builds the app inside `stage`, signs it for this machine, and returns its path.
///
/// The icon is scaled to every size macOS asks for, from the one image.
///
/// # Errors
///
/// Returns an error when a file cannot be written or a macOS tool fails.
pub fn build_app(parts: &Parts<'_>, stage: &Path) -> Result<PathBuf, String> {
    let app = stage.join(app_file_name(parts.name));
    let io = |error: std::io::Error| error.to_string();
    let macos = app.join("Contents/MacOS");
    let resources = app.join("Contents/Resources");
    std::fs::create_dir_all(&macos).map_err(io)?;
    std::fs::create_dir_all(&resources).map_err(io)?;
    std::fs::copy(parts.game, macos.join("anybox")).map_err(io)?;
    link_or_copy(parts.pack, &resources.join("disc.pack"))?;
    let iconset = stage.join("anybox.iconset");
    std::fs::create_dir_all(&iconset).map_err(io)?;
    for size in [16u32, 32, 128, 256, 512] {
        for (scale, suffix) in [(1, ""), (2, "@2x")] {
            let side = (size * scale).to_string();
            run(
                Command::new("sips")
                    .args(["-z", &side, &side])
                    .arg(parts.icon)
                    .arg("--out")
                    .arg(iconset.join(format!("icon_{size}x{size}{suffix}.png")))
                    .stdout(std::process::Stdio::null()),
                "scaling the icon",
            )?;
        }
    }
    run(
        Command::new("iconutil")
            .args(["-c", "icns"])
            .arg(&iconset)
            .arg("-o")
            .arg(resources.join("anybox.icns")),
        "making the icon set",
    )?;
    std::fs::remove_dir_all(&iconset).map_err(io)?;
    std::fs::write(
        app.join("Contents/Info.plist"),
        info_plist(parts.name, env!("CARGO_PKG_VERSION")),
    )
    .map_err(io)?;
    // Signed for this machine only (ad hoc): another Mac asks before opening it.
    run(
        Command::new("codesign")
            .args(["--force", "--deep", "--sign", "-"])
            .arg(&app)
            .stderr(std::process::Stdio::null()),
        "signing the app",
    )?;
    Ok(app)
}

/// Writes the disk image `dmg` holding the app in `stage`, and `Applications` to drag it onto.
///
/// The image is read-only and not recompressed: the pack is already compressed, and compressing
/// it again costs minutes for nothing.
///
/// # Errors
///
/// Returns an error when the link cannot be made or `hdiutil` fails.
pub fn build_dmg(name: &str, stage: &Path, dmg: &Path) -> Result<(), String> {
    std::os::unix::fs::symlink("/Applications", stage.join("Applications"))
        .map_err(|error| error.to_string())?;
    let _ = std::fs::remove_file(dmg);
    run(
        Command::new("hdiutil")
            .args(["create", "-quiet", "-volname", name, "-srcfolder"])
            .arg(stage)
            .args(["-ov", "-format", "UDRO"])
            .arg(dmg),
        "writing the disk image",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slug_is_lower_case_with_dashes_and_no_apostrophes() {
        assert_eq!(
            slug("Dragon's Lair 3D - Return to the Lair"),
            "dragons-lair-3d-return-to-the-lair"
        );
        assert_eq!(slug("  --A&B--  "), "a-b");
        assert_eq!(slug("Halo"), "halo");
        assert_eq!(slug(""), "");
    }

    #[test]
    fn the_plist_escapes_the_name_and_carries_the_identifier() {
        let plist = info_plist("Tom & <Jerry>", "1.2.3");
        assert!(plist.contains("<string>Tom &amp; &lt;Jerry&gt;</string>"));
        assert!(plist.contains("net.enzolombardi.anybox.tom-jerry"));
        assert!(plist.contains("<key>CFBundleVersion</key><string>1.2.3</string>"));
    }

    #[test]
    fn an_app_file_name_has_no_slash() {
        assert_eq!(app_file_name("a/b: c"), "a-b- c.app");
    }
}
