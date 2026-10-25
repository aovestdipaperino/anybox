//! Places `signatures.txt` beside `libruntime.a` in the target directory, so that directory is a
//! runtime `anybox --runtime` can use: the transpiler plans from the list the runtime embeds, the
//! converted `XbSymbolDatabase` signatures followed by those read from titles.

use std::path::{Path, PathBuf};

fn main() {
    let sources = [
        Path::new("data/signatures.txt"),
        Path::new("data/signatures-titles.txt"),
    ];
    let mut list = String::new();
    for source in sources {
        println!("cargo::rerun-if-changed={}", source.display());
        match std::fs::read_to_string(source) {
            Ok(text) => list.push_str(&text),
            Err(error) => println!(
                "cargo::warning=could not read {}: {error}",
                source.display()
            ),
        }
    }
    // OUT_DIR is target/<profile>/build/runtime-<hash>/out; the library goes to target/<profile>.
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let Some(profile_dir) = out.ancestors().nth(3) else {
        return;
    };
    if let Err(error) = std::fs::write(profile_dir.join("signatures.txt"), list) {
        println!("cargo::warning=could not write signatures.txt: {error}");
    }
}
