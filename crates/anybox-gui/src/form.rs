//! What the window knows without drawing anything: what a dropped file is for, the defaults a disc
//! suggests, and where each phase of a build stands.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anybox_core::progress::{Event, Phase, clock, percent, time_left};

/// The image formats offered as icons; `sips` reads them all.
const IMAGE_EXTENSIONS: [&str; 11] = [
    "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "heic", "webp", "icns", "ico",
];

/// What a file the user gives the window is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Target {
    Disc,
    Icon,
}

/// What `path`, dropped anywhere in the window, is for: a disc image or an image for the icon.
/// `None` for anything else.
pub(crate) fn classify(path: &Path) -> Option<Target> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    if extension == "iso" || extension == "xiso" {
        Some(Target::Disc)
    } else if IMAGE_EXTENSIONS.contains(&extension.as_str()) {
        Some(Target::Icon)
    } else {
        None
    }
}

/// The name a disc at `iso` suggests for the title: its file's, without the extension.
pub(crate) fn title_name(iso: &Path) -> String {
    iso.file_stem()
        .map(|stem| stem.to_string_lossy().trim().to_owned())
        .unwrap_or_default()
}

/// The disk image a build of the disc at `iso` writes: beside it, `.iso` become `.dmg`.
pub(crate) fn disk_image(iso: &Path) -> PathBuf {
    iso.with_extension("dmg")
}

/// Where one phase of a build stands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Stage {
    #[default]
    Waiting,
    /// Started; `done` of `total`, `elapsed` in, once it has said.
    Running {
        progress: Option<(u64, u64, Duration)>,
    },
    Finished(Duration),
}

/// Where every phase of a build stands, in [`Phase::ALL`] order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Phases([Stage; Phase::ALL.len()]);

impl Phases {
    /// The stage of `phase`.
    pub(crate) fn stage(&self, phase: Phase) -> Stage {
        self.0[index(phase)]
    }

    /// Takes `event` into account; a note changes no phase.
    pub(crate) fn apply(&mut self, event: &Event) {
        match *event {
            Event::Started(phase) => self.0[index(phase)] = Stage::Running { progress: None },
            Event::Progress {
                phase,
                done,
                total,
                elapsed,
            } => {
                self.0[index(phase)] = Stage::Running {
                    progress: Some((done, total, elapsed)),
                };
            }
            Event::Finished { phase, took } => self.0[index(phase)] = Stage::Finished(took),
            Event::Note(_) => {}
        }
    }
}

fn index(phase: Phase) -> usize {
    Phase::ALL
        .iter()
        .position(|each| *each == phase)
        .expect("every phase is in ALL")
}

impl Stage {
    /// How full the phase's bar is, from 0 to 100.
    pub(crate) fn fill(self) -> f32 {
        match self {
            Self::Waiting | Self::Running { progress: None } => 0.0,
            Self::Running {
                progress: Some((done, total, _)),
            } => {
                // A whole percentage is at most 100, so it is exact as an f32.
                #[allow(clippy::cast_precision_loss)]
                let fill = percent(done, total) as f32;
                fill
            }
            Self::Finished(_) => 100.0,
        }
    }

    /// Whether the bar should move without a value: a phase that reports no progress, running.
    pub(crate) fn indeterminate(self) -> bool {
        matches!(self, Self::Running { progress: None })
    }

    /// The words beside the phase's bar.
    pub(crate) fn caption(self, phase: Phase) -> String {
        match self {
            Self::Waiting => "waiting".to_owned(),
            Self::Running { progress: None } => "working".to_owned(),
            Self::Running {
                progress: Some((done, total, elapsed)),
            } => {
                let left = time_left(elapsed, done, total).map_or_else(
                    || "estimating time left".to_owned(),
                    |left| format!("about {} left", clock(left)),
                );
                format!(
                    "{}% · {} · {left}",
                    percent(done, total),
                    phase.unit().amount(done, total)
                )
            }
            Self::Finished(took) => format!("done in {}", clock(took)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dropped_file_goes_where_its_kind_belongs() {
        assert_eq!(classify(Path::new("/x/Halo.ISO")), Some(Target::Disc));
        assert_eq!(classify(Path::new("/x/cover.png")), Some(Target::Icon));
        assert_eq!(classify(Path::new("/x/cover.JPEG")), Some(Target::Icon));
        assert_eq!(classify(Path::new("/x/notes.txt")), None);
        assert_eq!(classify(Path::new("/x/no-extension")), None);
    }

    #[test]
    fn a_disc_suggests_its_name_and_a_disk_image_beside_it() {
        let iso = Path::new("/discs/Dragon's Lair 3D.iso");
        assert_eq!(title_name(iso), "Dragon's Lair 3D");
        assert_eq!(disk_image(iso), Path::new("/discs/Dragon's Lair 3D.dmg"));
    }

    #[test]
    fn a_phase_waits_runs_and_finishes() {
        let mut phases = Phases::default();
        assert_eq!(phases.stage(Phase::Compiling), Stage::Waiting);
        phases.apply(&Event::Started(Phase::Linking));
        let linking = phases.stage(Phase::Linking);
        assert!(linking.indeterminate());
        assert_eq!(linking.caption(Phase::Linking), "working");
        phases.apply(&Event::Progress {
            phase: Phase::Lifting,
            done: 25,
            total: 100,
            elapsed: Duration::from_secs(10),
        });
        let lifting = phases.stage(Phase::Lifting);
        assert!((lifting.fill() - 25.0).abs() < f32::EPSILON);
        assert_eq!(
            lifting.caption(Phase::Lifting),
            "25% · 25 of 100 functions · about 30s left"
        );
        phases.apply(&Event::Finished {
            phase: Phase::Lifting,
            took: Duration::from_secs(40),
        });
        assert_eq!(
            phases.stage(Phase::Lifting).caption(Phase::Lifting),
            "done in 40s"
        );
        assert_eq!(phases.stage(Phase::Compiling), Stage::Waiting);
    }
}
