//! What a build tells whoever runs it: each phase as it starts, advances and ends, and the lines
//! worth reading along the way.
//!
//! A front end gives the build a [`Reporter`] and draws the events as it likes: the command line
//! as a line on standard error, the window as progress bars.

use std::time::{Duration, Instant};

/// How often a phase reports its progress: often enough to look live, seldom enough that a phase
/// with tens of thousands of steps does not flood the front end.
const REPORT_EVERY: Duration = Duration::from_millis(50);

/// How long a phase runs before its time left is guessed: the first moments are no guide.
pub const SETTLE: Duration = Duration::from_secs(2);

/// A step of the build that takes long enough to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Translating the title's functions to C or Rust.
    Lifting,
    /// Compressing the disc's files into the pack.
    Compressing,
    /// Compiling the C or the Rust.
    Compiling,
    /// Linking the title with the runtime. It reports no progress, only its start and end.
    Linking,
    /// Writing the app and the disk image. It reports no progress, only its start and end.
    DiskImage,
}

impl Phase {
    /// Every phase, in the order a build runs them.
    pub const ALL: [Self; 5] = [
        Self::Lifting,
        Self::Compressing,
        Self::Compiling,
        Self::Linking,
        Self::DiskImage,
    ];

    /// The phase's name, as a line or a label shows it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Lifting => "lifting",
            Self::Compressing => "compressing",
            Self::Compiling => "compiling",
            Self::Linking => "linking",
            Self::DiskImage => "making the disk image",
        }
    }

    /// What the phase counts.
    #[must_use]
    pub fn unit(self) -> Unit {
        match self {
            Self::Lifting => Unit::Count("functions"),
            Self::Compressing | Self::Compiling => Unit::Bytes,
            Self::Linking | Self::DiskImage => Unit::Count("steps"),
        }
    }
}

/// What a phase counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Things, named in the plural.
    Count(&'static str),
    /// Bytes, shown in megabytes.
    Bytes,
}

impl Unit {
    /// `done of total`, in this unit.
    #[must_use]
    pub fn amount(self, done: u64, total: u64) -> String {
        match self {
            Self::Count(things) => format!("{done} of {total} {things}"),
            Self::Bytes => format!("{} of {} MB", done >> 20, total >> 20),
        }
    }
}

/// Something that happened in a build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A phase started.
    Started(Phase),
    /// A phase has done `done` of `total`, `elapsed` after it started. The total of
    /// [`Phase::Lifting`] grows as lifted functions name callees not seen before.
    Progress {
        phase: Phase,
        done: u64,
        total: u64,
        elapsed: Duration,
    },
    /// A phase ended, `took` after it started; a phase with nothing to do ends at once.
    Finished { phase: Phase, took: Duration },
    /// A line for the user: what a phase found, or a choice the build made.
    Note(String),
}

/// Receives a build's events. Phases report from several threads at once.
pub trait Reporter: Sync {
    /// Takes one event.
    fn report(&self, event: Event);
}

impl<F: Fn(Event) + Sync> Reporter for F {
    fn report(&self, event: Event) {
        self(event);
    }
}

/// `done` of `total` as a whole percentage, at most 100.
#[must_use]
pub fn percent(done: u64, total: u64) -> u64 {
    (done * 100).checked_div(total).unwrap_or(0).min(100)
}

/// The time the rest of `total` takes at the rate `done` took `elapsed`, once the phase has run
/// long enough to say.
#[must_use]
pub fn time_left(elapsed: Duration, done: u64, total: u64) -> Option<Duration> {
    if done == 0 || elapsed < SETTLE {
        return None;
    }
    let millis = elapsed.as_millis() * u128::from(total.saturating_sub(done)) / u128::from(done);
    Some(Duration::from_millis(
        u64::try_from(millis).unwrap_or(u64::MAX),
    ))
}

/// `duration` to the second, as `1h 02m 03s`, `2m 03s` or `3s`.
#[must_use]
pub fn clock(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{hours}h {minutes:02}m {seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

/// One phase in progress: it reports its start, its progress at most every [`REPORT_EVERY`], and
/// its end.
pub(crate) struct Tracker<'a> {
    phase: Phase,
    reporter: &'a dyn Reporter,
    start: Instant,
    reported: Option<Instant>,
}

impl std::fmt::Debug for Tracker<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracker")
            .field("phase", &self.phase)
            .field("start", &self.start)
            .finish_non_exhaustive()
    }
}

impl<'a> Tracker<'a> {
    /// Starts `phase`.
    pub(crate) fn start(phase: Phase, reporter: &'a dyn Reporter) -> Self {
        reporter.report(Event::Started(phase));
        Self {
            phase,
            reporter,
            start: Instant::now(),
            reported: None,
        }
    }

    /// Records that `done` of `total` are done; the end of the work is always reported.
    pub(crate) fn update(&mut self, done: u64, total: u64) {
        let now = Instant::now();
        if done < total && self.reported.is_some_and(|at| now - at < REPORT_EVERY) {
            return;
        }
        self.reported = Some(now);
        self.reporter.report(Event::Progress {
            phase: self.phase,
            done,
            total,
            elapsed: now - self.start,
        });
    }

    /// Ends the phase, and returns how long it took.
    pub(crate) fn finish(self) -> Duration {
        let took = self.start.elapsed();
        self.reporter.report(Event::Finished {
            phase: self.phase,
            took,
        });
        took
    }
}

/// Reports `line` as a [`Event::Note`].
pub(crate) fn note(reporter: &dyn Reporter, line: impl Into<String>) {
    reporter.report(Event::Note(line.into()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn the_time_left_is_the_rest_at_the_rate_so_far() {
        let ten = Duration::from_secs(10);
        assert_eq!(time_left(ten, 0, 100), None);
        assert_eq!(time_left(Duration::from_secs(1), 50, 100), None, "too soon");
        assert_eq!(time_left(ten, 25, 100), Some(Duration::from_secs(30)));
        assert_eq!(time_left(ten, 100, 100), Some(Duration::ZERO));
        assert_eq!(time_left(ten, 120, 100), Some(Duration::ZERO));
    }

    #[test]
    fn a_percentage_is_whole_and_at_most_a_hundred() {
        assert_eq!(percent(1, 3), 33);
        assert_eq!(percent(5, 0), 0);
        assert_eq!(percent(7, 5), 100);
    }

    #[test]
    fn a_clock_shows_only_the_units_it_needs() {
        assert_eq!(clock(Duration::from_secs(3)), "3s");
        assert_eq!(clock(Duration::from_secs(123)), "2m 03s");
        assert_eq!(clock(Duration::from_secs(3723)), "1h 02m 03s");
    }

    #[test]
    fn an_amount_is_in_the_phases_unit() {
        assert_eq!(Phase::Lifting.unit().amount(3, 9), "3 of 9 functions");
        assert_eq!(
            Phase::Compressing.unit().amount(1 << 20, 4 << 20),
            "1 of 4 MB"
        );
    }

    #[test]
    fn a_tracker_reports_its_start_its_end_and_the_last_step_however_soon() {
        let events = Mutex::new(Vec::new());
        let reporter = |event: Event| events.lock().unwrap().push(event);
        let mut tracker = Tracker::start(Phase::Lifting, &reporter);
        tracker.update(1, 3);
        tracker.update(2, 3);
        tracker.update(3, 3);
        tracker.finish();
        let events = events.into_inner().unwrap();
        let steps: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Progress { done, .. } => Some(*done),
                _ => None,
            })
            .collect();
        assert_eq!(events[0], Event::Started(Phase::Lifting));
        assert_eq!(
            steps,
            [1, 3],
            "the middle step came too soon after the first"
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finished {
                phase: Phase::Lifting,
                ..
            })
        ));
    }
}
