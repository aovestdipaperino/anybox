//! A build's events on standard error.
//!
//! On a terminal a phase in progress is one line, redrawn as it advances, with the percentage
//! done and the time left; a note clears it and is printed above. Elsewhere, a log say, a line
//! is written each time another tenth of a phase is done.

use std::io::{IsTerminal as _, Write as _};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use anybox_core::progress::{Event, Phase, Reporter, clock, percent, time_left};

/// Draws a build's events on standard error.
#[derive(Debug)]
pub(crate) struct Terminal {
    terminal: bool,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// Whether a progress line is on the screen.
    drawn: bool,
    /// The tenths of the phase in progress a log has been told of.
    tenths: u64,
}

impl Terminal {
    pub(crate) fn new() -> Self {
        Self {
            terminal: std::io::stderr().is_terminal(),
            state: Mutex::new(State::default()),
        }
    }

    /// Clears the progress line, if one is drawn.
    pub(crate) fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.clear_line(&mut state);
    }

    fn clear_line(&self, state: &mut State) {
        if self.terminal && state.drawn {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }
        state.drawn = false;
    }
}

impl Reporter for Terminal {
    fn report(&self, event: Event) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match event {
            Event::Started(_) => state.tenths = 0,
            Event::Progress {
                phase,
                done,
                total,
                elapsed,
            } => {
                let text = line(phase, done, total, elapsed);
                if self.terminal {
                    eprint!("\r{text}\x1b[K");
                    let _ = std::io::stderr().flush();
                    state.drawn = true;
                } else {
                    let tenths = (done * 10).checked_div(total).unwrap_or(0).min(10);
                    if tenths > state.tenths {
                        state.tenths = tenths;
                        eprintln!("{text}");
                    }
                }
            }
            Event::Finished { .. } => self.clear_line(&mut state),
            Event::Note(text) => {
                self.clear_line(&mut state);
                eprintln!("{text}");
            }
        }
    }
}

/// The line for `phase` at `done` of `total`, `elapsed` after it started.
fn line(phase: Phase, done: u64, total: u64, elapsed: Duration) -> String {
    let left = time_left(elapsed, done, total).map_or_else(
        || "estimating time left".to_owned(),
        |left| format!("about {} left", clock(left)),
    );
    format!(
        "{}: {:>3}% ({}), {left}",
        phase.name(),
        percent(done, total),
        phase.unit().amount(done, total)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_has_the_percentage_the_amount_and_the_time_left() {
        let ten = Duration::from_secs(10);
        assert_eq!(
            line(Phase::Lifting, 25, 100, ten),
            "lifting:  25% (25 of 100 functions), about 30s left"
        );
        assert_eq!(
            line(Phase::Compressing, 1 << 20, 4 << 20, ten),
            "compressing:  25% (1 of 4 MB), about 30s left"
        );
        assert_eq!(
            line(Phase::Compiling, 0, 0, Duration::ZERO),
            "compiling:   0% (0 of 0 MB), estimating time left"
        );
    }
}
