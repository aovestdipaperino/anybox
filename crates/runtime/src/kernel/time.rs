//! The kernel clock, published time, and servicing time: expired waits and due timers (spec
//! 7.2).

use serde::{Deserialize, Serialize};

use std::time::Instant;

use super::Kernel;
use super::object::{Body, WaitOn};
use super::thread::State;
use crate::boot::{DATA_EXPORT_SIZE, DATA_EXPORTS};

/// The NT clock at boot: 2004-01-01 00:00:00 in 100 ns units since 1601.
pub const BOOT_CLOCK: u64 = 127_173_888_000_000_000;

/// Ticks (100 ns) per millisecond.
pub const TICKS_PER_MS: u64 = 10_000;

/// Whether the console runs at 50 Hz (`ANYXBOX_REFRESH=50`), read once: time is serviced on
/// every kernel contact.
#[must_use]
pub fn refresh_is_50() -> bool {
    static FIFTY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FIFTY.get_or_init(|| std::env::var("ANYXBOX_REFRESH").is_ok_and(|v| v == "50"))
}

/// Ticks between vertical blanks: 60 Hz, or 50 Hz with `ANYXBOX_REFRESH=50` (spec 7.3).
#[must_use]
pub fn vblank_period() -> u64 {
    if refresh_is_50() { 200_000 } else { 166_667 }
}

/// The kernel clock.
#[derive(Debug, Serialize, Deserialize)]
pub struct Clock {
    /// NT time, never going backwards.
    ticks: u64,
    /// Added for `KeSystemTime` and `KeQuerySystemTime` (`NtSetSystemTime`).
    pub offset: i64,
    at_host_start: u64,
    /// The host instant matching `at_host_start`, when the clock follows the host.
    #[serde(skip)]
    host_start: Option<Instant>,
}

impl Clock {
    /// A clock at `BOOT_CLOCK` that only moves when told to.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ticks: BOOT_CLOCK,
            offset: 0,
            at_host_start: BOOT_CLOCK,
            host_start: None,
        }
    }

    /// Takes `saved`'s time and stands still there, not following the host, until
    /// [`follow_host`](Self::follow_host) is called as the resumed title starts running: the
    /// guest sees no jump across a hibernation, nor across the wait for the first turn after it.
    pub fn resume_from(&mut self, saved: &Self) {
        self.ticks = saved.ticks;
        self.offset = saved.offset;
        self.host_start = None;
    }

    /// From now on the clock follows the host (spec 6.4, 7.2).
    pub fn follow_host(&mut self) {
        self.host_start = Some(Instant::now());
        self.at_host_start = self.ticks;
    }

    /// Moves the clock to the host time, never backwards.
    pub fn advance(&mut self) {
        if let Some(start) = self.host_start {
            let elapsed = u64::try_from(start.elapsed().as_nanos() / 100).unwrap_or(u64::MAX);
            self.ticks = self.ticks.max(self.at_host_start.saturating_add(elapsed));
        }
    }

    /// Moves the clock forward by `ticks` (for tests and clocks that do not follow the host).
    pub fn pass(&mut self, ticks: u64) {
        self.ticks += ticks;
    }

    /// The clock, without the system time offset.
    #[must_use]
    pub fn now(&self) -> u64 {
        self.ticks
    }

    /// System time: the clock plus the offset.
    #[must_use]
    pub fn system_time(&self) -> u64 {
        self.ticks.saturating_add_signed(self.offset)
    }

    /// Ticks since boot.
    #[must_use]
    pub fn since_boot(&self) -> u64 {
        self.ticks - BOOT_CLOCK
    }

    /// The clock tick a guest due time names: negative is relative to now, positive is an
    /// absolute system time; never before now.
    #[must_use]
    pub fn due(&self, value: i64) -> u64 {
        if value < 0 {
            self.ticks.saturating_add(value.unsigned_abs())
        } else {
            value
                .cast_unsigned()
                .saturating_add_signed(-self.offset)
                .max(self.ticks)
        }
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel {
    /// Writes `KeTickCount`, `KeSystemTime` and `KeInterruptTime` (spec 7.2).
    pub fn publish_time(&self) {
        let slot = |ordinal: u32| DATA_EXPORTS + ordinal * DATA_EXPORT_SIZE;
        let since_boot = self.clock.since_boot();
        let ms = u32::try_from(since_boot / TICKS_PER_MS).unwrap_or(u32::MAX);
        let _ = self.w32(slot(156), ms);
        for (ordinal, value) in [(154, self.clock.system_time()), (120, since_boot)] {
            let low = value as u32;
            let high = (value >> 32) as u32;
            let mut bytes = [0u8; 12];
            bytes[0..4].copy_from_slice(&low.to_le_bytes());
            bytes[4..8].copy_from_slice(&high.to_le_bytes());
            bytes[8..12].copy_from_slice(&high.to_le_bytes());
            let _ = self.write(slot(ordinal), &bytes);
        }
    }

    /// Services time: moves the clock, publishes it, wakes expired waits, fires due timers in
    /// order of due time.
    pub fn service_time(&mut self) {
        self.clock.advance();
        self.publish_time();
        self.wake_expired();
        self.fire_timers();
        self.service_gpu();
        self.service_sound();
    }

    /// Fires every timer whose due time has come, earliest first: signals it, re-arms a
    /// periodic one by whole periods past now, queues its DPC and starts its APC routine.
    fn fire_timers(&mut self) {
        let now = self.clock.now();
        loop {
            let due = self
                .objects
                .timers()
                .filter_map(|(id, t)| t.due.filter(|&d| d <= now).map(|d| (d, id)))
                .min();
            let Some((due, id)) = due else {
                break;
            };
            let (dpc, apc) = {
                let Body::Timer(timer) = &mut self.objects.get_mut(id).body else {
                    unreachable!("the due object is a timer")
                };
                timer.signaled = true;
                timer.due = (timer.period > 0)
                    .then(|| due + timer.period * ((now - due) / timer.period + 1));
                (timer.dpc, timer.apc)
            };
            self.mirror(id);
            let (low, high) = (now as u32, (now >> 32) as u32);
            if dpc != 0 {
                self.queue_dpc(dpc, low, high);
            }
            if let Some((routine, context)) = apc {
                self.run_routine(routine, &[context, low, high]);
            }
        }
        self.wake_satisfiable();
    }

    /// Vertical blanks that are due, then the GPU's interrupt (spec 7.3): while its line is
    /// asserted, every connected interrupt's service routine is queued as a DPC. Pictures are
    /// dropped until a picture sink is installed.
    pub fn service_gpu(&mut self) {
        let period = vblank_period();
        let now = self.clock.now();
        if self.next_vblank == 0 {
            self.next_vblank = now + period;
        }
        while now >= self.next_vblank {
            self.gpu.vblank();
            self.next_vblank += period;
        }
        self.service_interrupts();
        // Overlay pictures go to the window, newest last.
        if let Some(picture) = self.gpu.take_pictures().pop()
            && crate::host::window_enabled()
        {
            crate::host::milestone("picture");
            crate::host::show(
                &picture.rgb,
                3,
                picture.width as usize,
                picture.height as usize,
            );
        }
        self.deliver_frames();
        if let Some((at, flips)) = self.fps_report
            && now >= at + 10_000_000
        {
            if at != 0 {
                eprintln!(
                    "anyxbox: {} frames per second; {}",
                    self.gpu.flips - flips,
                    self.gpu.fifo_state()
                );
            }
            self.fps_report = Some((now, self.gpu.flips));
        }
    }

    /// Sends finished 3D frames, with their bound textures, to the frame sink; a frame the sink
    /// cannot take now is dropped (the renderer is behind). Each texture is fingerprinted where it
    /// is in guest memory, and its bytes are copied only when the renderer does not hold it, or a
    /// dump may need them.
    fn deliver_frames(&mut self) {
        let frames = self.gpu.take_frames();
        let Some(sink) = &self.frame_sink else {
            return;
        };
        let every_byte = crate::render::dump::wanted();
        for frame in frames {
            let textures = crate::gpu::draw::bound(&frame.commands)
                .into_iter()
                .filter_map(|texture| {
                    let (texels, texels_len) = (
                        crate::gpu::guest_address(texture.offset),
                        texture.byte_len(),
                    );
                    let (palette, palette_len) = (
                        crate::gpu::guest_address(texture.palette),
                        4 * texture.palette_colours,
                    );
                    let seed = crate::render::gpu::identity_seed(&texture);
                    let id = self.memory.fingerprint(
                        palette,
                        palette_len,
                        self.memory.fingerprint(texels, texels_len, seed)?,
                    )?;
                    let held = !every_byte
                        && crate::render::gpu::RESIDENT
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .contains(&id);
                    let read = |address: u32, len: u32| {
                        let mut bytes = vec![0u8; len as usize];
                        (len == 0 || self.memory.read(address, &mut bytes)).then_some(bytes)
                    };
                    let (bytes, palette) = if held {
                        (Vec::new(), Vec::new())
                    } else {
                        (read(texels, texels_len)?, read(palette, palette_len)?)
                    };
                    Some(crate::render::gpu::TextureData {
                        texture,
                        id,
                        bytes,
                        palette,
                    })
                })
                .collect();
            let _ = sink.try_send((frame.commands, textures));
        }
    }

    /// Queues the service routine of every connected interrupt while the GPU's line is asserted.
    /// Returns whether one was queued.
    pub fn service_interrupts(&mut self) -> bool {
        let mut queued = false;
        if self.gpu.line() {
            for interrupt in self.interrupts.clone() {
                queued |= self.queue_dpc(interrupt, 0, 0);
            }
        }
        queued
    }

    /// The earliest clock tick at which something waiting could wake: a wait deadline or an
    /// armed timer.
    #[must_use]
    pub fn next_wake(&self) -> Option<u64> {
        let deadlines = self
            .threads
            .iter()
            .filter(|t| t.state == State::Waiting)
            .filter_map(|t| t.wait.as_ref().and_then(|w| w.deadline));
        let timers = self.objects.timers().filter_map(|(_, t)| t.due);
        deadlines.chain(timers).min()
    }

    /// Whether a waiting thread is waiting only for the clock.
    #[must_use]
    pub fn any_delay(&self) -> bool {
        self.threads
            .iter()
            .any(|t| t.wait.as_ref().is_some_and(|w| w.on == WaitOn::Delay))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resumed_clock_stands_still_until_the_title_runs_then_goes_on_from_its_time() {
        let mut saved = Clock::new();
        saved.pass(5_000_000);
        saved.offset = 7;
        let mut clock = Clock::new();
        clock.follow_host();
        clock.resume_from(&saved);
        std::thread::sleep(std::time::Duration::from_millis(20));
        clock.advance();
        assert_eq!(
            clock.now(),
            BOOT_CLOCK + 5_000_000,
            "no time passes before the title runs"
        );
        assert_eq!(clock.offset, 7);
        clock.follow_host();
        std::thread::sleep(std::time::Duration::from_millis(20));
        clock.advance();
        let moved = clock.now() - (BOOT_CLOCK + 5_000_000);
        assert!(
            (200_000..10_000_000).contains(&moved),
            "it follows the host from its own time: moved {moved} ticks"
        );
    }

    #[test]
    fn due_times_are_relative_when_negative_and_never_before_now() {
        let mut clock = Clock::new();
        clock.pass(1000);
        assert_eq!(clock.due(-500), BOOT_CLOCK + 1500);
        assert_eq!(clock.due(0), BOOT_CLOCK + 1000);
        assert_eq!(clock.due(5), BOOT_CLOCK + 1000);
        let later = (BOOT_CLOCK + 5000).cast_signed();
        assert_eq!(clock.due(later), BOOT_CLOCK + 5000);
        clock.offset = 100;
        assert_eq!(clock.due(later), BOOT_CLOCK + 4900);
        assert_eq!(clock.system_time(), BOOT_CLOCK + 1100);
    }
}
