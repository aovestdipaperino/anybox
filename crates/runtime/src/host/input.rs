//! The controller from the keyboard, and scripted presses and holds (spec 12.2).

use minifb::Key;

/// Digital buttons.
pub const UP: u16 = 0x01;
pub const DOWN: u16 = 0x02;
pub const LEFT: u16 = 0x04;
pub const RIGHT: u16 = 0x08;
pub const START: u16 = 0x10;
pub const BACK: u16 = 0x20;
/// The stick clicks.
pub const LEFT_THUMB: u16 = 0x40;
pub const RIGHT_THUMB: u16 = 0x80;

/// Analog button indices: A, B, X, Y, Black, White, left trigger, right trigger.
const A: usize = 0;
const B: usize = 1;
const X: usize = 2;
const Y: usize = 3;
const BLACK: usize = 4;
const WHITE: usize = 5;
const LEFT_TRIGGER: usize = 6;
const RIGHT_TRIGGER: usize = 7;

/// The controller's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pad {
    pub buttons: u16,
    pub analog: [u8; 8],
    /// LX, LY, RX, RY.
    pub sticks: [i16; 4],
}

impl Pad {
    /// Both states held at once: buttons of either, the stronger analog value, a moved stick.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            buttons: self.buttons | other.buttons,
            analog: std::array::from_fn(|i| self.analog[i].max(other.analog[i])),
            sticks: std::array::from_fn(|i| {
                if other.sticks[i] != 0 {
                    other.sticks[i]
                } else {
                    self.sticks[i]
                }
            }),
        }
    }
}

/// A stick axis from a negative and a positive key: full deflection.
fn axis(keys: &[Key], negative: Key, positive: Key) -> i16 {
    match (keys.contains(&negative), keys.contains(&positive)) {
        (true, false) => i16::MIN,
        (false, true) => i16::MAX,
        _ => 0,
    }
}

/// The controller the held keys make: arrows the left stick, W A S D the D-pad, Enter Start,
/// Backspace Back, Space Escape C V the A B X Y buttons, 1 2 Black and White, Q E or the Shift keys
/// the triggers, I J K L the right stick.
#[must_use]
pub fn from_keys(keys: &[Key]) -> Pad {
    let mut pad = Pad::default();
    for (key, bit) in [
        (Key::W, UP),
        (Key::S, DOWN),
        (Key::A, LEFT),
        (Key::D, RIGHT),
        (Key::Enter, START),
        (Key::Backspace, BACK),
    ] {
        if keys.contains(&key) {
            pad.buttons |= bit;
        }
    }
    let analog = [
        (Key::Space, A),
        (Key::Escape, B),
        (Key::C, X),
        (Key::V, Y),
        (Key::Key1, BLACK),
        (Key::Key2, WHITE),
        (Key::Q, LEFT_TRIGGER),
        (Key::LeftShift, LEFT_TRIGGER),
        (Key::E, RIGHT_TRIGGER),
        (Key::RightShift, RIGHT_TRIGGER),
    ];
    for (key, index) in analog {
        if keys.contains(&key) {
            pad.analog[index] = 255;
        }
    }
    pad.sticks = [
        axis(keys, Key::Left, Key::Right),
        axis(keys, Key::Down, Key::Up),
        axis(keys, Key::J, Key::L),
        axis(keys, Key::K, Key::I),
    ];
    pad
}

/// One control of the controller, as `ANYXBOX_PRESS` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    /// A digital button: its bit.
    Button(u16),
    /// An analog button or trigger: its index in `Pad::analog`.
    Analog(usize),
    /// A stick axis: its index in `Pad::sticks`.
    Axis(usize),
}

impl Control {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "up" => Self::Button(UP),
            "down" => Self::Button(DOWN),
            "left" => Self::Button(LEFT),
            "right" => Self::Button(RIGHT),
            "start" => Self::Button(START),
            "back" => Self::Button(BACK),
            "ls" => Self::Button(LEFT_THUMB),
            "rs" => Self::Button(RIGHT_THUMB),
            "a" => Self::Analog(A),
            "b" => Self::Analog(B),
            "x" => Self::Analog(X),
            "y" => Self::Analog(Y),
            "black" => Self::Analog(BLACK),
            "white" => Self::Analog(WHITE),
            "lt" => Self::Analog(LEFT_TRIGGER),
            "rt" => Self::Analog(RIGHT_TRIGGER),
            "lx" => Self::Axis(0),
            "ly" => Self::Axis(1),
            "rx" => Self::Axis(2),
            "ry" => Self::Axis(3),
            _ => return None,
        })
    }
}

/// A control held from `at` for `duration` seconds at `value` (0 to 1, or -1 to 1 for an axis).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Press {
    at: f64,
    duration: f64,
    control: Control,
    value: f64,
}

impl Press {
    /// Sets this control in `pad`.
    fn apply(&self, pad: &mut Pad) {
        // The values are range-checked when parsed.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        match self.control {
            Control::Button(bit) => pad.buttons |= bit,
            Control::Analog(index) => pad.analog[index] = (self.value * 255.0).round() as u8,
            Control::Axis(index) => {
                pad.sticks[index] = if self.value < 0.0 {
                    (self.value * 32768.0).round().max(-32768.0) as i16
                } else {
                    (self.value * 32767.0).round() as i16
                };
            }
        }
    }
}

/// Parses one `control[=value]@seconds[+duration]` entry; `hold` is the duration when it gives
/// none.
fn press(entry: &str, hold: f64) -> Result<Press, String> {
    let (what, when) = entry
        .split_once('@')
        .ok_or_else(|| format!("`{entry}` is not control[=value]@seconds[+duration]"))?;
    let number = |text: &str| {
        text.trim()
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("`{text}` is not a number"))
    };
    let (at, duration) = match when.split_once('+') {
        Some((at, duration)) => (number(at)?, number(duration)?),
        None => (number(when)?, hold),
    };
    if at < 0.0 || duration < 0.0 {
        return Err(format!("`{entry}` has a negative time"));
    }
    let (name, value) = match what.split_once('=') {
        Some((name, value)) => (name, Some(number(value)?)),
        None => (what, None),
    };
    let name = name.trim().to_lowercase();
    let control = Control::named(&name).ok_or_else(|| format!("`{name}` is not a control"))?;
    let value = match (control, value) {
        (Control::Button(_), Some(_)) => return Err(format!("`{name}` takes no value")),
        (Control::Axis(_), None) => return Err(format!("`{name}` needs a value from -1 to 1")),
        (Control::Axis(_), Some(value)) if !(-1.0..=1.0).contains(&value) => {
            return Err(format!("`{name}={value}` is not from -1 to 1"));
        }
        (Control::Analog(_), Some(value)) if !(0.0..=1.0).contains(&value) => {
            return Err(format!("`{name}={value}` is not from 0 to 1"));
        }
        (_, value) => value.unwrap_or(1.0),
    };
    Ok(Press {
        at,
        duration,
        control,
        value,
    })
}

/// Parses `ANYXBOX_PRESS`: comma-separated `control[=value]@seconds[+duration]` entries, timed in
/// seconds from boot, into the controller's state from each moment a press starts or ends on, in
/// time order. Each state holds every press that is on then; where two of one control overlap,
/// the later one wins.
///
/// The buttons `start`, `back`, `up`, `down`, `left`, `right`, `ls` and `rs` (the stick clicks)
/// are pressed; the analog buttons `a`, `b`, `x`, `y`, `black`, `white` and the triggers `lt` and
/// `rt` take a pressure from 0 to 1, full without one; the sticks `lx`, `ly`, `rx` and `ry` need
/// a position from -1 to 1 (`ly=1` is up). An entry without a duration is held `hold` seconds.
/// Holds may overlap: `lx=-1@5+2,a@6` runs left for two seconds and jumps on the way.
///
/// # Errors
///
/// The first entry that does not parse.
pub fn presses(text: &str, hold: f64) -> Result<Vec<(f64, Pad)>, String> {
    let presses = text
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| press(entry, hold))
        .collect::<Result<Vec<_>, _>>()?;
    let mut moments: Vec<f64> = presses
        .iter()
        .flat_map(|press| [press.at, press.at + press.duration])
        .collect();
    moments.sort_by(f64::total_cmp);
    moments.dedup();
    Ok(moments
        .into_iter()
        .map(|moment| {
            let mut on: Vec<&Press> = presses
                .iter()
                .filter(|press| press.at <= moment && moment < press.at + press.duration)
                .collect();
            on.sort_by(|a, b| a.at.total_cmp(&b.at));
            let mut pad = Pad::default();
            for press in on {
                press.apply(&mut pad);
            }
            (moment, pad)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_map_to_the_controller() {
        let pad = from_keys(&[
            Key::Enter,
            Key::W,
            Key::Space,
            Key::Up,
            Key::J,
            Key::RightShift,
        ]);
        assert_eq!(pad.buttons, START | UP);
        assert_eq!(pad.analog, [255, 0, 0, 0, 0, 0, 0, 255]);
        assert_eq!(pad.sticks, [0, i16::MAX, i16::MIN, 0]);
        assert_eq!(
            from_keys(&[Key::Left, Key::Right]).sticks[0],
            0,
            "opposite keys cancel"
        );
    }

    #[test]
    fn scripted_presses_parse_and_merge() {
        let presses = presses("start@3.5, a@10", 0.15).unwrap();
        assert_eq!(presses.len(), 4, "each press starts and ends");
        assert_eq!((presses[0].0, presses[0].1.buttons), (3.5, START));
        assert_eq!(presses[1].1, Pad::default(), "released after the hold");
        assert!((presses[1].0 - 3.65).abs() < 1e-9);
        assert_eq!(presses[2].1.analog[A], 255);
        for bad in [
            "jump@1",
            "start",
            "lx@2",
            "ly=2@3",
            "up=1@4",
            "a@x",
            "a=0.5@1+y",
        ] {
            assert!(super::presses(bad, 0.15).is_err(), "{bad}");
        }
        let merged = from_keys(&[Key::Up]).merge(presses[0].1);
        assert_eq!((merged.buttons, merged.sticks[1]), (START, i16::MAX));
    }

    #[test]
    fn sticks_and_triggers_are_held_for_their_durations() {
        let timeline = presses("lx=-1.0@5+2, a@6, rt=0.5@6.5+1, ry=1@1", 0.15).unwrap();
        let at = |time: f64| {
            timeline
                .iter()
                .rev()
                .find(|(moment, _)| *moment <= time)
                .map_or_else(Pad::default, |(_, pad)| *pad)
        };
        assert_eq!(at(1.0).sticks, [0, 0, 0, i16::MAX], "up on the right stick");
        assert_eq!(at(1.2).sticks, [0; 4], "released after the default hold");
        let jumping = at(6.05);
        assert_eq!(jumping.sticks[0], i16::MIN, "left, held");
        assert_eq!(jumping.analog[A], 255, "and A pressed on the way");
        let braking = at(6.6);
        assert_eq!((braking.analog[A], braking.analog[RIGHT_TRIGGER]), (0, 128));
        assert_eq!(braking.sticks[0], i16::MIN);
        assert_eq!(
            at(7.0).sticks[0],
            0,
            "the stick is let go after two seconds"
        );
        assert_eq!(at(7.5), Pad::default(), "everything is released at the end");
    }
}
