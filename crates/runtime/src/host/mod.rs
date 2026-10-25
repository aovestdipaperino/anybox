//! The host layer (spec 12): the window, presentation, and input.
//!
//! The window runs on the process's main thread. The newest image, a rendered 3D frame or an
//! overlay picture, waits in a shared slot; the window shows it through its own buffer. Once the
//! window has its GPU layer, 3D frames are shown there instead (spec 11.6), and the slot gets a
//! copy of one now and then. The frame rate is in the window's title.

pub mod audio;
pub mod crt;
pub mod fade;
pub mod input;
pub mod layer;
pub mod record;
pub mod titlebar;

use crate::hibernate::file::Picture;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use input::Pad;

/// The newest image to show: `0RGB` pixels, its size, and a serial number.
#[derive(Debug)]
struct Image {
    pixels: Vec<u32>,
    width: usize,
    height: usize,
    serial: u64,
}

static IMAGE: Mutex<Option<Image>> = Mutex::new(None);
/// Images delivered so far, those shown on the GPU layer included.
static DELIVERED: AtomicU64 = AtomicU64::new(0);
/// Whether the newest image is a 3D frame on the GPU layer rather than in the slot.
static ON_LAYER: AtomicBool = AtomicBool::new(false);
/// The pad scripted presses hold now.
static SCRIPTED: Mutex<Pad> = Mutex::new(Pad {
    buttons: 0,
    analog: [0; 8],
    sticks: [0; 4],
});

/// Whether the runtime has a window (`ANYXBOX_WINDOW=0` turns it off).
#[must_use]
pub fn window_enabled() -> bool {
    std::env::var("ANYXBOX_WINDOW").map_or(true, |v| v != "0")
}

/// Shows an image: `rgba` (4 bytes a pixel) or `rgb` (3), `width` by `height`.
pub fn show(bytes: &[u8], channels: usize, width: usize, height: usize) {
    let pixels = bytes
        .chunks_exact(channels)
        .map(|p| u32::from(p[0]) << 16 | u32::from(p[1]) << 8 | u32::from(p[2]))
        .collect();
    let serial = DELIVERED.fetch_add(1, Ordering::Relaxed) + 1;
    let mut slot = IMAGE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *slot = Some(Image {
        pixels,
        width,
        height,
        serial,
    });
    ON_LAYER.store(false, Ordering::Relaxed);
}

/// Says a new 3D frame is on the GPU layer, with a copy of a recent one when one has arrived:
/// its RGBA bytes, width and height.
pub fn show_on_layer(copy: Option<(&[u8], usize, usize)>) {
    match copy {
        Some((rgba, width, height)) => show(rgba, 4, width, height),
        None => {
            DELIVERED.fetch_add(1, Ordering::Relaxed);
        }
    }
    ON_LAYER.store(true, Ordering::Relaxed);
}

/// When the runtime booted, for milestones.
static BOOTED: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
/// The milestones already printed.
static MILESTONES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// Prints `first KIND N s after boot` the first time a milestone happens.
pub fn milestone(kind: &'static str) {
    let mut seen = MILESTONES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !seen.contains(&kind) {
        seen.push(kind);
        let seconds = BOOTED.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
        eprintln!("anyxbox: first {kind} {seconds:.1} s after boot");
    }
}

/// Persists the saves and exits the process.
fn quit() -> ! {
    crate::abi::with_kernel(|k| k.fs.persist());
    crate::hibernate::discard();
    std::process::exit(0);
}

/// Set when the title resumed from a snapshot: the window skips the boot logo.
static RESTORED: AtomicBool = AtomicBool::new(false);

/// The picture a resumed title's window fades in, until the window takes it.
static RESTORED_PICTURE: Mutex<Option<Picture>> = Mutex::new(None);

/// The title resumed from a snapshot, which kept `picture` for the window to fade in.
pub(crate) fn set_restored(picture: Option<Picture>) {
    *RESTORED_PICTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = picture;
    RESTORED.store(true, Ordering::Relaxed);
}

/// A picture as the snapshot keeps it, from the window's `width` x `height` `buffer`.
fn snapshot_picture(buffer: &[u32], width: usize, height: usize) -> Option<Picture> {
    Some(Picture {
        pixels: buffer.to_vec(),
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
    })
}

/// Set to switch off from outside the window, as `ANYXBOX_QUIT_AT` does.
static QUIT_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Set to reset from outside the window, as `ANYXBOX_RESET_AT` does.
static RESET_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Set to hibernate from outside the window, as `ANYXBOX_HIBERNATE_AT` does.
static HIBERNATE_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Hibernates without a window: on success the process ends; otherwise the title carries on.
fn hibernate_headless() {
    match crate::hibernate::hibernate(None) {
        Ok(path) => {
            eprintln!("anyxbox: hibernated to {}", path.display());
            std::process::exit(0);
        }
        Err(error) => eprintln!("anyxbox: cannot hibernate: {error}"),
    }
}

/// After `seconds`, quits or resets: through the window when there is one, which plays its
/// effects, else at once.
fn after(seconds: f64, reset: bool) {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs_f64(seconds.max(0.0)));
        match (window_enabled(), reset) {
            (true, false) => QUIT_REQUESTED.store(true, Ordering::Relaxed),
            (true, true) => RESET_REQUESTED.store(true, Ordering::Relaxed),
            (false, false) => quit(),
            (false, true) => crate::abi::reset_console(),
        }
    });
}

/// Starts the threads that quit at `ANYXBOX_QUIT_AT` seconds, reset at `ANYXBOX_RESET_AT`,
/// hibernate at `ANYXBOX_HIBERNATE_AT`, and
/// press and hold controls per `ANYXBOX_PRESS` (an entry without a duration held
/// `ANYXBOX_PRESS_MS`, 150 ms), timed from now (boot).
pub fn start_timers() {
    let _ = BOOTED.set(Instant::now());
    audio::start();
    for (variable, reset) in [("ANYXBOX_QUIT_AT", false), ("ANYXBOX_RESET_AT", true)] {
        if let Some(seconds) = std::env::var(variable)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            after(seconds, reset);
        }
    }
    if let Some(seconds) = std::env::var("ANYXBOX_HIBERNATE_AT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(seconds.max(0.0)));
            if window_enabled() {
                HIBERNATE_REQUESTED.store(true, Ordering::Relaxed);
            } else {
                hibernate_headless();
            }
        });
    }
    if let Ok(times) = std::env::var("ANYXBOX_DUMP_AT") {
        start_timed_dumps(&times);
    }
    record::start_playback();
    let Ok(text) = std::env::var("ANYXBOX_PRESS") else {
        return;
    };
    // How long a control is held without a duration: menus repeat a button held too long.
    let hold = std::env::var("ANYXBOX_PRESS_MS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(150);
    let changes = match input::presses(&text, f64::from(hold) / 1000.0) {
        Ok(changes) => changes,
        Err(error) => {
            eprintln!("anyxbox: ANYXBOX_PRESS: {error}");
            return;
        }
    };
    let start = Instant::now();
    std::thread::spawn(move || {
        for (at, pad) in changes {
            let due = start + Duration::from_secs_f64(at.max(0.0));
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
            set_scripted(pad);
        }
    });
}

/// Dumps the next frame at each of `times` (comma-separated seconds after boot), as F12 does:
/// with a recording played back, the same scenes can be dumped again.
fn start_timed_dumps(times: &str) {
    let mut times: Vec<f64> = times
        .split(',')
        .filter_map(|time| time.trim().parse().ok())
        .collect();
    times.sort_by(f64::total_cmp);
    let start = Instant::now();
    std::thread::spawn(move || {
        for at in times {
            let due = start + Duration::from_secs_f64(at.max(0.0));
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
            crate::render::dump_next();
        }
    });
}

fn set_scripted(pad: Pad) {
    *SCRIPTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = pad;
    apply(Pad::default());
}

/// The keyboard's pad, last applied, merged with the scripted one.
static KEYBOARD: Mutex<Pad> = Mutex::new(Pad {
    buttons: 0,
    analog: [0; 8],
    sticks: [0; 4],
});

/// Sets the kernel's controller to `keyboard` merged with the scripted presses, bumping the
/// packet number when it changes.
fn apply(keyboard: Pad) {
    let mut held = KEYBOARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *held = keyboard;
    let pad = keyboard.merge(
        *SCRIPTED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    crate::abi::with_kernel(|k| set_pad(k, pad));
}

/// Sets the kernel's controller to `pad`, bumping the packet number when it changes.
pub(crate) fn set_pad(kernel: &mut crate::kernel::Kernel, pad: Pad) {
    let state = &mut kernel.hooks.gamepad;
    if (state.buttons, state.analog, state.sticks) != (pad.buttons, pad.analog, pad.sticks) {
        state.buttons = pad.buttons;
        state.analog = pad.analog;
        state.sticks = pad.sticks;
        state.packet = state.packet.wrapping_add(1);
    }
}

/// The id of the Console menu's Reset item.
const MENU_RESET: usize = 1;
/// The id of the Console menu's Hibernate item.
const MENU_HIBERNATE: usize = 2;
/// The id of the Console menu's Perma Snapshot item.
const MENU_PERMA: usize = 3;
/// The id of the Console menu's Back to Perma Snapshot item.
const MENU_REWIND: usize = 4;

/// Adds the Console menu, with Reset (Cmd-R), Hibernate (Cmd-X), Perma Snapshot (Cmd-Shift-X)
/// and Back to Perma Snapshot (Cmd-Shift-Z), to the window's menu bar.
///
/// On macOS minifb gives a shortcut the Command key for `MENU_KEY_CTRL`, not for
/// `MENU_KEY_COMMAND`, which it ignores there: with it the shortcuts were the bare letters.
fn add_console_menu(window: &mut minifb::Window) {
    match minifb::Menu::new("Console") {
        Ok(mut menu) => {
            menu.add_item("Reset", MENU_RESET)
                .shortcut(minifb::Key::R, minifb::MENU_KEY_CTRL)
                .build();
            menu.add_item("Hibernate", MENU_HIBERNATE)
                .shortcut(minifb::Key::X, minifb::MENU_KEY_CTRL)
                .build();
            menu.add_item("Perma Snapshot", MENU_PERMA)
                .shortcut(
                    minifb::Key::X,
                    minifb::MENU_KEY_CTRL | minifb::MENU_KEY_SHIFT,
                )
                .build();
            menu.add_item("Back to Perma Snapshot", MENU_REWIND)
                .shortcut(
                    minifb::Key::Z,
                    minifb::MENU_KEY_CTRL | minifb::MENU_KEY_SHIFT,
                )
                .build();
            window.add_menu(&menu);
        }
        Err(error) => eprintln!("anyxbox: no Console menu: {error}"),
    }
}

/// Opens the title's window, 640 by 480 and resizable, with the Console menu and the title
/// bar's Reset button and volume slider.
fn open_window(name: &str) -> Option<minifb::Window> {
    let options = minifb::WindowOptions {
        resize: true,
        scale_mode: minifb::ScaleMode::AspectRatioStretch,
        ..minifb::WindowOptions::default()
    };
    let mut window = match minifb::Window::new(name, 640, 480, options) {
        Ok(window) => window,
        Err(error) => {
            eprintln!("anyxbox: cannot open a window: {error}");
            return None;
        }
    };
    window.set_target_fps(60);
    add_console_menu(&mut window);
    titlebar::add_title_bar_controls(&window);
    layer::attach(&window);
    Some(window)
}

/// Hibernates and exits when the menu (or `ANYXBOX_HIBERNATE_AT`) asks, or writes a perma
/// snapshot and carries on; the window shows `buffer`, `width` by `height`.
fn take_snapshot(
    menu: Option<usize>,
    window: &mut minifb::Window,
    buffer: &[u32],
    width: usize,
    height: usize,
) {
    if menu == Some(MENU_HIBERNATE) || HIBERNATE_REQUESTED.swap(false, Ordering::Relaxed) {
        match crate::hibernate::hibernate(snapshot_picture(buffer, width, height)) {
            Ok(path) => {
                eprintln!("anyxbox: hibernated to {}", path.display());
                fade_out(window, buffer, width, height);
                std::process::exit(0);
            }
            Err(error) => eprintln!("anyxbox: cannot hibernate: {error}"),
        }
    }
    if menu == Some(MENU_PERMA) {
        match crate::hibernate::perma_snapshot(snapshot_picture(buffer, width, height)) {
            Ok(path) => {
                eprintln!("anyxbox: perma snapshot in {}", path.display());
                titlebar::show_saved();
            }
            Err(error) => eprintln!("anyxbox: cannot take a perma snapshot: {error}"),
        }
    }
}

/// Takes the title back to its perma snapshot, fading in the picture it kept. Returns that
/// picture, `width` by `height`, if there is one; `None` (after saying why) when the title could
/// not go back and carries on.
#[allow(clippy::option_option)]
fn rewind(window: &mut minifb::Window) -> Option<Option<(Vec<u32>, usize, usize)>> {
    let snapshot = match crate::hibernate::restore::read_perma() {
        Ok(snapshot) => snapshot,
        Err(error) => {
            eprintln!("anyxbox: cannot go back to the perma snapshot: {error}");
            return None;
        }
    };
    let picture = crate::abi::rewind_console(snapshot);
    // The old title's sound and images go; the new one's follow.
    audio::flush();
    *IMAGE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    ON_LAYER.store(false, Ordering::Relaxed);
    eprintln!("anyxbox: back to the perma snapshot");
    let faded = picture.and_then(|picture| fade_in(window, picture));
    crate::hibernate::release();
    Some(faded)
}

/// Turns the window off as a CRT does, over the last picture: it collapses to a line, then to a
/// dot that fades, with a thump as the dot flares and the title silent. A window already closed
/// just ends.
fn switch_off(window: &mut minifb::Window, picture: &[u32], width: usize, height: usize) {
    crate::abi::with_kernel(|k| k.fs.persist());
    // The CRT switches off over the picture beneath the GPU's layer, a recent copy of it.
    layer::update(false);
    // The title falls silent as the picture goes.
    audio::mute();
    let crt = crt::Crt::switch_off(picture, width, height);
    let mut bumped = false;
    while let Some(frame) = crt.frame() {
        // The thump as the last dot flares.
        if !bumped && crt.at_dot() {
            audio::bump();
            bumped = true;
        }
        if window.update_with_buffer(&frame, width, height).is_err() {
            return;
        }
    }
    let black = vec![0; width * height];
    let _ = window.update_with_buffer(&black, width, height);
    // Long enough for the thump to die away.
    std::thread::sleep(Duration::from_millis(400));
}

/// Fades the window out over the last picture after a hibernation: from colour to black and
/// white, then to black, with the title silent. A window already closed just ends. The saves are
/// already persisted by the hibernation.
fn fade_out(window: &mut minifb::Window, picture: &[u32], width: usize, height: usize) {
    // The fade runs over the picture beneath the GPU's layer, a recent copy of it.
    layer::update(false);
    audio::mute();
    let fade = fade::Fade::out(picture);
    while let Some(frame) = fade.frame() {
        if window.update_with_buffer(&frame, width, height).is_err() {
            return;
        }
    }
    let black = vec![0; width * height];
    let _ = window.update_with_buffer(&black, width, height);
}

/// Fades a resumed title's `picture` in over the window: from black to black and white, then to
/// colour. Returns the picture to go on showing, or `None` when the picture does not fit its size
/// (nothing is drawn then); the caller releases the held title afterwards. A window that closes
/// meanwhile ends the fade early.
fn fade_in(window: &mut minifb::Window, picture: Picture) -> Option<(Vec<u32>, usize, usize)> {
    let (width, height) = (picture.width as usize, picture.height as usize);
    if width == 0 || height == 0 || picture.pixels.len() != width * height {
        return None;
    }
    layer::update(false);
    let fade = fade::Fade::inward(&picture.pixels);
    while let Some(frame) = fade.frame() {
        if window.update_with_buffer(&frame, width, height).is_err() {
            break;
        }
    }
    Some((picture.pixels, width, height))
}

/// Runs the window until quit, then exits the process. Without a window, parks forever.
pub fn main_loop() -> ! {
    let name = std::env::var("ANYXBOX_NAME").unwrap_or_else(|_| "anybox".to_owned());
    let Some(mut window) = window_enabled().then(|| open_window(&name)).flatten() else {
        // A resumed title waits for its fade-in, which no window will play: let it run.
        crate::hibernate::release();
        loop {
            std::thread::park();
        }
    };
    let (mut width, mut height) = (640, 480);
    // A resumed title's picture fades in first, with the title held frozen until it is done.
    let picture = RESTORED_PICTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    let faded = picture.and_then(|picture| fade_in(&mut window, picture));
    crate::hibernate::release();
    // The logo shows until the title's first image replaces it, warming up as a CRT does when
    // the console is switched on and after a reset. A restored title shows black instead.
    let restored = RESTORED.load(Ordering::Relaxed);
    let mut buffer = if let Some((pixels, w, h)) = faded {
        (width, height) = (w, h);
        pixels
    } else if restored {
        vec![0; width * height]
    } else {
        crt::boot_image(width, height)
    };
    let mut warm_up = (!restored).then(|| crt::Crt::warm_up(width, height));
    // Images up to this one were delivered before the boot the window shows.
    let mut booted_after = DELIVERED.load(Ordering::Relaxed);
    let (mut second, mut delivered_then) = (Instant::now(), booted_after);
    let mut keyboard = Pad::default();
    // With ANYXBOX_RECORD_INPUT, every change of the keyboard's controller goes to that file.
    let mut recorder = record::Recorder::from_env();
    loop {
        if !window.is_open() || QUIT_REQUESTED.load(Ordering::Relaxed) || titlebar::close_clicked()
        {
            crate::hibernate::discard();
            switch_off(&mut window, &buffer, width, height);
            std::process::exit(0);
        }
        let menu = window.is_menu_pressed();
        take_snapshot(menu, &mut window, &buffer, width, height);
        titlebar::blink();
        if (menu == Some(MENU_REWIND) || titlebar::rewind_clicked())
            && let Some(faded) = rewind(&mut window)
        {
            booted_after = DELIVERED.load(Ordering::Relaxed);
            warm_up = None;
            if let Some(picture) = faded {
                (buffer, width, height) = picture;
            }
        }
        if menu == Some(MENU_RESET)
            || titlebar::reset_clicked()
            || RESET_REQUESTED.swap(false, Ordering::Relaxed)
        {
            record::clear_playback();
            crate::abi::reset_console();
            // The logo warms up again until the new boot shows its first image, and a recording
            // and a playback start over with it, so they run from a boot.
            *IMAGE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            ON_LAYER.store(false, Ordering::Relaxed);
            booted_after = DELIVERED.load(Ordering::Relaxed);
            recorder = record::Recorder::from_env();
            record::start_playback();
            (width, height) = (640, 480);
            buffer = crt::boot_image(width, height);
            warm_up = Some(crt::Crt::warm_up(width, height));
        }
        if let Some(image) = IMAGE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            && image.serial > booted_after
        {
            (buffer, width, height) = (image.pixels, image.width, image.height);
        }
        if DELIVERED.load(Ordering::Relaxed) > booted_after {
            warm_up = None;
        }
        // Until the title shows an image, the logo warms up; then it holds.
        if let Some(frame) = crt::Crt::step(&mut warm_up, width, height) {
            buffer = frame;
        }
        layer::update(ON_LAYER.load(Ordering::Relaxed) && warm_up.is_none());
        if window.update_with_buffer(&buffer, width, height).is_err() {
            window.update();
        }
        if window.is_key_pressed(minifb::Key::F12, minifb::KeyRepeat::No) {
            crate::render::dump_next();
        }
        let pad = input::from_keys(&window.get_keys());
        if pad != keyboard {
            keyboard = pad;
            apply(pad);
            if let Some(recorder) = recorder.as_mut() {
                let frame = crate::abi::with_kernel(|k| k.gpu.flips).unwrap_or(0);
                recorder.record(record::When::Frame(frame), &pad);
            }
        }
        // The frame rate: images delivered in the last whole second, in the title.
        if second.elapsed() >= Duration::from_secs(1) {
            let delivered = DELIVERED.load(Ordering::Relaxed);
            window.set_title(&format!("{name} ({} fps)", delivered - delivered_then));
            (second, delivered_then) = (Instant::now(), delivered);
        }
    }
}
