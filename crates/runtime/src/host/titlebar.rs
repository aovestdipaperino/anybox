//! A Reset button, a volume slider and a floppy (which blinks after a perma snapshot and goes
//! back to it on a double-click) in the window's title bar, which `minifb` has no API for:
//! they are added with `AppKit` to the view that holds the window's own buttons and title, at its
//! right end. A
//! title bar accessory would be the usual way, but the window then reserves room for it below the
//! title bar, and `minifb`'s content view does not move to match.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission; the button
//! now sits at the right of the title bar instead of after the title.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{AllocAnyThread, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSApplication, NSAutoresizingMaskOptions, NSBezelStyle, NSButton, NSControlSize, NSImage,
    NSImageView, NSMenu, NSSlider, NSView, NSWindowButton,
};
use objc2_foundation::{NSObject, NSPoint, NSSize, NSString, NSUserDefaults};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

/// The gap between the button and the title bar's right edge, in points.
const RIGHT_MARGIN: f64 = 8.0;
/// The gap between the volume slider and the button, and the slider's width, in points.
const SLIDER_GAP: f64 = 10.0;
const SLIDER_WIDTH: f64 = 80.0;
/// The gap between the speaker icon and the slider, in points.
const ICON_GAP: f64 = 4.0;
/// The gap between the floppy and the button, in points.
const FLOPPY_GAP: f64 = 6.0;
/// How long the floppy blinks after a perma snapshot, and how long each blink is on or off.
const BLINKING: Duration = Duration::from_secs(2);
const BLINK_MS: u128 = 250;

/// The defaults domain the volume is kept in, shared by every title, and its key.
const DEFAULTS_SUITE: &str = "net.enzolombardi.anybox";
const VOLUME_KEY: &str = "volume";

thread_local! {
    /// The speaker icon beside the slider, which shows the volume; main thread only.
    static SPEAKER: RefCell<Option<Retained<NSImageView>>> = const { RefCell::new(None) };
    /// The floppy before the button, shown while the title has a perma snapshot; main thread
    /// only.
    static FLOPPY: RefCell<Option<Retained<NSButton>>> = const { RefCell::new(None) };
    /// When the last perma snapshot was written, while the floppy blinks for it.
    static SAVED_AT: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Set when the button is clicked, until [`reset_clicked`] takes it.
static RESET_CLICKED: AtomicBool = AtomicBool::new(false);

/// Set when the floppy is double-clicked, until [`rewind_clicked`] takes it.
static REWIND_CLICKED: AtomicBool = AtomicBool::new(false);

/// Set when the red close button is clicked; see [`close_clicked`].
static CLOSE_CLICKED: AtomicBool = AtomicBool::new(false);

define_class!(
    /// The button's target: its action only records the click, which the window loop acts on.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "AnyxboxResetTarget"]
    struct ResetTarget;

    impl ResetTarget {
        #[unsafe(method(reset:))]
        fn reset(&self, _sender: Option<&AnyObject>) {
            RESET_CLICKED.store(true, Ordering::Relaxed);
        }

        /// The red button's action: only recorded, so the window stays up for the CRT
        /// power-off, which the window loop plays before it closes the window itself.
        #[unsafe(method(closeClicked:))]
        fn close_clicked_action(&self, _sender: Option<&AnyObject>) {
            CLOSE_CLICKED.store(true, Ordering::Relaxed);
        }

        /// The floppy's action: a double-click asks to go back to the perma snapshot.
        #[unsafe(method(floppyClicked:))]
        fn floppy_clicked(&self, _sender: Option<&AnyObject>) {
            let double = NSApplication::sharedApplication(self.mtm())
                .currentEvent()
                .is_some_and(|event| event.clickCount() >= 2);
            if double {
                REWIND_CLICKED.store(true, Ordering::Relaxed);
            }
        }

        /// The volume slider's action: the sound follows it at once, and the volume is kept
        /// for the next title.
        #[unsafe(method(volumeChanged:))]
        fn volume_changed(&self, sender: Option<&AnyObject>) {
            let Some(slider) = sender.and_then(AnyObject::downcast_ref::<NSSlider>) else {
                return;
            };
            #[allow(clippy::cast_possible_truncation)]
            let volume = slider.doubleValue() as f32;
            super::audio::set_volume(volume);
            show_volume(volume);
            if let Some(defaults) = defaults() {
                defaults.setFloat_forKey(volume, &NSString::from_str(VOLUME_KEY));
            }
        }
    }
);

impl ResetTarget {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        // SAFETY: `init` is NSObject's initialiser, and the class adds no state to set up.
        unsafe { msg_send![Self::alloc(mtm), init] }
    }
}

/// Adds a Reset button at the right of the title bar of `window`, with a volume slider before it.
/// Does nothing off the main thread or without an `AppKit` window.
pub(crate) fn add_title_bar_controls(window: &minifb::Window) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return;
    };
    // SAFETY: an AppKit window handle's view is the window's live NSView, used on the main
    // thread while the window exists.
    let view: &NSView = unsafe { handle.ns_view.cast().as_ref() };
    let Some(ns_window) = view.window() else {
        return;
    };
    let target = ResetTarget::new(mtm);
    // SAFETY: the target is a ResetTarget, which implements the `reset:` action.
    let button = unsafe {
        NSButton::buttonWithTitle_target_action(
            &NSString::from_str("Reset"),
            Some(&target),
            Some(sel!(reset:)),
            mtm,
        )
    };
    // A small accessory-bar button fits within the title bar.
    button.setControlSize(NSControlSize::Small);
    button.setBezelStyle(NSBezelStyle::AccessoryBarAction);
    button.sizeToFit();
    // The red button asks for the power-off animation instead of closing at once.
    if let Some(red) = ns_window.standardWindowButton(NSWindowButton::CloseButton) {
        // SAFETY: the target is a ResetTarget, which implements `closeClicked:`.
        unsafe {
            red.setTarget(Some(&target));
            red.setAction(Some(sel!(closeClicked:)));
        }
    }
    // So does Quit (Cmd-Q), which would otherwise end the program at once.
    if let Some(menu) = NSApplication::sharedApplication(mtm).mainMenu() {
        redirect_quit(&menu, &target);
    }
    let Some(zoom) = ns_window.standardWindowButton(NSWindowButton::ZoomButton) else {
        return;
    };
    // SAFETY: the zoom button is in the title bar's view for as long as the window exists.
    let Some(title_bar) = (unsafe { zoom.superview() }) else {
        return;
    };
    let (zoom_frame, size) = (zoom.frame(), button.frame().size);
    // At the right end, centred on the window's buttons; a flexible left margin keeps it there
    // as the window resizes.
    button.setFrameOrigin(NSPoint::new(
        title_bar.frame().size.width - size.width - RIGHT_MARGIN,
        zoom_frame.origin.y + (zoom_frame.size.height - size.height) / 2.0,
    ));
    button.setAutoresizingMask(NSAutoresizingMaskOptions::ViewMinXMargin);
    title_bar.addSubview(&button);
    let left = add_floppy(&title_bar, &target, button.frame().origin.x, mtm);
    add_volume_slider(&title_bar, &target, left, mtm);
    // A control does not keep its target: this one lives as long as the program.
    std::mem::forget(target);
}

/// Adds the volume slider, and a speaker icon before it, to `title_bar`, ending `right` points
/// from its left edge; the slider starts at the volume kept from the last title.
fn add_volume_slider(title_bar: &NSView, target: &ResetTarget, right: f64, mtm: MainThreadMarker) {
    let key = NSString::from_str(VOLUME_KEY);
    let volume = defaults()
        .filter(|defaults| defaults.objectForKey(&key).is_some())
        .map_or(1.0, |defaults| defaults.floatForKey(&key).clamp(0.0, 1.0));
    super::audio::set_volume(volume);
    // SAFETY: the target is a ResetTarget, which implements `volumeChanged:`.
    let slider = unsafe {
        NSSlider::sliderWithValue_minValue_maxValue_target_action(
            f64::from(volume),
            0.0,
            1.0,
            Some(target),
            Some(sel!(volumeChanged:)),
            mtm,
        )
    };
    slider.setControlSize(NSControlSize::Small);
    slider.setContinuous(true);
    slider.setToolTip(Some(&NSString::from_str("Volume")));
    slider.sizeToFit();
    let height = slider.frame().size.height;
    slider.setFrameSize(NSSize::new(SLIDER_WIDTH, height));
    let bar = title_bar.frame().size.height;
    let slider_x = right - SLIDER_GAP - SLIDER_WIDTH;
    slider.setFrameOrigin(NSPoint::new(slider_x, (bar - height) / 2.0));
    slider.setAutoresizingMask(NSAutoresizingMaskOptions::ViewMinXMargin);
    title_bar.addSubview(&slider);
    let Some(image) = speaker_image(volume) else {
        return;
    };
    let icon = NSImageView::imageViewWithImage(&image, mtm);
    let size = icon.frame().size;
    icon.setFrameOrigin(NSPoint::new(
        slider_x - ICON_GAP - size.width,
        (bar - size.height) / 2.0,
    ));
    icon.setAutoresizingMask(NSAutoresizingMaskOptions::ViewMinXMargin);
    title_bar.addSubview(&icon);
    SPEAKER.with(|speaker| *speaker.borrow_mut() = Some(icon));
}

/// Adds the floppy to `title_bar`, ending `right` points from its left edge, shown if the title
/// has a perma snapshot; double-clicking it goes back to the snapshot. Returns where it starts.
fn add_floppy(title_bar: &NSView, target: &ResetTarget, right: f64, mtm: MainThreadMarker) -> f64 {
    // SAFETY: the target is a ResetTarget, which implements `floppyClicked:`.
    let floppy = unsafe {
        NSButton::buttonWithTitle_target_action(
            &NSString::from_str("\u{1F4BE}"),
            Some(target),
            Some(sel!(floppyClicked:)),
            mtm,
        )
    };
    floppy.setBordered(false);
    floppy.setToolTip(Some(&NSString::from_str(
        "Perma snapshot saved: double-click to go back to it",
    )));
    floppy.sizeToFit();
    let size = floppy.frame().size;
    let x = right - FLOPPY_GAP - size.width;
    floppy.setFrameOrigin(NSPoint::new(
        x,
        (title_bar.frame().size.height - size.height) / 2.0,
    ));
    floppy.setAutoresizingMask(NSAutoresizingMaskOptions::ViewMinXMargin);
    floppy.setHidden(!crate::hibernate::perma_exists());
    title_bar.addSubview(&floppy);
    FLOPPY.with(|slot| *slot.borrow_mut() = Some(floppy));
    x
}

/// Starts the floppy blinking, as a perma snapshot has been written. Main thread only.
pub(crate) fn show_saved() {
    SAVED_AT.set(Some(Instant::now()));
}

/// Shows or hides the floppy for this moment of its blinking, then leaves it shown, as the perma
/// snapshot is there; the window loop calls it every frame. Main thread only.
pub(crate) fn blink() {
    let Some(saved) = SAVED_AT.get() else {
        return;
    };
    let elapsed = saved.elapsed();
    let on = elapsed >= BLINKING || (elapsed.as_millis() / BLINK_MS).is_multiple_of(2);
    if elapsed >= BLINKING {
        SAVED_AT.set(None);
    }
    FLOPPY.with(|floppy| {
        if let Some(floppy) = floppy.borrow().as_ref() {
            floppy.setHidden(!on);
        }
    });
}

/// The defaults the volume is kept in.
fn defaults() -> Option<Retained<NSUserDefaults>> {
    NSUserDefaults::initWithSuiteName(
        NSUserDefaults::alloc(),
        Some(&NSString::from_str(DEFAULTS_SUITE)),
    )
}

/// The speaker symbol for `volume`: struck through when silent, with more waves as it rises.
fn speaker_image(volume: f32) -> Option<Retained<NSImage>> {
    let name = match volume {
        v if v <= 0.0 => "speaker.slash.fill",
        v if v < 0.34 => "speaker.wave.1.fill",
        v if v < 0.67 => "speaker.wave.2.fill",
        _ => "speaker.wave.3.fill",
    };
    NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str(name),
        Some(&NSString::from_str("Volume")),
    )
}

/// Shows `volume` on the speaker icon.
fn show_volume(volume: f32) {
    SPEAKER.with(|speaker| {
        if let (Some(icon), Some(image)) = (speaker.borrow().as_ref(), speaker_image(volume)) {
            icon.setImage(Some(&image));
        }
    });
}

/// Points every menu item in `menu` and its submenus that quits the application at the target's
/// `closeClicked:`, so quitting switches off as closing the window does.
fn redirect_quit(menu: &NSMenu, target: &ResetTarget) {
    for item in menu.itemArray() {
        if let Some(submenu) = item.submenu() {
            redirect_quit(&submenu, target);
        }
        if item.action() == Some(sel!(terminate:)) {
            // SAFETY: the target is a ResetTarget, which implements `closeClicked:`.
            unsafe {
                item.setTarget(Some(target));
                item.setAction(Some(sel!(closeClicked:)));
            }
        }
    }
}

/// Whether the button was clicked since the last call.
pub(crate) fn reset_clicked() -> bool {
    RESET_CLICKED.swap(false, Ordering::Relaxed)
}

/// Whether the floppy was double-clicked since the last call.
pub(crate) fn rewind_clicked() -> bool {
    REWIND_CLICKED.swap(false, Ordering::Relaxed)
}

/// Whether the red close button was clicked; it stays set, as the program is then switching off.
pub(crate) fn close_clicked() -> bool {
    CLOSE_CLICKED.load(Ordering::Relaxed)
}
