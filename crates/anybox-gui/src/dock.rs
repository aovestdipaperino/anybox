//! The anybox logo as the app's icon in the Dock and the app switcher. The window runs as a bare
//! program, not from an app bundle, so macOS has no icon for it unless it is given one.

use objc2::AllocAnyThread as _;
use objc2::MainThreadMarker;
use objc2_app_kit::{NSApplication, NSImage};
use objc2_foundation::NSData;

/// The logo in the shape of a macOS icon, made from `crates/runtime/assets/logo.png`.
const ICON: &[u8] = include_bytes!("../assets/icon.png");

/// Shows the logo as the app's icon. It must run on the main thread, after the app has started;
/// elsewhere it does nothing.
pub(crate) fn set_icon() {
    let Some(main_thread) = MainThreadMarker::new() else {
        return;
    };
    let data = NSData::with_bytes(ICON);
    let Some(image) = NSImage::initWithData(NSImage::alloc(), &data) else {
        return;
    };
    let app = NSApplication::sharedApplication(main_thread);
    // SAFETY: on the main thread, with an image that lives as long as AppKit keeps it.
    unsafe { app.setApplicationIconImage(Some(&image)) };
}
