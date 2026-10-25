//! The layer the GPU shows the title's 3D frames on, over `minifb`'s picture (spec 11.6): a
//! `CAMetalLayer` in a view of its own above `minifb`'s view, sized to the largest 4:3 rectangle
//! that fits the window. The renderer draws into it, so no frame is copied back to the CPU but
//! now and then. `minifb`'s picture beneath still shows the boot logo, the movies and the CRT
//! effects; the layer is shown only while the title draws 3D frames.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission; the frame
//! rate it also drew on the layer is not used, the frame rate is in the title.

use std::cell::RefCell;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly};
use objc2_app_kit::NSView;
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_quartz_core::{CAMetalLayer, CATransaction};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

/// The width over the height of the title's frames.
const ASPECT: f64 = 4.0 / 3.0;

/// The `CAMetalLayer`, for the renderer; null until [`attach`].
static LAYER: AtomicPtr<std::ffi::c_void> = AtomicPtr::new(std::ptr::null_mut());
/// The layer's size in pixels, width in the high half, for the renderer to draw at.
static SIZE: AtomicU64 = AtomicU64::new(0);

/// The view, kept on the main thread, which `AppKit` needs it on.
struct Views {
    host: Retained<NSView>,
    /// What was last set, so nothing is set again each frame.
    shown: bool,
    frame: CGRect,
}

thread_local! {
    static VIEWS: RefCell<Option<Views>> = const { RefCell::new(None) };
}

/// Adds the layer to `window`, hidden. Does nothing off the main thread or without an `AppKit`
/// window.
pub(crate) fn attach(window: &minifb::Window) {
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
    let Some(content) = ns_window.contentView() else {
        return;
    };
    let metal = CAMetalLayer::new();
    metal.setOpaque(true);
    metal.setContentsScale(ns_window.backingScaleFactor());
    // A layer-hosting view: the layer is set before the view is made layer-backed.
    let host = NSView::initWithFrame(NSView::alloc(mtm), content.bounds());
    host.setLayer(Some(&metal));
    host.setWantsLayer(true);
    host.setHidden(true);
    content.addSubview(&host);
    LAYER.store(
        Retained::as_ptr(&metal).cast_mut().cast(),
        Ordering::Release,
    );
    // The layer lives as long as the program; the view holds it too.
    std::mem::forget(metal);
    VIEWS.with(|views| {
        *views.borrow_mut() = Some(Views {
            host,
            shown: false,
            frame: CGRect::default(),
        });
    });
}

/// The `CAMetalLayer` the renderer draws to, once the window has one.
pub(crate) fn pointer() -> Option<*mut std::ffi::c_void> {
    let layer = LAYER.load(Ordering::Acquire);
    (!layer.is_null()).then_some(layer)
}

/// The layer's size in pixels.
pub(crate) fn size() -> (u32, u32) {
    let size = SIZE.load(Ordering::Relaxed);
    // Each half holds a 32-bit size.
    #[allow(clippy::cast_possible_truncation)]
    ((size >> 32) as u32, size as u32)
}

/// Shows or hides the layer and fits it to the window, on each turn of the window loop.
/// Returns whether the layer is showing.
pub(crate) fn update(show: bool) -> bool {
    VIEWS.with(|views| {
        let mut views = views.borrow_mut();
        let Some(views) = views.as_mut() else {
            return false;
        };
        // SAFETY: the host view is in the window's content view for as long as the window exists.
        let Some(content) = (unsafe { views.host.superview() }) else {
            return false;
        };
        // The part of the window the title bar leaves, in the content view's coordinates: the
        // content view itself reaches under the title bar.
        let bounds = match content.window() {
            Some(window) => content.convertRect_fromView(window.contentLayoutRect(), None),
            None => content.bounds(),
        };
        let (width, height) = if bounds.size.width / bounds.size.height > ASPECT {
            (bounds.size.height * ASPECT, bounds.size.height)
        } else {
            (bounds.size.width, bounds.size.width / ASPECT)
        };
        let frame = CGRect::new(
            CGPoint::new(
                bounds.origin.x + (bounds.size.width - width) / 2.0,
                bounds.origin.y + (bounds.size.height - height) / 2.0,
            ),
            CGSize::new(width, height),
        );
        if views.shown == show && views.frame == frame {
            return show;
        }
        // Without a transaction of its own, each change would fade in over a quarter second.
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        if views.frame != frame {
            views.host.setFrame(frame);
            let scale = views
                .host
                .window()
                .map_or(1.0, |window| window.backingScaleFactor());
            // Whole pixels, which the frame's size in points times the scale always nearly is.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pixels = |points: f64| (points * scale).round() as u64;
            SIZE.store(pixels(width) << 32 | pixels(height), Ordering::Relaxed);
            views.frame = frame;
        }
        if views.shown != show {
            views.host.setHidden(!show);
            views.shown = show;
        }
        CATransaction::commit();
        show
    })
}
