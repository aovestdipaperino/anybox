//! `anybox-gui`: turns an original Xbox disc image into a macOS app inside a disk image, from a
//! window. Drop the disc or the icon on it, and watch each phase of the build fill its bar. The
//! work is `anybox-core`'s; the runtime is this workspace's release build.

// GPUI idioms: the prelude glob, held tasks named `_` by convention, and view methods that keep
// `self` so they can be wired with `cx.listener`.
#![allow(
    clippy::wildcard_imports,
    clippy::used_underscore_binding,
    clippy::unused_self
)]

mod builder;
mod cli;
mod dock;
mod form;
mod splash;

use gpui_kit::*;

actions!(anybox, [Quit, ToggleCommandLineTool]);

/// The app's menu; the command-line item says whether it installs or uninstalls.
fn set_menus(cx: &mut App) {
    cx.set_menus(vec![Menu {
        name: "anybox".into(),
        items: vec![
            MenuItem::action(cli::menu_label(), ToggleCommandLineTool),
            MenuItem::separator(),
            MenuItem::action("Quit anybox", Quit),
        ],
        disabled: false,
    }]);
}

fn main() {
    gpui_kit::application()
        .with_assets(gpui_kit::assets::Assets)
        .run(|cx| {
            gpui_kit::init(cx);
            dock::set_icon();
            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
            cx.on_action(|_: &Quit, cx| cx.quit());
            cx.on_action(|_: &ToggleCommandLineTool, cx| {
                if let Some(window) = cx.active_window().or_else(|| cx.windows().first().copied()) {
                    let _ = window.update(cx, |_, window, cx| cli::toggle(window, cx, set_menus));
                }
            });
            set_menus(cx);
            // The app ends with its last window; the splash closing leaves the builder's.
            cx.on_window_closed(|cx, _| {
                if cx.windows().is_empty() {
                    cx.quit();
                }
            })
            .detach();
            open_builder(cx);
            splash::show(cx);
            cx.activate(true);
        });
}

/// Opens the builder's window.
fn open_builder(cx: &mut App) {
    let bounds = Bounds::centered(None, size(px(680.), px(820.)), cx);
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        titlebar: Some(TitlebarOptions {
            title: Some("anybox".into()),
            ..TitlebarOptions::default()
        }),
        window_min_size: Some(size(px(560.), px(660.))),
        ..WindowOptions::default()
    };
    gpui_kit::open_window(options, cx, |window, cx| {
        cx.new(|cx| builder::Builder::new(window, cx))
    })
    .expect("failed to open the anybox window");
}
