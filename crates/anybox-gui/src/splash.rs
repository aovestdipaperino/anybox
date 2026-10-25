//! The splash: the bare anybox logo, in a window without a title bar that floats over the
//! builder's window for its first two seconds.

use std::sync::Arc;
use std::time::Duration;

use gpui_kit::*;

/// How long the logo shows.
const SHOWN_FOR: Duration = Duration::from_secs(2);

/// The logo, white and green on black, the same one the runtime shows as a title starts.
const LOGO: &[u8] = include_bytes!("../../runtime/assets/logo.png");

/// The window's side, in points.
const SIDE: f32 = 360.;

/// The splash window's view.
struct Splash {
    logo: Arc<Image>,
}

impl Render for Splash {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .bg(black())
            .child(img(self.logo.clone()).size_full())
    }
}

/// Shows the logo above every other window for [`SHOWN_FOR`], then closes it.
pub(crate) fn show(cx: &mut App) {
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
            None,
            size(px(SIDE), px(SIDE)),
            cx,
        ))),
        titlebar: None,
        kind: WindowKind::PopUp,
        is_movable: false,
        is_resizable: false,
        is_minimizable: false,
        ..WindowOptions::default()
    };
    let logo = Arc::new(Image::from_bytes(ImageFormat::Png, LOGO.to_vec()));
    let Ok(splash) = cx.open_window(options, |_, cx| cx.new(|_| Splash { logo })) else {
        return;
    };
    cx.spawn(async move |cx| {
        cx.background_executor().timer(SHOWN_FOR).await;
        cx.update(|cx| {
            let _ = splash.update(cx, |_, window, _| window.remove_window());
        });
    })
    .detach();
}
