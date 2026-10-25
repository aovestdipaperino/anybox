//! The window: the build's inputs, the icon to drop, and a bar for each phase of the build.

use std::path::PathBuf;

use anybox_core::progress::{Event, Phase};
use futures::StreamExt as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::progress::Progress;
use gpui_kit::component::{ActiveTheme as _, Disableable as _, h_flex, v_flex};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::form::{Phases, Stage, Target, classify, disk_image, title_name};

/// The runtime when the window runs from a checkout rather than from `Anybox.app`: the release
/// build of this workspace's `runtime` crate, whose build leaves `signatures.txt` beside
/// `libruntime.a`. The app carries its own in `Contents/Resources/runtime`.
const CHECKOUT_RUNTIME: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/release");

/// The notes kept for the log; a verbose build writes thousands.
const MAX_NOTES: usize = 2000;

/// How a build went, or that none has run.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    Idle,
    Running,
    Done(PathBuf),
    Failed(String),
}

/// What the build's thread sends the window.
#[derive(Debug)]
enum Message {
    Event(Event),
    Ended(Result<(), String>),
}

/// The window's view.
pub(crate) struct Builder {
    iso: Entity<InputState>,
    name: Entity<InputState>,
    icon: Option<PathBuf>,
    /// The name the last disc suggested, so a name field still holding it follows the next disc.
    suggested_name: String,
    phases: Phases,
    notes: Vec<String>,
    log: ScrollHandle,
    status: Status,
    _build: Task<()>,
}

/// A text field holding `value`, showing `placeholder` when empty.
fn field(
    placeholder: &'static str,
    value: String,
    window: &mut Window,
    cx: &mut Context<Builder>,
) -> Entity<InputState> {
    cx.new(|cx| {
        InputState::new(window, cx)
            .placeholder(placeholder)
            .default_value(value)
    })
}

/// The text in `input`.
fn text(input: &Entity<InputState>, cx: &App) -> String {
    input.read(cx).value().trim().to_owned()
}

/// Replaces the text in `input` with `value`.
fn set(input: &Entity<InputState>, value: &str, window: &mut Window, cx: &mut Context<Builder>) {
    let value = SharedString::from(value.to_owned());
    input.update(cx, |state, cx| state.set_value(value, window, cx));
}

impl Builder {
    pub(crate) fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        Self {
            iso: field("the disc image, an .iso", String::new(), window, cx),
            name: field(
                "the title's name, shown by macOS",
                String::new(),
                window,
                cx,
            ),
            icon: None,
            suggested_name: String::new(),
            phases: Phases::default(),
            notes: Vec::new(),
            log: ScrollHandle::new(),
            status: Status::Idle,
            _build: Task::ready(()),
        }
    }

    /// Puts `path` where `target` says.
    fn take(&mut self, target: Target, path: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        let shown = path.to_string_lossy().into_owned();
        match target {
            Target::Disc => {
                set(&self.iso, &shown, window, cx);
                let next = title_name(&path);
                let have = text(&self.name, cx);
                if have.is_empty() || have == self.suggested_name {
                    set(&self.name, &next, window, cx);
                }
                self.suggested_name = next;
            }
            Target::Icon => self.icon = Some(path),
        }
        cx.notify();
    }

    /// Files dropped anywhere but the icon: each goes where its kind belongs.
    fn dropped(&mut self, paths: &ExternalPaths, window: &mut Window, cx: &mut Context<Self>) {
        for path in paths.paths() {
            if let Some(target) = classify(path) {
                self.take(target, path.clone(), window, cx);
            }
        }
    }

    /// Asks for a file for `target`.
    fn choose(&mut self, target: Target, window: &mut Window, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: None,
        });
        cx.spawn_in(window, async move |this, cx| {
            if let Ok(Ok(Some(paths))) = paths.await
                && let Some(path) = paths.into_iter().next()
            {
                let _ = this.update_in(cx, |this, window, cx| this.take(target, path, window, cx));
            }
        })
        .detach();
    }

    /// The build the fields describe, or what is missing.
    fn request(&self, cx: &App) -> Result<anybox_core::Request, String> {
        let iso = text(&self.iso, cx);
        if iso.is_empty() {
            return Err("drop a disc image on the window, or choose one".to_owned());
        }
        if !std::path::Path::new(&iso).is_file() {
            return Err(format!("{iso} is not a file"));
        }
        let name = text(&self.name, cx);
        if name.is_empty() {
            return Err("the title needs a name".to_owned());
        }
        let runtime = anybox_core::find_runtime(None)
            .unwrap_or_else(|| std::path::PathBuf::from(CHECKOUT_RUNTIME));
        let runtime = runtime.canonicalize().unwrap_or(runtime);
        if !runtime.join("libruntime.a").is_file() {
            return Err(format!(
                "no runtime in {}: build it with `cargo build --release -p runtime`",
                runtime.display()
            ));
        }
        // The window builds in C: without clang it says how to install it before starting.
        anybox_core::toolchain::check(false)?;
        let dmg = disk_image(std::path::Path::new(&iso));
        Ok(anybox_core::Request {
            iso: iso.into(),
            icon: self
                .icon
                .as_ref()
                .map(|icon| icon.to_string_lossy().into_owned())
                .unwrap_or_default(),
            name,
            dmg,
            runtime,
            // A temporary folder, made for the build and removed after it.
            work: None,
            pack_level: None,
            opt: None,
            // The default, C for clang; the command line's `--rust` asks for Rust.
            rust: false,
        })
    }

    /// Starts the build on a thread of its own; its events come back to the window as they happen.
    fn start(&mut self, cx: &mut Context<Self>) {
        if self.status == Status::Running {
            return;
        }
        let request = match self.request(cx) {
            Ok(request) => request,
            Err(message) => {
                self.status = Status::Failed(message);
                cx.notify();
                return;
            }
        };
        self.phases = Phases::default();
        self.notes.clear();
        self.status = Status::Running;
        let dmg = request.dmg.clone();
        let (sender, mut receiver) = futures::channel::mpsc::unbounded();
        std::thread::spawn(move || {
            let reporter = |event| {
                let _ = sender.unbounded_send(Message::Event(event));
            };
            let ended = anybox_core::build(&request, &reporter);
            let _ = sender.unbounded_send(Message::Ended(ended));
        });
        self._build = cx.spawn(async move |this, cx| {
            while let Some(message) = receiver.next().await {
                let dmg = dmg.clone();
                let taken = this.update(cx, |this, cx| {
                    match message {
                        Message::Event(Event::Note(line)) => {
                            if this.notes.len() == MAX_NOTES {
                                this.notes.remove(0);
                            }
                            this.notes.push(line);
                            this.log.scroll_to_bottom();
                        }
                        Message::Event(event) => this.phases.apply(&event),
                        Message::Ended(Ok(())) => this.status = Status::Done(dmg),
                        Message::Ended(Err(message)) => this.status = Status::Failed(message),
                    }
                    cx.notify();
                });
                if taken.is_err() {
                    break;
                }
            }
        });
        cx.notify();
    }

    /// Shows the written disk image in the Finder.
    fn reveal(&self, cx: &mut Context<Self>) {
        if let Status::Done(dmg) = &self.status {
            cx.reveal_path(dmg);
        }
    }

    fn render_icon(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let inside = match &self.icon {
            Some(icon) => img(icon.clone())
                .size_full()
                .object_fit(ObjectFit::Contain)
                .into_any_element(),
            None => v_flex()
                .items_center()
                .gap_1()
                .px_3()
                .text_xs()
                .text_center()
                .text_color(theme.muted_foreground)
                .child("Drop the icon here")
                .child("or click to choose")
                .child("(else the title's own image)")
                .into_any_element(),
        };
        v_flex()
            .gap_2()
            .items_center()
            .child(
                div()
                    .id("icon")
                    .size(px(148.))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_center()
                    .overflow_hidden()
                    .rounded(theme.radius_lg)
                    .border_2()
                    .border_color(theme.border)
                    .bg(theme.muted)
                    .cursor_pointer()
                    .drag_over::<ExternalPaths>(|style, _, _, cx| {
                        style.border_color(cx.theme().primary)
                    })
                    .on_drop(cx.listener(|this, paths: &ExternalPaths, window, cx| {
                        if let Some(path) = paths.paths().first() {
                            this.take(Target::Icon, path.clone(), window, cx);
                        }
                    }))
                    .on_click(
                        cx.listener(|this, _, window, cx| this.choose(Target::Icon, window, cx)),
                    )
                    .child(inside),
            )
            .when(self.icon.is_some(), |this| {
                this.child(
                    Button::new("clear-icon")
                        .ghost()
                        .compact()
                        .label("Use the title's image")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.icon = None;
                            cx.notify();
                        })),
                )
            })
    }

    fn render_field(
        &self,
        label: &'static str,
        input: &Entity<InputState>,
        choose: Option<Target>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        v_flex()
            .gap_1()
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(label),
            )
            .child(
                h_flex()
                    .gap_2()
                    .child(div().flex_1().child(Input::new(input)))
                    .when_some(choose, |this, target| {
                        this.child(Button::new(label).outline().label("Choose…").on_click(
                            cx.listener(move |this, _, window, cx| {
                                this.choose(target, window, cx);
                            }),
                        ))
                    }),
            )
    }

    fn render_phase(&self, index: usize, phase: Phase, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let stage = self.phases.stage(phase);
        let mut name = phase.name().to_owned();
        if let Some(first) = name.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        let color = match stage {
            Stage::Finished(_) => theme.success,
            _ => theme.progress_bar,
        };
        v_flex()
            .gap_1()
            .child(
                h_flex()
                    .justify_between()
                    .text_sm()
                    .child(div().font_weight(FontWeight::MEDIUM).child(name))
                    .child(
                        div()
                            .text_color(theme.muted_foreground)
                            .child(stage.caption(phase)),
                    ),
            )
            .child(
                Progress::new(("phase", index))
                    .value(stage.fill())
                    .loading(stage.indeterminate())
                    .color(color),
            )
    }

    fn render_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let (text, color) = match &self.status {
            Status::Idle => (
                "Drop a disc image or an icon anywhere in the window.".to_owned(),
                theme.muted_foreground,
            ),
            Status::Running => ("Building…".to_owned(), theme.foreground),
            Status::Done(dmg) => (format!("Wrote {}", dmg.display()), theme.success),
            Status::Failed(message) => (message.clone(), theme.danger),
        };
        div().flex_1().text_sm().text_color(color).child(text)
    }
}

impl Render for Builder {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let running = self.status == Status::Running;
        let done = matches!(self.status, Status::Done(_));
        let theme = cx.theme();
        let (background, foreground, border) = (theme.background, theme.foreground, theme.border);
        let phases: Vec<_> = Phase::ALL
            .iter()
            .enumerate()
            .map(|(index, phase)| self.render_phase(index, *phase, cx).into_any_element())
            .collect();
        v_flex()
            .id("anybox")
            .size_full()
            .p_5()
            .gap_4()
            .bg(background)
            .text_color(foreground)
            .on_drop(cx.listener(Self::dropped))
            .child(
                h_flex()
                    .gap_5()
                    .items_start()
                    .child(self.render_icon(cx))
                    .child(
                        v_flex()
                            .flex_1()
                            .gap_3()
                            .child(self.render_field(
                                "Disc image",
                                &self.iso,
                                Some(Target::Disc),
                                cx,
                            ))
                            .child(self.render_field("Name", &self.name, None, cx)),
                    ),
            )
            .child(
                h_flex()
                    .gap_3()
                    .items_center()
                    .child(self.render_status(cx))
                    .when(done, |this| {
                        this.child(
                            Button::new("reveal")
                                .outline()
                                .label("Show in Finder")
                                .on_click(cx.listener(|this, _, _, cx| this.reveal(cx))),
                        )
                    })
                    .child(
                        Button::new("build")
                            .primary()
                            .label("Build")
                            .loading(running)
                            .disabled(running)
                            .on_click(cx.listener(|this, _, _, cx| this.start(cx))),
                    ),
            )
            .child(v_flex().gap_3().children(phases))
            .child(
                div()
                    .id("log")
                    .flex_1()
                    .min_h(px(80.))
                    .p_2()
                    .rounded(cx.theme().radius)
                    .border_1()
                    .border_color(border)
                    .overflow_y_scroll()
                    .track_scroll(&self.log)
                    .font_family("Menlo")
                    .text_xs()
                    .children(self.notes.iter().map(|line| div().child(line.clone()))),
            )
    }
}
