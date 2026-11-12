# Changelog

## [0.2.0] - 2026-11-12

### Added
- A Bluetooth controller that speaks the Switch protocol (8BitDo pads, for one) works as an Xbox
  pad: its buttons map onto the Xbox layout, rumble reaches its motors, and the title bar shows a
  controller icon while it is connected. Its Capture button takes a perma snapshot and its Home
  button goes back to it.
- `ANYXBOX_FPS=1` prints the frame rate every two seconds, with how long each of the title's and
  the renderer's threads took over a frame and how long it waited.
- `anybox --census ISO` counts what a title's lifted code is made of.

### Changed
- A snapshot survives rebuilding the title with a newer anybox: it resumes as long as the code
  each thread stood in can still be resumed at the same place, and otherwise is refused with the
  reason. Snapshots made by 0.1.0 are not resumed.
- Faster rendering: vertex programs run eight vertices at a time, vertices are uploaded a batch at
  a time, and taking in the title's push buffer costs half as much.
- Loops in which a title only waits for something to change let other threads run, or sleep,
  instead of spinning (`ANYXBOX_SPIN=0` keeps the spinning).
- Lifted titles spend less time in the CPU: code reached only through data no longer cuts hot
  loops into pieces (`ANYBOX_WEAK_ROOTS=0` turns this off).

## [0.1.0] - 2026-10-25

First public release.
