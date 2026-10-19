---
tags:
  - streams
  - tui
  - cli
---

# Stream Viewer (`milk-termview`)

`milk-termview` shows a shared-memory stream as an image in the terminal. It maps the stream
read-only, so it never writes to it, and it only redraws the cells that changed. That keeps it
cheap enough to leave running next to a real-time loop, locally or over ssh.

```bash
milk-termview wfsim
milk-termview -c inferno -s log -r 1 dmdisp
milk-termview /milk/shm/aol0_wfsim.im.shm
```

## Screen

```text
 Cmap inferno ▼  Scale log ▼  Range 1-99% ▼  View ▼  Stream ▼          ? Help  q Quit
           ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄            ██ 1000
           ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄            ██ 750
           ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  image  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄            ██ 500
           ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄            ██ 250
           ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄            ██ 0
 ● LIVE  aol0_wfsim   1000.0 Hz  [0 : 1000] 1-99%  240x240 f32  CPU 1.2%  UI 20/20 fps
```

- **Menubar**: each menu button shows its current value. Click a button, or press F2 (Cmap),
  F3 (Scale), F4 (Range), F5 (View), F6 (Stream). Tab or F9 opens the last menu used.
- **Image**: each character cell shows two image samples (half blocks). The colorbar on the right
  is labelled with the display range.
- **Status bar**: the stream state, then the name, rate, range, geometry, CPU, refresh rate, zoom
  and active modes. On a narrow terminal the last segments are dropped first.

Stream states:

| State | Meaning |
|-------|---------|
| `LIVE` | New frames keep coming |
| `STALE 12s` | No new frame for over 1 s (or 5 frame periods); the age of the last one |
| `PAUSED` | Space froze the image |
| `GONE` | The stream was removed or replaced; the last image stays until a valid stream is back |

## Menus

| Menu | Items |
|------|-------|
| Cmap | 17 colormaps, each with a preview strip |
| Scale | linear, sqrt, log, log_strong, log_extreme, asinh, squared, cubed |
| Range | min-max or a percentile range (choosing one unlocks the range), Lock range, Set min / max... |
| View | Fit to window, Zoom in/out, Centre, Flip horizontally/vertically, Colors: auto/24-bit/256/16 |
| Stream | Pause, UI refresh rate (1-60 fps), Range smoothing, Low-bandwidth, Quit |

Moving through a menu (Up/Down, Home/End, PgUp/PgDn, the wheel, or the pointer) previews the
highlighted choice at once. Enter or a click keeps it; Esc or a click outside the menu goes back to
what was there before. Space applies an item and leaves the menu open. Left/Right move to the
neighbouring menu.

## Keys and mouse

| Keys | Action |
|------|--------|
| F2-F6 | Open the Cmap, Scale, Range, View, Stream menu |
| Tab, F9 | Open the last menu used |
| F1, ? | Key help |
| Arrows | Pan the view |
| `+` `=` / `-` `_` | Zoom in / out |
| `0` | Reset zoom and pan |
| `c` `C`, `s` `S`, `r` `R` | Next / previous colormap, scaling, range mode |
| `l` | Lock or unlock the current range |
| `m` / `M` | Type the min / max (locks the range) |
| `h` / `v` | Flip horizontally / vertically |
| Space | Pause: freeze the image (zoom and pan still work) |
| `b` | Low-bandwidth mode |
| `<` `,` / `>` `.` | Lower / raise the UI refresh rate |
| Ctrl-L | Repaint the screen |
| Ctrl-Z | Suspend |
| `q`, Ctrl-C | Quit |
| Wheel | Zoom at the pointer |
| Left drag | Pan |
| Right drag | Zoom to the selected box |

## Display range

The range is the colour scale's min and max. In an automatic mode (min-max or a percentile range)
it follows the image, smoothed so that noise does not make it flicker: it widens at once and
narrows gradually. Range smoothing can be turned off in the Stream menu.

`m` or `M` (or Range > Set min / max...) opens the range editor on the status row. Both fields are
pre-filled; the first key typed replaces a field, Tab moves to the other one, and Enter locks the
range once both values are numbers with min < max. `l` locks or unlocks the range shown.

## Slow terminals and ssh

- When the terminal cannot keep up, each screen update carries what it can show in about one frame
  period, so keys still get an answer within that time. The rest of the image follows in the next
  updates.
- Low-bandwidth mode (`b`, or Stream > Low-bandwidth) leaves a cell alone while its colour stays
  within a few levels of the exact one, and redraws a few rows exactly at each new frame. On a
  noisy but steady scene this cuts the output about ten-fold. When the stream stops, the image
  becomes exact.
- Fewer colours send fewer bytes: View > Colors, or `--colors 256`.
- Over ssh, writes rarely block (ssh buffers up to 2 MB), so a slow link can still delay keys behind
  queued output. Low-bandwidth mode, 256 colours and a lower refresh rate (`-f 5`) help there.

## Streams that change

Once a second the viewer checks the stream file. If it was removed or recreated (milk recreates a
stream by removing the file and creating a new one), the status shows `GONE`, and the viewer
switches to the new stream as soon as it is valid. A new size resets the zoom; the colormap,
scaling and range settings are kept. A stream truncated in place is dropped before it is read.

## Options

| Option | Meaning |
|--------|---------|
| `-c, --cmap NAME` | Initial colormap |
| `-s, --scale NAME` | Initial scaling |
| `-r, --range NAME` | Initial range: minmax, 0.1, 0.5, 1, 5, 10, 15, 20 (lower percentile) |
| `--min V --max V` | Lock the display range to [min, max] |
| `-f, --fps N` | UI refresh rate, 1-120 (20) |
| `--colors MODE` | auto, truecolor, 256, 16 (auto) |

Colour depth in `auto` mode: `COLORTERM=truecolor` or `24bit` gives 24-bit colour. Otherwise
`TERM` decides: known 24-bit terminals (kitty, foot, alacritty, wezterm, ...) get 24-bit, xterm-like
terminals 256, the rest 16. ssh forwards `TERM` but usually not `COLORTERM`, so pass `--colors` (or
pick View > Colors) when the colours look coarse.
