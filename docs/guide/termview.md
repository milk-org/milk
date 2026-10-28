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

- When the terminal cannot keep up, each screen update carries what it can show in about 50 ms (40
  ms above `-f 25`), so keys still get an answer within that time. The rest of the image follows in
  the next updates.
- Over ssh, writes rarely block: ssh buffers up to 2 MB, which a slow link takes seconds to send. So
  after each update the viewer asks the terminal where its cursor is (`CSI 6n`), having put the
  hidden cursor in a column that names the question. The terminal answers once it has shown
  everything sent before, so the viewer knows how much output is still on the way and how fast the
  link shows it. It keeps the link's round trip plus about 50 ms of image in flight: the link stays
  busy, and a key shows within about the round trip plus 50 ms. The menus and the status bar are
  never held back.
- Now and then (every 5 to 40 s, or sooner when the answers show that the round trip grew or
  shrank) the image waits for a moment while the queue drains, so that an answer times the bare
  round trip.
- Delays that vary a lot on the way back (by 150 ms or more) look like a queue: the image then goes
  slower, while keys stay prompt.
- While answers are late, for instance while the link stalls, the image waits for them, a little
  of it going through now and then (8 KB after 2 s, then less and less often). Once the link is
  back, that shows first, then all is as before. A terminal that stops answering for good keeps
  the image at that pace.
- Until the terminal answers the first question, at most 64 KB of image goes out. A terminal that
  does not answer within 2 s is not asked again (unless an answer comes later after all), and the
  viewer sends as fast as writes go through, as above.
- `q`, Ctrl-C and Ctrl-Z first read the answers still on the way, which would otherwise reach the
  shell as typed text. This lasts as long as the output queued ahead of them, and up to 3 s after
  the last answer, 30 s at most. With a terminal that does not answer, leaving in the first 3 s
  waits until then. Ctrl-C during the wait ends it, at the earliest a second (or two round trips)
  after it began; a second signal ends it at once. Answers still on the way then reach the shell.
  An answer that comes later than all that (more than 3 s, for the first one) reaches the shell
  too.
- Keys typed while the viewer waits are dropped, with a note once the shell is back. So is the rest
  of a line begun then, until Enter, Ctrl-U, a pause in typing of 1.5 s, or 10 s.
- Behind tmux or screen, the multiplexer answers the questions, so they time the path to the
  multiplexer, not to your terminal. Over ssh, run the multiplexer on your side of the link, or the
  viewer outside it.
- OpenSSH 9.5 and later obscure keystroke timing by default, and the answers keep that on while
  you only watch: on a fast link (10 MB/s) it costs about 10-15% of the image. `ssh -o
  ObscureKeystrokeTiming=no` avoids it.
- Low-bandwidth mode (`b`, or Stream > Low-bandwidth) leaves a cell alone while its colour stays
  within a few levels of the exact one, and redraws a few rows exactly at each new frame. On a
  noisy but steady scene this cuts the output about ten-fold. When the stream stops, the image
  becomes exact.
- Fewer colours send fewer bytes: View > Colors, or `--colors 256`. Together with low-bandwidth mode
  they let a slow link show more frames. A lower refresh rate (`-f 5`) gives each frame time to
  arrive whole once the link can carry one per period; it does not make keys faster.

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
