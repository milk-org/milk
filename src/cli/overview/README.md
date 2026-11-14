# overview (milk-CTRL)

Unified TUI dashboard for monitoring and controlling milk processes, streams, and FPS entries.
Provides a single-pane-of-glass view of the entire milk runtime environment.

## Purpose

`milk-CTRL` aggregates real-time data from shared memory (ImageStreamIO streams, FPS parameter
structures, and processinfo registry) into a multi-panel, differential double-buffered ANSI
terminal interface with sorting, regex filtering, lineage tracing, and interactive control.

User guide, with the key tables generated from the code: [docs/guide/milk-CTRL.md](../../../docs/guide/milk-CTRL.md).

## Extending milk-CTRL

Most additions are one table entry:

| To add | Edit |
| ------ | ---- |
| A list column | `render/overview_columns_{streams,procs,fps}.c`: one `OV_COLUMN` (header, hit-test, sort, collapse and `<`/`>` follow) |
| A view | `overview_view.c` (`ov_views[]`) and its body renderer in `render/overview_render.c` (`view_render[]`) |
| A sub-tab | `render/overview_render_subtabs.c` (label, renderer) and `input/overview_input_subtab.c` (click, keys) |
| A key | its documentation row in `overview_keymap_*.inc` (help, `-h`, status hints, docs), and its binding row next to the handler (`OV_KEYBIND`, see `input/overview_input_keys.h`) |
| A command-line option | `tools/milkCTRL_opts_parse.c` (pure, unit-tested) and its use in `tools/milkCTRL.c` |

After changing key rows, regenerate the docs tables: `milk-CTRL --keys-markdown` (CTest
`milkctrl_keys_doc` compares them).

## Architecture & Rendering

`milk-CTRL` is a standalone executable with zero dependency on `ncurses` or `CLIcore`. It uses
a custom double-buffered ANSI/VT100 shadow grid engine supporting:

- Differential rendering (`ov_buf_flush_delta`) that emits only modified screen cells.
- 24-bit colour, downgraded at flush time to 256/16 colours (`COLORTERM`/`TERM`) or monochrome
  (`NO_COLOR`, `TERM=dumb`; the selection then shows in reverse video).
- Full mouse reporting (SGR mode 1006) for click-to-focus, tab switching, and panel resizing;
  bracketed paste, so pasted text only reaches text prompts.
- A scan thread publishing models to the UI through an eventfd; the UI never blocks on a scan.
- The terminal is restored on every exit path, including fatal signals; library output is captured
  into the command log instead of corrupting the screen.

Viewing is read-only: the scan maps streams, FPS and processinfo files read-only and never writes
shared memory. CONTROL mode actions open the file they act on at that moment and check first that
it is still the entity on screen (inode, FPS keyword, process start time).

## Views

| Key | View | Content |
| --- | ---- | ------- |
| F2 | `DASH` | Split dashboard overview (streams + processes + FPS) |
| F3 | `STRM` | All active shared memory image streams (ImageStreamIO) |
| F4 | `PROC` | Managed processes and CPU/timing performance (via processinfo) |
| F5 | `FPS` | Function Parameter Structures (FPS) and tunable parameters |
| F6 | `CONN` | Dynamic dataflow lineage graph (producer -> stream -> consumer) |
| F7 | `LOOPS` | Closed feedback loop detection, status, and telemetry |
| F8 | `THEME` | Interactive color theme selector popup (also `Ctrl+T`) |
| `h` | `HELP` | Interactive help and key reference |
| `?` | Jump | Move the cursor to the first entry matching a regex |
| ENTER | Detail | DETAILS sub-tab (DASH), parameter tree / edit (F5), loop filter (LOOPS) |

## Control Mode

Press `c` to toggle Control Mode ON/OFF. Destructive and batch actions ask `y/N` first (the
target is checked again when you answer); a single SIGTERM and run/conf toggles act at once.

- **Streams**: delete the stream file (`DEL` / `^e`, y/N).
- **Processes**: `k` SIGTERM, `K` SIGKILL (y/N), `p` pause toggle (CTRLval 1/0), `^s` step,
  `e` exit, `z` reset loop counter, `C` remove all STOPPED/CRASHED entries (y/N), `^e` remove a
  dead process entry (y/N).
- **FPS**: `r` / `s` start or stop RUN / CONF, `k` SIGTERM, `K` SIGKILL (y/N), `^e` erase (y/N);
  `b` marks an FPS, `a` marks every visible one, and `k K r s` then act on the marks (y/N);
  `ESC` clears them.
- **Parameters**: edit with ENTER in F5 (values are parsed and checked against type and MIN/MAX
  before writing), `o` toggles ON/OFF parameters.

`W` writes a snapshot to `$TMPDIR` (default `/tmp`) and shows its path in the command log.

## Themes & Customization

`milk-CTRL` includes 10 built-in color themes with 24-bit TrueColor palettes:

- `dark` (Default Dark): Slate dark palette
- `night` (Observatory Red): Dark-adapted monochrome red palette
- `accessible` (High-Contrast CVD): Colorblind-friendly palette (Okabe-Ito)
- `light` (Paper Light): Clean light palette for daylight and publications
- `nordic` (Nordic Slate): Cool slate and arctic blue palette
- `dracula` (Dracula): Gothic dark slate with vibrant purple, pink, and cyan
- `solarized-dark` (Solarized Dark): Precision cyan and amber dark palette
- `solarized-light` (Solarized Light): Warm cream and cyan light palette
- `monokai` (Monokai Pro): Warm charcoal with radiant neon accents
- `matrix` (Matrix Phosphor): High-contrast phosphor green on pure black

Press `^T` or `F8` to open the interactive theme selector popup. Navigate themes with `UP` / `DOWN`
arrows with live preview; the popup automatically closes after 1s of inactivity or on `ESC`.
Themes can also be selected at launch using `-T <name>` (e.g. `milk-CTRL -T dracula`) or with
`MILK_CTRL_THEME`. Badges, pills and buttons derive their colours from the active theme.

## Environment

| Variable | Effect |
| -------- | ------ |
| `MILK_SHM_DIR` | Streams and FPS directory (else `/milk/shm`, `/dev/shm`, `/tmp`); `-d` sets it |
| `MILK_PROC_DIR` | processinfo directory; `-d` sets it too |
| `MILK_CTRL_THEME` | Initial theme (`-T` wins) |
| `COLORTERM`, `TERM` | `truecolor`/`24bit`: 24-bit; `*256color`: 256 colours; otherwise 16 |
| `NO_COLOR`, `TERM=dumb` | Monochrome |

## Directory & Source Organization

```text
src/cli/overview/
├── ansi/                    ← Double-buffered ANSI/VT100 shadow grid, delta flush, UTF-8 & key decoder
├── data/                    ← Shared-memory scan (`milkctrl_model`), graph lineage, loop detection & sorting
├── input/                   ← Keyboard/mouse handlers, hit-testing, control actions, modals & FPS editor
├── render/                  ← Panel, table column, graph, loop, help, inspector & theme rendering
├── tools/                   ← `milk-CTRL`, `milk-stream-info`, `milk-procinfo-info` & `milk-stream-graph`
├── tests/                   ← Keymap documentation check (`check_keys_doc.sh`) and CTest definitions
├── overview_cmdlog.c        ← Ring buffer for in-TUI command and status messages
├── overview_keymap.c        ← Keybinding catalog (`overview_keymap_*.inc`) and `--keys-markdown` export
├── overview_layout.c        ← Split-panel geometry, resizing, and responsive layout calculation
└── overview_view.c          ← Top-level view (`DASH`, `STRM`, `PROC`, `FPS`, `CONN`, `LOOPS`) descriptors
```

| Subdirectory | Role & Key Files |
| :--- | :--- |
| `ansi/` | Terminal raw mode (`overview_ansi_term.c`), cell shadow buffer (`overview_ansi_buf.c`), differential flush (`overview_ansi_delta.c`), UTF-8 width (`overview_ansi_utf8.c`), and SGR mouse / key parsing (`overview_ansi_key.c`) |
| `data/` | Static library `milkctrl_model` (`milkctrl_model.cmake`): background scan thread (`overview_scan.c`), SHM entity scanners (`overview_data_scan_*.c`), producer/consumer adjacency & lineage (`overview_data_adj.c`, `stream_graph*.c`), closed-loop detection (`overview_data_loops*.c`), `/proc` & `perf_event` sampling (`overview_data_sys*.c`), filtering (`overview_data_filter.c`), and sorting (`overview_data_sort*.c`) |
| `input/` | Key dispatch (`overview_input_keys.c`), mouse click/drag/resize (`overview_input_mouse*.c`), column hit-testing (`overview_input_hittest*.c`), navigation (`overview_input_nav*.c`), control-mode actions & confirmations (`overview_ctrl*.c`, `overview_input_confirm.c`), and inline FPS parameter editing (`overview_fps_edit.c`) |
| `render/` | Frame compositor (`overview_render.c`), column definitions (`overview_columns_*.c`), view renderers (`overview_render_{streams,procs,fps,graph,loops,insp,help}*.c`), detail sub-tabs (`overview_render_detail*.c`), and 10 truecolor themes (`overview_theme*.c`) |
| `tools/` | Entry points and CLI option parsers for `milk-CTRL` (`milkCTRL.c`, `milkCTRL_opts*.c`), `milk-stream-info` (`milk-stream-info*.c`), `milk-procinfo-info` (`milk-procinfo-info.c`), and `milk-stream-graph` (`milk-stream-graph*.c`) |

## Standalone Executables

| Executable | Source | Description |
| :--- | :--- | :--- |
| `milk-CTRL` | `tools/milkCTRL.c` | Unified multi-panel TUI dashboard (`milk-streamCTRL` symlink opens F3) |
| `milk-stream-info` | `tools/milk-stream-info.c` | Inspect stream metadata, holders, and producer/consumer links |
| `milk-procinfo-info` | `tools/milk-procinfo-info.c` | Inspect `processinfo` timing, CPU, and stream links |
| `milk-stream-graph` | `tools/milk-stream-graph.c` | Terminal and DOT/JSON dataflow graph exporter |

## Build Requirements

- Standalone executable target: `milk-CTRL`
- Libraries: `ImageStreamIO`, `milkprocessinfo`, `milkfps`, `m`, `rt`, `pthread`
- Dependencies: Independent of `CLIcore`, `ncurses`, or readline.
