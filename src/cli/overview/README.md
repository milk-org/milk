# overview (milk-CTRL)

Unified TUI dashboard for monitoring and controlling
milk processes, streams, and FPS entries. Provides
a single-pane-of-glass view of the entire milk
runtime environment.

## Purpose

`milk-CTRL` aggregates real-time data from shared memory
(streams, FPS, processinfo) into a multi-panel ncurses
interface with sorting, filtering, search, and
interactive control.

## Panels

| Panel   | Key   | Content                             |
| ------- | ----- | ----------------------------------- |
| STREAMS | F1    | All active SHM image streams        |
| PROCS   | F2    | Running processes (via processinfo) |
| FPS     | F3    | FPS parameter structures            |
| Detail  | Enter | Expanded view of selected item      |

## Files

| File                   | Description                         |
| ---------------------- | ----------------------------------- |
| `milk-CTRL.c`          | Entry point and main loop           |
| `overview_data.c/.h`   | Data scanning and model updates     |
| `overview_render.c`    | Panel rendering (largest file)      |
| `overview_input.c`     | Keyboard input handler              |
| `overview_ctrl.c/.h`   | Control actions (kill, spawn)       |
| `overview_scan.c`      | SHM directory scanner               |
| `overview_layout.c/.h` | Terminal layout calculations        |
| `overview_defs.h`      | Shared constants and enums          |
| `overview_ansi.h`      | ANSI color/style definitions        |
| `overview_theme.h`     | Theme configuration                 |
| `milk-stream-graph.c`  | `milk-stream-graph` standalone tool |
| `stream_graph.c/.h`    | BFS-based stream dependency graph   |

## Build Tier

CLI tier — requires `USE_CLI=ON` and ncurses.
