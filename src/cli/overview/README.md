# overview (milk-CTRL)

Unified TUI dashboard for monitoring and controlling milk processes, streams, and FPS entries.
Provides a single-pane-of-glass view of the entire milk runtime environment.

## Purpose

`milk-CTRL` aggregates real-time data from shared memory (ImageStreamIO streams, FPS parameter
structures, and processinfo registry) into a multi-panel, differential double-buffered ANSI
terminal interface with sorting, regex filtering, lineage tracing, and interactive control.

## Architecture & Rendering

`milk-CTRL` is a standalone executable with zero dependency on `ncurses` or `CLIcore`. It uses
a custom double-buffered ANSI/VT100 shadow grid engine supporting:
- Differential rendering (`ov_buf_flush_delta`) that emits only modified screen cells.
- 24-bit TrueColor and 256-color palette detection and ANSI SGR batching.
- Full mouse reporting (SGR mode 1006) for click-to-focus, tab switching, and panel resizing.
- Real-time event-driven updates driven by `poll()` and condition variable notifications.

## Views

| Key | View | Content |
| --- | ---- | ------- |
| F1 | `STRM` | All active shared memory image streams (ImageStreamIO) |
| F2 | `DASH` | Split dashboard overview (streams + processes + FPS) |
| F3 | `PROC` | Managed processes and CPU/timing performance (via processinfo) |
| F4 | `LOOP` | Sequencer and execution loops |
| F5 | `FPS` | Function Parameter Structures (FPS) and tunable parameters |
| F6 | `GRAPH`| Dynamic dataflow lineage graph (producer -> stream -> consumer) |
| F7 | `LOG` | Command execution history and system events log |
| F8 | `HELP` | Comprehensive interactive help and keybinding reference |
| ENTER | Detail | Toggle detailed inspection pane / parameter edit mode |

## Control Mode

Press `c` to toggle Control Mode ON/OFF:
- **Streams**: Delete shared-memory files (`DEL` / `CTRL+e`).
- **Processes**: Send signals (`k`: SIGTERM, `K`: SIGKILL, `p`: pause/resume,
  `^s`: step, `z`: reset).
- **FPS**: Manage tmux sessions (`k`/`K`) and toggle execution loops (`r`: run, `s`: config).
- **Parameters**: Edit scalar, string, timespec, and boolean parameters inline with ENTER.

## Build Requirements

- Standalone executable target: `milk-CTRL`
- Libraries: `ImageStreamIO`, `milkprocessinfo`, `milkfps`, `m`, `rt`, `pthread`
- Dependencies: Independent of `CLIcore`, `ncurses`, or readline.
