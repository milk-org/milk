# streamCTRL

Dedicated TUI for monitoring shared-memory image
streams. Provides detailed per-stream diagnostics
including semaphore states, timing, data rates, and
process associations.

## Purpose

`milk-streamCTRL` scans `/dev/shm/*.im.shm` and
displays real-time telemetry for each stream. Unlike
the broader `milk-CTRL` dashboard, this tool focuses
exclusively on stream diagnostics with higher detail.

## Files

| File                            | Description                       |
| ------------------------------- | --------------------------------- |
| `milk-streamCTRL.c`             | Entry point                       |
| `streamCTRL_TUI.c/.h`           | Main TUI rendering and input loop |
| `streamCTRL_TUIcompat.c/.h`     | Terminal compatibility layer      |
| `streamCTRL_scan.c/.h`          | SHM directory scanner             |
| `streamCTRL_find_streams.c/.h`  | Stream discovery                  |
| `streamCTRL_utilfuncs.c/.h`     | Shared utility functions          |
| `streamCTRL_print_inode.c/.h`   | Inode display helpers             |
| `streamCTRL_print_procpid.c/.h` | PID display helpers               |
| `streamCTRL_print_trace.c/.h`   | Stream trace rendering            |
| `streamCTRL_defs.h`             | Shared constants and struct defs  |
| `streamCTRL_ansi.h`             | ANSI color/style definitions      |
| `mmon_ui.c`                     | Minimal stream monitor mode       |

## Build Tier

CLI tier — requires `USE_CLI=ON` and ncurses.
