# Engine-Tier Terminal UIs (`src/tui/`)

Standalone ANSI terminal UIs (`milk-fpsCTRL`, `milk-procCTRL`, `milk-procCTRL-scan`) built at the
engine tier with zero `CLIcore` or `ncurses` dependency.

Both UIs share the lightweight ANSI frame-buffer compatibility header `milkTUI_compat.h` and can be
used when `USE_CLI=OFF`. For the unified multi-panel dashboard covering streams, processes, FPS, and
loops in one window, see [`milk-CTRL`](../cli/overview/README.md) (`src/cli/overview/`).

---

## Directory Organization

```text
src/tui/
├── milkTUI_compat.h ← Shared ANSI frame-buffer rendering primitives
├── fpsCTRL/         ← milk-fpsCTRL parameter tree & compute-unit controller TUI
└── procCTRL/        ← milk-procCTRL & milk-procCTRL-scan process monitor TUI
```

---

## Source Files

### Shared

| File               | Description                                                                   |
| :----------------- | :---------------------------------------------------------------------------- |
| `milkTUI_compat.h` | Shared ANSI frame-buffer TUI rendering primitives (`TUI_printfw`, attributes) |

### `fpsCTRL/` (`milk-fpsCTRL`)

| File                                 | Description                                                                  |
| :----------------------------------- | :--------------------------------------------------------------------------- |
| `fpsCTRL/milk-fpsCTRL.c`             | `milk-fpsCTRL` standalone entry point and CLI option parsing                 |
| `fpsCTRL/fpsCTRL_TUI.c`              | Main FPS scan, event loop, and screen layout orchestration                   |
| `fpsCTRL/fpsCTRL_FPSdisplay.c`       | FPS parameter tree and detail pane rendering                                 |
| `fpsCTRL/fpsCTRL_TUI_process_user_key.c` | Keyboard navigation, parameter editing, and unit action dispatch         |
| `fpsCTRL/fpsCTRL_units.c`            | Compute-unit start/stop polling (`milk-fps-*`), unit state, log & clipboard  |
| `fpsCTRL/level0node_summary.c`       | Top-level FPS node summary line rendering                                    |
| `fpsCTRL/print_nodeinfo.c`           | Selected tree node detail inspection rendering                               |
| `fpsCTRL/fpsCTRL_globals.c`          | Keyword tree storage and global TUI state                                    |
| `fpsCTRL/fpsCTRL_TUIcompat.c`        | Frame-buffer global state storage for `milk-fpsCTRL`                         |
| `fpsCTRL/fpsCTRL_ansi.h`             | ANSI escape sequence and key-decoding helpers for `milk-fpsCTRL`             |

### `procCTRL/` (`milk-procCTRL`, `milk-procCTRL-scan`)

| File                                      | Description                                                             |
| :---------------------------------------- | :---------------------------------------------------------------------- |
| `procCTRL/milk-procCTRL.c`                | `milk-procCTRL` standalone entry point                                  |
| `procCTRL/milk-procCTRL-scan.c`           | `milk-procCTRL-scan` headless processinfo scanner                       |
| `procCTRL/procCTRL_TUI.c`                 | Main `processinfo_CTRLscreen` event loop and lifecycle                  |
| `procCTRL/procCTRL_TUI_render.c`          | Header, process table, and detail view rendering                        |
| `procCTRL/procCTRL_TUI_input.c`           | Keyboard handling, signal dispatch, and process identity verification   |
| `procCTRL/procCTRL_TUI_sort.c`            | Process comparator (`proc_comp`) for sorting columns                    |
| `procCTRL/procCTRL_processinfo_scan.c`    | Shared-memory `proc.*.shm` scanning and liveness tracking               |
| `procCTRL/procCTRL_PIDcollectSystemInfo.c`| `/proc/<pid>` CPU, memory, and thread telemetry collection              |
| `procCTRL/procCTRL_GetCPUloads.c`         | System-wide `/proc/stat` CPU load sampling                              |
| `procCTRL/procCTRL_GetNumberCPUs.c`       | Online CPU count detection                                              |
| `procCTRL/procCTRL_TUIcompat.c`           | Frame-buffer global state storage for `milk-procCTRL`                   |
| `procCTRL/procCTRL_ansi.h`                | ANSI escape sequence and key-decoding helpers for `milk-procCTRL`       |

---

## Standalone Executables

| Executable           | Source                            | Description                                         |
| :------------------- | :-------------------------------- | :-------------------------------------------------- |
| `milk-fpsCTRL`       | `fpsCTRL/milk-fpsCTRL.c`          | FPS parameter tree and compute-unit controller TUI  |
| `milk-procCTRL`      | `procCTRL/milk-procCTRL.c`        | Real-time `processinfo` monitor and control TUI     |
| `milk-procCTRL-scan` | `procCTRL/milk-procCTRL-scan.c`   | Background `processinfo` shared-memory scanner      |

---

## Dependencies & Build Tier

- **Build Tier**: Engine tier (no `CLIcore` or `ncurses` dependency).
- **Internal Dependencies**:
  - `milk-fpsCTRL`: `milkcomputeunit`, `milkfps`, `milkprocessinfo`, `ImageStreamIO`.
  - `milk-procCTRL` / `milk-procCTRL-scan`: `milkprocessinfo`, `ImageStreamIO`.
