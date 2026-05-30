---
tags:
  - fps
  - tui
  - reference
---

# fpsCTRL Reference

`milk-fpsCTRL` is the primary interactive TUI tool for
managing Function Parameter Structure (FPS) instances.
It provides real-time control of compute units:
starting/stopping conf and run loops, editing
parameters, and monitoring FPS state.

See also: [FPS](fps.md) ·
[FPS Standalone Modes](FPS_Standalone_CMD_Modes.md) ·
[FPS Sequencer](sequencer.md)

## Command-Line Options

```
milk-fpsCTRL [OPTIONS] [fpsnamemask]
```

| Option             | Description                             |
| ------------------ | --------------------------------------- |
| `-m`, `--match`    | Enable match mode (filter by mask)      |
| `-n`, `--name`     | Set FPS name filter mask                |
| `-f`, `--fifo`     | Specify input FIFO file                 |
| `-q`, `--quiet`    | Suppress TUI output (`MILK_TUIPRINT_NONE=1`) |
| `-s`, `--stdio`    | Line-by-line stdio mode                 |
| `-h`, `--help`     | Print usage and exit                    |
| `-h1`              | Print one-line description and exit     |

The positional argument `fpsnamemask` sets the name
filter (default: `_ALL`).

## Environment Variables

| Variable                  | Effect                              |
| ------------------------- | ----------------------------------- |
| `MILK_FPS_LOGFILE`        | Path to FPS log file                |
| `FPS_FILTSTRING_NAME`     | Filter FPS entries by name          |
| `FPS_FILTSTRING_KEYWORD`  | Filter by keyword                   |
| `FPS_FILTSTRING_CALLFUNC` | Filter by call function in source   |
| `FPS_FILTSTRING_MODULE`   | Filter by source code module        |

## Display Modes

| Mode | Name       | Access           | Description             |
| ---- | ---------- | ---------------- | ----------------------- |
| 1    | Help       | `h`              | Built-in help screen    |
| 2    | FPS Log    | `?`              | FPS event log viewer    |
| 3    | Control    | `F2`, Ctrl+Right | Main FPS control view   |
| 4    | Scheduler  | `F3`, Ctrl+Left  | FPS sequencer display   |

The default display mode at startup is **Control**
(mode 3).

## Keyboard Shortcuts

### Navigation

| Key            | Action                                 |
| -------------- | -------------------------------------- |
| Up / Down      | Move selection up/down in FPS list     |
| PgUp / PgDn    | Page up/down through FPS list          |
| Left / Right   | Navigate parameter columns / levels    |
| Tab            | Toggle between FPS list and parameters |

### Mode Switching

| Key          | Action                                   |
| ------------ | ---------------------------------------- |
| `h`          | Show help screen (mode 1)                |
| `?`          | Show FPS log (mode 2)                    |
| `F2`         | Switch to control mode (mode 3)          |
| `F3`         | Switch to scheduler mode (mode 4)        |
| Ctrl+Right   | Next display mode                        |
| Ctrl+Left    | Previous display mode                    |
| `ESC`        | Return to control mode                   |

### FPS Control

| Key  | Action                                      |
| ---- | ------------------------------------------- |
| `T`  | Initialize tmux session for selected FPS    |
| `t`  | Kill tmux session for selected FPS          |
| `O`  | Start conf process (confstart)              |
| `C`  | Start conf process (legacy alias)           |
| `c`  | Stop (kill) conf process                    |
| `R`  | Start run process (runstart)                |
| `r`  | Stop run process (legacy runstop)           |
| `u`  | Update conf process                         |
| `E`  | Stop conf+run, erase FPS, kill tmux         |

### Parameter Editing

| Key      | Action                                    |
| -------- | ----------------------------------------- |
| `Enter`  | Edit selected parameter                   |
| `Space`  | Toggle ON/OFF parameter or select         |

### Display and Search

| Key  | Action                                      |
| ---- | ------------------------------------------- |
| `/`  | Enter search mode (filter by text)          |
| `S`  | Toggle sort mode                            |
| `]`  | Cycle to next sort column                   |
| `[`  | Toggle sort direction                       |
| `l`  | List all parameters for selected FPS        |
| `y`  | Yank FPS name to tmux paste buffer          |

### General

| Key        | Action                                  |
| ---------- | --------------------------------------- |
| `s`        | Rescan FPS instances                    |
| `x`        | Exit fpsCTRL                            |
| Ctrl+C     | Exit (signal handler)                   |

---

← [Documentation Index](index.md)
