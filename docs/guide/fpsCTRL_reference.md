---
tags:
  - fps
  - tui
  - reference
---

# fpsCTRL Reference

`milk-fpsCTRL` is the primary interactive TUI tool for managing Function Parameter Structure (FPS)
instances. It provides real-time control of compute units: starting/stopping conf and run loops,
editing parameters, and monitoring FPS state.

Starts and stops run in the background (the `milk-fps-*` tools), so the screen keeps refreshing
while a unit starts or stops. Their result shows in the line above the status bar. The row of each
FPS shows its conf and run process IDs, green while the process runs, yellow while it starts or
stops, red when it failed; verbose mode (`v`) adds the state, uptime and last exit of each.

See also: [FPS](fps.md)

## Command-Line Options

```bash
milk-fpsCTRL [OPTIONS] [fpsnamemask]
```

| Option          | Description                                  |
| --------------- | -------------------------------------------- |
| `-m`, `--match` | Enable match mode (filter by mask)           |
| `-n`, `--name`  | Set FPS name filter mask                     |
| `-f`, `--fifo`  | Removed with the command FIFO: exits with status 2; use `milk-fps-set` / `milk-fps-{conf,run}{start,stop}` |
| `-q`, `--quiet` | Suppress TUI output (`MILK_TUIPRINT_NONE=1`) |
| `-s`, `--stdio` | Removed with the command FIFO: exits with status 2 |
| `-h`, `--help`  | Print usage and exit                         |
| `-h1`           | Print one-line description and exit          |

The positional argument `fpsnamemask` sets the name filter (default: `_ALL`).

## Environment Variables

| Variable                  | Effect                            |
| ------------------------- | --------------------------------- |
| `MILK_FPS_LOGFILE`        | Path to FPS log file              |
| `FPS_FILTSTRING_NAME`     | Filter FPS entries by name        |
| `FPS_FILTSTRING_KEYWORD`  | Filter by keyword                 |
| `FPS_FILTSTRING_CALLFUNC` | Filter by call function in source |
| `FPS_FILTSTRING_MODULE`   | Filter by source code module      |

## Display Modes

| Mode | Name    | Access   | Description                                   |
| ---- | ------- | -------- | --------------------------------------------- |
| 1    | Help    | `h`      | Built-in help screen                          |
| 2    | FPS Log | `?`      | FPS event log viewer                          |
| 3    | Control | `F2`     | Main FPS control view                         |
| 4    | Output  | `Ctrl+a` | The selected unit's run output, followed live |

The default display mode at startup is **Control** (mode 3). In the output view, Ctrl-C stops the
unit (runstop), `r` restarts it, and `ESC`, `q`, `F2` or `Ctrl+a` go back to the control view.

## Keyboard Shortcuts

### Navigation

| Key          | Action                                 |
| ------------ | -------------------------------------- |
| Up / Down    | Move selection up/down in FPS list     |
| PgUp / PgDn  | Page up/down through FPS list          |
| Left / Right | Navigate parameter columns / levels    |

### Mode Switching

| Key        | Action                                              |
| ---------- | --------------------------------------------------- |
| `h`        | Show help screen (mode 1)                           |
| `?`        | Show FPS log (mode 2)                               |
| `F2`       | Switch to control mode (mode 3)                     |
| Ctrl+Right | Next display mode                                   |
| Ctrl+Left  | Previous display mode                               |
| `ESC`      | Exit output view (mode 4) or clear search filter    |

### FPS Control

| Key           | Action                                            |
| ------------- | ------------------------------------------------- |
| `O`           | Start conf process (confstart)                    |
| `Ctrl+o`      | Stop conf process (confstop)                      |
| `R`           | Start run process (runstart)                      |
| `Ctrl+r`      | Stop run process (runstop)                        |
| `Ctrl+a`      | Follow the run output (mode 4)                    |
| `u`           | Update conf process                               |
| `E`, `Ctrl+e` | Stop conf and run, erase the FPS (milk-fps-rm -f) |

`T` and `Ctrl+t` are gone: units have no tmux session.

### Parameter Editing

| Key     | Action                                       |
| ------- | -------------------------------------------- |
| `Enter` | Edit selected parameter                      |
| `Space` | Toggle ON/OFF parameter, or run a script one |

`Space` on a script parameter (type EXECFILENAME) starts the script in the unit's work directory,
detached, with the FPS data file as argument; its output goes to `<name>.fps.d/exec.out`.

### Display and Search

| Key       | Action                                             |
| --------- | -------------------------------------------------- |
| `/`       | Enter search mode (filter by text)                 |
| `v` / `V` | Toggle verbose display (state, uptime, last exit)  |
| `S`       | Toggle sort mode                                   |
| `]`       | Cycle to next sort column                          |
| `l`       | List all parameters for selected FPS               |
| `y`       | Copy `parameter = value` to the clipboard (OSC 52) |

### General

| Key    | Action                |
| ------ | --------------------- |
| `s`    | Rescan FPS instances  |
| `x`    | Exit fpsCTRL          |
| Ctrl+C | Exit (signal handler) |

---

← [Documentation Index](../index.md)
