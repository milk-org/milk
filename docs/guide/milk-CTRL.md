---
tags:
  - tui
  - streams
  - fps
  - procinfo
---

# Dashboard (`milk-CTRL`)

`milk-CTRL` shows the streams, FPS and processes of a milk runtime in one terminal, with the links
between them: which process writes or reads which stream, which FPS runs which process, and the
feedback loops they form. Viewing never writes shared memory. CONTROL mode (`c`) enables actions,
such as starting an FPS, pausing a process or deleting a stream, and asks `y/N` before the
destructive ones.

```bash
milk-CTRL
milk-CTRL --view procs 'wfs|dm'
milk-CTRL -d /tmp/mysim --interval 0.2 --log /tmp/ctrl.log
```

## Screen

```text
●▌ ● milk-CTRL ▐ [commit] [shm: /milk/shm]  [c] READ ONLY  [m] HOVER: OFF  [/] FILTER: ALL  12 stm 5 prc 4 fps
 ▌ F2:DASH ▐  [ F3:STRM ]  [ F4:PROC ]  [ F5:FPS ]  [ F6:CONN ]  [ F7:LOOPS ]               [h: HELP]
╔═ STREAMS (12) ══════════════════════╗╔═ PROCESSINFO (5) ═══════════════════════════════════╗
║ ...                                 ║║ ...                                                 ║
╔═ FPS (4) ═══════════════════════════╗╔═ CONNECTIONS ═ LOOPS ═ DETAILS ═ RESOURCES ═════════╗
║ ...                                 ║║ ...                                                 ║
 12:00:01 ● Control mode ON
 scan:2ms 1.0Hz  DASH › Streams [sort:name] [CONN]  TAB focus  / filter  D details  q quit  [dark]  [x] exit 12:00:05
```

- **Header**: commit, shared-memory directory, the mode badges (click to toggle CONTROL mode and
  hover highlights, or a panel's filter), entity counts and the dashboard's own CPU use.
- **Tab bar**: the views. Click a tab, or press its F-key; `[h: HELP]` opens the help.
- **Panels**: each list has a header row; click a column title to sort by it.
- **Command log**: results of actions, warnings, and messages from the libraries. `G` hides it,
  `v` / `V` make it smaller / larger.
- **Status line**: scan time and rate, the view and focused panel, the sort column, the sub-tab,
  the keys of the current context (CONTROL on PROC: `k term  p pause  e exit`; the F5 parameter
  tree: `← up  ESC back  ENTER edit`), the theme, an exit button and the clock.

On a narrow terminal (80 columns) the header, tab bar and status line drop their least important
items first: the commit, the shm path (shortened to its last directory), the logo and the general
key hints go before the mode badge, the counts and the filter badges. The tabs shorten to `F2:DASH`
and then `F2`. `[h: HELP]`, `[x] exit` and the clock always stay.

## Views

| View | Shows |
|------|-------|
| `F2` DASH | The streams, processes and FPS lists, and the sub-tab panel bottom right |
| `F3` STRM | Streams: type, size, rate, throughput, inode, owner, counter, semaphores, writer, readers |
| `F4` PROC | Processes: PID, priority, state, loop rate, uptime, trigger, execution time, duty cycle, CPU, loop counter, memory, missed triggers, status message |
| `F5` FPS | The FPS list on the left, the parameter tree of the FPS under the cursor on the right |
| `F6` CONN | The sub-tab panel full screen |
| `F7` LOOPS | The feedback loops on the left, the selected loop's circuit and members on the right |

Process states: `INIT`, `RUN`, `PAUS`, `SPIN`, `STOP`, `ERR`, `CRSH`, and `STAL` for a running
loop whose counter stopped moving. Loop states, worst first: `ERR`, `STAL`, `PAUS`, `RUN`, `IDLE`.

## Sub-tabs

The panel bottom right of DASH, and F6 full screen, has four sub-tabs. Click one, or press
Shift+TAB to cycle; `D` toggles DETAILS.

| Sub-tab | Shows |
|---------|-------|
| CONNECTIONS | Producers and consumers of the entry last selected in a list, with their health (below). `L` cycles the lineage mode (trigger, input, full); ENTER selects the row's stream in DASH; a double click right of the name selects the reader process |
| LOOPS | The feedback loops (top) and the selected one (bottom). ENTER or `f` isolates the loop: the lists show its members only. `r` renames it, `g` shows its CONNECTIONS, a double click opens F7 |
| DETAILS | The entry under the cursor of the focused list, else the last FPS, stream or process selected. For an FPS, its parameters: the arrows move a cursor, ENTER edits the value (CONTROL mode) |
| RESOURCES | The process behind that entry: memory, CPU cores, scheduling and hardware counters |

CONNECTIONS health columns, right of each row (dropped by width like list columns):

| Column | Shows |
|--------|-------|
| Hz | Update rate of the row's stream; `!` and the warning colour when its reader runs slower |
| STATE | The reader process's state, `OFF` for an FPS whose run process is down, `-` without a reader. `STALL` in red marks the first node upstream that stopped advancing; rows below it are dimmed |
| EXEC | The reader's median compute time |
| MISS | Its missed-trigger share |
| FPS | The FPS running the reader process |

Stream DETAILS lists the process trace: per hop the trigger mode, the compute time (processes
with timing enabled), the hand-off delay from the previous hop and how many frames behind the
input it ran (`lag`), with the end-to-end latency on top. F7 lists the FPS of a loop, including the
FPS whose run process is on the cycle.

## Lists

- **Sorting**: click a column title (again to reverse), or `s` name, `S` rate, `A` ancestry,
  `<` / `>` the previous / next sortable column, `[` reverse.
- **Columns**: Shift+LEFT / RIGHT moves the column cursor, `t` hides the column under it (it shrinks
  to `»`; `t` again shows it), `d` hides the wide optional columns.
- **Narrow panels**: `A` and NAME always show. The other columns are dropped, least important
  first, until the rest fit; `›` at the right of the header means some were dropped. Moving the
  column cursor onto a dropped column scrolls the columns to show it (`‹` means some are scrolled
  off on the left), so every column can be reached. NAME widens, up to 48 columns, when there is
  room. Cut text ends with `…`.
- **Numbers**: rates, throughput and counters are scaled with SI suffixes in at most 6 columns
  (`999.9`, `12.3k`, `1.20M`): Hz, B/s (bytes per second), COUNT, LOOPCNT, MISSED. EXEC and
  latencies carry their unit (`850ns`, `12.3us`, `1.25ms`, `2.40s`).
- **Empty lists** say so, with the directory scanned: `No streams — start a pipeline, or
  milk-CTRL -d DIR (now /milk/shm)`. F7 says `No feedback loops`.
- **Filter**: `/` types an extended, case-insensitive regex for the focused list; `f` toggles it,
  ESC clears it, `?` jumps to the first match without filtering.
- **Freeze**: SPACE keeps the selected entry and its relations highlighted while you move the
  cursor elsewhere; SPACE again releases it.
- **Lineage**: Shift+UP / DOWN jump to the nearest upstream / downstream entity. The `A` column
  shows, relative to the selection: `◀3` / `3▶` three steps upstream / downstream, `●` the
  selection, `▶` / `◀` writes / reads the selected stream, `↺` part of a loop, `⮂` of several.

## CONTROL mode

`c`, or a click on the header badge, toggles CONTROL mode. Actions apply to the entry under the
cursor of the focused panel, and only while that panel is on screen. Before acting, milk-CTRL opens
the file again and checks that it is still the same entity (stream inode, FPS name, process start
time). SIGKILL, erase, cleanup and batch actions ask `y/N` first.

FPS batches: `b` marks the FPS under the cursor, `a` marks every visible FPS; `k`, `K`, `r` and `s`
then act on all marked FPS after one `y/N`. ESC clears the marks.

## Keys

The tables below are generated from the key table of milk-CTRL (`milk-CTRL --keys-markdown`); a test
checks that they match. `h` or F1 opens the help, with every key and a longer description.

<!-- keys:begin -->
### Views (F2-F7; Ctrl+Left/Right or Ctrl+wheel cycle them)

| Keys | Action |
|------|--------|
| `F2` DASH | Streams, processes and FPS panels plus the sub-tab panel |
| `F3` STRM | Streams: size, rate, semaphores, writers |
| `F4` PROC | Processes: state (RUN/STOP/CRSH/STAL), rate, CPU |
| `F5` FPS | FPS list (left) and parameter tree (right) |
| `F6` CONN | Sub-tab panel full screen: connections, loops, details, resources |
| `F7` LOOPS | Feedback loops: members, rate, status (ERR/STAL/PAUS/RUN/IDLE) |
| `^Left/^Right` | Cycle views sequentially |

### Navigation and display

| Keys | Action |
|------|--------|
| `TAB` | Cycle active panel focus |
| `Left / Right` | Move focus to the adjacent panel |
| `UP / DOWN` | Navigate rows in focused panel |
| `PgUp / PgDn` | Scroll page up / down |
| `Home / End` | Jump to first / last item |
| `?` | Jump to the first match |
| `i` | Inspect selection |
| `/` | Filter panel items (regex search) |
| `f` | Toggle regex filter ON / OFF |
| `SPACE` | Freeze selection highlight |
| `+ / -` | Adjust scan rate (faster / slower) |
| `F` | Pause / resume display updates |
| `W` | Export snapshot to file ($TMPDIR, default /tmp) |
| `F8 / ^T` | Next theme + selector |
| `c` | Toggle Control Mode ON / OFF |
| `h` | Toggle this help overlay |
| `q / x` | Quit milk-CTRL |
| `ENTER / D` | DETAILS sub-tab (DASH, F6); select the CONNECTIONS row's stream; F5: tree / edit |
| `ESC` | Back out: parameter tree, DETAILS cursor, filters, FPS marks |
| `F3: Right / ENTER` | Open the stream inspector; Right: next page |
| `Shift+Up` | Jump to upstream producer (Ancestor) |
| `Shift+Down` | Jump to downstream consumer (Descendant) |
| `L` | Cycle lineage mode (Trigger / Input / Full) |
| `Shift+TAB` | Cycle graph tab (CONN, LOOPS, DETAIL, RES) |
| `G` | Toggle command log panel visibility |
| `v / V` | Decrease / increase command log height |
| `m` | Toggle mouse hover tracking (ON/OFF) |

### Columns and sorting

| Keys | Action |
|------|--------|
| `< / > / ]` | Cycle active sort column |
| `[` | Toggle sort direction (Asc / Desc) |
| `Shift+Left/Rt` | Select column for layout hiding |
| `t / T` | Toggle column visibility (hide/show) |
| `d` | Toggle compact layout mode |
| `s / S / A` | Sort by name / rate / ancestry (PROC, FPS: 's' outside CONTROL) |
| `( / )` | Adjust dashboard horizontal split ratio |
| `{ / }` | Adjust dashboard vertical split ratio |

### Feedback loops (F7, or LOOPS sub-tab)

| Keys | Action |
|------|--------|
| `r` | Rename feedback loop |
| `f` | Toggle loop isolation filter |
| `g` | Switch to graph CONNECTIONS tab |

### Control mode actions ('c' toggles; y/N asks before destructive ones)

| Keys | Action |
|------|--------|
| `DEL` | STRM: Delete stream shared memory file |
| `k` | PROC: Graceful kill (SIGTERM) |
| `K` | PROC: Immediate kill (SIGKILL) |
| `p` | PROC: Pause / run (CTRLval 1/0) |
| `^s` | PROC: Step single loop iteration |
| `e` | PROC: Clean exit request (CTRLval=3) |
| `z` | PROC: Zero loop performance counters |
| `C` | PROC: Clean up dead process entries |
| `CTRL+e` | PROC: Erase entry (stream, dead process, FPS) |
| `r` | FPS: Toggle Run loop (runstart / runstop) |
| `s` | FPS: Toggle Conf loop (confstart / confstop) |
| `o` | FPS: Toggle ON/OFF parameter (F5) |
| `b` | FPS: Toggle the batch mark on an FPS module |
| `a` | FPS: Mark every visible FPS module (batch) |
<!-- keys:end -->

## Mouse

Click selects an entry and focuses its panel, a double click acts like ENTER, the wheel scrolls
the list under the pointer and Ctrl+wheel cycles the views. Click tabs, sub-tabs and column titles;
drag the borders between panels to resize them. With `m` (hover), rows under the pointer are
highlighted, and in CONTROL mode they show action buttons.

## Options

| Option | Meaning |
|--------|---------|
| `-d, --shm-dir DIR` | Streams and processinfo directory |
| `-T, --theme NAME` | Colour theme (`-h` lists them) |
| `--view NAME` | Open this view: `dash`, `streams`, `procs`, `fps`, `conn`, `loops` |
| `PATTERN` | Initial regex filter of the view's list (the streams in DASH, CONN and LOOPS) |
| `--interval SEC` | Scan period, 0.1 to 10 s (1); `+` / `-` change it while running |
| `--log FILE` | Append the command log to FILE |
| `--debug` | With `--log`: also key events and the time of each scan |
| `-h`, `-h1`, `-hm` | Help, one-line description, monochrome help |

Exit status: 0 ok, 1 runtime error (no terminal, `--log` file not writable), 2 usage error.

Each `--log` line is a timestamp, a level and the message:

```text
2026-10-20T12:00:01.204 INFO  Shared memory directory: /milk/shm
2026-10-20T12:00:03.511 OK    FPS "wfsproc": RUNstart sent
2026-10-20T12:00:03.512 DEBUG key 114
2026-10-20T12:00:04.002 DEBUG scan 2.1 ms: 12 streams, 5 processes, 4 FPS
```

## Colours

F8 or Ctrl-T cycles the themes (a selector shows them); `-T` or `MILK_CTRL_THEME` picks one at
start. The colour depth follows the terminal: `COLORTERM=truecolor` or `24bit` gives 24-bit colour,
a `TERM` ending in `256color` gives 256, else 16. `NO_COLOR` or `TERM=dumb` gives monochrome, with
the selection in reverse video.

## Environment

| Variable | Meaning |
|----------|---------|
| `MILK_SHM_DIR` | Streams and FPS directory (else `/milk/shm`, `/dev/shm`, `/tmp`) |
| `MILK_PROC_DIR` | processinfo directory (`-d` sets both) |
| `MILK_CTRL_THEME` | Initial theme (`-T` wins) |

## See also

`milk-stream-info`, `milk-procinfo-info` and `milk-fps-info` (what `i` runs on the selection),
`milk-stream-graph`, [`milk-termview`](termview.md), [fpsCTRL](fpsCTRL_reference.md).
