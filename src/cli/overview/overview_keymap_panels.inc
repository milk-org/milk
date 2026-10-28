// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* Key table rows: processes, FPS, graph and loops, command log and mouse, theme.
 * Fields: display, label, help detail (NULL = label), help section (-1 = -h only),
 * -h group (-1 = help only), OV_KF_* flags, status-line hint (OV_NOHINT = none). */


    {
        "D / ENTER",
        "Toggle process detail pane",
        "Opens process inspector showing command line, start time, thread count, voluntary context switches, "
        "page faults, CPU core affinity, and perf counter metrics. In F3/F4 the key only says DETAILS is in "
        "F2/F6.",
        HS_PROCS,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "s / S",
        "Sort by name / loop rate (Hz)",
        "Press 's' to sort processes by name (outside CONTROL mode, where 's' steps the loop); press 'S' to "
        "sort by loop rate (Hz). Click the CPU% column title to sort by CPU use.",
        HS_PROCS,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "k",
        "Graceful kill (SIGTERM)",
        "Sends SIGTERM (signal 15) to the process PID. Gives the process opportunity to execute cleanup "
        "handlers, detach shared memory, and exit gracefully. Requires Control Mode ON.",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "k", "term", OV_CTX_CTRL | OV_CTX_FOCUS_PROCS | OV_CTX_FOCUS_SHOWN, 0 },
    },

    {
        "K",
        "Immediate kill (SIGKILL)",
        "Sends SIGKILL (signal 9) directly to the process PID after a y/N confirmation. Forces immediate kernel "
        "termination. Use when a process is stuck or unresponsive to SIGTERM. Requires Control Mode ON.",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        OV_NOHINT,
    },

    {
        "p",
        "Pause / run (CTRLval 1/0)",
        "Toggles CTRLval between 1 (pause) and 0 (run). Only loops that honour CTRLval react; no signal is sent. "
        "Requires Control Mode ON.",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "p", "pause", OV_CTX_CTRL | OV_CTX_FOCUS_PROCS | OV_CTX_FOCUS_SHOWN, 0 },
    },

    {
        "^s",
        "Step single loop iteration",
        "Sends step command (CTRLval=2) to advance a paused loop-controlled process by exactly one iteration, "
        "then re-pauses. Requires Control Mode ON.",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        OV_NOHINT,
    },

    {
        "e",
        "Clean exit request (CTRLval=3); stops a unit",
        "Sets process CTRLval=3 requesting a cooperative, clean exit at the end of the current iteration without "
        "raising signals. On the run process of a unit, stops the unit instead (milk-fps-runstop: the same "
        "request, then SIGINT, SIGTERM and SIGKILL if the process does not exit). Requires Control Mode ON.",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "e", "exit", OV_CTX_CTRL | OV_CTX_FOCUS_PROCS | OV_CTX_FOCUS_SHOWN, 0 },
    },

    {
        "z",
        "Zero loop performance counters",
        "Resets iteration loop counters, timing accumulators, and duty cycle statistics back to zero for "
        "benchmark timing. Requires Control Mode ON.",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        OV_NOHINT,
    },

    {
        "C",
        "Clean up dead process entries",
        "After a y/N confirmation, removes STOPPED and CRASHED processinfo entries (milk-procinfo-rm -c). Live "
        "processes are untouched. Requires Control Mode ON.",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "C", "cleanup", OV_CTX_CTRL | OV_CTX_FOCUS_PROCS | OV_CTX_FOCUS_SHOWN, 0 },
    },

    {
        "CTRL+e",
        "Erase entry (stream, dead process, FPS)",
        "After a y/N confirmation: deletes the selected stream, removes a dead process entry, or erases the "
        "selected FPS (milk-fps-rm -f in the background: stops CONF and RUN, then deletes its shm and unit "
        "directory). Requires Control Mode ON. FPS erase is refused unless the FPS name is [A-Za-z0-9_-].",
        HS_PROCS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "^e", "erase", OV_CTX_CTRL | OV_CTX_FOCUS_LIST | OV_CTX_FOCUS_SHOWN, OV_CTX_PARAM_TREE },
    },

    {
        "D / ENTER",
        "Inspect parameters / Edit value",
        "In Dashboard view, opens parameter summary. In F5 view, opens inline editor for the highlighted "
        "parameter (string, integer, float, or toggle).",
        HS_FPS,
        -1,
        0,
        { "ENTER", "edit", OV_CTX_PARAM_TREE, 0 },
    },

    {
        "r",
        "Toggle Run loop (runstart / runstop)",
        "Starts or stops the run process of the selected unit with milk-fps-runstart or milk-fps-runstop in the "
        "background; the result goes to the command log, and the screen keeps refreshing meanwhile. Requires "
        "Control Mode ON. Refused unless the FPS name is [A-Za-z0-9_-]; the unit checks its owner and "
        "parameters itself.",
        HS_FPS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "r", "run", OV_CTX_CTRL | OV_CTX_FOCUS_FPS | OV_CTX_FOCUS_SHOWN, OV_CTX_PARAM_TREE },
    },

    {
        "s",
        "Toggle Conf loop (confstart / confstop)",
        "Starts or stops the conf process of the selected unit with milk-fps-confstart or milk-fps-confstop in "
        "the background; the result goes to the command log. Requires Control Mode ON. Refused unless the FPS "
        "name is [A-Za-z0-9_-].",
        HS_FPS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "s", "conf", OV_CTX_CTRL | OV_CTX_FOCUS_FPS | OV_CTX_FOCUS_SHOWN, OV_CTX_PARAM_TREE },
    },

    {
        "k / K",
        "Send SIGTERM / SIGKILL to FPS processes",
        "Sends SIGTERM (k, immediate) or SIGKILL (K, after y/N) to the conf and run processes of the selected "
        "FPS module, or to every marked FPS (y/N): a forced stop, where r stops a unit cleanly. Requires "
        "Control Mode ON.",
        HS_FPS,
        -1,
        OV_KF_CTRL,
        { "k", "kill", OV_CTX_CTRL | OV_CTX_FOCUS_FPS | OV_CTX_FOCUS_SHOWN, OV_CTX_PARAM_TREE },
    },

    {
        "o",
        "Toggle ON/OFF parameter (F5)",
        "Flips the highlighted ON/OFF parameter in the F5 tree. Refused, with the reason, when the parameter is "
        "write-protected or the FPS file is not writable. Requires Control Mode ON.",
        HS_FPS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "o", "on/off", OV_CTX_CTRL | OV_CTX_PARAM_TREE, 0 },
    },

    {
        "b",
        "Toggle the batch mark on an FPS module",
        "Marks or unmarks the highlighted FPS (by name) and moves down. With marks set, k, K, r and s act on "
        "every marked FPS after a y/N confirmation; the FPS title shows the count and how many marks a filter "
        "hides. Requires Control Mode ON.",
        HS_FPS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "b", "mark", OV_CTX_CTRL | OV_CTX_FOCUS_FPS | OV_CTX_FOCUS_SHOWN, OV_CTX_PARAM_TREE },
    },

    {
        "a",
        "Mark every visible FPS module (batch)",
        "Adds every FPS row shown (the filter is respected) to the batch marks, e.g. to stop or start a whole "
        "pipeline together. Requires Control Mode ON.",
        HS_FPS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        OV_NOHINT,
    },

    {
        "ESC",
        "Clear FPS batch marks",
        "Last step of the ESC back-chain: once no filter is left to clear, ESC drops every FPS batch mark.",
        HS_FPS,
        -1,
        0,
        { "ESC", "unmark", OV_CTX_MSEL, OV_CTX_PARAM_TREE },
    },

    {
        "{ / }",
        "Adjust F5 split ratio (FPS list / params)",
        "Adjusts the vertical separator position between the left FPS module list and the right parameter tree "
        "panel in F5 view.",
        HS_FPS,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "Shift+Up",
        "Jump to upstream producer (Ancestor)",
        "Traces backwards through the dataflow graph to locate and highlight the process or FPS module that "
        "writes to the currently selected stream.",
        HS_GRAPH,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "Shift+Down",
        "Jump to downstream consumer (Descendant)",
        "Traces forward through the dataflow graph to locate and highlight the processes or FPS modules that "
        "consume the currently selected stream.",
        HS_GRAPH,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "L",
        "Cycle lineage mode (Trigger / Input / Full)",
        "Trigger follows semaphore-triggered dependencies only; Input adds sampled streams; Full also follows "
        "FPS stream parameters.",
        HS_GRAPH,
        OV_KG_NAV,
        0,
        { "L", "lineage", OV_CTX_FOCUS_GRAPH, OV_CTX_LOOP_LIST },
    },

    {
        "ENTER / D",
        "Select the row's stream / toggle DETAILS",
        "In CONNECTIONS, ENTER selects the highlighted row's stream in the STREAMS panel (DASH); a double click "
        "on the reader name selects that process. The tree stays rooted at the entry selected in the list panel "
        "last focused. D toggles the DETAILS sub-tab.",
        HS_GRAPH,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "Shift+TAB",
        "Cycle graph tab (CONN, LOOPS, DETAIL, RES, OUT)",
        "Cycles the graph panel display through: CONNECTIONS tree, LOOPS detection list, DETAILS connectivity "
        "inspector, RESOURCES allocation panel, and OUTPUT, the end of the selected unit's run output.",
        HS_GRAPH,
        OV_KG_NAV,
        0,
        { "\xe2\x87\xa7TAB", "sub-tab", OV_CTX_FOCUS_GRAPH, 0 },
    },

    {
        "r",
        "Rename feedback loop",
        "When focused on a loop in the LOOPS tab or F7 view, opens inline prompt to assign a custom name. Names "
        "are automatically persisted across sessions.",
        HS_GRAPH,
        OV_KG_LOOPS,
        0,
        { "r", "rename", OV_CTX_LOOP_LIST, 0 },
    },

    {
        "f",
        "Toggle loop isolation filter",
        "Filters the Streams, Processes, and FPS panels to isolate components belonging to the selected feedback "
        "loop. Press ESC to clear.",
        HS_GRAPH,
        OV_KG_LOOPS,
        0,
        { "f", "isolate", OV_CTX_LOOP_LIST, 0 },
    },

    {
        "g",
        "Switch to graph CONNECTIONS tab",
        "Switches the panel to the CONNECTIONS lineage tree to view the loop's dataflow circuit.",
        HS_GRAPH,
        OV_KG_LOOPS,
        0,
        OV_NOHINT,
    },

    {
        "G",
        "Toggle command log panel visibility",
        "Hides or reveals the command log panel strip at the bottom of the screen. When hidden, extra rows are "
        "allocated to main dashboard panels.",
        HS_CMDLOG,
        OV_KG_NAV,
        0,
        { "G", "log", 0, 0 },
    },

    {
        "v / V",
        "Decrease / increase command log height",
        "Expands or shrinks the number of display rows allocated to the command log panel (default is 4 rows).",
        HS_CMDLOG,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "( / )",
        "Adjust dashboard horizontal split ratio",
        "Moves the horizontal dividing line in Dashboard view (F2) between top panels (Streams) and bottom "
        "panels (Processes and FPS).",
        HS_CMDLOG,
        OV_KG_COLUMNS,
        0,
        OV_NOHINT,
    },

    {
        "{ / }",
        "Adjust dashboard vertical split ratio",
        "Moves the vertical dividing line in Dashboard view (F2) between left panels (Processes) and right "
        "panels (FPS and Graph).",
        HS_CMDLOG,
        OV_KG_COLUMNS,
        0,
        OV_NOHINT,
    },

    {
        "m",
        "Toggle mouse hover tracking (ON/OFF)",
        "Enables or disables live hover tracking. When ON, hovering over cells, PIDs, or borders displays "
        "instant contextual tooltips. Can be turned off to save bandwidth.",
        HS_MOUSE,
        OV_KG_NAV,
        0,
        { "m", "hover", 0, 0 },
    },

    {
        "F8 / ^T",
        "Open theme selector popup (↑/↓ to choose, ESC or 1s to close)",
        "Switches to the next theme and opens the selector with live swatches; Up/Down or ^T choose, ESC or 1 s "
        "idle closes. milk-CTRL -h lists the themes. Colours follow the terminal depth (COLORTERM/TERM); "
        "NO_COLOR or TERM=dumb gives monochrome.",
        HS_COLORS,
        -1,
        0,
        OV_NOHINT,
    },
