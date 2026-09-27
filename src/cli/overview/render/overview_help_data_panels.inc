    /* =========================================================
     * 3. Processes Panel (PROC)
     * ========================================================= */
    {
        NULL,
        "Processes Panel (PROC)",
        "Process registry monitor. Tracks all compute units and scripts registered in "
        "processinfo.list.shm. Displays process PID, loop count, iteration frequency, "
        "duty cycle, CPU%, memory RSS, and running status. Allows sending POSIX signals.",
        HF_SECTION,
        HS_PROCS,
    },
    {
        "D / ENTER",
        "Toggle process detail pane",
        "Opens process inspector showing command line, start time, thread count, "
        "voluntary context switches, page faults, CPU core affinity, and perf counter metrics.",
        HF_ENTRY,
        HS_PROCS,
    },
    {
        "s / S",
        "Sort by Name / CPU%",
        "Press 's' to sort processes alphabetically by process name; press 'S' to sort "
        "by active CPU% utilization or execution frequency.",
        HF_ENTRY,
        HS_PROCS,
    },
    {
        "k",
        "Graceful kill (SIGTERM)",
        "Sends SIGTERM (signal 15) to the process PID. Gives the process opportunity "
        "to execute cleanup handlers, detach shared memory, and exit gracefully. "
        "Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },
    {
        "K",
        "Immediate kill (SIGKILL)",
        "Sends SIGKILL (signal 9) directly to the process PID. Forces immediate kernel "
        "termination. Use when a process is stuck or unresponsive to SIGTERM. "
        "Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },
    {
        "p",
        "Pause / resume process (SIGSTOP/CONT)",
        "Toggles process execution between paused and running by setting CTRLval or "
        "dispatching SIGSTOP/SIGCONT signals. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },
    {
        "^s",
        "Step single loop iteration",
        "Sends step command (CTRLval=2) to advance a paused loop-controlled process "
        "by exactly one iteration, then re-pauses. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },
    {
        "e",
        "Clean exit request (CTRLval=3)",
        "Sets process CTRLval=3 requesting a cooperative, clean exit at the end of the "
        "current iteration without raising signals. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },
    {
        "z",
        "Zero loop performance counters",
        "Resets iteration loop counters, timing accumulators, and duty cycle statistics "
        "back to zero for benchmark timing. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },
    {
        "C",
        "Cleanup / release allocations",
        "Requests process to execute internal memory and resource cleanup procedures "
        "without terminating the process. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },
    {
        "CTRL+e",
        "Erase stale / dead process entry",
        "Deactivates a dead or crashed process slot in processinfo.list.shm, removing "
        "orphaned entries from the dashboard. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_PROCS,
    },

    /* =========================================================
     * 4. FPS Panel (FPS)
     * ========================================================= */
    {
        NULL,
        "FPS Panel (FPS)",
        "Function Parameter Structure management. Tracks FPS modules, shared configuration "
        "parameters, and execution state. In F5 view, provides full interactive tree view "
        "for inspecting and editing live runtime parameters in shared memory.",
        HF_SECTION,
        HS_FPS,
    },
    {
        "D / ENTER",
        "Inspect parameters / Edit value",
        "In Dashboard view, opens parameter summary. In F5 view, opens inline editor "
        "for the highlighted parameter (string, integer, float, or toggle).",
        HF_ENTRY,
        HS_FPS,
    },
    {
        "r",
        "Toggle Run loop (runstart / runstop)",
        "Dispatches runstart or runstop command to the selected FPS module tmux session "
        "or control process, starting or stopping compute loop. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_FPS,
    },
    {
        "s",
        "Toggle Conf loop (confstart / confstop)",
        "Dispatches confstart or confstop command to the selected FPS module, "
        "toggling configuration loop monitoring. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_FPS,
    },
    {
        "k / K",
        "Send SIGTERM / SIGKILL to FPS session",
        "Sends termination signals to the tmux session and worker processes managing "
        "the selected FPS module. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_FPS,
    },
    {
        "SPACE",
        "Toggle multi-select on FPS module",
        "Toggles multi-selection checkbox on the highlighted FPS module for performing "
        "batch run/stop or kill operations on multiple modules at once. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_FPS,
    },
    {
        "a",
        "Select all FPS modules (batch)",
        "Selects all listed FPS modules simultaneously for batch operations "
        "(e.g. stopping or starting an entire pipeline together). Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_FPS,
    },
    {
        "c",
        "Clear FPS multi-selections",
        "Clears all multi-selection marks across all FPS modules, returning to "
        "single-selection mode.",
        HF_ENTRY,
        HS_FPS,
    },
    {
        "{ / }",
        "Adjust F5 split ratio (FPS list / params)",
        "Adjusts the vertical separator position between the left FPS module list "
        "and the right parameter tree panel in F5 view.",
        HF_ENTRY,
        HS_FPS,
    },

    /* =========================================================
     * 5. Graph & Lineage (CONN)
     * ========================================================= */
    {
        NULL,
        "Graph & Lineage Panel (CONN)",
        "Visual dataflow graph and dependency tracker. Traces directed edges between "
        "streams and processes. Distinguishes trigger streams (driving compute loops) "
        "from passive input streams and output products.",
        HF_SECTION,
        HS_GRAPH,
    },
    {
        "Shift+Up",
        "Jump to upstream producer (Ancestor)",
        "Traces backwards through the dataflow graph to locate and highlight the process "
        "or FPS module that writes to the currently selected stream.",
        HF_ENTRY,
        HS_GRAPH,
    },
    {
        "Shift+Down",
        "Jump to downstream consumer (Descendant)",
        "Traces forward through the dataflow graph to locate and highlight the processes "
        "or FPS modules that consume the currently selected stream.",
        HF_ENTRY,
        HS_GRAPH,
    },
    {
        "L",
        "Cycle lineage mode (Trigger vs Input)",
        "Toggles lineage graph filtering: Triggers only (shows strictly semaphore-triggered "
        "compute dependencies) vs All Inputs (includes sampled/passive streams).",
        HF_ENTRY,
        HS_GRAPH,
    },
    {
        "ENTER / D",
        "Inspect node connection details",
        "Opens detailed connectivity report for the highlighted node, listing all upstream "
        "writers, downstream readers, trigger modes, and semaphore indices.",
        HF_ENTRY,
        HS_GRAPH,
    },
    {
        "Shift+TAB",
        "Cycle graph tab (CONN, LOOPS, DETAIL, RES)",
        "Cycles the graph panel display through: CONNECTIONS tree, LOOPS detection list, "
        "DETAILS connectivity inspector, and RESOURCES allocation panel.",
        HF_ENTRY,
        HS_GRAPH,
    },
    {
        "r",
        "Rename feedback loop",
        "When focused on a loop in the LOOPS tab or F7 view, opens inline prompt to assign "
        "a custom name. Names are automatically persisted across sessions.",
        HF_ENTRY,
        HS_GRAPH,
    },
    {
        "f",
        "Toggle loop isolation filter",
        "Filters the Streams, Processes, and FPS panels to isolate components belonging to "
        "the selected feedback loop. Press ESC to clear.",
        HF_ENTRY,
        HS_GRAPH,
    },
    {
        "g",
        "Switch to graph CONNECTIONS tab",
        "Switches the panel to the CONNECTIONS lineage tree to view the loop's dataflow circuit.",
        HF_ENTRY,
        HS_GRAPH,
    },

    /* =========================================================
     * 6. Command Log & Display
     * ========================================================= */
    {
        NULL,
        "Command Log & Display",
        "Real-time audit log and layout configuration. Shows timestamped records of user "
        "commands, process signaling, state changes, and error warnings. Provides "
        "interactive panel split resizing.",
        HF_SECTION,
        HS_CMDLOG,
    },
    {
        "G",
        "Toggle command log panel visibility",
        "Hides or reveals the command log panel strip at the bottom of the screen. "
        "When hidden, extra rows are allocated to main dashboard panels.",
        HF_ENTRY,
        HS_CMDLOG,
    },
    {
        "v / V",
        "Decrease / increase command log height",
        "Expands or shrinks the number of display rows allocated to the command log "
        "panel (default is 4 rows).",
        HF_ENTRY,
        HS_CMDLOG,
    },
    {
        "( / )",
        "Adjust dashboard horizontal split ratio",
        "Moves the horizontal dividing line in Dashboard view (F2) between top panels "
        "(Streams) and bottom panels (Processes and FPS).",
        HF_ENTRY,
        HS_CMDLOG,
    },
    {
        "{ / }",
        "Adjust dashboard vertical split ratio",
        "Moves the vertical dividing line in Dashboard view (F2) between left panels "
        "(Processes) and right panels (FPS and Graph).",
        HF_ENTRY,
        HS_CMDLOG,
    },

    /* =========================================================
     * 7. Mouse Interactions
     * ========================================================= */
    {
        NULL,
        "Mouse Interactions",
        "Full terminal mouse support. Enables direct point-and-click selection, wheel "
        "scrolling, drag-to-resize panel borders, and column sorting.",
        HF_SECTION,
        HS_MOUSE,
    },
    {
        "m",
        "Toggle mouse hover tracking (ON/OFF)",
        "Enables or disables live hover tracking. When ON, hovering over cells, PIDs, "
        "or borders displays instant contextual tooltips. Can be turned off to save bandwidth.",
        HF_ENTRY,
        HS_MOUSE,
    },
    {
        "Left Click",
        "Select item / Focus panel",
        "Clicking anywhere on a row focuses that panel and selects the clicked item. "
        "Clicking tabs (DASH, STRM, etc.) switches views; clicking [h: HELP] toggles help.",
        HF_ENTRY,
        HS_MOUSE,
    },
    {
        "Double Click",
        "Open detailed inspection pane",
        "Double-clicking any stream, process, or FPS module immediately opens its "
        "full inspection pane or parameter editor.",
        HF_ENTRY,
        HS_MOUSE,
    },
    {
        "Mouse Wheel",
        "Scroll focused list or table",
        "Spinning the mouse scroll wheel scrolls the hovered list up or down without "
        "needing to press keyboard arrow keys.",
        HF_ENTRY,
        HS_MOUSE,
    },
    {
        "Header Click",
        "Click column header to sort",
        "Clicking on any column heading (e.g. NAME, TYPE, HZ, CPU%) automatically sorts "
        "the table by that column. Clicking again reverses the sort.",
        HF_ENTRY,
        HS_MOUSE,
    },
    {
        "Border Drag",
        "Click and drag borders to resize",
        "Clicking on any panel separator border and dragging with the mouse dynamically "
        "resizes panel proportions in real time.",
        HF_ENTRY,
        HS_MOUSE,
    },

    /* =========================================================
     * 8. Theme & Colors
     * ========================================================= */
    {
        NULL,
        "Theme & Color Legend",
        "Color coding conventions and selectable theme styles in milk-CTRL. "
        "Supports 10 dark, light, high-contrast, and vibrant palettes.",
        HF_SECTION,
        HS_COLORS,
    },
    {
        "F8 / ^T",
        "Open theme selector popup (↑/↓ to choose, ESC or 1s to close)",
        "Brings up a theme selector popup with live palette swatches. Select with "
        "Up/Down arrows or ^T; closes after 1s inactivity or on ESC. Available: "
        "dark, night, accessible, light, nordic, dracula, solarized-dark, "
        "solarized-light, monokai, matrix.",
        HF_ENTRY,
        HS_COLORS,
    },
    {
        "Legend",
        "System color semantics",
        "Stream (SHM)  |  Process (procinfo)  |  FPS module\n"
        "Active / Running  |  Idle / Paused\n"
        "Stale / Warning  |  Error / Crashed / Signal Kill",
        HF_ENTRY | HF_COLORS,
        HS_COLORS,
    },
