// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_help_data.c
 * @brief Static command reference database and introductory guide data
 */

#include "overview_help_data.h"

/* clang-format off */
const help_entry_t g_help_entries[] =
{
    /* =========================================================
     * 0. Introduction & Overview
     * ========================================================= */
    {
        NULL,
        "Introduction: What is milk-CTRL & Features",
        "milk-CTRL is the unified, real-time diagnostic and control dashboard for the "
        "milk framework. It monitors and orchestrates three core shared-memory pillars: "
        "ImageStreamIO (streams), FPS (compute modules & parameters), and processinfo "
        "(telemetry & loop rates). Press 1, 'i', or ENTER to open the interactive guide.",
        HF_SECTION,
        HS_INTRO,
    },
    {
        "1 / i",
        "Display full interactive introduction guide",
        "Toggles the full-screen interactive Introduction guide explaining what milk-CTRL "
        "is, its three core architectural pillars, primary capabilities, and operational "
        "workflows.",
        HF_ENTRY,
        HS_INTRO,
    },
    {
        "Overview",
        "High-performance adaptive optics dashboard",
        "Designed for microsecond-latency adaptive optics and image processing pipelines. "
        "Provides single-pane-of-glass observability and control across all streams, "
        "processes, and FPS compute units without impacting compute loop latency.",
        HF_ENTRY,
        HS_INTRO,
    },
    {
        "Pillars",
        "ImageStreamIO, FPS, and processinfo",
        "1. ImageStreamIO: Zero-copy SHM circular buffers and semaphores (/milk/shm/). "
        "2. FPS: Standardized compute units with parameter trees in isolated tmux sessions. "
        "3. processinfo: Real-time telemetry, loop frequencies, and CPU core affinities.",
        HF_ENTRY,
        HS_INTRO,
    },
    {
        "Features",
        "Topology graphs, loops, and control mode",
        "Features include dataflow connection graphs (F6), closed feedback loop detection "
        "(F7), live parameter tree editing (F5), Control Mode actions (press 'c'), regex "
        "filtering ('/'), and command logging ('G').",
        HF_ENTRY,
        HS_INTRO,
    },

    /* =========================================================
     * 1. Global & Navigation
     * ========================================================= */
    {
        NULL,
        "Global & Navigation",
        "Global controls available from any view or panel. Includes view switching "
        "(F2-F7), panel focus cycling (TAB), real-time scan rate tuning, display pause, "
        "regex filtering, snapshot export, and application exit.",
        HF_SECTION,
        HS_NAV,
    },
    {
        "F2 - F7",
        "Switch views (DASH, STRM, PROC, FPS, CONN, LOOPS)",
        "Switches full-screen or grid dashboard view: F2=Dashboard (all panels), "
        "F3=Streams (SHM), F4=Processes (procinfo), F5=FPS (module list & param tree), "
        "F6=Node Graph (dataflow connections), F7=Feedback Loops (circuit & overlap).",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "^Left/^Right",
        "Cycle views sequentially",
        "Cycles forward or backward through the 6 dashboard view modes "
        "(DASH -> STRM -> PROC -> FPS -> CONN -> LOOPS). Equivalent to pressing F2 through F7.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "TAB",
        "Cycle active panel focus",
        "Moves active focus between visible panels: Streams -> Processes -> FPS -> Graph "
        "in Dashboard view; or between FPS list and Parameter tree in F5 view.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "Left / Right",
        "Panel focus / horizontal table scroll",
        "In Dashboard view, moves focus between adjacent panels. When a wide table "
        "exceeds screen width, scrolls table columns horizontally.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "UP / DOWN",
        "Navigate rows in focused panel (or j/k)",
        "Moves the row selection cursor up or down in the currently focused list or "
        "table. Vim navigation keys j and k are also supported.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "PgUp / PgDn",
        "Scroll page up / down",
        "Scrolls the focused list up or down by one full page of rows for rapid "
        "navigation in large stream or process tables.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "Home / End",
        "Jump to first / last item",
        "Instantly jumps the selection cursor to the very top (first item) or "
        "bottom (last item) of the currently focused list.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "/",
        "Filter panel items (regex search)",
        "Opens interactive regex filter for focused panel (STREAMS, PROCESSINFO, or FPS). "
        "Displays matching items with FILTER ON badge in panel border & header. Enter applies; "
        "Esc clears.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "f",
        "Toggle regex filter ON / OFF",
        "Toggles regex filtering on or off for focused panel without losing the filter query. "
        "When OFF, all panel items are displayed while retaining query for quick resumption.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "SPACE",
        "Freeze selection highlight",
        "Freezes the selection cursor on the current item name, preventing it from "
        "jumping when tables re-sort or items appear/disappear during live scans.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "+ / -",
        "Adjust scan rate (faster / slower)",
        "Increases or decreases the background shared-memory scan interval. Faster rates "
        "give lower latency; slower rates reduce CPU and terminal bandwidth.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "F",
        "Pause / resume display updates",
        "Freezes all UI rendering at the current frame without stopping background "
        "processes. Press F again to resume live real-time rendering.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "W",
        "Export snapshot to file (/tmp)",
        "Writes a complete timestamped diagnostic report of all active streams, "
        "processes, and FPS modules to /tmp/milk-CTRL_snapshot_<timestamp>.txt.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "F8 / ^T",
        "Cycle color theme",
        "Cycles through available color themes (dark, night, accessible, light, nordic). "
        "Can also be cycled by clicking the theme badge in the status bar.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "c",
        "Toggle Control Mode ON / OFF",
        "Toggles safety interlock for administrative commands. When ON, enables "
        "process signaling (kill, pause, step, exit) and stream deletion. Press 'c' to toggle.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "h",
        "Toggle this help overlay",
        "Opens or closes this interactive help overlay. While open, use UP/DOWN to browse "
        "commands and ENTER to expand or collapse panel sections.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "q / x",
        "Quit milk-CTRL",
        "Exits milk-CTRL cleanly and restores terminal settings. Background streams "
        "and shared-memory processes remain running untouched.",
        HF_ENTRY,
        HS_NAV,
    },

    /* =========================================================
     * 2. Streams Panel (STRM)
     * ========================================================= */
    {
        NULL,
        "Streams Panel (STRM)",
        "ImageStreamIO shared memory panel. Displays all active image streams (*.im.shm) "
        "in the SHM directory. Shows dimensions, datatype, write timestamps, update rates "
        "(Hz), throughput (MB/s), and reader semaphores.",
        HF_SECTION,
        HS_STREAMS,
    },
    {
        "D / ENTER",
        "Toggle detailed inspection pane",
        "Opens the stream inspector showing full metadata, dimensions, datatype, circular "
        "buffer depth, memory address, shared semaphore count, and active writer PID.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "s",
        "Sort streams by Name (A-Z)",
        "Sorts the stream table alphabetically by stream name. Press '[' to toggle "
        "between ascending (A-Z) and descending (Z-A) order.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "S",
        "Sort streams by Activity (Hz)",
        "Sorts the stream table by update frequency (Hz), placing the most actively "
        "written streams at the top of the list.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "A",
        "Sort streams by Ancestry / topology",
        "Sorts streams in topological dataflow order from upstream source inputs down "
        "to downstream consumer outputs based on dependency graph analysis.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "< / > / ]",
        "Cycle active sort column",
        "Cycles the active sort column through Name, Type, Size, Update Rate (Hz), "
        "Bandwidth (MB/s), and Writer PID.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "[",
        "Toggle sort direction (Asc / Desc)",
        "Reverses the sort order between ascending and descending for whichever "
        "column is currently active.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "Shift+Left/Rt",
        "Select column for layout hiding",
        "Moves the highlighted column selection cursor left or right across table "
        "headers so you can inspect column properties or toggle its visibility.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "t / T",
        "Toggle column visibility (hide/show)",
        "Hides or reveals the currently selected table column. Useful for customizing "
        "table density on compact or high-resolution displays.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "d",
        "Toggle compact layout mode",
        "Automatically hides secondary metadata columns (semaphores, memory size) "
        "to fit the table cleanly into narrow terminal windows.",
        HF_ENTRY,
        HS_STREAMS,
    },
    {
        "DEL",
        "Delete stream shared memory file",
        "Unlinks and removes the selected stream from /dev/shm. Any processes holding "
        "the file descriptor remain mapped until closed. Requires Control Mode ON.",
        HF_ENTRY | HF_CTRL_MODE,
        HS_STREAMS,
    },

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
};
/* clang-format on */

const int g_help_total = (int) (sizeof(g_help_entries) / sizeof(g_help_entries[0]));

/* clang-format off */
const intro_item_t g_intro_items[] = {
    { IL_HEADER,    "■ WHAT IS milk-CTRL?", NULL },
    { IL_TEXT,      NULL, "milk-CTRL is the unified, real-time diagnostic and control dashboard" },
    { IL_TEXT,      NULL, "for the milk (Modular Image Processing Library Kernel) framework." },
    { IL_TEXT,      NULL, "It provides a high-performance, flicker-free terminal interface" },
    { IL_TEXT,      NULL, "specially designed for mission-critical astronomical adaptive optics" },
    { IL_TEXT,      NULL, "(AO), wavefront control, and image processing pipelines." },
    { IL_BLANK,     NULL, NULL },
    { IL_TEXT,      NULL, "Built on zero-copy shared memory, milk-CTRL aggregates live system" },
    { IL_TEXT,      NULL, "state into a single dashboard without adding IPC overhead or latency." },
    { IL_BLANK,     NULL, NULL },

    { IL_HEADER,    "■ THE THREE CORE PILLARS", NULL },
    { IL_SUBHEADER, "1. ImageStreamIO (Streams — STRM / F3)", NULL },
    { IL_BULLET,    "• Zero-Copy SHM: ", "Circular buffers & image arrays in /milk/shm/." },
    { IL_BULLET,    "• Ultra-High Speed: ", "Transfers multi-dimensional data up to 10+ kHz." },
    { IL_BULLET,    "• Data Diversity: ", "Supports FLOAT, DOUBLE, UINT16, INT32, etc." },
    { IL_BULLET,    "• Synchronization: ", "POSIX read/write semaphores coordinate clients." },
    { IL_BLANK,     NULL, NULL },

    { IL_SUBHEADER, "2. FPS — Function Processing System (FPS — FPS / F5)", NULL },
    { IL_BULLET,    "• Compute Units: ", "Modular compute engines for algorithms & AO." },
    { IL_BULLET,    "• Tmux Isolation: ", "Runs compute tasks in isolated tmux sessions." },
    { IL_BULLET,    "• Dual Loops: ", "Separates parameter config (conf) from processing (run)." },
    { IL_BULLET,    "• Parameter Trees: ", "Hierarchical parameter directories editable live." },
    { IL_BLANK,     NULL, NULL },

    { IL_SUBHEADER, "3. processinfo (Processes — PROC / F4)", NULL },
    { IL_BULLET,    "• Telemetry: ", "Heartbeat registration, loop frequencies (Hz) & jitter." },
    { IL_BULLET,    "• Resources: ", "CPU core pinning masks, RSS memory, & cycle times." },
    { IL_BULLET,    "• Lifecycle: ", "Tracks RUNNING, PAUSED, STOPPED, & CRASHED states." },
    { IL_BLANK,     NULL, NULL },

    { IL_HEADER,    "■ WHAT CAN milk-CTRL DO?", NULL },
    { IL_BULLET,    "• Multi-Panel (F2): ", "View Streams, Procs, FPS, & Graph together." },
    { IL_BULLET,    "• Full Monitors (F3-F5): ", "Deep-dive into streams, telemetry, or FPS." },
    { IL_BULLET,    "• Data Lineage (F6): ", "Maps upstream producers & downstream consumers." },
    { IL_BULLET,    "• Feedback Loops (LOOPS/F7): ", "Detects closed directed loops (WFS -> DM)." },
    { IL_BULLET,    "• Control Mode ('c'): ", "Start/stop FPS, signal procs, or delete streams." },
    { IL_BULLET,    "• Parameter Editing: ", "Browse and edit FPS variables with limits check." },
    { IL_BULLET,    "• Filtering & Search: ", "Regex filter ('/'), toggle ('f'), freeze (SPACE)." },
    { IL_BULLET,    "• Snapshots & Logging: ", "Save state snapshots ('W') & command ring ('G')." },
    { IL_BLANK,     NULL, NULL },

    { IL_HEADER,    "■ QUICK START & KEYSTROKES", NULL },
    { IL_KEY,       "  1 / i", "Toggle this Introduction guide on or off" },
    { IL_KEY,       "  2 / k", "Switch to the Controls & Keybindings reference" },
    { IL_KEY,       "  F2 - F7", "Switch view panels (DASH, STRM, PROC, FPS, CONN, LOOPS)" },
    { IL_KEY,       "  TAB", "Cycle active panel focus (Streams -> Procs -> FPS -> Graph)" },
    { IL_KEY,       "  ↑ / ↓ (j / k)", "Navigate items in focused list, or scroll this guide" },
    { IL_KEY,       "  c", "Toggle Control Mode ON to enable management actions" },
    { IL_KEY,       "  /", "Search topics in help, or regex filter in dashboard" },
    { IL_KEY,       "  f", "Toggle regex filter ON/OFF for focused panel" },
    { IL_KEY,       "  SPACE", "Freeze selection highlight during rapid live updates" },
    { IL_KEY,       "  ESC", "Close help overlay or exit current prompt" },
    { IL_KEY,       "  q / x", "Quit milk-CTRL cleanly" },
};
/* clang-format on */

const int g_intro_total = (int) (sizeof(g_intro_items) / sizeof(g_intro_items[0]));

/**
 * ov_help_nb_sections - get the total number of help sections.
 *
 * Return: Number of help sections (HS_COUNT).
 */
int ov_help_nb_sections(void)
{
    return HS_COUNT;
}

/**
 * ov_help_section_name - get human-readable name of a section.
 * @sec: Section index.
 *
 * Return: Short section name string.
 */
const char *ov_help_section_name(
    int sec)
{
    switch (sec)
    {
    case HS_INTRO:
        return "Intro";
    case HS_NAV:
        return "Global & Nav";
    case HS_STREAMS:
        return "Streams";
    case HS_PROCS:
        return "Processes";
    case HS_FPS:
        return "FPS";
    case HS_GRAPH:
        return "Graph";
    case HS_CMDLOG:
        return "CmdLog";
    case HS_MOUSE:
        return "Mouse";
    case HS_COLORS:
        return "Colors";
    default:
        return "Help";
    }
}

/**
 * ov_help_focus_section - map layout focus to its corresponding help section.
 * @focus: Active panel focus enum.
 *
 * Return: Corresponding section index (HS_STREAMS, HS_PROCS, etc.).
 */
int ov_help_focus_section(
    ov_focus_t focus)
{
    switch (focus)
    {
    case OV_FOCUS_STREAMS:
        return HS_STREAMS;
    case OV_FOCUS_PROCS:
        return HS_PROCS;
    case OV_FOCUS_FPS:
        return HS_FPS;
    case OV_FOCUS_GRAPH:
        return HS_GRAPH;
    default:
        return HS_NAV;
    }
}

/**
 * ov_help_section_tag - return short 4-letter panel tag for search results.
 * @sec: Section index.
 *
 * Return: Short uppercase badge string.
 */
const char *ov_help_section_tag(
    int sec)
{
    switch (sec)
    {
    case HS_INTRO:
        return "INTR";
    case HS_NAV:
        return "NAV";
    case HS_STREAMS:
        return "STRM";
    case HS_PROCS:
        return "PROC";
    case HS_FPS:
        return "FPS";
    case HS_GRAPH:
        return "CONN";
    case HS_CMDLOG:
        return "LOG";
    case HS_MOUSE:
        return "MOUS";
    case HS_COLORS:
        return "COLR";
    default:
        return "HELP";
    }
}

/**
 * ov_help_section_color - return semantic theme color for section badge.
 * @sec: Section index.
 *
 * Return: Theme ov_rgb_t color.
 */
ov_rgb_t ov_help_section_color(
    int sec)
{
    switch (sec)
    {
    case HS_INTRO:
        return (ov_rgb_t) { 255, 215, 80 };
    case HS_NAV:
        return OV_FG_TITLE;
    case HS_STREAMS:
        return OV_FG_STREAM;
    case HS_PROCS:
        return OV_FG_PROC;
    case HS_FPS:
        return OV_FG_FPS;
    case HS_GRAPH:
        return OV_FG_CONN;
    case HS_CMDLOG:
        return (ov_rgb_t) { 180, 210, 170 };
    case HS_MOUSE:
        return OV_FG_WARN;
    case HS_COLORS:
        return (ov_rgb_t) { 230, 130, 255 };
    default:
        return OV_FG_DIM;
    }
}
