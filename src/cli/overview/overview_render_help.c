// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help.c
 * @brief   Structured, panel-grouped help overlay with detailed contextual help
 *
 * Organizes help into sections that match the milk-CTRL GUI panels.
 * Keystrokes are rendered with standardized bold formatting and a distinct
 * shade for commands requiring Control Mode ON. When an item is selected,
 * a dedicated detail pane displays in-depth explanations and live target
 * information for the currently selected GUI item.
 */

#include "overview_render_internal.h"

/* ---- Help content definition ---- */

/** Flag values for help entries */
#define HF_SECTION 1   /* Section header row                         */
#define HF_ENTRY 2     /* Standard keystroke command entry           */
#define HF_CTRL_MODE 4 /* Requires Control Mode ON (press 'c')       */
#define HF_COLORS 8    /* Render as theme color legend               */

typedef struct
{
    const char *key;     /* Keystroke label (e.g. "F2 - F6", "k", "DEL") */
    const char *label;   /* Concise summary for list view                */
    const char *detail;  /* In-depth explanation for detail pane         */
    int         flags;   /* Bitmask of HF_* flags                        */
    int         section; /* Parent section index                         */
} help_entry_t;

/* Section indices matching GUI panels */
enum
{
    HS_INTRO = 0, /* Introduction & Overview    */
    HS_NAV,       /* Global & Navigation        */
    HS_STREAMS,   /* Streams Panel (STRM)       */
    HS_PROCS,     /* Processes Panel (PROC)     */
    HS_FPS,       /* FPS Panel (FPS)            */
    HS_GRAPH,     /* Graph & Lineage (CONN)     */
    HS_CMDLOG,    /* Command Log & Display      */
    HS_MOUSE,     /* Mouse Interactions         */
    HS_COLORS,    /* Theme & Colors             */
    HS_COUNT
};

/* clang-format off */
static const help_entry_t HELP[] =
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
        "Filter items (regex search)",
        "Opens interactive regex filter. Displays only matching items with a fast-blinking "
        "FILTER ON indicator in header and status bar. Enter applies; Esc clears.",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "f",
        "Toggle regex filter ON / OFF",
        "Toggles regex filtering on or off without erasing the filter query. When OFF, "
        "all items are displayed while retaining the filter query for quick re-activation.",
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
        "Supports dark, observatory red, high-contrast, paper light, and nordic themes.",
        HF_SECTION,
        HS_COLORS,
    },
    {
        "F8 / ^T",
        "Cycle color theme (dark, night, accessible, light, nordic)",
        "Cycles through available color palettes: dark (default slate), night "
        "(observatory dark-adapted red), accessible (colorblind-friendly high-contrast), "
        "light (paper light for daylight/papers), and nordic (arctic slate). Selection "
        "is automatically saved to ~/.milk/ctrl_theme.",
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

static const int HELP_TOTAL = (int) (sizeof(HELP) / sizeof(HELP[0]));

/**
 * help_is_expanded - check if a section is expanded.
 * @lay: layout state
 * @sec: section index (HS_NAV .. HS_COLORS)
 *
 * Return: 1 if expanded, 0 if collapsed.
 */
static inline int help_is_expanded(
    const OV_LAYOUT *lay,
    int              sec)
{
    return (lay->help_expand >> sec) & 1;
}

/**
 * ov_help_nb_sections - return total number of help sections.
 *
 * Return: HS_COUNT
 */
int ov_help_nb_sections(void)
{
    return HS_COUNT;
}

/**
 * ov_help_section_name - human-readable name of a help section.
 * @sec: section index
 *
 * Return: short name string.
 */
static const char *ov_help_section_name(int sec)
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
 * @focus: active panel focus enum
 *
 * Return: section index (HS_STREAMS, HS_PROCS, etc.)
 */
int ov_help_focus_section(ov_focus_t focus)
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
 * @sec: section index
 */
static const char *ov_help_section_tag(int sec)
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
 * ov_help_section_color - return theme color for section badge.
 * @sec: section index
 */
static ov_rgb_t ov_help_section_color(int sec)
{
    switch (sec)
    {
    case HS_INTRO:
        return (ov_rgb_t){ 255, 215, 80 };
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
        return (ov_rgb_t){ 180, 210, 170 };
    case HS_MOUSE:
        return OV_FG_WARN;
    case HS_COLORS:
        return (ov_rgb_t){ 230, 130, 255 };
    default:
        return OV_FG_DIM;
    }
}

typedef struct
{
    int index; /* Index in HELP[] array */
    int score; /* Composite relevance score */
} help_search_match_t;

static int compare_search_matches(
    const void *a,
    const void *b)
{
    const help_search_match_t *ma = (const help_search_match_t *) a;
    const help_search_match_t *mb = (const help_search_match_t *) b;
    if (mb->score != ma->score)
    {
        return mb->score - ma->score; /* descending score */
    }
    return ma->index - mb->index;     /* stable tie-breaker */
}

/**
 * help_score_entry - score a help entry against a search query.
 * @entry: help entry to evaluate
 * @query: user search string
 *
 * Supports multi-token queries where all tokens must match (AND-logic).
 *
 * Return: score >= 0 (0 means no match).
 */
static int help_score_entry(
    const help_entry_t *entry,
    const char         *query)
{
    if (entry == NULL || query == NULL || query[0] == '\0')
    {
        return 0;
    }

    char qbuf[64];
    strncpy(qbuf, query, sizeof(qbuf) - 1);
    qbuf[sizeof(qbuf) - 1] = '\0';

    char *tokens[8];
    int   ntok    = 0;
    char *saveptr = NULL;
    char *tok     = strtok_r(qbuf, " \t", &saveptr);
    while (tok != NULL && ntok < 8)
    {
        tokens[ntok++] = tok;
        tok = strtok_r(NULL, " \t", &saveptr);
    }

    if (ntok == 0)
    {
        return 0;
    }

    int         total_score = 0;
    const char *sec_name    = ov_help_section_name(entry->section);
    const char *sec_tag     = ov_help_section_tag(entry->section);

    for (int t = 0; t < ntok; t++)
    {
        const char *w = tokens[t];
        int wlen      = (int) strlen(w);
        if (wlen == 0)
        {
            continue;
        }

        int tok_score = 0;

        /* 1. Keystroke exact or prefix match */
        if (entry->key != NULL)
        {
            if (strcasecmp(entry->key, w) == 0)
            {
                tok_score += 350; /* Exact match on key, e.g. "k" or "F2" */
            }
            else if (strncasecmp(entry->key, w, (size_t) wlen) == 0)
            {
                tok_score += 180; /* Key prefix match */
            }
            else if (strcasestr(entry->key, w) != NULL)
            {
                tok_score += 90;
            }
        }

        /* 2. Label match (primary title summary) */
        if (entry->label != NULL)
        {
            const char *p = strcasestr(entry->label, w);
            if (p != NULL)
            {
                int at_boundary = (p == entry->label || *(p - 1) == ' ' || *(p - 1) == '/' ||
                                   *(p - 1) == '(' || *(p - 1) == '[' || *(p - 1) == '-');
                if (at_boundary)
                {
                    if (p[wlen] == '\0' || p[wlen] == ' ' || p[wlen] == '/' ||
                        p[wlen] == ')' || p[wlen] == ']')
                    {
                        tok_score += 160;
                    }
                    else
                    {
                        tok_score += 120;
                    }
                }
                else
                {
                    tok_score += 60;
                }
            }
        }

        /* 3. Section/topic match */
        if (sec_name != NULL && strcasestr(sec_name, w) != NULL)
        {
            tok_score += 70;
        }
        if (sec_tag != NULL && strcasecmp(sec_tag, w) == 0)
        {
            tok_score += 80;
        }

        /* 4. Detail documentation match */
        if (entry->detail != NULL)
        {
            const char *p = strcasestr(entry->detail, w);
            if (p != NULL)
            {
                tok_score += 35;
            }
        }

        /* Every token must match somewhere (AND logic) */
        if (tok_score == 0)
        {
            return 0;
        }

        total_score += tok_score;
    }

    /* Command entries get a priority boost over section headers */
    if (entry->flags & HF_ENTRY)
    {
        total_score += 15;
    }

    return total_score;
}

/**
 * help_visible_rows - count visible rows and populate mapping array.
 * @lay: layout state (for expand bitmask or active search query)
 * @map: output array mapping visible row index to HELP[] index
 *
 * Return: number of visible rows.
 */
static int help_visible_rows(
    const OV_LAYOUT *lay,
    int             *map)
{
    /* If search query is non-empty, populate map with ranked search matches */
    if (lay->help_search[0] != '\0')
    {
        help_search_match_t matches[128];
        int                 n_matches = 0;

        for (int i = 0; i < HELP_TOTAL; i++)
        {
            int s = help_score_entry(&HELP[i], lay->help_search);
            if (s > 0 && n_matches < 128)
            {
                matches[n_matches].index = i;
                matches[n_matches].score = s;
                n_matches++;
            }
        }

        if (n_matches > 1)
        {
            qsort(matches, (size_t) n_matches, sizeof(help_search_match_t),
                  compare_search_matches);
        }

        for (int i = 0; i < n_matches; i++)
        {
            map[i] = matches[i].index;
        }
        return n_matches;
    }

    int vis = 0;
    for (int i = 0; i < HELP_TOTAL; i++)
    {
        if (HELP[i].flags & HF_SECTION)
        {
            map[vis++] = i;
        }
        else if (HELP[i].flags & (HF_ENTRY | HF_COLORS))
        {
            if (help_is_expanded(lay, HELP[i].section))
            {
                map[vis++] = i;
            }
        }
    }
    return vis;
}

/**
 * ov_help_visible_count - public helper returning number of visible rows.
 * @lay: layout state
 *
 * Return: visible row count.
 */
int ov_help_visible_count(const OV_LAYOUT *lay)
{
    int map[128];
    return help_visible_rows(lay, map);
}

/**
 * ov_help_section_first_vis_row - find visible row index for a section header.
 * @lay: layout state
 * @sec: section index
 *
 * Return: visible row index in map[], or 0 if not found.
 */
int ov_help_section_first_vis_row(
    const OV_LAYOUT *lay,
    int              sec)
{
    int map[128];
    int nvis = help_visible_rows(lay, map);
    for (int vr = 0; vr < nvis; vr++)
    {
        int idx = map[vr];
        if (HELP[idx].section == sec && (HELP[idx].flags & HF_SECTION))
        {
            return vr;
        }
    }
    return 0;
}

/**
 * ov_help_open - open help overlay with contextual initial focus and expansion.
 * @lay: layout state (modified)
 */
void ov_help_open(OV_LAYOUT *lay)
{
    lay->show_help          = 1;
    lay->help_expand        = 0;
    lay->filter_editing     = 0;
    lay->help_search[0]     = '\0';
    lay->help_search_active = 0;
    lay->help_search_cursor = 0;
    lay->help_mode          = 0;
    lay->help_intro_scroll  = 0;
    lay->help_sel           = 0;
}

/**
 * ov_help_toggle_at - toggle expansion for section header at visible row.
 * @lay:     layout state (help_expand bitmask modified)
 * @vis_row: 0-based visible row index
 *
 * Return: section index if toggled, or -1 if row is not a section header.
 */
int ov_help_toggle_at(
    OV_LAYOUT *lay,
    int        vis_row)
{
    if (lay->help_search[0] != '\0')
    {
        return -1;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    if (vis_row < 0 || vis_row >= nvis)
    {
        return -1;
    }

    int idx = map[vis_row];
    if (!(HELP[idx].flags & HF_SECTION))
    {
        return -1;
    }

    int sec = HELP[idx].section;
    lay->help_expand ^= (1U << sec);
    return sec;
}

/**
 * ov_help_expand_at - expand or collapse section at visible row.
 * @lay:     layout state (help_expand bitmask modified)
 * @vis_row: 0-based visible row index
 * @expand:  1 to expand, 0 to collapse
 *
 * Return: section index if modified, or -1 if row is not eligible.
 */
int ov_help_expand_at(
    OV_LAYOUT *lay,
    int        vis_row,
    int        expand)
{
    if (lay->help_search[0] != '\0')
    {
        return -1;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    if (vis_row < 0 || vis_row >= nvis)
    {
        return -1;
    }

    int idx = map[vis_row];
    int sec = HELP[idx].section;

    if (expand)
    {
        if (HELP[idx].flags & HF_SECTION)
        {
            if (!help_is_expanded(lay, sec))
            {
                lay->help_expand |= (1U << sec);
                return sec;
            }
            else
            {
                int new_nvis = ov_help_visible_count(lay);
                if (lay->help_sel + 1 < new_nvis)
                {
                    lay->help_sel++;
                }
                return sec;
            }
        }
    }
    else
    {
        if (HELP[idx].flags & HF_SECTION)
        {
            if (help_is_expanded(lay, sec))
            {
                lay->help_expand &= ~(1U << sec);
                return sec;
            }
        }
        else
        {
            /* On child item: collapse parent section and land on section header */
            lay->help_expand &= ~(1U << sec);
            lay->help_sel = ov_help_section_first_vis_row(lay, sec);
            return sec;
        }
    }

    return -1;
}

/**
 * ov_help_get_rect - compute bounding box for help overlay.
 * @lay: layout state
 * @pr:  output top row (1-based)
 * @pc:  output left column (1-based)
 * @ph:  output height in rows
 * @pw:  output width in columns
 *
 * Takes the whole available terminal space: starts below the dedicated tab bar
 * (row 3) and extends across the entire terminal width to the row above the status bar.
 */
static void ov_help_get_rect(
    const OV_LAYOUT *lay,
    int             *pr,
    int             *pc,
    int             *ph,
    int             *pw)
{
    int W = lay->term_cols;
    int H = lay->term_rows;

    *pc = 1;
    *pw = W;

    if (H <= 6)
    {
        *pr = 1;
        *ph = H;
    }
    else
    {
        *pr = 3;
        *ph = H - 3;
    }
}

/**
 * ov_help_handle_click - handle mouse click within help overlay.
 * @lay: layout state
 * @mr:  clicked terminal row (1-based)
 * @mc:  clicked terminal column (1-based)
 *
 * Return: 1 if click was handled inside help overlay, 0 otherwise.
 */
int ov_help_handle_click(
    OV_LAYOUT *lay,
    int        mr,
    int        mc)
{
    int pr, pc, ph, pw;
    ov_help_get_rect(lay, &pr, &pc, &ph, &pw);

    /* Click outside popup -> close help overlay */
    if (mr < pr || mr >= pr + ph || mc < pc || mc >= pc + pw)
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        if (mr == lay->r_tabs.row)
        {
            int tx = 1;
            for (int v = 0; v < OV_VIEW_COUNT; v++)
            {
                int tw = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
                if (mc >= tx && mc < tx + tw)
                {
                    lay->view = (ov_view_t) v;
                    break;
                }
                tx += tw;
            }
        }
        return 1;
    }

    /* Click close button area on top border or header row */
    if ((mr == pr && mc >= pc + pw - 6) || (mr == pr + 1 && mc >= pc + pw - 6))
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        return 1;
    }

    /* Click on header row (pr + 1): Mode selector tabs */
    if (mr == pr + 1)
    {
        int tab1_w = (pw >= 100) ? 26 : 14;
        int tab2_w = (pw >= 100) ? 32 : 20;
        if (mc >= pc + 2 && mc < pc + 2 + tab1_w)
        {
            lay->help_mode         = 1;
            lay->help_intro_scroll = 0;
            return 1;
        }
        if (mc >= pc + 2 + tab1_w && mc < pc + 2 + tab1_w + tab2_w)
        {
            lay->help_mode = 0;
            return 1;
        }
    }

    /* If in Intro mode (help_mode == 1), clicks in body do not alter list */
    if (lay->help_mode == 1)
    {
        return 1;
    }

    /* Click on search bar area on row pr + 2 (in Controls mode) */
    if (mr == pr + 2 && mc >= pc + 1 && mc < pc + pw - 1)
    {
        if (lay->help_search[0] != '\0' && mc >= pc + pw - 16)
        {
            /* Clicked on [ESC: clear] button */
            lay->help_search[0]     = '\0';
            lay->help_search_cursor = 0;
            lay->help_search_active = 0;
            lay->help_sel           = 0;
        }
        else
        {
            /* Clicked on search input box */
            lay->help_search_active = 1;
            lay->help_search_cursor = (int) strlen(lay->help_search);
        }
        return 1;
    }

    /* Detail pane height and split line */
    int detail_h = (ph >= 36) ? 10 : ((ph >= 28) ? 8 : ((ph >= 22) ? 6 : 5));
    int split_r  = (pr + ph - 1) - detail_h;
    int list_top = pr + 4;
    int list_h   = split_r - list_top;

    /* Check if click is inside the list area */
    if (mr >= list_top && mr < split_r)
    {
        int map[128];
        int nvis = help_visible_rows(lay, map);

        int sel = lay->help_sel;
        if (sel < 0)
        {
            sel = 0;
        }
        if (sel >= nvis)
        {
            sel = nvis - 1;
        }

        int scroll = 0;
        if (sel >= list_h)
        {
            scroll = sel - list_h + 1;
        }

        int vis_row = (mr - list_top) + scroll;
        if (vis_row >= 0 && vis_row < nvis)
        {
            int idx = map[vis_row];
            if (HELP[idx].section == HS_INTRO && (lay->help_sel == vis_row || mr == list_top))
            {
                /* Clicking on Introduction entry/header opens full intro guide */
                lay->help_mode         = 1;
                lay->help_intro_scroll = 0;
                return 1;
            }
            if (lay->help_search[0] == '\0' && lay->help_sel == vis_row)
            {
                /* Clicking selected header toggles expansion */
                ov_help_toggle_at(lay, vis_row);
            }
            else
            {
                lay->help_sel = vis_row;
            }
            return 1;
        }
    }

    return 0;
}

/**
 * ov_help_print_wrapped - word-wrap and print text within a specified bounding box.
 * @text:      text string to print
 * @row:       starting row
 * @col:       left column
 * @max_w:     maximum visible column width
 * @max_lines: maximum lines to print
 * @fg:        foreground color
 * @bg:        background color
 */
static void ov_help_print_wrapped(
    const char *text,
    int         row,
    int         col,
    int         max_w,
    int         max_lines,
    ov_rgb_t    fg,
    ov_rgb_t    bg)
{
    if (text == NULL || max_w <= 0 || max_lines <= 0)
    {
        return;
    }

    const char *p             = text;
    int         lines_printed = 0;

    while (*p != '\0' && lines_printed < max_lines)
    {
        /* Skip leading whitespace on new line */
        while (*p == ' ')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }

        int len        = 0;
        int last_space = -1;
        while (p[len] != '\0' && p[len] != '\n' && len < max_w)
        {
            if (p[len] == ' ')
            {
                last_space = len;
            }
            len++;
        }

        int line_len = len;
        if (p[len] == '\n')
        {
            line_len = len;
            len++;
        }
        else if (p[len] != '\0' && last_space > 0)
        {
            line_len = last_space;
            len      = last_space + 1;
        }

        ov_buf_pos(row + lines_printed, col);
        ov_theme_bg(bg);
        ov_theme_fg(fg);
        ov_buf_printf("%.*s", line_len, p);

        int pad = max_w - line_len;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }

        p += len;
        lines_printed++;
    }

    /* Clear any remaining allocated lines */
    for (int r = lines_printed; r < max_lines; r++)
    {
        ov_buf_pos(row + r, col);
        ov_theme_bg(bg);
        ov_buf_hline(' ', max_w);
    }
}

/**
 * ov_help_render_detail - render in-depth help and target info for selected item.
 * @lay:      layout state
 * @m:        system model
 * @entry:    currently highlighted help entry
 * @split_r:  row of separator line
 * @pc:       left column of help box
 * @pw:       width of help box
 * @detail_h: height of detail pane
 */
static void ov_help_render_detail(
    const OV_LAYOUT    *lay,
    const OV_MODEL     *m,
    const help_entry_t *entry,
    int                 split_r,
    int                 pc,
    int                 pw,
    int                 detail_h)
{
    int inner_w = pw - 4;
    int col     = pc + 2;

    /* Row 1: Header (Key / Command title + Status badges) */
    ov_buf_pos(split_r + 1, col);
    ov_theme_bg(OV_BG_PANEL);

    if (entry->flags & HF_SECTION)
    {
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        if (entry->section == HS_INTRO)
        {
            ov_buf_printf("■ INTRODUCTION: %s", entry->label);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            const char *hint = "[Press 1, 'i', or ENTER to open full interactive guide]";
            int rem = inner_w - (16 + (int) strlen(entry->label) + (int) strlen(hint));
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("%s", hint);
        }
        else
        {
            ov_buf_printf("■ PANEL OVERVIEW: %s", entry->label);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            const char *hint = help_is_expanded(lay, entry->section)
                                   ? "[Press ← / ENTER to collapse]"
                                   : "[Press → / ENTER to expand]";
            int rem = inner_w - (18 + (int) strlen(entry->label) + (int) strlen(hint));
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%s", hint);
        }
    }
    else if (entry->flags & HF_ENTRY)
    {
        ov_buf_bold();
        if (entry->flags & HF_CTRL_MODE)
        {
            if (lay->ctrl_mode)
            {
                ov_buf_fg(255, 95, 75);
                ov_buf_printf("Key: [ %s ]", entry->key);
                ov_buf_fg(255, 80, 80);
                ov_buf_printf("  ⚡ CONTROL MODE: ON (Ready)");
            }
            else
            {
                ov_buf_fg(205, 140, 50);
                ov_buf_printf("Key: [ %s ]", entry->key);
                ov_buf_fg(205, 140, 50);
                ov_buf_printf("  🔒 REQUIRES CONTROL MODE (Press 'c')");
            }
        }
        else
        {
            ov_buf_fg(130, 205, 255);
            ov_buf_printf("Key: [ %s ]", entry->key);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  (Standard Command)");
        }
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);

        /* Section scope badge on right */
        const char *sname   = ov_help_section_name(entry->section);
        int         key_len = (int) strlen(entry->key) + 9;
        int         badge_l = (int) strlen(sname) + 10;
        int         rem     = inner_w - (key_len + 35 + badge_l);
        if (rem > 0)
        {
            ov_buf_hline(' ', rem);
        }
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("[Panel: %s]", sname);
    }
    else
    {
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("■ %s", entry->label);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_hline(' ', inner_w - (2 + (int) strlen(entry->label)));
    }

    /* Rows 2..N: Word-wrapped descriptive detail */
    int text_lines = detail_h - 3;
    if (text_lines > 0)
    {
        ov_help_print_wrapped(entry->detail, split_r + 2, col, inner_w, text_lines,
                              OV_FG_TEXT, OV_BG_PANEL);
    }

    /* Last row: Live Target info for the currently selected GUI item */
    int target_row = split_r + detail_h - 1;
    ov_buf_pos(target_row, col);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_bold();
    ov_buf_printf("Target: ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    int  target_avail = inner_w - 8;
    char tbuf[160];
    tbuf[0] = '\0';
    ov_rgb_t target_fg = OV_FG_DIM;

    if (entry->section == HS_INTRO)
    {
        target_fg = OV_FG_TITLE;
        snprintf(tbuf, sizeof(tbuf),
                 "Intro Guide: Press '1', 'i', or ENTER to open the interactive introduction");
    }
    else if (entry->section == HS_STREAMS)
    {
        int si = ov_get_selected_stream_idx(lay, m);
        if (si >= 0 && si < m->nb_streams)
        {
            const OV_STREAM *s = &m->streams[si];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t){ 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will delete stream '%s' (%s %s) from SHM",
                         s->name, s->size_str, render_dtype(s->datatype));
            }
            else
            {
                target_fg = OV_FG_STREAM;
                snprintf(tbuf, sizeof(tbuf), "Selected stream '%s' (%s %s, %.1f Hz)",
                         s->name, s->size_str, render_dtype(s->datatype), s->update_hz);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No stream selected (Streams panel is empty)");
        }
    }
    else if (entry->section == HS_PROCS)
    {
        int pi = ov_get_selected_proc_idx(lay, m);
        if (pi >= 0 && pi < m->nb_procs)
        {
            const OV_PROC *p = &m->procs[pi];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t){ 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will signal process '%s' (PID %d, %s)",
                         p->name, (int) p->PID, p->statusmsg);
            }
            else
            {
                target_fg = OV_FG_PROC;
                snprintf(tbuf, sizeof(tbuf), "Selected process '%s' (PID %d, CPU %.1f%%, %s)",
                         p->name, (int) p->PID, p->cpu_used, p->statusmsg);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No process selected (Processes panel is empty)");
        }
    }
    else if (entry->section == HS_FPS)
    {
        int fi = ov_get_selected_fps_idx(lay, m);
        if (fi >= 0 && fi < m->nb_fps)
        {
            const OV_FPS *f = &m->fps[fi];
            if (entry->flags & HF_CTRL_MODE)
            {
                target_fg = (ov_rgb_t){ 255, 95, 75 };
                snprintf(tbuf, sizeof(tbuf), "Will control FPS module '%s' (run=%s, conf=%s)",
                         f->name, f->run_alive ? "ON" : "OFF", f->conf_alive ? "ON" : "OFF");
            }
            else
            {
                target_fg = OV_FG_FPS;
                snprintf(tbuf, sizeof(tbuf),
                         "Selected FPS module '%s' (run=%s, conf=%s, RSS: %ld KB)",
                         f->name, f->run_alive ? "ON" : "OFF", f->conf_alive ? "ON" : "OFF",
                         (long) f->mem_rss_kb);
            }
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "No FPS module selected");
        }
    }
    else if (entry->section == HS_GRAPH)
    {
        if (m != NULL && lay->sel_graph >= 0 && lay->sel_graph < m->nb_nodes)
        {
            const OV_NODE *n = &m->nodes[lay->sel_graph];
            target_fg = OV_FG_CONN;
            snprintf(tbuf, sizeof(tbuf), "Selected node '%s' (type: %s)", n->name,
                     (n->type == OV_NODE_STREAM) ? "Stream" :
                     ((n->type == OV_NODE_PROC) ? "Process" : "FPS"));
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "Node graph cursor active");
        }
    }
    else
    {
        /* Global & Display sections: show active GUI selection */
        int sel_s = ov_get_selected_stream_idx(lay, m);
        int sel_p = ov_get_selected_proc_idx(lay, m);
        int sel_f = ov_get_selected_fps_idx(lay, m);
        if (lay->focus == OV_FOCUS_STREAMS && m && sel_s >= 0 && sel_s < m->nb_streams)
        {
            target_fg = OV_FG_STREAM;
            snprintf(tbuf, sizeof(tbuf), "Active selection: Stream '%s' (Panel: Streams)",
                     m->streams[sel_s].name);
        }
        else if (lay->focus == OV_FOCUS_PROCS && m && sel_p >= 0 && sel_p < m->nb_procs)
        {
            target_fg = OV_FG_PROC;
            snprintf(tbuf, sizeof(tbuf), "Active selection: Process '%s' (PID %d) (Panel: Procs)",
                     m->procs[sel_p].name, (int) m->procs[sel_p].PID);
        }
        else if (lay->focus == OV_FOCUS_FPS && m && sel_f >= 0 && sel_f < m->nb_fps)
        {
            target_fg = OV_FG_FPS;
            snprintf(tbuf, sizeof(tbuf), "Active selection: FPS '%s' (Panel: FPS)",
                     m->fps[sel_f].name);
        }
        else
        {
            target_fg = OV_FG_DIM;
            snprintf(tbuf, sizeof(tbuf), "Global action — applies across all dashboard views");
        }
    }

    ov_theme_fg(target_fg);
    ov_buf_printf("%s", tbuf);
    int chars_used = (int) strlen(tbuf);
    int target_pad = target_avail - chars_used;
    if (target_pad > 0)
    {
        ov_buf_hline(' ', target_pad);
    }
}

/* clang-format off */
typedef enum
{
    IL_BLANK = 0,
    IL_HEADER,
    IL_SUBHEADER,
    IL_BULLET,
    IL_TEXT,
    IL_KEY
} intro_line_type_t;

typedef struct
{
    intro_line_type_t type;
    const char       *prefix;
    const char       *text;
} intro_item_t;

static const intro_item_t INTRO_ITEMS[] = {
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
    { IL_KEY,       "  f", "Toggle regex filter ON/OFF without losing query string" },
    { IL_KEY,       "  SPACE", "Freeze selection highlight during rapid live updates" },
    { IL_KEY,       "  ESC", "Close help overlay or exit current prompt" },
    { IL_KEY,       "  q / x", "Quit milk-CTRL cleanly" },
};
/* clang-format on */

/**
 * ov_help_render_intro - render full interactive introduction guide to milk-CTRL.
 * @lay: layout state
 * @pr:  top row (1-based)
 * @pc:  left column (1-based)
 * @ph:  height in rows
 * @pw:  width in columns
 */
static void ov_help_render_intro(
    const OV_LAYOUT *lay,
    int              pr,
    int              pc,
    int              ph,
    int              pw)
{
    const char *title =
        (pw >= 84)
            ? "ABOUT milk-CTRL (1/i: Intro • 2/k: Controls • ↑↓: Scroll • ESC: Close)"
            : ((pw >= 52) ? "ABOUT milk-CTRL (1/i: Intro • 2/k: Controls • ESC: Close)"
                          : "ABOUT milk-CTRL");
    ov_draw_panel_border(pr, pc, ph, pw, title, OV_FG_TITLE, 1, 0);

    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    int inner_w = pw - 4;
    int col     = pc + 2;

    /* Header Row (pr + 1): Mode selector tabs */
    ov_buf_pos(pr + 1, col);
    ov_theme_bg(OV_BG_PANEL);

    int tab1_w = (pw >= 100) ? 26 : 14;
    int tab2_w = (pw >= 100) ? 32 : 20;

    /* Tab 1 (Active): Intro & Overview */
    ov_buf_bg(240, 175, 20);
    ov_buf_fg(20, 20, 25);
    ov_buf_bold();
    ov_buf_printf("%s", (pw >= 100) ? " [▶ 1: INTRO & OVERVIEW ◀] " : " [▶ 1: INTRO ◀] ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_printf(" ");

    /* Tab 2 (Inactive): Keystrokes & Controls */
    ov_buf_bg(45, 50, 65);
    ov_buf_fg(190, 200, 220);
    ov_buf_bold();
    ov_buf_printf("%s", (pw >= 100) ? " [ 2: KEYSTROKES & CONTROLS ] " : " [2: CONTROLS] ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    /* Close hint on right */
    const char *close_hint = "[ESC: Close Help] [X] ";
    int used_hdr = tab1_w + 1 + tab2_w;
    int rem_hdr  = inner_w - used_hdr - (int) strlen(close_hint);
    if (rem_hdr > 0)
    {
        ov_buf_hline(' ', rem_hdr);
    }
    ov_buf_fg(255, 120, 100);
    ov_buf_bold();
    ov_buf_printf("%s", close_hint);
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    /* Row pr + 2: Subtitle */
    ov_buf_pos(pr + 2, col);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("Unified Real-Time Dashboard for Shared Memory, Telemetry, and AO Pipelines");
    int sub_rem = inner_w - 75;
    if (sub_rem > 0)
    {
        ov_buf_hline(' ', sub_rem);
    }

    /* Row pr + 3: Divider */
    ov_buf_pos(pr + 3, pc);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("├");
    for (int c = pc + 1; c < pc + pw - 1; c++)
    {
        ov_buf_printf("─");
    }
    ov_buf_printf("┤");

    /* Body viewport calculation */
    int body_top    = pr + 4;
    int body_bot    = pr + ph - 2;
    int body_h      = body_bot - body_top + 1;
    int total_lines = (int) (sizeof(INTRO_ITEMS) / sizeof(INTRO_ITEMS[0]));
    int max_scroll  = (total_lines > body_h) ? (total_lines - body_h) : 0;

    int scroll = lay->help_intro_scroll;
    if (scroll < 0)
    {
        scroll = 0;
    }
    if (scroll > max_scroll)
    {
        scroll = max_scroll;
    }

    for (int r = 0; r < body_h; r++)
    {
        int row_idx = r + scroll;
        int cur_row = body_top + r;
        ov_buf_pos(cur_row, col);
        ov_theme_bg(OV_BG_PANEL);

        if (row_idx >= total_lines)
        {
            ov_buf_hline(' ', inner_w);
            continue;
        }

        const intro_item_t *item = &INTRO_ITEMS[row_idx];
        int printed_len = 0;

        switch (item->type)
        {
        case IL_HEADER:
            ov_buf_bold();
            ov_buf_bg(40, 50, 75);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf(" %s ", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            printed_len = (int) strlen(item->prefix) + 2;
            break;

        case IL_SUBHEADER:
            ov_buf_bold();
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("  %s", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            printed_len = (int) strlen(item->prefix) + 2;
            break;

        case IL_BULLET:
            ov_buf_bold();
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf("    %s", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf("%s", item->text ? item->text : "");
            printed_len = (int) strlen(item->prefix) + 4 +
                          (item->text ? (int) strlen(item->text) : 0);
            break;

        case IL_TEXT:
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf("  %s", item->text ? item->text : "");
            printed_len = (item->text ? (int) strlen(item->text) : 0) + 2;
            break;

        case IL_KEY:
            ov_buf_bold();
            ov_buf_fg(130, 205, 255);
            ov_buf_printf("  %-16s", item->prefix ? item->prefix : "");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" %s", item->text ? item->text : "");
            printed_len = 2 + 16 + 1 + (item->text ? (int) strlen(item->text) : 0);
            break;

        case IL_BLANK:
        default:
            printed_len = 0;
            break;
        }

        int pad = inner_w - printed_len;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }
    }

    if (scroll > 0)
    {
        ov_buf_pos(body_top, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▲");
    }
    if (scroll < max_scroll)
    {
        ov_buf_pos(body_bot, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▼");
    }

    /* Bottom divider */
    {
        ov_buf_pos(body_bot + 1, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├─");

        char bstatus[80];
        if (max_scroll > 0)
        {
            snprintf(bstatus, sizeof(bstatus),
                     " [↑↓ / PgUp/PgDn: Scroll (%d/%d) • 2/k: Controls • ESC: Close] ",
                     scroll + 1, total_lines);
        }
        else
        {
            snprintf(bstatus, sizeof(bstatus),
                     " [2 / k: Controls Reference • ESC: Close] ");
        }

        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("%s", bstatus);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);

        int used_b = 2 + (int) strlen(bstatus);
        int rem_b  = (pw - 2) - used_b;
        if (rem_b > 0)
        {
            for (int i = 0; i < rem_b; i++)
            {
                ov_buf_printf("─");
            }
        }
        ov_buf_printf("┤");
    }

    ov_buf_reset_attr();
}

/* ---- Render entry point ---- */

/**
 * ov_render_help - render the help overlay panel.
 * @lay: layout state
 * @m:   system model
 */
void ov_render_help(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    int pr, pc, ph, pw;
    ov_help_get_rect(lay, &pr, &pc, &ph, &pw);

    if (lay->help_mode == 1)
    {
        ov_help_render_intro(lay, pr, pc, ph, pw);
        return;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    /* Draw outer panel border */
    const char *title =
        (pw >= 84)
            ? "HELP & CONTROLS (1/i: Intro • 2/k: Controls • ↑↓ nav • / search • ESC close)"
            : ((pw >= 54) ? "HELP & CONTROLS (1/i: Intro • / search • ESC close)" : "HELP");
    ov_draw_panel_border(pr, pc, ph, pw, title, OV_FG_BRIGHT, 1, 0);

    /* Clear interior background */
    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    int inner_w = pw - 4;

    /* Row 1: Header Mode Selector and Context */
    {
        ov_buf_pos(pr + 1, pc + 2);
        ov_theme_bg(OV_BG_PANEL);

        int tab1_w = (pw >= 100) ? 26 : 14;
        int tab2_w = (pw >= 100) ? 32 : 20;

        /* Tab 1 (Inactive): Intro & Overview */
        ov_buf_bg(45, 50, 65);
        ov_buf_fg(190, 200, 220);
        ov_buf_bold();
        ov_buf_printf("%s", (pw >= 100) ? " [ 1: INTRO & OVERVIEW ] " : " [1: INTRO] ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf(" ");

        /* Tab 2 (Active): Keystrokes & Controls */
        ov_buf_bg(240, 175, 20);
        ov_buf_fg(20, 20, 25);
        ov_buf_bold();
        ov_buf_printf("%s", (pw >= 100) ? " [▶ 2: KEYSTROKES & CONTROLS ◀] "
                                        : " [▶ 2: CONTROLS ◀] ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("  ");

        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Context: ");
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);

        const char *vname = (lay->view == OV_VIEW_DASHBOARD) ? "DASH"
                            : (lay->view == OV_VIEW_STREAMS) ? "STRM"
                            : (lay->view == OV_VIEW_PROCS)   ? "PROC"
                            : (lay->view == OV_VIEW_FPS)     ? "FPS"
                                                             : "CONN";
        const char *pname = (lay->focus == OV_FOCUS_STREAMS) ? "Streams"
                            : (lay->focus == OV_FOCUS_PROCS) ? "Processes"
                            : (lay->focus == OV_FOCUS_FPS)   ? "FPS"
                                                             : "Graph";
        ov_buf_printf("[%s / %s]", vname, pname);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);

        int sel_s = ov_get_selected_stream_idx(lay, m);
        int sel_p = ov_get_selected_proc_idx(lay, m);
        int sel_f = ov_get_selected_fps_idx(lay, m);

        if (lay->focus == OV_FOCUS_STREAMS && m && sel_s >= 0 && sel_s < m->nb_streams)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->streams[sel_s].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
        else if (lay->focus == OV_FOCUS_PROCS && m && sel_p >= 0 && sel_p < m->nb_procs)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->procs[sel_p].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
        else if (lay->focus == OV_FOCUS_FPS && m && sel_f >= 0 && sel_f < m->nb_fps)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->fps[sel_f].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }

        /* Control Mode badge on right side of header row */
        int ctrl_badge_col = pc + pw - 27;
        if (ctrl_badge_col > pc + tab1_w + tab2_w + 35)
        {
            ov_buf_pos(pr + 1, ctrl_badge_col);
            if (lay->ctrl_mode)
            {
                ov_buf_bg(220, 40, 40);
                ov_buf_fg(255, 255, 255);
                ov_buf_bold();
                ov_buf_printf("  CONTROL MODE: ON   ");
            }
            else
            {
                ov_buf_bg(35, 75, 45);
                ov_buf_fg(160, 230, 160);
                ov_buf_bold();
                ov_buf_printf("  CONTROL MODE: OFF  ");
            }
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
    }

    /* Row 2: Search Bar or Search Feature Notification Note */
    {
        ov_buf_pos(pr + 2, pc + 2);
        ov_theme_bg(OV_BG_PANEL);

        if (lay->help_search_active || lay->help_search[0] != '\0')
        {
            ov_buf_bold();
            ov_buf_fg(255, 220, 100);
            ov_buf_printf("Search: ");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            /* Search input box */
            if (lay->help_search_active)
            {
                ov_buf_bg(25, 45, 65);
                ov_buf_fg(255, 255, 255);
            }
            else
            {
                ov_buf_bg(35, 40, 50);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_bold();

            int qbox_w = 26;
            if (qbox_w > pw - 38)
            {
                qbox_w = pw - 38;
            }
            if (qbox_w < 12)
            {
                qbox_w = 12;
            }

            char qdisp[48];
            snprintf(qdisp, sizeof(qdisp), "%s%s", lay->help_search,
                     lay->help_search_active ? "█" : "");
            ov_buf_printf(" %-*.*s ", qbox_w, qbox_w, qdisp);

            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);

            /* Matches count */
            char count_str[32];
            if (lay->help_search[0] == '\0')
            {
                snprintf(count_str, sizeof(count_str), " (type query)");
            }
            else
            {
                snprintf(count_str, sizeof(count_str), " (%d match%s)", nvis,
                         (nvis == 1) ? "" : "es");
            }
            ov_buf_printf("%s", count_str);

            /* Clear / cancel button */
            const char *btn_str = (lay->help_search[0] == '\0')
                                      ? " [ESC: cancel] "
                                      : " [ESC: clear] ";
            int used = 8 + qbox_w + 2 + (int) strlen(count_str);
            int rem  = inner_w - used - (int) strlen(btn_str);
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_buf_fg(255, 120, 100);
            ov_buf_printf("%s", btn_str);
        }
        else
        {
            /* 1-line note notifying users of the search feature */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("Search: ");
            ov_buf_bold();
            ov_buf_fg(255, 220, 100);
            ov_buf_printf("[/]");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(
                " Press '/' to search topics & commands (e.g. \"kill\", \"stream\", \"fps\")");

            int used = 8 + 3 + 68;
            int rem  = inner_w - used;
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
        }
    }

    /* Row 3: Top divider */
    {
        ov_buf_pos(pr + 3, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├");
        for (int c = pc + 1; c < pc + pw - 1; c++)
        {
            ov_buf_printf("─");
        }
        ov_buf_printf("┤");
    }

    /* Detailed Help split calculation */
    int detail_h = (ph >= 36) ? 10 : ((ph >= 28) ? 8 : ((ph >= 22) ? 6 : 5));
    int split_r  = (pr + ph - 1) - detail_h;
    int list_top = pr + 4;
    int list_h   = split_r - list_top;
    if (list_h < 4)
    {
        list_h   = 4;
        split_r  = list_top + list_h;
        detail_h = (pr + ph - 1) - split_r;
    }

    /* Cursor bounds check */
    int sel = lay->help_sel;
    if (sel < 0)
    {
        sel = 0;
    }
    if (sel >= nvis)
    {
        sel = nvis - 1;
    }

    /* Scroll so cursor is within list viewport */
    int scroll = 0;
    if (sel >= list_h)
    {
        scroll = sel - list_h + 1;
    }

    /* Render visible rows in list area */
    if (lay->help_search[0] != '\0' && nvis == 0)
    {
        ov_buf_pos(list_top + 1, pc + 4);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("No matching commands found for \"%s\"", lay->help_search);
        ov_buf_pos(list_top + 2, pc + 4);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Try searching: stream, proc, fps, kill, sort, filter, view");
    }

    for (int vr = 0; vr < list_h && vr + scroll < nvis; vr++)
    {
        int                 idx    = map[vr + scroll];
        const help_entry_t *h      = &HELP[idx];
        int                 row    = list_top + vr;
        int                 is_sel = ((vr + scroll) == sel);

        ov_buf_pos(row, pc + 2);
        if (is_sel)
        {
            ov_buf_bg(45, 55, 85);
        }
        else
        {
            ov_theme_bg(OV_BG_PANEL);
        }

        if (lay->help_search[0] != '\0')
        {
            /* Search match row with section badge */
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf(" ▶ ");
            }
            else
            {
                ov_buf_printf("   ");
            }

            /* Section badge */
            ov_buf_bold();
            ov_theme_fg(ov_help_section_color(h->section));
            ov_buf_printf("[%-4s] ", ov_help_section_tag(h->section));
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_buf_bg(45, 55, 85);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
            }

            /* Keystroke or category indicator */
            ov_buf_bold();
            if (h->flags & HF_SECTION)
            {
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_printf("%-13s", "Topic");
                ov_buf_printf("   ");
            }
            else if (h->flags & HF_CTRL_MODE)
            {
                if (lay->ctrl_mode)
                {
                    ov_buf_fg(255, 95, 75);
                    ov_buf_printf("%-13s", h->key ? h->key : "");
                    ov_buf_fg(255, 80, 80);
                    ov_buf_printf(" ⚡ ");
                }
                else
                {
                    ov_buf_fg(200, 140, 50);
                    ov_buf_printf("%-13s", h->key ? h->key : "");
                    ov_buf_fg(160, 115, 45);
                    ov_buf_printf(" 🔒 ");
                }
            }
            else
            {
                ov_buf_fg(130, 205, 255);
                ov_buf_printf("%-13s", h->key ? h->key : "");
                ov_buf_printf("   ");
            }

            /* Summary label */
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_buf_bg(45, 55, 85);
                ov_buf_fg(255, 255, 255);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_printf("%s", h->label);

            int used = 3 + 7 + 13 + 3 + (int) strlen(h->label);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else if (h->flags & HF_SECTION)
        {
            int         expanded = help_is_expanded(lay, h->section);
            const char *chev     = expanded ? "▾" : "▸";

            if (is_sel)
            {
                ov_buf_bg(55, 65, 95);
                ov_buf_bold();
                ov_buf_fg(255, 220, 100);
                ov_buf_printf("▶ %s %s", chev, h->label);
            }
            else
            {
                ov_buf_bg(36, 40, 52);
                ov_buf_bold();
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_printf("  %s %s", chev, h->label);
            }

            /* Count child entries in section */
            int nchildren = 0;
            for (int k = 0; k < HELP_TOTAL; k++)
            {
                if (HELP[k].section == h->section && !(HELP[k].flags & HF_SECTION))
                {
                    nchildren++;
                }
            }

            char tag[32];
            snprintf(tag, sizeof(tag), "(%d keys)", nchildren);
            int used = 4 + (int) strlen(h->label) + 1 + (int) strlen(tag);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" %s", tag);
        }
        else if (h->flags & HF_COLORS)
        {
            ov_theme_fg(OV_FG_DIM);
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf("    ▶ ");
            }
            else
            {
                ov_buf_printf("      ");
            }

            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("● stream ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_printf("● proc ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_printf("● fps ");
            ov_theme_fg(OV_FG_CONN);
            ov_buf_printf("● conn ");
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("● active ");
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("● warn ");
            ov_theme_fg(OV_FG_ERROR);
            ov_buf_printf("● error");

            int used = 6 + 53;
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else
        {
            /* Standard keystroke entry with tab offset */
            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
                ov_buf_printf("    ▶ ");
            }
            else
            {
                ov_buf_printf("      ");
            }

            /* Keystroke column in standard bold font */
            ov_buf_bold();
            if (h->flags & HF_CTRL_MODE)
            {
                if (lay->ctrl_mode)
                {
                    ov_buf_fg(255, 95, 75);
                    ov_buf_printf("%-13s", h->key);
                    ov_buf_fg(255, 80, 80);
                    ov_buf_printf(" ⚡ ");
                }
                else
                {
                    ov_buf_fg(200, 140, 50);
                    ov_buf_printf("%-13s", h->key);
                    ov_buf_fg(160, 115, 45);
                    ov_buf_printf(" 🔒 ");
                }
            }
            else
            {
                ov_buf_fg(130, 205, 255);
                ov_buf_printf("%-13s", h->key);
                ov_buf_printf("    ");
            }

            /* Summary label */
            ov_buf_reset_attr();
            if (is_sel)
            {
                ov_buf_bg(45, 55, 85);
                ov_buf_fg(255, 255, 255);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_printf("%s", h->label);

            int used = 6 + 13 + 4 + (int) strlen(h->label);
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }

        ov_buf_reset_attr();
    }

    /* Scroll indicators */
    if (scroll > 0)
    {
        ov_buf_pos(list_top, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▲");
    }
    if (scroll + list_h < nvis)
    {
        ov_buf_pos(split_r - 1, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▼");
    }

    /* Divider above Detailed Help */
    {
        ov_buf_pos(split_r, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├─");
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf(" DETAILED HELP ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);

        int         used_div = 17;
        const char *hint     = (lay->help_search[0] != '\0')
                                   ? "[↑↓ nav • ESC clear search]"
                                   : "[↑↓ nav • →/← expand • [/] search • ESC close]";
        int         hint_len = (int) strlen(hint);
        int         div_pad  = (pw - 2) - used_div - hint_len - 1;
        if (div_pad > 0)
        {
            for (int i = 0; i < div_pad; i++)
            {
                ov_buf_printf("─");
            }
            ov_buf_printf(" %s─┤", hint);
        }
        else
        {
            for (int i = 0; i < (pw - 2) - used_div; i++)
            {
                ov_buf_printf("─");
            }
            ov_buf_printf("┤");
        }
    }

    /* Render detailed help for currently selected item */
    if (sel >= 0 && sel < nvis)
    {
        ov_help_render_detail(lay, m, &HELP[map[sel]], split_r, pc, pw, detail_h);
    }

    ov_buf_reset_attr();
}
