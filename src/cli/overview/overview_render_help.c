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
    HS_NAV = 0, /* Global & Navigation       */
    HS_STREAMS, /* Streams Panel (STRM)      */
    HS_PROCS,   /* Processes Panel (PROC)    */
    HS_FPS,     /* FPS Panel (FPS)           */
    HS_GRAPH,   /* Graph & Lineage (CONN)    */
    HS_CMDLOG,  /* Command Log & Display     */
    HS_MOUSE,   /* Mouse Interactions        */
    HS_COLORS,  /* Theme & Colors            */
    HS_COUNT
};

/* clang-format off */
static const help_entry_t HELP[] =
{
    /* =========================================================
     * 1. Global & Navigation
     * ========================================================= */
    {
        NULL,
        "Global & Navigation",
        "Global controls available from any view or panel. Includes view switching "
        "(F2-F6), panel focus cycling (TAB), real-time scan rate tuning, display pause, "
        "regex filtering, snapshot export, and application exit.",
        HF_SECTION,
        HS_NAV,
    },
    {
        "F2 - F6",
        "Switch views (DASH, STRM, PROC, FPS, CONN)",
        "Switches full-screen or grid dashboard view: F2=Dashboard (all panels), "
        "F3=Streams (SHM), F4=Processes (procinfo), F5=FPS (module list & param tree), "
        "F6=Node Graph (dataflow connections).",
        HF_ENTRY,
        HS_NAV,
    },
    {
        "^Left/^Right",
        "Cycle views sequentially",
        "Cycles forward or backward through the 5 dashboard view modes "
        "(DASH -> STRM -> PROC -> FPS -> CONN). Equivalent to pressing F2 through F6.",
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
        "Opens interactive regex filter. Displays only matching items with a blinking "
        "FILTER ON indicator in header and status bar. Enter applies; Esc clears.",
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
        "Clicking dashboard header tabs (DASH, STRM, etc.) switches views.",
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
        "Color coding conventions used throughout milk-CTRL. Distinguishes data stream "
        "types, process states, and system health status.",
        HF_SECTION,
        HS_COLORS,
    },
    {
        "Legend",
        "System color semantics",
        "Cyan = Stream (SHM)  |  Purple = Process (procinfo)  |  Blue = FPS module\n"
        "Green = Active / Running  |  Gray = Idle / Paused\n"
        "Amber = Stale / Warning  |  Red = Error / Crashed / Signal Kill",
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
 * help_visible_rows - count visible rows and populate mapping array.
 * @lay: layout state (for expand bitmask)
 * @map: output array mapping visible row index to HELP[] index
 *
 * Return: number of visible rows.
 */
static int help_visible_rows(
    const OV_LAYOUT *lay,
    int             *map)
{
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
    lay->show_help      = 1;
    lay->help_expand    = 0;
    lay->filter_editing = 0;
    int sec             = ov_help_focus_section(lay->focus);
    lay->help_sel       = ov_help_section_first_vis_row(lay, sec);
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
    int W = lay->term_cols;
    int H = lay->term_rows;

    int pw = (W >= 94) ? 88 : (W - 4);
    if (pw < 48)
    {
        pw = (W > 4) ? (W - 2) : 44;
    }

    int ph = (H >= 34) ? 30 : ((H >= 24) ? (H - 4) : (H - 2));
    if (ph < 16)
    {
        ph = (H > 2) ? (H - 2) : 14;
    }

    int pr = (H - ph) / 2;
    int pc = (W - pw) / 2;
    if (pr < 1)
    {
        pr = 1;
    }
    if (pc < 1)
    {
        pc = 1;
    }

    /* Click outside popup -> close help overlay */
    if (mr < pr || mr >= pr + ph || mc < pc || mc >= pc + pw)
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        return 1;
    }

    /* Click close button area on top border */
    if (mr == pr && mc >= pc + pw - 6)
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        return 1;
    }

    /* Detail pane height and split line */
    int detail_h = (ph >= 26) ? 7 : ((ph >= 20) ? 5 : 4);
    int split_r  = (pr + ph - 1) - detail_h;

    /* Check if click is inside the list area */
    if (mr >= pr + 3 && mr < split_r)
    {
        int map[128];
        int nvis = help_visible_rows(lay, map);

        int list_h = split_r - (pr + 3);
        int sel    = lay->help_sel;
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

        int vis_row = (mr - (pr + 3)) + scroll;
        if (vis_row >= 0 && vis_row < nvis)
        {
            if (lay->help_sel == vis_row)
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

    if (entry->section == HS_STREAMS)
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
    int map[128];
    int nvis = help_visible_rows(lay, map);

    int W = lay->term_cols;
    int H = lay->term_rows;

    int pw = (W >= 94) ? 88 : (W - 4);
    if (pw < 48)
    {
        pw = (W > 4) ? (W - 2) : 44;
    }

    int ph = (H >= 34) ? 30 : ((H >= 24) ? (H - 4) : (H - 2));
    if (ph < 16)
    {
        ph = (H > 2) ? (H - 2) : 14;
    }

    int pr = (H - ph) / 2;
    int pc = (W - pw) / 2;
    if (pr < 1)
    {
        pr = 1;
    }
    if (pc < 1)
    {
        pc = 1;
    }

    /* Draw outer panel border */
    const char *title =
        (pw >= 76) ? "HELP & CONTROLS  (↑↓ nav • →/← expand/collapse • ESC close)"
                   : ((pw >= 54) ? "HELP (↑↓ nav • →/← expand • ESC close)" : "HELP");
    ov_draw_panel_border(pr, pc, ph, pw, title, OV_FG_BRIGHT, 1, 0);

    /* Clear interior background */
    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    /* Row 1: Header Context and Control Mode Status */
    {
        ov_buf_pos(pr + 1, pc + 2);
        ov_theme_bg(OV_BG_PANEL);
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
        if (ctrl_badge_col > pc + 36)
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

    /* Row 2: Top divider */
    {
        ov_buf_pos(pr + 2, pc);
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
    int detail_h = (ph >= 26) ? 7 : ((ph >= 20) ? 5 : 4);
    int split_r  = (pr + ph - 1) - detail_h;
    int list_top = pr + 3;
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

    int inner_w = pw - 4;

    /* Render visible rows in list area */
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

        if (h->flags & HF_SECTION)
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
        const char *hint     = "[↑↓ nav • →/← expand • ESC close]";
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
