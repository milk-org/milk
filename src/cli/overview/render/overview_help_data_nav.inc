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
