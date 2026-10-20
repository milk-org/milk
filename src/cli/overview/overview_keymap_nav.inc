// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* Key table rows: introduction, global and navigation keys, streams panel.
 * Fields: display, label, help detail (NULL = label), help section (-1 = -h only),
 * -h group (-1 = help only), OV_KF_* flags, status-line hint (OV_NOHINT = none). */

    {
        "1 / i",
        "Display full interactive introduction guide",
        "Toggles the full-screen interactive Introduction guide explaining what milk-CTRL is, its three core "
        "architectural pillars, primary capabilities, and operational workflows.",
        HS_INTRO,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "F2 - F7",
        "Switch views (DASH, STRM, PROC, FPS, CONN, LOOPS)",
        "Switches full-screen or grid dashboard view: F2=Dashboard (all panels), F3=Streams (SHM), F4=Processes "
        "(procinfo), F5=FPS (module list & param tree), F6=Node Graph (dataflow connections), F7=Feedback Loops "
        "(circuit & overlap).",
        HS_NAV,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "^Left/^Right",
        "Cycle views sequentially",
        "Cycles forward or backward through the 6 dashboard view modes (DASH -> STRM -> PROC -> FPS -> CONN -> "
        "LOOPS). Equivalent to pressing F2 through F7.",
        HS_NAV,
        OV_KG_VIEWS,
        0,
        OV_NOHINT,
    },

    {
        "TAB",
        "Cycle active panel focus",
        "Moves active focus between visible panels: Streams -> Processes -> FPS -> Graph in Dashboard view; or "
        "between FPS list and Parameter tree in F5 view.",
        HS_NAV,
        OV_KG_NAV,
        0,
        { "TAB", "focus", OV_CTX_DASH, 0 },
    },

    {
        "Left / Right",
        "Move focus to the adjacent panel",
        "In DASH (F2), moves focus to the neighbouring panel. In F5, Right/ENTER enter the parameter tree and "
        "Left goes back up. No effect in other views.",
        HS_NAV,
        OV_KG_NAV,
        0,
        { "\xe2\x86\x90", "up", OV_CTX_PARAM_TREE, 0 },
    },

    {
        "UP / DOWN",
        "Navigate rows in focused panel",
        "Moves the row cursor up or down in the focused list. The cursor follows its entry, not the row number, "
        "when the list is rescanned or re-sorted.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "PgUp / PgDn",
        "Scroll page up / down",
        "Scrolls the focused list up or down by one full page of rows for rapid navigation in large stream or "
        "process tables.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "Home / End",
        "Jump to first / last item",
        "Instantly jumps the selection cursor to the very top (first item) or bottom (last item) of the "
        "currently focused list.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "?",
        "Jump to the first match",
        "Prompts for a regex and moves the cursor of the focused panel to the first matching entry; the cursor "
        "stays there across rescans. Says so when a panel filter hides it.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "i",
        "Inspect selection",
        "Suspends the dashboard and runs milk-stream-info, milk-procinfo-info or milk-fps-info on the selection "
        "(names passed as arguments, no shell); ENTER returns.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "/",
        "Filter panel items (regex search)",
        "Opens interactive regex filter for focused panel (STREAMS, PROCESSINFO, or FPS). Displays matching "
        "items with FILTER ON badge in panel border & header. Enter applies; Esc clears.",
        HS_NAV,
        OV_KG_NAV,
        0,
        { "/", "filter", OV_CTX_FOCUS_LIST | OV_CTX_FOCUS_SHOWN, 0 },
    },

    {
        "f",
        "Toggle regex filter ON / OFF",
        "Toggles regex filtering on or off for focused panel without losing the filter query. When OFF, all "
        "panel items are displayed while retaining query for quick resumption.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "SPACE",
        "Freeze selection highlight",
        "Freezes the selection cursor on the current item name, preventing it from jumping when tables re-sort "
        "or items appear/disappear during live scans.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "+ / -",
        "Adjust scan rate (faster / slower)",
        "Increases or decreases the background shared-memory scan interval. Faster rates give lower latency; "
        "slower rates reduce CPU and terminal bandwidth.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "F",
        "Pause / resume display updates",
        "Freezes all UI rendering at the current frame without stopping background processes. Press F again to "
        "resume live real-time rendering.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "W",
        "Export snapshot to file ($TMPDIR, default /tmp)",
        "Writes a diagnostic report of all streams, processes and FPS modules to a new private file "
        "$TMPDIR/milk-CTRL_snapshot_<timestamp>_<random>.txt and shows its path, or the error, in the command "
        "log.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "F8 / ^T",
        "Next theme + selector",
        "Switches to the next theme and opens the selector (see Theme & Colors). Clicking the theme badge in the "
        "status bar does the same.",
        HS_NAV,
        OV_KG_NAV,
        0,
        OV_NOHINT,
    },

    {
        "c",
        "Toggle Control Mode ON / OFF",
        "Toggles safety interlock for administrative commands. When ON, enables process signaling (kill, pause, "
        "step, exit) and stream deletion. Press 'c' to toggle.",
        HS_NAV,
        OV_KG_NAV,
        0,
        { "c", "control", 0, OV_CTX_CTRL },
    },

    {
        "h",
        "Toggle this help overlay",
        "Opens or closes this interactive help overlay. While open, use UP/DOWN to browse commands and ENTER to "
        "expand or collapse panel sections.",
        HS_NAV,
        OV_KG_NAV,
        0,
        { "h", "help", 0, 0 },
    },

    {
        "q / x",
        "Quit milk-CTRL",
        "Exits milk-CTRL cleanly and restores terminal settings. Background streams and shared-memory processes "
        "remain running untouched.",
        HS_NAV,
        OV_KG_NAV,
        0,
        { "q", "quit", 0, 0 },
    },

    {
        "ENTER / D",
        "DETAILS sub-tab (DASH, F6); select the CONNECTIONS row's stream; F5: tree / edit",
        NULL,
        -1,
        OV_KG_NAV,
        0,
        { "D", "details", OV_CTX_FOCUS_LIST | OV_CTX_FOCUS_SHOWN, OV_CTX_CTRL | OV_CTX_F5 },
    },

    {
        "ESC",
        "Back out: parameter tree, DETAILS cursor, filters, FPS marks",
        NULL,
        -1,
        OV_KG_NAV,
        0,
        { "ESC", "back", OV_CTX_PARAM_TREE, 0 },
    },

    {
        "D / ENTER",
        "Toggle detailed inspection pane",
        "Opens the stream inspector showing full metadata, dimensions, datatype, circular buffer depth, memory "
        "address, shared semaphore count, and active writer PID. In F3/F4 the key only says DETAILS is in F2/F6.",
        HS_STREAMS,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "s",
        "Sort streams by Name (A-Z)",
        "Sorts the stream table alphabetically by stream name. Press '[' to toggle between ascending (A-Z) and "
        "descending (Z-A) order.",
        HS_STREAMS,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "S",
        "Sort streams by Activity (Hz)",
        "Sorts the stream table by update frequency (Hz), placing the most actively written streams at the top "
        "of the list.",
        HS_STREAMS,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "A",
        "Sort streams by Ancestry / topology",
        "Sorts streams in topological dataflow order from upstream source inputs down to downstream consumer "
        "outputs based on dependency graph analysis.",
        HS_STREAMS,
        -1,
        0,
        OV_NOHINT,
    },

    {
        "< / > / ]",
        "Cycle active sort column",
        "Cycles the active sort column through Name, Type, Size, Update Rate (Hz), Bandwidth (MB/s), and Writer "
        "PID.",
        HS_STREAMS,
        OV_KG_COLUMNS,
        0,
        OV_NOHINT,
    },

    {
        "[",
        "Toggle sort direction (Asc / Desc)",
        "Reverses the sort order between ascending and descending for whichever column is currently active.",
        HS_STREAMS,
        OV_KG_COLUMNS,
        0,
        OV_NOHINT,
    },

    {
        "Shift+Left/Rt",
        "Select column for layout hiding",
        "Moves the highlighted column selection cursor left or right across table headers so you can inspect "
        "column properties or toggle its visibility.",
        HS_STREAMS,
        OV_KG_COLUMNS,
        0,
        { "\xe2\x87\xa7\xe2\x86\x90\xe2\x86\x92", "columns", OV_CTX_FOCUS_LIST | OV_CTX_FOCUS_SHOWN,
          OV_CTX_CTRL | OV_CTX_F5 },
    },

    {
        "t / T",
        "Toggle column visibility (hide/show)",
        "Hides or reveals the currently selected table column. Useful for customizing table density on compact "
        "or high-resolution displays.",
        HS_STREAMS,
        OV_KG_COLUMNS,
        0,
        OV_NOHINT,
    },

    {
        "d",
        "Toggle compact layout mode",
        "Automatically hides secondary metadata columns (semaphores, memory size) to fit the table cleanly into "
        "narrow terminal windows.",
        HS_STREAMS,
        OV_KG_COLUMNS,
        0,
        OV_NOHINT,
    },

    {
        "s / S / A",
        "Sort by name / rate / ancestry (PROC, FPS: 's' outside CONTROL)",
        NULL,
        -1,
        OV_KG_COLUMNS,
        0,
        { "s/S", "sort", OV_CTX_FOCUS_LIST | OV_CTX_FOCUS_SHOWN, OV_CTX_CTRL | OV_CTX_F5 },
    },

    {
        "F3: Right / ENTER",
        "Open the stream inspector; Right: next page",
        "F3 shows the stream list and an inspector for the stream under the cursor (beside the list, or under it "
        "on narrow terminals). Right or ENTER enters it; there the arrows and PgUp/PgDn scroll, Right shows the next "
        "page, Left or ESC goes back to the list. Shift+TAB or a click on a page label also changes the page. Pages: "
        "INFO (metadata, times, creator and owner), SEMS (every semaphore's value, reader and writer).",
        HS_STREAMS,
        OV_KG_NAV,
        0,
        { "\xe2\x86\x92", "inspect", OV_CTX_F3 | OV_CTX_FOCUS_SHOWN, OV_CTX_INSP },
    },

    {
        "DEL",
        "Delete stream shared memory file",
        "After a y/N confirmation, unlinks the selected stream from the shm directory. Processes holding it stay "
        "mapped until they close it. Requires Control Mode ON.",
        HS_STREAMS,
        OV_KG_CTRL,
        OV_KF_CTRL,
        { "DEL", "delete", OV_CTX_CTRL | OV_CTX_FOCUS_STREAMS | OV_CTX_FOCUS_SHOWN, 0 },
    },
