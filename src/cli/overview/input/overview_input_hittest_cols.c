// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_hittest_cols.c
 * @brief Header click detection, column sort hit-tests, and global hover resolution
 */

#include "overview_input_internal.h"

/**
 * ov_input_hit_panel_tab - detect which tab label was clicked.
 * @mc:        Mouse column (1-based).
 * @panel_col: Panel left column.
 * @tabs:      Array of tab label strings.
 * @num_tabs:  Number of tabs in array.
 *
 * Return: Tab index [0..num_tabs-1], or -1 if no tab hit.
 */
int ov_input_hit_panel_tab(int mc, int panel_col, const char **tabs, int num_tabs)
{
    int cur = panel_col + 2;
    for (int ii = 0; ii < num_tabs; ii++)
    {
        /* Tab text is " LABEL ", so width = strlen + 2 */
        int tw = (int) strlen(tabs[ii]) + 2;
        if (mc >= cur && mc < cur + tw)
        {
            return ii;
        }
        cur += tw + 1; /* +1 for gap between tabs */
    }
    return -1;
}

static const char *stream_col_names[] = { "NAME", "TYP",   "SIZE",  "Hz",
                                          "MB/s", "INODE", "COUNT", "ANCESTRY" };

/**
 * ov_input__streams_header_click - handle mouse clicks on streams table column headers.
 * @lay: Pointer to layout structure.
 * @mc:  Mouse column coordinate.
 */
void ov_input__streams_header_click(OV_LAYOUT *lay, int mc)
{
    int           table_x = mc - lay->r_streams.col - 2 + lay->panel[OV_FOCUS_STREAMS].hscroll;
    OV_COL_LAYOUT cols[12];
    int           num_cols = ov_get_stream_col_layout(lay->compact_mode, cols);
    int           col_idx =
        ov_header_hittest_sort_key(cols, num_cols, lay->panel[OV_FOCUS_STREAMS].collapsed, table_x);

    if (col_idx >= 0)
    {
        if (lay->panel[OV_FOCUS_STREAMS].sort_key == col_idx)
        {
            lay->panel[OV_FOCUS_STREAMS].sort_dir = !lay->panel[OV_FOCUS_STREAMS].sort_dir;
        }
        else
        {
            lay->panel[OV_FOCUS_STREAMS].sort_key = col_idx;
            lay->panel[OV_FOCUS_STREAMS].sort_dir = 0;
        }
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort STREAMS by %s %s",
                       stream_col_names[col_idx],
                       lay->panel[OV_FOCUS_STREAMS].sort_dir ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

static const char *proc_col_names[] = { "NAME", "PID",    "STAT", "Hz",      "MEM", "ANCESTRY",
                                        "PRIO", "UPTIME", "CPU%", "LOOPCNT", "DUTY" };

/**
 * ov_input__procs_header_click - handle mouse clicks on process table column headers.
 * @lay: Pointer to layout structure.
 * @mc:  Mouse column coordinate.
 */
void ov_input__procs_header_click(OV_LAYOUT *lay, int mc)
{
    int           table_x = mc - lay->r_procs.col - 2 + lay->panel[OV_FOCUS_PROCS].hscroll;
    OV_COL_LAYOUT cols[16];
    int           num_cols = ov_get_proc_col_layout(lay->compact_mode, cols);
    int           col_idx =
        ov_header_hittest_sort_key(cols, num_cols, lay->panel[OV_FOCUS_PROCS].collapsed, table_x);

    if (col_idx >= 0)
    {
        if (lay->panel[OV_FOCUS_PROCS].sort_key == col_idx)
        {
            lay->panel[OV_FOCUS_PROCS].sort_dir = !lay->panel[OV_FOCUS_PROCS].sort_dir;
        }
        else
        {
            lay->panel[OV_FOCUS_PROCS].sort_key = col_idx;
            lay->panel[OV_FOCUS_PROCS].sort_dir = 0;
        }
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort PROCS by %s %s", proc_col_names[col_idx],
                       lay->panel[OV_FOCUS_PROCS].sort_dir ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

static const char *fps_col_names[] = { "NAME", "CPID", "MEM", "ANCESTRY", "RPID", "TMX", "STR" };

/**
 * ov_input__fps_header_click - handle mouse clicks on FPS table column headers.
 * @lay: Pointer to layout structure.
 * @mc:  Mouse column coordinate.
 */
void ov_input__fps_header_click(OV_LAYOUT *lay, int mc)
{
    int           table_x = mc - lay->r_fps.col - 2 + lay->panel[OV_FOCUS_FPS].hscroll;
    OV_COL_LAYOUT cols[8];
    int           num_cols = ov_get_fps_col_layout(lay->compact_mode, lay->view, cols);
    int           col_idx =
        ov_header_hittest_sort_key(cols, num_cols, lay->panel[OV_FOCUS_FPS].collapsed, table_x);

    if (col_idx >= 0)
    {
        if (lay->panel[OV_FOCUS_FPS].sort_key == col_idx)
        {
            lay->panel[OV_FOCUS_FPS].sort_dir = !lay->panel[OV_FOCUS_FPS].sort_dir;
        }
        else
        {
            lay->panel[OV_FOCUS_FPS].sort_key = col_idx;
            lay->panel[OV_FOCUS_FPS].sort_dir = 0;
        }
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort FPS by %s %s", fps_col_names[col_idx],
                       lay->panel[OV_FOCUS_FPS].sort_dir ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

/**
 * ov_hittest_resolve_globals - Resolve hover index into global indices for cross-panel lineage
 * @lay: Pointer to layout structure
 * @m:   Pointer to data model snapshot
 */
void ov_hittest_resolve_globals(OV_LAYOUT *lay, const OV_MODEL *m)
{
    lay->panel[OV_FOCUS_STREAMS].hover_idx = -1;
    lay->panel[OV_FOCUS_PROCS].hover_idx   = -1;
    lay->panel[OV_FOCUS_FPS].hover_idx     = -1;

    if (!lay->mouse_hover || lay->hover_idx < 0)
    {
        return;
    }

    /* hover_idx is a drawn row: resolve it through the same rows the panel draws */
    if (lay->hover_view == OV_FOCUS_STREAMS)
    {
        lay->panel[OV_FOCUS_STREAMS].hover_idx =
            ov_row_to_idx(lay, m, OV_FOCUS_STREAMS, lay->hover_idx);
    }
    else if (lay->hover_view == OV_FOCUS_PROCS)
    {
        lay->panel[OV_FOCUS_PROCS].hover_idx =
            ov_row_to_idx(lay, m, OV_FOCUS_PROCS, lay->hover_idx);
    }
    else if (lay->hover_view == OV_FOCUS_FPS)
    {
        lay->panel[OV_FOCUS_FPS].hover_idx = ov_row_to_idx(lay, m, OV_FOCUS_FPS, lay->hover_idx);
    }
    else if (lay->hover_view == OV_FOCUS_GRAPH)
    {
        int                 n;
        const SG_TREE_NODE *rows = ov_graph_rows(lay, m, &n);
        if (lay->hover_idx >= 0 && lay->hover_idx < n)
        {
            lay->panel[OV_FOCUS_STREAMS].hover_idx = rows[lay->hover_idx].stream_idx;
        }
    }
}
