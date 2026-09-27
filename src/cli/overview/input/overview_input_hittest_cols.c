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
int ov_input_hit_panel_tab(
    int          mc,
    int          panel_col,
    const char **tabs,
    int          num_tabs)
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
void ov_input__streams_header_click(
    OV_LAYOUT *lay,
    int        mc)
{
    int           table_x = mc - lay->r_streams.col - 2 + lay->hscroll_stream;
    OV_COL_LAYOUT cols[12];
    int           num_cols = ov_get_stream_col_layout(lay->compact_mode, cols);
    int col_idx = ov_header_hittest_sort_key(cols, num_cols,
                                             lay->col_collapsed_stream, table_x);

    if (col_idx >= 0)
    {
        if (lay->sort_key_stream == col_idx)
        {
            lay->sort_dir_stream = !lay->sort_dir_stream;
        }
        else
        {
            lay->sort_key_stream = col_idx;
            lay->sort_dir_stream = 0;
        }
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort STREAMS by %s %s",
                       stream_col_names[col_idx], lay->sort_dir_stream ? "▼" : "▲");
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
void ov_input__procs_header_click(
    OV_LAYOUT *lay,
    int        mc)
{
    int           table_x = mc - lay->r_procs.col - 2 + lay->hscroll_proc;
    OV_COL_LAYOUT cols[16];
    int           num_cols = ov_get_proc_col_layout(lay->compact_mode, cols);
    int col_idx = ov_header_hittest_sort_key(cols, num_cols,
                                             lay->col_collapsed_proc, table_x);

    if (col_idx >= 0)
    {
        if (lay->sort_key_proc == col_idx)
        {
            lay->sort_dir_proc = !lay->sort_dir_proc;
        }
        else
        {
            lay->sort_key_proc = col_idx;
            lay->sort_dir_proc = 0;
        }
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort PROCS by %s %s",
                       proc_col_names[col_idx], lay->sort_dir_proc ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

static const char *fps_col_names[] = { "NAME", "CPID", "MEM", "ANCESTRY",
                                       "RPID", "TMX", "STR" };

/**
 * ov_input__fps_header_click - handle mouse clicks on FPS table column headers.
 * @lay: Pointer to layout structure.
 * @mc:  Mouse column coordinate.
 */
void ov_input__fps_header_click(
    OV_LAYOUT *lay,
    int        mc)
{
    int           table_x = mc - lay->r_fps.col - 2 + lay->hscroll_fps;
    OV_COL_LAYOUT cols[8];
    int           num_cols = ov_get_fps_col_layout(lay->compact_mode, lay->view, cols);
    int col_idx = ov_header_hittest_sort_key(cols, num_cols,
                                             lay->col_collapsed_fps, table_x);

    if (col_idx >= 0)
    {
        if (lay->sort_key_fps == col_idx)
        {
            lay->sort_dir_fps = !lay->sort_dir_fps;
        }
        else
        {
            lay->sort_key_fps = col_idx;
            lay->sort_dir_fps = 0;
        }
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort FPS by %s %s",
                       fps_col_names[col_idx], lay->sort_dir_fps ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

/**
 * ov_hittest_resolve_globals - Resolve hover index into global indices for cross-panel lineage
 * @lay: Pointer to layout structure
 * @m:   Pointer to data model snapshot
 */
void ov_hittest_resolve_globals(
    OV_LAYOUT      *lay,
    const OV_MODEL *m)
{
    lay->hover_global_stream = -1;
    lay->hover_global_proc   = -1;
    lay->hover_global_fps    = -1;

    if (!lay->mouse_hover || lay->hover_idx < 0)
    {
        return;
    }

    if (lay->hover_view == OV_FOCUS_STREAMS)
    {
        int count = m->nb_streams;
        int fidx[OV_MAX_STREAMS];
        for (int i = 0; i < count; i++)
        {
            fidx[i] = i;
        }
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_STREAMS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->streams[i].name;
            }
            count = ov_filter_build(filt, names, m->nb_streams, fidx, OV_MAX_STREAMS);
        }
        if (lay->hover_idx < count)
        {
            lay->hover_global_stream = fidx[lay->hover_idx];
        }
    }
    else if (lay->hover_view == OV_FOCUS_PROCS)
    {
        int count = m->nb_procs;
        int fidx[OV_MAX_PROCS];
        for (int i = 0; i < count; i++)
        {
            fidx[i] = i;
        }
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_PROCS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->procs[i].name;
            }
            count = ov_filter_build(filt, names, m->nb_procs, fidx, OV_MAX_PROCS);
        }
        if (lay->hover_idx < count)
        {
            lay->hover_global_proc = fidx[lay->hover_idx];
        }
    }
    else if (lay->hover_view == OV_FOCUS_FPS)
    {
        int count = m->nb_fps;
        int fidx[OV_MAX_FPS];
        for (int i = 0; i < count; i++)
        {
            fidx[i] = i;
        }
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_FPS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_FPS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->fps[i].name;
            }
            count = ov_filter_build(filt, names, m->nb_fps, fidx, OV_MAX_FPS);
        }
        if (lay->hover_idx < count)
        {
            lay->hover_global_fps = fidx[lay->hover_idx];
        }
    }
    else if (lay->hover_view == OV_FOCUS_GRAPH)
    {
        int start_node = ov_input_get_graph_start_node(lay, m);
        if (start_node >= 0)
        {
            SG_RENDER_NODE rnodes[OV_MAX_NODES];
            int n_rnodes = sg_compute_render_nodes(m, start_node,
                                                   lay->lineage_mode, rnodes);
            if (lay->hover_idx < n_rnodes)
            {
                int node_idx = rnodes[lay->hover_idx].node_idx;
                if (node_idx >= 0 && node_idx < m->nb_nodes)
                {
                    const OV_NODE *node = &m->nodes[node_idx];
                    if (node->type == OV_NODE_STREAM)
                    {
                        lay->hover_global_stream = node->index;
                    }
                    else if (node->type == OV_NODE_PROC)
                    {
                        lay->hover_global_proc = node->index;
                    }
                    else if (node->type == OV_NODE_FPS)
                    {
                        lay->hover_global_fps = node->index;
                    }
                }
            }
        }
    }
}
