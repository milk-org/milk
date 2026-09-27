// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_hittest.c
 * @brief Mouse hit-testing, region boundaries, tooltips, and header click resolution.
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

void ov_input__streams_header_click(OV_LAYOUT *lay, int mc)
{
    int           table_x = mc - lay->r_streams.col - 2 + lay->hscroll_stream;
    OV_COL_LAYOUT cols[12];
    int           num_cols = ov_get_stream_col_layout(lay->compact_mode, cols);
    int col_idx = ov_header_hittest_sort_key(cols, num_cols, lay->col_collapsed_stream, table_x);

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

void ov_input__procs_header_click(OV_LAYOUT *lay, int mc)
{
    int           table_x = mc - lay->r_procs.col - 2 + lay->hscroll_proc;
    OV_COL_LAYOUT cols[16];
    int           num_cols = ov_get_proc_col_layout(lay->compact_mode, cols);
    int col_idx = ov_header_hittest_sort_key(cols, num_cols, lay->col_collapsed_proc, table_x);

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
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort PROCS by %s %s", proc_col_names[col_idx],
                       lay->sort_dir_proc ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

static const char *fps_col_names[] = { "NAME", "CPID", "MEM", "ANCESTRY", "RPID", "TMX", "STR" };

void ov_input__fps_header_click(OV_LAYOUT *lay, int mc)
{
    int           table_x = mc - lay->r_fps.col - 2 + lay->hscroll_fps;
    OV_COL_LAYOUT cols[8];
    int           num_cols = ov_get_fps_col_layout(lay->compact_mode, lay->view, cols);
    int col_idx = ov_header_hittest_sort_key(cols, num_cols, lay->col_collapsed_fps, table_x);

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
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Sort FPS by %s %s", fps_col_names[col_idx],
                       lay->sort_dir_fps ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}


void ov_hittest(OV_LAYOUT *lay, const OV_MODEL *m, int mr, int mc)
{
    lay->hover_view         = -1;
    lay->hover_idx          = -1;
    lay->hover_is_header    = 0;
    lay->hover_col_logical  = mc;
    lay->hover_tooltip[0]   = '\0';
    lay->fps_split_hover    = 0;
    lay->dash_split_v_hover = 0;
    lay->dash_split_h_hover = 0;
    lay->cmdlog_split_hover = 0;

    if (!lay->mouse_hover)
    {
        return;
    }

    if (mr == lay->r_header.row)
    {
        int commit_w    = (int) strlen(MILK_GIT_COMMIT) + 3;
        int shmdir_w    = (int) strlen(ov_get_shmdir()) + 8;
        int badge_start = lay->r_header.col + 18 + commit_w + shmdir_w;
        int badge_w     = lay->ctrl_mode ? 13 : 15;
        if (mc >= badge_start && mc < badge_start + badge_w)
        {
            snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                     "Control Mode: Toggle write access & actions (key: c)");
            return;
        }
        int hover_badge_start = badge_start + badge_w + 1;
        int hover_badge_w     = lay->mouse_hover ? 15 : 16;
        if (mc >= hover_badge_start && mc < hover_badge_start + hover_badge_w)
        {
            snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                     "Mouse Hover: Toggle hover tooltips (key: m)");
            return;
        }
        /* Check header filter badges */
        for (int fb = 0; fb < lay->r_filter_count; fb++)
        {
            int fb_start = lay->r_filter_start[fb];
            int fb_w     = lay->r_filter_width[fb];
            if (mc >= fb_start && mc < fb_start + fb_w)
            {
                ov_focus_t  fpanel     = lay->r_filter_panel[fb];
                const char *panel_full = (fpanel == OV_FOCUS_STREAMS) ? "Streams"
                                         : (fpanel == OV_FOCUS_PROCS) ? "Processes"
                                         : (fpanel == OV_FOCUS_FPS)   ? "FPS"
                                                                      : "Panel";
                const char *fpat       = (fpanel != OV_FOCUS_GRAPH)
                                             ? ov_get_panel_filter_pattern(lay, fpanel)
                                             : ov_get_filter_pattern(lay);
                int is_act = (fpanel != OV_FOCUS_GRAPH) ? ov_is_panel_filter_active(lay, fpanel)
                                                        : ov_is_filter_active(lay);

                if (is_act)
                {
                    snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                             "%s Filter ON: Click or 'f' to pause, ESC to clear", panel_full);
                }
                else if (fpat[0] != '\0')
                {
                    snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                             "%s Filter OFF: Click or 'f' to resume, ESC to clear", panel_full);
                }
                else
                {
                    snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                             "%s Filter: Click or press [/] to set regex filter", panel_full);
                }
                return;
            }
        }
    }

    if (mr == lay->r_tabs.row)
    {
        int tab_widths[OV_VIEW_COUNT];
        int tabs_total_width = 0;
        for (int v = 0; v < OV_VIEW_COUNT; v++)
        {
            tab_widths[v] = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
            tabs_total_width += tab_widths[v];
        }
        int help_width = 11;
        int help_col   = (lay->term_cols >= tabs_total_width + help_width)
                             ? (lay->term_cols - help_width + 1)
                             : (tabs_total_width + 1);

        if (mc >= help_col && mc < help_col + help_width)
        {
            snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                     "Help: Toggle interactive help & keybinding menu (key: h)");
            return;
        }

        int tx = 1;
        for (int v = 0; v < OV_VIEW_COUNT; v++)
        {
            if (mc >= tx && mc < tx + tab_widths[v])
            {
                snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                         "View: Switch to %s view (key: F%d)", ov_view_label((ov_view_t) v), v + 2);
                return;
            }
            tx += tab_widths[v];
        }
    }

    int cmdlog_top = (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
    if (mr == cmdlog_top - 1 || mr == cmdlog_top)
    {
        lay->cmdlog_split_hover = 1;
    }

    if (lay->view == OV_VIEW_FPS)
    {
        if (mc >= lay->r_fps_list.width - 1 && mc <= lay->r_fps_list.width + 2)
        {
            lay->fps_split_hover = 1;
        }
    }
    else if (lay->view == OV_VIEW_DASHBOARD)
    {
        int h_split_row = lay->r_streams.row + lay->r_streams.height;
        int v_split_col = lay->r_streams.width;

        if (mr >= h_split_row - 1 && mr <= h_split_row + 1)
        {
            lay->dash_split_h_hover = 1;
        }
        if (mc >= v_split_col - 1 && mc <= v_split_col + 2)
        {
            lay->dash_split_v_hover = 1;
        }
    }

    if (lay->view == OV_VIEW_FPS)
    {
        if (INSIDE(lay->r_fps_params, mr, mc))
        {
            lay->hover_view = OV_FOCUS_FPS;
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->hover_view = OV_FOCUS_FPS;
            int body_row    = mr - lay->r_fps.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_fps + body_row;
                if (idx < m->nb_fps)
                {
                    lay->hover_idx = idx;
                }
            }
        }
    }
    else if (lay->view == OV_VIEW_STREAMS && INSIDE(lay->r_streams, mr, mc))
    {
        lay->hover_view = OV_FOCUS_STREAMS;
        int body_row    = mr - lay->r_streams.row - 3;
        if (body_row == -1 || body_row == -2)
        {
            lay->hover_is_header = 1;
        }
        else if (body_row >= 0)
        {
            int idx = lay->scroll_stream + body_row;
            if (idx < m->nb_streams)
            {
                lay->hover_idx = idx;
            }
        }
    }
    else if (lay->view == OV_VIEW_PROCS && INSIDE(lay->r_procs, mr, mc))
    {
        lay->hover_view = OV_FOCUS_PROCS;
        int body_row    = mr - lay->r_procs.row - 3;
        if (body_row == -1 || body_row == -2)
        {
            lay->hover_is_header = 1;
        }
        else if (body_row >= 0)
        {
            int idx = lay->scroll_proc + body_row;
            if (idx < m->nb_procs)
            {
                lay->hover_idx = idx;
            }
        }
    }
    else if ((lay->view == OV_VIEW_GRAPH || lay->view == OV_VIEW_LOOPS) &&
             INSIDE(lay->r_graph, mr, mc))
    {
        lay->hover_view = OV_FOCUS_GRAPH;
        int body_row    = mr - lay->r_graph.row - 2;
        if (body_row >= 0)
        {
            if (lay->graph_tab_mode == 0 && lay->view != OV_VIEW_LOOPS)
            {
                int idx = lay->scroll_graph + body_row;
                if (idx < m->nb_edges)
                {
                    lay->hover_idx = idx;
                }
            }
            else if (lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS)
            {
                int idx = lay->scroll_loop + body_row;
                if (idx < m->nb_loops)
                {
                    lay->hover_idx = idx;
                }
            }
            else
            {
                lay->hover_idx = body_row;
            }
        }
    }
    else if (lay->view == OV_VIEW_DASHBOARD)
    {
        if (INSIDE(lay->r_streams, mr, mc))
        {
            lay->hover_view = OV_FOCUS_STREAMS;
            int body_row    = mr - lay->r_streams.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_stream + body_row;
                if (idx < m->nb_streams)
                {
                    lay->hover_idx = idx;
                }
            }
        }
        else if (INSIDE(lay->r_procs, mr, mc))
        {
            lay->hover_view = OV_FOCUS_PROCS;
            int body_row    = mr - lay->r_procs.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_proc + body_row;
                if (idx < m->nb_procs)
                {
                    lay->hover_idx = idx;
                }
            }
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->hover_view = OV_FOCUS_FPS;
            int body_row    = mr - lay->r_fps.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_fps + body_row;
                if (idx < m->nb_fps)
                {
                    lay->hover_idx = idx;
                }
            }
        }
        else if (INSIDE(lay->r_graph, mr, mc))
        {
            lay->hover_view = OV_FOCUS_GRAPH;
            int body_row    = mr - lay->r_graph.row - 2;
            if (body_row >= 0)
            {
                if (lay->graph_tab_mode == 0)
                {
                    int idx = lay->scroll_graph + body_row;
                    if (idx < m->nb_edges)
                    {
                        lay->hover_idx = idx;
                    }
                }
                else if (lay->graph_tab_mode == 1)
                {
                    int idx = lay->scroll_loop + body_row;
                    if (idx < m->nb_loops)
                    {
                        lay->hover_idx = idx;
                    }
                }
                else
                {
                    lay->hover_idx = body_row;
                }
            }
        }
    }
}

void ov_hittest_resolve_globals(OV_LAYOUT *lay, const OV_MODEL *m)
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
        if (lay->graph_tab_mode == 0)
        {
            int start_node = ov_input_get_graph_start_node(lay, m);
            if (start_node >= 0)
            {
                SG_TREE_NODE rnodes[OV_MAX_NODES];
                int nb_rnodes = sg_compute_render_tree(m, start_node, lay->lineage_mode, rnodes);
                if (lay->hover_idx < nb_rnodes)
                {
                    const SG_TREE_NODE *rn       = &rnodes[lay->hover_idx];
                    int                 proc_idx = -1;
                    if (rn->reader_name[0] != '\0')
                    {
                        for (int i = 0; i < m->nb_procs; i++)
                        {
                            if (strcmp(m->procs[i].name, rn->reader_name) == 0)
                            {
                                proc_idx = i;
                                break;
                            }
                        }
                    }

                    int disp_len = 0;
                    for (int i = 0; rn->tree_prefix[i] != '\0';)
                    {
                        disp_len++;
                        i += utf8_char_length((unsigned char) rn->tree_prefix[i]);
                    }
                    if (rn->is_target)
                    {
                        disp_len += 2;
                    }
                    for (int i = 0; rn->name[i] != '\0';)
                    {
                        disp_len++;
                        i += utf8_char_length((unsigned char) rn->name[i]);
                    }

                    int click_on_proc = 0;
                    if (lay->hover_col_logical - lay->r_graph.col - 1 > disp_len + 1)
                    {
                        click_on_proc = 1;
                    }

                    if (click_on_proc && proc_idx >= 0)
                    {
                        lay->hover_global_proc = proc_idx;
                    }
                    else if (rn->stream_idx >= 0)
                    {
                        lay->hover_global_stream = rn->stream_idx;
                    }
                }
            }
        }
        else if (lay->graph_tab_mode == 2)
        {
            ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int        fsel  = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;

            int active_fps = -1;
            if (focus == OV_FOCUS_FPS && fsel >= 0 && fsel < m->nb_fps)
            {
                active_fps = fsel;
            }
            else if (focus != OV_FOCUS_STREAMS && focus != OV_FOCUS_PROCS)
            {
                if (fsel >= 0 && fsel < m->nb_fps)
                {
                    active_fps = fsel;
                }
            }

            if (active_fps >= 0)
            {
                const OV_FPS *f = &m->fps[active_fps];
                if (f->nb_disp_params > 0)
                {
                    int header_rows = 3 + (f->description[0] != '\0' ? 1 : 0);
                    int param_row   = lay->hover_idx - header_rows;
                    if (param_row >= 0)
                    {
                        int                  dp     = lay->param_scroll + param_row;
                        const OV_FPS_PARAMS *params = ov_fps_get_params(f->name);
                        if (params != NULL && dp >= 0 && dp < params->nb_disp_params)
                        {
                            if (params->disp_param_type[dp] == FPTYPE_STREAMNAME)
                            {
                                int si = ov_find_stream_by_name(m, params->disp_param_value[dp]);
                                if (si >= 0)
                                {
                                    lay->hover_global_stream = si;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

