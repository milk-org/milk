// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_mouse_panels.c
 * @brief Panel-specific mouse click hit-testing, selection, and double-click actions
 */

#include "overview_input_internal.h"
#include <string.h>

/**
 * ov_input_mouse_panel_click - dispatch mouse clicks within view panels.
 * @lay:    Pointer to layout structure.
 * @m:      Pointer to data model snapshot.
 * @mr:     Mouse row coordinate.
 * @mc:     Mouse column coordinate.
 * @is_dbl: 1 if double-click detected, 0 otherwise.
 *
 * Return: 1 if click was consumed.
 */
int ov_input_mouse_panel_click(
    OV_LAYOUT      *lay,
    const OV_MODEL *m,
    int             mr,
    int             mc,
    int             is_dbl)
{
    if (lay->view == OV_VIEW_FPS)
    {
        /* F5: Check for split drag */
        if (mc == lay->r_fps_list.width || mc == lay->r_fps_list.width + 1)
        {
            lay->fps_split_dragging = 1;
            return 1;
        }

        /* F5: left = fps list, right = params */
        if (INSIDE(lay->r_fps_params, mr, mc))
        {
            lay->focus           = OV_FOCUS_FPS;
            lay->fps_param_focus = 1;
            int body_row         = mr - lay->r_fps_params.row - 2;
            if (body_row >= 0)
            {
                int fsel = lay->sel_fps;
                if (fsel >= 0 && fsel < m->nb_fps)
                {
                    const OV_FPS   *fps = &m->fps[fsel];
                    fps_tree_item_t items[1024];
                    int nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);
                    if (nitems > 0)
                    {
                        int idx = lay->fps_param_scroll + body_row;
                        if (idx >= nitems)
                        {
                            lay->fps_param_sel = nitems - 1;
                        }
                        else
                        {
                            lay->fps_param_sel = idx;
                        }
                    }
                }
            }
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->focus           = OV_FOCUS_FPS;
            lay->fps_param_focus = 0;
            int body_row         = mr - lay->r_fps.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                ov_input__fps_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_fps + body_row;
                if (idx < m->nb_fps)
                {
                    if (lay->sel_fps != idx)
                    {
                        lay->sel_fps         = idx;
                        lay->sel_name_fps[0] = '\0';
                    }
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 2;
                    }
                }
            }
        }
    }
    else if (lay->view == OV_VIEW_STREAMS && INSIDE(lay->r_streams, mr, mc))
    {
        lay->focus   = OV_FOCUS_STREAMS;
        int body_row = mr - lay->r_streams.row - 3;
        if (body_row == -1 || body_row == -2)
        {
            ov_input__streams_header_click(lay, mc);
        }
        else if (body_row >= 0)
        {
            int idx = lay->scroll_stream + body_row;
            if (idx < m->nb_streams)
            {
                lay->sel_stream         = idx;
                lay->sel_name_stream[0] = '\0';
                if (is_dbl)
                {
                    lay->graph_tab_mode = 2;
                }
            }
        }
    }
    else if (lay->view == OV_VIEW_PROCS && INSIDE(lay->r_procs, mr, mc))
    {
        lay->focus   = OV_FOCUS_PROCS;
        int body_row = mr - lay->r_procs.row - 3;
        if (body_row == -1 || body_row == -2)
        {
            ov_input__procs_header_click(lay, mc);
        }
        else if (body_row >= 0)
        {
            int idx = lay->scroll_proc + body_row;
            if (idx < m->nb_procs)
            {
                lay->sel_proc         = idx;
                lay->sel_name_proc[0] = '\0';
                if (is_dbl)
                {
                    lay->graph_tab_mode = 2;
                }
            }
        }
    }
    else if ((lay->view == OV_VIEW_GRAPH || lay->view == OV_VIEW_LOOPS) &&
             INSIDE(lay->r_graph, mr, mc))
    {
        lay->focus = OV_FOCUS_GRAPH;

        /* Tab header click */
        if (lay->view == OV_VIEW_GRAPH && mr == lay->r_graph.row)
        {
            const char *dtabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
            int         ti      = ov_input_hit_panel_tab(mc, lay->r_graph.col, dtabs, 4);
            if (ti >= 0)
            {
                lay->graph_tab_mode = ti;
                ov_scan_force_update();
            }
        }
        else
        {
            int body_row = mr - lay->r_graph.row - 2;
            if (body_row >= 0)
            {
                if (lay->graph_tab_mode == 0)
                {
                    int start_node = ov_input_get_graph_start_node(lay, m);
                    if (start_node >= 0)
                    {
                        SG_TREE_NODE rnodes[OV_MAX_NODES];
                        int          nb_rnodes =
                            sg_compute_render_tree(m, start_node, lay->lineage_mode, rnodes);

                        int idx = lay->scroll_graph + body_row;
                        if (idx < nb_rnodes)
                        {
                            lay->sel_graph         = idx;
                            const SG_TREE_NODE *rn = &rnodes[idx];

                            int proc_idx = -1;
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

                            /* Estimate click position to decide stream vs proc */
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
                            if (mc - lay->r_graph.col - 1 > disp_len + 1)
                            {
                                click_on_proc = 1;
                            }

                            if (click_on_proc && proc_idx >= 0)
                            {
                                lay->focus            = OV_FOCUS_PROCS;
                                lay->sel_proc         = proc_idx;
                                lay->sel_name_proc[0] = '\0';
                            }
                            else if (rn->stream_idx >= 0)
                            {
                                lay->focus              = OV_FOCUS_STREAMS;
                                lay->sel_stream         = rn->stream_idx;
                                lay->sel_name_stream[0] = '\0';
                            }

                            if (is_dbl)
                            {
                                lay->view = OV_VIEW_DASHBOARD;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (lay->view == OV_VIEW_DASHBOARD)
    {
        /* Dashboard: four non-overlapping rects */
        if (INSIDE(lay->r_streams, mr, mc))
        {
            lay->focus   = OV_FOCUS_STREAMS;
            int body_row = mr - lay->r_streams.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                ov_input__streams_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_stream + body_row;
                if (idx < m->nb_streams)
                {
                    lay->sel_stream         = idx;
                    lay->sel_name_stream[0] = '\0';
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 2;
                    }
                }
            }
        }
        else if (INSIDE(lay->r_procs, mr, mc))
        {
            lay->focus   = OV_FOCUS_PROCS;
            int body_row = mr - lay->r_procs.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                ov_input__procs_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_proc + body_row;
                if (idx < m->nb_procs)
                {
                    lay->sel_proc         = idx;
                    lay->sel_name_proc[0] = '\0';
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 2;
                    }
                }
            }
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->focus   = OV_FOCUS_FPS;
            int body_row = mr - lay->r_fps.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                ov_input__fps_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_fps + body_row;
                if (idx < m->nb_fps)
                {
                    lay->sel_fps         = idx;
                    lay->sel_name_fps[0] = '\0';
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 2;
                    }
                }
            }
        }
        else if (INSIDE(lay->r_graph, mr, mc))
        {
            lay->focus = OV_FOCUS_GRAPH;

            /* Tab header click */
            if (mr == lay->r_graph.row)
            {
                const char *dtabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
                int         ti      = ov_input_hit_panel_tab(mc, lay->r_graph.col, dtabs, 4);
                if (ti >= 0)
                {
                    lay->graph_tab_mode = ti;
                    ov_scan_force_update();
                }
            }
            else
            {
                int body_row = mr - lay->r_graph.row - 2;
                if (body_row >= 0)
                {
                    if (lay->graph_tab_mode == 0)
                    {
                        int start_node = ov_input_get_graph_start_node(lay, m);
                        if (start_node >= 0)
                        {
                            SG_TREE_NODE rnodes[OV_MAX_NODES];
                            int nb_rnodes = sg_compute_render_tree(m, start_node,
                                                                   lay->lineage_mode, rnodes);

                            int idx = lay->scroll_graph + body_row;
                            if (idx < nb_rnodes)
                            {
                                lay->sel_graph         = idx;
                                const SG_TREE_NODE *rn = &rnodes[idx];

                                int proc_idx = -1;
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

                                /* Estimate click position to decide stream vs proc */
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
                                if (mc - lay->r_graph.col - 1 > disp_len + 1)
                                {
                                    click_on_proc = 1;
                                }

                                if (click_on_proc && proc_idx >= 0)
                                {
                                    lay->focus            = OV_FOCUS_PROCS;
                                    lay->sel_proc         = proc_idx;
                                    lay->sel_name_proc[0] = '\0';
                                    if (is_dbl)
                                    {
                                        lay->view = OV_VIEW_PROCS;
                                    }
                                }
                                else if (rn->stream_idx >= 0)
                                {
                                    lay->focus              = OV_FOCUS_STREAMS;
                                    lay->sel_stream         = rn->stream_idx;
                                    lay->sel_name_stream[0] = '\0';
                                    if (is_dbl)
                                    {
                                        lay->view = OV_VIEW_STREAMS;
                                    }
                                }
                            }
                        }
                    }
                    else if (lay->graph_tab_mode == 1)
                    {
                        int max_rows = lay->r_graph.height - 3;
                        int list_rows =
                            (max_rows >= 6) ? ((max_rows > 8) ? (max_rows / 2) : 3) : max_rows;
                        if (body_row >= 0 && body_row < list_rows)
                        {
                            int li = lay->scroll_loop + body_row;
                            if (li >= 0 && li < m->nb_loops)
                            {
                                lay->sel_loop = li;
                                if (is_dbl)
                                {
                                    lay->view = OV_VIEW_LOOPS;
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
                                int param_row   = body_row - header_rows;
                                if (param_row >= 0)
                                {
                                    int                  dp     = lay->param_scroll + param_row;
                                    const OV_FPS_PARAMS *params = ov_fps_get_params(f->name);
                                    if (params != NULL && dp >= 0 &&
                                        dp < params->nb_disp_params)
                                    {
                                        lay->param_sel = dp;
                                        if (params->disp_param_type[dp] == FPTYPE_STREAMNAME)
                                        {
                                            int si = ov_find_stream_by_name(
                                                m, params->disp_param_value[dp]);
                                            if (si >= 0)
                                            {
                                                lay->focus              = OV_FOCUS_STREAMS;
                                                lay->sel_stream         = si;
                                                lay->sel_name_stream[0] = '\0';
                                                if (is_dbl)
                                                {
                                                    lay->view = OV_VIEW_STREAMS;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* Check if clicking on cmdlog border to start dragging */
    int cmdlog_top =
        (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
    if (mr == cmdlog_top - 1 || mr == cmdlog_top)
    {
        lay->cmdlog_dragging = 1;
    }

    return 1;
}
