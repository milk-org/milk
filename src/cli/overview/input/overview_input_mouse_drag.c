// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_mouse_drag.c
 * @brief Mouse drag splitter adjustments and mouse wheel scroll dispatching
 */

#include "overview_input_internal.h"

/**
 * ov_input_mouse_drag - handle mouse drag events for splitters and command log.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 * @mr:  Mouse row.
 * @mc:  Mouse column.
 *
 * Return: 1 if handled.
 */
int ov_input_mouse_drag(
    OV_LAYOUT      *lay,
    const OV_MODEL *m,
    int             mr,
    int             mc)
{
    (void) m;

    /* Global: Command log panel height drag */
    int cmdlog_top =
        (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
    if (lay->cmdlog_dragging || (mr == cmdlog_top - 1 || mr == cmdlog_top))
    {
        lay->cmdlog_dragging = 1;
        int new_h            = lay->term_rows - 1 - mr;
        if (new_h < 0)
        {
            new_h = 0;
        }
        if (new_h > lay->term_rows / 2)
        {
            new_h = lay->term_rows / 2;
        }
        if (new_h != lay->cmdlog_rows)
        {
            lay->cmdlog_rows = new_h;
            ov_buf_force_clear();
        }
        return 1;
    }

    if (lay->view == OV_VIEW_FPS)
    {
        if (lay->fps_split_dragging ||
            (mc >= lay->r_fps_list.width - 1 && mc <= lay->r_fps_list.width + 2))
        {
            lay->fps_split_dragging = 1;
            float ratio             = (float) mc / lay->term_cols;
            if (ratio < 0.1f)
            {
                ratio = 0.1f;
            }
            if (ratio > 0.9f)
            {
                ratio = 0.9f;
            }
            lay->fps_split_ratio = ratio;
            return 1;
        }
    }
    if (lay->view == OV_VIEW_DASHBOARD)
    {
        int h_split_row = lay->r_streams.row + lay->r_streams.height;
        int v_split_col = lay->r_streams.width;

        int handled = 0;
        if (lay->dash_split_h_dragging || (mr >= h_split_row - 1 && mr <= h_split_row + 1))
        {
            lay->dash_split_h_dragging = 1;
            int log_h                  = lay->cmdlog_rows;
            if (log_h < 0)
            {
                log_h = 0;
            }
            int body_top = 4;
            int body_h   = lay->term_rows - 4 - log_h;
            if (body_h < 4)
            {
                body_h = 4;
            }
            float ratio = (float) (mr - body_top) / body_h;
            if (ratio < 0.1f)
            {
                ratio = 0.1f;
            }
            if (ratio > 0.9f)
            {
                ratio = 0.9f;
            }
            lay->dash_split_h_ratio = ratio;
            handled                 = 1;
        }
        if (lay->dash_split_v_dragging || (mc >= v_split_col - 1 && mc <= v_split_col + 2))
        {
            lay->dash_split_v_dragging = 1;
            float ratio                = (float) mc / lay->term_cols;
            if (ratio < 0.1f)
            {
                ratio = 0.1f;
            }
            if (ratio > 0.9f)
            {
                ratio = 0.9f;
            }
            lay->dash_split_v_ratio = ratio;
            handled                 = 1;
        }
        if (handled)
        {
            return 1;
        }
    }
    return 1;
}

/**
 * ov_input_mouse_wheel - handle mouse wheel events for panel scrolling.
 * @key: Mouse wheel key code.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if handled.
 */
int ov_input_mouse_wheel(
    int             key,
    OV_LAYOUT      *lay,
    const OV_MODEL *m)
{
    /* Ctrl+scroll: cycle views (#12) */
    if (key == OV_KEY_CTRL_SCROLL_UP || key == OV_KEY_CTRL_SCROLL_DOWN)
    {
        int v = (int) lay->view;
        if (key == OV_KEY_CTRL_SCROLL_UP)
        {
            v--;
            if (v < 0)
            {
                v = OV_VIEW_COUNT - 1;
            }
        }
        else
        {
            v++;
            if (v >= OV_VIEW_COUNT)
            {
                v = 0;
            }
        }
        lay->view = (ov_view_t) v;
        return 1;
    }

    if (key == OV_KEY_MOUSE_UP || key == OV_KEY_MOUSE_DOWN)
    {
        int mr  = ov_mouse_row;
        int mc  = ov_mouse_col;
        int dir = (key == OV_KEY_MOUSE_UP) ? -3 : 3;

        int *sel    = NULL;
        int *scroll = NULL;
        int  count  = 0;
        int  page_h = 10;

        if (lay->view == OV_VIEW_FPS)
        {
            if (INSIDE(lay->r_fps_params, mr, mc))
            {
                sel      = &lay->fps_param_sel;
                scroll   = &lay->fps_param_scroll;
                count    = 0;
                int fsel = lay->sel_fps;
                if (fsel >= 0 && fsel < m->nb_fps)
                {
                    const OV_FPS   *fps = &m->fps[fsel];
                    fps_tree_item_t items[1024];
                    count = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);
                }
                page_h = lay->r_fps_params.height - 3;
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                sel    = &lay->sel_fps;
                scroll = &lay->scroll_fps;
                count  = ov_input_get_filtered_count(OV_FOCUS_FPS, lay, m);
                page_h = lay->r_fps.height - 3;
            }
        }
        else if (lay->view == OV_VIEW_STREAMS)
        {
            sel    = &lay->sel_stream;
            scroll = &lay->scroll_stream;
            count  = ov_input_get_filtered_count(OV_FOCUS_STREAMS, lay, m);
            page_h = lay->r_streams.height - 3;
        }
        else if (lay->view == OV_VIEW_PROCS)
        {
            sel    = &lay->sel_proc;
            scroll = &lay->scroll_proc;
            count  = ov_input_get_filtered_count(OV_FOCUS_PROCS, lay, m);
            page_h = lay->r_procs.height - 3;
        }
        else if (lay->view == OV_VIEW_GRAPH)
        {
            if (lay->graph_tab_mode == 0)
            {
                sel            = &lay->sel_graph;
                scroll         = &lay->scroll_graph;
                int start_node = ov_input_get_graph_start_node(lay, m);
                if (start_node >= 0)
                {
                    SG_RENDER_NODE rnodes[OV_MAX_NODES];
                    count = sg_compute_render_nodes(m, start_node,
                                                   lay->lineage_mode, rnodes);
                }
                else
                {
                    count = m->nb_edges;
                }
                page_h = lay->r_graph.height - 3;
            }
            else if (lay->graph_tab_mode == 1)
            {
                sel    = &lay->sel_loop;
                scroll = &lay->scroll_loop;
                count  = m->nb_loops;
                page_h = (lay->r_graph.height - 3 >= 6)
                             ? ((lay->r_graph.height - 3 > 8)
                                    ? ((lay->r_graph.height - 3) / 2) : 3)
                             : (lay->r_graph.height - 3);
            }
            else if (lay->graph_tab_mode == 2)
            {
                page_h = lay->r_graph.height - 3;
                lay->scroll_detail += dir;
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
                if (lay->scroll_detail > lay->detail_total_lines - page_h)
                {
                    lay->scroll_detail = lay->detail_total_lines - page_h;
                }
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
                return 1;
            }
            else
            {
                /* RESOURCES panel */
                return 1;
            }
        }
        else if (lay->view == OV_VIEW_LOOPS)
        {
            sel    = &lay->sel_loop;
            scroll = &lay->scroll_loop;
            count  = m->nb_loops;
            page_h = lay->r_graph.height - 3;
        }
        else if (lay->view == OV_VIEW_DASHBOARD)
        {
            if (INSIDE(lay->r_streams, mr, mc))
            {
                sel    = &lay->sel_stream;
                scroll = &lay->scroll_stream;
                count  = ov_input_get_filtered_count(OV_FOCUS_STREAMS, lay, m);
                page_h = lay->r_streams.height - 3;
            }
            else if (INSIDE(lay->r_procs, mr, mc))
            {
                sel    = &lay->sel_proc;
                scroll = &lay->scroll_proc;
                count  = ov_input_get_filtered_count(OV_FOCUS_PROCS, lay, m);
                page_h = lay->r_procs.height - 3;
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                sel    = &lay->sel_fps;
                scroll = &lay->scroll_fps;
                count  = ov_input_get_filtered_count(OV_FOCUS_FPS, lay, m);
                page_h = lay->r_fps.height - 3;
            }
            else if (INSIDE(lay->r_graph, mr, mc))
            {
                if (lay->graph_tab_mode == 0)
                {
                    sel            = &lay->sel_graph;
                    scroll         = &lay->scroll_graph;
                    int start_node = ov_input_get_graph_start_node(lay, m);
                    if (start_node >= 0)
                    {
                        SG_RENDER_NODE rnodes[OV_MAX_NODES];
                        count = sg_compute_render_nodes(m, start_node,
                                                       lay->lineage_mode, rnodes);
                    }
                    else
                    {
                        count = m->nb_edges;
                    }
                    page_h = lay->r_graph.height - 3;
                }
                else if (lay->graph_tab_mode == 1)
                {
                    sel    = &lay->sel_loop;
                    scroll = &lay->scroll_loop;
                    count  = m->nb_loops;
                    page_h =
                        (lay->r_graph.height - 3 >= 6)
                            ? ((lay->r_graph.height - 3 > 8)
                                   ? ((lay->r_graph.height - 3) / 2) : 3)
                            : (lay->r_graph.height - 3);
                }
                else if (lay->graph_tab_mode == 2)
                {
                    page_h = lay->r_graph.height - 3;
                    lay->scroll_detail += dir;
                    if (lay->scroll_detail < 0)
                    {
                        lay->scroll_detail = 0;
                    }
                    if (lay->scroll_detail > lay->detail_total_lines - page_h)
                    {
                        lay->scroll_detail = lay->detail_total_lines - page_h;
                    }
                    if (lay->scroll_detail < 0)
                    {
                        lay->scroll_detail = 0;
                    }
                    return 1;
                }
                else
                {
                    /* RESOURCES panel */
                    return 1;
                }
            }
        }

        if (sel != NULL && scroll != NULL)
        {
            *scroll += dir;
            if (*scroll < 0)
            {
                *scroll = 0;
            }
            if (page_h > 0 && *scroll > count - page_h)
            {
                *scroll = count - page_h;
                if (*scroll < 0)
                {
                    *scroll = 0;
                }
            }

            *sel += dir;
            if (*sel < *scroll)
            {
                *sel = *scroll;
            }
            if (page_h > 0 && *sel >= *scroll + page_h)
            {
                *sel = *scroll + page_h - 1;
            }
            if (*sel >= count)
            {
                *sel = count - 1;
            }
            if (*sel < 0)
            {
                *sel = 0;
            }

            if (sel == &lay->sel_stream)
            {
                lay->sel_name_stream[0] = '\0';
            }
            else if (sel == &lay->sel_proc)
            {
                lay->sel_name_proc[0] = '\0';
            }
            else if (sel == &lay->sel_fps)
            {
                lay->sel_name_fps[0] = '\0';
            }
        }
        return 1;
    }

    return 0;
}
