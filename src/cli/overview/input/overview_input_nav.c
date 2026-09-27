// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_nav.c
 * @brief Cursor movement, scrolling, and page navigation across panels and graph
 */

#include "overview_input_internal.h"

/**
 * ov_input__handle_navigation - dispatch cursor keys, paging, and scroll navigation.
 * @key: Pressed key code (arrows, Home/End, PgUp/PgDn, etc.).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if navigation key was consumed, 0 otherwise.
 */
int ov_input__handle_navigation(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    int *sel                            = NULL;
    int *scroll __attribute__((unused)) = NULL;
    int  count                          = 0;
    int  page_h                         = 10;

    /* F5 view (OV_VIEW_FPS) param-tree intercept */
    if (lay->view == OV_VIEW_FPS)
    {
        int rc = ov_input_nav_fps(key, lay, m);
        if (rc != 0)
        {
            return rc;
        }
    }

    switch (lay->focus)
    {
    case OV_FOCUS_STREAMS:
        sel    = &lay->sel_stream;
        scroll = &lay->scroll_stream;
        {
            int fidx[OV_MAX_STREAMS];
            count = ov_filter_streams(lay, m, NULL, fidx, OV_MAX_STREAMS);
        }
        page_h = lay->r_streams.height - 3;
        break;

    case OV_FOCUS_PROCS:
        sel    = &lay->sel_proc;
        scroll = &lay->scroll_proc;
        {
            int fidx[OV_MAX_PROCS];
            count = ov_filter_procs(lay, m, NULL, fidx, OV_MAX_PROCS);
        }
        page_h = lay->r_procs.height - 3;
        break;

    case OV_FOCUS_FPS:
        sel    = &lay->sel_fps;
        scroll = &lay->scroll_fps;
        {
            int fidx[OV_MAX_FPS];
            count = ov_filter_fps(lay, m, NULL, fidx, OV_MAX_FPS);
        }
        page_h = lay->r_fps.height - 3;
        break;

    case OV_FOCUS_GRAPH:
        if (lay->graph_tab_mode == 0)
        {
            sel    = &lay->sel_graph;
            scroll = &lay->scroll_graph;
            {
                int start_node = ov_input_get_graph_start_node(lay, m);
                if (start_node >= 0)
                {
                    SG_RENDER_NODE rnodes[OV_MAX_NODES];
                    count = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                }
                else
                {
                    count = 0;
                }
            }
            page_h = lay->r_graph.height - 3;
        }
        else if (lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS)
        {
            sel    = &lay->sel_loop;
            scroll = &lay->scroll_loop;
            count  = m->nb_loops;
            page_h = (lay->r_graph.height - 3 >= 6)
                         ? ((lay->r_graph.height - 3 > 8) ? ((lay->r_graph.height - 3) / 2) : 3)
                         : (lay->r_graph.height - 3);
            break;
        }
        else if (lay->graph_tab_mode == 2)
        {
            /* DETAILS tab: param nav if FPS selected */
            int fps_sel = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;
            int has_fps_params =
                (fps_sel >= 0 && fps_sel < m->nb_fps && m->fps[fps_sel].nb_disp_params > 0);
            int nparams = has_fps_params ? m->fps[fps_sel].nb_disp_params : 0;

            if (has_fps_params && lay->param_sel >= 0)
            {
                /* Navigate parameter cursor */
                if (key == OV_KEY_UP)
                {
                    if (lay->param_sel > 0)
                    {
                        lay->param_sel--;
                    }
                }
                else if (key == OV_KEY_DOWN)
                {
                    if (lay->param_sel < nparams - 1)
                    {
                        lay->param_sel++;
                    }
                }
                else if (key == OV_KEY_PGUP)
                {
                    lay->param_sel -= page_h;
                    if (lay->param_sel < 0)
                    {
                        lay->param_sel = 0;
                    }
                }
                else if (key == OV_KEY_PGDN)
                {
                    lay->param_sel += page_h;
                    if (lay->param_sel >= nparams)
                    {
                        lay->param_sel = nparams - 1;
                    }
                }
                else if (key == OV_KEY_HOME)
                {
                    lay->param_sel = 0;
                }
                else if (key == OV_KEY_END)
                {
                    lay->param_sel = nparams - 1;
                    if (lay->param_sel < 0)
                    {
                        lay->param_sel = 0;
                    }
                }
                else if (key == OV_KEY_ESC)
                {
                    lay->param_sel = -1;
                }
                else if (key == OV_KEY_ENTER || key == '\r' || key == '\n')
                {
                    if (!lay->ctrl_mode)
                    {
                        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                                       "Edit requires "
                                       "CONTROL mode (press c to toggle CTRL mode ON/OFF)");
                    }
                    else
                    {
                        ov_fps_inline_edit(lay, m->fps[fps_sel].name, lay->param_sel);
                    }
                }
                return 1;
            }

            /* Init param_sel on first nav key */
            if (has_fps_params && lay->param_sel < 0 && (key == OV_KEY_DOWN || key == OV_KEY_UP))
            {
                lay->param_sel = 0;
                return 1;
            }

            /* Fallback: detail scroll */
            sel    = NULL;
            scroll = NULL;
            count  = lay->detail_total_lines;
            page_h = lay->r_graph.height - 3;

            if (key == OV_KEY_UP)
            {
                if (lay->scroll_detail > 0)
                {
                    lay->scroll_detail--;
                }
            }
            else if (key == OV_KEY_DOWN)
            {
                if (lay->scroll_detail < lay->detail_total_lines - page_h)
                {
                    lay->scroll_detail++;
                }
            }
            else if (key == OV_KEY_PGUP)
            {
                lay->scroll_detail -= page_h;
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
            }
            else if (key == OV_KEY_PGDN)
            {
                lay->scroll_detail += page_h;
                if (lay->scroll_detail > lay->detail_total_lines - page_h)
                {
                    lay->scroll_detail = lay->detail_total_lines - page_h;
                }
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
            }
            else if (key == OV_KEY_HOME)
            {
                lay->scroll_detail = 0;
            }
            else if (key == OV_KEY_END)
            {
                lay->scroll_detail = lay->detail_total_lines - page_h;
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
            }
            return 1;
        }
        else
        {
            sel    = NULL;
            scroll = NULL;
        }
        break;

    default:
        break;
    }

    if (sel != NULL)
    {
        int old_sel    = *sel;
        int navigated  = 0;
        int is_nav_key = 0;

        if (key == OV_KEY_UP)
        {
            is_nav_key = 1;
            if (*sel > 0)
            {
                (*sel)--;
                navigated = 1;
            }
        }
        else if (key == OV_KEY_DOWN)
        {
            is_nav_key = 1;
            if (*sel < count - 1)
            {
                (*sel)++;
                navigated = 1;
            }
        }
        else if (key == OV_KEY_PGUP)
        {
            is_nav_key = 1;
            *sel -= page_h;
            if (*sel < 0)
            {
                *sel = 0;
            }
            navigated = 1;
        }
        else if (key == OV_KEY_PGDN)
        {
            is_nav_key = 1;
            *sel += page_h;
            if (*sel >= count)
            {
                *sel = count - 1;
            }
            if (*sel < 0)
            {
                *sel = 0;
            }
            navigated = 1;
        }
        else if (key == OV_KEY_HOME)
        {
            is_nav_key = 1;
            *sel       = 0;
            navigated  = 1;
        }
        else if (key == OV_KEY_END)
        {
            is_nav_key = 1;
            *sel       = count - 1;
            if (*sel < 0)
            {
                *sel = 0;
            }
            navigated = 1;
        }

        if (is_nav_key)
        {
            if (navigated)
            {
                if (lay->focus == OV_FOCUS_STREAMS)
                {
                    lay->sel_name_stream[0] = '\0';
                }
                else if (lay->focus == OV_FOCUS_PROCS)
                {
                    lay->sel_name_proc[0] = '\0';
                }
                else if (lay->focus == OV_FOCUS_FPS)
                {
                    lay->sel_name_fps[0] = '\0';
                    if (lay->view == OV_VIEW_FPS)
                    {
                        lay->fps_param_focus = 0;
                    }
                }

                if (lay->focus == OV_FOCUS_GRAPH && lay->graph_tab_mode == 0 && *sel != old_sel)
                {
                    int start_node = ov_input_get_graph_start_node(lay, m);
                    if (start_node >= 0)
                    {
                        SG_RENDER_NODE rnodes[OV_MAX_NODES];
                        int            n_rnodes =
                            sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                        if (*sel < n_rnodes)
                        {
                            const SG_RENDER_NODE *rn   = &rnodes[*sel];
                            const OV_NODE        *node = &m->nodes[rn->node_idx];

                            if (node->type == OV_NODE_STREAM)
                            {
                                lay->sel_stream         = node->index;
                                lay->sel_name_stream[0] = '\0';
                                if (lay->sel_stream < lay->scroll_stream)
                                {
                                    lay->scroll_stream = lay->sel_stream;
                                }
                                if (lay->r_streams.height > 3 &&
                                    lay->sel_stream >=
                                        lay->scroll_stream + lay->r_streams.height - 3)
                                {
                                    lay->scroll_stream =
                                        lay->sel_stream - (lay->r_streams.height - 3) + 1;
                                }
                            }
                            else if (node->type == OV_NODE_PROC)
                            {
                                lay->sel_proc         = node->index;
                                lay->sel_name_proc[0] = '\0';
                                if (lay->sel_proc < lay->scroll_proc)
                                {
                                    lay->scroll_proc = lay->sel_proc;
                                }
                                if (lay->r_procs.height > 3 &&
                                    lay->sel_proc >= lay->scroll_proc + lay->r_procs.height - 3)
                                {
                                    lay->scroll_proc =
                                        lay->sel_proc - (lay->r_procs.height - 3) + 1;
                                }
                            }
                            else if (node->type == OV_NODE_FPS)
                            {
                                lay->sel_fps         = node->index;
                                lay->sel_name_fps[0] = '\0';
                                if (lay->sel_fps < lay->scroll_fps)
                                {
                                    lay->scroll_fps = lay->sel_fps;
                                }
                                if (lay->r_fps.height > 3 &&
                                    lay->sel_fps >= lay->scroll_fps + lay->r_fps.height - 3)
                                {
                                    lay->scroll_fps = lay->sel_fps - (lay->r_fps.height - 3) + 1;
                                }
                            }
                        }
                    }
                }
            }
            return 1;
        }
    }

    return 0;
}
