// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_nav.c
 * @brief Navigation, selection, scrolling, and lineage/ancestry navigation.
 */

#include "overview_input_internal.h"

/**
 * ov_input_get_graph_start_node - find graph node index corresponding to active selection.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: Node index in m->nodes, or -1 if no matching selection.
 */
int ov_input_get_graph_start_node(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_focus_t eff_focus   = lay->freeze ? lay->freeze_focus : lay->focus;
    int        target_type = -1;
    int        target_idx  = -1;

    if (eff_focus == OV_FOCUS_STREAMS || eff_focus == OV_FOCUS_GRAPH)
    {
        target_type = OV_NODE_STREAM;
        target_idx  = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    }
    else if (eff_focus == OV_FOCUS_PROCS)
    {
        target_type = OV_NODE_PROC;
        target_idx  = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
    }
    else if (eff_focus == OV_FOCUS_FPS)
    {
        target_type = OV_NODE_FPS;
        target_idx  = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;
    }

    if (target_type != -1 && target_idx != -1)
    {
        for (int i = 0; i < m->nb_nodes; i++)
        {
            if (m->nodes[i].type == target_type && m->nodes[i].index == target_idx)
            {
                return i;
            }
        }
    }
    return -1;
}

/**
 * ov_input_get_filtered_count - count visible items in a panel taking active filter into account.
 * @focus: Target focus panel (OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS).
 * @lay:   Pointer to layout structure.
 * @m:     Pointer to data model snapshot.
 *
 * Return: Number of items matching current filter.
 */
int ov_input_get_filtered_count(int focus, const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int count = 0;
    if (focus == OV_FOCUS_STREAMS)
    {
        count            = m->nb_streams;
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_STREAMS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->streams[i].name;
            }
            int fidx[OV_MAX_STREAMS];
            count = ov_filter_build(filt, names, count, fidx, OV_MAX_STREAMS);
        }
    }
    else if (focus == OV_FOCUS_PROCS)
    {
        count            = m->nb_procs;
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_PROCS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->procs[i].name;
            }
            int fidx[OV_MAX_PROCS];
            count = ov_filter_build(filt, names, count, fidx, OV_MAX_PROCS);
        }
    }
    else if (focus == OV_FOCUS_FPS)
    {
        count            = m->nb_fps;
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_FPS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_FPS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->fps[i].name;
            }
            int fidx[OV_MAX_FPS];
            count = ov_filter_build(filt, names, count, fidx, OV_MAX_FPS);
        }
    }
    return count;
}

/**
 * find_relative_node_of_type - traverse BFS graph edges upstream or downstream.
 * @m:           Pointer to data model snapshot.
 * @start_node:  Source node index in graph.
 * @target_type: Desired node type (stream, proc, fps) or -1 for any.
 * @upstream:    1 for upstream parents, 0 for downstream children.
 *
 * Return: Target relative node index, or -1 if none found.
 */
static int find_relative_node_of_type(const OV_MODEL *m,
                                      int             start_node,
                                      int             target_type,
                                      int             upstream)
{
    int queue[OV_MAX_NODES];
    int visited[OV_MAX_NODES];
    memset(visited, 0, sizeof(visited));
    int head = 0, tail = 0;

    queue[tail++]       = start_node;
    visited[start_node] = 1;

    while (head < tail)
    {
        int curr = queue[head++];

        for (int i = 0; i < m->nb_edges; i++)
        {
            int next = -1;
            if (upstream && m->edges[i].tgt_node == curr)
            {
                next = m->edges[i].src_node;
            }
            else if (!upstream && m->edges[i].src_node == curr)
            {
                next = m->edges[i].tgt_node;
            }

            if (next >= 0 && !visited[next])
            {
                if (target_type < 0 || m->nodes[next].type == target_type)
                {
                    return next;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }
    }
    return -1;
}

/**
 * ov_input__handle_ancestry_nav - navigate to immediate upstream parent or downstream child in DAG.
 * @key: Pressed key code (Shift+Up or Shift+Down).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if ancestry navigation key was handled, 0 otherwise.
 */
int ov_input__handle_ancestry_nav(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key != OV_KEY_SHIFT_UP && key != OV_KEY_SHIFT_DOWN)
    {
        return 0;
    }

    int current_node = -1;
    int target_type  = -1;
    if (lay->focus == OV_FOCUS_STREAMS && lay->sel_stream >= 0 && lay->sel_stream < m->nb_streams)
    {
        current_node = m->streams[lay->sel_stream].node_idx;
        target_type  = OV_NODE_STREAM;
    }
    else if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0 && lay->sel_proc < m->nb_procs)
    {
        current_node = m->procs[lay->sel_proc].node_idx;
        target_type  = OV_NODE_PROC;
    }
    else if (lay->focus == OV_FOCUS_FPS && lay->sel_fps >= 0 && lay->sel_fps < m->nb_fps)
    {
        current_node = m->fps[lay->sel_fps].node_idx;
        target_type  = OV_NODE_FPS;
    }
    else if (lay->focus == OV_FOCUS_GRAPH)
    {
        // Find selected graph node
        int start_node = ov_input_get_graph_start_node(lay, m);
        if (start_node >= 0)
        {
            SG_RENDER_NODE rnodes[OV_MAX_NODES];
            int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
            if (lay->sel_graph >= 0 && lay->sel_graph < n_rnodes)
            {
                current_node = rnodes[lay->sel_graph].node_idx;
            }
        }
        target_type = -1; // Any type
    }

    if (current_node < 0 || current_node >= m->nb_nodes)
    {
        return 1;
    }

    int target_node =
        find_relative_node_of_type(m, current_node, target_type, key == OV_KEY_SHIFT_UP);

    if (target_node >= 0 && target_node < m->nb_nodes)
    {
        const OV_NODE *tn = &m->nodes[target_node];
        if (lay->focus == OV_FOCUS_GRAPH)
        {
            // Find target_node in graph render list
            int start_node = ov_input_get_graph_start_node(lay, m);
            if (start_node >= 0)
            {
                SG_RENDER_NODE rnodes[OV_MAX_NODES];
                int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                for (int i = 0; i < n_rnodes; i++)
                {
                    if (rnodes[i].node_idx == target_node)
                    {
                        lay->sel_graph = i;
                        break;
                    }
                }
            }
        }
        else if (lay->focus == OV_FOCUS_STREAMS && tn->type == OV_NODE_STREAM)
        {
            lay->sel_stream         = tn->index;
            lay->sel_name_stream[0] = '\0';
        }
        else if (lay->focus == OV_FOCUS_PROCS && tn->type == OV_NODE_PROC)
        {
            lay->sel_proc         = tn->index;
            lay->sel_name_proc[0] = '\0';
        }
        else if (lay->focus == OV_FOCUS_FPS && tn->type == OV_NODE_FPS)
        {
            lay->sel_fps         = tn->index;
            lay->sel_name_fps[0] = '\0';
            lay->fps_param_focus = 0;
        }
    }
    return 1;
}

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

    /* -------------------------------------------------------
     * F5 view (OV_VIEW_FPS) param-tree intercept.
     *
     * When fps_param_focus == 1 (right-side param panel
     * is active), all navigation keys are consumed by the
     * param tree. RIGHT / ENTER from the list side switch
     * focus to the param panel.
     * ------------------------------------------------------- */
    if (lay->view == OV_VIEW_FPS)
    {
        int fsel       = lay->sel_fps;
        int has_params = (fsel >= 0 && fsel < m->nb_fps && m->fps[fsel].nb_disp_params > 0);

        int             nitems = 0;
        fps_tree_item_t items[1024];
        if (has_params)
        {
            nitems = ov_get_fps_tree_items(&m->fps[fsel], lay->fps_param_path, items, 1024);
        }

        /* RIGHT from list → enter param panel */
        if (lay->fps_param_focus == 0 &&
            (key == OV_KEY_RIGHT || key == OV_KEY_ENTER || key == '\r' || key == '\n') &&
            has_params)
        {
            lay->fps_param_focus = 1;
            if (nitems > 0)
            {
                if (lay->fps_param_sel < 0)
                {
                    lay->fps_param_sel = 0;
                }
                else if (lay->fps_param_sel >= nitems)
                {
                    lay->fps_param_sel = nitems - 1;
                }
            }
            return 1;
        }

        /* All nav/edit keys when param panel is focused */
        if (lay->fps_param_focus == 1)
        {
            if (nitems > 0)
            {
                if (lay->fps_param_sel < 0)
                {
                    lay->fps_param_sel = 0;
                }
                else if (lay->fps_param_sel >= nitems)
                {
                    lay->fps_param_sel = nitems - 1;
                }
            }
            /* ESC / LEFT from param panel → back to list or ascend dir */
            if (key == OV_KEY_LEFT || key == OV_KEY_ESC)
            {
                if (lay->fps_param_path[0] == '\0')
                {
                    lay->fps_param_focus = 0;
                }
                else
                {
                    ov_input_save_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);

                    /* Ascend directory */
                    char  exited_dir[100] = { 0 };
                    char *last_dot        = strrchr(lay->fps_param_path, '.');
                    if (last_dot)
                    {
                        strncpy(exited_dir, last_dot + 1, sizeof(exited_dir) - 1);
                        *last_dot = '\0';
                    }
                    else
                    {
                        strncpy(exited_dir, lay->fps_param_path, sizeof(exited_dir) - 1);
                        lay->fps_param_path[0] = '\0';
                    }

                    ov_input_load_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);

                    int found_sel = -1;
                    if (has_params)
                    {
                        fps_tree_item_t parent_items[1024];
                        int             n_parent_items = ov_get_fps_tree_items(
                            &m->fps[fsel], lay->fps_param_path, parent_items, 1024);

                        for (int i = 0; i < n_parent_items; i++)
                        {
                            if (parent_items[i].is_dir &&
                                strcmp(parent_items[i].name, exited_dir) == 0)
                            {
                                found_sel = i;
                                break;
                            }
                        }
                    }
                    if (found_sel != -1)
                    {
                        lay->fps_param_sel = found_sel;
                    }
                }
                return 1;
            }

            int ph = lay->r_fps_params.height - 3;
            if (ph < 1)
            {
                ph = 1;
            }

            if (key == OV_KEY_UP)
            {
                if (lay->fps_param_sel > 0)
                {
                    lay->fps_param_sel--;
                }
                return 1;
            }
            if (key == OV_KEY_DOWN)
            {
                if (lay->fps_param_sel < nitems - 1)
                {
                    lay->fps_param_sel++;
                }
                return 1;
            }
            if (key == OV_KEY_PGUP)
            {
                lay->fps_param_sel -= ph;
                if (lay->fps_param_sel < 0)
                {
                    lay->fps_param_sel = 0;
                }
                return 1;
            }
            if (key == OV_KEY_PGDN)
            {
                lay->fps_param_sel += ph;
                if (lay->fps_param_sel >= nitems)
                {
                    lay->fps_param_sel = nitems - 1;
                }
                return 1;
            }
            if (key == OV_KEY_HOME)
            {
                lay->fps_param_sel = 0;
                return 1;
            }
            if (key == OV_KEY_END)
            {
                lay->fps_param_sel = nitems - 1;
                if (lay->fps_param_sel < 0)
                {
                    lay->fps_param_sel = 0;
                }
                return 1;
            }
            if (key == OV_KEY_RIGHT || key == OV_KEY_ENTER || key == '\r' || key == '\n')
            {
                if (lay->fps_param_sel >= 0 && lay->fps_param_sel < nitems)
                {
                    fps_tree_item_t *item = &items[lay->fps_param_sel];
                    if (item->is_dir)
                    {
                        ov_input_save_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);

                        /* Descend directory */
                        if (lay->fps_param_path[0] == '\0')
                        {
                            strncpy(lay->fps_param_path, item->name,
                                    sizeof(lay->fps_param_path) - 1);
                        }
                        else
                        {
                            char tmp[200];
                            snprintf(tmp, sizeof(tmp), "%s.%s", lay->fps_param_path, item->name);
                            strncpy(lay->fps_param_path, tmp, sizeof(lay->fps_param_path) - 1);
                        }

                        ov_input_load_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);
                    }
                    else if (key == OV_KEY_ENTER || key == '\r' || key == '\n')
                    {
                        if (!lay->ctrl_mode)
                        {
                            ov_cmdlog_push(
                                &lay->cmdlog, OV_CMDLOG_WARN,
                                "Edit requires CONTROL mode (press c to toggle CTRL mode ON/OFF)");
                        }
                        else
                        {
                            ov_fps_inline_edit(lay, m->fps[fsel].name, item->param_idx);
                        }
                    }
                }
                return 1;
            }
            if (key == 'o')
            {
                if (lay->fps_param_sel >= 0 && lay->fps_param_sel < nitems)
                {
                    fps_tree_item_t *item = &items[lay->fps_param_sel];
                    if (!item->is_dir)
                    {
                        int                  pi     = item->param_idx;
                        const OV_FPS_PARAMS *params = ov_fps_get_params(m->fps[fsel].name);
                        if (params != NULL && pi >= 0 && pi < params->nb_disp_params &&
                            params->disp_param_type[pi] == FPTYPE_ONOFF)
                        {
                            if (!lay->ctrl_mode)
                            {
                                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                                               "Toggle requires CONTROL mode (press c to toggle "
                                               "CTRL mode ON/OFF)");
                            }
                            else
                            {
                                char kw[FUNCTION_PARAMETER_STRMAXLEN] = { 0 };
                                int  newval                           = 0;
                                if (ov_fcache_toggle_param(m->fps[fsel].name, pi, kw, sizeof(kw),
                                                           &newval) == 0)
                                {
                                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                                   "Toggled parameter %s to %s", kw,
                                                   newval ? "ON" : "OFF");
                                }
                            }
                        }
                    }
                }
                return 1;
            }
            return 0; /* pass through unmapped keys */
        }
    } /* if OV_VIEW_FPS */

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
                    /* Reset param tree cursor when FPS selection moves */
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

    /* Auto-scroll to keep selection visible happens in the main function or here if handled,
       but wait, if UP/DOWN/etc are not pressed we shouldn't do anything.
       We only return 1 if one of those keys was pressed.
       But the auto-scroll needs to happen if sel changes. We can do it before returning 1. */
    return 0; /* Not a navigation key */
}

