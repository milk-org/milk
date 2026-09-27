// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_toggles.c
 * @brief View switching, option toggles, interval controls, and dashboard focus navigation
 */

#include "overview_input_internal.h"

/**
 * ov_input__handle_view_switch - handle F-key view switches (F2..F7) and tabs.
 * @key: Pressed key code.
 * @lay: Pointer to layout structure.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_view_switch(
    int        key,
    OV_LAYOUT *lay)
{
    if (key >= OV_KEY_F2 && key <= OV_KEY_F7)
    {
        int vi = key - OV_KEY_F2;
        if (vi < OV_VIEW_COUNT)
        {
            lay->view = (ov_view_t) vi;
            if (vi == OV_VIEW_STREAMS)
            {
                lay->focus = OV_FOCUS_STREAMS;
            }
            else if (vi == OV_VIEW_PROCS)
            {
                lay->focus = OV_FOCUS_PROCS;
            }
            else if (vi == OV_VIEW_FPS)
            {
                lay->focus = OV_FOCUS_FPS;
            }
            else if (vi == OV_VIEW_GRAPH || vi == OV_VIEW_LOOPS)
            {
                lay->focus = OV_FOCUS_GRAPH;
            }
        }
        return 1;
    }

    if (key == OV_KEY_CTRL_LEFT)
    {
        int v     = (int) lay->view;
        v         = (v - 1 + OV_VIEW_COUNT) % OV_VIEW_COUNT;
        lay->view = (ov_view_t) v;
        if (v == OV_VIEW_STREAMS)
        {
            lay->focus = OV_FOCUS_STREAMS;
        }
        else if (v == OV_VIEW_PROCS)
        {
            lay->focus = OV_FOCUS_PROCS;
        }
        else if (v == OV_VIEW_FPS)
        {
            lay->focus = OV_FOCUS_FPS;
        }
        else if (v == OV_VIEW_GRAPH || v == OV_VIEW_LOOPS)
        {
            lay->focus = OV_FOCUS_GRAPH;
        }
        return 1;
    }
    if (key == OV_KEY_CTRL_RIGHT)
    {
        int v     = (int) lay->view;
        v         = (v + 1) % OV_VIEW_COUNT;
        lay->view = (ov_view_t) v;
        if (v == OV_VIEW_STREAMS)
        {
            lay->focus = OV_FOCUS_STREAMS;
        }
        else if (v == OV_VIEW_PROCS)
        {
            lay->focus = OV_FOCUS_PROCS;
        }
        else if (v == OV_VIEW_FPS)
        {
            lay->focus = OV_FOCUS_FPS;
        }
        else if (v == OV_VIEW_GRAPH || v == OV_VIEW_LOOPS)
        {
            lay->focus = OV_FOCUS_GRAPH;
        }
        return 1;
    }

    if (key == OV_KEY_TAB)
    {
        lay->focus = (ov_focus_t) (((int) lay->focus + 1) % OV_FOCUS_COUNT);
        return 1;
    }

    if (key == OV_KEY_BTAB)
    {
        lay->graph_tab_mode = (lay->graph_tab_mode + 1) % 4;
        return 1;
    }

    return 0;
}

/**
 * ov_input__handle_misc_toggles - handle miscellaneous toggle shortcuts
 * (freeze, help, theme, etc.).
 * @key: Pressed key code.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_misc_toggles(
    int             key,
    OV_LAYOUT      *lay,
    const OV_MODEL *m)
{
    if (key == '+' || key == '=')
    {
        float cur = ov_scan_get_interval();
        ov_scan_set_interval(cur * 0.7f);
        return 1;
    }
    if (key == '-')
    {
        float cur = ov_scan_get_interval();
        ov_scan_set_interval(cur * 1.4f);
        return 1;
    }
    if (key == 'D')
    {
        lay->graph_tab_mode = (lay->graph_tab_mode == 2) ? 0 : 2;
        return 1;
    }
    /* Compact column mode toggle (#13) */
    if (key == 'd')
    {
        lay->compact_mode = !lay->compact_mode;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                       lay->compact_mode ? "Compact mode ON" : "Compact mode OFF");
        return 1;
    }
    if (key == 'L')
    {
        lay->lineage_mode = (lay->lineage_mode + 1) % 3;
        return 1;
    }
    if (key == 'v' || key == 'V')
    {
        int prev_rows = lay->cmdlog_rows;
        int step      = (key == 'v') ? -1 : 1;
        if (lay->cmdlog_rows == 0 && step > 0)
        {
            lay->cmdlog_rows = 4;
        }
        else
        {
            lay->cmdlog_rows += step;
        }
        if (lay->cmdlog_rows < 0)
        {
            lay->cmdlog_rows = 0;
        }
        if (lay->cmdlog_rows > lay->term_rows / 2)
        {
            lay->cmdlog_rows = lay->term_rows / 2;
        }
        if (lay->cmdlog_rows != prev_rows)
        {
            ov_buf_force_clear();
        }
        return 1;
    }
    if (key == '{' || key == '}')
    {
        float step = (key == '{') ? -0.05f : 0.05f;
        if (lay->view == OV_VIEW_FPS)
        {
            lay->fps_split_ratio += step;
            if (lay->fps_split_ratio < 0.1f)
            {
                lay->fps_split_ratio = 0.1f;
            }
            if (lay->fps_split_ratio > 0.9f)
            {
                lay->fps_split_ratio = 0.9f;
            }
            return 1;
        }
        else if (lay->view == OV_VIEW_DASHBOARD)
        {
            lay->dash_split_v_ratio += step;
            if (lay->dash_split_v_ratio < 0.1f)
            {
                lay->dash_split_v_ratio = 0.1f;
            }
            if (lay->dash_split_v_ratio > 0.9f)
            {
                lay->dash_split_v_ratio = 0.9f;
            }
            return 1;
        }
    }
    if (key == '(' || key == ')')
    {
        float step = (key == '(') ? -0.05f : 0.05f;
        if (lay->view == OV_VIEW_DASHBOARD)
        {
            lay->dash_split_h_ratio += step;
            if (lay->dash_split_h_ratio < 0.1f)
            {
                lay->dash_split_h_ratio = 0.1f;
            }
            if (lay->dash_split_h_ratio > 0.9f)
            {
                lay->dash_split_h_ratio = 0.9f;
            }
            return 1;
        }
    }

    if (key == 'F')
    {
        lay->paused = !lay->paused;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "%s Display %s",
                       lay->paused ? "⏸️" : "▶️",
                       lay->paused ? "paused" : "resumed");
        return 1;
    }
    if (key == 'W')
    {
        ov_model_export_snapshot(m);
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_OK, "📸 Snapshot exported");
        return 1;
    }
    if (key == OV_KEY_F8 || key == ctrl('t'))
    {
        int count = ov_theme_count();
        int next  = (ov_theme_get_active_index() + 1) % count;
        ov_theme_set(next);
        lay->theme_popup_sel    = next;
        lay->theme_popup_active = 1;
        clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup_ts);
        const ov_theme_t *th = ov_theme_get_active();
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "🎨 Theme: %s (%s)",
                       th->name, th->desc);
        return 1;
    }

    /* Loop filter toggle on Enter when on Loops */
    if ((key == '\n' || key == '\r' || key == 10 || key == 13) &&
        (lay->view == OV_VIEW_LOOPS ||
         (lay->focus == OV_FOCUS_GRAPH && lay->graph_tab_mode == 1)))
    {
        lay->loop_filter_active = !lay->loop_filter_active;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                       lay->loop_filter_active ? "Loop isolation filter: ON"
                                               : "Loop isolation filter: OFF");
        return 1;
    }

    /* Graph jump on Enter (must be before detail mode toggle) */
    if ((key == '\n' || key == '\r') &&
        lay->focus == OV_FOCUS_GRAPH && lay->graph_tab_mode == 0)
    {
        int start_node = ov_input_get_graph_start_node(lay, m);
        if (start_node >= 0)
        {
            SG_RENDER_NODE rnodes[OV_MAX_NODES];
            int n_rnodes = sg_compute_render_nodes(m, start_node,
                                                   lay->lineage_mode, rnodes);
            if (lay->sel_graph < n_rnodes)
            {
                const SG_RENDER_NODE *rn   = &rnodes[lay->sel_graph];
                const OV_NODE        *node = &m->nodes[rn->node_idx];
                if (node->type == OV_NODE_STREAM)
                {
                    lay->focus      = OV_FOCUS_STREAMS;
                    lay->sel_stream = node->index;
                }
                else if (node->type == OV_NODE_PROC)
                {
                    lay->focus    = OV_FOCUS_PROCS;
                    lay->sel_proc = node->index;
                }
                else if (node->type == OV_NODE_FPS)
                {
                    lay->focus   = OV_FOCUS_FPS;
                    lay->sel_fps = node->index;
                }
                lay->view   = OV_VIEW_DASHBOARD;
                lay->freeze = 0;
            }
        }
        return 1;
    }

    /* ENTER — open DETAILS for list panels; toggle when on graph */
    if ((key == 10 || key == 13) && m != NULL)
    {
        /* In dedicated FPS view, ENTER is used to edit parameters */
        if (lay->view == OV_VIEW_FPS)
        {
            return 0;
        }

        if (lay->focus == OV_FOCUS_STREAMS || lay->focus == OV_FOCUS_PROCS ||
            lay->focus == OV_FOCUS_FPS)
        {
            /* Always jump to DETAILS sub-tab */
            lay->graph_tab_mode = 2;
        }
        else
        {
            lay->graph_tab_mode = (lay->graph_tab_mode == 2) ? 0 : 2;
        }
        return 1;
    }

    if (key == ' ')
    {
        if (lay->freeze)
        {
            lay->freeze = 0;
        }
        else
        {
            lay->freeze            = 1;
            lay->freeze_focus      = lay->focus;
            lay->freeze_sel_stream = lay->sel_stream;
            lay->freeze_sel_proc   = lay->sel_proc;
            lay->freeze_sel_fps    = lay->sel_fps;
        }
        return 1;
    }

    if (key == OV_KEY_LEFT || key == OV_KEY_RIGHT)
    {
        if (lay->view != OV_VIEW_DASHBOARD)
        {
            if (lay->view == OV_VIEW_FPS)
            {
                return 0;
            }
            return 1;
        }

        if (key == OV_KEY_LEFT)
        {
            if (lay->focus == OV_FOCUS_STREAMS)
            {
                lay->focus = OV_FOCUS_GRAPH;
            }
            else if (lay->focus == OV_FOCUS_PROCS)
            {
                lay->focus = OV_FOCUS_STREAMS;
            }
            else if (lay->focus == OV_FOCUS_FPS)
            {
                lay->focus = OV_FOCUS_PROCS;
            }
            else if (lay->focus == OV_FOCUS_GRAPH)
            {
                lay->focus = OV_FOCUS_FPS;
            }
        }
        else
        {
            if (lay->focus == OV_FOCUS_STREAMS)
            {
                lay->focus = OV_FOCUS_PROCS;
            }
            else if (lay->focus == OV_FOCUS_PROCS)
            {
                lay->focus = OV_FOCUS_FPS;
            }
            else if (lay->focus == OV_FOCUS_FPS)
            {
                lay->focus = OV_FOCUS_GRAPH;
            }
            else if (lay->focus == OV_FOCUS_GRAPH)
            {
                lay->focus = OV_FOCUS_STREAMS;
            }
        }
        return 1;
    }

    return 0;
}
