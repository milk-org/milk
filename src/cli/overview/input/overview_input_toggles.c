// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_toggles.c
 * @brief View switching, option toggles, interval controls, and dashboard focus navigation
 */

#include "overview_input_internal.h"
#include <limits.h>
#include <errno.h>

/**
 * ov_input__handle_view_switch - handle F-key view switches (F2..F7) and tabs.
 * @key: Pressed key code.
 * @lay: Pointer to layout structure.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_view_switch(int key, OV_LAYOUT *lay)
{
    if (key >= OV_KEY_F1 && key <= OV_KEY_F8 && ov_view_by_fkey(key - OV_KEY_F1 + 1) >= 0)
    {
        ov_set_view(lay, (ov_view_t) ov_view_by_fkey(key - OV_KEY_F1 + 1));
        return 1;
    }
    if (key == OV_KEY_CTRL_LEFT || key == OV_KEY_CTRL_RIGHT)
    {
        ov_set_view(lay, (ov_view_t) ov_view_cycle(lay->view, (key == OV_KEY_CTRL_LEFT) ? -1 : 1));
        return 1;
    }

    if (key == OV_KEY_TAB)
    {
        /* Only panels on screen take focus: DASH cycles its four, F5 its list and tree */
        if (lay->view == OV_VIEW_DASHBOARD)
        {
            lay->focus = (ov_focus_t) (((int) lay->focus + 1) % OV_FOCUS_COUNT);
        }
        else if (lay->view == OV_VIEW_FPS)
        {
            lay->fps_param_focus = !lay->fps_param_focus;
        }
        return 1;
    }

    if (key == OV_KEY_BTAB)
    {
        lay->subtab = (ov_subtab_t) ((lay->subtab + 1) % OV_SUBTAB_COUNT);
        return 1;
    }

    return 0;
}

/**
 * ov_view_details_hint - in a view without the graph panel, say where DETAILS lives.
 * Return: 1 when the hint was given (key consumed, nothing toggled).
 */
static int ov_view_details_hint(OV_LAYOUT *lay)
{
    if (lay->view == OV_VIEW_DASHBOARD || lay->view == OV_VIEW_GRAPH)
    {
        return 0;
    }
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "DETAILS is shown in F2:DASH and F6:GRAPH");
    return 1;
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
int ov_input__handle_misc_toggles(int key, OV_LAYOUT *lay, const OV_MODEL *m)
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
        if (ov_view_details_hint(lay))
        {
            return 1;
        }
        lay->subtab = (lay->subtab == OV_SUB_DETAILS) ? OV_SUB_CONN : OV_SUB_DETAILS;
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
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Display %s",
                       lay->paused ? "paused" : "resumed");
        return 1;
    }
    if (key == 'W')
    {
        char path[PATH_MAX];
        if (ov_model_export_snapshot(m, path, sizeof(path)) == 0)
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_OK, "Snapshot written to %s", path);
        }
        else
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_FAIL, "Snapshot failed: %s", strerror(errno));
        }
        return 1;
    }
    if (key == OV_KEY_F8 || key == ctrl('t'))
    {
        int count = ov_theme_count();
        int next  = (ov_theme_get_active_index() + 1) % count;
        ov_theme_set(next);
        lay->theme_popup.sel    = next;
        lay->theme_popup.active = 1;
        clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup.ts);
        const ov_theme_t *th = ov_theme_get_active();
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Theme: %s (%s)", th->name, th->desc);
        return 1;
    }

    /* Loop filter toggle on Enter when on Loops */
    if ((key == '\n' || key == '\r' || key == 10 || key == 13) &&
        (lay->view == OV_VIEW_LOOPS ||
         (lay->focus == OV_FOCUS_GRAPH && lay->subtab == OV_SUB_LOOPS)))
    {
        lay->loop_filter_active = !lay->loop_filter_active;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                       lay->loop_filter_active ? "Loop isolation filter: ON"
                                               : "Loop isolation filter: OFF");
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
        if (ov_view_details_hint(lay))
        {
            return 1;
        }

        if (lay->focus == OV_FOCUS_STREAMS || lay->focus == OV_FOCUS_PROCS ||
            lay->focus == OV_FOCUS_FPS)
        {
            /* Always jump to DETAILS sub-tab */
            lay->subtab = OV_SUB_DETAILS;
        }
        else
        {
            lay->subtab = (lay->subtab == OV_SUB_DETAILS) ? OV_SUB_CONN : OV_SUB_DETAILS;
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
            lay->freeze                             = 1;
            lay->freeze_focus                       = lay->focus;
            lay->panel[OV_FOCUS_STREAMS].freeze_row = lay->panel[OV_FOCUS_STREAMS].sel;
            lay->panel[OV_FOCUS_PROCS].freeze_row   = lay->panel[OV_FOCUS_PROCS].sel;
            lay->panel[OV_FOCUS_FPS].freeze_row     = lay->panel[OV_FOCUS_FPS].sel;
            for (int p = 0; p < OV_NB_LIST_PANELS; p++)
            {
                lay->panel[p].freeze_tgt = lay->panel[p].tgt;
            }
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
