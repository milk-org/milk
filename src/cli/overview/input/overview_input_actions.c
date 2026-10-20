// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_actions.c
 * @brief Control actions (kill, pause, step, stream delete, preview bar actions).
 */

#include "overview_input_internal.h"

/**
 * ov_input_get_sel_stream - the stream under the STREAMS cursor (by identity), or NULL.
 */
const OV_STREAM *ov_input_get_sel_stream(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int i = ov_sel_idx(lay, m, OV_FOCUS_STREAMS);
    return (i >= 0) ? &m->streams[i] : NULL;
}

/**
 * ov_input_get_sel_proc - the process under the PROCS cursor (by identity), or NULL.
 */
const OV_PROC *ov_input_get_sel_proc(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int i = ov_sel_idx(lay, m, OV_FOCUS_PROCS);
    return (i >= 0) ? &m->procs[i] : NULL;
}

/**
 * ov_input_get_sel_fps - the FPS under the FPS cursor (by identity), or NULL.
 */
const OV_FPS *ov_input_get_sel_fps(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int i = ov_sel_idx(lay, m, OV_FOCUS_FPS);
    return (i >= 0) ? &m->fps[i] : NULL;
}

/**
 * ov_input__exec_preview_btn - execute a preview bar action button.
 * @btn_id: ID of the button pressed (OV_BTN_*).
 * @lay:    Pointer to layout structure.
 * @m:      Pointer to data model snapshot.
 */
void ov_input__exec_preview_btn(int btn_id, OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_CMDLOG *log = &lay->cmdlog;

    /* Buttons act on the entry the preview line shows: the frozen one when frozen */
    ov_focus_t       pfocus = lay->freeze ? lay->freeze_focus : lay->focus;
    int              si     = ov_get_selected_stream_idx(lay, m);
    int              pi     = ov_get_selected_proc_idx(lay, m);
    int              fi     = ov_get_selected_fps_idx(lay, m);
    const OV_STREAM *ps     = (si >= 0) ? &m->streams[si] : NULL;
    const OV_PROC   *pp     = (pi >= 0) ? &m->procs[pi] : NULL;
    const OV_FPS    *pf     = (fi >= 0) ? &m->fps[fi] : NULL;

    /* Inspect works without CONTROL mode */
    if (btn_id == OV_BTN_INSPECT)
    {
        const void *item = (pfocus == OV_FOCUS_STREAMS) ? (const void *) ps
                           : (pfocus == OV_FOCUS_PROCS) ? (const void *) pp
                           : (pfocus == OV_FOCUS_FPS)   ? (const void *) pf
                                                        : NULL;
        if (item != NULL)
        {
            ov_ctrl_inspect_item(pfocus, item);
        }
        return;
    }

    if (!lay->ctrl_mode)
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                       "\xf0\x9f\x9a\xab Action requires CONTROL mode"
                       " (press c to toggle CTRL mode ON/OFF)");
        return;
    }

    switch (btn_id)
    {
    case OV_BTN_PROC_PAUSE:
    {
        const OV_PROC *p = pp;
        if (p)
        {
            ov_ctrl_proc_set_ctrlval(p, -1, log);
        }
        break;
    }
    case OV_BTN_PROC_EXIT:
    {
        const OV_PROC *p = pp;
        if (p)
        {
            ov_ctrl_proc_set_ctrlval(p, 3, log);
        }
        break;
    }
    case OV_BTN_PROC_KILL:
    {
        const OV_PROC *p = pp;
        if (p)
        {
            ov_ctrl_proc_kill(p, log);
        }
        break;
    }
    case OV_BTN_PROC_STEP:
    {
        const OV_PROC *p = pp;
        if (p)
        {
            ov_ctrl_proc_set_ctrlval(p, 2, log);
        }
        break;
    }
    case OV_BTN_FPS_CONF:
    {
        const OV_FPS *f = pf;
        if (f)
        {
            ov_ctrl_fps_conf_toggle(f, log);
        }
        break;
    }
    case OV_BTN_FPS_RUN:
    {
        const OV_FPS *f = pf;
        if (f)
        {
            ov_ctrl_fps_run_toggle(f, log);
        }
        break;
    }
    case OV_BTN_FPS_KILL:
    {
        const OV_FPS *f = pf;
        if (f)
        {
            ov_ctrl_fps_signal_pid(f, SIGTERM, log);
        }
        break;
    }
    case OV_BTN_STREAM_DEL:
    {
        const OV_STREAM *s = ps;
        if (s)
        {
            ov_ctrl_stream_delete(s, log);
        }
        break;
    }
    default:
        break;
    }

    ov_scan_force_update();
}


/**
 * ov_input__handle_actions - dispatch keyboard action keys (kill, pause, step, delete).
 * @key: Pressed key code.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if key was an action and consumed, 0 otherwise.
 */
int ov_input__handle_actions(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_CMDLOG *log = &lay->cmdlog;

    if (!ov_view_has_panel(lay->view, lay->focus))
    {
        return 0; /* never act on a panel that is not on screen */
    }

    if (key == ctrl('e'))
    {
        if (!lay->ctrl_mode)
        {
            ov_cmdlog_push(log, OV_CMDLOG_WARN,
                           "🚫 CTRL+e requires CONTROL mode (press c to toggle CTRL mode ON/OFF)");
            return 1;
        }

        if (lay->focus == OV_FOCUS_FPS)
        {
            const OV_FPS *f = ov_input_get_sel_fps(lay, m);
            if (f)
            {
                ov_ctrl_fps_remove(f, log);
            }
        }
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            const OV_PROC *p = ov_input_get_sel_proc(lay, m);
            if (p)
            {
                ov_ctrl_proc_remove(p, log);
            }
        }
        else if (lay->focus == OV_FOCUS_STREAMS)
        {
            const OV_STREAM *s = ov_input_get_sel_stream(lay, m);
            if (s)
            {
                ov_ctrl_stream_delete(s, log);
            }
        }
        ov_scan_force_update();
        return 1;
    }

    if (key == 'i')
    {
        if (lay->focus == OV_FOCUS_STREAMS)
        {
            const OV_STREAM *s = ov_input_get_sel_stream(lay, m);
            if (s)
            {
                ov_ctrl_inspect_item(OV_FOCUS_STREAMS, s);
            }
        }
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            const OV_PROC *p = ov_input_get_sel_proc(lay, m);
            if (p)
            {
                ov_ctrl_inspect_item(OV_FOCUS_PROCS, p);
            }
        }
        else if (lay->focus == OV_FOCUS_FPS)
        {
            const OV_FPS *f = ov_input_get_sel_fps(lay, m);
            if (f)
            {
                ov_ctrl_inspect_item(OV_FOCUS_FPS, f);
            }
        }
        return 1;
    }

    int is_ctrl_action = 0;

    if (lay->focus == OV_FOCUS_PROCS)
    {
        if (key == 'C')
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                ov_ctrl_procs_cleanup(log);
            }
        }
        else if (key == 'k' || key == 'K' || key == 'p' || key == 's' || key == ctrl('s') ||
                 key == 'e' || key == 'z')
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                const OV_PROC *p = ov_input_get_sel_proc(lay, m);
                if (p)
                {
                    if (key == 'k')
                    {
                        ov_ctrl_proc_kill(p, log);
                    }
                    else if (key == 'K')
                    {
                        ov_ctrl_proc_sigkill(p, log);
                    }
                    else if (key == 'p')
                    {
                        ov_ctrl_proc_set_ctrlval(p, -1, log);
                    }
                    else if (key == 's' || key == ctrl('s'))
                    {
                        ov_ctrl_proc_set_ctrlval(p, 2, log);
                    }
                    else if (key == 'e')
                    {
                        ov_ctrl_proc_set_ctrlval(p, 3, log);
                    }
                    else if (key == 'z')
                    {
                        ov_ctrl_proc_zero_counters(p, log);
                    }
                }
            }
        }
    }
    else if (lay->focus == OV_FOCUS_FPS)
    {
        /* Space: toggle multi-select (#8) */
        if (key == ' ' && lay->ctrl_mode)
        {
            int           fi = lay->sel_fps;
            const OV_FPS *f  = ov_input_get_sel_fps(lay, m);
            if (f)
            {
                int idx = (int) (f - m->fps);
                if (idx >= 0 && idx < 200)
                {
                    lay->multi_sel_fps[idx] ^= 1;
                    lay->multi_sel_count += lay->multi_sel_fps[idx] ? 1 : -1;
                }
            }
            /* Advance cursor */
            int cnt = ov_input_get_filtered_count(OV_FOCUS_FPS, lay, m);
            if (fi + 1 < cnt)
            {
                lay->sel_fps++;
            }
            return 0;
        }
        /* 'a': select/deselect all filtered (#8) */
        if (key == 'a' && lay->ctrl_mode)
        {
            /* If any selected, deselect all;
             * otherwise select all filtered */
            if (lay->multi_sel_count > 0)
            {
                memset(lay->multi_sel_fps, 0, sizeof(lay->multi_sel_fps));
                lay->multi_sel_count = 0;
            }
            else
            {
                /* Would need filtered indices here;
                 * for simplicity, select all FPS */
                for (int j = 0; j < m->nb_fps; j++)
                {
                    lay->multi_sel_fps[j] = 1;
                }
                lay->multi_sel_count = m->nb_fps;
            }
            return 0;
        }
        if (key == 'k' || key == 'K' || key == 'r' || key == 's')
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                /* Batch dispatch over multi-selected
                 * or single selected (#8) */
                if (lay->multi_sel_count > 0)
                {
                    for (int j = 0; j < m->nb_fps; j++)
                    {
                        if (!lay->multi_sel_fps[j])
                        {
                            continue;
                        }
                        const OV_FPS *f = &m->fps[j];
                        if (key == 'k')
                        {
                            ov_ctrl_fps_signal_pid(f, SIGTERM, log);
                        }
                        else if (key == 'K')
                        {
                            ov_ctrl_fps_signal_pid(f, SIGKILL, log);
                        }
                        else if (key == 'r')
                        {
                            ov_ctrl_fps_run_toggle(f, log);
                        }
                        else if (key == 's')
                        {
                            ov_ctrl_fps_conf_toggle(f, log);
                        }
                    }
                }
                else
                {
                    const OV_FPS *f = ov_input_get_sel_fps(lay, m);
                    if (f)
                    {
                        if (key == 'k')
                        {
                            ov_ctrl_fps_signal_pid(f, SIGTERM, log);
                        }
                        else if (key == 'K')
                        {
                            ov_ctrl_fps_signal_pid(f, SIGKILL, log);
                        }
                        else if (key == 'r')
                        {
                            ov_ctrl_fps_run_toggle(f, log);
                        }
                        else if (key == 's')
                        {
                            ov_ctrl_fps_conf_toggle(f, log);
                        }
                    }
                }
            }
        }
    }
    else if (lay->focus == OV_FOCUS_STREAMS)
    {
        if (key == OV_KEY_DEL)
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                const OV_STREAM *s = ov_input_get_sel_stream(lay, m);
                if (s)
                {
                    ov_ctrl_stream_delete(s, log);
                }
            }
        }
    }

    if (is_ctrl_action)
    {
        if (!lay->ctrl_mode)
        {
            char keyname[16];
            if (key == ctrl('s'))
            {
                snprintf(keyname, sizeof(keyname), "CTRL+s");
            }
            else if (key == OV_KEY_DEL)
            {
                snprintf(keyname, sizeof(keyname), "DEL");
            }
            else
            {
                snprintf(keyname, sizeof(keyname), "'%c'", key);
            }

            ov_cmdlog_push(log, OV_CMDLOG_WARN,
                           "🚫 %s requires CONTROL mode (press c to toggle CTRL mode ON/OFF)",
                           keyname);
        }
        else
        {
            ov_scan_force_update();
        }
        return 1;
    }

    return 0;
}
