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
                       "Action requires CONTROL mode"
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
    case OV_BTN_PROC_REMOVE:
    {
        if (pi >= 0)
        {
            ov_confirm_open(lay, m, OV_ACT_PROC_REMOVE, 0, OV_FOCUS_PROCS, pi);
        }
        break;
    }
    case OV_BTN_PROC_CLEANUP:
    {
        ov_confirm_open(lay, m, OV_ACT_PROC_CLEANUP, 0, OV_FOCUS_PROCS, -1);
        break;
    }
    case OV_BTN_FPS_ERASE:
    {
        if (fi >= 0)
        {
            ov_confirm_open(lay, m, OV_ACT_FPS_ERASE, 0, OV_FOCUS_FPS, fi);
        }
        break;
    }
    case OV_BTN_STREAM_DEL:
    {
        if (ps != NULL)
        {
            ov_confirm_open(lay, m, OV_ACT_STREAM_DELETE, 0, OV_FOCUS_STREAMS, si);
        }
        break;
    }
    default:
        break;
    }

    ov_scan_force_update();
}

/** need_ctrl - an action key outside CONTROL mode: say how to enable it. */
static int need_ctrl(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    char name[16];
    snprintf(name, sizeof(name),
             (key == ctrl('s'))    ? "CTRL+s"
             : (key == ctrl('e'))  ? "CTRL+e"
             : (key == OV_KEY_DEL) ? "DEL"
                                   : "'%c'",
             key);
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                   "%s requires CONTROL mode (press c to toggle CTRL mode ON/OFF)", name);
    return 1;
}

/** acted - an action ran: rescan now so its effect shows. */
static int acted(void)
{
    ov_scan_force_update();
    return 1;
}

/** erase - ^E: delete the stream / remove the dead process / erase the FPS under the cursor, after y/N. */
static int erase(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    static const ov_act_t act[OV_NB_LIST_PANELS] = { OV_ACT_STREAM_DELETE, OV_ACT_PROC_REMOVE,
                                                     OV_ACT_FPS_ERASE };
    (void) key;
    int idx = (lay->focus < OV_NB_LIST_PANELS) ? ov_sel_idx(lay, m, lay->focus) : -1;
    if (idx >= 0)
    {
        ov_confirm_open(lay, m, act[lay->focus], 0, lay->focus, idx);
    }
    return 1;
}

/** inspect - i: milk-stream-info / milk-procinfo-info / milk-fps-info on the entry under the cursor. */
static int inspect(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key;
    const void *item =
        (lay->focus == OV_FOCUS_STREAMS) ? (const void *) ov_input_get_sel_stream(lay, m)
        : (lay->focus == OV_FOCUS_PROCS) ? (const void *) ov_input_get_sel_proc(lay, m)
        : (lay->focus == OV_FOCUS_FPS)   ? (const void *) ov_input_get_sel_fps(lay, m)
                                         : NULL;
    if (item != NULL)
    {
        ov_ctrl_inspect_item(lay->focus, item);
    }
    return 1;
}

/** proc_cleanup - C: remove all stopped / crashed process entries, after y/N. */
static int proc_cleanup(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key;
    ov_confirm_open(lay, m, OV_ACT_PROC_CLEANUP, 0, OV_FOCUS_PROCS, -1);
    return acted();
}

/** proc_action - k SIGTERM, K SIGKILL (y/N), p pause, s / ^S step, e exit, z zero the counters. */
static int proc_action(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_CMDLOG     *log = &lay->cmdlog;
    const OV_PROC *p   = ov_input_get_sel_proc(lay, m);
    if (p == NULL)
    {
        return acted();
    }
    switch (key)
    {
    case 'k':
        ov_ctrl_proc_kill(p, log);
        break;
    case 'K':
        ov_confirm_open(lay, m, OV_ACT_PROC_SIGKILL, 0, OV_FOCUS_PROCS, (int) (p - m->procs));
        break;
    case 'p':
        ov_ctrl_proc_set_ctrlval(p, -1, log);
        break;
    case 'e':
        ov_ctrl_proc_set_ctrlval(p, 3, log);
        break;
    case 'z':
        ov_ctrl_proc_zero_counters(p, log);
        break;
    default: /* s, ^S */
        ov_ctrl_proc_set_ctrlval(p, 2, log);
        break;
    }
    return acted();
}

/** fps_mark - b: toggle the batch mark of the FPS under the cursor and move down; a: mark every visible row. */
static int fps_mark(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key == 'b')
    {
        const OV_FPS *f = ov_input_get_sel_fps(lay, m);
        if (f != NULL)
        {
            ov_msel_set(lay, f->name, !ov_msel_has(lay, f->name));
            ov_sel_row(lay, m, OV_FOCUS_FPS, lay->panel[OV_FOCUS_FPS].sel + 1);
        }
        return acted();
    }
    int fidx[OV_MAX_FPS];
    int n = ov_rows(lay, m, OV_FOCUS_FPS, fidx, OV_MAX_FPS);
    for (int r = 0; r < n; r++)
    {
        ov_msel_set(lay, m->fps[fidx[r]].name, 1);
    }
    return acted();
}

/** fps_batch - k K r s with FPS marks: the same action on every marked FPS, after y/N. */
static int fps_batch(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_act_t verb = (key == 'r')   ? OV_ACT_BATCH_RUN
                    : (key == 's') ? OV_ACT_BATCH_CONF
                                   : OV_ACT_BATCH_SIGNAL;
    int      arg  = (key == 'k')   ? SIGTERM
                    : (key == 'K') ? SIGKILL
                                   : ov_confirm_batch_arg(lay, m, key == 's');
    ov_confirm_open(lay, m, verb, arg, OV_FOCUS_FPS, -1);
    return acted();
}

/** fps_action - k SIGTERM, K SIGKILL (y/N), r run start / stop, s conf start / stop of the cursor FPS. */
static int fps_action(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_CMDLOG    *log = &lay->cmdlog;
    const OV_FPS *f   = ov_input_get_sel_fps(lay, m);
    if (f == NULL)
    {
        return acted();
    }
    switch (key)
    {
    case 'k':
        ov_ctrl_fps_signal_pid(f, SIGTERM, log);
        break;
    case 'K':
        ov_confirm_open(lay, m, OV_ACT_FPS_SIGKILL, 0, OV_FOCUS_FPS, (int) (f - m->fps));
        break;
    case 'r':
        ov_ctrl_fps_run_toggle(f, log);
        break;
    default: /* s */
        ov_ctrl_fps_conf_toggle(f, log);
        break;
    }
    return acted();
}

/** stream_delete - DEL: delete the stream under the cursor, after y/N. */
static int stream_delete(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key;
    int si = ov_sel_idx(lay, m, OV_FOCUS_STREAMS);
    if (si >= 0)
    {
        ov_confirm_open(lay, m, OV_ACT_STREAM_DELETE, 0, OV_FOCUS_STREAMS, si);
    }
    return acted();
}

#define PROC_KEYS 'k', 'K', 'p', 's', ctrl('s'), 'e', 'z'

/* Only the focused panel acts, and only while it is on screen; outside CONTROL mode, a hint */
static const OV_KEYBIND action_keys[] = {
    { { ctrl('e') }, OV_CTX_FOCUS_SHOWN | OV_CTX_CTRL, 0, erase },
    { { ctrl('e') }, OV_CTX_FOCUS_SHOWN, OV_CTX_CTRL, need_ctrl },
    { { 'i' }, OV_CTX_FOCUS_SHOWN, 0, inspect },
    { { 'C' }, OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_PROCS | OV_CTX_CTRL, 0, proc_cleanup },
    { { PROC_KEYS }, OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_PROCS | OV_CTX_CTRL, 0, proc_action },
    { { 'C', PROC_KEYS }, OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_PROCS, OV_CTX_CTRL, need_ctrl },
    { { 'b', 'a' }, OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_FPS | OV_CTX_CTRL, 0, fps_mark },
    { { 'k', 'K', 'r', 's' },
      OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_FPS | OV_CTX_CTRL | OV_CTX_MSEL,
      0,
      fps_batch },
    { { 'k', 'K', 'r', 's' }, OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_FPS | OV_CTX_CTRL, 0, fps_action },
    { { 'b', 'a', 'k', 'K', 'r', 's' },
      OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_FPS,
      OV_CTX_CTRL,
      need_ctrl },
    { { OV_KEY_DEL }, OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_STREAMS | OV_CTX_CTRL, 0, stream_delete },
    { { OV_KEY_DEL }, OV_CTX_FOCUS_SHOWN | OV_CTX_FOCUS_STREAMS, OV_CTX_CTRL, need_ctrl },
};

#undef PROC_KEYS

/** ov_input__handle_actions - action keys (erase, inspect, CONTROL actions). Return: 1 consumed. */
int ov_input__handle_actions(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_key_dispatch(action_keys, OV_KEYBIND_N(action_keys), key, lay, m);
}
