// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_confirm.c
 * @brief y/N confirmation for destructive and batch verbs; the target is re-resolved before acting.
 */

#include "overview_input_internal.h"
#include <signal.h>

/**
 * ov_confirm_open - ask y/N for @verb on entry @idx of @panel (@idx < 0: no single target).
 */
void ov_confirm_open(OV_LAYOUT      *lay,
                     const OV_MODEL *m,
                     ov_act_t        verb,
                     int             arg,
                     ov_focus_t      panel,
                     int             idx)
{
    lay->confirm.verb  = verb;
    lay->confirm.arg   = arg;
    lay->confirm.panel = panel;
    ov_tgt_set(&lay->confirm.tgt, m, panel, idx);
}

/**
 * ov_confirm_batch_arg - start (1) unless every present marked FPS already runs (@conf: its CONF side).
 */
int ov_confirm_batch_arg(const OV_LAYOUT *lay, const OV_MODEL *m, int conf)
{
    for (int k = 0; k < lay->multi_sel_count; k++)
    {
        int fi = ov_tgt_find(m, OV_FOCUS_FPS, &lay->multi_sel_fps[k]);
        if (fi >= 0 && !(conf ? m->fps[fi].conf_alive : m->fps[fi].run_alive))
        {
            return 1;
        }
    }
    return 0;
}

/** ov_confirm_batch - run the batch verb on every marked FPS, by name, then clear the marks. */
static void ov_confirm_batch(OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_CONFIRM *c    = &lay->confirm;
    OV_CMDLOG        *log  = &lay->cmdlog;
    int               done = 0, skipped = 0, missing = 0;

    for (int k = 0; k < lay->multi_sel_count; k++)
    {
        int fi = ov_tgt_find(m, OV_FOCUS_FPS, &lay->multi_sel_fps[k]);
        if (fi < 0)
        {
            missing++;
            continue;
        }
        const OV_FPS *f = &m->fps[fi];
        if (c->verb == OV_ACT_BATCH_SIGNAL)
        {
            if (!f->run_alive && !f->conf_alive)
            {
                skipped++;
                continue;
            }
            ov_ctrl_fps_signal_pid(f, c->arg, log);
        }
        else
        {
            int conf = (c->verb == OV_ACT_BATCH_CONF);
            if ((conf ? !!f->conf_alive : !!f->run_alive) == c->arg)
            {
                skipped++; /* already in the requested state */
                continue;
            }
            if (conf)
            {
                ov_ctrl_fps_conf_toggle(f, log);
            }
            else
            {
                ov_ctrl_fps_run_toggle(f, log);
            }
        }
        done++;
    }
    ov_cmdlog_push(log, OV_CMDLOG_INFO, "Batch on %d marked FPS: %d acted, %d skipped, %d missing",
                   lay->multi_sel_count, done, skipped, missing);
    ov_msel_clear(lay);
}

/** ov_confirm_exec - act on the confirmed verb if its target still exists (C2 identity checks apply). */
static void ov_confirm_exec(OV_LAYOUT *lay, const OV_MODEL *m)
{
    const OV_CONFIRM *c   = &lay->confirm;
    OV_CMDLOG        *log = &lay->cmdlog;
    int               idx = -1;

    if (c->verb != OV_ACT_PROC_CLEANUP && c->verb < OV_ACT_BATCH_SIGNAL)
    {
        idx = ov_tgt_find(m, (ov_focus_t) c->panel, &c->tgt);
        if (idx < 0)
        {
            ov_cmdlog_push(log, OV_CMDLOG_WARN, "\"%s\" is gone: nothing done", c->tgt.name);
            return;
        }
    }
    switch (c->verb)
    {
    case OV_ACT_PROC_SIGKILL:
        ov_ctrl_proc_sigkill(&m->procs[idx], log);
        break;
    case OV_ACT_PROC_REMOVE:
        ov_ctrl_proc_remove(&m->procs[idx], log);
        break;
    case OV_ACT_PROC_CLEANUP:
        ov_ctrl_procs_cleanup(log);
        break;
    case OV_ACT_FPS_SIGKILL:
        ov_ctrl_fps_signal_pid(&m->fps[idx], SIGKILL, log);
        break;
    case OV_ACT_FPS_ERASE:
        ov_ctrl_fps_remove(&m->fps[idx], log);
        break;
    case OV_ACT_STREAM_DELETE:
        ov_ctrl_stream_delete(&m->streams[idx], log);
        break;
    default:
        ov_confirm_batch(lay, m);
        break;
    }
    ov_scan_force_update();
}

/**
 * ov_input__handle_confirm - while the y/N modal is open, y acts and any other key or click cancels.
 * Return: 1 if the modal consumed @key.
 */
int ov_input__handle_confirm(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay->confirm.verb == OV_ACT_NONE)
    {
        return 0;
    }
    if (key == OV_KEY_MOUSE_MOVE || key == OV_KEY_MOUSE_RELEASE || key == OV_KEY_MOUSE_DRAG ||
        key == OV_KEY_MOUSE_UP || key == OV_KEY_MOUSE_DOWN)
    {
        return 1;
    }
    if (key == 'y' || key == 'Y')
    {
        ov_confirm_exec(lay, m);
    }
    else
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Cancelled");
    }
    lay->confirm.verb = OV_ACT_NONE;
    return 1;
}
