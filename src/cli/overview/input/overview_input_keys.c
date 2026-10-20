// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_keys.c
 * @brief Key binding context and first-match dispatch.
 */

#include "overview_input_internal.h"

/** ov_key_ctx - the OV_CTX_* bits of the current state. */
unsigned ov_key_ctx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    static const unsigned focus_bit[OV_FOCUS_COUNT] = { OV_CTX_FOCUS_STREAMS, OV_CTX_FOCUS_PROCS,
                                                        OV_CTX_FOCUS_FPS, OV_CTX_FOCUS_GRAPH };
    unsigned              c                         = 0;
    c |= (m != NULL) ? OV_CTX_MODEL : 0;
    c |= lay->ctrl_mode ? OV_CTX_CTRL : 0;
    c |= lay->freeze ? OV_CTX_FROZEN : 0;
    c |= (lay->view == OV_VIEW_DASHBOARD) ? OV_CTX_DASH : 0;
    c |= (lay->view == OV_VIEW_FPS) ? OV_CTX_F5 : 0;
    c |= (lay->view == OV_VIEW_LOOPS ||
          (lay->focus == OV_FOCUS_GRAPH && lay->subtab == OV_SUB_LOOPS))
             ? OV_CTX_LOOP_LIST
             : 0;
    c |= ((int) lay->focus >= 0 && lay->focus < OV_FOCUS_COUNT) ? focus_bit[lay->focus] : 0;
    c |= (lay->view == OV_VIEW_FPS && lay->fps_param_focus) ? OV_CTX_PARAM_TREE : 0;
    c |= (lay->multi_sel_count > 0) ? OV_CTX_MSEL : 0;
    c |= ov_view_has_panel(lay->view, lay->focus) ? OV_CTX_FOCUS_SHOWN : 0;
    c |= (lay->focus < OV_NB_LIST_PANELS) ? OV_CTX_FOCUS_LIST : 0;
    return c;
}

/**
 * ov_key_dispatch - run the first row of @t that binds @key in the current context; when its handler
 * declines (returns 0), try the following rows. Return: the handler's value, 0 if none consumed it.
 */
int ov_key_dispatch(const OV_KEYBIND *t, int n, int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    unsigned ctx = ov_key_ctx(lay, m);
    for (int i = 0; i < n; i++)
    {
        int hit = 0;
        for (int k = 0; k < 8 && t[i].keys[k] != 0 && !hit; k++)
        {
            hit = (t[i].keys[k] == key || t[i].keys[k] == OV_KEY_ANY);
        }
        if (hit && (ctx & t[i].need) == t[i].need && (ctx & t[i].forbid) == 0)
        {
            int r = t[i].fn(key, lay, m);
            if (r != 0)
            {
                return r;
            }
        }
    }
    return 0;
}
