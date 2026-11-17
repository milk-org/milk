// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_nav_insp.c
 * @brief Keys of the F3 stream inspector: enter and leave it, scroll its page, change page.
 *
 * Purpose:     in F3 the navigation keys move either the stream list or the inspector beside it;
 *              this decides which, and handles the inspector's own keys.
 * Does:        ov_input_nav_insp(): Right or ENTER on the list enter the inspector; there Up/Down,
 *              PgUp/PgDn and Home/End scroll the page, Right shows the next page, Left or ESC go
 *              back to the list. ov_insp_tab_step() changes page and starts it from its top.
 * Used by:     input/overview_input_nav.c (the first row of its bindings, in F3) and
 *              input/overview_input_toggles.c (Shift+TAB: next page). Declared in
 *              render/overview_render_internal.h.
 * See also:    render/overview_render_insp.c (draws the inspector and clamps the scroll);
 *              docs/guide/milk-CTRL.md, "Streams view (F3)".
 */

#include "overview_input_internal.h"

/** ov_insp_tab_step - show inspector page @step away (wrapping), from its top. */
void ov_insp_tab_step(OV_LAYOUT *lay, int step)
{
    lay->strm_insp_tab    = (lay->strm_insp_tab + step + OV_INSP_TAB_COUNT) % OV_INSP_TAB_COUNT;
    lay->strm_insp_scroll = 0;
}

/**
 * ov_input_nav_insp - F3: RIGHT/ENTER on the list enter the inspector; there UP/DOWN/PgUp/PgDn/Home/End scroll,
 * RIGHT shows the next page, LEFT/ESC go back to the list. Return: 1 consumed, 0 for the list's keys.
 */
int ov_input_nav_insp(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    int page = lay->r_strm_insp.height - 3;
    if (lay->r_strm_insp.width <= 0)
    {
        lay->strm_insp_focus = 0;
        return 0;
    }
    if (!lay->strm_insp_focus)
    {
        if (key == OV_KEY_RIGHT || key == OV_KEY_ENTER || key == '\r' || key == '\n')
        {
            lay->strm_insp_focus = 1;
            return 1;
        }
        return 0;
    }
    switch (key)
    {
    case OV_KEY_LEFT:
    case OV_KEY_ESC:
        lay->strm_insp_focus = 0;
        return 1;
    case OV_KEY_RIGHT:
        ov_insp_tab_step(lay, 1);
        return 1;
    case OV_KEY_UP:
    case OV_KEY_DOWN:
        lay->strm_insp_scroll += (key == OV_KEY_UP) ? -1 : 1; /* clamped when drawn */
        return 1;
    case OV_KEY_PGUP:
    case OV_KEY_PGDN:
        lay->strm_insp_scroll += (key == OV_KEY_PGUP) ? -page : page;
        return 1;
    case OV_KEY_HOME:
    case OV_KEY_END:
        lay->strm_insp_scroll = (key == OV_KEY_HOME) ? 0 : 1 << 20;
        return 1;
    default:
        return 0;
    }
}
