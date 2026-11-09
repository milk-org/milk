// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_bar.c
 * @brief   Priority layout of one-line bars (header, status line): what fits when the terminal is
 *          narrow.
 *
 * Purpose:     on a narrow terminal the header and the status line drop their least important
 *              items first, or shorten them, instead of being cut at the right edge.
 * Does:        ov_bar_layout() takes the items of a bar (OV_BAR_ITEM: full and short widths,
 *              priority) and, by ascending priority (screen order within one priority), shows
 *              each in its full form, else its short form, else hides it.
 * Used by:     render/overview_render_header.c and overview_render_status.c. Declared in
 *              render/overview_render_internal.h.
 * See also:    render/overview_render_tabs.c (the tab bar's own fitting, ov_tabbar_level);
 *              docs/guide/milk-CTRL.md, "Screen".
 */

#include "overview_render_internal.h"

/**
 * ov_bar_layout - choose the items of a bar that fit in @width columns: by ascending priority (screen order
 * within one priority), each in its full form, else its short form, else hidden. Return: columns used.
 */
int ov_bar_layout(OV_BAR_ITEM *it, int n, int width)
{
    int used = 0, maxp = 0;
    for (int i = 0; i < n; i++)
    {
        it[i].shown = 0;
        maxp        = (it[i].prio > maxp) ? it[i].prio : maxp;
    }
    for (int p = 0; p <= maxp; p++)
    {
        for (int i = 0; i < n; i++)
        {
            if (it[i].prio != p)
            {
                continue;
            }
            if (used + it[i].w <= width)
            {
                it[i].shown = 1;
                used += it[i].w;
            }
            else if (it[i].w_short > 0 && used + it[i].w_short <= width)
            {
                it[i].shown = 2;
                used += it[i].w_short;
            }
        }
    }
    return used;
}
