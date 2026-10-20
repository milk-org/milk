// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_subtabs.c
 * @brief   Sub-tabs of the bottom-right panel (DASH) and of F6: tab bar, hit rectangles, body dispatch.
 */

#include "overview_render_internal.h"
#include "overview_render_loops.h"

/** render_res - RESOURCES of the DETAILS entry. */
static int render_res(OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_render_resources_panel(lay, m);
}

/** render_loops - LOOPS list and the selected loop. */
static int render_loops(OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_render_loops_panel(lay, m);
    return 1;
}

/** render_conn - CONNECTIONS tree. */
static int render_conn(OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_render_graph_panel(lay, m);
    return 1;
}

/* Indexed by ov_subtab_t */
static const struct
{
    const char *label;                                /* tab text */
    const char *status;                               /* status-line tag */
    int (*render)(OV_LAYOUT *lay, const OV_MODEL *m); /* 0: nothing to show */
} subtabs[OV_SUBTAB_COUNT] = {
    [OV_SUB_CONN]    = { "CONNECTIONS", "CONN", render_conn },
    [OV_SUB_LOOPS]   = { "LOOPS", "LOOPS", render_loops },
    [OV_SUB_DETAILS] = { "DETAILS", "DETAIL", ov_render_detail_panel },
    [OV_SUB_RES]     = { "RESOURCES", "RES", render_res },
};

/* Sub-tab labels as last drawn: 1-based screen column, width */
static struct
{
    int col;
    int w;
} s_hit[OV_SUBTAB_COUNT];

/** ov_subtab_bar - frame the bottom-right panel in @color with the sub-tab labels; record where they are. */
void ov_subtab_bar(const OV_LAYOUT *lay, ov_rgb_t color)
{
    OV_RECT     r = lay->r_graph;
    const char *labels[OV_SUBTAB_COUNT];
    int         col = r.col + 2; /* as ov_draw_panel_tabs lays out " LABEL " + one gap */
    for (int t = 0; t < OV_SUBTAB_COUNT; t++)
    {
        labels[t]    = subtabs[t].label;
        s_hit[t].col = col;
        s_hit[t].w   = (int) strlen(labels[t]) + 2;
        col += s_hit[t].w + 1;
    }
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, labels, OV_SUBTAB_COUNT, lay->subtab, color,
                       lay->focus == OV_FOCUS_GRAPH);
}

/** ov_subtab_hit - sub-tab under screen column @mc of the panel's top border, or -1. */
int ov_subtab_hit(int mc)
{
    for (int t = 0; t < OV_SUBTAB_COUNT; t++)
    {
        if (s_hit[t].w > 0 && mc >= s_hit[t].col && mc < s_hit[t].col + s_hit[t].w)
        {
            return t;
        }
    }
    return -1;
}

/** ov_subtab_status - status-line tag of sub-tab @t. */
const char *ov_subtab_status(ov_subtab_t t)
{
    return ((int) t >= 0 && t < OV_SUBTAB_COUNT) ? subtabs[t].status : "";
}

/** ov_render_subtab - draw the current sub-tab in the bottom-right panel (CONNECTIONS if it has nothing). */
void ov_render_subtab(OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_subtab_t t =
        ((int) lay->subtab >= 0 && lay->subtab < OV_SUBTAB_COUNT) ? lay->subtab : OV_SUB_CONN;
    if (!subtabs[t].render(lay, m))
    {
        ov_render_graph_panel(lay, m);
    }
}
