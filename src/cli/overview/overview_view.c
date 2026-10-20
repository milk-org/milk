// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_view.c
 * @brief   View descriptor table and view switching.
 */

#include <string.h>

#include "overview_layout.h"

const OV_VIEW_DESC ov_views[OV_VIEW_COUNT] = {
    [OV_VIEW_DASHBOARD] = { "DASH", "dash", 2,
                            OV_PANEL_BIT(OV_FOCUS_STREAMS) | OV_PANEL_BIT(OV_FOCUS_PROCS) |
                                OV_PANEL_BIT(OV_FOCUS_FPS) | OV_PANEL_BIT(OV_FOCUS_GRAPH),
                            -1 },
    [OV_VIEW_STREAMS] = { "STRM", "streams", 3, OV_PANEL_BIT(OV_FOCUS_STREAMS), OV_FOCUS_STREAMS },
    [OV_VIEW_PROCS]   = { "PROC", "procs", 4, OV_PANEL_BIT(OV_FOCUS_PROCS), OV_FOCUS_PROCS },
    [OV_VIEW_FPS]     = { "FPS", "fps", 5, OV_PANEL_BIT(OV_FOCUS_FPS), OV_FOCUS_FPS },
    [OV_VIEW_GRAPH]   = { "CONN", "conn", 6, OV_PANEL_BIT(OV_FOCUS_GRAPH), OV_FOCUS_GRAPH },
    [OV_VIEW_LOOPS]   = { "LOOPS", "loops", 7, OV_PANEL_BIT(OV_FOCUS_GRAPH), OV_FOCUS_GRAPH },
};

/** ov_view_cycle - view @step places after @v in tab order, wrapping. */
int ov_view_cycle(int v, int step)
{
    return (((v + step) % OV_VIEW_COUNT) + OV_VIEW_COUNT) % OV_VIEW_COUNT;
}

/** ov_view_by_fkey - view bound to function key F@fkey, or -1. */
int ov_view_by_fkey(int fkey)
{
    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        if (ov_views[v].fkey == fkey)
        {
            return v;
        }
    }
    return -1;
}

/** ov_view_by_name - view whose CLI name is @name, or -1. */
int ov_view_by_name(const char *name)
{
    for (int v = 0; name != NULL && v < OV_VIEW_COUNT; v++)
    {
        if (strcmp(ov_views[v].cli_name, name) == 0)
        {
            return v;
        }
    }
    return -1;
}

/** ov_view_has_panel - 1 if view @v shows panel @f. */
int ov_view_has_panel(ov_view_t v, ov_focus_t f)
{
    return (int) v >= 0 && v < OV_VIEW_COUNT && (ov_views[v].panels & OV_PANEL_BIT(f)) != 0;
}

/** ov_set_view - switch to view @v and give focus to the panel it opens on. */
void ov_set_view(OV_LAYOUT *lay, ov_view_t v)
{
    if ((int) v < 0 || v >= OV_VIEW_COUNT)
    {
        return;
    }
    lay->view = v;
    if (ov_views[v].default_focus >= 0)
    {
        lay->focus = (ov_focus_t) ov_views[v].default_focus;
    }
}
