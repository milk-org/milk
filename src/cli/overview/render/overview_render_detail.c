// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_detail.c
 * @brief   Detail panel dispatcher for milk-CTRL.
 */

#include "overview_render_detail_internal.h"

/**
 * ov_detail_target - model index and panel of the entry the DETAILS pane shows (the focused list's
 * entry, else the selected FPS, stream, then process). Return: index, or -1 with *panel unchanged.
 */
int ov_detail_target(const OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t *panel)
{
    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int        sel[OV_NB_LIST_PANELS];
    sel[OV_FOCUS_STREAMS] = ov_get_selected_stream_idx(lay, m);
    sel[OV_FOCUS_PROCS]   = ov_get_selected_proc_idx(lay, m);
    sel[OV_FOCUS_FPS]     = ov_get_selected_fps_idx(lay, m);

    if (focus < OV_FOCUS_GRAPH && sel[focus] >= 0)
    {
        *panel = focus;
        return sel[focus];
    }
    static const ov_focus_t order[] = { OV_FOCUS_FPS, OV_FOCUS_STREAMS, OV_FOCUS_PROCS };
    for (int i = 0; i < 3; i++)
    {
        if (sel[order[i]] >= 0)
        {
            *panel = order[i];
            return sel[order[i]];
        }
    }
    return -1;
}

/**
 * ov_render_detail_panel - dispatch detail inspector pane based on active focus and selection.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if detail was drawn, 0 if nothing to show (caller falls back to graph panel).
 */
int ov_render_detail_panel(OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT    r        = lay->r_graph;
    int        max_rows = r.height - 2;
    int        row      = r.row + 1;
    ov_focus_t panel    = OV_FOCUS_STREAMS;
    int        idx      = ov_detail_target(lay, m, &panel);

    if (idx < 0)
    {
        return 0;
    }
    if (panel == OV_FOCUS_STREAMS)
    {
        return ov_fps__render_detail_stream(lay, m, idx, r, max_rows, row);
    }
    if (panel == OV_FOCUS_PROCS)
    {
        return ov_fps__render_detail_proc(lay, m, idx, r, max_rows, row);
    }
    return ov_fps__render_detail_fps(lay, m, idx, r, max_rows, row);
}
