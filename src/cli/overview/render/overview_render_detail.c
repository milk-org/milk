// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_detail.c
 * @brief   Detail panel dispatcher for milk-CTRL.
 */

#include "overview_render_detail_internal.h"

/**
 * ov_render_detail_panel - dispatch detail inspector pane based on active focus and selection.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if detail was drawn, 0 if nothing to show (caller falls back to graph panel).
 */
int ov_render_detail_panel(OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r        = lay->r_graph;
    int     max_rows = r.height - 2;
    int     row      = r.row + 1;

    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int        ssel  = ov_get_selected_stream_idx(lay, m);
    int        psel  = ov_get_selected_proc_idx(lay, m);
    int        fsel  = ov_get_selected_fps_idx(lay, m);

    /* If a list panel is directly focused, show its item's details. */
    if (focus == OV_FOCUS_STREAMS && ssel >= 0 && ssel < m->nb_streams)
    {
        return ov_fps__render_detail_stream(lay, m, ssel, r, max_rows, row);
    }
    if (focus == OV_FOCUS_PROCS && psel >= 0 && psel < m->nb_procs)
    {
        return ov_fps__render_detail_proc(lay, m, psel, r, max_rows, row);
    }
    if (focus == OV_FOCUS_FPS && fsel >= 0 && fsel < m->nb_fps)
    {
        return ov_fps__render_detail_fps(lay, m, fsel, r, max_rows, row);
    }

    /* Focus is on the graph panel (or no focused list panel).
     * Still render details for whatever list item is selected,
     * so that clicking/tabbing into the graph panel does not
     * make the panel jump to CONNECTIONS view. */
    if (fsel >= 0 && fsel < m->nb_fps)
    {
        return ov_fps__render_detail_fps(lay, m, fsel, r, max_rows, row);
    }
    if (ssel >= 0 && ssel < m->nb_streams)
    {
        return ov_fps__render_detail_stream(lay, m, ssel, r, max_rows, row);
    }
    if (psel >= 0 && psel < m->nb_procs)
    {
        return ov_fps__render_detail_proc(lay, m, psel, r, max_rows, row);
    }

    return 0;
}
