// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_mouse_panels.c
 * @brief Panel-specific mouse click hit-testing, selection, and double-click actions
 */

#include "overview_input_internal.h"
#include <string.h>

/**
 * list_click - click in list panel @f drawn in @r: the header rows sort, a row moves the cursor and a
 * double click opens DETAILS on it.
 */
static void
list_click(OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t f, OV_RECT r, int mr, int mc, int is_dbl)
{
    int body_row = mr - r.row - 3;
    lay->focus   = f;
    if (body_row == -1 || body_row == -2)
    {
        ov_input__table_header_click(lay, f, mc);
    }
    else if (body_row >= 0 && ov_sel_row(lay, m, f, lay->panel[f].scroll + body_row) >= 0 && is_dbl)
    {
        lay->subtab = OV_SUB_DETAILS; /* rows past the list are ignored */
    }
}

/**
 * ov_input_mouse_panel_click - dispatch mouse clicks within view panels.
 * @lay:    Pointer to layout structure.
 * @m:      Pointer to data model snapshot.
 * @mr:     Mouse row coordinate.
 * @mc:     Mouse column coordinate.
 * @is_dbl: 1 if double-click detected, 0 otherwise.
 *
 * Return: 1 if click was consumed.
 */
int ov_input_mouse_panel_click(OV_LAYOUT *lay, const OV_MODEL *m, int mr, int mc, int is_dbl)
{
    if (lay->view == OV_VIEW_FPS)
    {
        /* F5: Check for split drag */
        if (mc == lay->r_fps_list.width || mc == lay->r_fps_list.width + 1)
        {
            lay->fps_split_dragging = 1;
            return 1;
        }

        /* F5: left = fps list, right = params */
        if (INSIDE(lay->r_fps_params, mr, mc))
        {
            lay->focus           = OV_FOCUS_FPS;
            lay->fps_param_focus = 1;
            int body_row         = mr - lay->r_fps_params.row - 2;
            if (body_row >= 0)
            {
                int fsel = ov_sel_idx(lay, m, OV_FOCUS_FPS);
                if (fsel >= 0)
                {
                    const OV_FPS   *fps = &m->fps[fsel];
                    fps_tree_item_t items[1024];
                    int nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);
                    if (nitems > 0)
                    {
                        int idx = lay->fps_param_scroll + body_row;
                        if (idx >= nitems)
                        {
                            lay->fps_param_sel = nitems - 1;
                        }
                        else
                        {
                            lay->fps_param_sel = idx;
                        }
                    }
                }
            }
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->fps_param_focus = 0;
            list_click(lay, m, OV_FOCUS_FPS, lay->r_fps, mr, mc, is_dbl);
        }
    }
    else
    {
        const OV_RECT *lists[OV_NB_LIST_PANELS] = { &lay->r_streams, &lay->r_procs, &lay->r_fps };
        int            hit                      = 0;
        for (int f = 0; f < OV_NB_LIST_PANELS && !hit; f++)
        {
            if (ov_view_has_panel(lay->view, (ov_focus_t) f) && INSIDE(*lists[f], mr, mc))
            {
                list_click(lay, m, (ov_focus_t) f, *lists[f], mr, mc, is_dbl);
                hit = 1;
            }
        }
        if (!hit && ov_view_has_panel(lay->view, OV_FOCUS_GRAPH) && INSIDE(lay->r_graph, mr, mc))
        {
            ov_input__subtab_click(lay, m, mr, mc, is_dbl);
        }
    }

    /* Check if clicking on cmdlog border to start dragging */
    int cmdlog_top = (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
    if (mr == cmdlog_top - 1 || mr == cmdlog_top)
    {
        lay->cmdlog_dragging = 1;
    }

    return 1;
}
