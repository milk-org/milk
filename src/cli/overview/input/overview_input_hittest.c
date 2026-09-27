// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_hittest.c
 * @brief Mouse hit-testing, region boundaries, hover detection, and splitters
 */

#include "overview_input_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_hittest - evaluate mouse coordinates against panel bounding boxes and elements.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 * @mr:  Mouse row coordinate.
 * @mc:  Mouse column coordinate.
 */
void ov_hittest(
    OV_LAYOUT      *lay,
    const OV_MODEL *m,
    int             mr,
    int             mc)
{
    lay->hover_view         = -1;
    lay->hover_idx          = -1;
    lay->hover_is_header    = 0;
    lay->hover_col_logical  = mc;
    lay->hover_tooltip[0]   = '\0';
    lay->fps_split_hover    = 0;
    lay->dash_split_v_hover = 0;
    lay->dash_split_h_hover = 0;
    lay->cmdlog_split_hover = 0;

    if (!lay->mouse_hover)
    {
        return;
    }

    if (mr == lay->r_header.row)
    {
        int commit_w    = (int) strlen(MILK_GIT_COMMIT) + 3;
        int shmdir_w    = (int) strlen(ov_get_shmdir()) + 8;
        int badge_start = lay->r_header.col + 18 + commit_w + shmdir_w;
        int badge_w     = lay->ctrl_mode ? 13 : 15;
        if (mc >= badge_start && mc < badge_start + badge_w)
        {
            snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                     "Control Mode: Toggle write access & actions (key: c)");
            return;
        }
        int hover_badge_start = badge_start + badge_w + 1;
        int hover_badge_w     = lay->mouse_hover ? 15 : 16;
        if (mc >= hover_badge_start && mc < hover_badge_start + hover_badge_w)
        {
            snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                     "Mouse Hover: Toggle hover tooltips (key: m)");
            return;
        }
        /* Check header filter badges */
        for (int fb = 0; fb < lay->r_filter_count; fb++)
        {
            int fb_start = lay->r_filter_start[fb];
            int fb_w     = lay->r_filter_width[fb];
            if (mc >= fb_start && mc < fb_start + fb_w)
            {
                ov_focus_t  fpanel     = lay->r_filter_panel[fb];
                const char *panel_full = (fpanel == OV_FOCUS_STREAMS) ? "Streams"
                                         : (fpanel == OV_FOCUS_PROCS) ? "Processes"
                                         : (fpanel == OV_FOCUS_FPS)   ? "FPS"
                                                                      : "Panel";
                const char *fpat       = (fpanel != OV_FOCUS_GRAPH)
                                             ? ov_get_panel_filter_pattern(lay, fpanel)
                                             : ov_get_filter_pattern(lay);
                int is_act = (fpanel != OV_FOCUS_GRAPH)
                                 ? ov_is_panel_filter_active(lay, fpanel)
                                 : ov_is_filter_active(lay);

                if (is_act)
                {
                    snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                             "%s Filter ON: Click or 'f' to pause, ESC to clear", panel_full);
                }
                else if (fpat[0] != '\0')
                {
                    snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                             "%s Filter OFF: Click or 'f' to resume, ESC to clear", panel_full);
                }
                else
                {
                    snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                             "%s Filter: Click or press [/] to set regex filter", panel_full);
                }
                return;
            }
        }
    }

    if (mr == lay->r_tabs.row)
    {
        int tab_widths[OV_VIEW_COUNT];
        int tabs_total_width = 0;
        for (int v = 0; v < OV_VIEW_COUNT; v++)
        {
            tab_widths[v] = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
            tabs_total_width += tab_widths[v];
        }
        int help_width = 11;
        int help_col   = (lay->term_cols >= tabs_total_width + help_width)
                             ? (lay->term_cols - help_width + 1)
                             : (tabs_total_width + 1);

        if (mc >= help_col && mc < help_col + help_width)
        {
            snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                     "Help: Toggle interactive help & keybinding menu (key: h)");
            return;
        }

        int tx = 1;
        for (int v = 0; v < OV_VIEW_COUNT; v++)
        {
            if (mc >= tx && mc < tx + tab_widths[v])
            {
                snprintf(lay->hover_tooltip, sizeof(lay->hover_tooltip),
                         "View: Switch to %s view (key: F%d)",
                         ov_view_label((ov_view_t) v), v + 2);
                return;
            }
            tx += tab_widths[v];
        }
    }

    int cmdlog_top = (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
    if (mr == cmdlog_top - 1 || mr == cmdlog_top)
    {
        lay->cmdlog_split_hover = 1;
    }

    if (lay->view == OV_VIEW_FPS)
    {
        if (mc >= lay->r_fps_list.width - 1 && mc <= lay->r_fps_list.width + 2)
        {
            lay->fps_split_hover = 1;
        }
    }
    else if (lay->view == OV_VIEW_DASHBOARD)
    {
        int h_split_row = lay->r_streams.row + lay->r_streams.height;
        int v_split_col = lay->r_streams.width;

        if (mr >= h_split_row - 1 && mr <= h_split_row + 1)
        {
            lay->dash_split_h_hover = 1;
        }
        if (mc >= v_split_col - 1 && mc <= v_split_col + 2)
        {
            lay->dash_split_v_hover = 1;
        }
    }

    if (lay->view == OV_VIEW_FPS)
    {
        if (INSIDE(lay->r_fps_params, mr, mc))
        {
            lay->hover_view = OV_FOCUS_FPS;
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->hover_view = OV_FOCUS_FPS;
            int body_row    = mr - lay->r_fps.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_fps + body_row;
                if (idx < m->nb_fps)
                {
                    lay->hover_idx = idx;
                }
            }
        }
    }
    else if (lay->view == OV_VIEW_STREAMS && INSIDE(lay->r_streams, mr, mc))
    {
        lay->hover_view = OV_FOCUS_STREAMS;
        int body_row    = mr - lay->r_streams.row - 3;
        if (body_row == -1 || body_row == -2)
        {
            lay->hover_is_header = 1;
        }
        else if (body_row >= 0)
        {
            int idx = lay->scroll_stream + body_row;
            if (idx < m->nb_streams)
            {
                lay->hover_idx = idx;
            }
        }
    }
    else if (lay->view == OV_VIEW_PROCS && INSIDE(lay->r_procs, mr, mc))
    {
        lay->hover_view = OV_FOCUS_PROCS;
        int body_row    = mr - lay->r_procs.row - 3;
        if (body_row == -1 || body_row == -2)
        {
            lay->hover_is_header = 1;
        }
        else if (body_row >= 0)
        {
            int idx = lay->scroll_proc + body_row;
            if (idx < m->nb_procs)
            {
                lay->hover_idx = idx;
            }
        }
    }
    else if ((lay->view == OV_VIEW_GRAPH || lay->view == OV_VIEW_LOOPS) &&
             INSIDE(lay->r_graph, mr, mc))
    {
        lay->hover_view = OV_FOCUS_GRAPH;
        int body_row    = mr - lay->r_graph.row - 2;
        if (body_row >= 0)
        {
            if (lay->graph_tab_mode == 0 && lay->view != OV_VIEW_LOOPS)
            {
                int idx = lay->scroll_graph + body_row;
                if (idx < m->nb_edges)
                {
                    lay->hover_idx = idx;
                }
            }
            else if (lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS)
            {
                int idx = lay->scroll_loop + body_row;
                if (idx < m->nb_loops)
                {
                    lay->hover_idx = idx;
                }
            }
            else
            {
                lay->hover_idx = body_row;
            }
        }
    }
    else if (lay->view == OV_VIEW_DASHBOARD)
    {
        if (INSIDE(lay->r_streams, mr, mc))
        {
            lay->hover_view = OV_FOCUS_STREAMS;
            int body_row    = mr - lay->r_streams.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_stream + body_row;
                if (idx < m->nb_streams)
                {
                    lay->hover_idx = idx;
                }
            }
        }
        else if (INSIDE(lay->r_procs, mr, mc))
        {
            lay->hover_view = OV_FOCUS_PROCS;
            int body_row    = mr - lay->r_procs.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_proc + body_row;
                if (idx < m->nb_procs)
                {
                    lay->hover_idx = idx;
                }
            }
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->hover_view = OV_FOCUS_FPS;
            int body_row    = mr - lay->r_fps.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                lay->hover_is_header = 1;
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_fps + body_row;
                if (idx < m->nb_fps)
                {
                    lay->hover_idx = idx;
                }
            }
        }
        else if (INSIDE(lay->r_graph, mr, mc))
        {
            lay->hover_view = OV_FOCUS_GRAPH;
            int body_row    = mr - lay->r_graph.row - 2;
            if (body_row >= 0)
            {
                if (lay->graph_tab_mode == 0)
                {
                    int idx = lay->scroll_graph + body_row;
                    if (idx < m->nb_edges)
                    {
                        lay->hover_idx = idx;
                    }
                }
                else if (lay->graph_tab_mode == 1)
                {
                    int idx = lay->scroll_loop + body_row;
                    if (idx < m->nb_loops)
                    {
                        lay->hover_idx = idx;
                    }
                }
                else
                {
                    lay->hover_idx = body_row;
                }
            }
        }
    }
}
