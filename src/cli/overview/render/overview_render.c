// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render.c
 * @brief   Main frame orchestration and view dispatcher for milk-CTRL.
 */

#include "overview_render_internal.h"
#include "overview_render_fps_params.h"
#include "overview_render_loops.h"
#include <stdio.h>
/**
 * ov_render__dispatch_view - route rendering to active view panels.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 * @rel: Pointer to relationship lookup tables.
 */
static void ov_render__dispatch_view(OV_LAYOUT *lay, const OV_MODEL *m, const OV_RELATED *rel)
{
    switch (lay->view)
    {
    case OV_VIEW_DASHBOARD:
        ov_render_preview_line(lay, m);
        ov_render_streams_panel(lay, m, rel);
        ov_render_procs_panel(lay, m, rel);
        ov_render_fps_panel(lay, m, rel);
        int rendered = 0;
        if (lay->graph_tab_mode == 1)
        {
            ov_render_loops_panel(lay, m);
            rendered = 1;
        }
        else if (lay->graph_tab_mode == 2)
        {
            rendered = ov_render_detail_panel(lay, m);
        }
        else if (lay->graph_tab_mode == 3)
        {
            rendered = ov_render_resources_panel(lay, m);
        }

        if (!rendered)
        {
            ov_render_graph_panel(lay, m);
        }
        break;
    case OV_VIEW_GRAPH:
        ov_render_graph_panel(lay, m);
        break;
    case OV_VIEW_LOOPS:
        ov_render_loops_view(lay, m);
        break;
    case OV_VIEW_STREAMS:
        ov_render_streams_panel(lay, m, rel);
        break;
    case OV_VIEW_PROCS:
        ov_render_procs_panel(lay, m, rel);
        break;
    case OV_VIEW_FPS:
        ov_render_fps_param_info(lay, m);
        ov_render_fps_panel(lay, m, rel);
        int cur_fsel = ov_get_selected_fps_idx(lay, m);
        if (cur_fsel >= 0 && cur_fsel < m->nb_fps && m->fps[cur_fsel].nb_disp_params > 0)
        {
            ov_render_fps_params_panel(lay, m);
        }
        else
        {
            /* No params: draw empty right panel */
            ov_draw_panel_border(lay->r_fps_params.row, lay->r_fps_params.col,
                                 lay->r_fps_params.height, lay->r_fps_params.width, "PARAMS",
                                 OV_FG_DIM, 0, 0);
        }
        break;
    default:
        break;
    }
}
/**
 * ov_render__draw_edge_highlights - draw split-pane hover highlights for draggable separators.
 * @lay: Pointer to layout structure.
 */
static void ov_render__draw_edge_highlights(const OV_LAYOUT *lay)
{
    /* Highlight movable edges if hovering */
    if (lay->mouse_hover && !lay->show_help)
    {
        ov_theme_fg(OV_FG_WARN);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_bold();

        if (lay->cmdlog_split_hover)
        {
            int cmdlog_top =
                (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
            if (cmdlog_top > 1)
            {
                ov_buf_pos(cmdlog_top - 1, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
            }
        }

        if (lay->view == OV_VIEW_DASHBOARD)
        {
            if (lay->dash_split_h_hover)
            {
                int r = lay->r_streams.row + lay->r_streams.height - 1;
                ov_buf_pos(r, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
                ov_buf_pos(r + 1, 1);
                ov_buf_hline_utf8(OV_BOX_H_D, lay->term_cols);
            }
            if (lay->dash_split_v_hover)
            {
                int c = lay->r_streams.width;
                for (int rr = lay->r_streams.row; rr < lay->r_fps.row + lay->r_fps.height; rr++)
                {
                    ov_buf_pos(rr, c);
                    ov_buf_printf("%s", OV_BOX_V_D);
                    ov_buf_pos(rr, c + 1);
                    ov_buf_printf("%s", OV_BOX_V_D);
                }
            }
        }
        else if (lay->view == OV_VIEW_FPS)
        {
            if (lay->fps_split_hover)
            {
                int c = lay->r_fps_list.width;
                for (int rr = lay->r_fps_list.row;
                     rr < lay->r_fps_list.row + lay->r_fps_list.height; rr++)
                {
                    ov_buf_pos(rr, c);
                    ov_buf_printf("%s", OV_BOX_V_D);
                    ov_buf_pos(rr, c + 1);
                    ov_buf_printf("%s", OV_BOX_V_D);
                }
            }
        }

        ov_buf_reset_attr();
    }
}
/**
 * ov_render_frame - compose and render one full overview frame.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 */
void ov_render_frame(OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_buf_reset_size(lay->term_rows, lay->term_cols);

    /* Perform global hit-test to populate hover state */
    ov_hittest(lay, m, ov_mouse_row, ov_mouse_col);
    ov_hittest_resolve_globals(lay, m);

    ov_render__sync_selection(lay, m);

    /* Compute cross-panel relation set once per frame */
    OV_RELATED rel;
    ov_compute_related(lay, m, &rel);

    ov_render_header(lay, m);
    ov_render_tabs(lay);

    /* Render view panels if help overlay is not active */
    if (!lay->show_help)
    {
        ov_render__dispatch_view(lay, m, &rel);
    }

    if (lay->show_help)
    {
        ov_render_help(lay, m);
    }

    if (!lay->show_help)
    {
        ov_render_cmdlog(lay);
    }
    ov_render_status(lay, m);

    ov_render__draw_edge_highlights(lay);

    /* End frame overlays */
    ov_render_theme_popup(lay);
    ov_draw_tooltip(lay);

    ov_buf_flush_delta(lay->term_rows, lay->term_cols);
}
