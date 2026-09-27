// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_loops.c
 * @brief GUI rendering for dedicated fullscreen F7 LOOPS view in milk-CTRL.
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "overview_render_loops.h"
#include "overview_data_loops.h"
#include "overview_render_internal.h"
#include "overview_theme.h"

/**
 * ov_render_loops_view - Render the dedicated fullscreen F7 LOOPS view.
 * @lay: Layout state
 * @m:   System model
 */
void ov_render_loops_view(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int W = lay->term_cols;

    int body_top = lay->r_graph.row;
    int body_h   = lay->r_graph.height;
    if (body_h < 4)
    {
        body_h = 4;
    }

    /* Left pane: Loop table (40% width). Right pane: Detail circuit (60% width) */
    int lw = (int) (W * 0.42f);
    if (lw < 35)
    {
        lw = 35;
    }
    if (lw > W - 25)
    {
        lw = W - 25;
    }
    int rw = W - lw;

    /* Draw left panel frame */
    ov_draw_panel_border(body_top, 1, body_h, lw, "FEEDBACK LOOPS", OV_FG_LOOP, 1, 0);

    /* Draw right panel frame */
    char rtitle[128];
    int  sel_idx = lay->sel_loop;
    if (sel_idx < 0)
    {
        sel_idx = 0;
    }
    if (sel_idx >= m->nb_loops)
    {
        sel_idx = m->nb_loops - 1;
    }

    if (m->nb_loops > 0 && sel_idx < m->nb_loops)
    {
        snprintf(rtitle, sizeof(rtitle), "LOOP L%02d CIRCUIT & OVERLAP DETAILS",
                 m->loops[sel_idx].loop_id);
    }
    else
    {
        snprintf(rtitle, sizeof(rtitle), "LOOP CIRCUIT & DETAILS");
    }
    ov_draw_panel_border(body_top, lw + 1, body_h, rw, rtitle, OV_FG_TITLE, 0, 0);

    /* Left Pane: Loop List */
    int lrow = body_top + 1;
    ov_buf_pos(lrow, 2);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);
    char lhtext[128];
    int  lhlen = snprintf(lhtext, sizeof(lhtext), " %-4s %-16s %-6s %-8s %s", "ID", "NAME", "NODES",
                          "RATE", "STATUS");
    ov_buf_printf("%s", lhtext);
    render_pad_spaces(lhlen, lw);
    lrow++;

    int max_lrows = body_h - 3;
    int rendered  = 0;

    for (int i = 0; i < m->nb_loops && rendered < max_lrows; i++)
    {
        const OV_LOOP *lp     = &m->loops[i];
        int            is_sel = (i == sel_idx);

        ov_buf_pos(lrow, 2);
        ov_theme_bg(is_sel ? OV_BG_SELECTED : OV_BG_PANEL);

        ov_theme_fg(is_sel ? OV_FG_WARN : OV_FG_DIM);
        ov_buf_printf("%s", is_sel ? "\xe2\x96\xb6" : " ");

        ov_theme_fg(OV_FG_LOOP);
        ov_buf_printf("L%02d ", lp->loop_id);

        ov_theme_fg(is_sel ? OV_FG_BRIGHT : (lp->has_custom_name ? OV_FG_TITLE : OV_FG_TEXT));
        ov_buf_printf("%-16.16s ", lp->name);

        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("%1ds %1dp  ", lp->nb_streams, lp->nb_procs + lp->nb_fps);

        ov_theme_fg(OV_FG_TEXT);
        if (lp->min_hz > 0.0)
        {
            ov_buf_printf("%7.0f  ", lp->min_hz);
        }
        else
        {
            ov_buf_printf("   idle  ");
        }

        if (lp->is_running)
        {
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("RUN");
        }
        else if (lp->is_paused)
        {
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("PAUS");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("IDLE");
        }

        render_pad_to_col(lw);
        lrow++;
        rendered++;
    }

    for (; rendered < max_lrows; rendered++, lrow++)
    {
        clear_row(lrow, 2, lw - 2, OV_BG_PANEL);
    }

    /* Right Pane: Selected Loop Deep Inspector */
    int rrow       = body_top + 1;
    int max_rrows  = body_h - 2;
    int r_rendered = 0;

    if (m->nb_loops > 0 && sel_idx < m->nb_loops)
    {
        const OV_LOOP *cl = &m->loops[sel_idx];

        /* Overview Summary Banner */
        ov_buf_pos(rrow, lw + 2);
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_LOOP);
        ov_buf_printf(" Loop L%02d: %s", cl->loop_id, cl->name);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("  |  Bottleneck: %.1f Hz  |  Nodes: %d (%d streams, %d procs)", cl->min_hz,
                      cl->nb_nodes, cl->nb_streams, cl->nb_procs);
        render_pad_to_col(lw + rw);
        rrow++;
        r_rendered++;

        /* Path Circuit Box */
        ov_buf_pos(rrow, lw + 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf(" Topo Circuit:");
        render_pad_to_col(lw + rw);
        rrow++;
        r_rendered++;

        char pathbuf[256];
        ov_format_loop_path(m, cl, pathbuf, sizeof(pathbuf));

        ov_buf_pos(rrow, lw + 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_STREAM);
        ov_buf_printf("  %s", pathbuf);
        render_pad_to_col(lw + rw);
        rrow++;
        r_rendered++;

        /* Overlap Summary Box */
        ov_buf_pos(rrow, lw + 2);
        ov_theme_bg(OV_BG_PANEL);
        if (cl->overlap_mask != 0)
        {
            ov_theme_fg(OV_FG_LOOP_SHARED);
            ov_buf_printf(" \xe2\xae\x82 Overlap Analysis: Shares %d node(s) with other loops",
                          cl->nb_shared_nodes);
        }
        else
        {
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf(" Overlap Analysis: 100%% Isolated Loop (no shared resources)");
        }
        render_pad_to_col(lw + rw);
        rrow++;
        r_rendered++;

        /* Member Streams Table */
        ov_buf_pos(rrow, lw + 2);
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_STREAM_HDR);
        ov_buf_printf("  %-16s %-8s %-12s %-10s %s", "STREAM", "DTYPE", "DIMENSIONS", "UPDATE HZ",
                      "LOOP STATUS");
        render_pad_to_col(lw + rw);
        rrow++;
        r_rendered++;

        for (int i = 0; i < cl->nb_streams && r_rendered < max_rrows; i++)
        {
            int si = cl->stream_indices[i];
            if (si < 0 || si >= m->nb_streams)
            {
                continue;
            }
            const OV_STREAM *s = &m->streams[si];

            ov_buf_pos(rrow, lw + 2);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("  %-16.16s ", s->name);

            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%-8s ", render_dtype(s->datatype));

            char szb[32];
            if (s->naxis == 1)
            {
                snprintf(szb, sizeof(szb), "%u", (unsigned) s->size[0]);
            }
            else
            {
                snprintf(szb, sizeof(szb), "%ux%u", (unsigned) s->size[0], (unsigned) s->size[1]);
            }
            ov_buf_printf("%-12.12s ", szb);

            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf("%9.1f  ", s->update_hz);

            if (s->nb_loops > 1)
            {
                ov_theme_fg(OV_FG_LOOP_SHARED);
                ov_buf_printf("\xe2\xae\x82 SHARED (%d loops)", s->nb_loops);
            }
            else
            {
                ov_theme_fg(OV_FG_ACTIVE);
                ov_buf_printf("Exclusive");
            }

            render_pad_to_col(lw + rw);
            rrow++;
            r_rendered++;
        }

        /* Member Processes Table */
        if (r_rendered < max_rrows)
        {
            ov_buf_pos(rrow, lw + 2);
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_PROC_HDR);
            ov_buf_printf("  %-16s %-8s %-8s %-10s %s", "PROCESS", "PID", "STATUS", "RATE (Hz)",
                          "LOOP STATUS");
            render_pad_to_col(lw + rw);
            rrow++;
            r_rendered++;

            for (int i = 0; i < cl->nb_procs && r_rendered < max_rrows; i++)
            {
                int pi = cl->proc_indices[i];
                if (pi < 0 || pi >= m->nb_procs)
                {
                    continue;
                }
                const OV_PROC *p = &m->procs[pi];

                ov_buf_pos(rrow, lw + 2);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_PROC);
                ov_buf_printf("  %-16.16s ", p->name);

                ov_theme_fg(OV_FG_DIM);
                ov_buf_printf("%-8d ", (int) p->PID);

                if (p->loopstat == 1)
                {
                    ov_theme_fg(OV_FG_ACTIVE);
                    ov_buf_printf("RUN      ");
                }
                else if (p->loopstat == 2)
                {
                    ov_theme_fg(OV_FG_WARN);
                    ov_buf_printf("PAUS     ");
                }
                else
                {
                    ov_theme_fg(OV_FG_DIM);
                    ov_buf_printf("IDLE     ");
                }

                ov_theme_fg(OV_FG_TEXT);
                ov_buf_printf("%9.1f  ", p->loop_hz);

                if (p->nb_loops > 1)
                {
                    ov_theme_fg(OV_FG_LOOP_SHARED);
                    ov_buf_printf("\xe2\xae\x82 SHARED (%d loops)", p->nb_loops);
                }
                else
                {
                    ov_theme_fg(OV_FG_ACTIVE);
                    ov_buf_printf("Exclusive");
                }

                render_pad_to_col(lw + rw);
                rrow++;
                r_rendered++;
            }
        }
    }

    for (; r_rendered < max_rrows; r_rendered++, rrow++)
    {
        clear_row(rrow, lw + 2, rw - 2, OV_BG_PANEL);
    }

    ov_buf_reset_attr();
}
