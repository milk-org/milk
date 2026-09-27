// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_loops.c
 * @brief GUI rendering for LOOPS tab and fullscreen view in milk-CTRL
 *
 * Implements visualization of feedback loops, showing loop table, status,
 * bottleneck rates, cycle topology diagrams, and explicit breakdown of
 * exclusive vs overlapping/shared streams and processes.
 */

#include "overview_render_loops.h"
#include "overview_data_loops.h"
#include "overview_render_internal.h"
#include "overview_theme.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/**
 * format_loop_path - Build a compact path string for loop cycle.
 * @m:    Model
 * @lp:   Loop
 * @buf:  Output string buffer
 * @sz:   Buffer capacity
 */
static void format_loop_path(const OV_MODEL *m, const OV_LOOP *lp, char *buf, size_t sz)
{
    buf[0] = '\0';
    for (int i = 0; i < lp->nb_nodes; i++)
    {
        int ni = lp->node_indices[i];
        if (ni < 0 || ni >= m->nb_nodes)
        {
            continue;
        }
        const OV_NODE *n = &m->nodes[ni];

        /* Check if node is shared */
        int is_shared = 0;
        if (n->type == OV_NODE_STREAM && n->index >= 0 && n->index < m->nb_streams)
        {
            is_shared = (m->streams[n->index].nb_loops > 1);
        }
        else if (n->type == OV_NODE_PROC && n->index >= 0 && n->index < m->nb_procs)
        {
            is_shared = (m->procs[n->index].nb_loops > 1);
        }
        else if (n->type == OV_NODE_FPS && n->index >= 0 && n->index < m->nb_fps)
        {
            is_shared = (m->fps[n->index].nb_loops > 1);
        }

        char seg[64];
        snprintf(seg, sizeof(seg), "%s%s%s%s",
                 (i == 0) ? "" : " \xe2\x94\x80\xe2\x94\x80\xe2\x96\xb6 ", /* ──▶ */
                 (n->type == OV_NODE_STREAM) ? "" : "[", n->name,
                 (n->type == OV_NODE_STREAM) ? (is_shared ? "*" : "") : (is_shared ? "*]" : "]"));

        if (strlen(buf) + strlen(seg) < sz - 8)
        {
            strncat(buf, seg, sz - strlen(buf) - 1);
        }
        else
        {
            strncat(buf, " ...", sz - strlen(buf) - 1);
            break;
        }
    }
    strncat(buf, " \xe2\x86\xba", sz - strlen(buf) - 1); /* ↺ */
}

/**
 * ov_render_loops_panel - Render the LOOPS tab inside the dashboard graph panel.
 * @lay: Layout state
 * @m:   System model
 */
void ov_render_loops_panel(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT     r      = lay->r_graph;
    const char *tabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, tabs, 4, lay->graph_tab_mode, OV_FG_LOOP,
                       lay->focus == OV_FOCUS_GRAPH);

    int max_rows = r.height - 3;
    int row      = r.row + 1;

    /* Header row */
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);
    char htext[256];
    snprintf(htext, sizeof(htext), " %-4s %-20s %-8s %-10s %-8s %s", "ID", "NAME",
             "NODES", "RATE (Hz)", "STATUS", "OVERLAP");
    ov_buf_printf("%s", htext);
    render_pad_to_col(r.col + r.width - 1);
    row++;

    if (m->nb_loops == 0)
    {
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("  ");
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("No closed feedback loops detected (mode: %s)",
                      sg_mode_label(lay->lineage_mode));
        render_pad_to_col(r.col + r.width - 1);
        row++;
        for (int i = 1; i < max_rows; i++, row++)
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return;
    }

    int sel_idx = lay->sel_loop;
    if (sel_idx < 0)
    {
        sel_idx = 0;
    }
    if (sel_idx >= m->nb_loops)
    {
        sel_idx = m->nb_loops - 1;
    }

    /* Split panel: top half list of loops, bottom half selected loop details */
    int list_rows   = max_rows;
    int detail_rows = 0;
    if (max_rows >= 6)
    {
        list_rows   = (max_rows > 8) ? (max_rows / 2) : 3;
        detail_rows = max_rows - list_rows;
    }

    int rendered_list = 0;
    int scroll        = lay->scroll_loop;
    if (scroll < 0)
    {
        scroll = 0;
    }

    /* Loop list rows */
    for (int li = scroll; li < m->nb_loops && rendered_list < list_rows; li++)
    {
        const OV_LOOP *lp     = &m->loops[li];
        int            is_sel = (li == sel_idx && lay->focus == OV_FOCUS_GRAPH);

        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(is_sel ? OV_BG_SELECTED : OV_BG_PANEL);

        /* Selection cursor */
        if (is_sel)
        {
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("\xe2\x96\xb6"); /* ▶ */
        }
        else
        {
            ov_buf_printf(" ");
        }

        /* Loop ID badge */
        ov_theme_fg(OV_FG_LOOP);
        ov_buf_printf("L%02d ", lp->loop_id);

        /* Loop Name */
        ov_theme_fg(is_sel ? OV_FG_BRIGHT : (lp->has_custom_name ? OV_FG_TITLE : OV_FG_TEXT));
        ov_buf_printf("%-20.20s ", lp->name);

        /* Node counts */
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("%2ds %2dp  ", lp->nb_streams, lp->nb_procs + lp->nb_fps);

        /* Frequency (Hz) */
        ov_theme_fg(OV_FG_TEXT);
        if (lp->min_hz > 0.0)
        {
            ov_buf_printf("%9.1f  ", lp->min_hz);
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("     idle  ");
        }

        /* Status */
        if (lp->is_error)
        {
            ov_theme_fg(OV_FG_ERROR);
            ov_buf_printf("ERR   ");
        }
        else if (lp->is_paused)
        {
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("PAUS  ");
        }
        else if (lp->is_running)
        {
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("RUN   ");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("IDLE  ");
        }

        /* Overlap summary */
        if (lp->overlap_mask != 0)
        {
            ov_theme_fg(OV_FG_LOOP_SHARED);
            char ovbuf[64];
            int  first_ov = -1;
            for (int k = 0; k < m->nb_loops; k++)
            {
                if (k != li && (lp->overlap_mask & (UINT32_C(1) << k)))
                {
                    first_ov = k;
                    break;
                }
            }
            if (first_ov >= 0)
            {
                snprintf(ovbuf, sizeof(ovbuf), "\xe2\xae\x82 with L%02d (%d shared)", first_ov + 1,
                         lp->nb_shared_nodes);
            }
            else
            {
                snprintf(ovbuf, sizeof(ovbuf), "\xe2\xae\x82 %d shared", lp->nb_shared_nodes);
            }
            ov_buf_printf("%-24.24s", ovbuf);
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%-24s", "exclusive");
        }

        render_pad_to_col(r.col + r.width - 1);
        row++;
        rendered_list++;
    }

    /* Fill remaining list rows if any */
    for (; rendered_list < list_rows; rendered_list++, row++)
    {
        clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
    }

    /* Divider and Detail section */
    if (detail_rows > 0 && sel_idx < m->nb_loops)
    {
        const OV_LOOP *cl = &m->loops[sel_idx];

        /* Divider line */
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("\xe2\x94\x80\xe2\x94\x80[ Loop L%02d Details ", cl->loop_id);
        int dlen = 22;
        if (cl->overlap_mask != 0)
        {
            ov_theme_fg(OV_FG_LOOP_SHARED);
            ov_buf_printf("\xe2\xae\x82 %d Shared Nodes ", cl->nb_shared_nodes);
            dlen += 18;
        }
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("]");
        dlen += 1;
        for (int c = dlen; c < r.width - 2; c++)
        {
            ov_buf_printf("\xe2\x94\x80");
        }
        row++;
        detail_rows--;

        /* Rename input mode line */
        if (lay->renaming_loop && detail_rows > 0)
        {
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(OV_BG_SELECTED);
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf(" Rename L%02d: [ %s_ ]  (Enter: save, Esc: cancel)", cl->loop_id,
                          lay->rename_buf);
            render_pad_to_col(r.col + r.width - 1);
            row++;
            detail_rows--;
        }

        /* Path line */
        if (detail_rows > 0)
        {
            char pathbuf[256];
            format_loop_path(m, cl, pathbuf, sizeof(pathbuf));

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" Path: ");
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("%.*s", r.width - 12, pathbuf);
            render_pad_to_col(r.col + r.width - 1);
            row++;
            detail_rows--;
        }

        /* Overlap details or members line */
        if (detail_rows > 0)
        {
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);

            if (cl->overlap_mask != 0)
            {
                ov_theme_fg(OV_FG_LOOP_SHARED);
                ov_buf_printf(" Shared: ");
                int printed_sh = 9;

                for (int ni = 0; ni < cl->nb_nodes; ni++)
                {
                    int            nidx    = cl->node_indices[ni];
                    const OV_NODE *n       = &m->nodes[nidx];
                    int            n_loops = 0;
                    if (n->type == OV_NODE_STREAM && n->index >= 0 && n->index < m->nb_streams)
                    {
                        n_loops = m->streams[n->index].nb_loops;
                    }
                    else if (n->type == OV_NODE_PROC && n->index >= 0 && n->index < m->nb_procs)
                    {
                        n_loops = m->procs[n->index].nb_loops;
                    }
                    else if (n->type == OV_NODE_FPS && n->index >= 0 && n->index < m->nb_fps)
                    {
                        n_loops = m->fps[n->index].nb_loops;
                    }

                    if (n_loops > 1 && printed_sh < r.width - 25)
                    {
                        ov_theme_fg(OV_FG_WARN);
                        ov_buf_printf("%s* ", n->name);
                        printed_sh += (int) strlen(n->name) + 2;
                    }
                }
                ov_theme_fg(OV_FG_DIM);
                ov_buf_printf("(*shared across %d loops)", cl->nb_shared_nodes);
                render_pad_to_col(r.col + r.width - 1);
            }
            else
            {
                ov_theme_fg(OV_FG_ACTIVE);
                ov_buf_printf(" Exclusive: ");
                ov_theme_fg(OV_FG_DIM);
                ov_buf_printf("All %d nodes dedicated strictly to loop L%02d", cl->nb_nodes,
                              cl->loop_id);
                render_pad_to_col(r.col + r.width - 1);
            }
            row++;
            detail_rows--;
        }

        /* Action shortcuts / hints line */
        if (detail_rows > 0)
        {
            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);

            if (lay->loop_filter_active)
            {
                ov_theme_fg(OV_FG_WARN);
                ov_buf_printf(" [FILTER ACTIVE: Loop L%02d]  [f] Clear Filter  [r] Rename",
                              cl->loop_id);
                render_pad_to_col(r.col + r.width - 1);
            }
            else
            {
                ov_buf_printf(" [r] Rename Loop   [f] Filter Dashboard to Loop   [Shift-Tab] Mode");
                render_pad_to_col(r.col + r.width - 1);
            }
            row++;
            detail_rows--;
        }

        /* Clear remaining detail rows */
        for (; detail_rows > 0; detail_rows--, row++)
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }

    ov_buf_reset_attr();
}

/**
 * ov_render_loops_view - Render the dedicated fullscreen F7 LOOPS view.
 * @lay: Layout state
 * @m:   System model
 */
void ov_render_loops_view(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int W = lay->term_cols;
    int H = lay->term_rows;

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
        format_loop_path(m, cl, pathbuf, sizeof(pathbuf));

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
