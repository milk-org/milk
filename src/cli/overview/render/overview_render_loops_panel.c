// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_loops_panel.c
 * @brief GUI rendering for LOOPS tab inside the dashboard graph panel.
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "overview_render_loops.h"
#include "overview_data_loops.h"
#include "overview_render_internal.h"
#include "overview_theme.h"

/**
 * ov_format_loop_path - Build a compact path string for loop cycle.
 * @m:   Model
 * @lp:  Loop
 * @buf: Output string buffer
 * @sz:  Buffer capacity
 */
void ov_format_loop_path(const OV_MODEL *m, const OV_LOOP *lp, char *buf, size_t sz)
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
    snprintf(htext, sizeof(htext), " %-4s %-20s %-8s %-10s %-8s %s", "ID", "NAME", "NODES",
             "RATE (Hz)", "STATUS", "OVERLAP");
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
            ov_format_loop_path(m, cl, pathbuf, sizeof(pathbuf));

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
                ov_theme_fg(OV_FG_DIM);
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
