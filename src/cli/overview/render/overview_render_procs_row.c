// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_procs_row.c
 * @brief   Single process row rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/**
 * ov_procs_render_single_row - render a single process row in the processes panel.
 * @lay:    Pointer to layout structure.
 * @m:      Pointer to data model snapshot.
 * @rel:    Pointer to relationship lookup tables.
 * @row:    Terminal screen row index.
 * @i:      Row index within visible panel list.
 * @fi:     Index in filtered process array.
 * @pi:     Process index in data model.
 * @sdepth: Lineage depth for ancestry indicators.
 * @has_re: Non-zero if regex filter is active and compiled.
 * @re:     Pointer to compiled regex.
 * @r:      Bounding rectangle of processes panel.
 */
void ov_procs_render_single_row(const OV_LAYOUT  *lay,
                                const OV_MODEL   *m,
                                const OV_RELATED *rel,
                                int               row,
                                int               i,
                                int               fi,
                                int               pi,
                                int8_t            sdepth,
                                int               has_re,
                                const regex_t    *re,
                                OV_RECT           r)
{
    const OV_PROC *p         = &m->procs[pi];
    int            is_sel    = (fi == lay->panel[OV_FOCUS_PROCS].sel &&
                                (lay->focus == OV_FOCUS_PROCS || lay->focus == OV_FOCUS_GRAPH));
    int            is_frozen = (lay->freeze && lay->freeze_focus == OV_FOCUS_PROCS &&
                                fi == lay->panel[OV_FOCUS_PROCS].freeze_row);
    ov_focus_t     eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int            has_rel   = (rel != NULL && bget(rel->procs, pi));
    int            is_rel    = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_PROCS && has_rel);
    int            is_loop_member = 0;
    if ((lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS) && lay->sel_loop >= 0 &&
        lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        if (p->loop_mask & active_mask)
        {
            is_loop_member = 1;
        }
    }

    ov_rgb_t row_bg = OV_BG_PANEL;
    if (is_sel)
    {
        row_bg = OV_BG_SELECTED;
    }
    else if (is_frozen)
    {
        row_bg = OV_BG_FROZEN;
    }
    else if (is_loop_member)
    {
        row_bg = (p->nb_loops > 1) ? OV_BG_LOOP_SHARED : OV_BG_LOOP;
    }
    else if (is_rel)
    {
        row_bg = OV_BG_RELATED;
    }
    else if (p->stale_count >= OV_STALE_SCANS)
    {
        row_bg = OV_BG_STALE;
    }
    else if (p->is_new > 0)
    {
        row_bg = OV_BG_NEW_ITEM;
    }
    else if (lay->mouse_hover && lay->panel[OV_FOCUS_PROCS].hover_idx == pi)
    {
        row_bg = OV_BG_HOVER;
    }
    row_bg = zebra_bg(row_bg, i);

    int avail = r.width - 2;
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(row_bg);
    ov_buf_printf(" ");
    OV_ROW_CTX rc = {
        lay, m, rel, pi, is_sel, is_frozen, is_rel, eff_focus, sdepth, 0, has_re ? re : NULL
    };
    int printed = ov_table_row(&ov_procs_table, &rc, row_bg, avail);

    if (lay->mouse_hover && lay->hover_view == OV_FOCUS_PROCS && lay->hover_idx == fi)
    {
        char loc_upt[12] = "-";
        if (p->start_time_sec > 0)
        {
            format_uptime(loc_upt, sizeof(loc_upt), p->start_time_sec);
        }
        ov_row_tooltip("PID: %d | Up: %s | CPU: %4.1f%% | Exec: %zu ns", (int) p->PID, loc_upt,
                       p->cpu_used, (size_t) p->dtmedian_exec_ns);

        int btn_w = 8; /* " [Kill] " */
        int rem   = avail - printed;
        if (lay->ctrl_mode && rem >= btn_w) /* action buttons: CONTROL mode only */
        {
            ov_buf_hline(' ', rem - btn_w);
            ov_theme_bg(OV_FG_ERROR);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" [Kill] ");
            printed += rem;
        }
    }

    /* Pad remainder */
    {
        int rem = avail - printed;
        if (rem > 0)
        {
            ov_buf_hline(' ', rem);
        }
    }
    ov_buf_reset_attr();
}
