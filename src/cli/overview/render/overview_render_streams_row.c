// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_streams_row.c
 * @brief   Single stream row rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <inttypes.h>

/**
 * ov_streams_render_single_row - render a single stream row in the streams panel.
 * @lay:    Pointer to layout structure.
 * @m:      Pointer to data model snapshot.
 * @rel:    Pointer to relationship lookup tables.
 * @row:    Terminal screen row index.
 * @i:      Row index within visible panel list.
 * @fi:     Index in filtered stream array.
 * @si:     Stream index in data model.
 * @sdepth: Lineage depth for ancestry indicators.
 * @has_re: Non-zero if regex filter is active and compiled.
 * @re:     Pointer to compiled regex.
 * @r:      Bounding rectangle of streams panel.
 */
void ov_streams_render_single_row(const OV_LAYOUT  *lay,
                                  const OV_MODEL   *m,
                                  const OV_RELATED *rel,
                                  int               row,
                                  int               i,
                                  int               fi,
                                  int               si,
                                  int8_t            sdepth,
                                  int               has_re,
                                  const regex_t    *re,
                                  OV_RECT           r)
{
    const OV_STREAM *s         = &m->streams[si];
    int              is_sel    = (fi == lay->panel[OV_FOCUS_STREAMS].sel &&
                                  (lay->focus == OV_FOCUS_STREAMS || lay->focus == OV_FOCUS_GRAPH));
    int              is_frozen = (lay->freeze && lay->freeze_focus == OV_FOCUS_STREAMS &&
                                  fi == lay->panel[OV_FOCUS_STREAMS].freeze_row);
    ov_focus_t       eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int is_rel         = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_STREAMS && rel != NULL &&
                          bget(rel->streams, si));
    int is_loop_member = 0;
    if ((lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS) && lay->sel_loop >= 0 &&
        lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        if (s->loop_mask & active_mask)
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
        row_bg = (s->nb_loops > 1) ? OV_BG_LOOP_SHARED : OV_BG_LOOP;
    }
    else if (is_rel)
    {
        row_bg = OV_BG_RELATED;
    }
    else if (s->is_new > 0)
    {
        row_bg = OV_BG_NEW_ITEM;
    }
    else if (lay->mouse_hover && lay->panel[OV_FOCUS_STREAMS].hover_idx == si)
    {
        row_bg = OV_BG_HOVER;
    }
    row_bg = zebra_bg(row_bg, i);

    int avail = r.width - 2;
    ov_buf_pos(row, r.col + 1);
    ov_theme_bg(row_bg);
    render_focus_strip(row, r.col + 1, lay->focus == OV_FOCUS_STREAMS, OV_FG_STREAM, row_bg);
    OV_ROW_CTX rc      = { lay,
                           m,
                           rel,
                           si,
                           is_sel,
                           is_frozen,
                           is_rel,
                           eff_focus,
                           sdepth,
                           (rel != NULL) ? rel->sel_pid : 0,
                           has_re ? re : NULL };
    int        printed = ov_table_row(&ov_streams_table, &rc, row_bg, avail);

    if (lay->mouse_hover && lay->hover_view == OV_FOCUS_STREAMS && lay->hover_idx == fi)
    {
        ov_row_tooltip("Stream: %s | Dimensions: %dD (%s) | Semaphores: %d | inode: %" PRIu64,
                       s->name, s->naxis, s->size_str, s->nb_sem, (uint64_t) s->inode);

        int btn_w = 10; /* " [Delete] " */
        int rem   = r.width - printed;
        if (lay->ctrl_mode && rem >= btn_w) /* action buttons: CONTROL mode only */
        {
            render_pad_spaces(printed, r.width - btn_w);
            ov_theme_bg(OV_FG_ERROR);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" [Delete] ");
            printed = r.width;
        }
    }

    render_pad_spaces(printed, r.width);
    ov_buf_reset_attr();
}
