// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_streams.c
 * @brief   STREAMS panel rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_streams__render_rows - render filtered stream rows and scrollbar.
 * @lay:           Pointer to layout structure.
 * @m:             Pointer to data model snapshot.
 * @rel:           Pointer to related entities lookup.
 * @r:             Bounding rectangle of streams panel.
 * @filt_idx:      Array of stream indices matching active filter.
 * @filt_n:        Count of matching streams.
 * @active_filter: Active filter string.
 */
static void ov_streams__render_rows(const OV_LAYOUT  *lay,
                                    const OV_MODEL   *m,
                                    const OV_RELATED *rel,
                                    OV_RECT           r,
                                    const int        *filt_idx,
                                    int               filt_n,
                                    const char       *active_filter)
{
    int hrow = r.row + 1;

    /* Compute lineage depths when a stream is selected. sel_stream is a position in
     * the filtered list; convert via filt_idx[] to obtain model-level stream index. */
    int8_t local_depth[OV_MAX_STREAMS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->mouse_hover && lay->hover_global_stream >= 0)
        {
            for (int i = 0; i < filt_n; i++)
            {
                if (filt_idx[i] == lay->hover_global_stream)
                {
                    eff_sel = i;
                    break;
                }
            }
        }
        else if (lay->freeze && lay->freeze_focus == OV_FOCUS_STREAMS &&
                 lay->freeze_sel_stream >= 0 && lay->freeze_sel_stream < filt_n)
        {
            eff_sel = lay->freeze_sel_stream;
        }
        else if (lay->focus == OV_FOCUS_STREAMS && lay->sel_stream >= 0 && lay->sel_stream < filt_n)
        {
            eff_sel = lay->sel_stream;
        }

        if (eff_sel >= 0)
        {
            int        root_si = filt_idx[eff_sel];
            SG_LINEAGE lin;
            sg_compute_lineage(m, root_si, SG_MODE_FULL, &lin);

            for (int a = 0; a < lin.nb_ancestors; a++)
            {
                int si = lin.ancestors[a].stream_idx;
                if (si >= 0 && si < m->nb_streams)
                {
                    int d = lin.ancestors[a].depth;
                    if (d > 127)
                    {
                        d = 127;
                    }
                    local_depth[si] = (int8_t) (-d);
                }
            }
            for (int di = 0; di < lin.nb_descendants; di++)
            {
                int si = lin.descendants[di].stream_idx;
                if (si >= 0 && si < m->nb_streams)
                {
                    int dp = lin.descendants[di].depth;
                    if (dp > 127)
                    {
                        dp = 127;
                    }
                    local_depth[si] = (int8_t) dp;
                }
            }
        }
    }

    int max_rows = r.height - 4;
    int start    = lay->scroll_stream;

    regex_t re;
    int     has_re = 0;
    if (active_filter[0] != '\0')
    {
        if (regcomp(&re, active_filter, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    if (filt_n == 0 && active_filter[0] != '\0')
    {
        int row = hrow + 2;
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        char msg[128];
        snprintf(msg, sizeof(msg), "  No matching streams for '/%s/'", active_filter);
        ov_buf_printf("%s", msg);
        render_pad_spaces((int) strlen(msg), r.width);
        for (int i = 1; i < max_rows; i++)
        {
            clear_row(hrow + 2 + i, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
        if (has_re)
        {
            regfree(&re);
        }
        render_scroll_indicators(r, 0, max_rows, 0, OV_FG_STREAM);
        return;
    }

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 2 + i;
        int fi  = start + i;
        if (fi < filt_n)
        {
            int si = filt_idx[fi];
            ov_streams_render_single_row(lay, m, rel, row, i, fi, si, local_depth[si], has_re,
                                         has_re ? &re : NULL, r);
        }
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(r, lay->scroll_stream, max_rows, filt_n, OV_FG_STREAM);
    if (has_re)
    {
        regfree(&re);
    }
}

/**
 * ov_render_streams_panel - render the entire streams panel (border, header, rows, footer).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 * @rel: Pointer to relationship lookup tables.
 */
void ov_render_streams_panel(const OV_LAYOUT *lay, const OV_MODEL *m, const OV_RELATED *rel)
{
    OV_RECT r = lay->r_streams;

    int         filt_idx[OV_MAX_STREAMS];
    int         filt_n        = ov_filter_streams(lay, m, rel, filt_idx, OV_MAX_STREAMS);
    const char *active_filter = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);

    /* Panel title with prominent filter indicator */
    {
        int loop_id = (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
                          ? lay->sel_loop + 1
                          : -1;
        ov_draw_panel_border_filter(r.row, r.col, r.height, r.width, "STREAMS", OV_FG_STREAM,
                                    lay->focus == OV_FOCUS_STREAMS, 0, lay->ctrl_blink, loop_id,
                                    lay->filter_stream, lay->filter_stream_active, filt_n,
                                    m->nb_streams);
    }

    ov_streams__render_header(lay, r);
    ov_streams__render_rows(lay, m, rel, r, filt_idx, filt_n, active_filter);

    int max_rows = r.height - 4;
    ov_streams__render_footer(lay, m, r, filt_idx, filt_n, max_rows);

    ov_buf_reset_attr();
}
