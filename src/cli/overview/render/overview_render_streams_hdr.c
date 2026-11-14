// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_streams_hdr.c
 * @brief   Streams panel footer rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/**
 * ov_streams__render_footer - render total and filtered bandwidth footer.
 * @lay:      Pointer to layout structure.
 * @m:        Pointer to data model snapshot.
 * @r:        Bounding rectangle of streams panel.
 * @filt_idx: Array of stream indices matching active filter.
 * @filt_n:   Number of streams currently visible after filtering.
 * @max_rows: Maximum visible data rows in panel.
 */
void ov_streams__render_footer(const OV_LAYOUT *lay,
                               const OV_MODEL  *m,
                               OV_RECT          r,
                               const int       *filt_idx,
                               int              filt_n,
                               int              max_rows)
{
    /* Total over all streams: once per published model (it walks every stream record) */
    static uint64_t s_seq;
    static double   s_total_all_bps;
    if (m->seq == 0 || m->seq != s_seq)
    {
        s_seq           = m->seq;
        s_total_all_bps = 0.0;
        for (int i = 0; i < m->nb_streams; i++)
        {
            const OV_STREAM *s = &m->streams[i];
            if (s->update_hz > 0.1)
            {
                s_total_all_bps +=
                    s->update_hz * (double) s->nelement * dtype_bytesize(s->datatype);
            }
        }
    }
    double total_all_bps = s_total_all_bps;

    /* Total over the filtered subset */
    int    is_subset     = (filt_n < m->nb_streams);
    double total_flt_bps = is_subset ? 0.0 : total_all_bps;
    for (int i = 0; is_subset && i < filt_n; i++)
    {
        const OV_STREAM *s = &m->streams[filt_idx[i]];
        if (s->update_hz > 0.1)
        {
            total_flt_bps += s->update_hz * (double) s->nelement * dtype_bytesize(s->datatype);
        }
    }

    int brow = r.row + r.height - 1;

    /* Right side: total (always) */
    double total_all_mb = total_all_bps / (1024.0 * 1024.0);
    char   rbuf[40];
    if (total_all_mb >= 1000.0)
    {
        snprintf(rbuf, sizeof(rbuf), " %.1f GB/s ", total_all_mb / 1024.0);
    }
    else
    {
        snprintf(rbuf, sizeof(rbuf), " %.1f MB/s ", total_all_mb);
    }
    int rlen  = (int) strlen(rbuf);
    int below = filt_n - lay->panel[OV_FOCUS_STREAMS].scroll - max_rows;
    int dw    = 0;
    if (below > 0)
    {
        dw      = 3;
        int tmp = below;
        while (tmp > 0)
        {
            dw++;
            tmp /= 10;
        }
    }
    int rcol = r.col + r.width - rlen - dw - 4;
    if (rcol > r.col + 1)
    {
        ov_buf_pos(brow, rcol);
        ov_theme_fg(OV_FG_ACTIVE);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("%s", rbuf);
    }

    /* Left side: filtered (only when filter is active) */
    if (is_subset)
    {
        double flt_mb = total_flt_bps / (1024.0 * 1024.0);
        char   lbuf[40];
        if (flt_mb >= 1000.0)
        {
            snprintf(lbuf, sizeof(lbuf), " %.1f GB/s ", flt_mb / 1024.0);
        }
        else
        {
            snprintf(lbuf, sizeof(lbuf), " %.1f MB/s ", flt_mb);
        }
        int llen = (int) strlen(lbuf);
        int lcol = r.col + 2;
        if (lcol + llen < rcol)
        {
            ov_buf_pos(brow, lcol);
            ov_theme_fg(OV_FG_WARN);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", lbuf);
        }
    }
    ov_buf_reset_attr();
}
