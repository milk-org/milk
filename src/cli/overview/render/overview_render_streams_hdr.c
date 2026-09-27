// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_streams_hdr.c
 * @brief   Streams panel header and footer rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/**
 * ov_streams__render_header - render column headers for streams panel.
 * @lay: Pointer to layout structure.
 * @r:   Bounding rectangle of streams panel.
 */
void ov_streams__render_header(const OV_LAYOUT *lay, OV_RECT r)
{
    int hrow = r.row + 1;
    int hs   = lay->hscroll_stream;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");

    typedef struct
    {
        int         logical_col;
        const char *label;
        int         width;
        int         align_right;
    } STRM_COL_SPEC;

    STRM_COL_SPEC cols[12];
    int           num_cols = 0;

    {
        int         sk = lay->sort_key_stream;
        int         sd = lay->sort_dir_stream;
        static char c_anc[32], c_name[32], c_typ[32];
        static char c_size[32];
        static char c_hz[32], c_mbps[32], c_ino[32], c_cnt[32];
        int         w_anc  = sort_col_label(c_anc, sizeof(c_anc), "A", 7, sk, sd, 3);
        int         w_name = sort_col_label(c_name, sizeof(c_name), "NAME", 0, sk, sd, 14);
        int         w_typ  = sort_col_label(c_typ, sizeof(c_typ), "TYP", 1, sk, sd, 4);
        int         w_size = sort_col_label(c_size, sizeof(c_size), "SIZE", 2, sk, sd, 11);
        int         w_hz   = sort_col_label(c_hz, sizeof(c_hz), "Hz", 3, sk, sd, 6);
        int         w_mbps = sort_col_label(c_mbps, sizeof(c_mbps), "MB/s", 4, sk, sd, 7);
        int         w_ino  = sort_col_label(c_ino, sizeof(c_ino), "INODE", 5, sk, sd, 10);
        int         w_cnt  = sort_col_label(c_cnt, sizeof(c_cnt), "COUNT", 6, sk, sd, 10);

        cols[num_cols++] = (STRM_COL_SPEC) { 0, c_anc, w_anc, 0 };
        cols[num_cols++] = (STRM_COL_SPEC) { 1, c_name, w_name, 0 };
        cols[num_cols++] = (STRM_COL_SPEC) { 2, c_typ, w_typ, 1 };
        cols[num_cols++] = (STRM_COL_SPEC) { 3, c_size, w_size, 1 };
        cols[num_cols++] = (STRM_COL_SPEC) { 4, c_hz, w_hz, 1 };
        cols[num_cols++] = (STRM_COL_SPEC) { 5, c_mbps, w_mbps, 1 };
        if (!lay->compact_mode)
        {
            cols[num_cols++] = (STRM_COL_SPEC) { 6, c_ino, w_ino, 1 };
        }
        cols[num_cols++] = (STRM_COL_SPEC) { 7, "OWNER", 7, 1 };
        if (!lay->compact_mode)
        {
            cols[num_cols++] = (STRM_COL_SPEC) { 8, c_cnt, w_cnt, 1 };
            cols[num_cols++] = (STRM_COL_SPEC) { 9, "SEMS", 10, 1 };
        }
        cols[num_cols++] = (STRM_COL_SPEC) { 10, "WPID", 7, 1 };
        cols[num_cols++] = (STRM_COL_SPEC) { 11, "RPID", 7, 0 };
    }

    int hs_rem  = hs;
    int printed = 1;
    int avail   = r.width - 2;

    for (int c = 0; c < num_cols; c++)
    {
        if (c > 0)
        {
            int prev_logical   = cols[c - 1].logical_col;
            int prev_collapsed = (lay->col_collapsed_stream & (1U << prev_logical)) != 0;
            if (!prev_collapsed)
            {
                ov_theme_bg(OV_BG_HEADER);
                if (hs_rem > 0)
                {
                    hs_rem--;
                }
                else if (printed < avail)
                {
                    ov_buf_printf(" ");
                    printed++;
                }
            }
        }

        /* Format the header cell string with correct alignment */
        char cell_buf[128];
        if (cols[c].align_right)
        {
            snprintf(cell_buf, sizeof(cell_buf), "%*s", cols[c].width, cols[c].label);
        }
        else
        {
            snprintf(cell_buf, sizeof(cell_buf), "%-*s", cols[c].width, cols[c].label);
        }

        ov_render_cell(cols[c].logical_col, c, OV_FG_STREAM_HDR, OV_BG_HEADER, cell_buf, &hs_rem,
                       &printed, avail, lay->highlight_col_stream, lay->col_collapsed_stream);
    }
    render_pad_spaces(printed, r.width);

    /* Separator between header and data rows */
    render_separator(hrow + 1, r.col + 1, r.width - 2, OV_FG_STREAM_HDR);
}

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
    /* Totals over ALL streams */
    double total_all_bps = 0.0;
    for (int i = 0; i < m->nb_streams; i++)
    {
        const OV_STREAM *s = &m->streams[i];
        if (s->update_hz > 0.1)
        {
            total_all_bps += s->update_hz * (double) s->nelement * dtype_bytesize(s->datatype);
        }
    }

    /* Totals over filtered subset */
    double total_flt_bps = 0.0;
    for (int i = 0; i < filt_n; i++)
    {
        int              si = filt_idx[i];
        const OV_STREAM *s  = &m->streams[si];
        if (s->update_hz > 0.1)
        {
            total_flt_bps += s->update_hz * (double) s->nelement * dtype_bytesize(s->datatype);
        }
    }

    int brow      = r.row + r.height - 1;
    int is_subset = (filt_n < m->nb_streams);

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
    int below = filt_n - lay->scroll_stream - max_rows;
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
