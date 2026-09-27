// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_fps_hdr.c
 * @brief FPS panel header and footer summary statistics rendering.
 */

#include "overview_render_internal.h"

/**
 * ov_fps__render_header - render column headers for FPS panel.
 * @lay: Pointer to layout structure.
 * @r:   Bounding rectangle of FPS panel.
 */
void ov_fps__render_header(
    const OV_LAYOUT *lay,
    OV_RECT          r)
{
    int hrow = r.row + 1;
    int hs   = lay->hscroll_fps;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");

    typedef struct
    {
        int         logical_col;
        const char *label;
        int         width;
        int         align_right;
    } FPS_COL_SPEC;

    FPS_COL_SPEC cols[8];
    int          num_cols = 0;

    {
        int         sk = lay->sort_key_fps;
        int         sd = lay->sort_dir_fps;
        static char c_anc[32], c_name[32];
        static char c_c[32], c_r[32], c_mem[32];
        static char c_tmx[32], c_str[32];
        int         w_anc  = sort_col_label(c_anc, sizeof(c_anc), "A", 3, sk, sd, 3);
        int         w_name = sort_col_label(c_name, sizeof(c_name), "NAME", 0, sk, sd, 18);
        int         w_tmx  = sort_col_label(c_tmx, sizeof(c_tmx), "TMX", 5, sk, sd, 3);
        int         w_c    = sort_col_label(c_c, sizeof(c_c), "CPID", 1, sk, sd, 7);
        int         w_r    = sort_col_label(c_r, sizeof(c_r), "RPID", 4, sk, sd, 7);
        int         w_str  = sort_col_label(c_str, sizeof(c_str), "STR", 6, sk, sd, 3);
        int         w_mem  = sort_col_label(c_mem, sizeof(c_mem), "MEM", 2, sk, sd, 5);
        int         desc_w = (lay->view == OV_VIEW_FPS) ? 30 : 20;

        cols[num_cols++] = (FPS_COL_SPEC) { 0, c_anc, w_anc, 0 };
        cols[num_cols++] = (FPS_COL_SPEC) { 1, c_name, w_name, 0 };
        cols[num_cols++] = (FPS_COL_SPEC) { 2, c_tmx, w_tmx, 1 };
        cols[num_cols++] = (FPS_COL_SPEC) { 3, c_c, w_c, 1 };
        cols[num_cols++] = (FPS_COL_SPEC) { 4, c_r, w_r, 1 };
        cols[num_cols++] = (FPS_COL_SPEC) { 5, c_str, w_str, 1 };
        cols[num_cols++] = (FPS_COL_SPEC) { 6, c_mem, w_mem, 1 };
        if (!lay->compact_mode)
        {
            cols[num_cols++] = (FPS_COL_SPEC) { 7, "DESCRIPTION", desc_w, 0 };
        }
    }

    int hs_rem  = hs;
    int printed = 1;
    int avail   = r.width - 2;

    for (int c = 0; c < num_cols; c++)
    {
        if (c > 0)
        {
            int prev_logical   = cols[c - 1].logical_col;
            int prev_collapsed = (lay->col_collapsed_fps & (1U << prev_logical)) != 0;
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

        ov_render_cell(cols[c].logical_col, c, OV_FG_FPS_HDR, OV_BG_HEADER, cell_buf, &hs_rem,
                       &printed, avail, lay->highlight_col_fps, lay->col_collapsed_fps);
    }
    render_pad_spaces(printed, r.width);

    /* Separator between header and data rows */
    render_separator(hrow + 1, r.col + 1, r.width - 2, OV_FG_FPS_HDR);
}

/**
 * ov_fps__render_footer - render totals and active counts on FPS panel footer.
 * @lay:      Pointer to layout structure.
 * @m:        Pointer to data model snapshot.
 * @r:        Bounding rectangle of FPS panel.
 * @fidx:     Array of visible FPS indices.
 * @filt_n:   Count of matching FPS modules.
 * @max_rows: Maximum visible rows in panel.
 */
void ov_fps__render_footer(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    OV_RECT          r,
    const int       *fidx,
    int              filt_n,
    int              max_rows)
{
    /* Totals over ALL FPS */
    int     tot_conf  = 0;
    int     tot_run   = 0;
    int     tot_crash = 0;
    int     tot_idle  = 0;
    int64_t tot_mem   = 0;
    for (int j = 0; j < m->nb_fps; j++)
    {
        const OV_FPS *f = &m->fps[j];
        if (f->conf_alive)
        {
            tot_conf++;
        }
        if (f->run_alive)
        {
            tot_run++;
        }
        tot_mem += f->mem_rss_kb;
        if (f->runpid > 0 && !f->run_alive)
        {
            tot_crash++;
        }
        if (f->conf_alive && !f->run_alive && f->runpid <= 0)
        {
            tot_idle++;
        }
    }

    /* Totals over filtered subset */
    int     flt_conf = 0;
    int     flt_run  = 0;
    int64_t flt_mem  = 0;
    for (int j = 0; j < filt_n; j++)
    {
        const OV_FPS *f = &m->fps[fidx[j]];
        if (f->conf_alive)
        {
            flt_conf++;
        }
        if (f->run_alive)
        {
            flt_run++;
        }
        flt_mem += f->mem_rss_kb;
    }

    int brow      = r.row + r.height - 1;
    int is_subset = (filt_n < m->nb_fps);

    /* Right side: total stats (always) */
    char tmem[16];
    format_mem_kb(tmem, sizeof(tmem), tot_mem);
    char rbuf[120];
    int  roff = snprintf(rbuf, sizeof(rbuf), " %d conf \u2502 %d run", tot_conf, tot_run);
    if (tot_crash > 0)
    {
        roff += snprintf(rbuf + roff, sizeof(rbuf) - (size_t) roff, " \u2502 %d crash", tot_crash);
    }
    if (tot_idle > 0)
    {
        roff += snprintf(rbuf + roff, sizeof(rbuf) - (size_t) roff, " \u2502 %d idle", tot_idle);
    }
    snprintf(rbuf + roff, sizeof(rbuf) - (size_t) roff, " \u2502 %s ", tmem);
    int rlen  = (int) strlen(rbuf);
    int below = filt_n - lay->scroll_fps - max_rows;
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
        ov_theme_fg(tot_run > 0 ? OV_FG_ACTIVE : OV_FG_DIM);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("%s", rbuf);
    }

    /* Left side: filtered stats */
    if (is_subset)
    {
        char fmem[16];
        format_mem_kb(fmem, sizeof(fmem), flt_mem);
        char lbuf[80];
        snprintf(lbuf, sizeof(lbuf), " %d conf \u2502 %d run \u2502 %s ", flt_conf, flt_run, fmem);
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
