// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_fps_hdr.c
 * @brief FPS panel footer summary statistics rendering.
 */

#include "overview_render_internal.h"

/**
 * ov_fps__render_footer - render totals and active counts on FPS panel footer.
 * @lay:      Pointer to layout structure.
 * @m:        Pointer to data model snapshot.
 * @r:        Bounding rectangle of FPS panel.
 * @fidx:     Array of visible FPS indices.
 * @filt_n:   Count of matching FPS modules.
 * @max_rows: Maximum visible rows in panel.
 */
void ov_fps__render_footer(const OV_LAYOUT *lay,
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
    int below = filt_n - lay->panel[OV_FOCUS_FPS].scroll - max_rows;
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
