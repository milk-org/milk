// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_procs_hdr.c
 * @brief PROCS panel footer summary statistics rendering.
 */

#include "overview_render_internal.h"

/**
 * ov_procs__render_footer - render process CPU and memory statistics footer.
 * @lay:      Pointer to layout structure.
 * @m:        Pointer to data model snapshot.
 * @r:        Bounding rectangle of procs panel.
 * @pidx:     Array of indices matching active filter.
 * @filt_n:   Count of matching processes.
 * @max_rows: Maximum visible rows in panel.
 */
void ov_procs__render_footer(const OV_LAYOUT *lay,
                             const OV_MODEL  *m,
                             OV_RECT          r,
                             const int       *pidx,
                             int              filt_n,
                             int              max_rows)
{
    (void) max_rows;
    /* Compute totals over ALL procs */
    int     tot_run = 0, tot_stal = 0;
    double  tot_cpu = 0.0;
    int64_t tot_mem = 0;
    for (int j = 0; j < m->nb_procs; j++)
    {
        const OV_PROC *p = &m->procs[j];
        tot_run += (ov_proc_state(p) == OV_ST_RUN);
        tot_stal += (ov_proc_state(p) == OV_ST_STAL);
        if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE ||
            p->loopstat == PROCESSINFO_LOOPSTAT_SPIN || p->loopstat == PROCESSINFO_LOOPSTAT_INIT)
        {
            tot_cpu += p->cpu_used;
        }
        tot_mem += p->mem_rss_kb;
    }

    /* Compute totals over filtered subset */
    int     flt_run = 0;
    double  flt_cpu = 0.0;
    int64_t flt_mem = 0;
    for (int j = 0; j < filt_n; j++)
    {
        const OV_PROC *p = &m->procs[pidx[j]];
        flt_run += (ov_proc_state(p) == OV_ST_RUN);
        if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE ||
            p->loopstat == PROCESSINFO_LOOPSTAT_SPIN || p->loopstat == PROCESSINFO_LOOPSTAT_INIT)
        {
            flt_cpu += p->cpu_used;
        }
        flt_mem += p->mem_rss_kb;
    }

    int brow      = r.row + r.height - 1;
    int is_subset = (filt_n < m->nb_procs);

    /* Right side: total stats (always) */
    char tmem[16];
    format_mem_kb(tmem, sizeof(tmem), tot_mem);
    char rbuf[80];
    char stal[24] = "";
    if (tot_stal > 0)
    {
        snprintf(stal, sizeof(stal), " %d STAL", tot_stal);
    }
    snprintf(rbuf, sizeof(rbuf), " %d RUN%s \u2502 %.0f%% CPU \u2502 %s ", tot_run, stal,
             (double) tot_cpu, tmem);
    int rlen  = (int) strlen(rbuf);
    int below = filt_n - lay->panel[OV_FOCUS_PROCS].scroll - (r.height - 3);
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

    /* Left side: filtered stats */
    if (is_subset)
    {
        char fmem[16];
        format_mem_kb(fmem, sizeof(fmem), flt_mem);
        char lbuf[80];
        snprintf(lbuf, sizeof(lbuf), " %d RUN \u2502 %.0f%% CPU \u2502 %s ", flt_run,
                 (double) flt_cpu, fmem);
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
