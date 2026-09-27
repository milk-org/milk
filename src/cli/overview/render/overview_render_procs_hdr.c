// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_procs_hdr.c
 * @brief PROCS panel header and footer summary statistics rendering.
 */

#include "overview_render_internal.h"

/**
 * ov_procs__render_header - Render process panel column headers.
 * @lay: Layout configuration.
 * @r:   Bounding rectangle of panel.
 */
void ov_procs__render_header(const OV_LAYOUT *lay, OV_RECT r)
{
    int hrow    = r.row + 1;
    int hs      = lay->hscroll_proc;
    int hs_rem  = hs;
    int printed = 1;
    int avail   = r.width - 2;

    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");

    typedef struct
    {
        int         logical_col;
        const char *label;
        int         width;
        int         align_right;
    } PROC_COL_SPEC;

    PROC_COL_SPEC cols[16];
    int           num_cols = 0;

    int  sk = lay->sort_key_proc;
    int  sd = lay->sort_dir_proc;
    char c_anc[32], c_name[32], c_pid[32];
    char c_prio[32], c_stat[32], c_hz[32];
    char c_upt[32], c_duty[32], c_cpu[32];
    char c_lpcnt[32], c_mem[32];
    char c_cpu_padded[64];
    int  w_anc   = sort_col_label(c_anc, sizeof(c_anc), "A", 5, sk, sd, 3);
    int  w_name  = sort_col_label(c_name, sizeof(c_name), "NAME", 0, sk, sd, 14);
    int  w_pid   = sort_col_label(c_pid, sizeof(c_pid), "PID", 1, sk, sd, 7);
    int  w_prio  = sort_col_label(c_prio, sizeof(c_prio), "PRIO", 6, sk, sd, 4);
    int  w_stat  = sort_col_label(c_stat, sizeof(c_stat), "STAT", 2, sk, sd, 5);
    int  w_hz    = sort_col_label(c_hz, sizeof(c_hz), "Hz", 3, sk, sd, 6);
    int  w_upt   = sort_col_label(c_upt, sizeof(c_upt), "UPTIME", 7, sk, sd, 6);
    int  w_duty  = sort_col_label(c_duty, sizeof(c_duty), "DUTY", 10, sk, sd, 5);
    int  w_cpu   = sort_col_label(c_cpu, sizeof(c_cpu), "CPU%", 8, sk, sd, 6);
    int  w_lpcnt = sort_col_label(c_lpcnt, sizeof(c_lpcnt), "LOOPCNT", 9, sk, sd, 10);
    int  w_mem   = sort_col_label(c_mem, sizeof(c_mem), "MEM", 4, sk, sd, 5);

    cols[num_cols++] = (PROC_COL_SPEC) { 0, c_anc, w_anc, 0 };
    cols[num_cols++] = (PROC_COL_SPEC) { 1, c_name, w_name, 0 };
    cols[num_cols++] = (PROC_COL_SPEC) { 2, c_pid, w_pid, 1 };
    cols[num_cols++] = (PROC_COL_SPEC) { 3, c_prio, w_prio, 1 };
    cols[num_cols++] = (PROC_COL_SPEC) { 4, c_stat, w_stat, 1 };
    cols[num_cols++] = (PROC_COL_SPEC) { 5, c_hz, w_hz, 1 };
    cols[num_cols++] = (PROC_COL_SPEC) { 6, c_upt, w_upt, 1 };
    if (!lay->compact_mode)
    {
        cols[num_cols++] = (PROC_COL_SPEC) { 7, "TRG", 3, 1 };
        cols[num_cols++] = (PROC_COL_SPEC) { 8, "trig-strm", 10, 0 };
        cols[num_cols++] = (PROC_COL_SPEC) { 9, "exec", 8, 1 };
        cols[num_cols++] = (PROC_COL_SPEC) { 10, c_duty, w_duty, 1 };
    }

    snprintf(c_cpu_padded, sizeof(c_cpu_padded), "  %s  ", c_cpu);
    cols[num_cols++] = (PROC_COL_SPEC) { 11, c_cpu_padded, w_cpu + 4, 1 };

    cols[num_cols++] = (PROC_COL_SPEC) { 12, c_lpcnt, w_lpcnt, 1 };
    cols[num_cols++] = (PROC_COL_SPEC) { 13, c_mem, w_mem, 1 };
    if (!lay->compact_mode)
    {
        cols[num_cols++] = (PROC_COL_SPEC) { 14, "MISSED", 10, 1 };
    }
    cols[num_cols++] = (PROC_COL_SPEC) { 15, "MSG", 200, 0 };

    for (int c = 0; c < num_cols; c++)
    {
        if (c > 0)
        {
            int prev_logical   = cols[c - 1].logical_col;
            int prev_collapsed = (lay->col_collapsed_proc & (1U << prev_logical)) != 0;
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
        char cell_buf[256];
        if (cols[c].align_right)
        {
            snprintf(cell_buf, sizeof(cell_buf), "%*s", cols[c].width, cols[c].label);
        }
        else
        {
            snprintf(cell_buf, sizeof(cell_buf), "%-*s", cols[c].width, cols[c].label);
        }

        ov_render_cell(cols[c].logical_col, c, OV_FG_PROC_HDR, OV_BG_HEADER, cell_buf, &hs_rem,
                       &printed, avail, lay->highlight_col_proc, lay->col_collapsed_proc);
    }
    render_pad_spaces(printed, r.width);

    /* Separator between header and data rows */
    render_separator(hrow + 1, r.col + 1, r.width - 2, OV_FG_PROC_HDR);
}

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
    int     tot_run = 0;
    double  tot_cpu = 0.0;
    int64_t tot_mem = 0;
    for (int j = 0; j < m->nb_procs; j++)
    {
        const OV_PROC *p = &m->procs[j];
        if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE)
        {
            tot_run++;
        }
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
        if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE)
        {
            flt_run++;
        }
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
    snprintf(rbuf, sizeof(rbuf), " %d RUN \u2502 %.0f%% CPU \u2502 %s ", tot_run, (double) tot_cpu,
             tmem);
    int rlen  = (int) strlen(rbuf);
    int below = filt_n - lay->scroll_proc - (r.height - 3);
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
