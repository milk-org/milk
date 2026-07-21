// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_procs.c
 * @brief   PROCS panel rendering for milk-CTRL
 *
 * Split from overview_render.c for navigability.
 */

#include "overview_render_internal.h"

/* render_trigmode_label and format_mem_kb are
 * now static inline in overview_render_internal.h */


static int ov_procs__filter(const OV_LAYOUT  *lay,
                            const OV_MODEL   *m,
                            const OV_RELATED *rel,
                            int              *filt_idx,
                            int              *has_re,
                            regex_t          *re)
{
    const char *names[OV_MAX_PROCS];
    for (int i = 0; i < m->nb_procs; i++)
    {
        names[i] = m->procs[i].name;
    }
    int filt_n = ov_filter_build(lay->filter_proc, names, m->nb_procs, filt_idx, OV_MAX_PROCS);

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_PROCS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->procs, filt_idx[i]))
            {
                filt_idx[new_filt_n++] = filt_idx[i];
            }
        }
        filt_n = new_filt_n;
    }

    *has_re = 0;
    if (lay->filter_proc[0] != '\0')
    {
        if (regcomp(re, lay->filter_proc, REG_EXTENDED | REG_ICASE) == 0)
        {
            *has_re = 1;
        }
    }
    return filt_n;
}

/**
 * @brief Render the process panel column headers.
 */
static void ov_procs__render_header(const OV_LAYOUT *lay, int hrow, int hs, OV_RECT r)
{
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

    cols[num_cols++] = (PROC_COL_SPEC) { 0, c_anc, 4, 0 };
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
 * @brief Render process rows in the overview.
 */
static void ov_procs__render_rows(const OV_LAYOUT  *lay,
                                  const OV_MODEL   *m,
                                  const OV_RELATED *rel,
                                  int               hrow,
                                  int               hs,
                                  OV_RECT           r,
                                  const int        *filt_idx,
                                  int               filt_n,
                                  int               has_re,
                                  const regex_t    *re)
{
    int8_t local_depth[OV_MAX_PROCS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->freeze && lay->freeze_focus == OV_FOCUS_PROCS && lay->freeze_sel_proc >= 0 &&
            lay->freeze_sel_proc < filt_n)
        {
            eff_sel = lay->freeze_sel_proc;
        }
        else if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0 && lay->sel_proc < filt_n)
        {
            eff_sel = lay->sel_proc;
        }
        if (eff_sel >= 0)
        {
            int root_pi   = filt_idx[eff_sel];
            int root_node = m->procs[root_pi].node_idx;
            if (root_node >= 0)
            {
                int8_t node_depths[OV_MAX_NODES];
                sg_compute_node_depths(m, root_node, SG_MODE_FULL, node_depths);
                for (int pi = 0; pi < m->nb_procs; pi++)
                {
                    int n = m->procs[pi].node_idx;
                    if (n >= 0 && node_depths[n] != 127)
                    {
                        local_depth[pi] = node_depths[n];
                    }
                }
            }
        }
    }

    int max_rows = r.height - 4;
    int start    = lay->scroll_proc;

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 2 + i;
        int fi  = start + i;
        if (fi < filt_n)
        {
            int            pi     = filt_idx[fi];
            const OV_PROC *p      = &m->procs[pi];
            int            is_sel = (fi == lay->sel_proc &&
                          (lay->focus == OV_FOCUS_PROCS || lay->focus == OV_FOCUS_GRAPH));
            int            is_frozen =
                (lay->freeze && lay->freeze_focus == OV_FOCUS_PROCS && fi == lay->freeze_sel_proc);
            ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int        has_rel   = (rel != NULL && bget(rel->procs, pi));
            int        is_rel   = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_PROCS && has_rel);
            int        is_write = (has_rel && rel != NULL && bget(rel->proc_writes, pi));
            ov_rgb_t   row_bg   = OV_BG_PANEL;
            if (is_sel)
            {
                row_bg = OV_BG_SELECTED;
            }
            else if (is_frozen)
            {
                row_bg = OV_BG_FROZEN;
            }
            else if (is_rel)
            {
                row_bg = OV_BG_RELATED;
            }
            else if (p->stale_count >= 3)
            {
                row_bg = OV_BG_STALE;
            }
            else if (p->is_new > 0)
            {
                row_bg = OV_BG_NEW_ITEM;
            }
            else if (lay->mouse_hover && lay->hover_global_proc == pi)
            {
                row_bg = OV_BG_HOVER;
            }
            row_bg = zebra_bg(row_bg, i);

            /* Per-field colored cell rendering */
            int hs_rem  = lay->hscroll_proc;
            int printed = 1;
            int avail   = r.width - 2;

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);
            ov_buf_printf(" ");

#define PROC_FIELD_WITH_COL(vcol_idx, logical_idx, color, bg_color, fmt, ...)                     \
    do                                                                                            \
    {                                                                                             \
        char _fb[256];                                                                            \
        int  _fl = snprintf(_fb, sizeof(_fb), fmt, ##__VA_ARGS__);                                \
        ov_render_cell(logical_idx, vcol_idx, (color), (bg_color), _fb, &hs_rem, &printed, avail, \
                       lay->highlight_col_proc, lay->col_collapsed_proc);                         \
    } while (0)

#define PROC_FIELD(color, fmt, ...)                                                   \
    do                                                                                \
    {                                                                                 \
        int logical_idx = ov_get_logical_col_proc(vcol, lay->compact_mode);           \
        PROC_FIELD_WITH_COL(vcol, logical_idx, (color), cell_bg, fmt, ##__VA_ARGS__); \
        vcol++;                                                                       \
    } while (0)

            int      vcol    = 1;
            ov_rgb_t cell_bg = row_bg;

            /* Calculate Status Color */
            ov_rgb_t sc;
            switch (p->loopstat)
            {
            case PROCESSINFO_LOOPSTAT_INIT:
                sc = OV_FG_DIM;
                break;
            case PROCESSINFO_LOOPSTAT_ACTIVE:
                sc = OV_FG_ACTIVE;
                break;
            case PROCESSINFO_LOOPSTAT_PAUSE:
                sc = OV_FG_WARN;
                break;
            case PROCESSINFO_LOOPSTAT_STOP:
                sc = OV_FG_ZOMBIE;
                break;
            case PROCESSINFO_LOOPSTAT_ERROR:
                sc = OV_FG_ERROR;
                break;
            case PROCESSINFO_LOOPSTAT_SPIN:
                sc = OV_FG_WARN;
                break;
            case PROCESSINFO_LOOPSTAT_CRASHED:
                sc = OV_FG_ERROR;
                break;
            default:
                sc = OV_FG_DIM;
                break;
            }

            /* Ancestry column — rendered raw */
            int8_t sdepth      = local_depth[pi];
            char   anc_str[64] = "";
            if (sdepth != 0 && !is_sel && !is_frozen)
            {
                int abs_d = sdepth < 0 ? -sdepth : sdepth;
                if (abs_d > 99)
                {
                    abs_d = 99;
                }
                if (sdepth < 0)
                {
                    snprintf(anc_str, sizeof(anc_str),
                             abs_d < 10 ? "\xe2\x97\x80%d  " : "\xe2\x97\x80%d ", abs_d);
                }
                else
                {
                    snprintf(anc_str, sizeof(anc_str),
                             abs_d < 10 ? "%d\xe2\x96\xb6  " : "%d\xe2\x96\xb6 ", abs_d);
                }
            }
            else if (eff_focus == OV_FOCUS_STREAMS && has_rel)
            {
                snprintf(anc_str, sizeof(anc_str),
                         is_write ? "\xe2\x96\xb6   " : "\xe2\x97\x80   ");
            }
            else if (is_sel || is_frozen)
            {
                snprintf(anc_str, sizeof(anc_str), "\xe2\x97\x8f   ");
            }
            else
            {
                snprintf(anc_str, sizeof(anc_str), "    ");
            }

            ov_render_cell(0, 0, OV_FG_WARN, row_bg, anc_str, &hs_rem, &printed, avail,
                           lay->highlight_col_proc, lay->col_collapsed_proc);

            /* Name */
            {
                char       name_cell[128];
                regmatch_t pm[1];
                if (has_re && regexec(re, p->name, 1, pm, 0) == 0)
                {
                    int b_len = pm[0].rm_so;
                    if (b_len > 14)
                    {
                        b_len = 14;
                    }
                    int m_len = pm[0].rm_eo - pm[0].rm_so;
                    if (b_len + m_len > 14)
                    {
                        m_len = 14 - b_len;
                    }
                    int tail_len = 14 - (b_len + m_len);
                    if (tail_len < 0)
                    {
                        tail_len = 0;
                    }
                    snprintf(name_cell, sizeof(name_cell), "%.*s\x01%.*s\x02%.*s ", b_len, p->name,
                             m_len, p->name + b_len, tail_len, p->name + b_len + m_len);
                }
                else
                {
                    snprintf(name_cell, sizeof(name_cell), "%-14.14s ", p->name);
                }
                int logical_idx = ov_get_logical_col_proc(vcol, lay->compact_mode);
                PROC_FIELD_WITH_COL(vcol, logical_idx, OV_FG_PROC, cell_bg, "%s", name_cell);
                vcol++;
            }

            /* PID */
            PROC_FIELD(ov_pid_color(p->PID), "%7d ", (int) p->PID);

            /* PRIO */
            if (p->rt_priority > -1)
            {
                PROC_FIELD(OV_FG_WARN, "%4d ", p->rt_priority);
            }
            else
            {
                PROC_FIELD(OV_FG_DIM, "   - ");
            }

            /* Status */
            const char *sl;
            switch (p->loopstat)
            {
            case PROCESSINFO_LOOPSTAT_INIT:
                sl = "INIT";
                break;
            case PROCESSINFO_LOOPSTAT_ACTIVE:
                sl = " RUN";
                break;
            case PROCESSINFO_LOOPSTAT_PAUSE:
                sl = "PAUS";
                break;
            case PROCESSINFO_LOOPSTAT_STOP:
                sl = "STOP";
                break;
            case PROCESSINFO_LOOPSTAT_ERROR:
                sl = "ERR!";
                break;
            case PROCESSINFO_LOOPSTAT_SPIN:
                sl = "SPIN";
                break;
            case PROCESSINFO_LOOPSTAT_CRASHED:
                sl = "CRSH";
                break;
            default:
                sl = " ?? ";
                break;
            }
            PROC_FIELD(sc, "%4s ", sl);

            /* Hz */
            if (p->loop_hz > 0.1)
            {
                ov_rgb_t hzc = ov_rgb_lerp(OV_FG_DIM, OV_FG_ACTIVE, (float) (p->loop_hz / 5000.0));
                PROC_FIELD(hzc, "%6.1f ", p->loop_hz);
            }
            else
            {
                PROC_FIELD(OV_FG_DIM, "     - ");
            }

            /* UPTIME */
            {
                char uptstr[12] = "-";
                if (p->start_time_sec > 0)
                {
                    format_uptime(uptstr, sizeof(uptstr), p->start_time_sec);
                }
                PROC_FIELD(OV_FG_TEXT, "%6s ", uptstr);
            }

            /* Compact mode gated fields */
            if (!lay->compact_mode)
            {
                /* TRG */
                PROC_FIELD(OV_FG_CONN, "%3s ", render_trigmode_label(p->triggermode));

                /* trig-strm */
                if (p->trigstreamname[0] != '\0' && p->triggermode > 0)
                {
                    PROC_FIELD(OV_FG_STREAM, "%-10.10s ", p->trigstreamname);
                }
                else
                {
                    PROC_FIELD(OV_FG_DIM, "%-10s ", "-");
                }

                /* exec + arrow */
                {
                    char     exec_str[32];
                    ov_rgb_t ec = OV_FG_DIM;
                    if (p->MeasureTiming && p->dtmedian_exec_ns > 0)
                    {
                        double exec_ms = 1.0e-6 * (double) p->dtmedian_exec_ns;
                        if (has_rel)
                        {
                            snprintf(exec_str, sizeof(exec_str), "%7.3f%s", exec_ms,
                                     is_write ? " W" : " R");
                        }
                        else
                        {
                            snprintf(exec_str, sizeof(exec_str), "%7.3f  ", exec_ms);
                        }
                        if (exec_ms < 1.0)
                        {
                            ec = OV_FG_ACTIVE;
                        }
                        else if (exec_ms < 10.0)
                        {
                            ec = OV_FG_WARN;
                        }
                        else
                        {
                            ec = OV_FG_ERROR;
                        }
                    }
                    else
                    {
                        if (has_rel)
                        {
                            snprintf(exec_str, sizeof(exec_str), "      -%s",
                                     is_write ? " W" : " R");
                        }
                        else
                        {
                            snprintf(exec_str, sizeof(exec_str), "      -  ");
                        }
                    }
                    PROC_FIELD(ec, "%s", exec_str);
                }

                /* DUTY */
                if (p->MeasureTiming && p->dtmedian_exec_ns > 0 && p->dtmedian_iter_ns > 0)
                {
                    double duty =
                        100.0 * (double) p->dtmedian_exec_ns / (double) p->dtmedian_iter_ns;
                    ov_rgb_t dc = OV_FG_ACTIVE;
                    if (duty > 90.0)
                    {
                        dc = OV_FG_ERROR;
                    }
                    else if (duty > 50.0)
                    {
                        dc = OV_FG_WARN;
                    }
                    PROC_FIELD(dc, " %4.0f%%", duty);
                }
                else
                {
                    PROC_FIELD(OV_FG_DIM, "     -");
                }
            }

            /* CPU% */
            PROC_FIELD(OV_FG_TEXT, "  %5.1f%%  ", p->cpu_used);

            /* LOOPCNT */
            PROC_FIELD(p->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM, "%10" PRId64 " ",
                       (int64_t) p->loopcnt);

            /* MEM */
            {
                char memstr[16];
                format_mem_kb(memstr, sizeof(memstr), p->mem_rss_kb);
                PROC_FIELD(OV_FG_TEXT, "%5s ", memstr);
            }

            /* MISSED */
            if (!lay->compact_mode)
            {
                PROC_FIELD(p->triggermissed_cumul > 0 ? OV_FG_WARN : OV_FG_DIM, "%10" PRIu64 " ",
                           (uint64_t) p->triggermissed_cumul);
            }

            /* MSG */
            PROC_FIELD(OV_FG_TEXT, "%-200.200s", p->statusmsg);

#undef PROC_FIELD_WITH_COL
#undef PROC_FIELD

            if (lay->mouse_hover && lay->hover_view == OV_FOCUS_PROCS && lay->hover_idx == fi)
            {
                char loc_upt[12] = "-";
                if (p->start_time_sec > 0)
                {
                    format_uptime(loc_upt, sizeof(loc_upt), p->start_time_sec);
                }
                snprintf((char *) lay->hover_tooltip, sizeof(lay->hover_tooltip),
                         "PID: %d | Up: %s | CPU: %4.1f%% | Exec: %zu ns", (int) p->PID, loc_upt,
                         p->cpu_used, (size_t) p->dtmedian_exec_ns);

                int btn_w = 8; /* " [Kill] " */
                int rem   = avail - printed;
                if (rem >= btn_w)
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
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(r, lay->scroll_proc, max_rows, filt_n, OV_FG_PROC);

    /* ---- Footer stats on bottom border ---- */
    {
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
                p->loopstat == PROCESSINFO_LOOPSTAT_SPIN ||
                p->loopstat == PROCESSINFO_LOOPSTAT_INIT)
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
            const OV_PROC *p = &m->procs[filt_idx[j]];
            if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE)
            {
                flt_run++;
            }
            if (p->loopstat == PROCESSINFO_LOOPSTAT_ACTIVE ||
                p->loopstat == PROCESSINFO_LOOPSTAT_SPIN ||
                p->loopstat == PROCESSINFO_LOOPSTAT_INIT)
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
        snprintf(rbuf, sizeof(rbuf), " %d RUN \u2502 %.0f%% CPU \u2502 %s ", tot_run,
                 (double) tot_cpu, tmem);
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

        /* Left side: filtered stats (only if
         * filter is active) */
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
    }

    ov_buf_reset_attr();
}

/**
 * @brief Render the processes panel in the overview.
 */
void ov_render_procs_panel(const OV_LAYOUT *lay, const OV_MODEL *m, const OV_RELATED *rel)
{
    OV_RECT r = lay->r_procs;

    int     filt_idx[OV_MAX_PROCS];
    int     has_re;
    regex_t re;
    int     filt_n = ov_procs__filter(lay, m, rel, filt_idx, &has_re, &re);

    char title[80];
    if (lay->filter_proc[0] != '\0')
    {
        snprintf(title, sizeof(title), "PROCESSINFO /%s/", lay->filter_proc);
    }
    else
    {
        snprintf(title, sizeof(title), "PROCESSINFO");
    }
    ov_draw_panel_border(r.row, r.col, r.height, r.width, title, OV_FG_PROC,
                         lay->focus == OV_FOCUS_PROCS, 0);

    int hrow = r.row + 1;
    int hs   = lay->hscroll_proc;

    ov_procs__render_header(lay, hrow, hs, r);
    ov_procs__render_rows(lay, m, rel, hrow, hs, r, filt_idx, filt_n, has_re, &re);

    if (has_re)
    {
        regfree(&re);
    }
}
