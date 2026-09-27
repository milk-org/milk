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
void ov_procs_render_single_row(
    const OV_LAYOUT  *lay,
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
    const OV_PROC *p = &m->procs[pi];
            int is_sel        = (fi == lay->sel_proc &&
                                 (lay->focus == OV_FOCUS_PROCS || lay->focus == OV_FOCUS_GRAPH));
            int is_frozen =
                (lay->freeze && lay->freeze_focus == OV_FOCUS_PROCS && fi == lay->freeze_sel_proc);
            ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int        has_rel   = (rel != NULL && bget(rel->procs, pi));
            int        is_rel   = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_PROCS && has_rel);
            int        is_write = (has_rel && rel != NULL && bget(rel->proc_writes, pi));
            int        is_loop_member = 0;
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
            else if (p->nb_loops > 1)
            {
                snprintf(anc_str, sizeof(anc_str), "\xe2\xae\x82   "); /* ⮂ */
            }
            else if (p->nb_loops == 1)
            {
                snprintf(anc_str, sizeof(anc_str), "\xe2\x86\xba   "); /* ↺ */
            }
            else
            {
                snprintf(anc_str, sizeof(anc_str), "    ");
            }

            ov_rgb_t anc_color = (p->nb_loops > 1) ? OV_FG_LOOP_SHARED
                                                   : ((p->nb_loops == 1) ? OV_FG_LOOP : OV_FG_WARN);
            ov_render_cell(0, 0, anc_color, row_bg, anc_str, &hs_rem, &printed, avail,
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
