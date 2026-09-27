// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_resources_perf.c
 * @brief Telemetry, CPU cores, scheduling, and perf counters rendering.
 */

#include "overview_render_resources_internal.h"

#define skip_draw ((*line_idx) < lay->scroll_detail || (*ri) >= max_rows)

#define H_ov_buf_pos(r, c) \
    if (!(skip_draw)) ov_buf_pos(r, c)
#define H_ov_theme_bg(c) \
    if (!(skip_draw)) ov_theme_bg(c)
#define H_ov_theme_fg(c) \
    if (!(skip_draw)) ov_theme_fg(c)
#define H_ov_buf_bold() \
    if (!(skip_draw)) ov_buf_bold()
#define H_ov_buf_reset_attr() \
    if (!(skip_draw)) ov_buf_reset_attr()
#define H_ov_buf_printf(...) \
    if (!(skip_draw)) ov_buf_printf(__VA_ARGS__)
#define H_render_pad_spaces(n, w) \
    if (!(skip_draw)) render_pad_spaces(n, w)

typedef struct
{
    pid_t               pid;
    struct timespec     last_update;
    uint64_t            core_mask;
    ov_advanced_stats_t adv_stats;
    int                 has_adv_stats;
    ov_perf_counters_t  perf_cnt;
    int                 has_perf;
    int64_t             target_loopcnt;
} ov_detail_telemetry_cache_t;

static ov_detail_telemetry_cache_t s_detail_cache;

/**
 * detail_update_telemetry - throttle /proc reads for process telemetry.
 * @target_pid:     PID of process being inspected.
 * @target_loopcnt: Current iteration count of the process.
 */
static void detail_update_telemetry(pid_t target_pid, int64_t target_loopcnt)
{
    s_detail_cache.target_loopcnt = target_loopcnt;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (double) (now.tv_sec - s_detail_cache.last_update.tv_sec) +
                     (double) (now.tv_nsec - s_detail_cache.last_update.tv_nsec) * 1e-9;

    if (s_detail_cache.pid != target_pid || elapsed >= 0.5)
    {
        s_detail_cache.pid         = target_pid;
        s_detail_cache.last_update = now;

        /* Core mask */
        int active_cores[128];
        int num_active           = pid_get_core_utilization(target_pid, active_cores, 128);
        s_detail_cache.core_mask = 0;
        for (int ii = 0; ii < num_active; ii++)
        {
            if (active_cores[ii] >= 0 && active_cores[ii] < 64)
            {
                s_detail_cache.core_mask |= (1ULL << active_cores[ii]);
            }
        }

        /* Advanced stats */
        s_detail_cache.has_adv_stats =
            (pid_get_advanced_stats(target_pid, &s_detail_cache.adv_stats) == 0);

        /* Perf counters */
        s_detail_cache.has_perf =
            (pid_read_perf_counters(target_pid, target_loopcnt, &s_detail_cache.perf_cnt) == 0);
    }
}

/**
 * ov_render_resources_perf_section - render CPU cores, scheduler stats and perf counters.
 * @lay:        Layout configuration.
 * @m:          Snapshot model.
 * @target_pid: Process identifier.
 * @row:        Starting row.
 * @ri:         Row offset index (in/out).
 * @line_idx:   Logical line index for scrolling (in/out).
 * @max_rows:   Max visible rows.
 */
void ov_render_resources_perf_section(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    pid_t            target_pid,
    int              row,
    int             *ri,
    int             *line_idx,
    int              max_rows)
{
    OV_RECT r = lay->r_graph;

    /* Draw core mask */
    int64_t target_loopcnt = 0;
    for (int ii = 0; ii < m->nb_procs; ii++)
    {
        if (m->procs[ii].PID == target_pid && m->procs[ii].active)
        {
            target_loopcnt = m->procs[ii].loopcnt;
            break;
        }
    }
    detail_update_telemetry(target_pid, target_loopcnt);

    uint64_t core_mask = s_detail_cache.core_mask;

    H_ov_buf_pos(row + (*ri), r.col + 1);
    H_ov_theme_fg(OV_FG_TEXT);
    H_ov_buf_printf("   [");
    int  chars_written = 4;
    long num_cores     = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cores <= 0 || num_cores > 64)
    {
        num_cores = 64;
    }

    for (long cc = 0; cc < num_cores; cc++)
    {
        if (core_mask & (1ULL << cc))
        {
            H_ov_theme_fg(OV_FG_PROC);
            H_ov_buf_printf("■");
        }
        else
        {
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf("-");
        }
        chars_written++;
    }

    H_ov_theme_fg(OV_FG_TEXT);
    H_ov_buf_printf("]");
    chars_written++;
    H_render_pad_spaces(chars_written, r.width);
    if (!skip_draw)
    {
        (*ri)++;
    }
    (*line_idx)++;

    if (s_detail_cache.has_adv_stats)
    {
        ov_advanced_stats_t adv_stats = s_detail_cache.adv_stats;
        /* Blank separator */
        H_ov_buf_pos(row + (*ri), r.col + 1);
        H_ov_theme_fg(OV_FG_DIM);
        H_render_pad_spaces(0, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;

        /* Scheduling section header */
        H_ov_buf_pos(row + (*ri), r.col + 1);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf(" Scheduling & Memory:");
        H_render_pad_spaces(21, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;

        /* Threads + migrations */
        H_ov_buf_pos(row + (*ri), r.col + 1);
        H_ov_theme_fg(OV_FG_TEXT);
        char buf[128];
        int  nb = snprintf(buf, sizeof(buf), "   Threads: %4" PRIu64 "    Migrations: %4" PRIu64,
                           adv_stats.threads, adv_stats.migrations);
        H_ov_buf_printf("%s", buf);
        H_render_pad_spaces(nb, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;

        /* Context switches */
        H_ov_buf_pos(row + (*ri), r.col + 1);
        H_ov_theme_fg(OV_FG_TEXT);
        nb = snprintf(buf, sizeof(buf),
                      "   Ctx Sw:  %4" PRIu64 " (Vol) / %4" PRIu64 " (Invol)",
                      adv_stats.vol_ctxt, adv_stats.nonvol_ctxt);
        H_ov_buf_printf("%s", buf);
        H_render_pad_spaces(nb, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;

        /* Page faults */
        H_ov_buf_pos(row + (*ri), r.col + 1);
        H_ov_theme_fg(OV_FG_TEXT);
        nb = snprintf(buf, sizeof(buf),
                      "   Faults:  %4" PRIu64 " (Min) / %4" PRIu64 " (Maj)",
                      adv_stats.minflt, adv_stats.majflt);
        H_ov_buf_printf("%s", buf);
        H_render_pad_spaces(nb, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;
    }

    /* Blank separator */
    H_ov_buf_pos(row + (*ri), r.col + 1);
    H_ov_theme_fg(OV_FG_DIM);
    H_render_pad_spaces(0, r.width);
    if (!skip_draw)
    {
        (*ri)++;
    }
    (*line_idx)++;

    /* Hardware counters header */
    H_ov_buf_pos(row + (*ri), r.col + 1);
    H_ov_theme_fg(OV_FG_DIM);
    H_ov_buf_printf(" Hardware Counters:");
    H_render_pad_spaces(19, r.width);
    if (!skip_draw)
    {
        (*ri)++;
    }
    (*line_idx)++;

    /* Hardware counter values */
    int                has_perf = s_detail_cache.has_perf;
    ov_perf_counters_t perf_cnt = s_detail_cache.perf_cnt;

    H_ov_buf_pos(row + (*ri), r.col + 1);
    if (has_perf)
    {
        H_ov_theme_fg(OV_FG_TEXT);
        char buf[128];
        int  nb = snprintf(buf, sizeof(buf), "   Inst: %8" PRIu64 "    Cache Miss: %8" PRIu64,
                           (uint64_t) perf_cnt.instructions, (uint64_t) perf_cnt.cache_misses);
        H_ov_buf_printf("%s", buf);
        H_render_pad_spaces(nb, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;

        if (s_detail_cache.target_loopcnt > 0)
        {
            /* Per-iteration instructions + cache miss */
            H_ov_buf_pos(row + (*ri), r.col + 1);
            int nb2 = snprintf(buf, sizeof(buf),
                               "   Inst/Iter:  %.1f    Cache Miss/Iter: %.1f",
                               perf_cnt.inst_per_loop, perf_cnt.cache_miss_per_loop);
            H_ov_buf_printf("%s", buf);
            H_render_pad_spaces(nb2, r.width);
            if (!skip_draw)
            {
                (*ri)++;
            }
            (*line_idx)++;

            /* Per-iteration breakdown header */
            H_ov_buf_pos(row + (*ri), r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            nb2 = snprintf(buf, sizeof(buf), "   Miss/Iter Breakdown:");
            H_ov_buf_printf("%s", buf);
            H_render_pad_spaces(nb2, r.width);
            if (!skip_draw)
            {
                (*ri)++;
            }
            (*line_idx)++;

            /* Per-iterator breakdown values */
            H_ov_buf_pos(row + (*ri), r.col + 1);
            H_ov_theme_fg(OV_FG_TEXT);
            nb2 = snprintf(buf, sizeof(buf),
                           "     L1D: %.1f   LLC: %.1f"
                           "   dTLB: %.1f   Branch: %.1f",
                           perf_cnt.l1d_miss_per_loop, perf_cnt.llc_miss_per_loop,
                           perf_cnt.dtlb_miss_per_loop, perf_cnt.branch_miss_per_loop);
            H_ov_buf_printf("%s", buf);
            H_render_pad_spaces(nb2, r.width);
            if (!skip_draw)
            {
                (*ri)++;
            }
            (*line_idx)++;
        }
    }
    else
    {
        H_ov_theme_fg(OV_FG_WARN);
        H_ov_buf_printf("   [Requires Privileges / CAP_PERFMON]");
        H_render_pad_spaces(38, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;

        H_ov_buf_pos(row + (*ri), r.col + 1);
        H_ov_buf_printf("   [Run: milk-setup-caps]");
        H_render_pad_spaces(25, r.width);
        if (!skip_draw)
        {
            (*ri)++;
        }
        (*line_idx)++;
    }
}
