// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_resources.c
 * @brief System and process resource utilization inspector panel
 */

#include "overview_render_internal.h"
#include "overview_data_internal.h"

#define skip_draw (line_idx < lay->scroll_detail || ri >= max_rows)

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

#define H_detail_row(ri, line_idx, row, col, width, ...) \
    do \
    { \
        H_ov_buf_pos((row) + (ri), (col) + 1); \
        { \
            int _n = snprintf(NULL, 0, __VA_ARGS__); \
            H_ov_buf_printf(__VA_ARGS__); \
            H_render_pad_spaces(_n, (width)); \
        } \
        if (!skip_draw) \
        { \
            (ri)++; \
        } \
        (line_idx)++; \
    } while (0)

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
 * @target_pid:      PID of process being inspected
 * @target_loopcnt:  current iteration count of the process
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


int ov_render_resources_panel(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r        = lay->r_graph;
    int     max_rows = r.height - 2;
    int     row      = r.row + 1;

    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int        ssel  = ov_get_selected_stream_idx(lay, m);
    int        psel  = ov_get_selected_proc_idx(lay, m);
    int        fsel  = ov_get_selected_fps_idx(lay, m);

    pid_t       target_pid   = 0;
    const char *target_name  = "UNKNOWN";
    ov_rgb_t    target_color = OV_FG_DIM;

    if (focus == OV_FOCUS_STREAMS && ssel >= 0 && ssel < m->nb_streams)
    {
        target_pid   = m->streams[ssel].ownerPID;
        target_name  = m->streams[ssel].name;
        target_color = OV_FG_STREAM;
    }
    else if (focus == OV_FOCUS_PROCS && psel >= 0 && psel < m->nb_procs)
    {
        target_pid   = m->procs[psel].PID;
        target_name  = m->procs[psel].name;
        target_color = OV_FG_PROC;
    }
    else if (focus == OV_FOCUS_FPS && fsel >= 0 && fsel < m->nb_fps)
    {
        target_pid = m->fps[fsel].runpid;
        if (target_pid == 0)
        {
            target_pid = m->fps[fsel].confpid;
        }
        target_name  = m->fps[fsel].name;
        target_color = OV_FG_FPS;
    }

    const char *tabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, tabs, 4, lay->graph_tab_mode, target_color,
                       lay->focus == OV_FOCUS_GRAPH);

    int ri       = 0;
    int line_idx = 0;

    if (target_pid <= 0)
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf(" No active process for %s", target_name);
        H_render_pad_spaces(25 + strlen(target_name), r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }
    else
    {
        /* Title row */
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TITLE);
            H_ov_buf_bold();

            int policy   = sched_getscheduler(target_pid);
            int priority = 0;
            if (policy == SCHED_FIFO || policy == SCHED_RR)
            {
                struct sched_param param;
                if (sched_getparam(target_pid, &param) == 0)
                {
                    priority = param.sched_priority;
                }
            }
            else
            {
                priority = getpriority(PRIO_PROCESS, (id_t) target_pid);
            }

            const char *policy_name = "UNKNOWN";
            if (policy == SCHED_OTHER)
            {
                policy_name = "OTHER";
            }
            else if (policy == SCHED_FIFO)
            {
                policy_name = "FIFO";
            }
            else if (policy == SCHED_RR)
            {
                policy_name = "RR";
            }
#ifdef SCHED_BATCH
            else if (policy == SCHED_BATCH)
            {
                policy_name = "BATCH";
            }
#endif
#ifdef SCHED_IDLE
            else if (policy == SCHED_IDLE)
            {
                policy_name = "IDLE";
            }
#endif
#ifdef SCHED_DEADLINE
            else if (policy == SCHED_DEADLINE)
            {
                policy_name = "DEADLINE";
            }
#endif

            int n;
            if (policy != -1)
            {
                if (policy == SCHED_FIFO || policy == SCHED_RR)
                {
                    n = snprintf(NULL, 0, " %s  (PID %d, %s prio %d)", target_name,
                                 (int) target_pid, policy_name, priority);
                    H_ov_buf_printf(" %s  (PID %d, %s prio %d)", target_name, (int) target_pid,
                                    policy_name, priority);
                }
                else
                {
                    n = snprintf(NULL, 0, " %s  (PID %d, %s nice %d)", target_name,
                                 (int) target_pid, policy_name, priority);
                    H_ov_buf_printf(" %s  (PID %d, %s nice %d)", target_name, (int) target_pid,
                                    policy_name, priority);
                }
            }
            else
            {
                n = snprintf(NULL, 0, " %s  (PID %d)", target_name, (int) target_pid);
                H_ov_buf_printf(" %s  (PID %d)", target_name, (int) target_pid);
            }

            H_ov_buf_reset_attr();
            H_ov_theme_bg(OV_BG_PANEL);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }

        /* Memory header */
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf(" Memory Usage:");
            H_render_pad_spaces(14, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }

        /* Memory values */
        {
            uint64_t vm_size = 0, vm_rss = 0;
            {
                char stat_path[256];
                snprintf(stat_path, sizeof(stat_path), "/proc/%d/statm", (int) target_pid);
                FILE *fp = fopen(stat_path, "r");
                if (fp)
                {
                    if (fscanf(fp, "%" SCNu64 " %" SCNu64, &vm_size, &vm_rss) == 2)
                    {
                        /* Scale to MB (assuming 4 KB pages) */
                        vm_size = (vm_size * 4) / 1024;
                        vm_rss  = (vm_rss * 4) / 1024;
                    }
                    fclose(fp);
                }
            }

            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_TEXT);
            char buf[64];
            int  nb = snprintf(buf, sizeof(buf), "   RSS: %4" PRIu64 " MB   VIRT: %4" PRIu64 " MB",
                               vm_rss, vm_size);
            H_ov_buf_printf("%s", buf);
            H_render_pad_spaces(nb, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }

        /* CPU core activity header */
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf(" CPU Core Activity:");
            H_render_pad_spaces(19, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }

        /* Draw core mask — sysconf cached once per render frame */
        {
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

            H_ov_buf_pos(row + ri, r.col + 1);
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
            } // for cores

            H_ov_theme_fg(OV_FG_TEXT);
            H_ov_buf_printf("]");
            chars_written++;
            H_render_pad_spaces(chars_written, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }

        if (s_detail_cache.has_adv_stats)
        {
            ov_advanced_stats_t adv_stats = s_detail_cache.adv_stats;
            /* Blank separator */
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_DIM);
                H_render_pad_spaces(0, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;
            }

            /* Scheduling section header */
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_DIM);
                H_ov_buf_printf(" Scheduling & Memory:");
                H_render_pad_spaces(21, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;
            }

            /* Threads + migrations */
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int  nb =
                    snprintf(buf, sizeof(buf), "   Threads: %4" PRIu64 "    Migrations: %4" PRIu64,
                             adv_stats.threads, adv_stats.migrations);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;
            }

            /* Context switches */
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int  nb = snprintf(buf, sizeof(buf),
                                   "   Ctx Sw:  %4" PRIu64 " (Vol) / %4" PRIu64 " (Invol)",
                                   adv_stats.vol_ctxt, adv_stats.nonvol_ctxt);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;
            }

            /* Page faults */
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int  nb = snprintf(buf, sizeof(buf),
                                   "   Faults:  %4" PRIu64 " (Min) / %4" PRIu64 " (Maj)",
                                   adv_stats.minflt, adv_stats.majflt);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;
            }
        } // if has_adv

        /* Blank separator */
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_render_pad_spaces(0, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }

        /* Hardware counters header */
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf(" Hardware Counters:");
            H_render_pad_spaces(19, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }

        /* Hardware counter values */
        {
            int                has_perf = s_detail_cache.has_perf;
            ov_perf_counters_t perf_cnt = s_detail_cache.perf_cnt;

            H_ov_buf_pos(row + ri, r.col + 1);
            if (has_perf)
            {
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int  nb =
                    snprintf(buf, sizeof(buf), "   Inst: %8" PRIu64 "    Cache Miss: %8" PRIu64,
                             (uint64_t) perf_cnt.instructions, (uint64_t) perf_cnt.cache_misses);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;

                if (s_detail_cache.target_loopcnt > 0)
                {
                    /* Per-iteration instructions + cache miss */
                    {
                        H_ov_buf_pos(row + ri, r.col + 1);
                        int nb2 = snprintf(buf, sizeof(buf),
                                           "   Inst/Iter:  %.1f    Cache Miss/Iter: %.1f",
                                           perf_cnt.inst_per_loop, perf_cnt.cache_miss_per_loop);
                        H_ov_buf_printf("%s", buf);
                        H_render_pad_spaces(nb2, r.width);
                        if (!skip_draw)
                        {
                            ri++;
                        }
                        line_idx++;
                    }

                    /* Per-iteration breakdown header */
                    {
                        H_ov_buf_pos(row + ri, r.col + 1);
                        H_ov_theme_fg(OV_FG_DIM);
                        int nb2 = snprintf(buf, sizeof(buf), "   Miss/Iter Breakdown:");
                        H_ov_buf_printf("%s", buf);
                        H_render_pad_spaces(nb2, r.width);
                        if (!skip_draw)
                        {
                            ri++;
                        }
                        line_idx++;
                    }

                    /* Per-iterator breakdown values */
                    {
                        H_ov_buf_pos(row + ri, r.col + 1);
                        H_ov_theme_fg(OV_FG_TEXT);
                        int nb2 =
                            snprintf(buf, sizeof(buf),
                                     "     L1D: %.1f   LLC: %.1f"
                                     "   dTLB: %.1f   Branch: %.1f",
                                     perf_cnt.l1d_miss_per_loop, perf_cnt.llc_miss_per_loop,
                                     perf_cnt.dtlb_miss_per_loop, perf_cnt.branch_miss_per_loop);
                        H_ov_buf_printf("%s", buf);
                        H_render_pad_spaces(nb2, r.width);
                        if (!skip_draw)
                        {
                            ri++;
                        }
                        line_idx++;
                    }
                } // if target_loopcnt > 0
            }
            else
            {
                H_ov_theme_fg(OV_FG_WARN);
                H_ov_buf_printf("   [Requires Privileges / CAP_PERFMON]");
                H_render_pad_spaces(38, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;

                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_buf_printf("   [Run: milk-setup-caps]");
                H_render_pad_spaces(25, r.width);
                if (!skip_draw)
                {
                    ri++;
                }
                line_idx++;
            }
        }
    } // if target_pid > 0

    for (; ri < max_rows; ri++)
    {
        clear_row(row + ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    H_ov_buf_reset_attr();
    return 1;
}
