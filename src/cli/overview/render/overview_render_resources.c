// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_resources.c
 * @brief System and process resource utilization inspector panel.
 */

#include "overview_render_resources_internal.h"

#define skip_draw (line_idx < lay->scroll_detail || ri >= max_rows)

#define H_ov_buf_pos(r, c) \
    if (!(skip_draw))      \
    ov_buf_pos(r, c)
#define H_ov_theme_bg(c) \
    if (!(skip_draw))    \
    ov_theme_bg(c)
#define H_ov_theme_fg(c) \
    if (!(skip_draw))    \
    ov_theme_fg(c)
#define H_ov_buf_bold() \
    if (!(skip_draw))   \
    ov_buf_bold()
#define H_ov_buf_reset_attr() \
    if (!(skip_draw))         \
    ov_buf_reset_attr()
#define H_ov_buf_printf(...) \
    if (!(skip_draw))        \
    ov_buf_printf(__VA_ARGS__)
#define H_render_pad_spaces(n, w) \
    if (!(skip_draw))             \
    render_pad_spaces(n, w)

/**
 * ov_render_resources_panel - render system hardware, thread affinity, and perf metrics panel.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if panel was rendered, 0 otherwise.
 */
int ov_render_resources_panel(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r        = lay->r_graph;
    int     max_rows = r.height - 2;
    int     row      = r.row + 1;

    /* the entry DETAILS shows: the focused list's, else the last selected one (also while the tab has focus) */
    ov_focus_t  panel        = OV_FOCUS_STREAMS;
    int         idx          = ov_detail_target(lay, m, &panel);
    pid_t       target_pid   = 0;
    const char *target_name  = "UNKNOWN";
    ov_rgb_t    target_color = OV_FG_DIM;

    if (idx >= 0 && panel == OV_FOCUS_STREAMS)
    {
        target_pid   = m->streams[idx].ownerPID;
        target_name  = m->streams[idx].name;
        target_color = OV_FG_STREAM;
    }
    else if (idx >= 0 && panel == OV_FOCUS_PROCS)
    {
        target_pid   = m->procs[idx].PID;
        target_name  = m->procs[idx].name;
        target_color = OV_FG_PROC;
    }
    else if (idx >= 0 && panel == OV_FOCUS_FPS)
    {
        target_pid   = (m->fps[idx].runpid != 0) ? m->fps[idx].runpid : m->fps[idx].confpid;
        target_name  = m->fps[idx].name;
        target_color = OV_FG_FPS;
    }

    ov_subtab_bar(lay, target_color);

    int ri       = 0;
    int line_idx = 0;

    if (target_pid <= 0)
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf(" No active process for %s", target_name);
        H_render_pad_spaces(25 + (int) strlen(target_name), r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }
    else
    {
        /* Title row */
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
                n = snprintf(NULL, 0, " %s  (PID %d, %s prio %d)", target_name, (int) target_pid,
                             policy_name, priority);
                H_ov_buf_printf(" %s  (PID %d, %s prio %d)", target_name, (int) target_pid,
                                policy_name, priority);
            }
            else
            {
                n = snprintf(NULL, 0, " %s  (PID %d, %s nice %d)", target_name, (int) target_pid,
                             policy_name, priority);
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

        /* Memory header */
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf(" Memory Usage:");
        H_render_pad_spaces(14, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;

        /* Memory values */
        uint64_t vm_size = 0, vm_rss = 0;
        char     stat_path[256];
        snprintf(stat_path, sizeof(stat_path), "/proc/%d/statm", (int) target_pid);
        FILE *fp = fopen(stat_path, "r");
        if (fp)
        {
            if (fscanf(fp, "%" SCNu64 " %" SCNu64, &vm_size, &vm_rss) == 2)
            {
                /* Pages to MB */
                uint64_t page_kb = (uint64_t) sysconf(_SC_PAGESIZE) / 1024;
                vm_size          = (vm_size * page_kb) / 1024;
                vm_rss           = (vm_rss * page_kb) / 1024;
            }
            fclose(fp);
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

        /* CPU core activity header */
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf(" CPU Core Activity:");
        H_render_pad_spaces(19, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;

        /* Render CPU cores and hardware perf section */
        ov_render_resources_perf_section(lay, m, target_pid, row, &ri, &line_idx, max_rows);
    }

    for (; ri < max_rows; ri++)
    {
        clear_row(row + ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    H_ov_buf_reset_attr();
    return 1;
}
