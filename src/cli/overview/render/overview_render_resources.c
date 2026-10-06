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
                /* Scale to MB (assuming 4 KB pages) */
                vm_size = (vm_size * 4) / 1024;
                vm_rss  = (vm_rss * 4) / 1024;
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
