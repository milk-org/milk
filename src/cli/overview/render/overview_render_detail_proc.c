// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_detail_proc.c
 * @brief   Process telemetry and detail inspector pane for milk-CTRL.
 */

#include "overview_render_detail_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * @brief Render detailed process info in the panel.
 */
int ov_fps__render_detail_proc(OV_LAYOUT      *lay,
                                      const OV_MODEL *m,
                                      int             psel,
                                      OV_RECT         r,
                                      int             max_rows,
                                      int             row)
{
    const OV_PROC *p = &m->procs[psel];

    const char *tabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, tabs, 4, lay->graph_tab_mode, OV_FG_PROC,
                       lay->focus == OV_FOCUS_GRAPH);

    int ri       = 0;
    int line_idx = 0;

    /* Name + PID */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TITLE);
        H_ov_buf_bold();
        int n = snprintf(NULL, 0, " %s  (PID %d)", p->name, (int) p->PID);
        H_ov_buf_printf(" %s  (PID %d)", p->name, (int) p->PID);
        H_ov_buf_reset_attr();
        H_ov_theme_bg(OV_BG_PANEL);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Status + loop info */
    {
        const char *sl;
        switch (p->loopstat)
        {
        case 0:
            sl = "IDLE";
            break;
        case 1:
            sl = "RUNNING";
            break;
        case 2:
            sl = "PAUSED";
            break;
        case 3:
            sl = "TERMINATING";
            break;
        case 4:
            sl = "ERROR";
            break;
        default:
            sl = "UNKNOWN";
            break;
        }
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TEXT);
        H_ov_buf_printf(" Status: %s  Loops: ", sl);
        H_ov_theme_fg(p->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM);
        H_ov_buf_printf("%" PRId64, (int64_t) p->loopcnt);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf("  Hz: ");
        H_ov_theme_fg(p->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM);
        H_ov_buf_printf("%.1f", p->loop_hz);
        int n = snprintf(NULL, 0, " Status: %s  Loops: %" PRId64 "  Hz: %.1f", sl,
                         (int64_t) p->loopcnt, p->loop_hz);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* CPU and Mem */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TEXT);
        int n = snprintf(NULL, 0, " CPU: %5.1f%%  Mem: %" PRId64 " KB", p->cpu_used,
                         (int64_t) p->mem_rss_kb);
        H_ov_buf_printf(" CPU: %5.1f%%  Mem: %" PRId64 " KB", p->cpu_used, (int64_t) p->mem_rss_kb);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Trigger info */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_CONN);
        const char *tstream = (p->trigstreamname[0] != '\0') ? p->trigstreamname : "-";
        int         n = snprintf(NULL, 0, " Trigger: %s  stream: %s  sem: %d",
                                 render_trigmode_label(p->triggermode), tstream, p->triggersem);
        H_ov_buf_printf(" Trigger: %s  stream: %s  sem: %d", render_trigmode_label(p->triggermode),
                        tstream, p->triggersem);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Missed frames */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(p->triggermissed > 0 ? OV_FG_WARN : OV_FG_DIM);
        int n = snprintf(NULL, 0, " Missed: %d (cumul: %" PRIu64 ")", p->triggermissed,
                         (uint64_t) p->triggermissed_cumul);
        H_ov_buf_printf(" Missed: %d (cumul: %" PRIu64 ")", p->triggermissed,
                        (uint64_t) p->triggermissed_cumul);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Exec time + overhead */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        if (p->MeasureTiming && p->dtmedian_exec_ns > 0)
        {
            double exec_ms  = 1.0e-6 * (double) p->dtmedian_exec_ns;
            double overhead = 0.0;
            if (p->dtmedian_iter_ns > 0)
            {
                overhead = 100.0 * (double) p->dtmedian_exec_ns / (double) p->dtmedian_iter_ns;
            }
            H_ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0, " Exec: %.3f ms  Load: %.1f%%  RT: %d", exec_ms, overhead,
                             p->rt_priority);
            H_ov_buf_printf(" Exec: %.3f ms  Load: %.1f%%  RT: %d", exec_ms, overhead,
                            p->rt_priority);
            H_render_pad_spaces(n, r.width);
        }
        else
        {
            H_ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0, " Timing: disabled  RT: %d", p->rt_priority);
            H_ov_buf_printf(" Timing: disabled  RT: %d", p->rt_priority);
            H_render_pad_spaces(n, r.width);
        }
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    lay->detail_total_lines = line_idx;
    for (; ri < max_rows; ri++)
    {
        clear_row(row + ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    H_ov_buf_reset_attr();
    return 1;
} // ov_fps__render_detail_proc
