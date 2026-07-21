// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_status.c
 */

#include "overview_render_internal.h"


/* All overview headers included via
 * overview_render_internal.h */

/* forward declarations for scan API */

void ov_render_status(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r = lay->r_status;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);

    double interval = (double) ov_scan_get_interval();
    double rate_hz  = (interval > 0.0) ? (1.0 / interval) : 0.0;

    /* Context-sensitive ctrl hints */
    const char *ctrl_hint = "";
    if (lay->ctrl_mode)
    {
        switch (lay->focus)
        {
        case OV_FOCUS_FPS:
            ctrl_hint = "  r:run s:conf k:kill";
            break;
        case OV_FOCUS_STREAMS:
            ctrl_hint = "  DEL:delete";
            break;
        case OV_FOCUS_PROCS:
            ctrl_hint = "  p:pause e:exit k:kill";
            break;
        default:
            ctrl_hint = "";
            break;
        }
    }

    /* Sort key label */
    const char *sort_label = "";
    switch (lay->focus)
    {
    case OV_FOCUS_STREAMS:
        switch (lay->sort_key_stream)
        {
        case 1:
            sort_label = " [sort:typ]";
            break;
        case 2:
            sort_label = " [sort:size]";
            break;
        case 3:
            sort_label = " [sort:Hz]";
            break;
        case 4:
            sort_label = " [sort:MB/s]";
            break;
        case 5:
            sort_label = " [sort:inode]";
            break;
        case 6:
            sort_label = " [sort:count]";
            break;
        case 7:
            sort_label = " [sort:ancestry]";
            break;
        default:
            sort_label = " [sort:name]";
            break;
        }
        break;
    case OV_FOCUS_PROCS:
        switch (lay->sort_key_proc)
        {
        case 1:
            sort_label = " [sort:PID]";
            break;
        case 2:
            sort_label = " [sort:stat]";
            break;
        case 3:
            sort_label = " [sort:Hz]";
            break;
        case 4:
            sort_label = " [sort:MEM]";
            break;
        case 5:
            sort_label = " [sort:ancestry]";
            break;
        case 6:
            sort_label = " [sort:PRIO]";
            break;
        case 7:
            sort_label = " [sort:UPTIME]";
            break;
        case 8:
            sort_label = " [sort:CPU%]";
            break;
        case 9:
            sort_label = " [sort:LOOPCNT]";
            break;
        case 10:
            sort_label = " [sort:DUTY]";
            break;
        default:
            sort_label = " [sort:name]";
            break;
        }
        break;
    case OV_FOCUS_FPS:
        switch (lay->sort_key_fps)
        {
        case 1:
            sort_label = " [sort:CPID]";
            break;
        case 2:
            sort_label = " [sort:MEM]";
            break;
        case 3:
            sort_label = " [sort:ancestry]";
            break;
        case 4:
            sort_label = " [sort:RPID]";
            break;
        case 5:
            sort_label = " [sort:TMX]";
            break;
        case 6:
            sort_label = " [sort:STR]";
            break;
        default:
            sort_label = " [sort:name]";
            break;
        }
        break;
    default:
        break;
    }

    const char *detail_label = "";
    switch (lay->graph_tab_mode)
    {
    case 0:
        detail_label = " [CONN]";
        break;
    case 1:
        detail_label = " [DETAIL]";
        break;
    case 2:
        detail_label = " [RES]";
        break;
    }

    /* Filter editing prompt overrides normal status */
    if (lay->filter_editing)
    {
        const char *fstr = "";
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            fstr = lay->filter_stream;
            break;
        case OV_FOCUS_PROCS:
            fstr = lay->filter_proc;
            break;
        case OV_FOCUS_FPS:
            fstr = lay->filter_fps;
            break;
        default:
            break;
        }
        ov_theme_fg(OV_FG_TEXT);
        char pfx = lay->filter_jump ? '?' : '/';
        int  np  = snprintf(NULL, 0, " %c%s", pfx, fstr);
        ov_buf_printf(" %c%s", pfx, fstr);

        /* Blinking cursor */
        ov_buf_fg(255, 200, 50);
        ov_buf_printf("█");
        np += 1;

        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("  (ENTER=accept ESC=cancel)");
        np += 26;

        int pad = r.width - np - 1;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }
        ov_buf_reset_attr();
        return;
    }

    int n1 = 0;
    if (lay->paused)
    {
        ov_buf_fg(255, 50, 50);
        n1 = snprintf(NULL, 0, " [PAUSED] ");
        ov_buf_printf(" [PAUSED] ");
        ov_theme_fg(OV_FG_DIM);
    }
    else
    {
        n1 = snprintf(NULL, 0, " scan:%.0fms %.1fHz", m->scan_time_ms, rate_hz);
        ov_buf_printf(" scan:%.0fms %.1fHz", m->scan_time_ms, rate_hz);
    }
    /* Breadcrumb trail (#11) */
    {
        const char *view_name = "";
        switch (lay->view)
        {
        case OV_VIEW_DASHBOARD:
            view_name = "OVW";
            break;
        case OV_VIEW_STREAMS:
            view_name = "STR";
            break;
        case OV_VIEW_FPS:
            view_name = "FPS";
            break;
        case OV_VIEW_PROCS:
            view_name = "PRC";
            break;
        default:
            view_name = "???";
            break;
        }
        const char *panel_name = "";
        ov_rgb_t    panel_fg   = OV_FG_DIM;
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            panel_name = "Streams";
            panel_fg   = OV_FG_STREAM;
            break;
        case OV_FOCUS_FPS:
            panel_name = "FPS";
            panel_fg   = OV_FG_FPS;
            break;
        case OV_FOCUS_PROCS:
            panel_name = "Procs";
            panel_fg   = OV_FG_PROC;
            break;
        default:
            panel_name = "Graph";
            break;
        }
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("  %s", view_name);
        ov_buf_printf(" \xe2\x80\xba ");
        ov_theme_fg(panel_fg);
        ov_buf_printf("%s", panel_name);
        n1 += 3 + (int) strlen(view_name) + 3 + (int) strlen(panel_name);
        ov_theme_fg(OV_FG_DIM);
    }

    int n_hints = snprintf(NULL, 0, "%s%s%s  +/- TAB D S/s / p c G h q  (Click headers/tabs)",
                           ctrl_hint, sort_label, detail_label);
    ov_buf_printf("%s%s%s  +/- TAB D S/s / p c G h q  (Click headers/tabs)", ctrl_hint, sort_label,
                  detail_label);
    n1 += n_hints;

    /* [x] exit button */
    const char *exit_label = " [x] exit ";
    int         n_exit     = (int) strlen(exit_label);

    time_t     now    = time(NULL);
    struct tm *tm_ptr = localtime(&now);
    char       tstr[16];
    int        n2 = strftime(tstr, sizeof(tstr), "%H:%M:%S", tm_ptr);

    /* Clock on the right edge */
    int pad = r.width - n1 - n_exit - n2 - 1;
    if (pad > 0)
    {
        ov_buf_hline(' ', pad);
    }

    /* Render [x] exit with distinct color or hover highlight */
    int col_start = r.width - n_exit - n2;
    if (lay->mouse_hover && (ov_mouse_row == r.row) && (ov_mouse_col >= col_start) &&
        (ov_mouse_col < col_start + n_exit))
    {
        ov_buf_bg(220, 40, 40);   /* vibrant red background */
        ov_buf_fg(255, 255, 255); /* white text */
    }
    else
    {
        ov_buf_fg(200, 80, 80);
        ov_theme_bg(OV_BG_HEADER);
    }
    ov_buf_printf("%s", exit_label);

    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s ", tstr);
    ov_buf_reset_attr();
}
