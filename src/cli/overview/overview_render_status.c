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
        const char *fstr  = lay->filter;
        const char *pname = (lay->filter_panel == OV_FOCUS_STREAMS) ? "STREAMS"
                            : (lay->filter_panel == OV_FOCUS_PROCS) ? "PROCESSINFO"
                            : (lay->filter_panel == OV_FOCUS_FPS)   ? "FPS"
                                                                    : "FILTER";
        ov_theme_fg(OV_FG_TEXT);
        char pfx = lay->filter_jump ? '?' : '/';
        int  np  = snprintf(NULL, 0, " [%s] %c%s", pname, pfx, fstr);
        ov_buf_printf(" [%s] %c%s", pname, pfx, fstr);

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
        case OV_VIEW_GRAPH:
            view_name = "GRP";
            break;
        case OV_VIEW_LOOPS:
            view_name = "LOOP";
            break;
        default:
            view_name = "???";
            break;
        }
        const char *panel_name = "";
        ov_rgb_t    panel_fg   = OV_FG_DIM;
        if (lay->view == OV_VIEW_LOOPS ||
            (lay->view == OV_VIEW_DASHBOARD && lay->graph_tab_mode == 1 &&
             lay->focus == OV_FOCUS_GRAPH))
        {
            panel_name = "Loops";
            panel_fg   = OV_FG_LOOP;
        }
        else
        {
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
        }
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("  %s", view_name);
        ov_buf_printf(" \xe2\x80\xba ");
        ov_theme_fg(panel_fg);
        ov_buf_printf("%s", panel_name);
        n1 += 3 + (int) strlen(view_name) + 3 + (int) strlen(panel_name);
        ov_theme_fg(OV_FG_DIM);
    }

    ov_focus_t  fpanel   = ov_get_effective_filter_panel(lay);
    const char *pname    = (fpanel == OV_FOCUS_STREAMS) ? "STRM"
                           : (fpanel == OV_FOCUS_PROCS) ? "PROC"
                           : (fpanel == OV_FOCUS_FPS)   ? "FPS"
                                                        : "FILTER";
    const char *fpat     = (fpanel != OV_FOCUS_GRAPH) ? ov_get_panel_filter_pattern(lay, fpanel)
                                                      : ov_get_filter_pattern(lay);
    int is_act           = (fpanel != OV_FOCUS_GRAPH) ? ov_is_panel_filter_active(lay, fpanel)
                                                      : ov_is_filter_active(lay);

    if (is_act)
    {
        if ((lay->ctrl_blink % 4) < 2)
        {
            ov_buf_bg(255, 190, 0);   /* bright amber/gold */
            ov_buf_fg(20, 20, 20);    /* dark text */
        }
        else
        {
            ov_buf_bg(230, 80, 20);   /* vibrant red-orange */
            ov_buf_fg(255, 255, 255); /* white text */
        }
        ov_buf_bold();
        char fstatus[64];
        if (fpanel != OV_FOCUS_GRAPH)
        {
            snprintf(fstatus, sizeof(fstatus),
                     " [%s FILTER ON: /%.10s/ ('f' toggle, ESC clear)] ",
                     pname, fpat);
        }
        else
        {
            snprintf(fstatus, sizeof(fstatus),
                     " [FILTER ON: /%.12s/ ('f' toggle, ESC clear)] ",
                     fpat);
        }
        ov_buf_printf("%s", fstatus);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_DIM);
        n1 += (int) strlen(fstatus);
    }
    else if (fpat[0] != '\0')
    {
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_bold();
        char fstatus[64];
        if (fpanel != OV_FOCUS_GRAPH)
        {
            snprintf(fstatus, sizeof(fstatus),
                     " [%s Filter OFF: /%.10s/ ('f' enable)] ",
                     pname, fpat);
        }
        else
        {
            snprintf(fstatus, sizeof(fstatus),
                     " [Filter OFF: /%.12s/ ('f' enable)] ",
                     fpat);
        }
        ov_buf_printf("%s", fstatus);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_DIM);
        n1 += (int) strlen(fstatus);
    }

    /* Also show alert badges for other panels with active filters */
    ov_focus_t  bg_panels[3] = { OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS };
    const char *bg_names[3]  = { "STRM", "PROC", "FPS" };
    for (int p = 0; p < 3; p++)
    {
        if (bg_panels[p] != fpanel && ov_is_panel_filter_active(lay, bg_panels[p]))
        {
            const char *bg_pat = ov_get_panel_filter_pattern(lay, bg_panels[p]);
            char        bg_status[64];
            snprintf(bg_status, sizeof(bg_status), " [%s: /%.8s/] ", bg_names[p], bg_pat);
            ov_buf_bold();
            ov_buf_bg(220, 130, 20);
            ov_buf_fg(255, 255, 255);
            ov_buf_printf("%s", bg_status);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
            n1 += (int) strlen(bg_status);
        }
    }

    const char *exit_label = " [x] exit ";
    if (lay->show_help)
    {
        exit_label = (lay->help_search[0] != '\0' || lay->help_search_active)
                         ? " [ESC] clear "
                         : " [ESC] close ";
    }
    int n_exit = (int) strlen(exit_label);

    if (lay->show_help)
    {
        const char *help_hints;
        if (lay->help_search_active)
        {
            help_hints = " Type to search   ENTER Browse results   ESC Back";
        }
        else if (lay->help_search[0] != '\0')
        {
            help_hints = " ↑↓ Navigate results   [/] Edit search   ESC Clear search";
        }
        else if (lay->help_mode == 1)
        {
            help_hints = " ↑↓/PgUp/PgDn Scroll   2/k Controls Reference   ESC Close";
        }
        else
        {
            help_hints = " 1/i Intro   ↑↓ Nav   →/← Expand   [/] Search   ESC Close";
        }
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("%s", help_hints);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_DIM);
        n1 += (int) strlen(help_hints);
    }
    else
    {
        int n_hints = snprintf(NULL, 0, "%s%s%s  +/- TAB D S/s / p c m G h q  (Click headers/tabs)",
                               ctrl_hint, sort_label, detail_label);
        ov_buf_printf("%s%s%s  +/- TAB D S/s / p c m G h q  (Click headers/tabs)",
                      ctrl_hint, sort_label, detail_label);
        n1 += n_hints;
    }

    time_t     now    = time(NULL);
    struct tm *tm_ptr = localtime(&now);
    char       tstr[16];
    int        n2 = strftime(tstr, sizeof(tstr), "%H:%M:%S", tm_ptr);

    char th_label[32];
    snprintf(th_label, sizeof(th_label), " [%s] ", ov_active_theme->id);
    int n_th = (int) strlen(th_label);

    /* Clock on the right edge */
    int pad = r.width - n1 - n_th - n_exit - n2 - 1;
    if (pad > 0)
    {
        ov_buf_hline(' ', pad);
    }

    /* Render theme button with hover highlight */
    int col_th_start = r.width - n_th - n_exit - n2;
    if (lay->mouse_hover && (ov_mouse_row == r.row) && (ov_mouse_col >= col_th_start) &&
        (ov_mouse_col < col_th_start + n_th))
    {
        ov_theme_bg(OV_BG_SELECTED);
        ov_theme_fg(OV_FG_BRIGHT);
    }
    else
    {
        ov_theme_fg(OV_FG_TITLE);
        ov_theme_bg(OV_BG_HEADER);
    }
    ov_buf_printf("%s", th_label);

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
