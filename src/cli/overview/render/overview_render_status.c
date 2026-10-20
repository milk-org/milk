// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_status.c
 */

#include "overview_render_internal.h"


/* All overview headers included via
 * overview_render_internal.h */

/** ov_status_btn - record a @w-column button starting at the cursor; width 0 when it overflows @cols. */
static void ov_status_btn(OV_RECT *b, int row, int w, int cols)
{
    b->row    = row;
    b->col    = ov__cursor_col;
    b->height = 1;
    b->width  = (ov__cursor_col + w - 1 <= cols) ? w : 0;
}

/**
 * ov_render_status - Render the bottom status bar and key hints
 * @lay: Layout configuration
 * @m:   Data model snapshot
 */
void ov_render_status(OV_LAYOUT *lay, const OV_MODEL *m)
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
        switch (lay->panel[OV_FOCUS_STREAMS].sort_key)
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
        switch (lay->panel[OV_FOCUS_PROCS].sort_key)
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
        switch (lay->panel[OV_FOCUS_FPS].sort_key)
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

    /* The DASH graph panel's sub-tab, named as its tab bar names it */
    static const char *const subtab[] = { " [CONN]", " [LOOPS]", " [DETAIL]", " [RES]" };
    const char              *detail_label =
        (lay->view == OV_VIEW_DASHBOARD && lay->graph_tab_mode >= 0 && lay->graph_tab_mode < 4)
            ? subtab[lay->graph_tab_mode]
            : "";

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
        ov_theme_fg(OV_FG_WARN);
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
        ov_theme_fg(OV_FG_ERROR);
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
        const char *view_name  = ov_view_label(lay->view); /* the tab bar's names */
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
        n1 += 2 + (int) strlen(view_name) + 3 + (int) strlen(panel_name); /* "  " + " › " */
        ov_theme_fg(OV_FG_DIM);
    }

    ov_focus_t  fpanel = ov_get_effective_filter_panel(lay);
    const char *pname  = (fpanel == OV_FOCUS_STREAMS) ? "STRM"
                         : (fpanel == OV_FOCUS_PROCS) ? "PROC"
                         : (fpanel == OV_FOCUS_FPS)   ? "FPS"
                                                      : "FILTER";
    const char *fpat   = (fpanel != OV_FOCUS_GRAPH) ? ov_get_panel_filter_pattern(lay, fpanel)
                                                    : ov_get_filter_pattern(lay);
    int         is_act = (fpanel != OV_FOCUS_GRAPH) ? ov_is_panel_filter_active(lay, fpanel)
                                                    : ov_is_filter_active(lay);

    if (is_act)
    {
        if ((lay->ctrl_blink % 4) < 2)
        {
            ov_theme_bg(OV_BADGE_WARN);
            ov_theme_fg(ov_rgb_on(OV_BADGE_WARN));
        }
        else
        {
            ov_theme_bg(OV_BADGE_DANGER);
            ov_theme_fg(ov_rgb_on(OV_BADGE_DANGER));
        }
        ov_buf_bold();
        char fstatus[64];
        if (fpanel != OV_FOCUS_GRAPH)
        {
            snprintf(fstatus, sizeof(fstatus), " [%s FILTER ON: /%.10s/ ('f' toggle, ESC clear)] ",
                     pname, fpat);
        }
        else
        {
            snprintf(fstatus, sizeof(fstatus), " [FILTER ON: /%.12s/ ('f' toggle, ESC clear)] ",
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
            snprintf(fstatus, sizeof(fstatus), " [%s Filter OFF: /%.10s/ ('f' enable)] ", pname,
                     fpat);
        }
        else
        {
            snprintf(fstatus, sizeof(fstatus), " [Filter OFF: /%.12s/ ('f' enable)] ", fpat);
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
            ov_theme_bg(OV_BADGE_WARN);
            ov_theme_fg(ov_rgb_on(OV_BADGE_WARN));
            ov_buf_printf("%s", bg_status);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
            n1 += (int) strlen(bg_status);
        }
    }

    const char *exit_label = " [x] exit ";
    if (lay->help.show)
    {
        exit_label = (lay->help.search[0] != '\0' || lay->help.search_active) ? " [ESC] clear "
                                                                              : " [ESC] close ";
    }
    int n_exit = (int) strlen(exit_label);

    if (lay->help.show)
    {
        const char *help_hints;
        if (lay->help.search_active)
        {
            help_hints = " Type to search   ENTER Browse results   ESC Back";
        }
        else if (lay->help.search[0] != '\0')
        {
            help_hints = " ↑↓ Navigate results   [/] Edit search   ESC Clear search";
        }
        else if (lay->help.mode == 1)
        {
            help_hints = " ↑↓/PgUp/PgDn Scroll   2 Controls Reference   ESC Close";
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
        int n_hints = snprintf(NULL, 0, "%s%s%s  +/- TAB D S/s / c m G h q  (Click headers/tabs)",
                               ctrl_hint, sort_label, detail_label);
        ov_buf_printf("%s%s%s  +/- TAB D S/s / c m G h q  (Click headers/tabs)", ctrl_hint,
                      sort_label, detail_label);
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

    /* Buttons are hit-tested where they are drawn; one cut off by the terminal edge has no target */
    ov_status_btn(&lay->r_theme_btn, r.row, n_th, lay->term_cols);
    ov_status_btn(&lay->r_exit_btn, r.row, n_exit, lay->term_cols);
    lay->r_exit_btn.col += n_th; /* drawn right after the theme button */
    if (lay->r_exit_btn.col + n_exit - 1 > lay->term_cols)
    {
        lay->r_exit_btn.width = 0;
    }

    /* Render theme button with hover highlight */
    int col_th_start = ov__cursor_col;
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
    int col_start = ov__cursor_col;
    if (lay->mouse_hover && (ov_mouse_row == r.row) && (ov_mouse_col >= col_start) &&
        (ov_mouse_col < col_start + n_exit))
    {
        ov_theme_bg(OV_BADGE_DANGER);
        ov_theme_fg(ov_rgb_on(OV_BADGE_DANGER));
    }
    else
    {
        ov_theme_fg(OV_FG_ERROR);
        ov_theme_bg(OV_BG_HEADER);
    }
    ov_buf_printf("%s", exit_label);

    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s ", tstr);
    ov_buf_reset_attr();
}
