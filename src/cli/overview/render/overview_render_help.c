// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help.c
 * @brief   Help overlay panel coordinator and shell for milk-CTRL.
 */

#include "overview_render_help_internal.h"
#include <stdio.h>
#include <string.h>


/**
 * ov_render_help - render the help overlay panel.
 * @lay: layout state
 * @m:   system model
 */
void ov_render_help(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int pr, pc, ph, pw;
    ov_help_get_rect(lay, &pr, &pc, &ph, &pw);

    if (lay->help_mode == 1)
    {
        ov_help_render_intro(lay, pr, pc, ph, pw);
        return;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    /* Draw outer panel border */
    const char *title =
        (pw >= 84) ? "HELP & CONTROLS (1/i: Intro • 2/k: Controls • ↑↓ nav • / search • ESC close)"
                   : ((pw >= 54) ? "HELP & CONTROLS (1/i: Intro • / search • ESC close)" : "HELP");
    ov_draw_panel_border(pr, pc, ph, pw, title, OV_FG_BRIGHT, 1, 0);

    /* Clear interior background */
    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    int inner_w = pw - 4;

    /* Row 1: Header Mode Selector and Context */
    {
        ov_buf_pos(pr + 1, pc + 2);
        ov_theme_bg(OV_BG_PANEL);

        int tab1_w = (pw >= 100) ? 26 : 14;
        int tab2_w = (pw >= 100) ? 32 : 20;

        /* Tab 1 (Inactive): Intro & Overview */
        ov_theme_bg(OV_BG_PANEL_ALT);
        ov_theme_fg(OV_FG_MUTED);
        ov_buf_bold();
        ov_buf_printf("%s", (pw >= 100) ? " [ 1: INTRO & OVERVIEW ] " : " [1: INTRO] ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf(" ");

        /* Tab 2 (Active): Keystrokes & Controls */
        ov_buf_bg(240, 175, 20);
        ov_buf_fg(20, 20, 25);
        ov_buf_bold();
        ov_buf_printf("%s",
                      (pw >= 100) ? " [▶ 2: KEYSTROKES & CONTROLS ◀] " : " [▶ 2: CONTROLS ◀] ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("  ");

        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Context: ");
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);

        const char *vname = (lay->view == OV_VIEW_DASHBOARD) ? "DASH"
                            : (lay->view == OV_VIEW_STREAMS) ? "STRM"
                            : (lay->view == OV_VIEW_PROCS)   ? "PROC"
                            : (lay->view == OV_VIEW_FPS)     ? "FPS"
                                                             : "CONN";
        const char *pname = (lay->focus == OV_FOCUS_STREAMS) ? "Streams"
                            : (lay->focus == OV_FOCUS_PROCS) ? "Processes"
                            : (lay->focus == OV_FOCUS_FPS)   ? "FPS"
                                                             : "Graph";
        ov_buf_printf("[%s / %s]", vname, pname);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);

        int sel_s = ov_get_selected_stream_idx(lay, m);
        int sel_p = ov_get_selected_proc_idx(lay, m);
        int sel_f = ov_get_selected_fps_idx(lay, m);

        if (lay->focus == OV_FOCUS_STREAMS && m && sel_s >= 0 && sel_s < m->nb_streams)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->streams[sel_s].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
        else if (lay->focus == OV_FOCUS_PROCS && m && sel_p >= 0 && sel_p < m->nb_procs)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->procs[sel_p].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
        else if (lay->focus == OV_FOCUS_FPS && m && sel_f >= 0 && sel_f < m->nb_fps)
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" sel: ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_bold();
            ov_buf_printf("'%s'", m->fps[sel_f].name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }

        /* Control Mode badge on right side of header row */
        int ctrl_badge_col = pc + pw - 27;
        if (ctrl_badge_col > pc + tab1_w + tab2_w + 35)
        {
            ov_buf_pos(pr + 1, ctrl_badge_col);
            if (lay->ctrl_mode)
            {
                ov_buf_bg(220, 40, 40);
                ov_buf_fg(255, 255, 255);
                ov_buf_bold();
                ov_buf_printf("  CONTROL MODE: ON   ");
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL_ALT);
                ov_theme_fg(OV_FG_DIM);
                ov_buf_bold();
                ov_buf_printf("  CONTROL MODE: OFF  ");
            }
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }
    }

    /* Row 2: Search Bar or Search Feature Notification Note */
    {
        ov_buf_pos(pr + 2, pc + 2);
        ov_theme_bg(OV_BG_PANEL);

        if (lay->help_search_active || lay->help_search[0] != '\0')
        {
            ov_buf_bold();
            ov_buf_fg(255, 220, 100);
            ov_buf_printf("Search: ");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);

            /* Search input box */
            if (lay->help_search_active)
            {
                ov_theme_bg(OV_BG_SELECTED);
                ov_theme_fg(OV_FG_BRIGHT);
            }
            else
            {
                ov_theme_bg(OV_BG_PANEL_ALT);
                ov_theme_fg(OV_FG_TEXT);
            }
            ov_buf_bold();

            int qbox_w = 26;
            if (qbox_w > pw - 38)
            {
                qbox_w = pw - 38;
            }
            if (qbox_w < 12)
            {
                qbox_w = 12;
            }

            char qdisp[48];
            snprintf(qdisp, sizeof(qdisp), "%s%s", lay->help_search,
                     lay->help_search_active ? "█" : "");
            ov_buf_printf(" %-*.*s ", qbox_w, qbox_w, qdisp);

            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);

            /* Matches count */
            char count_str[32];
            if (lay->help_search[0] == '\0')
            {
                snprintf(count_str, sizeof(count_str), " (type query)");
            }
            else
            {
                snprintf(count_str, sizeof(count_str), " (%d match%s)", nvis,
                         (nvis == 1) ? "" : "es");
            }
            ov_buf_printf("%s", count_str);

            /* Clear / cancel button */
            const char *btn_str =
                (lay->help_search[0] == '\0') ? " [ESC: cancel] " : " [ESC: clear] ";
            int used = 8 + qbox_w + 2 + (int) strlen(count_str);
            int rem  = inner_w - used - (int) strlen(btn_str);
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
            ov_buf_fg(255, 120, 100);
            ov_buf_printf("%s", btn_str);
        }
        else
        {
            /* 1-line note notifying users of the search feature */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("Search: ");
            ov_buf_bold();
            ov_buf_fg(255, 220, 100);
            ov_buf_printf("[/]");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(
                " Press '/' to search topics & commands (e.g. \"kill\", \"stream\", \"fps\")");

            int used = 8 + 3 + 68;
            int rem  = inner_w - used;
            if (rem > 0)
            {
                ov_buf_hline(' ', rem);
            }
        }
    }

    /* Row 3: Top divider */
    {
        ov_buf_pos(pr + 3, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├");
        for (int c = pc + 1; c < pc + pw - 1; c++)
        {
            ov_buf_printf("─");
        }
        ov_buf_printf("┤");
    }

    /* Detailed Help split calculation */
    int detail_h = (ph >= 36) ? 10 : ((ph >= 28) ? 8 : ((ph >= 22) ? 6 : 5));
    int split_r  = (pr + ph - 1) - detail_h;
    int list_top = pr + 4;
    int list_h   = split_r - list_top;
    if (list_h < 4)
    {
        list_h   = 4;
        split_r  = list_top + list_h;
        detail_h = (pr + ph - 1) - split_r;
    }

    /* Cursor bounds check */
    int sel = lay->help_sel;
    if (sel < 0)
    {
        sel = 0;
    }
    if (sel >= nvis)
    {
        sel = nvis - 1;
    }

    /* Scroll so cursor is within list viewport */
    int scroll = 0;
    if (sel >= list_h)
    {
        scroll = sel - list_h + 1;
    }

    /* Render visible rows in list area */
    if (lay->help_search[0] != '\0' && nvis == 0)
    {
        ov_buf_pos(list_top + 1, pc + 4);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("No matching commands found for \"%s\"", lay->help_search);
        ov_buf_pos(list_top + 2, pc + 4);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Try searching: stream, proc, fps, kill, sort, filter, view");
    }

    for (int vr = 0; vr < list_h && vr + scroll < nvis; vr++)
    {
        for (int vr = 0; vr < list_h && vr + scroll < nvis; vr++)
        {
            ov_help_render_list_row(lay, vr, scroll, sel, map, nvis, list_top, pc, pw, inner_w);
        }
    }

    /* Scroll indicators */
    if (scroll > 0)
    {
        ov_buf_pos(list_top, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▲");
    }
    if (scroll + list_h < nvis)
    {
        ov_buf_pos(split_r - 1, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▼");
    }

    /* Divider above Detailed Help */
    {
        ov_buf_pos(split_r, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├─");
        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf(" DETAILED HELP ");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);

        int         used_div = 17;
        const char *hint     = (lay->help_search[0] != '\0')
                                   ? "[↑↓ nav • ESC clear search]"
                                   : "[↑↓ nav • →/← expand • [/] search • ESC close]";
        int         hint_len = (int) strlen(hint);
        int         div_pad  = (pw - 2) - used_div - hint_len - 1;
        if (div_pad > 0)
        {
            for (int i = 0; i < div_pad; i++)
            {
                ov_buf_printf("─");
            }
            ov_buf_printf(" %s─┤", hint);
        }
        else
        {
            for (int i = 0; i < (pw - 2) - used_div; i++)
            {
                ov_buf_printf("─");
            }
            ov_buf_printf("┤");
        }
    }

    /* Render detailed help for currently selected item */
    if (sel >= 0 && sel < nvis)
    {
        ov_help_render_detail(lay, m, &g_help_entries[map[sel]], split_r, pc, pw, detail_h);
    }

    ov_buf_reset_attr();
}
