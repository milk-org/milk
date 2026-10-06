// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help_intro.c
 * @brief   Introductory architectural guide rendering for milk-CTRL.
 */

#include "overview_render_help_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_help_render_intro - render full interactive introduction guide to milk-CTRL.
 * @lay: layout state
 * @pr:  top row (1-based)
 * @pc:  left column (1-based)
 * @ph:  height in rows
 * @pw:  width in columns
 */
void ov_help_render_intro(const OV_LAYOUT *lay, int pr, int pc, int ph, int pw)
{
    const char *title =
        (pw >= 84) ? "ABOUT milk-CTRL (1/i: Intro • 2/k: Controls • ↑↓: Scroll • ESC: Close)"
                   : ((pw >= 52) ? "ABOUT milk-CTRL (1/i: Intro • 2/k: Controls • ESC: Close)"
                                 : "ABOUT milk-CTRL");
    ov_draw_panel_border(pr, pc, ph, pw, title, OV_FG_TITLE, 1, 0);

    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    int inner_w = pw - 4;
    int col     = pc + 2;

    /* Header Row (pr + 1): Mode selector tabs */
    ov_buf_pos(pr + 1, col);
    ov_theme_bg(OV_BG_PANEL);

    int tab1_w = (pw >= 100) ? 26 : 14;
    int tab2_w = (pw >= 100) ? 32 : 20;

    /* Tab 1 (Active): Intro & Overview */
    ov_buf_bg(240, 175, 20);
    ov_buf_fg(20, 20, 25);
    ov_buf_bold();
    ov_buf_printf("%s", (pw >= 100) ? " [▶ 1: INTRO & OVERVIEW ◀] " : " [▶ 1: INTRO ◀] ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_printf(" ");

    /* Tab 2 (Inactive): Keystrokes & Controls */
    ov_theme_bg(OV_BG_PANEL_ALT);
    ov_theme_fg(OV_FG_MUTED);
    ov_buf_bold();
    ov_buf_printf("%s", (pw >= 100) ? " [ 2: KEYSTROKES & CONTROLS ] " : " [2: CONTROLS] ");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    /* Close hint on right */
    const char *close_hint = "[ESC: Close Help] [X] ";
    int         used_hdr   = tab1_w + 1 + tab2_w;
    int         rem_hdr    = inner_w - used_hdr - (int) strlen(close_hint);
    if (rem_hdr > 0)
    {
        ov_buf_hline(' ', rem_hdr);
    }
    ov_buf_fg(255, 120, 100);
    ov_buf_bold();
    ov_buf_printf("%s", close_hint);
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_PANEL);

    /* Row pr + 2: Subtitle */
    ov_buf_pos(pr + 2, col);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("Unified Real-Time Dashboard for Shared Memory, Telemetry, and AO Pipelines");
    int sub_rem = inner_w - 75;
    if (sub_rem > 0)
    {
        ov_buf_hline(' ', sub_rem);
    }

    /* Row pr + 3: Divider */
    ov_buf_pos(pr + 3, pc);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("├");
    for (int c = pc + 1; c < pc + pw - 1; c++)
    {
        ov_buf_printf("─");
    }
    ov_buf_printf("┤");

    /* Body viewport calculation */
    int body_top    = pr + 4;
    int body_bot    = pr + ph - 2;
    int body_h      = body_bot - body_top + 1;
    int total_lines = (int) (g_intro_total);
    int max_scroll  = (total_lines > body_h) ? (total_lines - body_h) : 0;

    int scroll = lay->help_intro_scroll;
    if (scroll < 0)
    {
        scroll = 0;
    }
    if (scroll > max_scroll)
    {
        scroll = max_scroll;
    }

    for (int r = 0; r < body_h; r++)
    {
        int row_idx = r + scroll;
        int cur_row = body_top + r;
        ov_buf_pos(cur_row, col);
        ov_theme_bg(OV_BG_PANEL);

        if (row_idx >= total_lines)
        {
            ov_buf_hline(' ', inner_w);
            continue;
        }

        const intro_item_t *item        = &g_intro_items[row_idx];
        int                 printed_len = 0;

        switch (item->type)
        {
        case IL_HEADER:
            ov_buf_bold();
            ov_theme_bg(OV_BG_PANEL_ALT);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf(" %s ", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            printed_len = (int) strlen(item->prefix) + 2;
            break;

        case IL_SUBHEADER:
            ov_buf_bold();
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf("  %s", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            printed_len = (int) strlen(item->prefix) + 2;
            break;

        case IL_BULLET:
            ov_buf_bold();
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf("    %s", item->prefix);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf("%s", item->text ? item->text : "");
            printed_len =
                (int) strlen(item->prefix) + 4 + (item->text ? (int) strlen(item->text) : 0);
            break;

        case IL_TEXT:
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf("  %s", item->text ? item->text : "");
            printed_len = (item->text ? (int) strlen(item->text) : 0) + 2;
            break;

        case IL_KEY:
            ov_buf_bold();
            ov_buf_fg(130, 205, 255);
            ov_buf_printf("  %-16s", item->prefix ? item->prefix : "");
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" %s", item->text ? item->text : "");
            printed_len = 2 + 16 + 1 + (item->text ? (int) strlen(item->text) : 0);
            break;

        case IL_BLANK:
        default:
            printed_len = 0;
            break;
        }

        int pad = inner_w - printed_len;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }
    }

    if (scroll > 0)
    {
        ov_buf_pos(body_top, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▲");
    }
    if (scroll < max_scroll)
    {
        ov_buf_pos(body_bot, pc + pw - 2);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("▼");
    }

    /* Bottom divider */
    {
        ov_buf_pos(body_bot + 1, pc);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("├─");

        char bstatus[80];
        if (max_scroll > 0)
        {
            snprintf(bstatus, sizeof(bstatus),
                     " [↑↓ / PgUp/PgDn: Scroll (%d/%d) • 2/k: Controls • ESC: Close] ", scroll + 1,
                     total_lines);
        }
        else
        {
            snprintf(bstatus, sizeof(bstatus), " [2 / k: Controls Reference • ESC: Close] ");
        }

        ov_buf_bold();
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("%s", bstatus);
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_DIM);

        int used_b = 2 + (int) strlen(bstatus);
        int rem_b  = (pw - 2) - used_b;
        if (rem_b > 0)
        {
            for (int i = 0; i < rem_b; i++)
            {
                ov_buf_printf("─");
            }
        }
        ov_buf_printf("┤");
    }

    ov_buf_reset_attr();
}
