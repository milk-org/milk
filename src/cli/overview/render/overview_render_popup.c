// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_popup.c
 * @brief   Theme selector modal popup rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <stdio.h>
#include <string.h>

void ov_render_theme_popup(OV_LAYOUT *lay)
{
    if (!lay->theme_popup_active)
    {
        return;
    }

    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    double elapsed = (now_ts.tv_sec - lay->theme_popup_ts.tv_sec) +
                     (now_ts.tv_nsec - lay->theme_popup_ts.tv_nsec) * 1e-9;
    if (elapsed >= 1.0)
    {
        lay->theme_popup_active = 0;
        return;
    }

    int nthemes = ov_theme_count();
    int pw      = 46;
    int ph      = nthemes + 2;

    int pr = lay->term_rows - ph;
    int pc = lay->term_cols - pw - 2;

    if (pr < 2)
    {
        pr = 2;
    }
    if (pc < 1)
    {
        pc = 1;
    }
    if (pr + ph > lay->term_rows)
    {
        ph = lay->term_rows - pr;
    }
    if (pc + pw > lay->term_cols)
    {
        pw = lay->term_cols - pc;
    }

    lay->r_theme_popup.row    = pr;
    lay->r_theme_popup.col    = pc;
    lay->r_theme_popup.height = ph;
    lay->r_theme_popup.width  = pw;

    /* Top border */
    ov_buf_pos(pr, pc);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_WARN);
    ov_buf_bold();
    ov_buf_printf("%s%s", OV_BOX_TL, OV_BOX_H);
    ov_theme_fg(OV_FG_TITLE);
    ov_buf_printf(" THEME SELECTOR (↑/↓ • ESC) ");
    ov_theme_fg(OV_FG_WARN);
    int top_rem = (pc + pw - 1) - ov__cursor_col;
    if (top_rem > 0)
    {
        ov_buf_hline_utf8(OV_BOX_H, top_rem);
    }
    ov_buf_printf("%s", OV_BOX_TR);
    ov_buf_reset_attr();

    /* Render theme items */
    for (int i = 0; i < nthemes && (pr + 1 + i) < (pr + ph - 1); i++)
    {
        int               row       = pr + 1 + i;
        int               is_sel    = (i == lay->theme_popup_sel);
        int               is_active = (i == ov_theme_get_active_index());
        const ov_theme_t *th        = ov_theme_get(i);

        ov_buf_pos(row, pc);

        /* Left border */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("%s", OV_BOX_V);

        /* Row content */
        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;
        ov_theme_bg(row_bg);

        if (is_sel)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf(" ▶ ");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("   ");
        }

        /* Swatch: 4 color preview blocks */
        ov_buf_bg(th->bg_terminal.r, th->bg_terminal.g, th->bg_terminal.b);
        ov_buf_fg(th->fg_title.r, th->fg_title.g, th->fg_title.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_stream.r, th->fg_stream.g, th->fg_stream.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_active.r, th->fg_active.g, th->fg_active.b);
        ov_buf_printf("■");
        ov_buf_fg(th->fg_warn.r, th->fg_warn.g, th->fg_warn.b);
        ov_buf_printf("■ ");

        /* Restore row background */
        ov_theme_bg(row_bg);

        /* Theme name */
        if (is_sel)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_BRIGHT);
        }
        else
        {
            ov_theme_fg(OV_FG_TEXT);
        }
        ov_buf_printf("%-18.18s ", th->name);

        /* Active tag */
        if (is_active)
        {
            ov_buf_bold();
            ov_theme_fg(OV_FG_ACTIVE);
            ov_buf_printf("● active");
        }
        else
        {
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("        ");
        }

        /* Pad row to right edge */
        render_pad_to_col(pc + pw - 1);

        /* Right border */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("%s", OV_BOX_V);
        ov_buf_reset_attr();
    }

    /* Bottom border with auto-close countdown */
    int bot_row = pr + ph - 1;
    ov_buf_pos(bot_row, pc);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_WARN);
    ov_buf_bold();
    ov_buf_printf("%s%s", OV_BOX_BL, OV_BOX_H);

    double remain = 1.0 - elapsed;
    if (remain < 0.0)
    {
        remain = 0.0;
    }
    char hint[48];
    snprintf(hint, sizeof(hint), " auto-closes in %.1fs ", remain);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("%s", hint);

    ov_theme_fg(OV_FG_WARN);
    int bot_rem = (pc + pw - 1) - ov__cursor_col;
    if (bot_rem > 0)
    {
        ov_buf_hline_utf8(OV_BOX_H, bot_rem);
    }
    ov_buf_printf("%s", OV_BOX_BR);
    ov_buf_reset_attr();
}
