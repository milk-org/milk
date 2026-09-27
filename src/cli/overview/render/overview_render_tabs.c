// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_tabs.c
 * @brief   Navigation tab bar and tooltip rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_render_tabs - render dedicated tab selection bar and prominent help button.
 * @lay: layout state
 *
 * Renders on row 2 (lay->r_tabs.row). Left side displays function key view tabs
 * ([F2:DASH] .. [F7:LOOPS]), and right side displays prominent [h: HELP] button
 * with slow blink color when idle, and active pill styling when help is open.
 */
void ov_render_tabs(OV_LAYOUT *lay)
{
    OV_RECT r = lay->r_tabs;
    ov_buf_pos(r.row, r.col);
    ov_theme_bg(OV_BG_HEADER);

    /* Render view tabs */
    int tabs_total_width = 0;
    int tab_widths[OV_VIEW_COUNT];
    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        tab_widths[v] = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
        tabs_total_width += tab_widths[v];
    }

    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        if (v == (int) lay->view)
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf(" %s", OV_LCARS_LEFT);
            ov_theme_bg(OV_FG_TITLE);
            ov_theme_fg(OV_BG_TERMINAL);
            ov_buf_bold();
            ov_buf_printf(" F%d:%s ", v + 2, ov_view_label((ov_view_t) v));
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf("%s ", OV_LCARS_RIGHT);
        }
        else
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" [");
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_bold();
            ov_buf_printf(" F%d:%s ", v + 2, ov_view_label((ov_view_t) v));
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("] ");
        }
    }

    /* Help button [h: HELP] - prominent with slow blink */
    int help_width = 11; /* visual width of " [h: HELP] " */
    int pad        = r.width - tabs_total_width - help_width;
    if (pad > 0)
    {
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', pad);
    }
    else
    {
        pad = 0;
    }

    /* Render prominent help button */
    if (lay->show_help)
    {
        /* Active state when help overlay is visible: light-blue solid pill */
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf(" %s", OV_LCARS_LEFT);
        ov_theme_bg(OV_FG_TITLE);
        ov_theme_fg(OV_BG_TERMINAL);
        ov_buf_bold();
        ov_buf_printf("h: HELP");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ov_theme_fg(OV_FG_TITLE);
        ov_buf_printf("%s ", OV_LCARS_RIGHT);
    }
    else
    {
        /* Slow blinking prominent amber badge (1s bright, 1s dim) */
        struct timespec now_ts;
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_printf(" ");
        if ((now_ts.tv_sec % 2) == 0)
        {
            /* Bright prominent state: vibrant gold/amber bg, dark crisp text */
            ov_buf_bg(240, 175, 20);
            ov_buf_fg(20, 20, 25);
        }
        else
        {
            /* Dim prominent state themed to panel bg with warning text */
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_WARN);
        }
        ov_buf_bold();
        ov_buf_printf("[h: HELP]");
        ov_buf_reset_attr();
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_printf(" ");
    }

    /* Pad trailing space if line not completely filled */
    int rendered_w = tabs_total_width + pad + help_width;
    if (rendered_w < r.width)
    {
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', r.width - rendered_w);
    }
    ov_theme_bg(OV_BG_HEADER);
}

/**
 * ov_draw_tooltip - render floating mouse hover tooltip box near cursor.
 * @lay: Pointer to layout structure.
 */
void ov_draw_tooltip(OV_LAYOUT *lay)
{
    if (!lay->mouse_hover || lay->hover_tooltip[0] == '\0')
    {
        return;
    }

    int len = (int) strlen(lay->hover_tooltip);
    if (len == 0)
    {
        return;
    }

    /* Try drawing above the cursor first */
    int tr = ov_mouse_row - 1;
    int tc = ov_mouse_col;

    /* Screen boundary clamping */
    if (tr < 0)
    {
        tr = ov_mouse_row + 1; /* flip below */
    }

    if (tc + len + 2 > lay->term_cols)
    {
        tc = lay->term_cols - len - 2;
    }
    if (tc < 0)
    {
        tc = 0;
    }

    ov_buf_pos(tr, tc);
    ov_theme_bg(OV_BG_HEADER); /* Pop out visually */
    ov_theme_fg(OV_FG_WARN);
    ov_buf_printf(" %s ", lay->hover_tooltip);

    /* Reset for next frame */
    lay->hover_tooltip[0] = '\0';
}

/**
 * ov_render_theme_popup - render interactive color theme selection popup.
 * @lay: Pointer to layout structure.
 */
