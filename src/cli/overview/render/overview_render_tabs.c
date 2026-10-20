// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_tabs.c
 * @brief   Navigation tab bar and tooltip rendering for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* Tab bar cells as last drawn (1-based screen column, width): the views, then the help button */
static struct
{
    int col;
    int w;
} s_tab[OV_VIEW_COUNT + 1];

/** ov_tabbar_hit - view tab under screen column @mc as last drawn, OV_TABBAR_HELP, or -1. */
int ov_tabbar_hit(int mc)
{
    for (int v = 0; v <= OV_VIEW_COUNT; v++)
    {
        if (s_tab[v].w > 0 && mc >= s_tab[v].col && mc < s_tab[v].col + s_tab[v].w)
        {
            return (v == OV_VIEW_COUNT) ? OV_TABBAR_HELP : v;
        }
    }
    return -1;
}

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

    /* View tabs: " [ F2:DASH ] ", or the pill " ▌ F2:DASH ▐ " for the current view */
    int tabs_total_width = 0;
    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        s_tab[v].col = r.col + tabs_total_width;
        s_tab[v].w   = (int) strlen(ov_views[v].label) + 9;
        tabs_total_width += s_tab[v].w;
        if (v == (int) lay->view)
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_printf(" %s", OV_LCARS_LEFT);
            ov_theme_bg(OV_FG_TITLE);
            ov_theme_fg(OV_BG_TERMINAL);
            ov_buf_bold();
            ov_buf_printf(" F%d:%s ", ov_views[v].fkey, ov_views[v].label);
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
            ov_buf_printf(" F%d:%s ", ov_views[v].fkey, ov_views[v].label);
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
    s_tab[OV_VIEW_COUNT].col = r.col + tabs_total_width + pad;
    s_tab[OV_VIEW_COUNT].w   = help_width;

    /* Render prominent help button */
    if (lay->help.show)
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
            ov_theme_bg(OV_BADGE_WARN);
            ov_theme_fg(ov_rgb_on(OV_BADGE_WARN));
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

static char s_row_tooltip[256]; /* set by the row under the mouse while the frame is drawn */

/** ov_row_tooltip - set the tooltip of the hovered row for this frame (overrides the hit-test text). */
void ov_row_tooltip(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_row_tooltip, sizeof(s_row_tooltip), fmt, ap);
    va_end(ap);
}

/**
 * ov_draw_tooltip - render floating mouse hover tooltip box near cursor.
 * @lay: Pointer to layout structure.
 */
void ov_draw_tooltip(OV_LAYOUT *lay)
{
    const char *text = (s_row_tooltip[0] != '\0') ? s_row_tooltip : lay->hover_tooltip;
    if (!lay->mouse_hover || text[0] == '\0')
    {
        s_row_tooltip[0] = '\0';
        return;
    }

    int len = (int) strlen(text);
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
    ov_buf_printf(" %s ", text);

    /* Reset for next frame */
    lay->hover_tooltip[0] = '\0';
    s_row_tooltip[0]      = '\0';
}

/**
 * ov_render_theme_popup - render interactive color theme selection popup.
 * @lay: Pointer to layout structure.
 */
