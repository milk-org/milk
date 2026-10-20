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
#include <time.h>

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

/* Tab forms by width level: 0 " [ F2:DASH ] " (pill when current), 1 " F2:DASH ", 2 " F2 " */
static int tab_text(char *buf, size_t sz, int v, int level)
{
    if (level >= 2)
    {
        return snprintf(buf, sz, " F%d ", ov_views[v].fkey);
    }
    return snprintf(buf, sz, " F%d:%s ", ov_views[v].fkey, ov_views[v].label);
}

/* Draw one view tab at @level; current view as a pill (level 0) or reversed text */
static void draw_tab(int v, int level, int current)
{
    char t[48];
    tab_text(t, sizeof(t), v, level);
    ov_theme_bg(OV_BG_HEADER);
    if (level == 0)
    {
        ov_theme_fg(current ? OV_FG_TITLE : OV_FG_DIM);
        ov_buf_printf(" %s", current ? OV_LCARS_LEFT : "[");
    }
    if (current)
    {
        ov_theme_bg(OV_FG_TITLE);
        ov_theme_fg(OV_BG_TERMINAL);
    }
    else
    {
        ov_theme_fg(OV_FG_TEXT);
    }
    ov_buf_bold();
    ov_buf_printf("%s", t);
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_HEADER);
    if (level == 0)
    {
        ov_theme_fg(current ? OV_FG_TITLE : OV_FG_DIM);
        ov_buf_printf("%s ", current ? OV_LCARS_RIGHT : "]");
    }
}

/* Draw the help button " [h: HELP] " (11 columns): a pill while help is open, else a slow blink */
static void draw_help(const OV_LAYOUT *lay)
{
    ov_theme_bg(OV_BG_HEADER);
    if (lay->help.show)
    {
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
        return;
    }
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    ov_buf_printf(" ");
    if ((now_ts.tv_sec % 2) == 0)
    {
        ov_theme_bg(OV_BADGE_WARN);
        ov_theme_fg(ov_rgb_on(OV_BADGE_WARN));
    }
    else
    {
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
    }
    ov_buf_bold();
    ov_buf_printf("[h: HELP]");
    ov_buf_reset_attr();
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");
}

/**
 * ov_tabbar_level - widest tab form whose bar, help button included, fits @width columns (2 if none does).
 */
int ov_tabbar_level(int width)
{
    for (int level = 0; level < 2; level++)
    {
        int w = OV_TABBAR_HELP_W;
        for (int v = 0; v < OV_VIEW_COUNT; v++)
        {
            char t[48];
            w += tab_text(t, sizeof(t), v, level) + (level == 0 ? 4 : 0);
        }
        if (w <= width)
        {
            return level;
        }
    }
    return 2;
}

/**
 * ov_render_tabs - render the view tabs on the left and the help button on the right of row lay->r_tabs.
 * Narrow terminals get shorter tab forms (ov_tabbar_level); the help button is always kept.
 */
void ov_render_tabs(OV_LAYOUT *lay)
{
    OV_RECT r     = lay->r_tabs;
    int     level = ov_tabbar_level(r.width);
    int     used  = 0;
    ov_buf_pos(r.row, r.col);
    for (int v = 0; v < OV_VIEW_COUNT; v++)
    {
        char t[48];
        int  w = tab_text(t, sizeof(t), v, level) + (level == 0 ? 4 : 0);
        if (used + w + OV_TABBAR_HELP_W > r.width)
        {
            s_tab[v].w = 0;
            continue;
        }
        s_tab[v].col = r.col + used;
        s_tab[v].w   = w;
        used += w;
        draw_tab(v, level, v == (int) lay->view);
    }

    int pad = r.width - used - OV_TABBAR_HELP_W;
    pad     = (pad > 0) ? pad : 0;
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_hline(' ', pad);
    s_tab[OV_VIEW_COUNT].col = r.col + used + pad;
    s_tab[OV_VIEW_COUNT].w   = OV_TABBAR_HELP_W;
    draw_help(lay);
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
