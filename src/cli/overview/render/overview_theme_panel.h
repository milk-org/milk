// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_theme_panel.h
 * @brief Panel border, header, filter badge, and tab rendering helpers.
 */

#ifndef OVERVIEW_THEME_PANEL_H
#define OVERVIEW_THEME_PANEL_H

#include "overview_theme_draw.h"

/**
 * ov_draw_panel_border - draw a panel frame with title.
 * @row:         top-left row
 * @col:         top-left column
 * @height:      total panel height
 * @width:       total panel width
 * @title:       title string (NULL = no title)
 * @tcolor:      title text color
 * @is_focused:  1 if panel has focus, 0 otherwise
 * @drop_shadow: 1 to draw drop shadow, 0 otherwise
 */
static inline void ov_draw_panel_border(int         row,
                                        int         col,
                                        int         height,
                                        int         width,
                                        const char *title,
                                        ov_rgb_t    tcolor,
                                        int         is_focused,
                                        int         drop_shadow)
{
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_theme_bg(OV_BG_TERMINAL);

    const char *tl = is_focused ? OV_BOX_TL_D : OV_BOX_TL;
    const char *tr = is_focused ? OV_BOX_TR_D : OV_BOX_TR;
    const char *bl = is_focused ? OV_BOX_BL_D : OV_BOX_BL;
    const char *br = is_focused ? OV_BOX_BR_D : OV_BOX_BR;
    const char *h  = is_focused ? OV_BOX_H_D : OV_BOX_H;
    const char *v  = is_focused ? OV_BOX_V_D : OV_BOX_V;

    /* top edge */
    ov_buf_pos(row, col);
    ov_buf_printf("%s", tl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", tr);

    /* title overlay */
    if (title && title[0])
    {
        ov_buf_pos(row, col + 2);
        ov_buf_bold();
        if (is_focused)
        {
            ov_theme_bg(tcolor);
            ov_theme_fg(OV_BG_TERMINAL);
            ov_buf_printf(" %s ", title);
        }
        else
        {
            ov_theme_fg(OV_FG_MUTED);
            ov_theme_bg(OV_BG_TERMINAL);
            ov_buf_printf(" %s ", title);
        }
        ov_buf_reset_attr();
    }

    /* sides */
    for (int r = row + 1; r < row + height - 1; r++)
    {
        ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_pos(r, col);
        ov_buf_printf("%s", v);
        ov_buf_pos(r, col + width - 1);
        ov_buf_printf("%s", v);
    }

    /* bottom edge */
    ov_buf_pos(row + height - 1, col);
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_buf_printf("%s", bl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", br);

    /* drop shadow */
    if (drop_shadow)
    {
        ov_theme_fg(OV_FG_DIM);
        ov_theme_bg(OV_BG_TERMINAL);
        /* bottom shadow */
        ov_buf_pos(row + height, col + 1);
        ov_buf_hline_utf8("▒", width);
        /* right shadow */
        for (int r = row + 1; r < row + height; r++)
        {
            ov_buf_pos(r, col + width);
            ov_buf_printf("▒");
        }
        ov_buf_pos(row + height, col + width);
        ov_buf_printf("▒");
    }

    ov_buf_reset_attr();
}

/**
 * ov_draw_panel_border_filter - draw panel frame with title and prominent filter badge.
 * @row:           top-left row
 * @col:           top-left column
 * @height:        total panel height
 * @width:         total panel width
 * @title:         base panel name (e.g. "STREAMS", "PROCESSINFO", "FPS")
 * @tcolor:        panel theme color
 * @is_focused:    1 if panel has focus, 0 otherwise
 * @drop_shadow:   1 to draw drop shadow, 0 otherwise
 * @ctrl_blink:    global blink tick for animated indicators
 * @loop_id:       loop ID if loop isolation active, or -1
 * @filter_pat:    regex filter pattern string, or NULL/""
 * @filter_active: 1 if regex filter is actively applied, 0 if paused/off
 * @filt_count:    number of items visible after filter
 * @total_count:   total number of items in panel
 */
static inline void ov_draw_panel_border_filter(int         row,
                                               int         col,
                                               int         height,
                                               int         width,
                                               const char *title,
                                               ov_rgb_t    tcolor,
                                               int         is_focused,
                                               int         drop_shadow,
                                               uint32_t    ctrl_blink,
                                               int         loop_id,
                                               const char *filter_pat,
                                               int         filter_active,
                                               int         filt_count,
                                               int         total_count)
{
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_theme_bg(OV_BG_TERMINAL);

    const char *tl = is_focused ? OV_BOX_TL_D : OV_BOX_TL;
    const char *tr = is_focused ? OV_BOX_TR_D : OV_BOX_TR;
    const char *bl = is_focused ? OV_BOX_BL_D : OV_BOX_BL;
    const char *br = is_focused ? OV_BOX_BR_D : OV_BOX_BR;
    const char *h  = is_focused ? OV_BOX_H_D : OV_BOX_H;
    const char *v  = is_focused ? OV_BOX_V_D : OV_BOX_V;

    /* top edge */
    ov_buf_pos(row, col);
    ov_buf_printf("%s", tl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", tr);

    /* title overlay */
    if (title && title[0])
    {
        ov_buf_pos(row, col + 2);
        ov_buf_bold();
        if (is_focused)
        {
            ov_theme_bg(tcolor);
            ov_theme_fg(OV_BG_TERMINAL);
            ov_buf_printf(" %s ", title);
        }
        else
        {
            ov_theme_fg(OV_FG_MUTED);
            ov_theme_bg(OV_BG_TERMINAL);
            ov_buf_printf(" %s ", title);
        }
        ov_buf_reset_attr();

        int has_filter_pat = (filter_pat != NULL && filter_pat[0] != '\0');
        if (loop_id >= 0)
        {
            ov_buf_bold();
            ov_theme_bg(OV_FG_LOOP);
            ov_theme_fg(OV_BG_TERMINAL);
            ov_buf_printf(" [LOOP L%02d] ", loop_id);
            ov_buf_reset_attr();
            ov_theme_fg(OV_FG_LOOP);
            ov_theme_bg(OV_BG_TERMINAL);
            ov_buf_printf(" (%d/%d) ", filt_count, total_count);
            ov_buf_reset_attr();
        }
        else if (filter_active && has_filter_pat)
        {
            /* HIGH-VISIBILITY FILTER ON BADGE — visible whether focused or unselected! */
            ov_buf_bold();
            if (is_focused)
            {
                if ((ctrl_blink % 4) < 2)
                {
                    ov_buf_bg(255, 190, 0); /* bright amber/gold */
                    ov_buf_fg(20, 20, 20);  /* dark text */
                }
                else
                {
                    ov_buf_bg(230, 80, 20);   /* vibrant red-orange */
                    ov_buf_fg(255, 255, 255); /* white text */
                }
            }
            else
            {
                /* Unselected panel: solid vivid amber-orange pill */
                ov_buf_bg(220, 130, 20);
                ov_buf_fg(255, 255, 255);
            }
            ov_buf_printf(" [FILTER ON: /%.12s/] ", filter_pat);
            ov_buf_reset_attr();

            ov_buf_bold();
            ov_theme_fg(OV_FG_WARN);
            ov_theme_bg(OV_BG_TERMINAL);
            ov_buf_printf(" (%d/%d) ", filt_count, total_count);
            ov_buf_reset_attr();
        }
        else if (has_filter_pat)
        {
            /* Paused filter (query retained but disabled) */
            ov_buf_bold();
            ov_theme_bg(OV_BG_PANEL_ALT);
            ov_theme_fg(OV_FG_WARN);
            ov_buf_printf(" [FILTER OFF: /%.8s/] ", filter_pat);
            ov_buf_reset_attr();

            ov_theme_fg(OV_FG_DIM);
            ov_theme_bg(OV_BG_TERMINAL);
            ov_buf_printf(" (%d) ", total_count);
            ov_buf_reset_attr();
        }
        else
        {
            ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
            ov_theme_bg(OV_BG_TERMINAL);
            ov_buf_printf(" (%d) ", total_count);
            ov_buf_reset_attr();
        }
    }

    /* sides */
    for (int r = row + 1; r < row + height - 1; r++)
    {
        ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_pos(r, col);
        ov_buf_printf("%s", v);
        ov_buf_pos(r, col + width - 1);
        ov_buf_printf("%s", v);
    }

    /* bottom edge */
    ov_buf_pos(row + height - 1, col);
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_buf_printf("%s", bl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", br);

    /* drop shadow */
    if (drop_shadow)
    {
        ov_theme_fg(OV_FG_DIM);
        ov_theme_bg(OV_BG_TERMINAL);
        /* bottom shadow */
        ov_buf_pos(row + height, col + 1);
        ov_buf_hline_utf8("▒", width);
        /* right shadow */
        for (int r = row + 1; r < row + height; r++)
        {
            ov_buf_pos(r, col + width);
            ov_buf_printf("▒");
        }
        ov_buf_pos(row + height, col + width);
        ov_buf_printf("▒");
    }

    ov_buf_reset_attr();
}

/**
 * ov_draw_panel_tabs - draw a panel frame with multiple tabs.
 * @row:        top-left row
 * @col:        top-left column
 * @height:     total panel height
 * @width:      total panel width
 * @tabs:       array of tab label strings
 * @num_tabs:   number of tabs
 * @active_tab: index of currently active tab
 * @tcolor:     accent theme color
 * @is_focused: 1 if panel has focus, 0 otherwise
 */
static inline void ov_draw_panel_tabs(int          row,
                                      int          col,
                                      int          height,
                                      int          width,
                                      const char **tabs,
                                      int          num_tabs,
                                      int          active_tab,
                                      ov_rgb_t     tcolor,
                                      int          is_focused)
{
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_theme_bg(OV_BG_TERMINAL);

    const char *tl = is_focused ? OV_BOX_TL_D : OV_BOX_TL;
    const char *tr = is_focused ? OV_BOX_TR_D : OV_BOX_TR;
    const char *bl = is_focused ? OV_BOX_BL_D : OV_BOX_BL;
    const char *br = is_focused ? OV_BOX_BR_D : OV_BOX_BR;
    const char *h  = is_focused ? OV_BOX_H_D : OV_BOX_H;
    const char *v  = is_focused ? OV_BOX_V_D : OV_BOX_V;

    /* top edge */
    ov_buf_pos(row, col);
    ov_buf_printf("%s", tl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", tr);

    /* title overlay: rendering tabs */
    int current_col = col + 2;
    for (int i = 0; i < num_tabs; i++)
    {
        ov_buf_pos(row, current_col);
        ov_buf_bold();
        if (i == active_tab)
        {
            if (is_focused)
            {
                ov_theme_bg(tcolor);
                ov_theme_fg(OV_BG_TERMINAL);
            }
            else
            {
                ov_theme_bg(OV_FG_DIM);
                ov_theme_fg(OV_BG_TERMINAL);
            }
        }
        else
        {
            ov_theme_fg(OV_FG_MUTED);
            ov_theme_bg(OV_BG_TERMINAL);
        }

        char tab_text[64];
        snprintf(tab_text, sizeof(tab_text), " %s ", tabs[i]);
        ov_buf_printf("%s", tab_text);

        ov_buf_reset_attr();
        current_col += (int) strlen(tab_text) + 1; // 1 space between tabs
    }

    if (current_col + 15 < col + width)
    {
        ov_buf_pos(row, current_col + 1);
        ov_theme_fg(OV_FG_MUTED);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_printf("(Click tab)");
    }

    /* sides */
    for (int r = row + 1; r < row + height - 1; r++)
    {
        ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_pos(r, col);
        ov_buf_printf("%s", v);
        ov_buf_pos(r, col + width - 1);
        ov_buf_printf("%s", v);
    }

    /* bottom edge */
    ov_buf_pos(row + height - 1, col);
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_buf_printf("%s", bl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", br);

    ov_buf_reset_attr();
}

#endif /* OVERVIEW_THEME_PANEL_H */
