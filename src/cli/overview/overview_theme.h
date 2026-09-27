// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_theme.h
 * @brief btop-inspired dark theme for milk-CTRL
 *
 * Defines semantic color tokens used throughout the TUI.
 * Uses TrueColor (24-bit) RGB values and provides helpers
 * for gradient interpolation and sparkline rendering.
 *
 * Supports 3-tier color fallback:
 *   level 3 = TrueColor  (ov_buf_fg/ov_buf_bg)
 *   level 2 = 256-color  (ov_buf_fg_256/ov_buf_bg_256)
 *   level 1 = 16-color   (ANSI 30-37/90-97)
 */

#ifndef OVERVIEW_THEME_H
#define OVERVIEW_THEME_H

#include "overview_ansi.h"
#include "overview_data.h"

/* =========================================================
 * RGB color struct
 * ========================================================= */

typedef struct
{
    int r;
    int g;
    int b;
} ov_rgb_t;

/* =========================================================
 * Theme structure & palette
 * ========================================================= */

typedef struct
{
    const char *id;
    const char *name;
    const char *desc;

    /* Panel backgrounds */
    ov_rgb_t bg_terminal;
    ov_rgb_t bg_panel;
    ov_rgb_t bg_panel_alt;
    ov_rgb_t bg_header;
    ov_rgb_t bg_selected;
    ov_rgb_t bg_related;
    ov_rgb_t bg_frozen;
    ov_rgb_t bg_hover;
    ov_rgb_t bg_pid_match;
    ov_rgb_t bg_stale;
    ov_rgb_t bg_new_item;
    ov_rgb_t bg_loop;
    ov_rgb_t bg_loop_shared;

    /* Foreground — text */
    ov_rgb_t fg_title;
    ov_rgb_t fg_dim;
    ov_rgb_t fg_text;
    ov_rgb_t fg_bright;
    ov_rgb_t fg_muted;

    /* Foreground — node types */
    ov_rgb_t fg_stream;
    ov_rgb_t fg_fps;
    ov_rgb_t fg_proc;

    /* Dimmed accent colors for column headers */
    ov_rgb_t fg_stream_hdr;
    ov_rgb_t fg_fps_hdr;
    ov_rgb_t fg_proc_hdr;

    /* Foreground — status */
    ov_rgb_t fg_active;
    ov_rgb_t fg_idle;

    /* Animation pulse parameters & status fg */
    ov_rgb_t anim_pulse_bg_min;
    ov_rgb_t anim_pulse_bg_max;
    ov_rgb_t anim_pulse_fg_min;
    ov_rgb_t anim_pulse_fg_max;
    ov_rgb_t fg_warn;
    ov_rgb_t fg_error;
    ov_rgb_t fg_zombie;

    /* Foreground — graph & loops */
    ov_rgb_t fg_conn;
    ov_rgb_t fg_edge_active;
    ov_rgb_t fg_loop;
    ov_rgb_t fg_loop_shared;

    /* Gradient endpoints for bars/sparklines */
    ov_rgb_t grad_lo;
    ov_rgb_t grad_hi;
    ov_rgb_t grad_cpu_lo;
    ov_rgb_t grad_cpu_hi;
} ov_theme_t;

extern const ov_theme_t *ov_active_theme;

/* Theme API declarations */
int               ov_theme_count(void);
const ov_theme_t *ov_theme_get(int index);
const ov_theme_t *ov_theme_get_active(void);
int               ov_theme_get_active_index(void);
int               ov_theme_find_by_id(const char *id);
void              ov_theme_set(int index);
void              ov_theme_cycle(void);
void              ov_theme_init(const char *preferred_theme);
void              ov_theme_save_preference(void);

/* =========================================================
 * Semantic color palette (maps to active theme)
 * ========================================================= */

/* Panel backgrounds */
#define OV_BG_TERMINAL (ov_active_theme->bg_terminal)
#define OV_BG_PANEL (ov_active_theme->bg_panel)
#define OV_BG_PANEL_ALT (ov_active_theme->bg_panel_alt)
#define OV_BG_HEADER (ov_active_theme->bg_header)
#define OV_BG_SELECTED (ov_active_theme->bg_selected)
#define OV_BG_RELATED (ov_active_theme->bg_related)
#define OV_BG_FROZEN (ov_active_theme->bg_frozen)
#define OV_BG_HOVER (ov_active_theme->bg_hover)
#define OV_BG_PID_MATCH (ov_active_theme->bg_pid_match)
#define OV_BG_STALE (ov_active_theme->bg_stale)
#define OV_BG_NEW_ITEM (ov_active_theme->bg_new_item)
#define OV_BG_LOOP (ov_active_theme->bg_loop)
#define OV_BG_LOOP_SHARED (ov_active_theme->bg_loop_shared)

/* Foreground — text */
#define OV_FG_TITLE (ov_active_theme->fg_title)
#define OV_FG_DIM (ov_active_theme->fg_dim)
#define OV_FG_TEXT (ov_active_theme->fg_text)
#define OV_FG_BRIGHT (ov_active_theme->fg_bright)
#define OV_FG_MUTED (ov_active_theme->fg_muted)

/* Foreground — node types */
#define OV_FG_STREAM (ov_active_theme->fg_stream)
#define OV_FG_FPS (ov_active_theme->fg_fps)
#define OV_FG_PROC (ov_active_theme->fg_proc)

/* Dimmed accent colors for column headers */
#define OV_FG_STREAM_HDR (ov_active_theme->fg_stream_hdr)
#define OV_FG_FPS_HDR (ov_active_theme->fg_fps_hdr)
#define OV_FG_PROC_HDR (ov_active_theme->fg_proc_hdr)

/* Foreground — status */
#define OV_FG_ACTIVE (ov_active_theme->fg_active)
#define OV_FG_IDLE (ov_active_theme->fg_idle)

/* Animation Parameters */
#define OV_ANIM_PULSE_SPEED 0.15f
#define OV_ANIM_PULSE_BG_MIN (ov_active_theme->anim_pulse_bg_min)
#define OV_ANIM_PULSE_BG_MAX (ov_active_theme->anim_pulse_bg_max)
#define OV_ANIM_PULSE_FG_MIN (ov_active_theme->anim_pulse_fg_min)
#define OV_ANIM_PULSE_FG_MAX (ov_active_theme->anim_pulse_fg_max)
#define OV_FG_WARN (ov_active_theme->fg_warn)
#define OV_FG_ERROR (ov_active_theme->fg_error)
#define OV_FG_ZOMBIE (ov_active_theme->fg_zombie)

/* Foreground — graph & loops */
#define OV_FG_CONN (ov_active_theme->fg_conn)
#define OV_FG_EDGE_ACTIVE (ov_active_theme->fg_edge_active)
#define OV_FG_LOOP (ov_active_theme->fg_loop)
#define OV_FG_LOOP_SHARED (ov_active_theme->fg_loop_shared)

/* Gradient endpoints for bars/sparklines */
#define OV_GRAD_LO (ov_active_theme->grad_lo)
#define OV_GRAD_HI (ov_active_theme->grad_hi)

#define OV_GRAD_CPU_LO (ov_active_theme->grad_cpu_lo)
#define OV_GRAD_CPU_HI (ov_active_theme->grad_cpu_hi)

/* =========================================================
 * Borders & box-drawing
 * ========================================================= */

#define OV_BOX_TL "╭"
#define OV_BOX_TR "╮"
#define OV_BOX_BL "╰"
#define OV_BOX_BR "╯"
#define OV_BOX_H "─"
#define OV_BOX_V "│"

/* Double borders for focused panels */
#define OV_BOX_TL_D "╔"
#define OV_BOX_TR_D "╗"
#define OV_BOX_BL_D "╚"
#define OV_BOX_BR_D "╝"
#define OV_BOX_H_D "═"
#define OV_BOX_V_D "║"
#define OV_BOX_LT "├"
#define OV_BOX_RT "┤"
#define OV_BOX_TB "┬"
#define OV_BOX_BT "┴"
#define OV_BOX_X "┼"

/* Arrow characters */
#define OV_ARROW_R "→"
#define OV_ARROW_L "←"
#define OV_ARROW_D "↓"
#define OV_ARROW_U "↑"
#define OV_TRI_R "▶"
#define OV_TRI_L "◀"
#define OV_TRI_D "▼"
#define OV_TRI_U "▲"
#define OV_BULLET "●"
#define OV_DIAMOND "◆"

/* LCARS block elements */
#define OV_LCARS_LEFT "▌"
#define OV_LCARS_RIGHT "▐"

/* Sparkline block characters (1/8 to full) */
static const char *OV_SPARK_CHARS[] = { " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" };
#define OV_SPARK_LEVELS 9

/* =========================================================
 * Helper: emit themed fg/bg
 * ========================================================= */

static inline void ov_theme_fg(ov_rgb_t c)
{
    ov_buf_fg(c.r, c.g, c.b);
}

static inline void ov_theme_bg(ov_rgb_t c)
{
    ov_buf_bg(c.r, c.g, c.b);
}

static inline void ov_theme_ul(ov_rgb_t c)
{
    ov_buf_ul_color(c.r, c.g, c.b);
}

/**
 * ov_pid_color - uniform PID coloring.
 *
 * Returns OV_FG_ACTIVE for alive processes,
 * OV_FG_ZOMBIE for zombie processes,
 * OV_FG_DIM for dead/zero PIDs.
 */
static inline ov_rgb_t ov_pid_color(pid_t pid)
{
    if (pid <= 0)
    {
        return OV_FG_DIM;
    }
    ov_pid_status_t st = pid_get_status(pid);
    switch (st)
    {
    case OV_PID_ALIVE:
        return OV_FG_ACTIVE;
    case OV_PID_ZOMBIE:
        return OV_FG_ZOMBIE;
    default:
        return OV_FG_DIM;
    }
}

/* =========================================================
 * Helper: gradient interpolation
 * ========================================================= */

/**
 * ov_rgb_lerp - linear interpolation between two colors.
 * @a:   start color
 * @b:   end color
 * @t:   factor 0.0 (=a) to 1.0 (=b)
 */
static inline ov_rgb_t ov_rgb_lerp(ov_rgb_t a, ov_rgb_t b, float t)
{
    if (t < 0.0f)
    {
        t = 0.0f;
    }
    if (t > 1.0f)
    {
        t = 1.0f;
    }
    return (ov_rgb_t) {
        a.r + (int) ((float) (b.r - a.r) * t),
        a.g + (int) ((float) (b.g - a.g) * t),
        a.b + (int) ((float) (b.b - a.b) * t),
    };
}

/**
 * ov_buf_gradient_bar - render a horizontal gradient bar.
 * @row:   screen row
 * @col:   start column
 * @len:   total bar width (characters)
 * @fill:  fraction filled 0.0 to 1.0
 * @lo:    color at 0%
 * @hi:    color at 100%
 */
static inline void ov_buf_gradient_bar(int      row,
                                       int      col,
                                       int      len,
                                       float    fill,
                                       ov_rgb_t lo,
                                       ov_rgb_t hi)
{
    if (fill < 0.0f)
    {
        fill = 0.0f;
    }
    if (fill > 1.0f)
    {
        fill = 1.0f;
    }
    int filled = (int) (fill * (float) len + 0.5f);

    ov_buf_pos(row, col);
    for (int i = 0; i < len; i++)
    {
        if (i < filled)
        {
            float    t = (len > 1) ? (float) i / (float) (len - 1) : 0.0f;
            ov_rgb_t c = ov_rgb_lerp(lo, hi, t);
            ov_theme_bg(c);
            ov_buf_printf(" ");
        }
        else
        {
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_MUTED);
            ov_buf_printf("%s", OV_BOX_H);
        }
    }
    ov_buf_reset_attr();
}

/**
 * ov_buf_printf_gradient - print text with a gradient foreground color.
 * @a:   start color
 * @b:   end color
 * @fmt: format string
 */
static inline void ov_buf_printf_gradient(ov_rgb_t a, ov_rgb_t b, const char *fmt, ...)
{
    char    tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    if (n > 0)
    {
        if (n >= (int) sizeof(tmp))
        {
            n = (int) sizeof(tmp) - 1;
        }

        int total_chars = 0;
        int i           = 0;
        while (i < n)
        {
            int nb = 0, w = 1;
            ov_utf8_next_cluster(&tmp[i], n - i, &nb, &w);
            if (nb <= 0)
            {
                break;
            }
            total_chars++;
            i += nb;
        }

        i            = 0;
        int char_idx = 0;
        while (i < n)
        {
            int nb = 0, w = 1;
            ov_utf8_next_cluster(&tmp[i], n - i, &nb, &w);
            if (nb <= 0)
            {
                break;
            }

            float t = (total_chars > 1) ? (float) char_idx / (float) (total_chars - 1) : 0.0f;
            ov_theme_fg(ov_rgb_lerp(a, b, t));

            ov_buf_append_cluster(&tmp[i], nb, w);
            i += nb;
            char_idx++;
        }
    }
}

/**
 * ov_buf_sparkline - draw a sparkline from a value array.
 * @row:     screen row
 * @col:     start column
 * @vals:    array of values [0.0, 1.0]
 * @len:     number of values to render
 * @color:   foreground color
 */
static inline void ov_buf_sparkline(int row, int col, const float *vals, int len, ov_rgb_t color)
{
    ov_buf_pos(row, col);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(color);

    for (int i = 0; i < len; i++)
    {
        float v = vals[i];
        if (v < 0.0f)
        {
            v = 0.0f;
        }
        if (v > 1.0f)
        {
            v = 1.0f;
        }
        int idx = (int) (v * (float) (OV_SPARK_LEVELS - 1) + 0.5f);
        ov_buf_printf("%s", OV_SPARK_CHARS[idx]);
    }
    ov_buf_reset_attr();
}

/* =========================================================
 * Helper: draw a rounded panel border
 * ========================================================= */

/**
 * ov_draw_panel_border - draw a panel frame with title.
 * @row:    top-left row
 * @col:    top-left column
 * @height: total panel height
 * @width:  total panel width
 * @title:  title string (NULL = no title)
 * @tcolor: title text color
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
 * ov_draw_panel_tabs - draw a panel frame with multiple tabs.
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
        current_col += strlen(tab_text) + 1; // 1 space between tabs
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

static inline ov_rgb_t ov_theme_highlight_bg(ov_rgb_t base_bg)
{
    ov_rgb_t highlight;
    int sum = base_bg.r + base_bg.g + base_bg.b;
    if (sum > 384)
    {
        /* Light palette: darken slightly */
        highlight.r = (base_bg.r > 18) ? base_bg.r - 18 : 0;
        highlight.g = (base_bg.g > 18) ? base_bg.g - 18 : 0;
        highlight.b = (base_bg.b > 15) ? base_bg.b - 15 : 0;
    }
    else
    {
        /* Dark palette: lighten slightly */
        highlight.r = (base_bg.r + 15 <= 255) ? base_bg.r + 15 : 255;
        highlight.g = (base_bg.g + 15 <= 255) ? base_bg.g + 15 : 255;
        highlight.b = (base_bg.b + 18 <= 255) ? base_bg.b + 18 : 255;
    }
    return highlight;
}

#endif /* OVERVIEW_THEME_H */
