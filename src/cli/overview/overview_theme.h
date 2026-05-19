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
 * Semantic color palette
 * ========================================================= */

/* Panel backgrounds */
#define OV_BG_TERMINAL    (ov_rgb_t){  20,  22,  28 }
#define OV_BG_PANEL       (ov_rgb_t){  30,  32,  40 }
#define OV_BG_PANEL_ALT   (ov_rgb_t){  25,  27,  35 }
#define OV_BG_HEADER      (ov_rgb_t){  40,  44,  58 }
#define OV_BG_SELECTED    (ov_rgb_t){  50,  60,  90 }
#define OV_BG_RELATED     (ov_rgb_t){  38,  50,  42 }  /* soft green tint for related items */
#define OV_BG_FROZEN      (ov_rgb_t){  40,  90, 140 }  /* bright blue tint for frozen selection */
#define OV_BG_HOVER       (ov_rgb_t){  38,  42,  55 }
#define OV_BG_PID_MATCH   (ov_rgb_t){  50, 180,  50 }  /* green bg for PID match */
#define OV_BG_STALE       (ov_rgb_t){  55,  45,  20 }  /* amber tint for stale procs */
#define OV_BG_NEW_ITEM    (ov_rgb_t){  40,  60,  50 }  /* green flash for new items */

/* Foreground — text */
#define OV_FG_TITLE       (ov_rgb_t){ 130, 170, 255 }
#define OV_FG_DIM         (ov_rgb_t){ 100, 105, 120 }
#define OV_FG_TEXT        (ov_rgb_t){ 200, 205, 215 }
#define OV_FG_BRIGHT      (ov_rgb_t){ 240, 245, 255 }
#define OV_FG_MUTED       (ov_rgb_t){  70,  75,  85 }

/* Foreground — node types */
#define OV_FG_STREAM      (ov_rgb_t){  80, 200, 220 }
#define OV_FG_FPS         (ov_rgb_t){ 130, 170, 255 }
#define OV_FG_PROC        (ov_rgb_t){ 180, 140, 255 }

/* Dimmed accent colors for column headers */
#define OV_FG_STREAM_HDR  (ov_rgb_t){  55, 140, 155 }
#define OV_FG_FPS_HDR     (ov_rgb_t){  90, 120, 180 }
#define OV_FG_PROC_HDR    (ov_rgb_t){ 125, 100, 180 }

/* Foreground — status */
#define OV_FG_ACTIVE      (ov_rgb_t){  80, 220,  80 }
#define OV_FG_IDLE        (ov_rgb_t){ 130, 140, 160 }

/* Animation Parameters */
#define OV_ANIM_PULSE_SPEED    0.15f
#define OV_ANIM_PULSE_BG_MIN   (ov_rgb_t){  80,  10,  10 }
#define OV_ANIM_PULSE_BG_MAX   (ov_rgb_t){ 180,  20,  20 }
#define OV_ANIM_PULSE_FG_MIN   (ov_rgb_t){ 160,  80,  80 }
#define OV_ANIM_PULSE_FG_MAX   (ov_rgb_t){ 255, 220, 220 }
#define OV_FG_WARN        (ov_rgb_t){ 255, 180,   0 }
#define OV_FG_ERROR       (ov_rgb_t){ 240,  60,  60 }
#define OV_FG_ZOMBIE      (ov_rgb_t){ 180, 120,  40 }

/* Foreground — graph */
#define OV_FG_CONN        (ov_rgb_t){ 100, 130, 180 }
#define OV_FG_EDGE_ACTIVE (ov_rgb_t){ 140, 200, 255 }

/* Gradient endpoints for bars/sparklines */
#define OV_GRAD_LO        (ov_rgb_t){  60,  90, 140 }
#define OV_GRAD_HI        (ov_rgb_t){ 100, 200, 255 }

#define OV_GRAD_CPU_LO    (ov_rgb_t){  60, 180,  60 }
#define OV_GRAD_CPU_HI    (ov_rgb_t){ 240,  60,  60 }

/* =========================================================
 * Borders & box-drawing
 * ========================================================= */

#define OV_BOX_TL   "╭"
#define OV_BOX_TR   "╮"
#define OV_BOX_BL   "╰"
#define OV_BOX_BR   "╯"
#define OV_BOX_H    "─"
#define OV_BOX_V    "│"

/* Double borders for focused panels */
#define OV_BOX_TL_D "╔"
#define OV_BOX_TR_D "╗"
#define OV_BOX_BL_D "╚"
#define OV_BOX_BR_D "╝"
#define OV_BOX_H_D  "═"
#define OV_BOX_V_D  "║"
#define OV_BOX_LT   "├"
#define OV_BOX_RT   "┤"
#define OV_BOX_TB   "┬"
#define OV_BOX_BT   "┴"
#define OV_BOX_X    "┼"

/* Arrow characters */
#define OV_ARROW_R  "→"
#define OV_ARROW_L  "←"
#define OV_ARROW_D  "↓"
#define OV_ARROW_U  "↑"
#define OV_TRI_R    "▶"
#define OV_TRI_L    "◀"
#define OV_TRI_D    "▼"
#define OV_TRI_U    "▲"
#define OV_BULLET   "●"
#define OV_DIAMOND  "◆"

/* LCARS block elements */
#define OV_LCARS_LEFT  "▌"
#define OV_LCARS_RIGHT "▐"

/* Sparkline block characters (1/8 to full) */
static const char *OV_SPARK_CHARS[] =
{
    " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"
};
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
    if(pid <= 0)
    {
        return OV_FG_DIM;
    }
    ov_pid_status_t st = pid_get_status(pid);
    switch(st)
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
static inline ov_rgb_t ov_rgb_lerp(
    ov_rgb_t a,
    ov_rgb_t b,
    float    t)
{
    if(t < 0.0f)
    {
        t = 0.0f;
    }
    if(t > 1.0f)
    {
        t = 1.0f;
    }
    return (ov_rgb_t)
    {
        a.r + (int)((float)(b.r - a.r) * t),
            a.g + (int)((float)(b.g - a.g) * t),
            a.b + (int)((float)(b.b - a.b) * t),
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
static inline void ov_buf_gradient_bar(
    int      row,
    int      col,
    int      len,
    float    fill,
    ov_rgb_t lo,
    ov_rgb_t hi)
{
    if(fill < 0.0f)
    {
        fill = 0.0f;
    }
    if(fill > 1.0f)
    {
        fill = 1.0f;
    }
    int filled = (int)(fill * (float) len + 0.5f);

    ov_buf_pos(row, col);
    for(int i = 0; i < len; i++)
    {
        if(i < filled)
        {
            float t = (len > 1)
                      ? (float) i / (float)(len - 1)
                      : 0.0f;
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
static inline void ov_buf_printf_gradient(
    ov_rgb_t a,
    ov_rgb_t b,
    const char *fmt,
    ...)
{
    char tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    if(n > 0)
    {
        if(n >= (int)sizeof(tmp))
        {
            n = (int)sizeof(tmp) - 1;
        }

        int total_chars = 0;
        int i = 0;
        while(i < n)
        {
            int char_len = utf8_char_length((unsigned char)tmp[i]);
            if(i + char_len > n)
            {
                char_len = n - i;
            }
            total_chars++;
            i += char_len;
        }

        i = 0;
        int char_idx = 0;
        while(i < n)
        {
            int char_len = utf8_char_length((unsigned char)tmp[i]);
            if(i + char_len > n)
            {
                char_len = n - i;
            }

            float t = (total_chars > 1) ? (float)char_idx / (float)(total_chars - 1) : 0.0f;
            ov_theme_fg(ov_rgb_lerp(a, b, t));

            ov_buf_append_char(&tmp[i], char_len);
            i += char_len;
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
static inline void ov_buf_sparkline(
    int         row,
    int         col,
    const float *vals,
    int         len,
    ov_rgb_t    color)
{
    ov_buf_pos(row, col);
    ov_theme_bg(OV_BG_PANEL);
    ov_theme_fg(color);

    for(int i = 0; i < len; i++)
    {
        float v = vals[i];
        if(v < 0.0f)
        {
            v = 0.0f;
        }
        if(v > 1.0f)
        {
            v = 1.0f;
        }
        int idx = (int)(v *
                        (float)(OV_SPARK_LEVELS - 1)
                        + 0.5f);
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
static inline void ov_draw_panel_border(
    int        row,
    int        col,
    int        height,
    int        width,
    const char *title,
    ov_rgb_t   tcolor,
    int        is_focused,
    int        drop_shadow)
{
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_theme_bg(OV_BG_TERMINAL);

    const char *tl = is_focused ? OV_BOX_TL_D : OV_BOX_TL;
    const char *tr = is_focused ? OV_BOX_TR_D : OV_BOX_TR;
    const char *bl = is_focused ? OV_BOX_BL_D : OV_BOX_BL;
    const char *br = is_focused ? OV_BOX_BR_D : OV_BOX_BR;
    const char *h  = is_focused ? OV_BOX_H_D  : OV_BOX_H;
    const char *v  = is_focused ? OV_BOX_V_D  : OV_BOX_V;

    /* top edge */
    ov_buf_pos(row, col);
    ov_buf_printf("%s", tl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", tr);

    /* title overlay */
    if(title && title[0])
    {
        ov_buf_pos(row, col + 2);
        ov_buf_bold();
        if(is_focused)
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
    for(int r = row + 1; r < row + height - 1; r++)
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
    if(drop_shadow)
    {
        ov_theme_fg(OV_FG_DIM);
        ov_theme_bg(OV_BG_TERMINAL);
        /* bottom shadow */
        ov_buf_pos(row + height, col + 1);
        ov_buf_hline_utf8("▒", width);
        /* right shadow */
        for(int r = row + 1; r < row + height; r++)
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
static inline void ov_draw_panel_tabs(
    int        row,
    int        col,
    int        height,
    int        width,
    const char **tabs,
    int        num_tabs,
    int        active_tab,
    ov_rgb_t   tcolor,
    int        is_focused)
{
    ov_theme_fg(is_focused ? tcolor : OV_FG_DIM);
    ov_theme_bg(OV_BG_TERMINAL);

    const char *tl = is_focused ? OV_BOX_TL_D : OV_BOX_TL;
    const char *tr = is_focused ? OV_BOX_TR_D : OV_BOX_TR;
    const char *bl = is_focused ? OV_BOX_BL_D : OV_BOX_BL;
    const char *br = is_focused ? OV_BOX_BR_D : OV_BOX_BR;
    const char *h  = is_focused ? OV_BOX_H_D  : OV_BOX_H;
    const char *v  = is_focused ? OV_BOX_V_D  : OV_BOX_V;

    /* top edge */
    ov_buf_pos(row, col);
    ov_buf_printf("%s", tl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", tr);

    /* title overlay: rendering tabs */
    int current_col = col + 2;
    for(int i = 0; i < num_tabs; i++)
    {
        ov_buf_pos(row, current_col);
        ov_buf_bold();
        if(i == active_tab)
        {
            if(is_focused)
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

    if(current_col + 15 < col + width)
    {
        ov_buf_pos(row, current_col + 1);
        ov_theme_fg(OV_FG_MUTED);
        ov_theme_bg(OV_BG_TERMINAL);
        ov_buf_printf("(Click tab)");
    }

    /* sides */
    for(int r = row + 1; r < row + height - 1; r++)
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

#endif /* OVERVIEW_THEME_H */
