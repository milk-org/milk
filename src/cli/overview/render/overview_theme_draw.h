// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_theme_draw.h
 * @brief Drawing primitives, gradients, and sparkline helpers for milk-CTRL themes.
 */

#ifndef OVERVIEW_THEME_DRAW_H
#define OVERVIEW_THEME_DRAW_H

#include <stdarg.h>
#include <stdio.h>

#include "overview_ansi.h"
#include "overview_data.h"

/* Helper: emit themed fg/bg */

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

static inline ov_rgb_t ov_theme_highlight_bg(ov_rgb_t base_bg)
{
    ov_rgb_t highlight;
    int      sum = base_bg.r + base_bg.g + base_bg.b;
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

#endif /* OVERVIEW_THEME_DRAW_H */
