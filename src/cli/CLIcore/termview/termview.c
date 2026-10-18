// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file termview.c
 * @brief Termview module (TrueColor Standalone Rewrite)
 */

#include "termview.h"
#include "termview_ansi.h"
#include "termview_internal.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkcommon/pixel_dispatch.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/select.h>
#include <time.h>
#include <stdbool.h>

/* -----------------------------------------------------------------------
 * Typedefs & Constants
 * ----------------------------------------------------------------------- */
typedef struct
{
    uint8_t r, g, b;
} rgb_t;

#define TV_KEY_UP 0x1001
#define TV_KEY_DOWN 0x1002
#define TV_KEY_LEFT 0x1003
#define TV_KEY_RIGHT 0x1004
#define TV_KEY_MOUSE 0x1005
#define TV_KEY_EOF 0x10FF

#define TV_CTRL(c) ((c) & 0x1F)
#define TV_ESC_TIMEOUT_US 50000L

#define TV_BAR_WIDTH 12 /* colorbar + labels */
#define TV_INFO_ROWS 4  /* rows below the image */

typedef struct
{
    int button;   // 0: Left, 1: Middle, 2: Right, 64: ScrollUp, 65: ScrollDown
    int x;        // 1-indexed
    int y;        // 1-indexed
    int is_press; // 1 for press/drag, 0 for release
    int is_drag;  // 1 if drag
} tv_mouse_event_t;

static tv_mouse_event_t last_mouse_event = { 0 };

/* -----------------------------------------------------------------------
 * Global State
 * ----------------------------------------------------------------------- */
static int loop = 1;
static int wrow = 0, wcol = 0;
static int force_redraw = 1;
static int exit_status  = 0;

static termview_options_t current_options;
static double             view_zoom     = 1.0;
static double             view_center_x = 0.5;
static double             view_center_y = 0.5;

static double current_min_val = 0.0;
static double current_max_val = 1.0;

/* -----------------------------------------------------------------------
 * Grid-based Rendering State
 * ----------------------------------------------------------------------- */
typedef struct
{
    uint8_t bg_r, bg_g, bg_b;
    uint8_t fg_r, fg_g, fg_b;
    char    ch[4]; // utf-8 char (half-block is 3 bytes, full-block 3 bytes)
} tv_cell_t;

static rgb_t colormap_lut[TV_LUT_N + 1]; /* [TV_LUT_N]: NaN/Inf pixels */
static int   lut_colormap   = -1;
static int   lut_scale_mode = -1;

static rgb_t get_colormap_color(termview_colormap_t cmap, double v)
{
    if (v < 0.0)
    {
        v = 0.0;
    }
    if (v > 1.0)
    {
        v = 1.0;
    }

    rgb_t c = { 0, 0, 0 };
    if (cmap == COLORMAP_GREYSCALE)
    {
        c.r = c.g = c.b = (uint8_t) (v * 255.0f);
    }
    else
    {
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (cmap == COLORMAP_HEAT)
        {
            if (v < 0.25f)
            {
                b = v * 4.0f;
            }
            else if (v < 0.5f)
            {
                b = 1.0f;
                g = (v - 0.25f) * 4.0f;
            }
            else if (v < 0.75f)
            {
                b = 1.0f - (v - 0.5f) * 4.0f;
                g = 1.0f;
                r = (v - 0.5f) * 4.0f;
            }
            else
            {
                g = 1.0f - (v - 0.75f) * 4.0f;
                r = 1.0f;
            }
        }
        else if (cmap == COLORMAP_COLD)
        {
            if (v < 0.33f)
            {
                b = v * 3.0f;
            }
            else if (v < 0.66f)
            {
                b = 1.0f;
                g = (v - 0.33f) * 3.0f;
            }
            else
            {
                b = 1.0f;
                g = 1.0f;
                r = (v - 0.66f) * 3.0f;
            }
        }
        else if (cmap == COLORMAP_JET)
        {
            if (v < 0.25f)
            {
                b = 1.0f;
                g = v * 4.0f;
            }
            else if (v < 0.5f)
            {
                b = 1.0f - (v - 0.25f) * 4.0f;
                g = 1.0f;
            }
            else if (v < 0.75f)
            {
                g = 1.0f;
                r = (v - 0.5f) * 4.0f;
            }
            else
            {
                g = 1.0f - (v - 0.75f) * 4.0f;
                r = 1.0f;
            }
        }
        else if (cmap == COLORMAP_INFERNO)
        {
            if (v < 0.33f)
            {
                r = v * 1.5f;
                b = v * 2.0f;
            }
            else if (v < 0.66f)
            {
                r = 0.5f + (v - 0.33f) * 1.5f;
                g = (v - 0.33f) * 1.5f;
                b = 0.66f - (v - 0.33f) * 2.0f;
            }
            else
            {
                r = 1.0f;
                g = 0.5f + (v - 0.66f) * 1.5f;
            }
        }
        else if (cmap == COLORMAP_VIRIDIS)
        {
            if (v < 0.33f)
            {
                r = 0.2f;
                g = v * 1.5f;
                b = 0.4f + v * 1.2f;
            }
            else if (v < 0.66f)
            {
                r = 0.2f;
                g = 0.5f + (v - 0.33f) * 1.0f;
                b = 0.8f - (v - 0.33f) * 1.5f;
            }
            else
            {
                r = 0.2f + (v - 0.66f) * 2.4f;
                g = 0.83f + (v - 0.66f) * 0.5f;
                b = 0.3f - (v - 0.66f) * 0.9f;
            }
        }
        else if (cmap == COLORMAP_MAGMA)
        {
            if (v < 0.25f)
            {
                r = v * 2.0f;
                g = 0.0f;
                b = v * 2.0f;
            }
            else if (v < 0.5f)
            {
                r = 0.5f + (v - 0.25f) * 2.0f;
                g = 0.0f;
                b = 0.5f - (v - 0.25f) * 2.0f;
            }
            else if (v < 0.75f)
            {
                r = 1.0f;
                g = (v - 0.5f) * 2.0f;
                b = 0.0f;
            }
            else
            {
                r = 1.0f;
                g = 0.5f + (v - 0.75f) * 2.0f;
                b = (v - 0.75f) * 4.0f;
            }
        }
        else if (cmap == COLORMAP_PLASMA)
        {
            if (v < 0.33f)
            {
                r = v * 1.5f;
                g = 0.0f;
                b = 0.5f + v * 1.5f;
            }
            else if (v < 0.66f)
            {
                r = 0.5f + (v - 0.33f) * 1.5f;
                g = (v - 0.33f) * 1.5f;
                b = 1.0f - (v - 0.33f) * 1.5f;
            }
            else
            {
                r = 1.0f;
                g = 0.5f + (v - 0.66f) * 1.5f;
                b = 0.5f - (v - 0.66f) * 1.5f;
            }
        }
        else if (cmap == COLORMAP_BONE)
        {
            if (v < 0.33f)
            {
                r = v * 1.5f;
                g = v * 1.5f;
                b = v * 2.0f;
            }
            else if (v < 0.66f)
            {
                r = 0.5f + (v - 0.33f) * 1.0f;
                g = 0.5f + (v - 0.33f) * 1.5f;
                b = 0.66f + (v - 0.33f) * 1.0f;
            }
            else
            {
                r = 0.83f + (v - 0.66f) * 0.5f;
                g = 1.0f;
                b = 1.0f;
            }
        }
        else if (cmap == COLORMAP_RAINBOW)
        {
            float h = (1.0f - v) * 5.0f;
            int   i = (int) h;
            float f = h - i;
            float q = 1.0f - f;
            switch (i)
            {
            case 0:
                r = 1.0f;
                g = f;
                b = 0.0f;
                break;
            case 1:
                r = q;
                g = 1.0f;
                b = 0.0f;
                break;
            case 2:
                r = 0.0f;
                g = 1.0f;
                b = f;
                break;
            case 3:
                r = 0.0f;
                g = q;
                b = 1.0f;
                break;
            case 4:
                r = f;
                g = 0.0f;
                b = 1.0f;
                break;
            case 5:
                r = 1.0f;
                g = 0.0f;
                b = 1.0f;
                break;
            default:
                r = 1.0f;
                g = 0.0f;
                b = 1.0f;
                break;
            }
        }
        else if (cmap == COLORMAP_TURBO)
        {
            float k1 =
                0.1357f +
                v * (4.5974f - v * (42.3277f - v * (130.5887f - v * (150.5666f - v * 58.1375f))));
            float k2 = 0.0914f +
                       v * (2.1941f + v * (4.8429f - v * (14.1850f - v * (4.2773f + v * 2.8251f))));
            float k3 =
                0.1066f +
                v * (12.6419f - v * (60.5820f - v * (110.3627f - v * (89.9031f - v * 27.3482f))));
            r = k1;
            g = k2;
            b = k3;
        }
        else if (cmap == COLORMAP_OCEAN)
        {
            r = v * 0.5f;
            g = v * 0.8f;
            b = 0.2f + v * 0.8f;
        }
        else if (cmap == COLORMAP_COPPER)
        {
            r = v * 1.25f;
            g = v * 0.7812f;
            b = v * 0.4975f;
        }
        else if (cmap == COLORMAP_SPRING)
        {
            r = 1.0f;
            g = v;
            b = 1.0f - v;
        }
        else if (cmap == COLORMAP_SUMMER)
        {
            r = v;
            g = 0.5f + v * 0.5f;
            b = 0.4f;
        }
        else if (cmap == COLORMAP_AUTUMN)
        {
            r = 1.0f;
            g = v;
            b = 0.0f;
        }
        else if (cmap == COLORMAP_WINTER)
        {
            r = 0.0f;
            g = v;
            b = 1.0f - v * 0.5f;
        }
        if (r > 1.0f)
        {
            r = 1.0f;
        }
        if (r < 0.0f)
        {
            r = 0.0f;
        }
        if (g > 1.0f)
        {
            g = 1.0f;
        }
        if (g < 0.0f)
        {
            g = 0.0f;
        }
        if (b > 1.0f)
        {
            b = 1.0f;
        }
        if (b < 0.0f)
        {
            b = 0.0f;
        }
        c.r = (uint8_t) (r * 255.0f);
        c.g = (uint8_t) (g * 255.0f);
        c.b = (uint8_t) (b * 255.0f);
    }
    return c;
}

static void build_colormap_lut(termview_colormap_t cmap, termview_scale_t scale)
{
    for (int i = 0; i < TV_LUT_N; i++)
    {
        double v = i / (double) (TV_LUT_N - 1);
        if (scale == SCALE_SQRT)
        {
            v = sqrt(v);
        }
        else if (scale == SCALE_LOG)
        {
            v = log(v * 9.0 + 1.0) / log(10.0);
        }
        else if (scale == SCALE_LOG_STRONG)
        {
            v = log(v * 99.0 + 1.0) / log(100.0);
        }
        else if (scale == SCALE_LOG_EXTREME)
        {
            v = log(v * 999.0 + 1.0) / log(1000.0);
        }
        else if (scale == SCALE_ASINH)
        {
            v = asinh(v * 10.0) / asinh(10.0);
        }
        else if (scale == SCALE_SQUARED)
        {
            v = v * v;
        }
        else if (scale == SCALE_CUBED)
        {
            v = v * v * v;
        }

        colormap_lut[i] = get_colormap_color(cmap, v);
    }
    colormap_lut[TV_LUT_N] = (rgb_t) { 255, 0, 255 };
    lut_colormap           = cmap;
    lut_scale_mode         = scale;
}


/* -----------------------------------------------------------------------
 * Keyboard and Mouse Input
 * ----------------------------------------------------------------------- */

/** tv_read_byte() - One byte within @timeout_us: the byte, -1 on timeout, -2 on EOF or error. */
static int tv_read_byte(long timeout_us)
{
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    struct timeval tv;
    tv.tv_sec  = timeout_us / 1000000;
    tv.tv_usec = timeout_us % 1000000;

    int ready = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
    if (ready == 0)
    {
        return -1;
    }
    if (ready < 0)
    {
        return (errno == EINTR) ? -1 : -2;
    }

    unsigned char c;
    ssize_t       n = read(STDIN_FILENO, &c, 1);
    if (n == 1)
    {
        return c;
    }
    if (n < 0 && (errno == EINTR || errno == EAGAIN))
    {
        return -1;
    }
    return -2; /* readable without data: hangup */
}

/** tv_parse_sgr_mouse() - Decode "<b;x;y" + final M/m into last_mouse_event; 0 if valid. */
static int tv_parse_sgr_mouse(const char *par, int fin)
{
    int v[3] = { 0, 0, 0 };
    int k    = 0;
    for (const char *p = par + 1; *p != '\0'; p++)
    {
        if (*p == ';')
        {
            if (++k > 2)
            {
                return -1;
            }
        }
        else if (*p >= '0' && *p <= '9')
        {
            if (v[k] < 100000)
            {
                v[k] = v[k] * 10 + (*p - '0');
            }
        }
        else
        {
            return -1;
        }
    }
    if (k != 2)
    {
        return -1;
    }

    int b                    = v[0];
    last_mouse_event.is_drag = ((b & 32) != 0);
    if (b & 64)
    {
        if (fin == 'm')
        {
            return -1; /* wheel "release" carries no action */
        }
        last_mouse_event.button  = 64 + (b & 3);
        last_mouse_event.is_drag = 0;
    }
    else
    {
        last_mouse_event.button = b & 3;
    }
    last_mouse_event.x        = v[1];
    last_mouse_event.y        = v[2];
    last_mouse_event.is_press = (fin == 'M');
    return 0;
}

/**
 * tv_read_key() - Next key, TV_KEY_* code, -1 if none, or TV_KEY_EOF.
 * Escape sequences are collected whole (waiting up to TV_ESC_TIMEOUT_US per byte); incomplete or
 * unknown sequences are discarded so their bytes never act as commands.
 */
static int tv_read_key(long timeout_us)
{
    int c = tv_read_byte(timeout_us);
    if (c == -2)
    {
        return TV_KEY_EOF;
    }
    if (c != 27)
    {
        return c;
    }

    int c1 = tv_read_byte(TV_ESC_TIMEOUT_US);
    if (c1 == -2)
    {
        return TV_KEY_EOF;
    }
    if (c1 == -1)
    {
        return 27; /* lone ESC */
    }
    if (c1 == 'O')
    {
        int c2 = tv_read_byte(TV_ESC_TIMEOUT_US);
        switch (c2)
        {
        case -2:
            return TV_KEY_EOF;
        case 'A':
            return TV_KEY_UP;
        case 'B':
            return TV_KEY_DOWN;
        case 'C':
            return TV_KEY_RIGHT;
        case 'D':
            return TV_KEY_LEFT;
        default:
            return -1;
        }
    }
    if (c1 != '[')
    {
        return -1; /* Alt+key: unbound */
    }

    char par[32];
    int  np       = 0;
    int  overflow = 0;
    int  fin      = -1;
    while (fin < 0)
    {
        int b = tv_read_byte(TV_ESC_TIMEOUT_US);
        if (b == -2)
        {
            return TV_KEY_EOF;
        }
        if (b < 0x20 || b > 0x7E)
        {
            return -1; /* timeout or stray byte: drop the partial sequence */
        }
        if (b >= 0x40)
        {
            fin = b;
        }
        else if (np < (int) sizeof(par) - 1)
        {
            par[np++] = (char) b;
        }
        else
        {
            overflow = 1;
        }
    }
    par[np] = '\0';
    if (overflow)
    {
        return -1;
    }

    if (fin == 'M' && np == 0)
    {
        /* X10 mouse report: three raw bytes follow, consumed and ignored */
        for (int i = 0; i < 3; i++)
        {
            if (tv_read_byte(TV_ESC_TIMEOUT_US) == -2)
            {
                return TV_KEY_EOF;
            }
        }
        return -1;
    }
    if ((fin == 'M' || fin == 'm') && par[0] == '<')
    {
        return (tv_parse_sgr_mouse(par, fin) == 0) ? TV_KEY_MOUSE : -1;
    }
    switch (fin)
    {
    case 'A':
        return TV_KEY_UP;
    case 'B':
        return TV_KEY_DOWN;
    case 'C':
        return TV_KEY_RIGHT;
    case 'D':
        return TV_KEY_LEFT;
    default:
        return -1;
    }
}

/* -----------------------------------------------------------------------
 * Data Retrieval & Colormaps
 * ----------------------------------------------------------------------- */
/** get_pixel_value() - Element @idx of the data array as double; caller checks bounds. */
static inline double get_pixel_value(const IMAGE *img, uint8_t datatype, size_t idx)
{
#define GET_PIXEL_CASE(DT, ACC, CTYPE) \
    case DT:                           \
        return (double) img->array.ACC[idx];

    switch (datatype)
    {
        FOREACH_REAL_DATATYPE(GET_PIXEL_CASE)
    default:
        return 0.0;
    }
#undef GET_PIXEL_CASE
}


static inline int compare_doubles(const void *a, const void *b)
{
    double arg1 = *(const double *) a;
    double arg2 = *(const double *) b;
    if (arg1 < arg2)
    {
        return -1;
    }
    if (arg1 > arg2)
    {
        return 1;
    }
    return 0;
}

/* -----------------------------------------------------------------------
 * Main Screen Loop
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Termview Refactored Context
 * ----------------------------------------------------------------------- */
typedef struct
{
    tv_stream_t    *st;
    IMAGE          *img;
    int             target_fps;
    double          current_cpu_usage;
    double          current_stream_fps;
    uint64_t        last_cnt0_for_fps;
    uint64_t        last_cnt0;
    struct timespec last_time_real;
    struct timespec last_time_cpu;
    struct timespec last_render_real;

    int             popup_type;
    struct timespec popup_expiry_time;

    int         mouse_is_dragging;
    int         roi_is_dragging;
    int         last_mouse_x;
    int         last_mouse_y;
    int         roi_start_x;
    int         roi_start_y;
    int         roi_end_x;
    int         roi_end_y;
    int         input_mode;
    char        input_buf[64];
    int         input_len;
    const char *input_err; /**< validation message shown in the prompt line */

    double    *display_buffer;
    int        buffer_size;
    tv_cell_t *screen;
    tv_cell_t *prev_screen;
    int        screen_size;
    char      *frame_buffer;
    size_t     frame_buffer_size;
    double    *sort_buffer; /**< percentile scratch, grown on demand */
    int        sort_size;

    long timeout_us;
} tv_context_t;


static void termview_update_fps_stats(tv_context_t *ctx)
{
    clock_gettime(CLOCK_MONOTONIC, &ctx->last_render_real);
    double real_diff = (ctx->last_render_real.tv_sec - ctx->last_time_real.tv_sec) +
                       (ctx->last_render_real.tv_nsec - ctx->last_time_real.tv_nsec) / 1e9;
    if (real_diff >= 1.0)
    {
        struct timespec now_cpu;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now_cpu);
        double cpu_diff        = (now_cpu.tv_sec - ctx->last_time_cpu.tv_sec) +
                                 (now_cpu.tv_nsec - ctx->last_time_cpu.tv_nsec) / 1e9;
        ctx->current_cpu_usage = (cpu_diff / real_diff) * 100.0;

        uint64_t current_cnt0 = ctx->img->md[0].cnt0;
        /* a counter that went backwards means the stream was reset: re-baseline */
        ctx->current_stream_fps = (current_cnt0 >= ctx->last_cnt0_for_fps)
                                      ? (double) (current_cnt0 - ctx->last_cnt0_for_fps) / real_diff
                                      : 0.0;
        ctx->last_cnt0_for_fps  = current_cnt0;

        ctx->last_time_real = ctx->last_render_real;
        ctx->last_time_cpu  = now_cpu;
    }
}

/** tv_layout_ok() - True when the image area is drawable; also guards every geometry divisor. */
static int tv_layout_ok(void)
{
    return (wrow - TV_INFO_ROWS) >= 2 && (wcol - TV_BAR_WIDTH - 1) >= 4;
}

static void termview_handle_mouse_event(tv_context_t *ctx)
{
    force_redraw = 1;
    if (!tv_layout_ok())
    {
        ctx->mouse_is_dragging = 0;
        ctx->roi_is_dragging   = 0;
        return;
    }

    int    mx            = last_mouse_event.x;
    int    my            = last_mouse_event.y;
    double sign_x        = current_options.flip_h ? -1.0 : 1.0;
    double sign_y        = current_options.flip_v ? -1.0 : 1.0;
    double xsize         = ctx->st->xsize;
    double ysize         = ctx->st->ysize;
    int    disp_cols     = wcol - TV_BAR_WIDTH - 1;
    int    disp_img_rows = (wrow - TV_INFO_ROWS) * 2;
    double step          = fmax(xsize / view_zoom / disp_cols, ysize / view_zoom / disp_img_rows);

    if (last_mouse_event.button == 64)
    {
        // Scroll Up -> Zoom in
        view_zoom *= 1.2;
    }
    else if (last_mouse_event.button == 65)
    {
        // Scroll Down -> Zoom out
        view_zoom /= 1.2;
    }
    else if (last_mouse_event.button == 0) // Left Button
    {
        if (last_mouse_event.is_press && !last_mouse_event.is_drag)
        {
            ctx->mouse_is_dragging = 1;
            ctx->last_mouse_x      = mx;
            ctx->last_mouse_y      = my;
        }
        else if (last_mouse_event.is_press && last_mouse_event.is_drag && ctx->mouse_is_dragging)
        {
            /* grab: the image follows the pointer on both axes, whatever the flips */
            int dx = mx - ctx->last_mouse_x;
            int dy = my - ctx->last_mouse_y;
            view_center_x -= sign_x * dx * step / xsize;
            view_center_y -= sign_y * dy * 2.0 * step / ysize;
            ctx->last_mouse_x = mx;
            ctx->last_mouse_y = my;
        }
        else if (!last_mouse_event.is_press)
        {
            ctx->mouse_is_dragging = 0;
        }
    }
    else if (last_mouse_event.button == 2) // Right Button
    {
        if (last_mouse_event.is_press && !last_mouse_event.is_drag)
        {
            ctx->roi_is_dragging = 1;
            ctx->roi_start_x     = mx;
            ctx->roi_start_y     = my;
            ctx->roi_end_x       = mx;
            ctx->roi_end_y       = my;
        }
        else if (last_mouse_event.is_press && last_mouse_event.is_drag && ctx->roi_is_dragging)
        {
            ctx->roi_end_x = mx;
            ctx->roi_end_y = my;
        }
        else if (!last_mouse_event.is_press && ctx->roi_is_dragging)
        {
            ctx->roi_end_x       = mx;
            ctx->roi_end_y       = my;
            ctx->roi_is_dragging = 0;

            int min_x = (ctx->roi_start_x < ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
            int max_x = (ctx->roi_start_x > ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
            int min_y = (ctx->roi_start_y < ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
            int max_y = (ctx->roi_start_y > ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;

            if (max_x - min_x >= 2 && max_y - min_y >= 2)
            {
                double center_disp_x = disp_cols / 2.0;
                double center_disp_y = disp_img_rows / 2.0;
                double roi_center_mx = (min_x + max_x) / 2.0;
                double roi_center_my = (min_y + max_y) / 2.0;
                double roi_row       = (roi_center_my - 1.0) * 2.0 + 0.5;

                double new_cx =
                    view_center_x * xsize + sign_x * (roi_center_mx - 1.0 - center_disp_x) * step;
                double new_cy = view_center_y * ysize + sign_y * (roi_row - center_disp_y) * step;

                double roi_w_char = max_x - min_x + 1;
                double roi_h_char = max_y - min_y + 1;
                double new_step_x = (roi_w_char * step) / disp_cols;
                double new_step_y = (roi_h_char * 2 * step) / disp_img_rows;
                double new_step   = fmax(new_step_x, new_step_y);

                view_center_x = new_cx / xsize;
                view_center_y = new_cy / ysize;
                view_zoom *= (step / new_step);
            }
        }
    }
}

static void termview_handle_keyboard_event(tv_context_t *ctx, int ch)
{
    switch (ch)
    {
    case 10: // ENTER key (LF)
    case 13: // ENTER key (CR)
    case 27: // ESC key
        if (ctx->popup_type > 0)
        {
            ctx->popup_type = 0;
            force_redraw    = 1;
        }
        break;
    case 'q':
    case TV_CTRL('c'):
    case TV_CTRL('\\'):
        loop = 0;
        break;
    case TV_KEY_EOF:
        loop        = 0;
        exit_status = 1;
        break;
    case TV_CTRL('z'):
        tv_term_suspend();
        break;
    case TV_CTRL('l'):
        if (ctx->prev_screen)
        {
            memset(ctx->prev_screen, 0xFF, (size_t) ctx->screen_size * sizeof(tv_cell_t));
        }
        force_redraw = 1;
        break;
    case 'c':
        force_redraw             = 1;
        current_options.colormap = (current_options.colormap + 1) % COLORMAP_NB;
        ctx->popup_type          = 1;
        clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
        ctx->popup_expiry_time.tv_sec += 2;
        break;
    case 's':
        force_redraw          = 1;
        current_options.scale = (current_options.scale + 1) % SCALE_NB;
        ctx->popup_type       = 2;
        clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
        ctx->popup_expiry_time.tv_sec += 2;
        break;
    case 'r':
        force_redraw          = 1;
        current_options.range = (current_options.range + 1) % RANGE_NB;
        ctx->popup_type       = 3;
        clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
        ctx->popup_expiry_time.tv_sec += 2;
        break;
    case 'l':
        force_redraw                 = 1;
        current_options.range_locked = !current_options.range_locked;
        if (current_options.range_locked)
        {
            current_options.manual_min = current_min_val;
            current_options.manual_max = current_max_val;
        }
        break;
    case 'm':
    case 'M':
        force_redraw      = 1;
        ctx->input_mode   = (ch == 'm') ? 1 : 2;
        ctx->input_len    = 0;
        ctx->input_buf[0] = '\0';
        ctx->input_err    = NULL;
        break;
    case 'h':
        force_redraw           = 1;
        current_options.flip_h = !current_options.flip_h;
        break;
    case 'v':
        force_redraw           = 1;
        current_options.flip_v = !current_options.flip_v;
        break;
    case '+':
    case '=':
        force_redraw = 1;
        view_zoom *= 1.2;
        break;
    case '-':
    case '_':
        force_redraw = 1;
        view_zoom /= 1.2;
        break;
    case '0':
        force_redraw  = 1;
        view_zoom     = 1.0;
        view_center_x = 0.5;
        view_center_y = 0.5;
        break;
    /* arrows move the viewport in screen space: Left shows what lies left of the screen */
    case TV_KEY_LEFT:
        force_redraw = 1;
        view_center_x -= (current_options.flip_h ? -0.1 : 0.1) / view_zoom;
        break;
    case TV_KEY_RIGHT:
        force_redraw = 1;
        view_center_x += (current_options.flip_h ? -0.1 : 0.1) / view_zoom;
        break;
    case TV_KEY_UP:
        force_redraw = 1;
        if (ctx->popup_type == 1)
        {
            current_options.colormap = (current_options.colormap + COLORMAP_NB - 1) % COLORMAP_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 2)
        {
            current_options.scale = (current_options.scale + SCALE_NB - 1) % SCALE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 3)
        {
            current_options.range = (current_options.range + RANGE_NB - 1) % RANGE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else
        {
            view_center_y -= (current_options.flip_v ? -0.1 : 0.1) / view_zoom;
        }
        break;
    case TV_KEY_DOWN:
        force_redraw = 1;
        if (ctx->popup_type == 1)
        {
            current_options.colormap = (current_options.colormap + 1) % COLORMAP_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 2)
        {
            current_options.scale = (current_options.scale + 1) % SCALE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 3)
        {
            current_options.range = (current_options.range + 1) % RANGE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else
        {
            view_center_y += (current_options.flip_v ? -0.1 : 0.1) / view_zoom;
        }
        break;
    case '<':
    case ',':
        force_redraw = 1;
        ctx->target_fps -= 1;
        if (ctx->target_fps < 1)
        {
            ctx->target_fps = 1;
        }
        break;
    case '>':
    case '.':
        force_redraw = 1;
        ctx->target_fps += 1;
        if (ctx->target_fps > 120)
        {
            ctx->target_fps = 120;
        }
        break;
    case TV_KEY_MOUSE:
        termview_handle_mouse_event(ctx);
        break;
    }
}

static void termview_render_image(tv_context_t *ctx,
                                  int           disp_char_rows,
                                  int           disp_cols,
                                  double        center_img_x,
                                  double        center_img_y,
                                  double        sign_x,
                                  double        sign_y,
                                  double        step,
                                  uint32_t      xsize,
                                  uint32_t      ysize,
                                  double        min_val,
                                  double        lut_scale)
{
#define IS_IN_BOUNDS(jj, yy)                                                          \
    (((int) (center_img_x + sign_x * ((jj) - center_disp_x) * step) >= 0) &&          \
     ((int) (center_img_x + sign_x * ((jj) - center_disp_x) * step) < (int) xsize) && \
     ((int) (center_img_y + sign_y * ((yy) - center_disp_y) * step) >= 0) &&          \
     ((int) (center_img_y + sign_y * ((yy) - center_disp_y) * step) < (int) ysize))

    double center_disp_x = disp_cols / 2.0;
    double center_disp_y = (disp_char_rows * 2) / 2.0;

    for (int i = 0; i < disp_char_rows; i++)
    {
        for (int j = 0; j < disp_cols; j++)
        {
            int  img_y_top     = (int) (center_img_y + sign_y * ((i * 2) - center_disp_y) * step);
            int  img_x_top     = (int) (center_img_x + sign_x * (j - center_disp_x) * step);
            bool in_bounds_top = (img_x_top >= 0 && img_x_top < (int) xsize && img_y_top >= 0 &&
                                  img_y_top < (int) ysize);

            int  img_y_bot = (int) (center_img_y + sign_y * ((i * 2 + 1) - center_disp_y) * step);
            int  img_x_bot = (int) (center_img_x + sign_x * (j - center_disp_x) * step);
            bool in_bounds_bot = (img_x_bot >= 0 && img_x_bot < (int) xsize && img_y_bot >= 0 &&
                                  img_y_bot < (int) ysize);

            tv_cell_t *cell = &ctx->screen[i * wcol + j];

            if (!in_bounds_top && !in_bounds_bot)
            {
                bool near_top = IS_IN_BOUNDS(j - 1, i * 2) || IS_IN_BOUNDS(j + 1, i * 2) ||
                                IS_IN_BOUNDS(j, i * 2 - 1) || IS_IN_BOUNDS(j, i * 2 + 1);
                bool near_bot = IS_IN_BOUNDS(j - 1, i * 2 + 1) || IS_IN_BOUNDS(j + 1, i * 2 + 1) ||
                                IS_IN_BOUNDS(j, i * 2) || IS_IN_BOUNDS(j, i * 2 + 2);
                if (!near_top && !near_bot)
                {
                    continue; // left as space
                }
            }

            double val_top = in_bounds_top ? ctx->display_buffer[(i * 2) * disp_cols + j] : 0.0;
            double val_bot = in_bounds_bot ? ctx->display_buffer[(i * 2 + 1) * disp_cols + j] : 0.0;

            rgb_t rgb_top = colormap_lut[tv_lut_idx(val_top, min_val, lut_scale)];
            rgb_t rgb_bot = colormap_lut[tv_lut_idx(val_bot, min_val, lut_scale)];

            if (!in_bounds_top)
            {
                if (IS_IN_BOUNDS(j - 1, i * 2) || IS_IN_BOUNDS(j + 1, i * 2) ||
                    IS_IN_BOUNDS(j, i * 2 - 1) || IS_IN_BOUNDS(j, i * 2 + 1))
                {
                    rgb_top = (rgb_t) { 64, 64, 64 }; // Dark gray border
                }
                else
                {
                    rgb_top = (rgb_t) { 0, 0, 0 };
                }
            }

            if (!in_bounds_bot)
            {
                if (IS_IN_BOUNDS(j - 1, i * 2 + 1) || IS_IN_BOUNDS(j + 1, i * 2 + 1) ||
                    IS_IN_BOUNDS(j, i * 2) || IS_IN_BOUNDS(j, i * 2 + 2))
                {
                    rgb_bot = (rgb_t) { 64, 64, 64 }; // Dark gray border
                }
                else
                {
                    rgb_bot = (rgb_t) { 0, 0, 0 };
                }
            }

            if (ctx->roi_is_dragging)
            {
                int rx_min =
                    (ctx->roi_start_x < ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
                int rx_max =
                    (ctx->roi_start_x > ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
                int ry_min =
                    (ctx->roi_start_y < ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
                int ry_max =
                    (ctx->roi_start_y > ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;

                int term_x = j + 1;
                int term_y = i + 1;

                if (term_x >= rx_min && term_x <= rx_max && term_y >= ry_min && term_y <= ry_max)
                {
                    if (term_x == rx_min || term_x == rx_max || term_y == ry_min ||
                        term_y == ry_max)
                    {
                        rgb_top = (rgb_t) { 255, 255, 0 }; // Yellow border
                        rgb_bot = (rgb_t) { 255, 255, 0 };
                    }
                    else
                    {
                        // Dim the inside of the ROI
                        rgb_top.r /= 2;
                        rgb_top.g /= 2;
                        rgb_top.b /= 2;
                        rgb_bot.r /= 2;
                        rgb_bot.g /= 2;
                        rgb_bot.b /= 2;
                    }
                }
            }

            cell->bg_r  = rgb_top.r;
            cell->bg_g  = rgb_top.g;
            cell->bg_b  = rgb_top.b;
            cell->fg_r  = rgb_bot.r;
            cell->fg_g  = rgb_bot.g;
            cell->fg_b  = rgb_bot.b;
            cell->ch[0] = (char) 0xE2;
            cell->ch[1] = (char) 0x96;
            cell->ch[2] = (char) 0x84;
            cell->ch[3] = '\0';
        }
    }
#undef IS_IN_BOUNDS
}

static void termview_render_colorbar(tv_context_t *ctx,
                                     int           disp_char_rows,
                                     int           bar_col_start,
                                     double        min_val,
                                     double        max_val)
{
    for (int i = 0; i < disp_char_rows; i++)
    {
        double norm_val = (disp_char_rows > 1) ? 1.0 - (double) i / (disp_char_rows - 1) : 0.5;
        rgb_t  rgb      = colormap_lut[tv_lut_idx(norm_val, 0.0, TV_LUT_N - 1)];

        tv_cell_t *c1 = &ctx->screen[i * wcol + bar_col_start - 1];
        tv_cell_t *c2 = &ctx->screen[i * wcol + bar_col_start];
        c1->fg_r      = rgb.r;
        c1->fg_g      = rgb.g;
        c1->fg_b      = rgb.b;
        c1->bg_r      = 0;
        c1->bg_g      = 0;
        c1->bg_b      = 0;
        c1->ch[0]     = (char) 0xE2;
        c1->ch[1]     = (char) 0x96;
        c1->ch[2]     = (char) 0x88;
        c1->ch[3]     = '\0';

        c2->fg_r  = rgb.r;
        c2->fg_g  = rgb.g;
        c2->fg_b  = rgb.b;
        c2->bg_r  = 0;
        c2->bg_g  = 0;
        c2->bg_b  = 0;
        c2->ch[0] = (char) 0xE2;
        c2->ch[1] = (char) 0x96;
        c2->ch[2] = (char) 0x88;
        c2->ch[3] = '\0';

        char label[32];
        int  label_len = 0;
        if (i == 0)
        {
            label_len = snprintf(label, sizeof(label), "%.2g", max_val);
        }
        else if (i == disp_char_rows / 2)
        {
            label_len = snprintf(label, sizeof(label), "%.2g", (min_val + max_val) / 2);
        }
        else if (i == disp_char_rows - 1)
        {
            label_len = snprintf(label, sizeof(label), "%.2g", min_val);
        }

        label_len = tv_fmt_len(label_len, sizeof(label));
        for (int k = 0; k < label_len; k++)
        {
            if (bar_col_start + 2 + k < wcol)
            {
                tv_cell_t *lc = &ctx->screen[i * wcol + bar_col_start + 2 + k];
                lc->ch[0]     = label[k];
                lc->ch[1]     = '\0';
            }
        }
    }
}

static void termview_render_infobar(tv_context_t *ctx,
                                    uint32_t      xsize,
                                    uint32_t      ysize,
                                    double        step,
                                    double        min_val,
                                    double        max_val)
{
    int info_row = wrow - 3;

    const char *cmap_str  = tv_cmap_names[current_options.colormap];
    const char *scale_str = tv_scale_names[current_options.scale];
    const char *range_str = tv_range_names[current_options.range];

    char info[256];
    int  len;

    int name_len = (int) strlen(ctx->st->name);
    len = snprintf(info, sizeof(info),
                   "Image: %s [%u x %u] Type: %d  |  CPU: %5.1f%%  UI: %3d FPS  Stream: %5.1f Hz",
                   ctx->st->name, xsize, ysize, ctx->st->datatype, ctx->current_cpu_usage,
                   ctx->target_fps, ctx->current_stream_fps);
    len = tv_fmt_len(len, sizeof(info));
    for (int k = 0; k < len && k < wcol; k++)
    {
        tv_cell_t *c = &ctx->screen[info_row * wcol + k];
        c->ch[0]     = info[k];
        c->ch[1]     = '\0';
        if (k >= 7 && k < 7 + name_len)
        {
            c->fg_r = 0;
            c->fg_g = 255;
            c->fg_b = 255; // Cyan
        }
    }
    len = snprintf(info, sizeof(info),
                   "Val: [%.4g : %.4g] Zoom: %.2fx (%.2f px/char H, %.2f px/char V)", min_val,
                   max_val, view_zoom, step, step * 2.0);
    len = tv_fmt_len(len, sizeof(info));
    for (int k = 0; k < len && k < wcol; k++)
    {
        ctx->screen[(info_row + 1) * wcol + k].ch[0] = info[k];
        ctx->screen[(info_row + 1) * wcol + k].ch[1] = '\0';
    }

    if (ctx->input_mode > 0)
    {
        len = snprintf(info, sizeof(info), "Enter Manual %s: %s_%s%s",
                       ctx->input_mode == 1 ? "Min" : "Max", ctx->input_buf,
                       ctx->input_err ? "   " : "", ctx->input_err ? ctx->input_err : "");
        len = tv_fmt_len(len, sizeof(info));
        for (int k = 0; k < len && k < wcol; k++)
        {
            tv_cell_t *c = &ctx->screen[(info_row + 2) * wcol + k];
            c->ch[0]     = info[k];
            c->ch[1]     = '\0';
            c->fg_r      = 255;
            c->fg_g      = 255;
            c->fg_b      = 0; // Yellow text
        }
        for (int k = len; k < wcol; k++)
        {
            ctx->screen[(info_row + 2) * wcol + k].ch[0] = ' ';
            ctx->screen[(info_row + 2) * wcol + k].ch[1] = '\0';
        }
    }
    else
    {
        char info1[128];
        int  len1 = snprintf(info1, sizeof(info1), "[C]map: %s  [S]cale: %s  [R]ange: %s [",
                             cmap_str, scale_str, range_str);
        len1      = tv_fmt_len(len1, sizeof(info1));
        const char *lock_str = current_options.range_locked ? "LOCKED" : "AUTO";
        int         lock_len = (int) strlen(lock_str);
        char        info2[256];
        int         len2 =
            snprintf(info2, sizeof(info2), "%s%s]  (< >:FPS l:Lock m/M:MinMax ", info1, lock_str);
        len2 = tv_fmt_len(len2, sizeof(info2));

        const char *h_str = current_options.flip_h ? "h:FLIPH" : "h:fliph";
        const char *v_str = current_options.flip_v ? "v:FLIPV" : "v:flipv";

        len = snprintf(info, sizeof(info), "%s%s %s q:Quit)", info2, h_str, v_str);
        len = tv_fmt_len(len, sizeof(info));

        for (int k = 0; k < len && k < wcol; k++)
        {
            tv_cell_t *c = &ctx->screen[(info_row + 2) * wcol + k];
            c->ch[0]     = info[k];
            c->ch[1]     = '\0';
            if (k >= len1 && k < len1 + lock_len)
            {
                if (current_options.range_locked)
                {
                    c->fg_r = 255;
                    c->fg_g = 50;
                    c->fg_b = 50; // Red for Locked
                }
                else
                {
                    c->fg_r = 50;
                    c->fg_g = 255;
                    c->fg_b = 50; // Green for Auto
                }
            }
            if (current_options.flip_h && k >= len2 && k < len2 + (int) strlen(h_str))
            {
                c->fg_r = 255;
                c->fg_g = 255;
                c->fg_b = 0; // Yellow
            }
            if (current_options.flip_v && k >= len2 + (int) strlen(h_str) + 1 &&
                k < len2 + (int) strlen(h_str) + 1 + (int) strlen(v_str))
            {
                c->fg_r = 255;
                c->fg_g = 255;
                c->fg_b = 0; // Yellow
            }
        }
        for (int k = len; k < wcol; k++)
        {
            ctx->screen[(info_row + 2) * wcol + k].ch[0] = ' ';
            ctx->screen[(info_row + 2) * wcol + k].ch[1] = '\0';
        }
    }
}

static void termview_render_popup(tv_context_t *ctx, int disp_char_rows, int disp_cols)
{
    if (ctx->popup_type <= 0)
    {
        return;
    }
    int                num_opts     = 0;
    const char        *title        = "";
    const char *const *names        = NULL;
    int                selected_idx = 0;
    int                box_w        = 16;
    if (ctx->popup_type == 1)
    {
        title        = "COLORMAP";
        names        = tv_cmap_names;
        num_opts     = COLORMAP_NB;
        selected_idx = current_options.colormap;
        box_w        = 40; // Wider box to fit colormap preview
    }
    else if (ctx->popup_type == 2)
    {
        title        = "SCALE";
        names        = tv_scale_names;
        num_opts     = SCALE_NB;
        selected_idx = current_options.scale;
    }
    else if (ctx->popup_type == 3)
    {
        title        = "RANGE";
        names        = tv_range_names;
        num_opts     = RANGE_NB;
        selected_idx = current_options.range;
    }

    int box_h   = num_opts + 2;
    int start_r = (disp_char_rows - box_h) / 2;
    int start_c = (disp_cols - box_w) / 2;
    if (start_r < 0)
    {
        start_r = 0;
    }
    if (start_c < 0)
    {
        start_c = 0;
    }

    for (int i = 0; i < box_h; i++)
    {
        int r = start_r + i;
        if (r >= disp_char_rows)
        {
            break;
        }
        for (int j = 0; j < box_w; j++)
        {
            int c = start_c + j;
            if (c >= disp_cols)
            {
                break;
            }

            tv_cell_t *cell = &ctx->screen[r * wcol + c];
            cell->bg_r      = 50;
            cell->bg_g      = 50;
            cell->bg_b      = 50; // dark gray bg
            cell->fg_r      = 255;
            cell->fg_g      = 255;
            cell->fg_b      = 255; // white fg
            cell->ch[0]     = ' ';
            cell->ch[1]     = '\0';

            if (i == 0)
            {
                int tlen   = (int) strlen(title);
                int tstart = (box_w - tlen) / 2;
                if (j >= tstart && j < tstart + tlen)
                {
                    cell->ch[0] = title[j - tstart];
                    cell->fg_r  = 255;
                    cell->fg_g  = 255;
                    cell->fg_b  = 0; // Yellow title
                }
            }
            else if (i <= num_opts)
            {
                int opt_idx = i - 1;
                int is_sel  = (opt_idx == selected_idx);
                if (is_sel)
                {
                    cell->bg_r = 200;
                    cell->bg_g = 200;
                    cell->bg_b = 200; // light gray selected
                    cell->fg_r = 0;
                    cell->fg_g = 0;
                    cell->fg_b = 0; // black fg
                }

                // Text portion
                if (j >= 2 && j - 2 < (int) strlen(names[opt_idx]))
                {
                    cell->ch[0] = names[opt_idx][j - 2];
                }

                // Compressed colormap rendering
                if (ctx->popup_type == 1 && j >= 16 && j < box_w - 2)
                {
                    int    bar_len = box_w - 2 - 16;
                    double v       = (double) (j - 16) / (double) (bar_len - 1);
                    rgb_t  color   = get_colormap_color((termview_colormap_t) opt_idx, v);
                    cell->bg_r     = color.r;
                    cell->bg_g     = color.g;
                    cell->bg_b     = color.b;
                    cell->ch[0]    = ' ';
                    cell->ch[1]    = '\0';
                }
            }
        }
    }
}

/** tv_invalidate() - Forget what the terminal shows so the next frame repaints every cell. */
static void tv_invalidate(tv_context_t *ctx)
{
    if (ctx->prev_screen)
    {
        memset(ctx->prev_screen, 0xFF, (size_t) ctx->screen_size * sizeof(tv_cell_t));
    }
    force_redraw = 1;
}

static void termview_flush_ansi(tv_context_t *ctx)
{
    static const char sync_on[]  = "\033[?2026h";
    static const char sync_off[] = "\033[?2026l";

    size_t fb_pos = sizeof(sync_on) - 1;
    int    cur_r = -1, cur_c = -1;
    int    last_bg_r = -1, last_bg_g = -1, last_bg_b = -1;
    int    last_fg_r = -1, last_fg_g = -1, last_fg_b = -1;
    int    changed = 0;

    memcpy(ctx->frame_buffer, sync_on, sizeof(sync_on) - 1);
    for (int r = 0; r < wrow; r++)
    {
        for (int c = 0; c < wcol; c++)
        {
            tv_cell_t *cell = &ctx->screen[r * wcol + c];
            tv_cell_t *prev = &ctx->prev_screen[r * wcol + c];

            if (memcmp(cell, prev, sizeof(tv_cell_t)) != 0)
            {
                changed = 1;
                // Update cursor if not sequential
                if (cur_r != r || cur_c != c)
                {
                    tv_move(ctx->frame_buffer, &fb_pos, r + 1, c + 1);
                }

                // Update BG
                if (cell->bg_r != last_bg_r || cell->bg_g != last_bg_g || cell->bg_b != last_bg_b)
                {
                    tv_bg(ctx->frame_buffer, &fb_pos, cell->bg_r, cell->bg_g, cell->bg_b);
                    last_bg_r = cell->bg_r;
                    last_bg_g = cell->bg_g;
                    last_bg_b = cell->bg_b;
                }

                // Update FG
                if (cell->fg_r != last_fg_r || cell->fg_g != last_fg_g || cell->fg_b != last_fg_b)
                {
                    tv_fg(ctx->frame_buffer, &fb_pos, cell->fg_r, cell->fg_g, cell->fg_b);
                    last_fg_r = cell->fg_r;
                    last_fg_g = cell->fg_g;
                    last_fg_b = cell->fg_b;
                }

                // Write Char
                for (int k = 0; k < 4 && cell->ch[k] != '\0'; k++)
                {
                    ctx->frame_buffer[fb_pos++] = cell->ch[k];
                }

                cur_r = r;
                cur_c = c + 1;
                *prev = *cell;
            }
        }
    }
    if (!changed)
    {
        return;
    }

    tv_reset(ctx->frame_buffer, &fb_pos);
    memcpy(ctx->frame_buffer + fb_pos, sync_off, sizeof(sync_off) - 1);
    fb_pos += sizeof(sync_off) - 1;

    if (tv_write_all(ctx->frame_buffer, fb_pos) != 0)
    {
        tv_invalidate(ctx); /* partial frame on screen: repaint everything next time */
    }
}

/** tv_draw_text() - Write ASCII @text into grid row @row from column @col, clipped to the width. */
static void tv_draw_text(tv_context_t *ctx, int row, int col, const char *text)
{
    for (int k = 0; text[k] != '\0' && col + k < wcol; k++)
    {
        if (col + k >= 0)
        {
            ctx->screen[row * wcol + col + k].ch[0] = text[k];
        }
    }
}

static const double tv_range_lo[RANGE_NB] = { 0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20 };
static const double tv_range_hi[RANGE_NB] = { 1.0, 0.999, 0.995, 0.99, 0.95, 0.90, 0.85, 0.80 };

/**
 * tv_compute_range() - Display range from the finite samples (out-of-image samples are NaN).
 * Returns -1 on allocation failure. The result is always finite with min < max.
 */
static int tv_compute_range(tv_context_t *ctx, int num_pixels, double *pmin, double *pmax)
{
    double min_val = 0.0, max_val = 1.0;

    if (current_options.range_locked)
    {
        min_val = current_options.manual_min;
        max_val = current_options.manual_max;
    }
    else if (current_options.range == RANGE_MINMAX)
    {
        int nvalid = 0;
        for (int k = 0; k < num_pixels; k++)
        {
            double v = ctx->display_buffer[k];
            if (!tv_finite(v))
            {
                continue;
            }
            if (nvalid++ == 0)
            {
                min_val = max_val = v;
            }
            else if (v < min_val)
            {
                min_val = v;
            }
            else if (v > max_val)
            {
                max_val = v;
            }
        }
    }
    else
    {
        if (num_pixels > ctx->sort_size)
        {
            double *tmp =
                (double *) realloc(ctx->sort_buffer, (size_t) num_pixels * sizeof(double));
            if (tmp == NULL)
            {
                return -1;
            }
            ctx->sort_buffer = tmp;
            ctx->sort_size   = num_pixels;
        }
        int nvalid = 0;
        for (int k = 0; k < num_pixels; k++)
        {
            if (tv_finite(ctx->display_buffer[k]))
            {
                ctx->sort_buffer[nvalid++] = ctx->display_buffer[k];
            }
        }
        if (nvalid > 0)
        {
            qsort(ctx->sort_buffer, (size_t) nvalid, sizeof(double), compare_doubles);
            min_val = ctx->sort_buffer[(int) (tv_range_lo[current_options.range] * (nvalid - 1))];
            max_val = ctx->sort_buffer[(int) (tv_range_hi[current_options.range] * (nvalid - 1))];
        }
    }

    if (!tv_finite(min_val) || !tv_finite(max_val))
    {
        min_val = 0.0;
        max_val = 1.0;
    }
    if (!(max_val > min_val))
    {
        double w = fmax(fabs(min_val) * 1e-6, 0.5);
        min_val -= w;
        max_val += w;
    }
    *pmin = min_val;
    *pmax = max_val;
    return 0;
}

/** tv_handle_input_key() - Line editor for the manual min/max prompt; 1 if the key was consumed. */
static int tv_handle_input_key(tv_context_t *ctx, int ch)
{
    if (ch == 10 || ch == 13)
    {
        double v;
        if (ctx->input_len == 0)
        {
            ctx->input_mode = 0;
        }
        else if (tv_parse_double(ctx->input_buf, &v) != 0)
        {
            ctx->input_err = "invalid number";
        }
        else
        {
            /* the other bound is the locked one, or the range shown right now */
            double lo = current_options.range_locked ? current_options.manual_min : current_min_val;
            double hi = current_options.range_locked ? current_options.manual_max : current_max_val;
            if (ctx->input_mode == 1)
            {
                lo = v;
            }
            else
            {
                hi = v;
            }
            if (!(hi > lo))
            {
                ctx->input_err = "min must be < max";
            }
            else
            {
                current_options.manual_min   = lo;
                current_options.manual_max   = hi;
                current_options.range_locked = true;
                ctx->input_mode              = 0;
            }
        }
    }
    else if (ch == 27)
    {
        ctx->input_mode = 0;
    }
    else if (ch == 127 || ch == 8)
    {
        if (ctx->input_len > 0)
        {
            ctx->input_buf[--ctx->input_len] = '\0';
        }
        ctx->input_err = NULL;
    }
    else if (ch >= 32 && ch <= 126)
    {
        if (ctx->input_len < (int) sizeof(ctx->input_buf) - 1)
        {
            ctx->input_buf[ctx->input_len++] = (char) ch;
            ctx->input_buf[ctx->input_len]   = '\0';
        }
        ctx->input_err = NULL;
    }
    else
    {
        return 0; /* mouse, control and special keys keep their normal meaning */
    }
    if (ctx->input_mode == 0)
    {
        ctx->input_len    = 0;
        ctx->input_buf[0] = '\0';
        ctx->input_err    = NULL;
    }
    force_redraw = 1;
    return 1;
}

/** tv_sample() - Nearest-neighbour samples of the view; NaN marks out-of-image samples. */
static void tv_sample(tv_context_t *ctx,
                      int           disp_img_rows,
                      int           disp_cols,
                      double        center_img_x,
                      double        center_img_y,
                      double        sign_x,
                      double        sign_y,
                      double        step)
{
    const tv_stream_t *st            = ctx->st;
    int                xsize         = (int) st->xsize;
    int                ysize         = (int) st->ysize;
    double             center_disp_x = disp_cols / 2.0;
    double             center_disp_y = disp_img_rows / 2.0;
    int                buf_idx       = 0;

    for (int i = 0; i < disp_img_rows; i++)
    {
        int img_y = (int) (center_img_y + sign_y * (i - center_disp_y) * step);
        for (int j = 0; j < disp_cols; j++)
        {
            int img_x = (int) (center_img_x + sign_x * (j - center_disp_x) * step);
            if (img_x < 0 || img_x >= xsize || img_y < 0 || img_y >= ysize)
            {
                ctx->display_buffer[buf_idx++] = NAN;
                continue;
            }
            size_t idx = (size_t) img_y * (size_t) xsize + (size_t) img_x;
            if (st->datatype == _DATATYPE_FLOAT)
            {
                ctx->display_buffer[buf_idx++] = st->img.array.F[idx];
            }
            else if (st->datatype == _DATATYPE_UINT16)
            {
                ctx->display_buffer[buf_idx++] = st->img.array.UI16[idx];
            }
            else
            {
                ctx->display_buffer[buf_idx++] = get_pixel_value(&st->img, st->datatype, idx);
            }
        }
    }
}

/** tv_grow() - Make sure the grids, sample and frame buffers fit the terminal; -1 on failure. */
static int tv_grow(tv_context_t *ctx, int disp_samples)
{
    if (wrow * wcol > ctx->screen_size)
    {
        size_t     n          = (size_t) wrow * (size_t) wcol;
        tv_cell_t *tmp_screen = (tv_cell_t *) realloc(ctx->screen, n * sizeof(tv_cell_t));
        if (tmp_screen == NULL)
        {
            return -1;
        }
        ctx->screen        = tmp_screen;
        tv_cell_t *tmp_prv = (tv_cell_t *) realloc(ctx->prev_screen, n * sizeof(tv_cell_t));
        if (tmp_prv == NULL)
        {
            return -1;
        }
        ctx->prev_screen = tmp_prv;
        ctx->screen_size = (int) n;
        tv_invalidate(ctx);
    }
    if (disp_samples > ctx->buffer_size)
    {
        double *tmp =
            (double *) realloc(ctx->display_buffer, (size_t) disp_samples * sizeof(double));
        if (tmp == NULL)
        {
            return -1;
        }
        ctx->display_buffer = tmp;
        ctx->buffer_size    = disp_samples;
    }
    size_t needed_fb = tv_framebuf_size(wrow, wcol);
    if (needed_fb > ctx->frame_buffer_size)
    {
        char *tmp_fb = (char *) realloc(ctx->frame_buffer, needed_fb);
        if (tmp_fb == NULL)
        {
            return -1;
        }
        ctx->frame_buffer      = tmp_fb;
        ctx->frame_buffer_size = needed_fb;
    }
    return 0;
}

void termview_options_init(termview_options_t *opt)
{
    memset(opt, 0, sizeof(*opt));
    opt->colormap     = COLORMAP_GREYSCALE;
    opt->scale        = SCALE_LINEAR;
    opt->range        = RANGE_MINMAX;
    opt->range_locked = false;
    opt->manual_min   = 0.0;
    opt->manual_max   = 1.0;
    opt->flip_h       = false;
    opt->flip_v       = false;
    opt->fps          = 20;
}

int termview_run(const char *name, const termview_options_t *opt)
{
    char        err[512];
    tv_stream_t st;

    if (tv_stream_open(&st, name, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "Error: %s\n", err);
        return 1;
    }
    if (tv_term_init(err, sizeof(err)) != 0)
    {
        fprintf(stderr, "Error: %s\n", err);
        tv_stream_close(&st);
        return 1;
    }

    current_options = *opt;

    tv_context_t  context_instance = { 0 };
    tv_context_t *ctx              = &context_instance;
    ctx->st                        = &st;
    ctx->img                       = &st.img;
    ctx->target_fps                = (opt->fps >= 1 && opt->fps <= 120) ? opt->fps : 20;
    ctx->last_cnt0                 = (uint64_t) -1;
    ctx->last_cnt0_for_fps         = st.img.md->cnt0;
    int alloc_failed               = 0;

    clock_gettime(CLOCK_MONOTONIC, &ctx->last_time_real);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ctx->last_time_cpu);
    ctx->last_render_real = ctx->last_time_real;

    if (tv_term_enter() != 0)
    {
        tv_term_restore();
        tv_stream_close(&st);
        fprintf(stderr, "Error: cannot configure the terminal\n");
        return 1;
    }
    tv_term_size(&wrow, &wcol);

    while (loop && !tv_sig_quit && !tv_term_hup)
    {
        if (tv_sig_cont)
        {
            tv_sig_cont = 0;
            if (tv_term_enter() != 0)
            {
                break;
            }
            tv_sig_winch = 1;
        }
        if (tv_sig_winch)
        {
            tv_sig_winch = 0; /* cleared before the query: a resize during it is not lost */
            tv_term_size(&wrow, &wcol);
            tv_invalidate(ctx);
        }

        long            target_us = 1000000L / ctx->target_fps;
        struct timespec now_real;
        clock_gettime(CLOCK_MONOTONIC, &now_real);
        long elapsed_since_last = (now_real.tv_sec - ctx->last_render_real.tv_sec) * 1000000L +
                                  (now_real.tv_nsec - ctx->last_render_real.tv_nsec) / 1000L;
        ctx->timeout_us         = target_us - elapsed_since_last;
        if (ctx->timeout_us < 0)
        {
            ctx->timeout_us = 0;
        }

        if (ctx->popup_type > 0)
        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            long popup_remain_us = (ctx->popup_expiry_time.tv_sec - now.tv_sec) * 1000000L +
                                   (ctx->popup_expiry_time.tv_nsec - now.tv_nsec) / 1000L;
            if (popup_remain_us <= 0)
            {
                ctx->popup_type = 0;
                force_redraw    = 1;
            }
            else if (popup_remain_us < ctx->timeout_us)
            {
                ctx->timeout_us = popup_remain_us;
            }
        }

        int first_wait = 1;
        while (loop && !tv_sig_quit && !tv_sig_cont)
        {
            long wait_us = first_wait ? ctx->timeout_us : 0;
            int  ch      = tv_read_key(wait_us);
            first_wait   = 0;
            if (ch == -1)
            {
                break;
            }
            if (ctx->input_mode > 0 && tv_handle_input_key(ctx, ch))
            {
                continue;
            }
            termview_handle_keyboard_event(ctx, ch);
        }
        if (!loop || tv_sig_quit || tv_term_hup)
        {
            break;
        }

        if (ctx->popup_type > 0)
        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            long popup_remain_us = (ctx->popup_expiry_time.tv_sec - now.tv_sec) * 1000000L +
                                   (ctx->popup_expiry_time.tv_nsec - now.tv_nsec) / 1000L;
            if (popup_remain_us <= 0)
            {
                ctx->popup_type = 0;
                force_redraw    = 1;
            }
        }

        termview_update_fps_stats(ctx);

        if (!force_redraw && st.img.md->cnt0 == ctx->last_cnt0)
        {
            continue;
        }
        ctx->last_cnt0 = st.img.md->cnt0;
        force_redraw   = 0;

        uint32_t xsize = st.xsize;
        uint32_t ysize = st.ysize;

        int disp_char_rows = wrow - TV_INFO_ROWS;
        int disp_cols      = wcol - TV_BAR_WIDTH - 1;
        int bar_col_start  = wcol - TV_BAR_WIDTH;
        int disp_img_rows  = disp_char_rows * 2;

        if (tv_grow(ctx, tv_layout_ok() ? disp_img_rows * disp_cols : 0) != 0)
        {
            alloc_failed = 1;
            break;
        }

        for (int i = 0; i < wrow * wcol; i++)
        {
            tv_cell_t *c = &ctx->screen[i];
            memset(c, 0, sizeof(*c));
            c->fg_r  = 255;
            c->fg_g  = 255;
            c->fg_b  = 255;
            c->ch[0] = ' ';
        }

        if (!tv_layout_ok())
        {
            char msg[64];
            snprintf(msg, sizeof(msg), "terminal too small (%dx%d)", wcol, wrow);
            int len = (int) strlen(msg);
            tv_draw_text(ctx, wrow / 2, (len < wcol) ? (wcol - len) / 2 : 0, msg);
            ctx->mouse_is_dragging = 0;
            ctx->roi_is_dragging   = 0;
            termview_flush_ansi(ctx);
            continue;
        }

        double zmax = 64.0 * fmax((double) xsize / disp_cols, (double) ysize / disp_img_rows);
        tv_view_sanitize(&view_zoom, &view_center_x, &view_center_y, zmax);

        double view_w_img = (double) xsize / view_zoom;
        double view_h_img = (double) ysize / view_zoom;
        double step       = fmax(view_w_img / disp_cols, view_h_img / disp_img_rows);

        double center_img_x = view_center_x * xsize;
        double center_img_y = view_center_y * ysize;

        double sign_y = current_options.flip_v ? -1.0 : 1.0;
        double sign_x = current_options.flip_h ? -1.0 : 1.0;

        tv_sample(ctx, disp_img_rows, disp_cols, center_img_x, center_img_y, sign_x, sign_y, step);

        double min_val, max_val;
        if (tv_compute_range(ctx, disp_img_rows * disp_cols, &min_val, &max_val) != 0)
        {
            alloc_failed = 1;
            break;
        }
        current_min_val = min_val;
        current_max_val = max_val;

        double lut_scale = (TV_LUT_N - 1) / (max_val - min_val);
        if (!tv_finite(lut_scale))
        {
            lut_scale = 0.0;
        }

        if ((int) current_options.colormap != lut_colormap ||
            (int) current_options.scale != lut_scale_mode)
        {
            build_colormap_lut(current_options.colormap, current_options.scale);
        }

        termview_render_image(ctx, disp_char_rows, disp_cols, center_img_x, center_img_y, sign_x,
                              sign_y, step, xsize, ysize, min_val, lut_scale);
        termview_render_colorbar(ctx, disp_char_rows, bar_col_start, min_val, max_val);
        termview_render_infobar(ctx, xsize, ysize, step, min_val, max_val);
        termview_render_popup(ctx, disp_char_rows, disp_cols);
        termview_flush_ansi(ctx);
    }

    tv_term_restore();

    free(ctx->display_buffer);
    free(ctx->sort_buffer);
    free(ctx->screen);
    free(ctx->prev_screen);
    free(ctx->frame_buffer);
    tv_stream_close(&st);

    if (tv_sig_quit)
    {
        tv_term_reraise(tv_sig_quit);
    }
    if (alloc_failed)
    {
        fprintf(stderr, "Error: out of memory\n");
        return 1;
    }
    if (tv_term_hup)
    {
        return 1;
    }
    return exit_status;
}
