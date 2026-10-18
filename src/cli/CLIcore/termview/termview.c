// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file termview.c
 * @brief Termview module (TrueColor Standalone Rewrite)
 */

#include "termview.h"
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

/** tv_parse_sgr_mouse() - Decode "<b;x;y" + final M/m into @ev; 0 if valid. */
static int tv_parse_sgr_mouse(const char *par, int fin, tv_mouse_event_t *ev)
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

    int b       = v[0];
    ev->is_drag = ((b & 32) != 0);
    if (b & 64)
    {
        if (fin == 'm')
        {
            return -1; /* wheel "release" carries no action */
        }
        ev->button  = 64 + (b & 3);
        ev->is_drag = 0;
    }
    else
    {
        ev->button = b & 3;
    }
    ev->x        = v[1];
    ev->y        = v[2];
    ev->is_press = (fin == 'M');
    return 0;
}

/**
 * tv_read_key() - Next key, TV_KEY_* code, -1 if none, or TV_KEY_EOF.
 * Escape sequences are collected whole (waiting up to TV_ESC_TIMEOUT_US per byte); incomplete or
 * unknown sequences are discarded so their bytes never act as commands.
 */
static int tv_read_key(long timeout_us, tv_mouse_event_t *ev)
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
        return (tv_parse_sgr_mouse(par, fin, ev) == 0) ? TV_KEY_MOUSE : -1;
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

static void termview_update_fps_stats(tv_state_t *ctx)
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

        uint64_t current_cnt0 = ctx->st->img.md->cnt0;
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
static int tv_layout_ok(const tv_state_t *ctx)
{
    return (ctx->rows - TV_INFO_ROWS) >= 2 && (ctx->cols - TV_BAR_WIDTH - 1) >= 4;
}

/** tv_invalidate() - Forget what the terminal shows so the next frame repaints every cell. */
static void tv_invalidate(tv_state_t *ctx)
{
    if (ctx->prev_screen)
    {
        for (int i = 0; i < ctx->screen_size; i++)
        {
            ctx->prev_screen[i] = TV_CELL_UNKNOWN;
        }
    }
    ctx->dirty = 1;
}

static void termview_handle_mouse_event(tv_state_t *ctx)
{
    ctx->dirty = 1;
    if (!tv_layout_ok(ctx))
    {
        ctx->mouse_is_dragging = 0;
        ctx->roi_is_dragging   = 0;
        return;
    }

    int    mx            = ctx->mouse.x;
    int    my            = ctx->mouse.y;
    double sign_x        = ctx->opt.flip_h ? -1.0 : 1.0;
    double sign_y        = ctx->opt.flip_v ? -1.0 : 1.0;
    double xsize         = ctx->st->xsize;
    double ysize         = ctx->st->ysize;
    int    disp_cols     = ctx->cols - TV_BAR_WIDTH - 1;
    int    disp_img_rows = (ctx->rows - TV_INFO_ROWS) * 2;
    double step = fmax(xsize / ctx->view.zoom / disp_cols, ysize / ctx->view.zoom / disp_img_rows);

    if (ctx->mouse.button == 64)
    {
        // Scroll Up -> Zoom in
        ctx->view.zoom *= 1.2;
    }
    else if (ctx->mouse.button == 65)
    {
        // Scroll Down -> Zoom out
        ctx->view.zoom /= 1.2;
    }
    else if (ctx->mouse.button == 0) // Left Button
    {
        if (ctx->mouse.is_press && !ctx->mouse.is_drag)
        {
            ctx->mouse_is_dragging = 1;
            ctx->last_mouse_x      = mx;
            ctx->last_mouse_y      = my;
        }
        else if (ctx->mouse.is_press && ctx->mouse.is_drag && ctx->mouse_is_dragging)
        {
            /* grab: the image follows the pointer on both axes, whatever the flips */
            int dx = mx - ctx->last_mouse_x;
            int dy = my - ctx->last_mouse_y;
            ctx->view.cx -= sign_x * dx * step / xsize;
            ctx->view.cy -= sign_y * dy * 2.0 * step / ysize;
            ctx->last_mouse_x = mx;
            ctx->last_mouse_y = my;
        }
        else if (!ctx->mouse.is_press)
        {
            ctx->mouse_is_dragging = 0;
        }
    }
    else if (ctx->mouse.button == 2) // Right Button
    {
        if (ctx->mouse.is_press && !ctx->mouse.is_drag)
        {
            ctx->roi_is_dragging = 1;
            ctx->roi_start_x     = mx;
            ctx->roi_start_y     = my;
            ctx->roi_end_x       = mx;
            ctx->roi_end_y       = my;
        }
        else if (ctx->mouse.is_press && ctx->mouse.is_drag && ctx->roi_is_dragging)
        {
            ctx->roi_end_x = mx;
            ctx->roi_end_y = my;
        }
        else if (!ctx->mouse.is_press && ctx->roi_is_dragging)
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
                    ctx->view.cx * xsize + sign_x * (roi_center_mx - 1.0 - center_disp_x) * step;
                double new_cy = ctx->view.cy * ysize + sign_y * (roi_row - center_disp_y) * step;

                double roi_w_char = max_x - min_x + 1;
                double roi_h_char = max_y - min_y + 1;
                double new_step_x = (roi_w_char * step) / disp_cols;
                double new_step_y = (roi_h_char * 2 * step) / disp_img_rows;
                double new_step   = fmax(new_step_x, new_step_y);

                ctx->view.cx = new_cx / xsize;
                ctx->view.cy = new_cy / ysize;
                ctx->view.zoom *= (step / new_step);
            }
        }
    }
}

static void termview_handle_keyboard_event(tv_state_t *ctx, int ch)
{
    switch (ch)
    {
    case 10: // ENTER key (LF)
    case 13: // ENTER key (CR)
    case 27: // ESC key
        if (ctx->popup_type > 0)
        {
            ctx->popup_type = 0;
            ctx->dirty      = 1;
        }
        break;
    case 'q':
    case TV_CTRL('c'):
    case TV_CTRL('\\'):
        ctx->running = 0;
        break;
    case TV_KEY_EOF:
        ctx->running     = 0;
        ctx->exit_status = 1;
        break;
    case TV_CTRL('z'):
        tv_term_suspend();
        break;
    case TV_CTRL('l'):
        tv_invalidate(ctx);
        break;
    case 'c':
        ctx->dirty        = 1;
        ctx->opt.colormap = (ctx->opt.colormap + 1) % COLORMAP_NB;
        ctx->popup_type   = 1;
        clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
        ctx->popup_expiry_time.tv_sec += 2;
        break;
    case 's':
        ctx->dirty      = 1;
        ctx->opt.scale  = (ctx->opt.scale + 1) % SCALE_NB;
        ctx->popup_type = 2;
        clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
        ctx->popup_expiry_time.tv_sec += 2;
        break;
    case 'r':
        ctx->dirty      = 1;
        ctx->opt.range  = (ctx->opt.range + 1) % RANGE_NB;
        ctx->popup_type = 3;
        clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
        ctx->popup_expiry_time.tv_sec += 2;
        break;
    case 'l':
        ctx->dirty            = 1;
        ctx->opt.range_locked = !ctx->opt.range_locked;
        if (ctx->opt.range_locked)
        {
            ctx->opt.manual_min = ctx->cur_min;
            ctx->opt.manual_max = ctx->cur_max;
        }
        break;
    case 'm':
    case 'M':
        ctx->dirty        = 1;
        ctx->input_mode   = (ch == 'm') ? 1 : 2;
        ctx->input_len    = 0;
        ctx->input_buf[0] = '\0';
        ctx->input_err    = NULL;
        break;
    case 'h':
        ctx->dirty      = 1;
        ctx->opt.flip_h = !ctx->opt.flip_h;
        break;
    case 'v':
        ctx->dirty      = 1;
        ctx->opt.flip_v = !ctx->opt.flip_v;
        break;
    case '+':
    case '=':
        ctx->dirty = 1;
        ctx->view.zoom *= 1.2;
        break;
    case '-':
    case '_':
        ctx->dirty = 1;
        ctx->view.zoom /= 1.2;
        break;
    case '0':
        ctx->dirty     = 1;
        ctx->view.zoom = 1.0;
        ctx->view.cx   = 0.5;
        ctx->view.cy   = 0.5;
        break;
    /* arrows move the viewport in screen space: Left shows what lies left of the screen */
    case TV_KEY_LEFT:
        ctx->dirty = 1;
        ctx->view.cx -= (ctx->opt.flip_h ? -0.1 : 0.1) / ctx->view.zoom;
        break;
    case TV_KEY_RIGHT:
        ctx->dirty = 1;
        ctx->view.cx += (ctx->opt.flip_h ? -0.1 : 0.1) / ctx->view.zoom;
        break;
    case TV_KEY_UP:
        ctx->dirty = 1;
        if (ctx->popup_type == 1)
        {
            ctx->opt.colormap = (ctx->opt.colormap + COLORMAP_NB - 1) % COLORMAP_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 2)
        {
            ctx->opt.scale = (ctx->opt.scale + SCALE_NB - 1) % SCALE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 3)
        {
            ctx->opt.range = (ctx->opt.range + RANGE_NB - 1) % RANGE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else
        {
            ctx->view.cy -= (ctx->opt.flip_v ? -0.1 : 0.1) / ctx->view.zoom;
        }
        break;
    case TV_KEY_DOWN:
        ctx->dirty = 1;
        if (ctx->popup_type == 1)
        {
            ctx->opt.colormap = (ctx->opt.colormap + 1) % COLORMAP_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 2)
        {
            ctx->opt.scale = (ctx->opt.scale + 1) % SCALE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else if (ctx->popup_type == 3)
        {
            ctx->opt.range = (ctx->opt.range + 1) % RANGE_NB;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
        }
        else
        {
            ctx->view.cy += (ctx->opt.flip_v ? -0.1 : 0.1) / ctx->view.zoom;
        }
        break;
    case '<':
    case ',':
        ctx->dirty = 1;
        ctx->target_fps -= 1;
        if (ctx->target_fps < 1)
        {
            ctx->target_fps = 1;
        }
        break;
    case '>':
    case '.':
        ctx->dirty = 1;
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

static void termview_render_image(tv_state_t *ctx,
                                  int         disp_char_rows,
                                  int         disp_cols,
                                  double      center_img_x,
                                  double      center_img_y,
                                  double      sign_x,
                                  double      sign_y,
                                  double      step,
                                  uint32_t    xsize,
                                  uint32_t    ysize,
                                  double      min_val,
                                  double      lut_scale)
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

            tv_rgb_t rgb_top = ctx->lut[tv_lut_idx(val_top, min_val, lut_scale)];
            tv_rgb_t rgb_bot = ctx->lut[tv_lut_idx(val_bot, min_val, lut_scale)];

            if (!in_bounds_top)
            {
                if (IS_IN_BOUNDS(j - 1, i * 2) || IS_IN_BOUNDS(j + 1, i * 2) ||
                    IS_IN_BOUNDS(j, i * 2 - 1) || IS_IN_BOUNDS(j, i * 2 + 1))
                {
                    rgb_top = (tv_rgb_t) { 64, 64, 64 }; // Dark gray border
                }
                else
                {
                    rgb_top = (tv_rgb_t) { 0, 0, 0 };
                }
            }

            if (!in_bounds_bot)
            {
                if (IS_IN_BOUNDS(j - 1, i * 2 + 1) || IS_IN_BOUNDS(j + 1, i * 2 + 1) ||
                    IS_IN_BOUNDS(j, i * 2) || IS_IN_BOUNDS(j, i * 2 + 2))
                {
                    rgb_bot = (tv_rgb_t) { 64, 64, 64 }; // Dark gray border
                }
                else
                {
                    rgb_bot = (tv_rgb_t) { 0, 0, 0 };
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
                        rgb_top = (tv_rgb_t) { 255, 255, 0 }; // Yellow border
                        rgb_bot = (tv_rgb_t) { 255, 255, 0 };
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

            ctx->screen[i * ctx->cols + j] =
                tv_cell(tv_rgb(rgb_top), tv_rgb(rgb_bot), TV_G_LOWER_HALF);
        }
    }
#undef IS_IN_BOUNDS
}

static void termview_render_colorbar(tv_state_t *ctx,
                                     int         disp_char_rows,
                                     int         bar_col_start,
                                     double      min_val,
                                     double      max_val)
{
    for (int i = 0; i < disp_char_rows; i++)
    {
        double   norm_val = (disp_char_rows > 1) ? 1.0 - (double) i / (disp_char_rows - 1) : 0.5;
        tv_rgb_t rgb      = ctx->lut[tv_lut_idx(norm_val, 0.0, TV_LUT_N - 1)];

        tv_cell_t bar                                  = tv_cell(0, tv_rgb(rgb), TV_G_FULL_BLOCK);
        ctx->screen[i * ctx->cols + bar_col_start - 1] = bar;
        ctx->screen[i * ctx->cols + bar_col_start]     = bar;

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
            if (bar_col_start + 2 + k < ctx->cols)
            {
                tv_cell_t *lc = &ctx->screen[i * ctx->cols + bar_col_start + 2 + k];
                *lc           = tv_cell_with_glyph(*lc, (uint8_t) label[k]);
            }
        }
    }
}

static void termview_render_infobar(tv_state_t *ctx,
                                    uint32_t    xsize,
                                    uint32_t    ysize,
                                    double      step,
                                    double      min_val,
                                    double      max_val)
{
    int info_row = ctx->rows - 3;

    const char *cmap_str  = tv_cmap_names[ctx->opt.colormap];
    const char *scale_str = tv_scale_names[ctx->opt.scale];
    const char *range_str = tv_range_names[ctx->opt.range];

    char info[256];
    int  len;

    int name_len = (int) strlen(ctx->st->name);
    len = snprintf(info, sizeof(info),
                   "Image: %s [%u x %u] Type: %d  |  CPU: %5.1f%%  UI: %3d FPS  Stream: %5.1f Hz",
                   ctx->st->name, xsize, ysize, ctx->st->datatype, ctx->current_cpu_usage,
                   ctx->target_fps, ctx->current_stream_fps);
    len = tv_fmt_len(len, sizeof(info));
    for (int k = 0; k < len && k < ctx->cols; k++)
    {
        tv_cell_t *c = &ctx->screen[info_row * ctx->cols + k];
        *c           = tv_cell_with_glyph(*c, (uint8_t) info[k]);
        if (k >= 7 && k < 7 + name_len)
        {
            *c = tv_cell_with_fg(*c, TV_RGB(0, 255, 255)); // Cyan
        }
    }
    len = snprintf(info, sizeof(info),
                   "Val: [%.4g : %.4g] Zoom: %.2fx (%.2f px/char H, %.2f px/char V)", min_val,
                   max_val, ctx->view.zoom, step, step * 2.0);
    len = tv_fmt_len(len, sizeof(info));
    for (int k = 0; k < len && k < ctx->cols; k++)
    {
        tv_cell_t *c = &ctx->screen[(info_row + 1) * ctx->cols + k];
        *c           = tv_cell_with_glyph(*c, (uint8_t) info[k]);
    }

    if (ctx->input_mode > 0)
    {
        len = snprintf(info, sizeof(info), "Enter Manual %s: %s_%s%s",
                       ctx->input_mode == 1 ? "Min" : "Max", ctx->input_buf,
                       ctx->input_err ? "   " : "", ctx->input_err ? ctx->input_err : "");
        len = tv_fmt_len(len, sizeof(info));
        for (int k = 0; k < len && k < ctx->cols; k++)
        {
            tv_cell_t *c = &ctx->screen[(info_row + 2) * ctx->cols + k];
            *c = tv_cell_with_fg(tv_cell_with_glyph(*c, (uint8_t) info[k]), TV_RGB(255, 255, 0));
        }
        for (int k = len; k < ctx->cols; k++)
        {
            tv_cell_t *c = &ctx->screen[(info_row + 2) * ctx->cols + k];
            *c           = tv_cell_with_glyph(*c, ' ');
        }
    }
    else
    {
        char info1[128];
        int  len1 = snprintf(info1, sizeof(info1), "[C]map: %s  [S]cale: %s  [R]ange: %s [",
                             cmap_str, scale_str, range_str);
        len1      = tv_fmt_len(len1, sizeof(info1));
        const char *lock_str = ctx->opt.range_locked ? "LOCKED" : "AUTO";
        int         lock_len = (int) strlen(lock_str);
        char        info2[256];
        int         len2 =
            snprintf(info2, sizeof(info2), "%s%s]  (< >:FPS l:Lock m/M:MinMax ", info1, lock_str);
        len2 = tv_fmt_len(len2, sizeof(info2));

        const char *h_str = ctx->opt.flip_h ? "h:FLIPH" : "h:fliph";
        const char *v_str = ctx->opt.flip_v ? "v:FLIPV" : "v:flipv";

        len = snprintf(info, sizeof(info), "%s%s %s q:Quit)", info2, h_str, v_str);
        len = tv_fmt_len(len, sizeof(info));

        for (int k = 0; k < len && k < ctx->cols; k++)
        {
            tv_cell_t *c = &ctx->screen[(info_row + 2) * ctx->cols + k];
            *c           = tv_cell_with_glyph(*c, (uint8_t) info[k]);
            if (k >= len1 && k < len1 + lock_len)
            {
                // Red for Locked, green for Auto
                *c = tv_cell_with_fg(*c, ctx->opt.range_locked ? TV_RGB(255, 50, 50)
                                                               : TV_RGB(50, 255, 50));
            }
            if (ctx->opt.flip_h && k >= len2 && k < len2 + (int) strlen(h_str))
            {
                *c = tv_cell_with_fg(*c, TV_RGB(255, 255, 0)); // Yellow
            }
            if (ctx->opt.flip_v && k >= len2 + (int) strlen(h_str) + 1 &&
                k < len2 + (int) strlen(h_str) + 1 + (int) strlen(v_str))
            {
                *c = tv_cell_with_fg(*c, TV_RGB(255, 255, 0)); // Yellow
            }
        }
        for (int k = len; k < ctx->cols; k++)
        {
            tv_cell_t *c = &ctx->screen[(info_row + 2) * ctx->cols + k];
            *c           = tv_cell_with_glyph(*c, ' ');
        }
    }
}

static void termview_render_popup(tv_state_t *ctx, int disp_char_rows, int disp_cols)
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
        selected_idx = ctx->opt.colormap;
        box_w        = 40; // Wider box to fit colormap preview
    }
    else if (ctx->popup_type == 2)
    {
        title        = "SCALE";
        names        = tv_scale_names;
        num_opts     = SCALE_NB;
        selected_idx = ctx->opt.scale;
    }
    else if (ctx->popup_type == 3)
    {
        title        = "RANGE";
        names        = tv_range_names;
        num_opts     = RANGE_NB;
        selected_idx = ctx->opt.range;
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

            uint32_t bg    = TV_RGB(50, 50, 50);    // dark gray bg
            uint32_t fg    = TV_RGB(255, 255, 255); // white fg
            uint8_t  glyph = ' ';

            if (i == 0)
            {
                int tlen   = (int) strlen(title);
                int tstart = (box_w - tlen) / 2;
                if (j >= tstart && j < tstart + tlen)
                {
                    glyph = (uint8_t) title[j - tstart];
                    fg    = TV_RGB(255, 255, 0); // Yellow title
                }
            }
            else if (i <= num_opts)
            {
                int opt_idx = i - 1;
                if (opt_idx == selected_idx)
                {
                    bg = TV_RGB(200, 200, 200); // light gray selected
                    fg = TV_RGB(0, 0, 0);
                }

                // Text portion
                if (j >= 2 && j - 2 < (int) strlen(names[opt_idx]))
                {
                    glyph = (uint8_t) names[opt_idx][j - 2];
                }

                // Compressed colormap rendering
                if (ctx->popup_type == 1 && j >= 16 && j < box_w - 2)
                {
                    int    bar_len = box_w - 2 - 16;
                    double v       = (double) (j - 16) / (double) (bar_len - 1);
                    bg             = tv_rgb(tv_cmap_color((termview_colormap_t) opt_idx, v));
                    glyph          = ' ';
                }
            }
            ctx->screen[r * ctx->cols + c] = tv_cell(bg, fg, glyph);
        }
    }
}

static void termview_flush_ansi(tv_state_t *ctx)
{
    size_t n =
        tv_flush_diff(ctx->frame_buffer, ctx->screen, ctx->prev_screen, ctx->rows, ctx->cols);
    if (n > 0 && tv_write_all(ctx->frame_buffer, n) != 0)
    {
        tv_invalidate(ctx); /* partial frame on screen: repaint everything next time */
    }
}

/** tv_draw_text() - Write ASCII @text into grid row @row from column @col, clipped to the width. */
static void tv_draw_text(tv_state_t *ctx, int row, int col, const char *text)
{
    for (int k = 0; text[k] != '\0' && col + k < ctx->cols; k++)
    {
        if (col + k >= 0)
        {
            tv_cell_t *c = &ctx->screen[row * ctx->cols + col + k];
            *c           = tv_cell_with_glyph(*c, (uint8_t) text[k]);
        }
    }
}

static const double tv_range_lo[RANGE_NB] = { 0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20 };
static const double tv_range_hi[RANGE_NB] = { 1.0, 0.999, 0.995, 0.99, 0.95, 0.90, 0.85, 0.80 };

/**
 * tv_compute_range() - Display range from the finite samples (out-of-image samples are NaN).
 * Returns -1 on allocation failure. The result is always finite with min < max.
 */
static int tv_compute_range(tv_state_t *ctx, int num_pixels, double *pmin, double *pmax)
{
    double min_val = 0.0, max_val = 1.0;

    if (ctx->opt.range_locked)
    {
        min_val = ctx->opt.manual_min;
        max_val = ctx->opt.manual_max;
    }
    else if (ctx->opt.range == RANGE_MINMAX)
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
            min_val = ctx->sort_buffer[(int) (tv_range_lo[ctx->opt.range] * (nvalid - 1))];
            max_val = ctx->sort_buffer[(int) (tv_range_hi[ctx->opt.range] * (nvalid - 1))];
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
static int tv_handle_input_key(tv_state_t *ctx, int ch)
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
            double lo = ctx->opt.range_locked ? ctx->opt.manual_min : ctx->cur_min;
            double hi = ctx->opt.range_locked ? ctx->opt.manual_max : ctx->cur_max;
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
                ctx->opt.manual_min   = lo;
                ctx->opt.manual_max   = hi;
                ctx->opt.range_locked = true;
                ctx->input_mode       = 0;
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
    ctx->dirty = 1;
    return 1;
}

/** tv_sample() - Nearest-neighbour samples of the view; NaN marks out-of-image samples. */
static void tv_sample(tv_state_t *ctx,
                      int         disp_img_rows,
                      int         disp_cols,
                      double      center_img_x,
                      double      center_img_y,
                      double      sign_x,
                      double      sign_y,
                      double      step)
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
static int tv_grow(tv_state_t *ctx, int disp_samples)
{
    if (ctx->rows * ctx->cols > ctx->screen_size)
    {
        size_t     n          = (size_t) ctx->rows * (size_t) ctx->cols;
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
    size_t needed_fb = tv_framebuf_size(ctx->rows, ctx->cols);
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

    tv_state_t *ctx = (tv_state_t *) calloc(1, sizeof(tv_state_t));
    if (ctx == NULL)
    {
        fprintf(stderr, "Error: out of memory\n");
        tv_stream_close(&st);
        return 1;
    }
    ctx->st                = &st;
    ctx->opt               = *opt;
    ctx->view.zoom         = 1.0;
    ctx->view.cx           = 0.5;
    ctx->view.cy           = 0.5;
    ctx->running           = 1;
    ctx->dirty             = 1;
    ctx->lut_cmap          = -1;
    ctx->lut_scale         = -1;
    ctx->cur_max           = 1.0;
    ctx->target_fps        = (opt->fps >= 1 && opt->fps <= 120) ? opt->fps : 20;
    ctx->last_cnt0         = (uint64_t) -1;
    ctx->last_cnt0_for_fps = st.img.md->cnt0;
    int alloc_failed       = 0;

    clock_gettime(CLOCK_MONOTONIC, &ctx->last_time_real);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ctx->last_time_cpu);
    ctx->last_render_real = ctx->last_time_real;

    if (tv_term_enter() != 0)
    {
        tv_term_restore();
        tv_stream_close(&st);
        free(ctx);
        fprintf(stderr, "Error: cannot configure the terminal\n");
        return 1;
    }
    tv_term_size(&ctx->rows, &ctx->cols);

    while (ctx->running && !tv_sig_quit && !tv_term_hup)
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
            tv_term_size(&ctx->rows, &ctx->cols);
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
                ctx->dirty      = 1;
            }
            else if (popup_remain_us < ctx->timeout_us)
            {
                ctx->timeout_us = popup_remain_us;
            }
        }

        int first_wait = 1;
        while (ctx->running && !tv_sig_quit && !tv_sig_cont)
        {
            long wait_us = first_wait ? ctx->timeout_us : 0;
            int  ch      = tv_read_key(wait_us, &ctx->mouse);
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
        if (!ctx->running || tv_sig_quit || tv_term_hup)
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
                ctx->dirty      = 1;
            }
        }

        termview_update_fps_stats(ctx);

        if (!ctx->dirty && st.img.md->cnt0 == ctx->last_cnt0)
        {
            continue;
        }
        ctx->last_cnt0 = st.img.md->cnt0;
        ctx->dirty     = 0;

        uint32_t xsize = st.xsize;
        uint32_t ysize = st.ysize;

        int disp_char_rows = ctx->rows - TV_INFO_ROWS;
        int disp_cols      = ctx->cols - TV_BAR_WIDTH - 1;
        int bar_col_start  = ctx->cols - TV_BAR_WIDTH;
        int disp_img_rows  = disp_char_rows * 2;

        if (tv_grow(ctx, tv_layout_ok(ctx) ? disp_img_rows * disp_cols : 0) != 0)
        {
            alloc_failed = 1;
            break;
        }

        tv_cell_t blank = tv_cell(TV_RGB(0, 0, 0), TV_RGB(255, 255, 255), ' ');
        for (int i = 0; i < ctx->rows * ctx->cols; i++)
        {
            ctx->screen[i] = blank;
        }

        if (!tv_layout_ok(ctx))
        {
            char msg[64];
            snprintf(msg, sizeof(msg), "terminal too small (%dx%d)", ctx->cols, ctx->rows);
            int len = (int) strlen(msg);
            tv_draw_text(ctx, ctx->rows / 2, (len < ctx->cols) ? (ctx->cols - len) / 2 : 0, msg);
            ctx->mouse_is_dragging = 0;
            ctx->roi_is_dragging   = 0;
            termview_flush_ansi(ctx);
            continue;
        }

        double zmax = 64.0 * fmax((double) xsize / disp_cols, (double) ysize / disp_img_rows);
        tv_view_sanitize(&ctx->view.zoom, &ctx->view.cx, &ctx->view.cy, zmax);

        double view_w_img = (double) xsize / ctx->view.zoom;
        double view_h_img = (double) ysize / ctx->view.zoom;
        double step       = fmax(view_w_img / disp_cols, view_h_img / disp_img_rows);

        double center_img_x = ctx->view.cx * xsize;
        double center_img_y = ctx->view.cy * ysize;

        double sign_y = ctx->opt.flip_v ? -1.0 : 1.0;
        double sign_x = ctx->opt.flip_h ? -1.0 : 1.0;

        tv_sample(ctx, disp_img_rows, disp_cols, center_img_x, center_img_y, sign_x, sign_y, step);

        double min_val, max_val;
        if (tv_compute_range(ctx, disp_img_rows * disp_cols, &min_val, &max_val) != 0)
        {
            alloc_failed = 1;
            break;
        }
        ctx->cur_min = min_val;
        ctx->cur_max = max_val;

        double lut_scale = (TV_LUT_N - 1) / (max_val - min_val);
        if (!tv_finite(lut_scale))
        {
            lut_scale = 0.0;
        }

        if ((int) ctx->opt.colormap != ctx->lut_cmap || (int) ctx->opt.scale != ctx->lut_scale)
        {
            tv_lut_build(ctx->lut, ctx->opt.colormap, ctx->opt.scale);
            ctx->lut_cmap  = ctx->opt.colormap;
            ctx->lut_scale = ctx->opt.scale;
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
    int status = ctx->exit_status;
    free(ctx);
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
    return status;
}
