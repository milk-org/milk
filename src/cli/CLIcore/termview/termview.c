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
#include <poll.h>
#include <time.h>
#include <stdbool.h>

#define TV_CTRL(c) ((c) & 0x1F)
#define TV_POPUP_US 2000000 /* quick-pick popups close after 2 s */

#define TV_BAR_WIDTH 12 /* colorbar + labels */
#define TV_INFO_ROWS 4  /* rows below the image */


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

/** tv_update_stats() - Refresh the CPU usage and stream rate shown in the info bar. */
static void tv_update_stats(tv_state_t *ctx, int64_t now_us)
{
    double real_diff = (now_us - ctx->stats_time_us) / 1e6;
    if (real_diff <= 0.0)
    {
        return;
    }
    struct timespec now_cpu;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now_cpu);
    double cpu_diff =
        (now_cpu.tv_sec - ctx->stats_cpu.tv_sec) + (now_cpu.tv_nsec - ctx->stats_cpu.tv_nsec) / 1e9;
    ctx->current_cpu_usage = cpu_diff / real_diff * 100.0;

    uint64_t cnt0 = ctx->st->img.md->cnt0;
    /* a counter that went backwards means the stream was reset: re-baseline */
    ctx->current_stream_fps = (cnt0 >= ctx->last_cnt0_for_fps)
                                  ? (double) (cnt0 - ctx->last_cnt0_for_fps) / real_diff
                                  : 0.0;
    ctx->last_cnt0_for_fps  = cnt0;
    ctx->stats_time_us      = now_us;
    ctx->stats_cpu          = now_cpu;
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
    ctx->dirty |= TV_D_FULL | TV_D_GEOM;
}

/** tv_popup_open() - Show quick-pick popup @type (1 cmap, 2 scale, 3 range) for TV_POPUP_US. */
static void tv_popup_open(tv_state_t *ctx, int type)
{
    ctx->popup_type      = type;
    ctx->popup_expiry_us = tv_now_us() + TV_POPUP_US;
}

/** tv_zmax() - Largest zoom for the current layout (a sample step of 1/64 pixel). */
static double tv_zmax(const tv_state_t *ctx)
{
    if (!tv_layout_ok(ctx))
    {
        return 1.0;
    }
    int disp_cols     = ctx->cols - TV_BAR_WIDTH - 1;
    int disp_img_rows = (ctx->rows - TV_INFO_ROWS) * 2;
    return 64.0 *
           fmax((double) ctx->st->xsize / disp_cols, (double) ctx->st->ysize / disp_img_rows);
}

void tv_set_view(tv_state_t *ctx, double cx, double cy, double zoom)
{
    ctx->view.cx   = cx;
    ctx->view.cy   = cy;
    ctx->view.zoom = zoom;
    tv_view_sanitize(&ctx->view.zoom, &ctx->view.cx, &ctx->view.cy, tv_zmax(ctx));
    ctx->dirty |= TV_D_GEOM | TV_D_CHROME;
}

void tv_set_range(tv_state_t *ctx, double lo, double hi)
{
    ctx->opt.manual_min   = lo;
    ctx->opt.manual_max   = hi;
    ctx->opt.range_locked = true;
    ctx->dirty |= TV_D_MAP | TV_D_CHROME;
}

static int tv_wrap(int v, int step, int n)
{
    return ((v + step) % n + n) % n;
}

void tv_do(tv_state_t *ctx, tv_action_t a, int arg)
{
    switch (a)
    {
    case TV_ACT_QUIT:
        ctx->running = 0;
        break;
    case TV_ACT_SUSPEND:
        tv_term_suspend();
        break;
    case TV_ACT_REPAINT:
        tv_invalidate(ctx);
        break;
    case TV_ACT_CMAP:
        ctx->opt.colormap = (termview_colormap_t) tv_wrap(ctx->opt.colormap, arg, COLORMAP_NB);
        tv_popup_open(ctx, 1);
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_SCALE:
        ctx->opt.scale = (termview_scale_t) tv_wrap(ctx->opt.scale, arg, SCALE_NB);
        tv_popup_open(ctx, 2);
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_RANGE:
        ctx->opt.range = (termview_range_t) tv_wrap(ctx->opt.range, arg, RANGE_NB);
        tv_popup_open(ctx, 3);
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_LOCK:
        ctx->opt.range_locked = !ctx->opt.range_locked;
        if (ctx->opt.range_locked)
        {
            ctx->opt.manual_min = ctx->cur_min;
            ctx->opt.manual_max = ctx->cur_max;
        }
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_PROMPT:
        ctx->input_mode   = (arg == 2) ? 2 : 1;
        ctx->input_len    = 0;
        ctx->input_buf[0] = '\0';
        ctx->input_err    = NULL;
        ctx->dirty |= TV_D_CHROME;
        break;
    case TV_ACT_FLIP_H:
        ctx->opt.flip_h = !ctx->opt.flip_h;
        ctx->dirty |= TV_D_GEOM | TV_D_CHROME;
        break;
    case TV_ACT_FLIP_V:
        ctx->opt.flip_v = !ctx->opt.flip_v;
        ctx->dirty |= TV_D_GEOM | TV_D_CHROME;
        break;
    case TV_ACT_ZOOM:
        tv_set_view(ctx, ctx->view.cx, ctx->view.cy, ctx->view.zoom * pow(1.2, arg));
        break;
    case TV_ACT_VIEW_RESET:
        tv_set_view(ctx, 0.5, 0.5, 1.0);
        break;
    /* arrows move the viewport in screen space: Left shows what lies left of the screen */
    case TV_ACT_PAN_X:
        tv_set_view(ctx, ctx->view.cx + arg * (ctx->opt.flip_h ? -0.1 : 0.1) / ctx->view.zoom,
                    ctx->view.cy, ctx->view.zoom);
        break;
    case TV_ACT_PAN_Y:
        tv_set_view(ctx, ctx->view.cx,
                    ctx->view.cy + arg * (ctx->opt.flip_v ? -0.1 : 0.1) / ctx->view.zoom,
                    ctx->view.zoom);
        break;
    case TV_ACT_FPS:
        ctx->target_fps += arg;
        ctx->target_fps = (ctx->target_fps < 1)     ? 1
                          : (ctx->target_fps > 120) ? 120
                                                    : ctx->target_fps;
        ctx->dirty |= TV_D_CHROME;
        break;
    case TV_ACT_ESCAPE:
        if (ctx->mouse_is_dragging || ctx->roi_is_dragging)
        {
            ctx->mouse_is_dragging = 0;
            ctx->roi_is_dragging   = 0;
            ctx->dirty |= TV_D_CHROME;
        }
        if (ctx->popup_type > 0)
        {
            ctx->popup_type = 0;
            ctx->dirty |= TV_D_CHROME;
        }
        break;
    case TV_ACT_POPUP_CLOSE:
        if (ctx->popup_type > 0)
        {
            ctx->popup_type = 0;
            ctx->dirty |= TV_D_CHROME;
        }
        break;
    }
}

/** tv_handle_key() - Map a key event to an action. */
void tv_handle_key(tv_state_t *ctx, const tv_event_t *ev)
{
    if ((ev->mods & TV_MOD_ALT) && ev->key < TV_K_UP)
    {
        return; /* Alt+character is unbound; special keys keep their meaning with any modifier */
    }
    /* Up/Down step through the open quick-pick popup */
    static const tv_action_t popup_act[4] = { TV_ACT_ESCAPE, TV_ACT_CMAP, TV_ACT_SCALE,
                                              TV_ACT_RANGE };
    switch (ev->key)
    {
    case 'q':
    case TV_CTRL('c'):
    case TV_CTRL('\\'):
        tv_do(ctx, TV_ACT_QUIT, 0);
        break;
    case TV_CTRL('z'):
        tv_do(ctx, TV_ACT_SUSPEND, 0);
        break;
    case TV_CTRL('l'):
        tv_do(ctx, TV_ACT_REPAINT, 0);
        break;
    case TV_K_ESC:
        tv_do(ctx, TV_ACT_ESCAPE, 0);
        break;
    case '\r':
    case '\n':
        tv_do(ctx, TV_ACT_POPUP_CLOSE, 0);
        break;
    case 'c':
        tv_do(ctx, TV_ACT_CMAP, 1);
        break;
    case 's':
        tv_do(ctx, TV_ACT_SCALE, 1);
        break;
    case 'r':
        tv_do(ctx, TV_ACT_RANGE, 1);
        break;
    case 'l':
        tv_do(ctx, TV_ACT_LOCK, 0);
        break;
    case 'm':
        tv_do(ctx, TV_ACT_PROMPT, 1);
        break;
    case 'M':
        tv_do(ctx, TV_ACT_PROMPT, 2);
        break;
    case 'h':
        tv_do(ctx, TV_ACT_FLIP_H, 0);
        break;
    case 'v':
        tv_do(ctx, TV_ACT_FLIP_V, 0);
        break;
    case '+':
    case '=':
        tv_do(ctx, TV_ACT_ZOOM, 1);
        break;
    case '-':
    case '_':
        tv_do(ctx, TV_ACT_ZOOM, -1);
        break;
    case '0':
        tv_do(ctx, TV_ACT_VIEW_RESET, 0);
        break;
    case TV_K_LEFT:
        tv_do(ctx, TV_ACT_PAN_X, -1);
        break;
    case TV_K_RIGHT:
        tv_do(ctx, TV_ACT_PAN_X, 1);
        break;
    case TV_K_UP:
    case TV_K_DOWN:
    {
        int step = (ev->key == TV_K_UP) ? -1 : 1;
        if (ctx->popup_type >= 1 && ctx->popup_type <= 3)
        {
            tv_do(ctx, popup_act[ctx->popup_type], step);
        }
        else
        {
            tv_do(ctx, TV_ACT_PAN_Y, step);
        }
        break;
    }
    case '<':
    case ',':
        tv_do(ctx, TV_ACT_FPS, -1);
        break;
    case '>':
    case '.':
        tv_do(ctx, TV_ACT_FPS, 1);
        break;
    default:
        break;
    }
}

/** tv_handle_mouse() - Pan (left drag), box zoom (right drag) and wheel zoom. */
void tv_handle_mouse(tv_state_t *ctx, const tv_event_t *ev)
{
    if (!tv_layout_ok(ctx))
    {
        ctx->mouse_is_dragging = 0;
        ctx->roi_is_dragging   = 0;
        return;
    }
    if (ev->button == TV_MB_WHEEL_UP || ev->button == TV_MB_WHEEL_DOWN)
    {
        tv_do(ctx, TV_ACT_ZOOM, (ev->button == TV_MB_WHEEL_UP) ? 1 : -1);
        return;
    }

    int    mx            = ev->x;
    int    my            = ev->y;
    double sign_x        = ctx->opt.flip_h ? -1.0 : 1.0;
    double sign_y        = ctx->opt.flip_v ? -1.0 : 1.0;
    double xsize         = ctx->st->xsize;
    double ysize         = ctx->st->ysize;
    int    disp_cols     = ctx->cols - TV_BAR_WIDTH - 1;
    int    disp_img_rows = (ctx->rows - TV_INFO_ROWS) * 2;
    double step = fmax(xsize / ctx->view.zoom / disp_cols, ysize / ctx->view.zoom / disp_img_rows);

    if (ev->button == TV_MB_LEFT)
    {
        if (ev->action == TV_MA_PRESS)
        {
            ctx->mouse_is_dragging = 1;
            ctx->last_mouse_x      = mx;
            ctx->last_mouse_y      = my;
        }
        else if (ev->action == TV_MA_DRAG && ctx->mouse_is_dragging)
        {
            /* grab: the image follows the pointer on both axes, whatever the flips */
            int dx = mx - ctx->last_mouse_x;
            int dy = my - ctx->last_mouse_y;
            tv_set_view(ctx, ctx->view.cx - sign_x * dx * step / xsize,
                        ctx->view.cy - sign_y * dy * 2.0 * step / ysize, ctx->view.zoom);
            ctx->last_mouse_x = mx;
            ctx->last_mouse_y = my;
        }
        else if (ev->action == TV_MA_RELEASE)
        {
            ctx->mouse_is_dragging = 0;
        }
    }
    else if (ev->button == TV_MB_RIGHT)
    {
        if (ev->action == TV_MA_PRESS)
        {
            ctx->roi_is_dragging = 1;
            ctx->roi_start_x     = mx;
            ctx->roi_start_y     = my;
            ctx->roi_end_x       = mx;
            ctx->roi_end_y       = my;
            ctx->dirty |= TV_D_CHROME;
        }
        else if (ev->action == TV_MA_DRAG && ctx->roi_is_dragging)
        {
            ctx->roi_end_x = mx;
            ctx->roi_end_y = my;
            ctx->dirty |= TV_D_CHROME;
        }
        else if (ev->action == TV_MA_RELEASE && ctx->roi_is_dragging)
        {
            ctx->roi_end_x       = mx;
            ctx->roi_end_y       = my;
            ctx->roi_is_dragging = 0;
            ctx->dirty |= TV_D_CHROME;

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

                tv_set_view(ctx, new_cx / xsize, new_cy / ysize,
                            ctx->view.zoom * (step / new_step));
            }
        }
    }
}

/** tv_handle_input_key() - Line editor for the manual min/max prompt; 1 if the key was consumed. */
static int tv_handle_input_key(tv_state_t *ctx, const tv_event_t *ev)
{
    uint32_t ch = ev->key;
    if (ev->mods & (TV_MOD_ALT | TV_MOD_CTRL))
    {
        return 0;
    }
    if (ch == '\n' || ch == '\r')
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
                tv_set_range(ctx, lo, hi);
                ctx->input_mode = 0;
            }
        }
    }
    else if (ch == TV_K_ESC)
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
    ctx->dirty |= TV_D_CHROME;
    return 1;
}

/** tv_flush_wheel() - Apply @notches of accumulated wheel motion as one zoom step. */
static void tv_flush_wheel(tv_state_t *ctx, int *notches)
{
    if (*notches != 0 && tv_layout_ok(ctx))
    {
        tv_do(ctx, TV_ACT_ZOOM, *notches);
    }
    *notches = 0;
}

/**
 * tv_dispatch_input() - Apply up to @budget decoded events. Consecutive drags of one button, and
 * consecutive wheel notches, merge into one update; any other event flushes them first, so the
 * order of effects is kept.
 */
void tv_dispatch_input(tv_state_t *ctx, tv_input_t *in, int64_t now_us, int budget)
{
    tv_event_t ev, drag;
    int        have_drag = 0;
    int        wheel     = 0;

    while (budget-- > 0 && ctx->running && tv_input_next(in, now_us, &ev))
    {
        int is_mouse = (ev.type == TV_EV_MOUSE);
        int is_wheel = is_mouse && (ev.button == TV_MB_WHEEL_UP || ev.button == TV_MB_WHEEL_DOWN);
        int is_drag  = is_mouse && ev.action == TV_MA_DRAG;

        if (have_drag && !(is_drag && drag.button == ev.button))
        {
            tv_handle_mouse(ctx, &drag);
            have_drag = 0;
        }
        if (!is_wheel)
        {
            tv_flush_wheel(ctx, &wheel);
        }
        if (is_drag)
        {
            drag      = ev;
            have_drag = 1;
        }
        else if (is_wheel)
        {
            wheel += (ev.button == TV_MB_WHEEL_UP) ? 1 : -1;
        }
        else if (is_mouse)
        {
            tv_handle_mouse(ctx, &ev);
        }
        else if (!(ctx->input_mode > 0 && tv_handle_input_key(ctx, &ev)))
        {
            tv_handle_key(ctx, &ev);
        }
    }
    if (have_drag)
    {
        tv_handle_mouse(ctx, &drag);
    }
    tv_flush_wheel(ctx, &wheel);
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
    if (tv_sig_cont)
    {
        return; /* stopped while rendering: the screen is not ours until re-entered */
    }
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
    ctx->dirty             = TV_D_FULL | TV_D_GEOM;
    ctx->lut_cmap          = -1;
    ctx->lut_scale         = -1;
    ctx->cur_max           = 1.0;
    ctx->target_fps        = (opt->fps >= 1 && opt->fps <= 120) ? opt->fps : 20;
    ctx->last_cnt0         = (uint64_t) -1;
    ctx->last_cnt0_for_fps = st.img.md->cnt0;
    int alloc_failed       = 0;

    ctx->stats_time_us = tv_now_us();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ctx->stats_cpu);

    if (tv_term_enter() != 0)
    {
        tv_term_restore();
        tv_stream_close(&st);
        free(ctx);
        fprintf(stderr, "Error: cannot configure the terminal\n");
        return 1;
    }
    tv_term_size(&ctx->rows, &ctx->cols);

    tv_input_t in;
    memset(&in, 0, sizeof(in));
    int64_t next_data      = tv_now_us();
    int64_t next_tick      = next_data + 1000000;
    int64_t last_render    = next_data - TV_INTERACTIVE_US;
    int     reenter_failed = 0;

    while (ctx->running && !tv_sig_quit && !tv_term_hup)
    {
        if (tv_sig_cont)
        {
            tv_sig_cont = 0;
            if (tv_term_enter() != 0)
            {
                ctx->exit_status = 1;
                reenter_failed   = 1;
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

        /* sleep until the earliest of: next data poll, frame cap, ESC timeout, tick, popup */
        int64_t now    = tv_now_us();
        int64_t period = 1000000 / ctx->target_fps;
        if (next_data > now + period)
        {
            next_data = now + period; /* the rate was raised: do not wait out the old period */
        }
        int64_t dl = next_data;
        if (ctx->dirty && last_render + TV_INTERACTIVE_US < dl)
        {
            dl = last_render + TV_INTERACTIVE_US;
        }
        int64_t esc_dl = tv_input_deadline(&in);
        if (esc_dl >= 0 && esc_dl < dl)
        {
            dl = esc_dl;
        }
        if (next_tick < dl)
        {
            dl = next_tick;
        }
        if (ctx->popup_type > 0 && ctx->popup_expiry_us < dl)
        {
            dl = ctx->popup_expiry_us;
        }

        struct pollfd pfd        = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
        int           timeout_ms = (dl <= now) ? 0 : (int) ((dl - now + 999) / 1000);
        int           ready      = poll(&pfd, 1, timeout_ms);
        if (ready < 0 && errno != EINTR)
        {
            ctx->exit_status = 1;
            break;
        }
        now = tv_now_us();
        if (ready > 0)
        {
            if ((pfd.revents & POLLIN) && tv_input_room(&in) > 0)
            {
                unsigned char buf[sizeof(in.buf)];
                ssize_t       n = read(STDIN_FILENO, buf, tv_input_room(&in));
                if (n > 0)
                {
                    tv_input_feed(&in, buf, (size_t) n, now);
                }
                else if (n == 0 || (errno != EINTR && errno != EAGAIN))
                {
                    ctx->exit_status = 1; /* terminal closed */
                    break;
                }
            }
            else if (!(pfd.revents & POLLIN) && (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)))
            {
                ctx->exit_status = 1;
                break;
            }
        }

        tv_dispatch_input(ctx, &in, now, 512);
        if (!ctx->running || tv_sig_quit || tv_term_hup)
        {
            break;
        }
        if (tv_sig_cont)
        {
            continue; /* resumed: restore the terminal before drawing anything */
        }

        if (now >= next_tick)
        {
            tv_update_stats(ctx, now);
            next_tick = (next_tick + 1000000 > now) ? next_tick + 1000000 : now + 1000000;
            ctx->dirty |= TV_D_CHROME;
        }
        if (ctx->popup_type > 0 && now >= ctx->popup_expiry_us)
        {
            ctx->popup_type = 0;
            ctx->dirty |= TV_D_CHROME;
        }
        if (now >= next_data)
        {
            next_data = (next_data + period > now) ? next_data + period : now + period;
            if (st.img.md->cnt0 != ctx->last_cnt0)
            {
                ctx->dirty |= TV_D_DATA;
            }
        }

        /* stream frames follow the data poll; input-driven frames are capped */
        if (!(ctx->dirty & TV_D_DATA) && !(ctx->dirty && now - last_render >= TV_INTERACTIVE_US))
        {
            continue;
        }
        last_render    = now;
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
    if (reenter_failed)
    {
        fprintf(stderr, "Error: cannot restore the viewer's terminal mode after resuming\n");
        return 1;
    }
    if (tv_term_hup)
    {
        return 1;
    }
    return status;
}
