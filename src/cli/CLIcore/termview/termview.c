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


#define TV_STALE_MIN_US 1000000 /* no new frame for this long (or 5 periods): STALE */

/* A slow terminal gets per flush what it drains in one frame period, within these bounds */
#define TV_FLUSH_SLICE_MIN_US 40000
#define TV_FLUSH_SLICE_MAX_US 50000
#define TV_FLUSH_BLOCKED_US 2000      /* a write that took longer met a terminal slower than us */
#define TV_FLUSH_MIN_BYTES 16384      /* the budget is never less than this */
#define TV_FLUSH_MAX_BYTES (4u << 20) /* above this the budget is dropped */

/* -----------------------------------------------------------------------
 * Main Screen Loop
 * ----------------------------------------------------------------------- */

/** tv_cnt0() - The stream's frame counter; the last value seen once the mapping is gone. */
static inline uint64_t tv_cnt0(const tv_state_t *ctx)
{
    return (ctx->st->img.md != NULL) ? ctx->st->img.md->cnt0 : ctx->last_cnt0;
}

/** tv_mapped() - The mapping can be read now; one just dropped is looked into at once. */
static int tv_mapped(tv_state_t *ctx)
{
    int had = (ctx->st->map != NULL);
    if (tv_stream_safe(ctx->st))
    {
        return 1;
    }
    ctx->recheck |= had;
    return 0;
}

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

    uint64_t cnt0 = tv_cnt0(ctx);
    /* a counter that went backwards means the stream was reset: re-baseline */
    ctx->current_stream_fps = (cnt0 >= ctx->last_cnt0_for_fps)
                                  ? (double) (cnt0 - ctx->last_cnt0_for_fps) / real_diff
                                  : 0.0;
    if (ctx->current_stream_fps > 0.0)
    {
        ctx->rate_hz = ctx->current_stream_fps;
    }
    ctx->last_cnt0_for_fps = cnt0;
    ctx->ui_fps            = ctx->flushes / real_diff;
    ctx->flushes           = 0;
    ctx->stats_time_us     = now_us;
    ctx->stats_cpu         = now_cpu;
}

int64_t tv_stale_age_us(const tv_state_t *ctx, int64_t now_us)
{
    int64_t age   = now_us - ctx->last_change_us;
    int64_t limit = TV_STALE_MIN_US;
    if (5.0 * ctx->period_us > (double) limit)
    {
        limit = (int64_t) (5.0 * ctx->period_us);
    }
    return (age > limit) ? age : 0;
}

/** tv_seen_at() - When the stream was last written (tv_now_us() time), from its write time. */
static int64_t tv_seen_at(const IMAGE_METADATA *md, int64_t now_us)
{
    struct timespec wall;
    clock_gettime(CLOCK_ISIO, &wall);
    double age = (double) (wall.tv_sec - md->writetime.tv_sec) +
                 (double) (wall.tv_nsec - md->writetime.tv_nsec) / 1e9;
    if (md->writetime.tv_sec > 0 && tv_finite(age) && age > 0.0 && age < 1e9)
    {
        return now_us - (int64_t) (age * 1e6);
    }
    return now_us; /* not set, or from another clock */
}

/** tv_note_cnt0() - Record a cnt0 value seen at @now_us; true if it is a new frame. */
static int tv_note_cnt0(tv_state_t *ctx, uint64_t cnt0, int64_t now_us)
{
    if (cnt0 == ctx->last_cnt0)
    {
        return 0;
    }
    int64_t seen = now_us;
    if (cnt0 > ctx->last_cnt0 && ctx->period_us >= 0.0)
    {
        /* frame interval, smoothed: sets how long the stream may be quiet before STALE */
        double obs     = (double) (now_us - ctx->last_change_us) / (double) (cnt0 - ctx->last_cnt0);
        ctx->period_us = (ctx->period_us > 0.0) ? 0.75 * ctx->period_us + 0.25 * obs : obs;
    }
    else if (ctx->period_us < 0.0)
    {
        /* first change seen (or after a pause): not ours to time, but the frame says its age */
        ctx->period_us = 0.0;
        seen           = tv_seen_at(ctx->st->img.md, now_us);
    }
    ctx->last_cnt0      = cnt0;
    ctx->last_change_us = seen;
    return 1;
}

/** tv_quiet_us() - Rest time: 1.5 frame periods, 1 s while the period is unknown, at most 5 s. */
static double tv_quiet_us(const tv_state_t *ctx)
{
    double q = (ctx->period_us > 0.0) ? 1.5 * ctx->period_us : 1e6;
    return (q < 5e6) ? q : 5e6;
}

/** tv_layout_ok() - True when the image area is drawable; also guards every geometry divisor. */
static int tv_layout_ok(const tv_state_t *ctx)
{
    tv_layout_t l;
    tv_layout_compute(&l, ctx->rows, ctx->cols);
    return l.valid;
}

/** tv_xform_now() - The current view transform, as the renderer uses it. */
static void tv_xform_now(const tv_state_t *ctx, tv_layout_t *l, tv_xform_t *x)
{
    tv_layout_compute(l, ctx->rows, ctx->cols);
    tv_xform_compute(x, l, &ctx->view, ctx->st->xsize, ctx->st->ysize, ctx->opt.flip_h,
                     ctx->opt.flip_v);
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
    ctx->ui.hover_sent = -1; /* the terminal's mouse mode is unknown too */
    ctx->dirty |= TV_D_FULL | TV_D_GEOM;
}

/**
 * tv_pause() - Freeze the image on a copy of the displayed slice (so zoom and pan keep working on
 * it), or go back to the live stream. Without memory for the copy, the pause is refused (bell).
 */
static void tv_pause(tv_state_t *ctx, int on)
{
    tv_stream_t *st = ctx->st;
    free(ctx->frozen);
    ctx->frozen = NULL;
    ctx->paused = on;
    if (on && tv_mapped(ctx))
    {
        size_t n = (size_t) st->xsize * st->ysize * (size_t) ImageStreamIO_typesize(st->datatype);
        ctx->frozen = malloc(n);
        if (ctx->frozen == NULL)
        {
            ctx->paused = 0; /* no memory for the copy: a pause would show live data; refuse */
            tv_write_all("\a", 1);
            return;
        }
        memcpy(ctx->frozen, st->img.array.raw, n);
    }
    if (!on)
    {
        ctx->recheck   = 1;    /* pick up a stream that was replaced meanwhile */
        ctx->period_us = -1.0; /* the pause is no frame interval */
    }
    /* sample what is frozen (or live again) and draw it exactly */
    ctx->dirty |= TV_D_DATA | TV_D_MAP | TV_D_CHROME;
}

/** tv_zmax() - Largest zoom for the current layout (a sample step of 1/64 pixel). */
static double tv_zmax(const tv_state_t *ctx)
{
    if (!tv_layout_ok(ctx))
    {
        return 1.0;
    }
    tv_layout_t l;
    tv_layout_compute(&l, ctx->rows, ctx->cols);
    return fmax(1.0, 64.0 * tv_view_base(&l, ctx->st->xsize, ctx->st->ysize));
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
    case TV_ACT_PAUSE:
        tv_pause(ctx, !ctx->paused);
        break;
    case TV_ACT_CMAP:
        tv_do(ctx, TV_ACT_SET_CMAP, tv_wrap(ctx->opt.colormap, arg, COLORMAP_NB));
        break;
    case TV_ACT_SCALE:
        tv_do(ctx, TV_ACT_SET_SCALE, tv_wrap(ctx->opt.scale, arg, SCALE_NB));
        break;
    case TV_ACT_RANGE:
        tv_do(ctx, TV_ACT_SET_RANGE, tv_wrap(ctx->opt.range, arg, RANGE_NB));
        break;
    case TV_ACT_SET_CMAP:
        ctx->opt.colormap = (termview_colormap_t) tv_wrap(arg, 0, COLORMAP_NB);
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_SET_SCALE:
        ctx->opt.scale = (termview_scale_t) tv_wrap(arg, 0, SCALE_NB);
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_SET_RANGE: /* choosing a range mode is asking for an automatic range */
        ctx->opt.range        = (termview_range_t) tv_wrap(arg, 0, RANGE_NB);
        ctx->opt.range_locked = false;
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_SET_COLORS:
        if (arg >= 0 && arg <= 3 && arg != ctx->opt.colors)
        {
            ctx->opt.colors = arg;
            tv_emit_init(&ctx->emit, arg ? arg : tv_term_color_level());
            tv_invalidate(ctx); /* every cell again, in the new codes */
        }
        ctx->dirty |= TV_D_CHROME;
        break;
    case TV_ACT_SET_FPS:
        ctx->target_fps = (arg < 1) ? 1 : (arg > 120) ? 120 : arg;
        ctx->dirty |= TV_D_CHROME;
        break;
    case TV_ACT_SMOOTH:
        ctx->opt.range_smooth = !ctx->opt.range_smooth;
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    case TV_ACT_CENTRE:
        tv_set_view(ctx, 0.5, 0.5, ctx->view.zoom);
        break;
    case TV_ACT_HELP:
    {
        int on = !ctx->ui.help;
        tv_ui_open(ctx, -1); /* a menu, or the range editor, gives way to the help box */
        ctx->ui.help     = on;
        ctx->ui.help_top = 0;
        ctx->mini.active = ctx->mini.active && !on;
        ctx->dirty |= TV_D_CHROME;
        break;
    }
    case TV_ACT_LOCK:
    {
        int lock = (arg < 0) ? !ctx->opt.range_locked : (arg != 0);
        if (lock && !ctx->opt.range_locked)
        {
            ctx->opt.manual_min = ctx->cur_min; /* the range shown now */
            ctx->opt.manual_max = ctx->cur_max;
        }
        ctx->opt.range_locked = lock;
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
        break;
    }
    case TV_ACT_PROMPT:
        tv_mini_open(ctx, (arg == 2) ? 1 : 0);
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
        tv_do(ctx, TV_ACT_SET_FPS, ctx->target_fps + arg);
        break;
    case TV_ACT_ESCAPE:
        if (ctx->mouse_is_dragging || ctx->roi_is_dragging)
        {
            ctx->mouse_is_dragging = 0;
            ctx->roi_is_dragging   = 0;
            ctx->dirty |= TV_D_CHROME;
        }
        break;
    case TV_ACT_LOWBW:
        ctx->opt.lowbw = !ctx->opt.lowbw;
        ctx->dirty |= TV_D_MAP | TV_D_CHROME;
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
    case 'c':
    case 'C':
        tv_do(ctx, TV_ACT_CMAP, (ev->key == 'c') ? 1 : -1);
        break;
    case 's':
    case 'S':
        tv_do(ctx, TV_ACT_SCALE, (ev->key == 's') ? 1 : -1);
        break;
    case 'r':
    case 'R':
        tv_do(ctx, TV_ACT_RANGE, (ev->key == 'r') ? 1 : -1);
        break;
    case 'l':
        tv_do(ctx, TV_ACT_LOCK, -1);
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
    case 'b':
        tv_do(ctx, TV_ACT_LOWBW, 0);
        break;
    case ' ':
        tv_do(ctx, TV_ACT_PAUSE, 0);
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
        tv_do(ctx, TV_ACT_PAN_Y, (ev->key == TV_K_UP) ? -1 : 1);
        break;
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

/**
 * tv_zoom_at() - Zoom by 1.2^@notches keeping the image point under cell (@x, @y) in place;
 * about the view centre when the pointer is outside the image.
 */
static void tv_zoom_at(tv_state_t *ctx, int notches, int x, int y)
{
    tv_layout_t l;
    tv_xform_t  xf;
    tv_xform_now(ctx, &l, &xf);
    y -= l.img_top;                /* screen row to image row, both 1-based */
    double dj = x - 0.5;           /* centre of column x (1-based) in samples */
    double di = 2.0 * (y - 1) + 1; /* between the cell's two sample rows */
    if (x < 1 || x > l.img_cols || y < 1 || y > l.img_rows)
    {
        tv_do(ctx, TV_ACT_ZOOM, notches);
        return;
    }
    double px   = tv_xf_img_x(&xf, dj);
    double py   = tv_xf_img_y(&xf, di);
    double zoom = ctx->view.zoom * pow(1.2, notches);
    double zmax = tv_zmax(ctx);
    zoom        = (zoom > zmax) ? zmax : (zoom < 0.1) ? 0.1 : zoom;
    double step = tv_view_base(&l, ctx->st->xsize, ctx->st->ysize) / zoom;
    double cx   = px - xf.sx * (dj - xf.cdx) * step;
    double cy   = py - xf.sy * (di - xf.cdy) * step;
    tv_set_view(ctx, cx / ctx->st->xsize, cy / ctx->st->ysize, zoom);
}

/** tv_handle_mouse() - Pan (left drag), box zoom (right drag) and wheel zoom at the pointer. */
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
        tv_zoom_at(ctx, (ev->button == TV_MB_WHEEL_UP) ? 1 : -1, ev->x, ev->y);
        return;
    }

    tv_layout_t l;
    tv_xform_t  xf;
    tv_xform_now(ctx, &l, &xf);
    int    mx    = ev->x;
    int    my    = ev->y - l.img_top; /* image rows, 1-based */
    double xsize = ctx->st->xsize;
    /* a box zoom selection stays inside the image area, so outline and result agree */
    int    bx    = (mx < 1) ? 1 : (mx > l.img_cols) ? l.img_cols : mx;
    int    by    = (my < 1) ? 1 : (my > l.img_rows) ? l.img_rows : my;
    double ysize = ctx->st->ysize;

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
            /* grab: the image point under the pointer follows it, whatever the flips */
            int dx = mx - ctx->last_mouse_x;
            int dy = my - ctx->last_mouse_y;
            tv_set_view(ctx, (xf.cx - xf.sx * dx * xf.step) / xsize,
                        (xf.cy - xf.sy * 2.0 * dy * xf.step) / ysize, ctx->view.zoom);
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
            ctx->roi_start_x     = bx;
            ctx->roi_start_y     = by;
            ctx->roi_end_x       = bx;
            ctx->roi_end_y       = by;
            ctx->dirty |= TV_D_CHROME;
        }
        else if (ev->action == TV_MA_DRAG && ctx->roi_is_dragging)
        {
            ctx->roi_end_x = bx;
            ctx->roi_end_y = by;
            ctx->dirty |= TV_D_CHROME;
        }
        else if (ev->action == TV_MA_RELEASE && ctx->roi_is_dragging)
        {
            ctx->roi_end_x       = bx;
            ctx->roi_end_y       = by;
            ctx->roi_is_dragging = 0;
            ctx->dirty |= TV_D_CHROME;

            int min_x = (ctx->roi_start_x < ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
            int max_x = (ctx->roi_start_x > ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
            int min_y = (ctx->roi_start_y < ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
            int max_y = (ctx->roi_start_y > ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;

            if (max_x - min_x >= 2 && max_y - min_y >= 2)
            {
                /* box edges in samples, mapped through the same transform as the image */
                double x0 = tv_xf_img_x(&xf, min_x - 1.0), x1 = tv_xf_img_x(&xf, max_x);
                double y0 = tv_xf_img_y(&xf, 2.0 * (min_y - 1)), y1 = tv_xf_img_y(&xf, 2.0 * max_y);
                double step = fmax(fabs(x1 - x0) / l.img_cols, fabs(y1 - y0) / (2.0 * l.img_rows));
                double base = tv_view_base(&l, ctx->st->xsize, ctx->st->ysize);
                tv_set_view(ctx, 0.5 * (x0 + x1) / xsize, 0.5 * (y0 + y1) / ysize, base / step);
            }
        }
    }
}

/** tv_flush_wheel() - Apply @notches of accumulated wheel motion as one zoom at the pointer. */
static void tv_flush_wheel(tv_state_t *ctx, int *notches)
{
    if (*notches != 0 && tv_layout_ok(ctx))
    {
        tv_zoom_at(ctx, *notches, ctx->wheel_x, ctx->wheel_y);
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
        if (is_mouse && tv_ui_wants_mouse(ctx, &ev))
        {
            tv_flush_wheel(ctx, &wheel);
            tv_ui_mouse(ctx, &ev); /* menus, the help box, the menubar and the status row */
        }
        else if (is_drag)
        {
            drag      = ev;
            have_drag = 1;
        }
        else if (is_wheel)
        {
            wheel += (ev.button == TV_MB_WHEEL_UP) ? 1 : -1;
            ctx->wheel_x = ev.x;
            ctx->wheel_y = ev.y;
        }
        else if (is_mouse)
        {
            tv_handle_mouse(ctx, &ev);
        }
        else if (!tv_ui_key(ctx, &ev))
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

static void termview_flush_ansi(tv_state_t *ctx)
{
    if (tv_sig_cont)
    {
        return; /* stopped while rendering: the screen is not ours until re-entered */
    }
    /* pointer motion reports only while a menu or the help box can use them */
    int hover = ctx->ui.open || ctx->ui.help;
    if (hover != ctx->ui.hover_sent)
    {
        ctx->ui.hover_sent = hover;
        /* in xterm and tmux, resetting 1003 ends mouse reporting: back to button events */
        tv_write_all(hover ? "\033[?1003h" : "\033[?1003l\033[?1002h", hover ? 8 : 16);
    }
    size_t n = tv_flush_diff(&ctx->emit, ctx->frame_buffer, ctx->screen, ctx->prev_screen,
                             ctx->rows, ctx->cols);
    if (n == 0)
    {
        return;
    }
    ctx->flushes++;
    int64_t t0 = tv_now_us();
    if (tv_write_all(ctx->frame_buffer, n) != 0)
    {
        tv_invalidate(ctx); /* partial frame on screen: repaint everything next time */
    }
    int64_t dt = tv_now_us() - t0;

    /* Image bytes per flush. Once a write blocks, a flush carries what the terminal drains in
     * a frame period, so a key never waits longer behind image bytes. A large write that goes
     * through at once means the terminal caught up: the budget doubles until it is dropped. */
    tv_emit_t *e = &ctx->emit;
    if (dt > TV_FLUSH_BLOCKED_US)
    {
        /* a slower link is believed at once, a faster one halfway: a short write that kernel
         * buffers absorbed overstates the rate */
        double rate     = (double) n / (double) dt;
        ctx->drain_rate = (rate > ctx->drain_rate && ctx->drain_rate > 0.0)
                              ? 0.5 * (ctx->drain_rate + rate)
                              : rate;
        int64_t slice   = 1000000 / ctx->target_fps;
        slice           = (slice < TV_FLUSH_SLICE_MIN_US)   ? TV_FLUSH_SLICE_MIN_US
                          : (slice > TV_FLUSH_SLICE_MAX_US) ? TV_FLUSH_SLICE_MAX_US
                                                            : slice;
        double b        = ctx->drain_rate * (double) slice;
        b               = (b > 2.0 * (double) n) ? 2.0 * (double) n : b; /* same reason */
        e->budget       = (b < TV_FLUSH_MIN_BYTES) ? TV_FLUSH_MIN_BYTES : (size_t) b;
    }
    else if (e->budget > 0 && n >= e->budget / 2)
    {
        e->budget = (e->budget > TV_FLUSH_MAX_BYTES / 2) ? 0 : 2 * e->budget;
        if (e->budget == 0)
        {
            ctx->drain_rate = 0.0;
        }
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

/** tv_grow() - Make sure the screen grids and the frame buffer fit the terminal. */
static int tv_grow(tv_state_t *ctx)
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

/** tv_dim() - Half-brightness version of a packed colour. */
static inline uint32_t tv_dim(uint32_t c)
{
    return (c >> 1) & 0x7F7F7F;
}

/** tv_draw_roi() - Box-zoom selection: yellow outline (interface), dimmed inside. */
static void tv_draw_roi(tv_state_t *ctx)
{
    const tv_layout_t *l  = &ctx->lay;
    int                x0 = (ctx->roi_start_x < ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
    int                x1 = (ctx->roi_start_x > ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
    int                y0 = (ctx->roi_start_y < ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
    int                y1 = (ctx->roi_start_y > ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
    uint32_t           yellow = TV_RGB(255, 255, 0);
    for (int r = y0; r <= y1 && r <= l->img_rows; r++)
    {
        for (int c = x0; c <= x1 && c <= l->img_cols; c++)
        {
            tv_cell_t *cell = &ctx->screen[(l->img_top + r - 1) * ctx->cols + (c - 1)];
            if (r == y0 || r == y1 || c == x0 || c == x1)
            {
                *cell = tv_cell(yellow, yellow, TV_G_LOWER_HALF) | TV_ATTR_CHROME;
            }
            else
            {
                /* still image cells: on a slow link the dimming follows the byte budget */
                *cell = tv_cell(tv_dim(tv_cell_bg(*cell)), tv_dim(tv_cell_fg(*cell)),
                                tv_cell_glyph(*cell));
            }
        }
    }
}

/** tv_compose() - Screen = cached image layer + colorbar + info bar + overlays. */
static void tv_compose(tv_state_t *ctx, int64_t now_us)
{
    const tv_layout_t *l     = &ctx->lay;
    tv_cell_t          blank = tv_cell(TV_RGB(0, 0, 0), TV_RGB(255, 255, 255), ' ');
    for (int i = 0; i < ctx->rows * ctx->cols; i++)
    {
        ctx->screen[i] = blank;
    }
    for (int r = 0; r < l->img_rows; r++)
    {
        memcpy(&ctx->screen[(l->img_top + r) * ctx->cols], &ctx->im.cells[(size_t) r * l->img_cols],
               (size_t) l->img_cols * sizeof(tv_cell_t));
    }
    if (ctx->roi_is_dragging)
    {
        tv_draw_roi(ctx);
    }
    if (ctx->im.blank && l->img_rows > 0)
    {
        static const char msg[] = "stream gone: waiting for it to come back";
        int               len   = (int) sizeof(msg) - 1;
        int               col   = (len < l->img_cols) ? (l->img_cols - len) / 2 : 0;
        int               row   = l->img_top + l->img_rows / 2;
        tv_draw_text(ctx, row, col, msg);
        for (int k = 0; k < len && col + k < l->img_cols; k++)
        {
            ctx->screen[row * ctx->cols + col + k] |= TV_ATTR_CHROME;
        }
    }
    tv_ui_render(ctx, now_us); /* marks its own cells, overlays on the image included */

    /* everything outside the image area is interface: sent ahead of image cells */
    for (int r = 0; r < ctx->rows; r++)
    {
        int in = (r >= l->img_top && r < l->img_top + l->img_rows);
        for (int c = in ? l->img_cols : 0; c < ctx->cols; c++)
        {
            ctx->screen[r * ctx->cols + c] |= TV_ATTR_CHROME;
        }
    }
}

/**
 * tv_stream_monitor() - Notice a stream that was truncated, removed or recreated, and switch to
 * the stream now at its path once that is readable. A new geometry resets the view; colormap,
 * scale and range are kept. Paused, the switch waits for the resume.
 */
static void tv_stream_monitor(tv_state_t *ctx, int64_t now_us)
{
    tv_stream_t *st       = ctx->st;
    int          was_gone = ctx->gone;
    ctx->gone             = (tv_stream_check(st) != TV_STREAM_OK);
    if (ctx->gone && !ctx->paused)
    {
        uint32_t xsize = st->xsize, ysize = st->ysize;
        if (tv_stream_reopen(st) == 0)
        {
            ctx->gone = 0;
            if (st->xsize != xsize || st->ysize != ysize)
            {
                ctx->view.zoom = 1.0;
                ctx->view.cx   = 0.5;
                ctx->view.cy   = 0.5;
            }
            ctx->last_cnt0         = st->img.md->cnt0;
            ctx->last_cnt0_for_fps = st->img.md->cnt0;
            ctx->period_us         = -1.0;
            ctx->last_change_us    = tv_seen_at(st->img.md, now_us);
            ctx->im.have_range     = 0; /* other values: no smoothing from the old ones */
            ctx->dirty |= TV_D_GEOM | TV_D_DATA | TV_D_CHROME;
        }
    }
    if (ctx->gone != was_gone)
    {
        ctx->dirty |= TV_D_CHROME;
    }
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
    opt->range_smooth = true;
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
    ctx->last_cnt0         = st.img.md->cnt0; /* the first frame is drawn by the TV_D_FULL above */
    ctx->period_us         = -1.0;            /* no frame interval observed yet */
    ctx->last_cnt0_for_fps = st.img.md->cnt0;
    int alloc_failed       = 0;

    ctx->stats_time_us = tv_now_us();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ctx->stats_cpu);
    tv_emit_init(&ctx->emit,
                 (opt->colors >= 1 && opt->colors <= 3) ? opt->colors : tv_term_color_level());

    ctx->last_change_us = tv_seen_at(st.img.md, ctx->stats_time_us); /* the newest frame's age */

    if (tv_term_enter() != 0)
    {
        tv_term_restore();
        tv_stream_close(&st);
        free(ctx);
        fprintf(stderr, "Error: cannot configure the terminal\n");
        return 1;
    }
    ImageStreamIO_set_printError(NULL); /* reopening a half-written stream must stay quiet */
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

        /* sleep until the earliest of: next data poll, frame cap (dirty, or image cells left
         * over by the last flush), ESC timeout, tick */
        int64_t now    = tv_now_us();
        int64_t period = 1000000 / ctx->target_fps;
        if (next_data > now + period)
        {
            next_data = now + period; /* the rate was raised: do not wait out the old period */
        }
        int64_t dl = ctx->paused ? next_tick : next_data; /* paused: about one wakeup a second */
        if ((ctx->dirty || ctx->emit.pending) && last_render + TV_INTERACTIVE_US < dl)
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

        if (now >= next_tick || ctx->recheck)
        {
            ctx->recheck = 0;
            tv_stream_monitor(ctx, now);
        }
        if (now >= next_tick)
        {
            tv_update_stats(ctx, now);
            next_tick = (next_tick + 1000000 > now) ? next_tick + 1000000 : now + 1000000;
            ctx->dirty |= TV_D_CHROME;
        }

        if (ctx->emit.pending)
        {
            ctx->dirty |= TV_D_CHROME; /* send the image cells the last flush had no room for */
        }
        if (now >= next_data && !ctx->paused)
        {
            next_data = (next_data + period > now) ? next_data + period : now + period;
            if (tv_mapped(ctx) && tv_note_cnt0(ctx, tv_cnt0(ctx), now))
            {
                ctx->dirty |= TV_D_DATA;
            }
            else if (ctx->im.held &&
                     (ctx->resting || (double) (now - ctx->last_change_us) > tv_quiet_us(ctx)))
            {
                ctx->dirty |= TV_D_HELD; /* the stream rests: redraw the held rows exactly */
            }
        }

        /* stream frames follow the data poll; input-driven frames are capped */
        if (!(ctx->dirty & TV_D_DATA) && !(ctx->dirty && now - last_render >= TV_INTERACTIVE_US))
        {
            continue;
        }
        last_render = now;
        /* a user change shows a frame that arrived since the poll; other renders (the tick, a
         * flush continuation) leave new frames to the data poll and its rate */
        if ((ctx->dirty & (TV_D_GEOM | TV_D_MAP | TV_D_FULL)) && !ctx->paused && tv_mapped(ctx) &&
            tv_note_cnt0(ctx, tv_cnt0(ctx), now))
        {
            ctx->dirty |= TV_D_DATA;
        }
        int dirty  = ctx->dirty;
        ctx->dirty = 0;

        tv_layout_t lay;
        tv_layout_compute(&lay, ctx->rows, ctx->cols);
        if (lay.rows != ctx->lay.rows || lay.cols != ctx->lay.cols)
        {
            dirty |= TV_D_GEOM;
        }
        ctx->lay = lay;
        if (tv_grow(ctx) != 0)
        {
            alloc_failed = 1;
            break;
        }
        if (!lay.valid)
        {
            tv_cell_t blank = tv_cell(TV_RGB(0, 0, 0), TV_RGB(255, 255, 255), ' ');
            for (int i = 0; i < ctx->rows * ctx->cols; i++)
            {
                ctx->screen[i] = blank;
            }
            char msg[64];
            snprintf(msg, sizeof(msg), "terminal too small (%dx%d)", ctx->cols, ctx->rows);
            int len = (int) strlen(msg);
            tv_draw_text(ctx, ctx->rows / 2, (len < ctx->cols) ? (ctx->cols - len) / 2 : 0, msg);
            for (int i = 0; i < ctx->rows * ctx->cols; i++)
            {
                ctx->screen[i] |= TV_ATTR_CHROME;
            }
            ctx->mouse_is_dragging = 0;
            ctx->roi_is_dragging   = 0;
            if (ctx->ui.open || ctx->ui.help)
            {
                tv_ui_open(ctx, -1); /* menus and help do not survive off screen */
            }
            ctx->mini.active   = 0;
            ctx->ui.nhits      = 0;
            ctx->im.have_range = 0;
            ctx->im.held       = 0;
            ctx->lay.rows      = 0; /* rebuild the tables once the size is usable again */
            termview_flush_ansi(ctx);
            continue;
        }

        /* geometry -> tables; data or geometry -> samples; anything but chrome -> recolour */
        int sampled = 0;
        if (dirty & (TV_D_GEOM | TV_D_FULL))
        {
            tv_view_sanitize(&ctx->view.zoom, &ctx->view.cx, &ctx->view.cy, tv_zmax(ctx));
            tv_xform_t xf;
            tv_xform_compute(&xf, &lay, &ctx->view, st.xsize, st.ysize, ctx->opt.flip_h,
                             ctx->opt.flip_v);
            if (tv_image_geom(&ctx->im, &lay, &xf, st.xsize, st.ysize) != 0)
            {
                alloc_failed = 1;
                break;
            }
        }
        /* the frozen copy while paused, else the mapping; nothing once the mapping is gone */
        const void *src = (ctx->paused && ctx->frozen) ? ctx->frozen
                          : tv_mapped(ctx)             ? st.img.array.raw
                                                       : NULL;
        if (src == NULL)
        {
            ctx->im.held = 0; /* no samples: no held rows to refresh */
            if (dirty & (TV_D_GEOM | TV_D_FULL))
            {
                tv_image_blank(&ctx->im); /* new geometry and no data to sample: empty image */
            }
            tv_compose(ctx, now);
            termview_flush_ansi(ctx);
            continue;
        }
        if (dirty & (TV_D_GEOM | TV_D_FULL | TV_D_DATA))
        {
            sampled      = tv_image_sample(&ctx->im, src, st.datatype);
            ctx->resting = !sampled && (dirty & TV_D_DATA);
            if (!sampled && ctx->im.held)
            {
                dirty |= TV_D_HELD; /* same samples: go on refreshing the held rows */
            }
            else if (!sampled && dirty == TV_D_DATA && ctx->im.settled)
            {
                continue; /* new frame, same picture: nothing to do */
            }
        }
        /* a range still converging finishes on unchanged samples (and on the 1 Hz tick) */
        int user = dirty & (TV_D_GEOM | TV_D_MAP | TV_D_FULL);
        if (sampled || user || (dirty & TV_D_HELD) || !ctx->im.settled)
        {
            if ((int) ctx->opt.colormap != ctx->lut_cmap || (int) ctx->opt.scale != ctx->lut_scale)
            {
                tv_lut_build(ctx->lut, ctx->opt.colormap, ctx->opt.scale);
                ctx->lut_cmap  = ctx->opt.colormap;
                ctx->lut_scale = ctx->opt.scale;
            }
            /* a user change shows its result at once; only stream updates are smoothed, and
             * redrawing held rows leaves the range alone */
            if (sampled || user || !ctx->im.settled)
            {
                tv_image_range(&ctx->im, &ctx->opt, !ctx->im.have_range || !sampled || user);
                ctx->cur_min = ctx->im.lo;
                ctx->cur_max = ctx->im.hi;
            }
            /* small changes are held only on stream updates, never on the user's own changes */
            tv_image_render(&ctx->im, ctx->lut, ctx->opt.lowbw && !user);
        }
        tv_compose(ctx, now);
        termview_flush_ansi(ctx);
    }

    tv_term_restore();
    ImageStreamIO_set_default_printError();

    tv_image_free(&ctx->im);
    free(ctx->frozen);
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
