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
        ctx->suspend = 1; /* once the outstanding answers are read: see tv_fc_settle() */
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

    while (budget-- > 0 && ctx->running && !ctx->suspend && tv_input_next(in, now_us, &ev))
    {
        if (ev.type == TV_EV_CPR)
        {
            tv_fc_answer(ctx, now_us, ev.x); /* not input: the terminal showed what came before */
            continue;
        }
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

/** tv_slice_us() - A frame period within 40-50 ms: what a slow terminal gets per flush. */
static int64_t tv_slice_us(const tv_state_t *ctx)
{
    int64_t slice = 1000000 / ctx->target_fps;
    return (slice < TV_FLUSH_SLICE_MIN_US)   ? TV_FLUSH_SLICE_MIN_US
           : (slice > TV_FLUSH_SLICE_MAX_US) ? TV_FLUSH_SLICE_MAX_US
                                             : slice;
}

static inline int64_t tv_max64(int64_t a, int64_t b)
{
    return (a > b) ? a : b;
}

/** tv_fc_quiet() - Unanswered bytes an answer may come behind and still time the round trip:
 * what the link shows in 5 ms, TV_FC_QUIET at least. */
static uint64_t tv_fc_quiet(const tv_fc_t *fc)
{
    double q = fc->rate * 5000.0;
    return (q > TV_FC_QUIET) ? (uint64_t) q : TV_FC_QUIET;
}

void tv_fc_window(tv_fc_t           *fc,
                  int64_t            now_us,
                  const tv_fc_req_t *r,
                  uint64_t           delta,
                  int64_t            slice_us)
{
    int64_t rtt     = now_us - r->us;
    int64_t horizon = tv_max64(3000000, 4 * (fc->base_us + slice_us));
    if (r->quiet)
    {
        /* nothing queued ahead: the plain round trip. While it holds, time it less often. If it
         * grew by as much as every queue seen since a cut, that queue was the old round trip's
         * doing: the rate comes back. */
        int64_t d = (rtt > fc->base_us) ? rtt - fc->base_us : fc->base_us - rtt;
        fc->gap_us =
            (d <= tv_max64(2000, fc->base_us / 8))
                ? ((2 * fc->gap_us < TV_FC_REBASE_MAX_US) ? 2 * fc->gap_us : TV_FC_REBASE_MAX_US)
                : TV_FC_REBASE_US;
        if (rtt > fc->base_us + 2 * slice_us && fc->cut_rate > fc->rate &&
            fc->cut_rtt_us <= rtt + fc->base_us + 2 * slice_us)
        {
            fc->rate    = fc->cut_rate;
            fc->rate_us = now_us;
        }
        fc->cut_rate   = 0.0;
        fc->cut_rtt_us = 0;
        fc->base_us    = rtt;
        fc->timed_us   = rtt;
        fc->quiet_us   = now_us;
    }
    else if (rtt < fc->base_us)
    {
        fc->base_us = rtt;
        if (rtt + 2 * slice_us < fc->timed_us)
        {
            /* the round trip shrank: the window keeps the rest of the old one queued: time it */
            fc->timed_us = rtt;
            fc->gap_us   = TV_FC_REBASE_US;
            fc->quiet_us = now_us - TV_FC_REBASE_US;
        }
    }
    /* rtt - base is the queue ahead of the request plus its own flush: about a slice when the
     * window is right */
    int64_t q = rtt - fc->base_us;

    /* delivery rate: the bytes shown since an earlier answer (far enough back for ssh's
     * bunching), over the longer of the times between the answers and between their requests,
     * so that answers held up and then let through together do not inflate it. A sample is low
     * when there was little to send: only one taken while the window held image back behind a
     * queue replaces a maximum gone stale. The first window's answer counts however soon it
     * comes after the startup answer. */
    double  sample = 0.0, ans = 0.0;
    int     sampled = 0, first = fc->first;
    int64_t least = tv_max64(5000, slice_us / 2);
    if (now_us - fc->grow_us > fc->base_us + slice_us)
    {
        fc->grow_us   = now_us;
        fc->grow_rate = fc->rate;
    }
    if (fc->ref_us != 0 && (first || now_us - fc->ref_us >= least))
    {
        int64_t span = tv_max64(tv_max64(now_us - fc->ref_us, r->us - fc->ref_sus), least);
        sample       = (double) (r->at - fc->ref_at) / (double) span;
        ans     = (double) (r->at - fc->ref_at) / (double) tv_max64(now_us - fc->ref_us, least);
        sample  = (fc->grow_rate > 0.0) ? fmin(sample, 2.0 * fc->grow_rate) : sample;
        sampled = 1;
        if (sample >= fc->rate || (now_us - fc->rate_us > horizon && fc->limited && q >= slice_us))
        {
            fc->rate    = sample;
            fc->rate_us = now_us;
        }
    }
    if (fc->ref_us == 0 || sampled)
    {
        fc->ref_us  = now_us;
        fc->ref_sus = r->us;
        fc->ref_at  = r->at;
    }
    if (fc->ramp_us != 0 && r->us == fc->ramp_us && fc->n > 0 && fc->req[fc->head].us == r->us)
    {
        fc->ref_us = 0; /* a drain's last request, with image in the same write: no reference */
    }

    /* Over 3 slices of queue for a slice of time, the link is slower than the maximum says (one
     * late answer is not enough: delays vary): the rate goes to what the answers themselves
     * show (sparse requests, as during a stall, say nothing), by half at most per round trip.
     * Over 1.25 slices for longer, the round trip may have grown: tv_fc_room() times it again. */
    fc->high_us = (q > 2 * slice_us) ? (fc->high_us ? fc->high_us : now_us) : 0;
    if (q > slice_us + slice_us / 4)
    {
        fc->lift_rtt_us = (fc->lift_us && fc->lift_rtt_us < rtt) ? fc->lift_rtt_us : rtt;
        fc->lift_us     = fc->lift_us ? fc->lift_us : now_us;
    }
    else
    {
        fc->lift_us = 0;
    }
    if (q > 3 * slice_us && fc->high_us && now_us - fc->high_us >= slice_us && sampled &&
        ans < fc->rate && now_us - fc->cut_us > fc->base_us + slice_us)
    {
        fc->cut_rtt_us = (fc->cut_rate > 0.0) ? fc->cut_rtt_us : 0;
        fc->cut_rate   = fmax(fc->cut_rate, fc->rate);
        fc->rate       = fmax(ans, 0.5 * fc->rate); /* by half at most per round trip: jitter */
        fc->rate_us    = now_us;
        fc->cut_us     = now_us;
    }
    fc->cut_rtt_us = (fc->cut_rate > 0.0) ? tv_max64(fc->cut_rtt_us, rtt) : 0;
    fc->full_us    = (fc->limited || fc->busy) ? now_us : fc->full_us;

    double w   = (double) fc->window;
    double est = fc->rate * (double) (fc->base_us + slice_us);
    if (first && sampled)
    {
        w = est; /* what the first window took: the link, at once */
    }
    else if (fc->limited && q < slice_us)
    {
        w = fmax(w + (double) delta, fmin(est, 2.0 * w)); /* no queue yet: a window more per trip */
    }
    else if (fc->rate > 0.0)
    {
        w = fmin(est, 2.0 * w);
    }
    if (fc->rate > 0.0 && now_us - fc->full_us > horizon)
    {
        w = fmin(w,
                 est / 4.0); /* long with little to send: start below it, the link may be slower */
    }
    fc->window  = (size_t) ((w < TV_FC_MIN) ? TV_FC_MIN : (w > TV_FC_MAX) ? TV_FC_MAX : w);
    fc->limited = 0;
    fc->busy    = 0;
    fc->first   = first && !sampled;
}

/** tv_fc_used() - Column @col is still asked: a request in it may yet be answered (in the last
 * column, @edge, also one past it, which a narrower terminal answers there). */
static int tv_fc_used(const uint64_t *used, int col, int edge)
{
    for (int c = col; c <= (edge ? 128 : col); c++)
    {
        if ((used[(c - 1) / 64] >> ((c - 1) % 64)) & 1)
        {
            return 1;
        }
    }
    return 0;
}

/*
 * The (hidden) cursor goes to the last row, in a column of its own among 17..16+TV_FC_IDS (from
 * the left, where a narrower terminal does not move it): the answer names its request, and its
 * row (2 or more) and column (past 16) keep it apart from modified F3 (CSI 1;<mod> R). A column
 * is not reused while an earlier request in it may still be answered: with none free, the
 * request waits for answers (unless none is out).
 */
size_t tv_fc_request(tv_state_t *ctx, char *out, int64_t now_us)
{
    tv_fc_t *fc = &ctx->fc;
    if (fc->n == TV_FC_REQS || fc->closing || fc->state == TV_FC_OFF || ctx->rows < 2)
    {
        return 0;
    }
    uint64_t used[2] = { fc->gone[0], fc->gone[1] };
    for (int k = 0; k < fc->n; k++)
    {
        int c = fc->req[(fc->head + k) % TV_FC_REQS].col - 1;
        used[c / 64] |= 1ull << (c % 64);
    }
    int ids = (ctx->cols > 17 + TV_FC_IDS) ? TV_FC_IDS : (ctx->cols > 18) ? ctx->cols - 17 : 1;
    int col = 0;
    for (unsigned i = 0; i < (unsigned) ids && col == 0; i++)
    {
        int c = (ids > 1) ? 17 + (int) ((fc->seq + i) % (unsigned) ids) : ctx->cols;
        col   = tv_fc_used(used, c, ids == 1) ? 0 : c;
        fc->seq += (col && ids > 1) ? i : 0;
    }
    if (col == 0 && fc->n > 0)
    {
        return 0;
    }
    col            = col ? col : (ids > 1) ? 17 + (int) (fc->seq % (unsigned) ids) : ctx->cols;
    tv_fc_req_t *r = &fc->req[(fc->head + fc->n++) % TV_FC_REQS];
    r->us          = now_us;
    r->col         = col;
    fc->seq += (ids > 1);
    return (size_t) snprintf(out, 32, "\033[%d;%dH\033[6n", ctx->rows, r->col);
}

void tv_fc_wrote(tv_fc_t *fc, size_t n, size_t req, int failed)
{
    fc->sent += n;
    if (req > 0 && failed)
    {
        fc->n--; /* it may not have reached the terminal */
    }
    else if (req > 0)
    {
        tv_fc_req_t *r = &fc->req[(fc->head + fc->n - 1) % TV_FC_REQS];
        r->at          = fc->sent;
        r->inflight    = fc->sent - fc->acked;
        /* at the rate known now; not while answers are late: a queue may hide behind them */
        r->quiet = fc->late_us == 0 && r->inflight <= tv_fc_quiet(fc);
    }
}

size_t tv_fc_lead(tv_state_t *ctx, char *out, int64_t now_us, size_t room)
{
    tv_fc_t *fc = &ctx->fc;
    /* draining, and nothing left to wait for: this one times the round trip, and bytes written
     * after it cannot delay its answer, so the image need not wait for it */
    int    quiet = fc->drain_us != 0 && fc->sent - fc->acked + 32 <= tv_fc_quiet(fc);
    size_t n     = 0;
    if (fc->ask || quiet || (room == 0 && fc->n == 0))
    {
        n = tv_fc_request(ctx, out, now_us);
    }
    if (n > 0)
    {
        fc->ask = 0;
        if (quiet)
        {
            fc->drain_us = 0;
            fc->quiet_us = now_us;
            fc->ramp_us  = now_us;
        }
    }
    return n;
}

void tv_fc_plan(tv_state_t *ctx, size_t room)
{
    tv_emit_t *e = &ctx->emit;
    e->budget = (room != SIZE_MAX && (ctx->budget == 0 || room < ctx->budget)) ? room : ctx->budget;
    e->hold   = (room == 0);
    ctx->fc.wait = e->hold && ctx->fc.state == TV_FC_ON; /* an answer sends what is held */
}

void tv_fc_flushed(tv_state_t *ctx, size_t room)
{
    const tv_emit_t *e  = &ctx->emit;
    tv_fc_t         *fc = &ctx->fc;
    fc->limited         = fc->limited || (fc->wbound && e->pending && e->budget == room);
    fc->busy            = fc->busy || e->pending; /* more to send than went (any limit): not idle */
}

int tv_fc_more(const tv_state_t *ctx)
{
    return ctx->emit.pending && !ctx->fc.wait;
}

/** tv_fc_req_due() - When the answer to @r is overdue: four round trips, then its queue drained
 * at a quarter of the measured rate (TV_FC_FLOOR at least), then TV_FC_LOST_US. */
static int64_t tv_fc_req_due(const tv_fc_t *fc, const tv_fc_req_t *r)
{
    if (fc->state == TV_FC_PROBE)
    {
        return r->us + TV_FC_PROBE_US;
    }
    double rate = (fc->rate / 4.0 > TV_FC_FLOOR / 1e6) ? fc->rate / 4.0 : TV_FC_FLOOR / 1e6;
    return r->us + TV_FC_LOST_US + 4 * fc->base_us + (int64_t) ((double) r->inflight / rate);
}

int tv_fc_match(const tv_state_t *ctx, int col)
{
    const tv_fc_t *fc   = &ctx->fc;
    int            edge = col == ctx->cols; /* a narrower terminal answers past its edge there */
    int            k    = 0;
    while (k < fc->n && fc->req[(fc->head + k) % TV_FC_REQS].col != col &&
           !(edge && fc->req[(fc->head + k) % TV_FC_REQS].col > col))
    {
        k++;
    }
    return k;
}

void tv_fc_answer(tv_state_t *ctx, int64_t now_us, int col)
{
    tv_fc_t *fc = &ctx->fc;
    int      k  = tv_fc_match(ctx, col);
    if (k == fc->n)
    {
        return; /* to a request given up on (a stop, a full ring), or not ours */
    }
    tv_fc_req_t r = fc->req[(fc->head + k) % TV_FC_REQS]; /* the k before it got no answer */
    /* an answer that late came behind a queue the trickle may hide: it times nothing */
    r.quiet =
        r.quiet && fc->late_us == 0 && (fc->state != TV_FC_ON || now_us < tv_fc_req_due(fc, &r));
    fc->head = (fc->head + k + 1) % TV_FC_REQS;
    fc->n -= k + 1;
    fc->gone[0] = fc->gone[1] = 0; /* answers come in order: the terminal is past all before */
    uint64_t delta            = (r.at > fc->acked) ? r.at - fc->acked : 0;
    fc->acked                 = (r.at > fc->acked) ? r.at : fc->acked;
    fc->credit                = 0; /* what trickled out is now answered for, or will be */
    fc->nlate                 = 0;
    /* answers to requests written while they were late: the trickle keeps its pace until one
     * is on time again, not one more per answer */
    fc->late_us = (fc->late_us && fc->n > 0 && fc->state == TV_FC_ON &&
                   now_us >= tv_fc_req_due(fc, &fc->req[fc->head]))
                      ? fc->late_us
                      : 0;
    if (fc->state != TV_FC_ON)
    {
        /* the startup request, possibly after it timed out: an answer that late came behind
         * a queue, so the round trip is timed again soon */
        fc->base_us  = now_us - r.us;
        fc->timed_us = fc->base_us;
        fc->gap_us   = TV_FC_REBASE_US;
        fc->quiet_us = (fc->state == TV_FC_PROBE) ? now_us : now_us - TV_FC_REBASE_US;
        fc->ref_us   = now_us;
        fc->ref_sus  = r.us;
        fc->ref_at   = r.at;
        fc->full_us  = now_us;
        fc->first    = 1;
        fc->state    = TV_FC_ON;
    }
    else
    {
        tv_fc_window(fc, now_us, &r, delta, tv_slice_us(ctx));
    }
    fc->ask = fc->ask || (fc->n == 0 && fc->sent > fc->acked); /* cover what no request did */
    /* send what was held now, not after the interactive cap: room came, a request is wanted, or
     * a drain can send its quiet request */
    if ((fc->wait || ctx->emit.pending) &&
        (fc->ask || tv_fc_room(ctx, now_us) > 0 ||
         (fc->drain_us != 0 && fc->sent - fc->acked + 32 <= tv_fc_quiet(fc))))
    {
        fc->release = 1;
        fc->wait    = 0;
        ctx->dirty |= TV_D_CHROME;
    }
}

int64_t tv_fc_due_us(const tv_state_t *ctx)
{
    const tv_fc_t *fc = &ctx->fc;
    if (fc->n == 0 || fc->state == TV_FC_OFF || fc->closing)
    {
        return 0;
    }
    return fc->late_us ? fc->late_us : tv_fc_req_due(fc, &fc->req[fc->head]);
}

/* A late answer is not a lost one: the request keeps its place, to be matched when its answer
 * comes (unless the ring is full: then the oldest goes, its column unused until an answer
 * comes). Meanwhile a little image goes now and then (less and less often), in case answers
 * stopped for good; it is not counted as answered. Without an answer to the startup request,
 * flow control stops (until one comes after all). */
void tv_fc_expire(tv_state_t *ctx, int64_t now_us)
{
    tv_fc_t *fc  = &ctx->fc;
    int64_t  due = tv_fc_due_us(ctx);
    if (due == 0 || now_us < due)
    {
        return;
    }
    if (fc->state == TV_FC_PROBE)
    {
        fc->state = TV_FC_OFF; /* the requests stay, for answers that come after all */
    }
    else
    {
        if (fc->n == TV_FC_REQS)
        {
            int c = fc->req[fc->head].col - 1;
            fc->gone[c / 64] |= 1ull << (c % 64);
            fc->head = (fc->head + 1) % TV_FC_REQS;
            fc->n--;
        }
        uint64_t out = fc->sent - fc->acked;
        fc->credit   = (out > fc->credit + TV_FC_MIN) ? fc->credit + TV_FC_MIN : out;
        fc->late_us =
            now_us + (TV_FC_LOST_US << ((fc->nlate < TV_FC_TRICKLES) ? fc->nlate : TV_FC_TRICKLES));
        fc->nlate++;
    }
    fc->wait    = 0;
    fc->release = 1;
    ctx->dirty |= TV_D_CHROME;
}

int64_t tv_fc_settle_end(const tv_state_t *ctx, int64_t now_us)
{
    const tv_fc_t *fc = &ctx->fc;
    if (fc->n == 0)
    {
        return now_us;
    }
    /* until an answer comes, 3 s from the startup request at most (perhaps behind earlier output
     * still on its way); then the newest request's due time */
    const tv_fc_req_t *r     = &fc->req[(fc->head + fc->n - 1) % TV_FC_REQS];
    int64_t            start = fc->req[fc->head].us + TV_FC_SETTLE_MAX_US;
    int64_t            due   = (fc->state == TV_FC_OFF) ? start : tv_fc_req_due(fc, r);
    due                      = (fc->state == TV_FC_PROBE && due > start) ? start : due;
    if (fc->state == TV_FC_OFF && due <= now_us)
    {
        return now_us;
    }
    /* answers on time: as long as the queue ahead of the newest takes, which the far terminal
     * is still showing; once one is overdue, the link may be gone */
    int on_time =
        fc->state == TV_FC_ON && fc->late_us == 0 && now_us < tv_fc_req_due(fc, &fc->req[fc->head]);
    int64_t lim = on_time ? TV_FC_SETTLE_LONG_US : TV_FC_SETTLE_MAX_US;
    return (due < now_us + TV_FC_SETTLE_MIN_US) ? now_us + TV_FC_SETTLE_MIN_US
           : (due > now_us + lim)               ? now_us + lim
                                                : due;
}

void tv_fc_resume(tv_fc_t *fc)
{
    for (int k = 0; k < fc->n; k++)
    {
        int c = fc->req[(fc->head + k) % TV_FC_REQS].col - 1; /* may be answered after all */
        fc->gone[c / 64] |= 1ull << (c % 64);
    }
    fc->n        = 0;
    fc->wait     = 0;
    fc->closing  = 0;
    fc->release  = 0;
    fc->limited  = 0;
    fc->busy     = 0;
    fc->first    = 0;
    fc->drain_us = 0;
    fc->high_us  = 0;
    fc->lift_us  = 0;
    fc->late_us  = 0;
    fc->nlate    = 0;
    fc->ref_us   = 0;
    fc->acked    = fc->sent;
    fc->credit   = 0;
    fc->ask      = 1; /* before the repaint: in PROBE, it is the startup request again */
}

void tv_settle_begin(tv_state_t *ctx, tv_settle_t *st, int64_t now_us)
{
    st->t0          = now_us;
    st->cut         = 0;
    st->state       = ctx->fc.state;
    st->end         = tv_fc_settle_end(ctx, now_us);
    ctx->fc.closing = 1;
}

void tv_settle_answer(tv_state_t *ctx, tv_settle_t *st, int64_t now_us, int col)
{
    tv_fc_t *fc = &ctx->fc;
    int      n  = fc->n;
    tv_fc_answer(ctx, now_us, col);
    if (fc->state != st->state)
    {
        st->state = fc->state; /* the first answer: the rest come on time, behind their queue */
        st->end   = tv_fc_settle_end(ctx, now_us);
    }
    else if (fc->state == TV_FC_ON && fc->n < n)
    {
        /* answers still coming, however late: the link is up, wait on behind its queue */
        int64_t more = now_us + TV_FC_SETTLE_MAX_US;
        more    = (more < st->t0 + TV_FC_SETTLE_LONG_US) ? more : st->t0 + TV_FC_SETTLE_LONG_US;
        st->end = (more > st->end) ? more : st->end;
    }
    st->end = (st->cut && st->cut < st->end) ? st->cut : st->end;
}

void tv_settle_cut(tv_state_t *ctx, tv_settle_t *st, int64_t now_us)
{
    /* not a short wait: a double tap to quit would send the answers to the shell */
    int64_t two = 2 * (ctx->fc.base_us + tv_slice_us(ctx));
    st->cut     = tv_max64(now_us, st->t0 + tv_max64(1000000, two));
    st->end     = (st->end < st->cut) ? st->end : st->cut;
}

/** tv_settle_read() - Wait until @end for input on stdin and feed it to @in: 1 on input or a
 * signal, 0 on timeout, -1 when the terminal hung up. */
static int tv_settle_read(tv_input_t *in, int64_t end)
{
    int64_t       now   = tv_now_us();
    struct pollfd pfd   = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
    int           ready = (now < end) ? poll(&pfd, 1, (int) ((end - now + 999) / 1000)) : 0;
    if (ready <= 0)
    {
        return (ready < 0 && errno == EINTR) ? 1 : 0; /* a resize is no pause in typing */
    }
    if (!(pfd.revents & POLLIN))
    {
        return -1;
    }
    unsigned char buf[sizeof(in->buf)];
    ssize_t       n = read(STDIN_FILENO, buf, tv_input_room(in));
    if ((n == 0 && (pfd.revents & (POLLHUP | POLLERR))) ||
        (n < 0 && errno != EINTR && errno != EAGAIN))
    {
        return -1;
    }
    tv_input_feed(in, buf, (n > 0) ? (size_t) n : 0, tv_now_us());
    return 1;
}

/**
 * tv_fc_settle() - Read the answers still owed before the shell gets the terminal back, dropping
 * keys typed meanwhile and the rest of a line begun then (all keys left, if @quitting).
 */
static int tv_fc_settle(tv_state_t *ctx, tv_input_t *in, int quitting)
{
    tv_fc_t    *fc = &ctx->fc;
    tv_settle_t st;
    int         nq = tv_sig_nquit, dropped = 0, partial = 0;
    int64_t     hard = 0; /* the discarding of a line begun meanwhile ends by then */
    tv_settle_begin(ctx, &st, tv_now_us());
    while (tv_sig_nquit == nq && !tv_term_hup && tv_input_room(in) > 0)
    {
        /* what was read with the key that brought us here may hold answers already; once none
         * is owed (or waited for), the keys after it stay for after a resume */
        tv_event_t ev;
        int64_t    now = tv_now_us();
        while (((fc->n > 0 && now < st.end) || partial || quitting) && tv_input_next(in, now, &ev))
        {
            if (ev.type == TV_EV_CPR)
            {
                tv_settle_answer(ctx, &st, now, ev.x);
            }
            else if (ev.type == TV_EV_KEY && ev.key == TV_CTRL('c'))
            {
                tv_settle_cut(ctx, &st, now);
                partial = 0; /* as in the shell, it ends a line typed meanwhile */
            }
            else if (ev.type == TV_EV_KEY)
            {
                dropped++;
                if (ev.key == '\r' || ev.key == '\n' || ev.key == TV_CTRL('u'))
                {
                    partial = 0; /* Enter or Ctrl-U ends a line */
                }
                else if (ev.key != 'q' && ev.key != TV_K_ESC && ev.key != TV_CTRL('z') &&
                         ev.key != TV_CTRL('\\'))
                {
                    partial = 1; /* the viewer's own q, Esc, Ctrl-Z, Ctrl-\ begin none */
                }
            }
        }
        hard = (fc->n == 0 && partial && hard == 0) ? now + TV_FC_LINE_MAX_US : hard;
        if ((fc->n == 0 && !partial) || (fc->n > 0 && now >= st.end) || (hard && now >= hard))
        {
            break;
        }
        /* answers in: the rest of a line begun meanwhile, until its end or a pause in typing */
        int64_t idle = now + TV_FC_LINE_IDLE_US;
        int     r    = tv_settle_read(in, (fc->n > 0) ? st.end : (idle < hard) ? idle : hard);
        if (r < 0)
        {
            tv_term_hup = 1;
        }
        else if (r == 0 && fc->n == 0)
        {
            break;
        }
    }
    fc->closing = 0;
    return dropped;
}

/** tv_note_dropped() - Say on stderr (the terminal is back) that @n typed keys were dropped. */
static void tv_note_dropped(int n)
{
    if (n > 0)
    {
        fprintf(stderr, "milk-termview: %d key%s typed while waiting for the terminal dropped\n", n,
                (n > 1) ? "s" : "");
    }
}

/* What the window leaves unanswered, a slice's worth at most; none while the queue drains to
 * time the round trip. Until the startup request is answered, the first window. */
size_t tv_fc_room(tv_state_t *ctx, int64_t now_us)
{
    tv_fc_t *fc = &ctx->fc;
    fc->wbound  = 0;
    if (fc->state == TV_FC_OFF || (fc->state == TV_FC_PROBE && fc->n == 0))
    {
        return SIZE_MAX;
    }
    uint64_t unanswered = fc->sent - fc->acked;
    unanswered -= (fc->credit < unanswered) ? fc->credit : unanswered;
    if (fc->state == TV_FC_PROBE)
    {
        return (unanswered < fc->window) ? (size_t) (fc->window - unanswered) : 0;
    }
    int64_t slice = tv_slice_us(ctx);
    int     lift =
        fc->lift_us != 0 && now_us - fc->lift_us > tv_max64(2000000, 8 * (fc->base_us + slice));
    if (fc->drain_us == 0 && (now_us - fc->quiet_us > fc->gap_us || lift))
    {
        /* due, or answers keep showing a queue: the round trip grew? */
        fc->drain_us     = now_us;
        fc->drain_rtt_us = lift ? tv_max64(fc->base_us, fc->lift_rtt_us) : fc->base_us;
        fc->high_us      = 0;
        fc->lift_us      = 0;
    }
    if (fc->drain_us != 0)
    {
        if (now_us - fc->drain_us <= 2 * fc->drain_rtt_us + 4 * slice + 200000)
        {
            return 0;
        }
        /* no quiet answer came: take the round trip the answers showed (an answer says less if
         * it can), and try again later */
        fc->drain_us = 0;
        fc->quiet_us = now_us;
        fc->ramp_us  = now_us;
        fc->base_us  = fc->drain_rtt_us;
        fc->timed_us = fc->base_us;
        fc->gap_us = (2 * fc->gap_us < TV_FC_REBASE_MAX_US) ? 2 * fc->gap_us : TV_FC_REBASE_MAX_US;
    }
    /* after a drain the window opens over a round trip, not at once: keys would wait behind it */
    double  w    = (double) fc->window;
    int64_t up   = now_us - fc->ramp_us + slice;
    int     ramp = up < fc->base_us + slice;
    w            = ramp ? w * (double) up / (double) (fc->base_us + slice) : w;
    size_t room  = (unanswered < (uint64_t) w) ? (size_t) w - (size_t) unanswered : 0;
    /* a slice per flush: answers come back a slice apart and each releases the next one, so the
     * link does not idle while the answer to a whole window travels back */
    size_t chunk = (size_t) ((double) fc->window * (double) slice / (double) (fc->base_us + slice));
    chunk        = (chunk < fc->window / (TV_FC_REQS - 2)) ? fc->window / (TV_FC_REQS - 2) : chunk;
    fc->wbound   = !ramp && room <= chunk; /* the window, not the slice or the ramp, limits it */
    /* not in crumbs: a flush per answer of what it freed would wake the viewer for little; but
     * while no queue shows, a chunk and what it leaves of the window both go, or a short round
     * trip idles the link once per answer */
    size_t least = (chunk / 2 < TV_FC_CRUMB) ? chunk / 2 : TV_FC_CRUMB;
    size_t rest  = (fc->window > chunk) ? (fc->window - chunk) / 2 : 0;
    rest         = (rest < 1024) ? 1024 : rest;
    least        = (least > rest && fc->high_us == 0) ? rest : least;
    return (room < least) ? 0 : (room < chunk) ? room : chunk;
}

void tv_budget_update(tv_state_t *ctx, size_t n, int64_t dt_us)
{
    /* Image bytes per flush. Once a write blocks, a flush carries what the terminal drains in
     * a frame period, so a key never waits longer behind image bytes. A large write that goes
     * through at once means the terminal caught up: the budget doubles until it is dropped. */
    if (dt_us > TV_FLUSH_BLOCKED_US)
    {
        /* a slower link is believed at once, a faster one halfway: a short write that kernel
         * buffers absorbed overstates the rate */
        double rate     = (double) n / (double) dt_us;
        ctx->drain_rate = (rate > ctx->drain_rate && ctx->drain_rate > 0.0)
                              ? 0.5 * (ctx->drain_rate + rate)
                              : rate;
        double b        = ctx->drain_rate * (double) tv_slice_us(ctx);
        b               = (b > 2.0 * (double) n) ? 2.0 * (double) n : b; /* same reason */
        ctx->budget     = (b < TV_FLUSH_MIN_BYTES) ? TV_FLUSH_MIN_BYTES : (size_t) b;
    }
    else if (ctx->budget > 0 && n >= ctx->budget / 2)
    {
        ctx->budget = (ctx->budget > TV_FLUSH_MAX_BYTES / 2) ? 0 : 2 * ctx->budget;
        if (ctx->budget == 0)
        {
            ctx->drain_rate = 0.0;
        }
    }
}

static void termview_flush_ansi(tv_state_t *ctx)
{
    if (tv_sig_cont)
    {
        return; /* stopped while rendering: the screen is not ours until re-entered */
    }
    /* pointer motion reports only while a menu or the help box can use them */
    tv_fc_t *fc    = &ctx->fc;
    int      hover = ctx->ui.open || ctx->ui.help;
    if (hover != ctx->ui.hover_sent)
    {
        ctx->ui.hover_sent = hover;
        /* in xterm and tmux, resetting 1003 ends mouse reporting: back to button events */
        tv_write_all(hover ? "\033[?1003h" : "\033[?1003l\033[?1002h", hover ? 8 : 16);
        tv_fc_wrote(fc, hover ? 8 : 16, 0, 0);
    }

    /* image bytes this flush: the budget from blocked writes, within what flow control allows;
     * interface cells always go */
    tv_emit_t *e    = &ctx->emit;
    int64_t    t0   = tv_now_us();
    size_t     room = tv_fc_room(ctx, t0);
    char       lead[32];
    size_t     nl = tv_fc_lead(ctx, lead, t0, room);
    if (nl > 0)
    {
        tv_fc_wrote(fc, nl, nl, tv_write_all(lead, nl) != 0);
        room = tv_fc_room(ctx, t0);
    }
    tv_fc_plan(ctx, room);
    size_t n =
        tv_flush_diff(e, ctx->frame_buffer, ctx->screen, ctx->prev_screen, ctx->rows, ctx->cols);
    tv_fc_flushed(ctx, room);
    if (n == 0)
    {
        return;
    }
    size_t req = tv_fc_request(ctx, ctx->frame_buffer + n, t0); /* room: tv_framebuf_size() */
    n += req;
    ctx->flushes++;
    int64_t t1     = tv_now_us();
    int     failed = tv_write_all(ctx->frame_buffer, n) != 0;
    if (failed)
    {
        tv_invalidate(ctx); /* partial frame on screen: repaint everything next time */
    }
    tv_fc_wrote(fc, n, req, failed);
    tv_budget_update(ctx, n, tv_now_us() - t1);
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

int tv_resize_waits(const tv_state_t *ctx, const tv_timers_t *tm, int64_t now_us)
{
    const tv_fc_t *fc = &ctx->fc;
    return tm->resized_us != 0 && fc->wait && fc->n > 0 &&
           now_us < tm->resized_us + fc->base_us + 2 * tv_slice_us(ctx);
}

int64_t tv_wake_at(const tv_state_t *ctx, const tv_timers_t *tm, int64_t now_us, int64_t esc_us)
{
    /* the earliest of: next data poll, frame cap (dirty, or image cells left over by the last
     * flush), a held resize, an overdue answer, ESC timeout, tick */
    int64_t dl    = ctx->paused ? tm->next_tick : tm->next_data; /* paused: one wakeup a second */
    int     defer = tv_resize_waits(ctx, tm, now_us);
    int     more  = (ctx->dirty && !defer) || tv_fc_more(ctx); /* image held: wait to be answered */
    if (more && tm->last_render + TV_INTERACTIVE_US < dl)
    {
        dl = (ctx->fc.release && ctx->dirty) ? now_us : tm->last_render + TV_INTERACTIVE_US;
    }
    int64_t until = tm->resized_us + ctx->fc.base_us + 2 * tv_slice_us(ctx);
    int64_t fc_dl = tv_fc_due_us(ctx);
    dl            = (defer && until < dl) ? until : dl;
    dl            = (fc_dl > 0 && fc_dl < dl) ? fc_dl : dl;
    dl            = (esc_us >= 0 && esc_us < dl) ? esc_us : dl;
    return (tm->next_tick < dl) ? tm->next_tick : dl;
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
    ctx->fc.window     = TV_FC_START;
    ctx->fc.ask        = 1; /* before the first paint: does the terminal answer? (TV_FC_PROBE) */
    tv_timers_t tm     = { .next_data = tv_now_us() };
    tm.next_tick       = tm.next_data + 1000000;
    tm.last_render     = tm.next_data - TV_INTERACTIVE_US;
    int reenter_failed = 0;

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
            tv_fc_resume(&ctx->fc);
        }
        if (tv_sig_winch)
        {
            tv_sig_winch = 0; /* cleared before the query: a resize during it is not lost */
            tv_term_size(&ctx->rows, &ctx->cols);
            tv_invalidate(ctx);
            tm.resized_us = tm.resized_us ? tm.resized_us : tv_now_us();
        }

        int64_t now    = tv_now_us();
        int64_t period = 1000000 / ctx->target_fps;
        if (tm.next_data > now + period)
        {
            tm.next_data = now + period; /* the rate was raised: do not wait out the old period */
        }
        int64_t dl = tv_wake_at(ctx, &tm, now, tv_input_deadline(&in));

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
                else if ((n == 0 && (pfd.revents & (POLLHUP | POLLERR))) ||
                         (n < 0 && errno != EINTR && errno != EAGAIN))
                {
                    ctx->exit_status = 1; /* terminal closed */
                    tv_term_hup      = 1;
                    break;
                }
                /* else nothing: raw reads return 0 when a stop let someone else take the input */
            }
            else if (!(pfd.revents & POLLIN) && (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)))
            {
                ctx->exit_status = 1;
                tv_term_hup      = 1;
                break;
            }
        }

        tv_dispatch_input(ctx, &in, now, 512);
        if (!ctx->running || tv_sig_quit || tv_term_hup)
        {
            break;
        }
        if (ctx->suspend || tv_sig_tstp)
        {
            ctx->suspend = 0;
            tv_sig_tstp  = 0;
            if (tv_term_can_suspend())
            {
                int dropped = tv_fc_settle(ctx, &in, 0);
                if (tv_sig_quit || tv_term_hup)
                {
                    break; /* quitting after all: the exit reads what is still owed */
                }
                tv_term_restore();
                tv_note_dropped(dropped);
                tv_term_suspend(); /* stops here; the resume is handled at the top of the loop */
            }
            continue;
        }
        if (tv_sig_cont)
        {
            continue; /* resumed: restore the terminal before drawing anything */
        }
        tv_fc_expire(ctx, now);

        if (now >= tm.next_tick || ctx->recheck)
        {
            ctx->recheck = 0;
            tv_stream_monitor(ctx, now);
        }
        if (now >= tm.next_tick)
        {
            tv_update_stats(ctx, now);
            tm.next_tick = (tm.next_tick + 1000000 > now) ? tm.next_tick + 1000000 : now + 1000000;
            ctx->dirty |= TV_D_CHROME;
        }

        if (tv_fc_more(ctx))
        {
            ctx->dirty |= TV_D_CHROME; /* send the image cells the last flush had no room for */
        }
        if (now >= tm.next_data && !ctx->paused)
        {
            tm.next_data = (tm.next_data + period > now) ? tm.next_data + period : now + period;
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

        if (tv_resize_waits(ctx, &tm, now))
        {
            continue; /* the new size is painted once the image may go */
        }
        /* stream frames follow the data poll; input-driven frames are capped */
        if (!(ctx->dirty & TV_D_DATA) &&
            !(ctx->dirty && (now - tm.last_render >= TV_INTERACTIVE_US || ctx->fc.release)))
        {
            continue;
        }
        tm.last_render  = now;
        ctx->fc.release = 0;
        /* a user change shows a frame that arrived since the poll; other renders (the tick, a
         * flush continuation) leave new frames to the data poll and its rate */
        if ((ctx->dirty & (TV_D_GEOM | TV_D_MAP | TV_D_FULL)) && !ctx->paused && tv_mapped(ctx) &&
            tv_note_cnt0(ctx, tv_cnt0(ctx), now))
        {
            ctx->dirty |= TV_D_DATA;
        }
        int dirty     = ctx->dirty;
        ctx->dirty    = 0;
        tm.resized_us = 0;

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

    int dropped = 0;
    if (tv_sig_quit != SIGHUP && !tv_term_hup && !reenter_failed && tv_term_is_active())
    {
        dropped = tv_fc_settle(ctx, &in, 1); /* an answer after the restore would reach the shell */
    }
    tv_term_restore();
    tv_note_dropped(dropped);
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
