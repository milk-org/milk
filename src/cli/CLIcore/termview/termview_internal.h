// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file termview_internal.h
 * @brief milk-termview internals shared by the viewer modules and unit tests.
 */

#ifndef TERMVIEW_INTERNAL_H
#define TERMVIEW_INTERNAL_H

#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include "ImageStreamIO/ImageStruct.h"
#include "termview.h"

/** Colormap LUT entries; index TV_LUT_N is the reserved NaN/Inf colour. */
#define TV_LUT_N 1024

/** Terminal size caps (keeps cell and frame-buffer arithmetic far from overflow). */
#define TV_MAX_ROWS 1000
#define TV_MAX_COLS 2000

/** tv_finite() - IEEE finiteness by exponent bit test; immune to -ffinite-math-only. */
static inline int tv_finite(double x)
{
    uint64_t u;
    memcpy(&u, &x, sizeof(u));
    return (u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

/**
 * tv_lut_idx() - Map @v to a LUT index in [0, TV_LUT_N - 1], or TV_LUT_N if @v is not finite.
 * @scale must be (TV_LUT_N - 1) / (hi - lo), finite and >= 0. Clamps in the double domain first.
 */
static inline int tv_lut_idx(double v, double lo, double scale)
{
    if (!tv_finite(v))
    {
        return TV_LUT_N;
    }
    double t = (v - lo) * scale;
    if (!(t > 0.0))
    {
        return 0;
    }
    if (!(t < (double) (TV_LUT_N - 1)))
    {
        return TV_LUT_N - 1;
    }
    return (int) t;
}

/** tv_parse_double() - strtod with full consumption (trailing blanks allowed); 0 if finite. */
static inline int tv_parse_double(const char *s, double *out)
{
    if (s == NULL || *s == '\0')
    {
        return -1;
    }
    char  *end = NULL;
    double v   = strtod(s, &end);
    if (end == s)
    {
        return -1;
    }
    while (*end == ' ' || *end == '\t')
    {
        end++;
    }
    if (*end != '\0' || !tv_finite(v))
    {
        return -1;
    }
    *out = v;
    return 0;
}

/** tv_view_sanitize() - Reset NaN view state, clamp zoom to [0.1, @zmax] and centre to [0, 1]. */
static inline void tv_view_sanitize(double *zoom, double *cx, double *cy, double zmax)
{
    if (!tv_finite(zmax) || zmax < 1.0)
    {
        zmax = 1.0;
    }
    if (!tv_finite(*zoom) || !(*zoom > 0.0))
    {
        *zoom = 1.0;
    }
    if (*zoom < 0.1)
    {
        *zoom = 0.1;
    }
    if (*zoom > zmax)
    {
        *zoom = zmax;
    }
    if (!tv_finite(*cx))
    {
        *cx = 0.5;
    }
    if (!tv_finite(*cy))
    {
        *cy = 0.5;
    }
    *cx = (*cx < 0.0) ? 0.0 : ((*cx > 1.0) ? 1.0 : *cx);
    *cy = (*cy < 0.0) ? 0.0 : ((*cy > 1.0) ? 1.0 : *cy);
}

/** tv_fmt_len() - Clamp an snprintf() return value to the bytes actually stored. */
static inline int tv_fmt_len(int n, size_t cap)
{
    if (n < 0)
    {
        return 0;
    }
    return ((size_t) n >= cap) ? (int) cap - 1 : n;
}

/* -----------------------------------------------------------------------
 * Screen cells: one 64-bit word per terminal cell, compared as an integer
 * ----------------------------------------------------------------------- */
typedef struct
{
    uint8_t r, g, b;
} tv_rgb_t;

/** Packed cell: bits 0-23 background RGB, 24-47 foreground RGB, 48-55 glyph, 56-63 attributes. */
typedef uint64_t tv_cell_t;

/** Glyph codes below 0x80 are ASCII; codes from 0x80 index the UTF-8 table in tv_flush.c. */
enum
{
    TV_G_LOWER_HALF = 0x80, /**< U+2584, top half shows bg, bottom half fg */
    TV_G_FULL_BLOCK,        /**< U+2588 */
    TV_G_COUNT
};

/** Value never produced by tv_cell(): marks a cell whose on-screen content is unknown. */
#define TV_CELL_UNKNOWN UINT64_MAX

/** Packed 24-bit colour, as stored in a cell. */
#define TV_RGB(r, g, b) (((uint32_t) (r) << 16) | ((uint32_t) (g) << 8) | (uint32_t) (b))

static inline uint32_t tv_rgb(tv_rgb_t c)
{
    return ((uint32_t) c.r << 16) | ((uint32_t) c.g << 8) | c.b;
}

static inline tv_cell_t tv_cell(uint32_t bg, uint32_t fg, uint8_t glyph)
{
    return (uint64_t) (bg & 0xFFFFFF) | ((uint64_t) (fg & 0xFFFFFF) << 24) |
           ((uint64_t) glyph << 48);
}

static inline uint32_t tv_cell_bg(tv_cell_t c)
{
    return (uint32_t) (c & 0xFFFFFF);
}

static inline uint32_t tv_cell_fg(tv_cell_t c)
{
    return (uint32_t) ((c >> 24) & 0xFFFFFF);
}

static inline uint8_t tv_cell_glyph(tv_cell_t c)
{
    return (uint8_t) (c >> 48);
}

static inline tv_cell_t tv_cell_with_fg(tv_cell_t c, uint32_t fg)
{
    return (c & ~((uint64_t) 0xFFFFFF << 24)) | ((uint64_t) (fg & 0xFFFFFF) << 24);
}

static inline tv_cell_t tv_cell_with_glyph(tv_cell_t c, uint8_t glyph)
{
    return (c & ~((uint64_t) 0xFF << 48)) | ((uint64_t) glyph << 48);
}

/** tv_framebuf_size() - Worst-case bytes tv_flush_diff() can emit for a rows x cols grid. */
size_t tv_framebuf_size(int rows, int cols);

/**
 * tv_flush_diff() - Escape sequences turning @prev into @cur, written to @out; @prev is updated.
 * Returns 0 when no cell changed (nothing to write).
 */
size_t tv_flush_diff(char *out, const tv_cell_t *cur, tv_cell_t *prev, int rows, int cols);

/* -----------------------------------------------------------------------
 * Colormaps (tv_cmap.c)
 * ----------------------------------------------------------------------- */

/** tv_cmap_color() - Colour of @cmap at @v in [0, 1] (clamped). */
tv_rgb_t tv_cmap_color(termview_colormap_t cmap, double v);

/** tv_lut_build() - Fill @lut[TV_LUT_N + 1] for @cmap and @scale; last entry: NaN colour. */
void tv_lut_build(tv_rgb_t *lut, termview_colormap_t cmap, termview_scale_t scale);

/* -----------------------------------------------------------------------
 * Name tables (tv_cmap.c)
 * ----------------------------------------------------------------------- */
extern const char *const tv_cmap_names[COLORMAP_NB];
extern const char *const tv_scale_names[SCALE_NB];
extern const char *const tv_range_names[RANGE_NB];
extern const char *const tv_range_keys[RANGE_NB];

/** tv_lookup_name() - Case-insensitive exact or unique-prefix match; index or -1. */
int tv_lookup_name(const char *const *names, int n, const char *s);

/* -----------------------------------------------------------------------
 * Terminal lifecycle (tv_term.c)
 * ----------------------------------------------------------------------- */
extern volatile sig_atomic_t tv_sig_quit;  /**< terminating signal number, 0 if none */
extern volatile sig_atomic_t tv_sig_winch; /**< window size changed */
extern volatile sig_atomic_t tv_sig_cont;  /**< resumed after a stop */
extern volatile sig_atomic_t tv_term_hup;  /**< terminal hung up or write failed */

int  tv_term_init(char *err, size_t errsz);
int  tv_term_enter(void);
void tv_term_restore(void);
void tv_term_suspend(void);
void tv_term_size(int *rows, int *cols);
int  tv_write_all(const char *buf, size_t n);
void tv_term_reraise(int sig);

/* -----------------------------------------------------------------------
 * Stream access (tv_stream.c)
 * ----------------------------------------------------------------------- */
typedef struct
{
    IMAGE    img;     /**< md/array point into the read-only mapping */
    int      fd;      /**< -1 when closed */
    void    *map;     /**< mmap base, NULL when closed */
    size_t   mapsize; /**< mapped length (file size at open) */
    dev_t    dev;
    ino_t    ino;
    uint32_t xsize; /**< geometry snapshot taken at open */
    uint32_t ysize; /**< max(size[1], 1) */
    uint32_t zsize; /**< size[2] for naxis 3, else 1 */
    uint8_t  naxis;
    uint8_t  datatype;
    char     name[STRINGMAXLEN_IMAGE_NAME]; /**< printable-ASCII copy of md->name */
    char     path[512];
} tv_stream_t;

int  tv_stream_open(tv_stream_t *s, const char *arg, char *err, size_t errsz);
void tv_stream_close(tv_stream_t *s);

/* -----------------------------------------------------------------------
 * Viewer state (termview.c)
 * ----------------------------------------------------------------------- */
typedef struct
{
    int button;   /**< 0 left, 1 middle, 2 right, 64/65 wheel up/down */
    int x;        /**< 1-indexed column */
    int y;        /**< 1-indexed row */
    int is_press; /**< 1 for press/drag, 0 for release */
    int is_drag;  /**< 1 if motion with a button held */
} tv_mouse_event_t;

/** Image-to-display mapping parameters; flips live in the options. */
typedef struct
{
    double zoom;
    double cx; /**< view centre, fraction of the image width */
    double cy; /**< view centre, fraction of the image height */
} tv_view_t;

typedef struct
{
    tv_stream_t       *st;
    termview_options_t opt;
    tv_view_t          view;
    int                rows, cols; /**< terminal size */
    int                running;
    int                dirty; /**< something changed: render the next frame */
    int                exit_status;

    tv_rgb_t lut[TV_LUT_N + 1]; /**< [TV_LUT_N]: NaN/Inf pixels */
    int      lut_cmap;
    int      lut_scale;
    double   cur_min, cur_max; /**< range of the last rendered frame */

    int             target_fps;
    double          current_cpu_usage;
    double          current_stream_fps;
    uint64_t        last_cnt0_for_fps;
    uint64_t        last_cnt0;
    struct timespec last_time_real;
    struct timespec last_time_cpu;
    struct timespec last_render_real;
    long            timeout_us;

    int             popup_type;
    struct timespec popup_expiry_time;

    tv_mouse_event_t mouse;
    int              mouse_is_dragging;
    int              roi_is_dragging;
    int              last_mouse_x;
    int              last_mouse_y;
    int              roi_start_x;
    int              roi_start_y;
    int              roi_end_x;
    int              roi_end_y;

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
} tv_state_t;

#endif /* TERMVIEW_INTERNAL_H */
