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
 * Input decoding (tv_input.c)
 * ----------------------------------------------------------------------- */

/** An incomplete escape sequence older than this is resolved (lone ESC) or dropped. */
#define TV_ESC_TIMEOUT_US 50000

enum
{
    TV_EV_KEY = 1,
    TV_EV_MOUSE
};

/** Keys: Unicode/ASCII and control bytes as themselves, special keys above U+10FFFF. */
enum
{
    TV_K_UP = 0x110000,
    TV_K_DOWN,
    TV_K_LEFT,
    TV_K_RIGHT,
    TV_K_HOME,
    TV_K_END,
    TV_K_PGUP,
    TV_K_PGDN,
    TV_K_INS,
    TV_K_DEL,
    TV_K_ESC,
    TV_K_F1, /* TV_K_F1 + 11 is F12 */
};

enum
{
    TV_MOD_SHIFT = 1,
    TV_MOD_ALT   = 2,
    TV_MOD_CTRL  = 4
};

enum
{
    TV_MB_LEFT = 0,
    TV_MB_MIDDLE,
    TV_MB_RIGHT,
    TV_MB_NONE,
    TV_MB_WHEEL_UP = 64,
    TV_MB_WHEEL_DOWN,
    TV_MB_WHEEL_LEFT,
    TV_MB_WHEEL_RIGHT
};

enum
{
    TV_MA_PRESS = 0,
    TV_MA_RELEASE,
    TV_MA_DRAG,
    TV_MA_MOVE
};

typedef struct
{
    uint8_t  type; /**< TV_EV_* */
    uint8_t  mods; /**< TV_MOD_* */
    uint8_t  button;
    uint8_t  action;
    uint32_t key;
    int16_t  x; /**< 1-indexed column */
    int16_t  y; /**< 1-indexed row */
} tv_event_t;

typedef struct
{
    unsigned char buf[256];
    size_t        len;
    int64_t       last_rx_us; /**< arrival time of the newest bytes */
    int           skip;       /**< inside an over-long sequence (internal state) */
    int           esc_stale;  /**< buf[0] is an ESC that timed out before more bytes came */
} tv_input_t;

/** tv_input_room() - Free bytes in the decoder buffer (what one read() may add). */
static inline size_t tv_input_room(const tv_input_t *in)
{
    return sizeof(in->buf) - in->len;
}

void tv_input_feed(tv_input_t *in, const unsigned char *data, size_t n, int64_t now_us);

/** tv_input_next() - Next decoded event at time @now_us; 1 if @ev was filled, 0 if none yet. */
int tv_input_next(tv_input_t *in, int64_t now_us, tv_event_t *ev);

/** tv_input_deadline() - When a pending partial sequence must be resolved, or -1. */
int64_t tv_input_deadline(const tv_input_t *in);

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
 * Image pipeline (tv_image.c)
 * ----------------------------------------------------------------------- */
/** Image-to-display mapping parameters; flips live in the options. */
typedef struct
{
    double zoom;
    double cx; /**< view centre, fraction of the image width */
    double cy; /**< view centre, fraction of the image height */
} tv_view_t;
#define TV_BAR_WIDTH 12 /* colorbar + labels */
#define TV_INFO_ROWS 4  /* rows below the image */

/** Screen areas, from the terminal size. */
typedef struct
{
    int rows, cols;
    int img_rows; /**< character rows of image; each shows 2 sample rows */
    int img_cols;
    int bar_col; /**< first colorbar column */
    int info_row;
    int valid; /**< large enough to draw the image */
} tv_layout_t;

/** Display sample <-> image pixel mapping; samples are square, 2 per character row. */
typedef struct
{
    double step;     /**< image pixels per display sample */
    double cx, cy;   /**< view centre in image pixels */
    double sx, sy;   /**< +1, or -1 when flipped */
    double cdx, cdy; /**< display centre in samples */
} tv_xform_t;

/** tv_xf_img_x() - Image x at display sample coordinate @dj (columns, 0 = left edge). */
static inline double tv_xf_img_x(const tv_xform_t *x, double dj)
{
    return x->cx + x->sx * (dj - x->cdx) * x->step;
}

/** tv_xf_img_y() - Image y at display sample coordinate @di (sample rows, 0 = top edge). */
static inline double tv_xf_img_y(const tv_xform_t *x, double di)
{
    return x->cy + x->sy * (di - x->cdy) * x->step;
}

/** tv_xf_disp_x() - Inverse of tv_xf_img_x(). */
static inline double tv_xf_disp_x(const tv_xform_t *x, double ix)
{
    return x->cdx + x->sx * (ix - x->cx) / x->step;
}

/** tv_xf_disp_y() - Inverse of tv_xf_img_y(). */
static inline double tv_xf_disp_y(const tv_xform_t *x, double iy)
{
    return x->cdy + x->sy * (iy - x->cy) / x->step;
}

/** Sampling tables, samples, display range and the image layer, reused across frames. */
typedef struct
{
    int        ncols, nrows; /**< samples: img_cols x (2 * img_rows) */
    int32_t   *col_src;      /**< image column per display column, -1 outside */
    uint8_t   *col_in;
    int64_t   *row_off; /**< element offset of the image row per sample row, -1 outside */
    uint8_t   *row_in;
    float     *samp;  /**< value - offset; NaN outside the image or not finite */
    float     *prev;  /**< previous frame's samples */
    tv_cell_t *cells; /**< img_rows x img_cols */
    size_t     col_cap, colin_cap, row_cap, rowin_cap, samp_cap, prev_cap, cells_cap;
    int        have_prev;
    double     offset, prev_offset; /**< subtracted before the float conversion */
    float      vmin, vmax;          /**< finite in-image samples, relative */
    long       nvalid;
    uint32_t   hist[4096];
    double     lo, hi; /**< display range, raw units */
    int        have_range;
    int        settled; /**< smoothing has reached the measured range */
} tv_image_t;

void   tv_layout_compute(tv_layout_t *l, int rows, int cols);
double tv_view_base(const tv_layout_t *l, uint32_t xsize, uint32_t ysize);
void   tv_xform_compute(tv_xform_t        *x,
                        const tv_layout_t *l,
                        const tv_view_t   *v,
                        uint32_t           xsize,
                        uint32_t           ysize,
                        bool               flip_h,
                        bool               flip_v);

/** tv_image_geom() - Rebuild the sampling tables for a new layout or view; -1 on alloc failure. */
int tv_image_geom(tv_image_t        *im,
                  const tv_layout_t *l,
                  const tv_xform_t  *x,
                  uint32_t           xsize,
                  uint32_t           ysize);

/** tv_image_sample() - Sample @data through the tables; returns 0 if identical to last frame. */
int tv_image_sample(tv_image_t *im, const void *data, uint8_t datatype);

/** tv_image_range() - Display range for @opt from the samples; @snap skips the smoothing. */
void tv_image_range(tv_image_t *im, const termview_options_t *opt, int snap);

/** tv_image_render() - Colour the samples into im->cells with @lut. */
void tv_image_render(tv_image_t *im, const tv_rgb_t *lut);

void tv_image_free(tv_image_t *im);

/* -----------------------------------------------------------------------
 * Viewer state (termview.c)
 * ----------------------------------------------------------------------- */
/** Dirty bits: what must be redone for the next frame. */
enum
{
    TV_D_DATA   = 0x01, /**< new stream frame: resample */
    TV_D_GEOM   = 0x02, /**< zoom, pan, flip or size changed: resample */
    TV_D_MAP    = 0x04, /**< colormap, scale or range changed: recolour */
    TV_D_CHROME = 0x08, /**< info bar, popup or prompt only */
    TV_D_FULL   = 0x10  /**< terminal content unknown: repaint every cell */
};

/** Input-driven frames are spaced at least this far apart (about 60 Hz). */
#define TV_INTERACTIVE_US 16000

/** User actions; tv_do() is the single place that applies them. */
typedef enum
{
    TV_ACT_QUIT,
    TV_ACT_SUSPEND,
    TV_ACT_REPAINT,
    TV_ACT_CMAP,  /**< arg: step */
    TV_ACT_SCALE, /**< arg: step */
    TV_ACT_RANGE, /**< arg: step */
    TV_ACT_LOCK,
    TV_ACT_PROMPT, /**< arg: 1 min, 2 max */
    TV_ACT_FLIP_H,
    TV_ACT_FLIP_V,
    TV_ACT_ZOOM, /**< arg: notches, > 0 zooms in */
    TV_ACT_VIEW_RESET,
    TV_ACT_PAN_X,      /**< arg: +1 right, -1 left, in screen space */
    TV_ACT_PAN_Y,      /**< arg: +1 down, -1 up, in screen space */
    TV_ACT_FPS,        /**< arg: step */
    TV_ACT_ESCAPE,     /**< cancel drags and close the popup */
    TV_ACT_POPUP_CLOSE /**< close the popup only */
} tv_action_t;

typedef struct
{
    tv_stream_t       *st;
    termview_options_t opt;
    tv_view_t          view;
    tv_layout_t        lay; /**< layout of the last frame */
    tv_image_t         im;
    int                rows, cols; /**< terminal size */
    int                running;
    int                dirty; /**< TV_D_* bits */
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
    int64_t         last_change_us; /**< when cnt0 last moved */
    double          rate_hz;        /**< last non-zero stream rate (display) */
    double          period_us;      /**< smoothed interval between new frames, 0 if unknown */
    int64_t         stats_time_us;
    struct timespec stats_cpu;

    int     popup_type;
    int64_t popup_expiry_us;

    int mouse_is_dragging;
    int roi_is_dragging;
    int last_mouse_x;
    int last_mouse_y;
    int roi_start_x;
    int roi_start_y;
    int roi_end_x;
    int roi_end_y;
    int wheel_x; /**< pointer position of the last wheel notch */
    int wheel_y;

    int         input_mode;
    char        input_buf[64];
    int         input_len;
    const char *input_err; /**< validation message shown in the prompt line */

    tv_cell_t *screen;
    tv_cell_t *prev_screen;
    int        screen_size;
    char      *frame_buffer;
    size_t     frame_buffer_size;
} tv_state_t;

/*
 * State changes from user input go through tv_do() (discrete actions), tv_set_view() (mouse pan
 * and box zoom) and tv_set_range() (typed limits); each sets the dirty bits it implies.
 */

/** tv_do() - Apply user action @a with argument @arg. */
void tv_do(tv_state_t *s, tv_action_t a, int arg);

/** tv_set_view() - Move the view to centre (@cx, @cy) at @zoom, clamped to the valid range. */
void tv_set_view(tv_state_t *s, double cx, double cy, double zoom);

/** tv_set_range() - Lock the display range to [@lo, @hi]; caller ensures lo < hi, both finite. */
void tv_set_range(tv_state_t *s, double lo, double hi);

/* Input handlers, exposed for the unit tests (termview.c) */
void tv_handle_key(tv_state_t *s, const tv_event_t *ev);
void tv_handle_mouse(tv_state_t *s, const tv_event_t *ev);
void tv_dispatch_input(tv_state_t *s, tv_input_t *in, int64_t now_us, int budget);

/** tv_now_us() - Monotonic clock in microseconds. */
static inline int64_t tv_now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

#endif /* TERMVIEW_INTERNAL_H */
