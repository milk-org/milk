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
    TV_G_DOWN,              /**< U+25BC, menu marker and scroll hint */
    TV_G_UP,                /**< U+25B2, scroll hint */
    TV_G_DOT,               /**< U+25CF, status pill and selected choice */
    TV_G_SHADE,             /**< U+2592, drop shadow */
    TV_G_HLINE,             /**< U+2500 */
    TV_G_VLINE,             /**< U+2502 */
    TV_G_TL,                /**< U+256D, rounded corners: top left */
    TV_G_TR,                /**< U+256E */
    TV_G_BL,                /**< U+2570 */
    TV_G_BR,                /**< U+256F */
    TV_G_COUNT
};

/** Value never produced by tv_cell(): marks a cell whose on-screen content is unknown. */
#define TV_CELL_UNKNOWN UINT64_MAX

/** Packed 24-bit colour, as stored in a cell. */
#define TV_RGB(r, g, b) (((uint32_t) (r) << 16) | ((uint32_t) (g) << 8) | (uint32_t) (b))

/**
 * tv_rgb - Pack an RGB triple @c into a 24-bit `0xRRGGBB` integer
 * @c: RGB color struct
 *
 * Return: Packed 24-bit color value.
 */
static inline uint32_t tv_rgb(tv_rgb_t c)
{
    return ((uint32_t) c.r << 16) | ((uint32_t) c.g << 8) | c.b;
}

/**
 * tv_cell - Pack background color, foreground color, and glyph index into a 64-bit `tv_cell_t`
 * @bg:    24-bit background RGB color (`bits 0..23`)
 * @fg:    24-bit foreground RGB color (`bits 24..47`)
 * @glyph: Glyph table index (`bits 48..55`)
 *
 * Return: Packed 64-bit cell word.
 */
static inline tv_cell_t tv_cell(uint32_t bg, uint32_t fg, uint8_t glyph)
{
    return (uint64_t) (bg & 0xFFFFFF) | ((uint64_t) (fg & 0xFFFFFF) << 24) |
           ((uint64_t) glyph << 48);
}

/**
 * tv_cell_bg - Extract 24-bit background RGB color from packed cell @c
 * @c: Packed cell word
 *
 * Return: 24-bit background RGB value.
 */
static inline uint32_t tv_cell_bg(tv_cell_t c)
{
    return (uint32_t) (c & 0xFFFFFF);
}

/**
 * tv_cell_fg - Extract 24-bit foreground RGB color from packed cell @c
 * @c: Packed cell word
 *
 * Return: 24-bit foreground RGB value.
 */
static inline uint32_t tv_cell_fg(tv_cell_t c)
{
    return (uint32_t) ((c >> 24) & 0xFFFFFF);
}

/**
 * tv_cell_glyph - Extract 8-bit glyph table index from packed cell @c
 * @c: Packed cell word
 *
 * Return: Glyph index (`0..255`).
 */
static inline uint8_t tv_cell_glyph(tv_cell_t c)
{
    return (uint8_t) (c >> 48);
}

/**
 * tv_cell_with_fg - Return a copy of cell @c with its 24-bit foreground color replaced by @fg
 * @c:  Original packed cell word
 * @fg: New 24-bit foreground RGB value
 *
 * Return: Updated packed cell word.
 */
static inline tv_cell_t tv_cell_with_fg(tv_cell_t c, uint32_t fg)
{
    return (c & ~((uint64_t) 0xFFFFFF << 24)) | ((uint64_t) (fg & 0xFFFFFF) << 24);
}

/**
 * tv_cell_with_glyph - Return a copy of cell @c with its glyph index replaced by @glyph
 * @c:     Original packed cell word
 * @glyph: New 8-bit glyph index
 *
 * Return: Updated packed cell word.
 */
static inline tv_cell_t tv_cell_with_glyph(tv_cell_t c, uint8_t glyph)
{
    return (c & ~((uint64_t) 0xFF << 48)) | ((uint64_t) glyph << 48);
}

/** Attribute bit: drawn by the user interface, not the image; sent ahead of image cells. */
#define TV_ATTR_CHROME ((uint64_t) 1 << 56)

/**
 * tv_cell_is_chrome - Check whether cell @c carries the UI chrome priority bit (`TV_ATTR_CHROME`)
 * @c: Packed cell word
 *
 * Return: 1 if @c is a UI chrome cell, 0 if it is an image cell.
 */
static inline int tv_cell_is_chrome(tv_cell_t c)
{
    return (c & TV_ATTR_CHROME) != 0;
}

/** tv_framebuf_size() - Worst-case bytes tv_flush_diff() can emit for a rows x cols grid. */
size_t tv_framebuf_size(int rows, int cols);

#define TV_QCACHE_BITS 12
#define TV_QCACHE (1 << TV_QCACHE_BITS)

/** Output state: colour depth, RGB -> palette memo, and the byte budget for slow terminals. */
typedef struct
{
    int      level; /**< 3 truecolor, 2 xterm-256, 1 sixteen colours */
    uint32_t qkey[TV_QCACHE];
    uint8_t  qval[TV_QCACHE];
    size_t   budget;     /**< image bytes per flush, 0 = unlimited */
    int      hold;       /**< interface cells only: the image waits (pending) */
    int      resume_row; /**< where the next budgeted flush starts */
    int      pending;    /**< the last flush left image cells unsent */
} tv_emit_t;

void tv_emit_init(tv_emit_t *e, int level);

/** tv_emit_code() - @rgb as sent at @e's depth: the RGB itself, or the palette index. */
uint32_t tv_emit_code(tv_emit_t *e, uint32_t rgb);

/**
 * tv_flush_diff() - Escape sequences turning @prev into @cur, written to @out; @prev is updated.
 * Cells whose appearance at @e's colour depth did not change are not sent. Interface cells go
 * first; image cells follow within e->budget (none with e->hold), rows rotating so that none
 * starves, and the rest waits for the next call (e->pending). Returns 0 when nothing is to write.
 */
size_t tv_flush_diff(tv_emit_t       *e,
                     char            *out,
                     const tv_cell_t *cur,
                     tv_cell_t       *prev,
                     int              rows,
                     int              cols);

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
extern volatile sig_atomic_t tv_sig_nquit; /**< terminating signals received */
extern volatile sig_atomic_t tv_sig_tstp;  /**< SIGTSTP: stop at the next chance */

int  tv_term_init(char *err, size_t errsz);
int  tv_term_enter(void);
void tv_term_restore(void);

/** tv_term_can_suspend() - SIGTSTP is neither blocked nor was ignored at startup. */
int tv_term_can_suspend(void);

/** tv_term_suspend() - Restore the terminal and stop until SIGCONT; nothing if it cannot. */
void tv_term_suspend(void);

/** tv_term_is_active() - The terminal is in the viewer's modes (entered, not restored). */
int  tv_term_is_active(void);
void tv_term_size(int *rows, int *cols);
int  tv_write_all(const char *buf, size_t n);

/** tv_term_color_level() - 3, 2 or 1 from COLORTERM, else TERM (24-bit names, then 256-colour). */
int  tv_term_color_level(void);
void tv_term_reraise(int sig);

/* -----------------------------------------------------------------------
 * Input decoding (tv_input.c)
 * ----------------------------------------------------------------------- */

/** An incomplete escape sequence older than this is resolved (lone ESC) or dropped. */
#define TV_ESC_TIMEOUT_US 50000

enum
{
    TV_EV_KEY = 1,
    TV_EV_MOUSE,
    TV_EV_CPR /**< cursor position report (answer to CSI 6n): y row, x column, 1-based */
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

/** tv_stream_open() - Validate and map stream @arg (name, name.im.shm or path) read-only. */
int  tv_stream_open(tv_stream_t *s, const char *arg, char *err, size_t errsz);
void tv_stream_close(tv_stream_t *s);

enum
{
    TV_STREAM_OK,    /**< the path still names the mapped file */
    TV_STREAM_MOVED, /**< removed or replaced: the mapping stays readable but is not the stream */
    TV_STREAM_CLOSED /**< no mapping (the file shrank under it and it was dropped) */
};

/** tv_stream_safe() - The mapping can be read: drops it first if the file shrank below it. */
int tv_stream_safe(tv_stream_t *s);

/** tv_stream_check() - Health check; unmaps at once a file truncated below the mapping. */
int tv_stream_check(tv_stream_t *s);

/** tv_stream_reopen() - Replace @s with the stream now at s->path, if it is valid; 0 on success. */
int tv_stream_reopen(tv_stream_t *s);

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
#define TV_BAR_WIDTH 13  /* gutter, colorbar, gap and 9 label columns */
#define TV_CHROME_ROWS 2 /* menubar above the image, status bar below */

/** Screen areas, from the terminal size. */
typedef struct
{
    int rows, cols;
    int img_top;  /**< screen row of the first image row (below the menubar) */
    int img_rows; /**< character rows of image; each shows 2 sample rows */
    int img_cols;
    int bar_col;    /**< first column right of the image: gutter, then the colorbar */
    int status_row; /**< status bar (or the range minibuffer) */
    int valid;      /**< large enough to draw the image */
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
    uint16_t  *shown; /**< LUT index on screen per sample (low-bandwidth mode) */
    size_t     col_cap, colin_cap, row_cap, rowin_cap, samp_cap, prev_cap, shown_cap, cells_cap;
    int        have_prev;
    double     offset, prev_offset; /**< subtracted before the float conversion */
    float      vmin, vmax;          /**< finite in-image samples, relative */
    long       nvalid;
    uint32_t   hist[4096];
    double     lo, hi; /**< display range, raw units */
    int        have_range;
    int        settled;    /**< smoothing has reached the measured range */
    int        have_shown; /**< shown[] matches the current tables */
    int        phase;      /**< low-bandwidth refresh row (mod TV_LOWBW_CYCLE) */
    long       held;       /**< samples the last render left at a stale colour */
    int        blank;      /**< cells are empty: no data for this geometry */
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

/* Low-bandwidth mode: a sample keeps the colour on screen while the exact one is within
 * TV_LOWBW_DIST2 (squared RGB distance, about 4 levels per channel). One cell row in
 * TV_LOWBW_CYCLE is redrawn exactly each render; once the stream rests or repeats a frame, the
 * viewer renders until nothing is held. */
#define TV_LOWBW_DIST2 48
#define TV_LOWBW_CYCLE 8

/** tv_image_render() - Colour the samples into im->cells with @lut; @lowbw holds small changes. */
void tv_image_render(tv_image_t *im, const tv_rgb_t *lut, int lowbw);

/** tv_image_blank() - Empty image layer for the current tables, when there is nothing to sample. */
void tv_image_blank(tv_image_t *im);

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
    TV_D_FULL   = 0x10, /**< terminal content unknown: repaint every cell */
    TV_D_HELD   = 0x20  /**< low-bandwidth: redraw rows still showing held colours */
};

/** Key code of Ctrl+@c. */
#define TV_CTRL(c) ((uint32_t) (c) & 0x1F)

/** Input-driven frames are spaced at least this far apart (about 60 Hz). */
#define TV_INTERACTIVE_US 16000

/** User actions; tv_do() is the single place that applies them. */
typedef enum
{
    TV_ACT_QUIT,
    TV_ACT_SUSPEND,
    TV_ACT_REPAINT,
    TV_ACT_CMAP,   /**< arg: step */
    TV_ACT_SCALE,  /**< arg: step */
    TV_ACT_RANGE,  /**< arg: step */
    TV_ACT_LOCK,   /**< arg: 1 lock the range shown, 0 unlock, -1 flip */
    TV_ACT_PROMPT, /**< arg: 1 min, 2 max */
    TV_ACT_FLIP_H,
    TV_ACT_FLIP_V,
    TV_ACT_ZOOM, /**< arg: notches, > 0 zooms in */
    TV_ACT_VIEW_RESET,
    TV_ACT_PAN_X,      /**< arg: +1 right, -1 left, in screen space */
    TV_ACT_PAN_Y,      /**< arg: +1 down, -1 up, in screen space */
    TV_ACT_FPS,        /**< arg: step */
    TV_ACT_ESCAPE,     /**< cancel drags */
    TV_ACT_LOWBW,      /**< toggle low-bandwidth mode */
    TV_ACT_PAUSE,      /**< freeze or resume the image */
    TV_ACT_SET_CMAP,   /**< arg: colormap */
    TV_ACT_SET_SCALE,  /**< arg: scale */
    TV_ACT_SET_RANGE,  /**< arg: range mode; unlocks the range */
    TV_ACT_SET_COLORS, /**< arg: 0 auto, 1 = 16, 2 = 256, 3 = 24-bit */
    TV_ACT_SET_FPS,    /**< arg: UI refresh rate */
    TV_ACT_SMOOTH,     /**< toggle range smoothing */
    TV_ACT_CENTRE,     /**< centre the view, keep the zoom */
    TV_ACT_HELP        /**< open or close the key help */
} tv_action_t;

/** Dropdown menus of the menubar, left to right. */
enum
{
    TV_M_CMAP,
    TV_M_SCALE,
    TV_M_RANGE,
    TV_M_VIEW,
    TV_M_STREAM,
    TV_M_COUNT
};

/** A clickable screen rectangle, inclusive, 0-based cells; hit-testing goes topmost first. */
typedef struct
{
    int16_t x0, y0, x1, y1;
    uint8_t kind; /**< TV_HIT_* */
    int16_t id;
} tv_hit_t;

enum
{
    TV_HIT_BUTTON, /**< menubar button: id is the menu, TV_M_COUNT Help, TV_M_COUNT + 1 Quit */
    TV_HIT_ITEM,   /**< dropdown item: id is its index */
    TV_HIT_PANEL,  /**< the rest of an open dropdown or of the help box */
    TV_HIT_FIELD,  /**< minibuffer field: id 0 min, 1 max */
    TV_HIT_CHROME  /**< menubar or status row elsewhere */
};

#define TV_HITS 64

/** Menus, help box and hover state. */
typedef struct
{
    int      open;       /**< a menu is open: menu */
    int      menu;       /**< the open (or last opened) menu, TV_M_* */
    int      sel;        /**< highlighted item */
    int      top;        /**< first item shown when the menu scrolls */
    int      saved;      /**< the menu's choice when it opened (restored on Esc) */
    int      saved_lock; /**< the range lock when the Range menu opened */
    double   saved_lo;   /**< the range shown when the menu opened */
    double   saved_hi;
    unsigned gen;        /**< bumped whenever what is open changes */
    unsigned hits_gen;   /**< gen of the last draw: older hit rects are not used */
    int      help;       /**< the help box is open */
    int      help_top;   /**< first help line shown */
    int      hover;      /**< menubar button under the pointer + 1, 0 for none */
    int      hover_sent; /**< pointer motion reports the terminal has: 1, 0, or -1 unknown */
    int16_t  btn_x[TV_M_COUNT + 2]; /**< left column of each menubar button */
    tv_hit_t hits[TV_HITS];
    int      nhits;
} tv_ui_t;

/**
 * End-to-end flow control. A cursor position request (CSI 6n) sent after a flush is answered
 * only once the terminal has shown everything before it, however much ssh or a relay buffered on
 * the way; image bytes are held while more than a window of them is unanswered. Each request puts
 * the (hidden) cursor in its own column, so an answer names the request it answers.
 */
enum
{
    TV_FC_PROBE, /**< no answer yet: a first window applies until one comes */
    TV_FC_ON,
    TV_FC_OFF /**< the startup request got no answer: no flow control, until it does */
};

#define TV_FC_REQS 64           /* requests outstanding at most */
#define TV_FC_IDS 72            /* columns that tell requests apart (17..88: two digits) */
#define TV_FC_START (64u << 10) /* window until the answers measure the link */
#define TV_FC_MIN (8u << 10)
#define TV_FC_CRUMB 4096 /* image bytes a flush carries at least (or half a slice, if less) */
#define TV_FC_MAX (4u << 20)
#define TV_FC_QUIET \
    1024 /* a request behind fewer unanswered bytes (or 5 ms of link) times the trip */
#define TV_FC_REBASE_US 5000000 /* without such a timing for this long the queue drains for one */
#define TV_FC_REBASE_MAX_US 40000000  /* ... less often while the timings agree */
#define TV_FC_PROBE_US 2000000        /* no answer to the startup request: no flow control */
#define TV_FC_LOST_US 2000000         /* an answer this late, past its round trip and queue ... */
#define TV_FC_FLOOR 8192              /* ... drained at least this fast (bytes/s): image trickles */
#define TV_FC_TRICKLES 4              /* ... every 2 s, then 4, 8, 16, 32 s while none comes */
#define TV_FC_SETTLE_MIN_US 200000    /* leaving: waiting for outstanding answers, ... */
#define TV_FC_SETTLE_MAX_US 3000000   /* ... at most this long after the last (or the start) ... */
#define TV_FC_SETTLE_LONG_US 30000000 /* ... and this long in all */
#define TV_FC_LINE_IDLE_US 1500000    /* a line typed meanwhile is dropped until this pause ... */
#define TV_FC_LINE_MAX_US 10000000    /* ... or this long after the last answer */

typedef struct
{
    uint64_t at;       /**< bytes written up to and with the request */
    uint64_t inflight; /**< of those, unanswered when it was written */
    int64_t  us;       /**< when it was written */
    int      col;      /**< where it put the cursor: the answer's column */
    int      quiet;    /**< behind so little that its answer times the round trip */
} tv_fc_req_t;

typedef struct
{
    int      state;
    int      n;           /**< requests outstanding */
    int      head;        /**< the oldest one in req[] */
    unsigned seq;         /**< requests so far: picks the column */
    int      wait;        /**< image held until an answer comes */
    int      ask;         /**< a request is wanted before the next flush */
    int      release;     /**< an answer let held image go: flush without the interactive cap */
    int      limited;     /**< the window held image back since the last answer */
    int      closing;     /**< reading the last answers: no new request */
    int      busy;        /**< more image than went since the last answer: not idle */
    int      first;       /**< the startup answer came: the next one measures the first window */
    int      nlate;       /**< trickles since the last answer */
    uint64_t gone[2];     /**< columns (bit col-1) of requests given up on, maybe still answered */
    uint64_t sent;        /**< bytes written so far */
    uint64_t acked;       /**< bytes the terminal has shown */
    uint64_t credit;      /**< of the unanswered bytes, what trickled out while answers were late */
    int64_t  base_us;     /**< round trip with nothing queued */
    int64_t  timed_us;    /**< base_us as a quiet answer or a drain last timed it */
    int64_t  quiet_us;    /**< when the round trip was last timed, or a timing begun or given up */
    int64_t  gap_us;      /**< time between such timings */
    int64_t  drain_us;    /**< image held since, to time it; 0 if not */
    int64_t  high_us;     /**< answers show a queue over two slices since; 0 if not */
    int64_t  lift_us;     /**< answers show a queue over 1.25 slices since: the trip grew? 0: no */
    int64_t  lift_rtt_us; /**< the shortest round trip among those answers */
    int64_t  late_us;     /**< the oldest answer is overdue: next trickle of image; 0 if not */
    int64_t  drain_rtt_us; /**< the round trip the current drain expects */
    int64_t  full_us;      /**< when an answer last came while the window held image back */
    double   rate;         /**< delivery rate, bytes/us (recent maximum); 0 until measured */
    int64_t  rate_us;      /**< when that maximum was seen */
    double   grow_rate;    /**< the rate a round trip ago: it at most doubles per round trip */
    int64_t  grow_us;
    double   cut_rate;   /**< the rate before a queue cut it (back if the round trip grew) */
    int64_t  cut_rtt_us; /**< the longest round trip since that cut */
    int64_t  cut_us;     /**< when the rate was last cut */

    int64_t     ramp_us; /**< a drain ended: the window opens over a round trip from then */
    int         wbound;  /**< the room tv_fc_room() gave is what the window leaves */
    int64_t     ref_us;  /**< the answer the next delivery sample counts from ... */
    int64_t     ref_sus; /**< ... and when its request was written */
    uint64_t    ref_at;
    size_t      window; /**< unanswered bytes allowed */
    tv_fc_req_t req[TV_FC_REQS];
} tv_fc_t;

/** Range minibuffer on the status row. */
typedef struct
{
    int         active;
    int         field;       /**< 0 min, 1 max */
    char        text[2][24]; /**< what is typed */
    int         fresh[2];    /**< the next typed key replaces the pre-filled text */
    const char *err;         /**< why Enter would be refused, NULL when valid */
} tv_mini_t;

typedef struct
{
    tv_stream_t       *st;
    termview_options_t opt;
    tv_view_t          view;
    tv_layout_t        lay; /**< layout of the last frame */
    tv_image_t         im;
    tv_emit_t          emit;
    double             drain_rate; /**< terminal bytes/us measured on blocked writes, 0 if fast */
    size_t             budget;     /**< image bytes per flush from blocked writes, 0 = no limit */
    tv_fc_t            fc;
    int                suspend;    /**< Ctrl-Z: stop once the outstanding answers are read */
    int                resting;    /**< the last new frame repeated the previous one */
    int                paused;     /**< Space: no new frames; the image comes from frozen */
    void              *frozen;     /**< copy of the displayed slice at pause, NULL if none */
    int                gone;       /**< the path no longer names the mapped stream */
    int                recheck;    /**< check the stream at the next loop pass */
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

    tv_ui_t   ui;
    tv_mini_t mini;
    int       flushes; /**< flushes that wrote bytes since the last stats update */
    double    ui_fps;  /**< screen updates per second */

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

/** tv_fc_window() - Learn from the answer to @r at @now_us, which acknowledged @delta bytes:
 * round trip, delivery rate, window (rate x (round trip + @slice_us)). */
void tv_fc_window(tv_fc_t           *fc,
                  int64_t            now_us,
                  const tv_fc_req_t *r,
                  uint64_t           delta,
                  int64_t            slice_us);

/** tv_fc_match() - Which outstanding request (0: the oldest) an answer in @col is for; n: none. */
int tv_fc_match(const tv_state_t *s, int col);

/** tv_fc_answer() - The terminal answered at @now_us with the cursor in column @col. */
void tv_fc_answer(tv_state_t *s, int64_t now_us, int col);

/** tv_fc_room() - Image bytes the next flush may carry at @now_us (SIZE_MAX: no limit); may start
 * or end a drain, and sets wbound. */
size_t tv_fc_room(tv_state_t *s, int64_t now_us);

/** tv_fc_request() - A request into @out (32 bytes) if one may go out; returns its length. */
size_t tv_fc_request(tv_state_t *s, char *out, int64_t now_us);

/** tv_fc_lead() - A request to write before the next flush (@room from tv_fc_room()), or 0. */
size_t tv_fc_lead(tv_state_t *s, char *out, int64_t now_us, size_t room);

/** tv_fc_wrote() - @n bytes written, the last @req of them a request (dropped if @failed). */
void tv_fc_wrote(tv_fc_t *fc, size_t n, size_t req, int failed);

/** tv_fc_plan() - The emitter's budget and hold for a flush that may carry @room image bytes. */
void tv_fc_plan(tv_state_t *s, size_t room);

/** tv_fc_flushed() - After the emitter: note whether the window (@room) held image back. */
void tv_fc_flushed(tv_state_t *s, size_t room);

/** tv_fc_more() - Image is left over and may go without waiting for an answer. */
int tv_fc_more(const tv_state_t *s);

/** tv_fc_due_us() - When the oldest outstanding answer is overdue; 0 if none can be. */
int64_t tv_fc_due_us(const tv_state_t *s);

/** tv_fc_expire() - At @now_us, act on an overdue answer (tv_fc_due_us()). */
void tv_fc_expire(tv_state_t *s, int64_t now_us);

/** tv_fc_settle_end() - Until when, leaving the terminal at @now_us, answers are waited for. */
int64_t tv_fc_settle_end(const tv_state_t *s, int64_t now_us);

/** tv_fc_resume() - After a stop: what was asked before is not waited for; ask again. */
void tv_fc_resume(tv_fc_t *fc);

/** Leaving the terminal: until when the answers still owed are waited for. */
typedef struct
{
    int64_t t0;  /**< the wait began */
    int64_t end; /**< it ends then, or once nothing is owed */
    int64_t cut; /**< a Ctrl-C: it ends by then; 0 if none */
    int     state;
} tv_settle_t;

/** tv_settle_begin() - Start waiting at @now_us for the answers still owed. */
void tv_settle_begin(tv_state_t *s, tv_settle_t *st, int64_t now_us);

/** tv_settle_answer() - An answer in column @col read while waiting: the wait follows answers. */
void tv_settle_answer(tv_state_t *s, tv_settle_t *st, int64_t now_us, int col);

/** tv_settle_cut() - Ctrl-C while waiting: a long wait ends soon (a short one still finishes). */
void tv_settle_cut(tv_state_t *s, tv_settle_t *st, int64_t now_us);

/** Main loop timers. */
typedef struct
{
    int64_t next_data;   /**< the next stream poll */
    int64_t next_tick;   /**< the next 1 Hz tick */
    int64_t last_render; /**< the last frame drawn */
    int64_t resized_us;  /**< a new size not painted yet: since when; 0 if none */
} tv_timers_t;

/** tv_resize_waits() - A new size's repaint (every interface cell) waits while the image does, for
 * an answer due within a round trip: a drag sends one per answer, not one per step. */
int tv_resize_waits(const tv_state_t *s, const tv_timers_t *tm, int64_t now_us);

/** tv_wake_at() - When the main loop wakes next, at @now_us, with the ESC deadline @esc_us (<0:
 * none). */
int64_t tv_wake_at(const tv_state_t *s, const tv_timers_t *tm, int64_t now_us, int64_t esc_us);

/** tv_budget_update() - Adapt the blocked-write budget to a flush of @n bytes that took @dt_us. */
void tv_budget_update(tv_state_t *s, size_t n, int64_t dt_us);

/** tv_stale_age_us() - How long the stream has gone without a new frame, 0 while it is live. */
int64_t tv_stale_age_us(const tv_state_t *s, int64_t now_us);

/* -----------------------------------------------------------------------
 * Interface: menubar, menus, status bar, minibuffer, help (tv_ui.c)
 * ----------------------------------------------------------------------- */
/** Keys and mouse, as the help box and -h show them: { keys, what they do }. */
extern const char *const tv_help_keys[][2];
extern const int         tv_help_nkeys;

/** tv_ui_render() - Draw the interface over the composed image and register its hit rects. */
void tv_ui_render(tv_state_t *s, int64_t now_us);

/** tv_ui_key() - Keys for an open menu, the help box or the minibuffer, and the menu keys; 1 if
 * consumed. */
int tv_ui_key(tv_state_t *s, const tv_event_t *ev);

/** tv_ui_wants_mouse() - @ev goes to tv_ui_mouse(), not to the image. */
int  tv_ui_wants_mouse(const tv_state_t *s, const tv_event_t *ev);
void tv_ui_mouse(tv_state_t *s, const tv_event_t *ev);

/** tv_ui_open() - Open menu @m (-1: close, reverting an uncommitted choice). */
void tv_ui_open(tv_state_t *s, int m);

/** tv_mini_open() - Open the range minibuffer on field @field (0 min, 1 max), pre-filled. */
void tv_mini_open(tv_state_t *s, int field);

/** tv_now_us() - Monotonic clock in microseconds. */
static inline int64_t tv_now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

#endif /* TERMVIEW_INTERNAL_H */
