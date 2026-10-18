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
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

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

#endif /* TERMVIEW_INTERNAL_H */
