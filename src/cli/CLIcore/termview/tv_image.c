// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_image.c
 * @brief Image pipeline: layout, view transform, sampling tables, typed samplers, display range
 *        and the cached image layer. Pure: memory in, memory out.
 */

#include <float.h>
#include <math.h>

#include "ImageStreamIO/ImageStruct.h"
#include "libmilkcommon/milk_compiler.h"
#include "libmilkcommon/pixel_dispatch.h"
#include "termview_internal.h"

/* sample centres that fall exactly on a pixel edge go to the same side despite rounding noise */
#define TV_TIE_EPS 1e-7

static const double tv_range_lo[RANGE_NB] = { 0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20 };
static const double tv_range_hi[RANGE_NB] = { 1.0, 0.999, 0.995, 0.99, 0.95, 0.90, 0.85, 0.80 };

/**
 * tv_layout_compute - Compute screen region geometry from terminal dimensions
 * @l:    Output layout structure to populate
 * @rows: Terminal height in character rows
 * @cols: Terminal width in character columns
 *
 * Partitions the terminal into the top menubar row, central image viewport,
 * right-hand colorbar column, and bottom status/info rows, setting `l->valid`
 * when the image viewport is at least 2 rows by 4 columns.
 *
 * Return: void
 */
void tv_layout_compute(tv_layout_t *l, int rows, int cols)
{
    l->rows       = rows;
    l->cols       = cols;
    l->img_top    = 1;
    l->img_rows   = rows - TV_CHROME_ROWS;
    l->img_cols   = cols - TV_BAR_WIDTH;
    l->bar_col    = cols - TV_BAR_WIDTH;
    l->status_row = rows - 1;
    l->valid      = (l->img_rows >= 2 && l->img_cols >= 4);
}

/**
 * tv_view_base - Compute the unzoomed image-pixels-per-sample step for a layout
 * @l:     Screen layout with valid `img_cols` and `img_rows`
 * @xsize: Image width in pixels
 * @ysize: Image height in pixels
 *
 * Returns the larger of the horizontal and vertical pixel-to-sample ratios so
 * the entire image fits inside the viewport at zoom = 1.0 with square aspect.
 *
 * Return: Base sample step in image pixels
 */
double tv_view_base(const tv_layout_t *l, uint32_t xsize, uint32_t ysize)
{
    return fmax((double) xsize / l->img_cols, (double) ysize / (2.0 * l->img_rows));
}

/**
 * tv_xform_compute - Build the sample-to-image coordinate transform
 * @x:      Output transform structure
 * @l:      Current screen layout
 * @v:      Viewport center and zoom state
 * @xsize:  Image width in pixels
 * @ysize:  Image height in pixels
 * @flip_h: True if horizontal mirror is active
 * @flip_v: True if vertical flip is active
 *
 * Combines the layout base step, zoom factor, normalized center coordinates,
 * and flip signs into a transform used by both the image sampler and mouse
 * coordinate mapping.
 *
 * Return: void
 */
void tv_xform_compute(tv_xform_t        *x,
                      const tv_layout_t *l,
                      const tv_view_t   *v,
                      uint32_t           xsize,
                      uint32_t           ysize,
                      bool               flip_h,
                      bool               flip_v)
{
    x->step = tv_view_base(l, xsize, ysize) / v->zoom;
    x->cx   = v->cx * xsize;
    x->cy   = v->cy * ysize;
    x->sx   = flip_h ? -1.0 : 1.0;
    x->sy   = flip_v ? -1.0 : 1.0;
    x->cdx  = l->img_cols / 2.0;
    x->cdy  = l->img_rows; /* half of the 2 * img_rows sample rows */
}

/** tv_grow_buf() - Make *@p hold at least @n elements of @size bytes; -1 on failure. */
static int tv_grow_buf(void **p, size_t *cap, size_t n, size_t size)
{
    if (n <= *cap)
    {
        return 0;
    }
    void *q = realloc(*p, n * size);
    if (q == NULL)
    {
        return -1;
    }
    *p   = q;
    *cap = n;
    return 0;
}

/**
 * tv_image_geom - Precompute column and row sampling lookup tables for the viewport
 * @im:    Image pipeline state whose buffers and index tables are updated
 * @l:     Current screen layout
 * @x:     Current sample-to-image coordinate transform
 * @xsize: Source image width in pixels
 * @ysize: Source image height in pixels
 *
 * Resizes internal sample/cell buffers if needed and populates `col_src`,
 * `col_in`, `row_off`, and `row_in` for the `img_cols` by `2 * img_rows`
 * half-block sample grid.
 *
 * Return: 0 on success, -1 on memory allocation failure
 */
int tv_image_geom(tv_image_t        *im,
                  const tv_layout_t *l,
                  const tv_xform_t  *x,
                  uint32_t           xsize,
                  uint32_t           ysize)
{
    int    nc = l->img_cols;
    int    nr = 2 * l->img_rows;
    size_t ns = (size_t) nc * nr;
    if (tv_grow_buf((void **) &im->col_src, &im->col_cap, (size_t) nc, sizeof(int32_t)) ||
        tv_grow_buf((void **) &im->col_in, &im->colin_cap, (size_t) nc, 1) ||
        tv_grow_buf((void **) &im->row_off, &im->row_cap, (size_t) nr, sizeof(int64_t)) ||
        tv_grow_buf((void **) &im->row_in, &im->rowin_cap, (size_t) nr, 1) ||
        tv_grow_buf((void **) &im->samp, &im->samp_cap, ns, sizeof(float)) ||
        tv_grow_buf((void **) &im->prev, &im->prev_cap, ns, sizeof(float)) ||
        tv_grow_buf((void **) &im->shown, &im->shown_cap, ns, sizeof(uint16_t)) ||
        tv_grow_buf((void **) &im->cells, &im->cells_cap, (size_t) nc * l->img_rows,
                    sizeof(tv_cell_t)))
    {
        return -1;
    }
    /* sample centres, floor(): columns and rows map symmetrically under flips */
    for (int j = 0; j < nc; j++)
    {
        double ix      = floor(tv_xf_img_x(x, j + 0.5) + TV_TIE_EPS);
        int    in      = (ix >= 0.0 && ix < (double) xsize);
        im->col_src[j] = in ? (int32_t) ix : -1;
        im->col_in[j]  = (uint8_t) in;
    }
    for (int i = 0; i < nr; i++)
    {
        double iy      = floor(tv_xf_img_y(x, i + 0.5) + TV_TIE_EPS);
        int    in      = (iy >= 0.0 && iy < (double) ysize);
        im->row_off[i] = in ? (int64_t) iy * xsize : -1;
        im->row_in[i]  = (uint8_t) in;
    }
    im->ncols      = nc;
    im->nrows      = nr;
    im->have_prev  = 0;
    im->have_shown = 0;
    return 0;
}

/**
 * tv_finitef - Test whether a 32-bit IEEE-754 float is finite via bit inspection
 * @v: Float value to test
 *
 * Inspects the exponent bits directly via memcpy() so `-Ofast` / `-ffast-math`
 * cannot optimize away NaN and infinity checks.
 *
 * Return: Non-zero if @v is finite, 0 if NaN or infinite
 */
static inline int tv_finitef(float v)
{
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return (u & 0x7F800000u) != 0x7F800000u;
}

/*
 * One sampler per datatype: nearest sample through the tables, minus @offset, as float.
 * Double values whose distance from the offset exceeds the float range draw with the NaN colour.
 */
#define TV_SAMPLER(DT, ACC, CTYPE)                                                        \
    static MILK_HOT void tv_sample_##ACC(const CTYPE *MILK_RESTRICT data, tv_image_t *im, \
                                         double offset)                                   \
    {                                                                                     \
        const int32_t *MILK_RESTRICT cs   = im->col_src;                                  \
        float                        vmin = FLT_MAX, vmax = -FLT_MAX;                     \
        long                         nvalid = 0;                                          \
        for (int i = 0; i < im->nrows; i++)                                               \
        {                                                                                 \
            float *MILK_RESTRICT row = im->samp + (size_t) i * im->ncols;                 \
            int64_t              ro  = im->row_off[i];                                    \
            if (ro < 0)                                                                   \
            {                                                                             \
                for (int j = 0; j < im->ncols; j++)                                       \
                {                                                                         \
                    row[j] = NAN;                                                         \
                }                                                                         \
                continue;                                                                 \
            }                                                                             \
            const CTYPE *MILK_RESTRICT src = data + ro;                                   \
            for (int j = 0; j < im->ncols; j++)                                           \
            {                                                                             \
                int32_t c = cs[j];                                                        \
                if (UNLIKELY(c < 0))                                                      \
                {                                                                         \
                    row[j] = NAN;                                                         \
                    continue;                                                             \
                }                                                                         \
                float v = (float) ((double) src[c] - offset);                             \
                row[j]  = v;                                                              \
                if (LIKELY(tv_finitef(v)))                                                \
                {                                                                         \
                    vmin = (v < vmin) ? v : vmin;                                         \
                    vmax = (v > vmax) ? v : vmax;                                         \
                    nvalid++;                                                             \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
        im->vmin   = vmin;                                                                \
        im->vmax   = vmax;                                                                \
        im->nvalid = nvalid;                                                              \
    }
FOREACH_REAL_DATATYPE(TV_SAMPLER)
#undef TV_SAMPLER

/** tv_pixel() - Element @idx of @data as double. */
static double tv_pixel(const void *data, uint8_t datatype, size_t idx)
{
    switch (datatype)
    {
#define TV_PIXEL_CASE(DT, ACC, CTYPE) \
    case DT:                          \
        return (double) ((const CTYPE *) data)[idx];
        FOREACH_REAL_DATATYPE(TV_PIXEL_CASE)
#undef TV_PIXEL_CASE
    default:
        return 0.0;
    }
}

/**
 * cmp_double - Compare two double values for qsort() ascending order
 * @a: Pointer to first double
 * @b: Pointer to second double
 *
 * Used by tv_offset() to sort up to 9 finite reference pixels when computing
 * the median offset for wide numeric types.
 *
 * Return: -1 if `*a < *b`, 1 if `*a > *b`, 0 if equal
 */
static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

/** tv_nearest_in() - Index nearest to @k (scanning outwards) whose entry is in the image, or -1. */
static int tv_nearest_in(const uint8_t *in, int n, int k)
{
    for (int d = 0; d < n; d++)
    {
        if (k + d < n && in[k + d])
        {
            return k + d;
        }
        if (k - d >= 0 && in[k - d])
        {
            return k - d;
        }
    }
    return -1;
}

/**
 * tv_offset() - Reference value for wide types: the median of a 3x3 grid of in-view pixels,
 * so one atypical pixel cannot pull the float precision away from the bulk of the data.
 */
static double tv_offset(const tv_image_t *im, const void *data, uint8_t datatype)
{
    double v[9];
    int    n = 0;
    for (int gi = 0; gi < 3; gi++)
    {
        int i = tv_nearest_in(im->row_in, im->nrows, (im->nrows * (2 * gi + 1)) / 6);
        for (int gj = 0; gj < 3 && i >= 0; gj++)
        {
            int j = tv_nearest_in(im->col_in, im->ncols, (im->ncols * (2 * gj + 1)) / 6);
            if (j < 0)
            {
                break;
            }
            double p = tv_pixel(data, datatype, (size_t) im->row_off[i] + (size_t) im->col_src[j]);
            if (tv_finite(p))
            {
                v[n++] = p;
            }
        }
    }
    if (n == 0)
    {
        return 0.0;
    }
    qsort(v, (size_t) n, sizeof(v[0]), cmp_double);
    return v[n / 2];
}

/**
 * tv_image_sample - Sample the source image through the viewport lookup tables
 * @im:       Image pipeline state
 * @data:     Pointer to raw source pixel buffer
 * @datatype: ImageStreamIO datatype code of @data
 *
 * Computes a median reference offset for wide types (32/64-bit integers and
 * doubles), dispatches the typed nearest-neighbor sampler to fill `im->samp`
 * along with `vmin`, `vmax`, and `nvalid`, and compares against `im->prev` to
 * detect whether any visible sample changed.
 *
 * Return: Non-zero if the sampled grid differs from the previous sample, 0 if unchanged
 */
int tv_image_sample(tv_image_t *im, const void *data, uint8_t datatype)
{
    /* wide types lose precision in float: sample relative to an in-view value */
    int wide   = (datatype == _DATATYPE_INT32 || datatype == _DATATYPE_UINT32 ||
                  datatype == _DATATYPE_INT64 || datatype == _DATATYPE_UINT64 ||
                  datatype == _DATATYPE_DOUBLE);
    im->offset = wide ? tv_offset(im, data, datatype) : 0.0;

    float *keep  = im->prev; /* the previous frame becomes the comparison buffer */
    im->prev     = im->samp;
    im->samp     = keep;
    size_t cap   = im->prev_cap;
    im->prev_cap = im->samp_cap;
    im->samp_cap = cap;

    switch (datatype)
    {
#define TV_SAMPLE_CASE(DT, ACC, CTYPE)                         \
    case DT:                                                   \
        tv_sample_##ACC((const CTYPE *) data, im, im->offset); \
        break;
        FOREACH_REAL_DATATYPE(TV_SAMPLE_CASE)
#undef TV_SAMPLE_CASE
    default:
        return 1;
    }

    im->blank    = 0;
    size_t bytes = (size_t) im->ncols * im->nrows * sizeof(float);
    int    changed =
        !(im->have_prev && im->offset == im->prev_offset && memcmp(im->samp, im->prev, bytes) == 0);
    im->have_prev   = 1;
    im->prev_offset = im->offset;
    return changed;
}

/** tv_fkey() - Order-preserving unsigned key of a finite float. */
static inline uint32_t tv_fkey(float v)
{
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/**
 * tv_fkey_inv - Convert an order-preserving unsigned radix key back to a float
 * @k: Unsigned 32-bit key produced by tv_fkey()
 *
 * Inverts the sign-bit/complement mapping of tv_fkey() to recover the exact
 * IEEE-754 float selected by radix selection.
 *
 * Return: Reconstructed float value
 */
static inline float tv_fkey_inv(uint32_t k)
{
    uint32_t u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    float    v;
    memcpy(&v, &u, sizeof(v));
    return v;
}

/**
 * tv_select() - The finite sample of rank @rank (0-based, in sorted order), exactly.
 * Radix select on float keys: 12 + 12 + 8 bits, one counting pass each, no allocation.
 */
static float tv_select(tv_image_t *im, long rank)
{
    static const int shift[3] = { 20, 8, 0 };
    static const int nbins[3] = { 4096, 4096, 256 };
    size_t           ns       = (size_t) im->ncols * im->nrows;
    uint32_t         prefix = 0, pmask = 0;
    for (int pass = 0; pass < 3; pass++)
    {
        memset(im->hist, 0, (size_t) nbins[pass] * sizeof(im->hist[0]));
        for (size_t k = 0; k < ns; k++)
        {
            float v = im->samp[k];
            if (!tv_finitef(v))
            {
                continue;
            }
            uint32_t key = tv_fkey(v);
            if ((key & pmask) == prefix)
            {
                im->hist[(key >> shift[pass]) & (uint32_t) (nbins[pass] - 1)]++;
            }
        }
        long cum = 0;
        int  b   = 0;
        for (; b < nbins[pass] - 1; b++)
        {
            if (cum + (long) im->hist[b] > rank)
            {
                break;
            }
            cum += im->hist[b];
        }
        rank -= cum;
        prefix |= (uint32_t) b << shift[pass];
        pmask |= (uint32_t) (nbins[pass] - 1) << shift[pass];
    }
    return tv_fkey_inv(prefix);
}

/** tv_percentiles() - Samples at fractions @plo, @phi of the finite samples (relative units). */
static void tv_percentiles(tv_image_t *im, double plo, double phi, double *lo, double *hi)
{
    if (!(im->vmax > im->vmin))
    {
        *lo = im->vmin;
        *hi = im->vmax;
        return;
    }
    *lo = tv_select(im, (long) (plo * (im->nvalid - 1)));
    *hi = tv_select(im, (long) (phi * (im->nvalid - 1)));
}

/**
 * tv_image_range - Compute and optionally smooth the active display intensity range
 * @im:   Image pipeline state containing sampled values and previous range
 * @opt:  Viewer options (range mode, manual lock bounds, smoothing flag)
 * @snap: Non-zero to bypass temporal smoothing and jump immediately to target bounds
 *
 * Determines `[im->lo, im->hi]` from manual bounds, min/max, or radix-selected
 * percentiles, applies hysteresis and half-step narrowing when smoothing is
 * enabled, and guarantees a non-degenerate finite interval.
 *
 * Return: void
 */
void tv_image_range(tv_image_t *im, const termview_options_t *opt, int snap)
{
    double lo, hi;
    im->settled = 1;
    if (opt->range_locked)
    {
        lo = opt->manual_min;
        hi = opt->manual_max;
    }
    else
    {
        if (im->nvalid == 0)
        {
            lo = 0.0;
            hi = 1.0;
        }
        else if (opt->range == RANGE_MINMAX)
        {
            lo = im->vmin + im->offset;
            hi = im->vmax + im->offset;
        }
        else
        {
            tv_percentiles(im, tv_range_lo[opt->range], tv_range_hi[opt->range], &lo, &hi);
            lo += im->offset;
            hi += im->offset;
        }

        /* hysteresis: widen at once, narrow halfway per frame, ignore changes below span/64 */
        if (opt->range_smooth && !snap && im->have_range && im->hi > im->lo && tv_finite(lo) &&
            tv_finite(hi))
        {
            double thr = (im->hi - im->lo) / 64.0;
            double dlo = im->lo, dhi = im->hi;
            if (lo < dlo - thr)
            {
                dlo = lo;
            }
            else if (lo > dlo + thr)
            {
                dlo += 0.5 * (lo - dlo);
            }
            if (hi > dhi + thr)
            {
                dhi = hi;
            }
            else if (hi < dhi - thr)
            {
                dhi += 0.5 * (hi - dhi);
            }
            im->settled = (fabs(dlo - lo) <= thr && fabs(dhi - hi) <= thr);
            lo          = dlo;
            hi          = dhi;
        }
    }

    if (!tv_finite(lo) || !tv_finite(hi))
    {
        lo = 0.0;
        hi = 1.0;
    }
    if (!(hi > lo))
    {
        double w = fmax(fabs(lo) * 1e-6, 0.5);
        lo -= w;
        hi += w;
    }
    im->lo         = lo;
    im->hi         = hi;
    im->have_range = 1;
}

/** tv_inb() - Sample (@j, @r) lies inside the image. */
static inline int tv_inb(const tv_image_t *im, int j, int r)
{
    return j >= 0 && j < im->ncols && r >= 0 && r < im->nrows && im->col_in[j] && im->row_in[r];
}

/** tv_near() - An out-of-image sample next to the image: drawn as the frame border. */
static inline int tv_near(const tv_image_t *im, int j, int r)
{
    return tv_inb(im, j - 1, r) || tv_inb(im, j + 1, r) || tv_inb(im, j, r - 1) ||
           tv_inb(im, j, r + 1);
}

/**
 * tv_shown_idx() - LUT index to draw for sample @k. With @hold, a colour change of at most
 * TV_LOWBW_DIST2 keeps what is on screen; the error stays bounded since @idx is always exact.
 */
static inline int tv_shown_idx(tv_image_t *im, const tv_rgb_t *lut, size_t k, int idx, int hold)
{
    if (hold)
    {
        int was = im->shown[k];
        if (idx < TV_LUT_N && was < TV_LUT_N)
        {
            int dr = lut[idx].r - lut[was].r, dg = lut[idx].g - lut[was].g;
            int db = lut[idx].b - lut[was].b;
            if (dr * dr + dg * dg + db * db <= TV_LOWBW_DIST2)
            {
                im->held += (was != idx);
                return was;
            }
        }
    }
    im->shown[k] = (uint16_t) idx;
    return idx;
}

/**
 * tv_image_render - Render sampled float values through the LUT into half-block cells
 * @im:    Image pipeline state with populated samples and range
 * @lut:   Color lookup table of `TV_LUT_N + 1` entries
 * @lowbw: Non-zero to suppress small color changes on non-phase rows
 *
 * Pairs consecutive sample rows into terminal cells using the lower-half-block
 * glyph (`TV_G_LOWER_HALF`), draws a 1-sample grey border immediately outside
 * the image bounds, and fills `im->cells`.
 *
 * Return: void
 */
void tv_image_render(tv_image_t *im, const tv_rgb_t *lut, int lowbw)
{
    static const uint32_t border = TV_RGB(64, 64, 64);

    double lo_rel = im->lo - im->offset;
    double scale  = (TV_LUT_N - 1) / (im->hi - im->lo);
    if (!tv_finite(scale) || !tv_finite(lo_rel))
    {
        scale  = 0.0;
        lo_rel = 0.0;
    }
    tv_cell_t blank = tv_cell(TV_RGB(0, 0, 0), TV_RGB(255, 255, 255), ' ');
    int       nc    = im->ncols;
    lowbw           = lowbw && im->have_shown;
    im->held        = 0;

    for (int i = 0; i < im->nrows / 2; i++)
    {
        int          rt   = 2 * i;
        int          rb   = rt + 1;
        const float *top  = im->samp + (size_t) rt * nc;
        const float *bot  = im->samp + (size_t) rb * nc;
        tv_cell_t   *cell = im->cells + (size_t) i * nc;
        int          hold = lowbw && (i % TV_LOWBW_CYCLE != im->phase); /* round-robin refresh */
        for (int j = 0; j < nc; j++)
        {
            int      it = tv_inb(im, j, rt);
            int      ib = tv_inb(im, j, rb);
            uint32_t ct, cb;
            if (it)
            {
                int k = tv_shown_idx(im, lut, (size_t) rt * nc + j,
                                     tv_lut_idx(top[j], lo_rel, scale), hold);
                ct    = tv_rgb(lut[k]);
            }
            else
            {
                ct = tv_near(im, j, rt) ? border : 0;
            }
            if (ib)
            {
                int k = tv_shown_idx(im, lut, (size_t) rb * nc + j,
                                     tv_lut_idx(bot[j], lo_rel, scale), hold);
                cb    = tv_rgb(lut[k]);
            }
            else
            {
                cb = tv_near(im, j, rb) ? border : 0;
            }
            cell[j] = (!it && !ib && ct == 0 && cb == 0) ? blank : tv_cell(ct, cb, TV_G_LOWER_HALF);
        }
    }
    im->phase      = (im->phase + 1) % TV_LOWBW_CYCLE;
    im->have_shown = 1;
}

/**
 * tv_image_blank - Fill the image cell layer with black space cells
 * @im: Image pipeline state to clear
 *
 * Used when no readable stream or frozen frame is available, resetting sample
 * history flags so the next valid frame triggers a full redraw.
 *
 * Return: void
 */
void tv_image_blank(tv_image_t *im)
{
    tv_cell_t blank = tv_cell(TV_RGB(0, 0, 0), TV_RGB(255, 255, 255), ' ');
    for (size_t k = 0; k < (size_t) im->ncols * (size_t) (im->nrows / 2); k++)
    {
        im->cells[k] = blank;
    }
    im->have_prev  = 0; /* whatever comes next is new */
    im->have_shown = 0;
    im->held       = 0;
    im->blank      = 1;
}

/**
 * tv_image_free - Free all heap buffers owned by an image pipeline state
 * @im: Image pipeline state to release and zero
 *
 * Releases lookup tables, sample buffers, and rendered cell arrays, and resets
 * `*im` to zero.
 *
 * Return: void
 */
void tv_image_free(tv_image_t *im)
{
    free(im->col_src);
    free(im->col_in);
    free(im->row_off);
    free(im->row_in);
    free(im->samp);
    free(im->prev);
    free(im->shown);
    free(im->cells);
    memset(im, 0, sizeof(*im));
}
