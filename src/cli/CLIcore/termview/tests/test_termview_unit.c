// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file test_termview_unit.c
 * @brief Unit tests of the termview helpers: NaN/Inf guards, LUT, name tables, range smoothing
 *        and stream lifecycle.
 *
 * Purpose:     the helpers the viewer relies on, checked as users build them: the tests get the
 *              viewer's flags (Release, the default: -Ofast), so a pass also proves the NaN/Inf
 *              guards survived the optimiser.
 * Does:        checks tv_finite(), tv_lut_idx(), tv_parse_double(), tv_view_sanitize() and
 *              tv_fmt_len(); the name tables (non-empty, unique, exact or unique-prefix lookup);
 *              the LUT (monotonic scalings, the NaN colour for every colormap and scaling); the
 *              range smoothing; a stream removed, created again with another size and type,
 *              then truncated in place.
 * Used by:     ctest, as test termview_unit (label termview), when BUILD_TESTING is on.
 * Uses:        termview_internal.h and the termview_core code under test (tv_cmap.c,
 *              tv_image.c, tv_stream.c); ImageStreamIO, to create the test streams.
 * See also:    docs/guide/termview.md, "Display range" and "Streams that change".
 * Constraints: creates its streams in a new directory under /tmp, set as MILK_SHM_DIR for the
 *              test and restored after.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "termview_internal.h"

static int tv_failures = 0;

#define TV_CHECK(cond)                                                               \
    do                                                                               \
    {                                                                                \
        if (!(cond))                                                                 \
        {                                                                            \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            tv_failures++;                                                           \
        }                                                                            \
    } while (0)

/* volatile keeps the compiler from constant-folding the special values */
static volatile double tv_zero = 0.0;

/**
 * test_finite - Verify IEEE-754 finite float/double detection under optimization
 *
 * Tests tv_finite() against zero, extreme finite values, NaN, and positive and
 * negative infinity constructed via volatile division so -Ofast cannot fold
 * or elide the checks.
 *
 * Return: void
 */
static void test_finite(void)
{
    double nan_v = tv_zero / tv_zero;
    double inf_v = 1.0 / tv_zero;
    TV_CHECK(tv_finite(0.0));
    TV_CHECK(tv_finite(-1e308));
    TV_CHECK(tv_finite(5e-324));
    TV_CHECK(!tv_finite(nan_v));
    TV_CHECK(!tv_finite(inf_v));
    TV_CHECK(!tv_finite(-inf_v));
}

/**
 * test_lut_idx - Verify LUT index mapping, clamping, and NaN/Inf sentinel routing
 *
 * Checks that tv_lut_idx() maps non-finite values to the reserved NaN slot
 * (TV_LUT_N) and clamps out-of-range or overflowing finite values to [0, TV_LUT_N - 1].
 *
 * Return: void
 */
static void test_lut_idx(void)
{
    double nan_v = tv_zero / tv_zero;
    double inf_v = 1.0 / tv_zero;
    double scale = (TV_LUT_N - 1) / 10.0;

    TV_CHECK(tv_lut_idx(nan_v, 0.0, scale) == TV_LUT_N);
    TV_CHECK(tv_lut_idx(inf_v, 0.0, scale) == TV_LUT_N);
    TV_CHECK(tv_lut_idx(-inf_v, 0.0, scale) == TV_LUT_N);
    TV_CHECK(tv_lut_idx(-5.0, 0.0, scale) == 0);
    TV_CHECK(tv_lut_idx(0.0, 0.0, scale) == 0);
    TV_CHECK(tv_lut_idx(10.0, 0.0, scale) == TV_LUT_N - 1);
    TV_CHECK(tv_lut_idx(1e300, 0.0, scale) == TV_LUT_N - 1);
    TV_CHECK(tv_lut_idx(5.0, 0.0, scale) == (TV_LUT_N - 1) / 2);
    /* huge finite inputs whose product overflows must still clamp, not wrap */
    TV_CHECK(tv_lut_idx(1e308, -1e308, 1e10) == TV_LUT_N - 1);
    TV_CHECK(tv_lut_idx(-1e308, 1e308, 1e10) == 0);
    TV_CHECK(tv_lut_idx(1.0, 0.0, 0.0) == 0);
}

/**
 * test_parse_double - Verify strict finite floating-point string parsing
 *
 * Ensures tv_parse_double() accepts valid numbers (with optional trailing
 * whitespace) and rejects NaN, infinities, overflows, trailing garbage, empty
 * strings, and NULL pointers.
 *
 * Return: void
 */
static void test_parse_double(void)
{
    double v = 0.0;
    TV_CHECK(tv_parse_double("1.5", &v) == 0 && v == 1.5);
    TV_CHECK(tv_parse_double("-2e3  ", &v) == 0 && v == -2000.0);
    TV_CHECK(tv_parse_double("nan", &v) != 0);
    TV_CHECK(tv_parse_double("inf", &v) != 0);
    TV_CHECK(tv_parse_double("-infinity", &v) != 0);
    TV_CHECK(tv_parse_double("1e999", &v) != 0);
    TV_CHECK(tv_parse_double("abc", &v) != 0);
    TV_CHECK(tv_parse_double("12abc", &v) != 0);
    TV_CHECK(tv_parse_double("", &v) != 0);
    TV_CHECK(tv_parse_double(NULL, &v) != 0);
}

/**
 * test_view_sanitize - Verify viewport zoom and center coordinate sanitization
 *
 * Confirms that tv_view_sanitize() resets non-finite or non-positive zoom/center
 * parameters to defaults and clamps extreme values to their valid bounds.
 *
 * Return: void
 */
static void test_view_sanitize(void)
{
    double nan_v = tv_zero / tv_zero;
    double inf_v = 1.0 / tv_zero;
    double z, cx, cy;

    z  = nan_v;
    cx = nan_v;
    cy = inf_v;
    tv_view_sanitize(&z, &cx, &cy, 100.0);
    TV_CHECK(z == 1.0 && cx == 0.5 && cy == 0.5);

    z  = 1e9;
    cx = -3.0;
    cy = 7.0;
    tv_view_sanitize(&z, &cx, &cy, 100.0);
    TV_CHECK(z == 100.0 && cx == 0.0 && cy == 1.0);

    z = 1e-9;
    tv_view_sanitize(&z, &cx, &cy, nan_v);
    TV_CHECK(z == 0.1);

    z = -2.0;
    tv_view_sanitize(&z, &cx, &cy, 100.0);
    TV_CHECK(z == 1.0);
}

/**
 * test_names - Verify case-insensitive exact and unambiguous prefix name lookup
 *
 * Tests tv_lookup_name() across colormap, scale, and range tables, checking
 * exact matches, unique prefix matches, ambiguous prefix rejection, and
 * full table round-trips.
 *
 * Return: void
 */
static void test_names(void)
{
    TV_CHECK(tv_lookup_name(tv_cmap_names, COLORMAP_NB, "viridis") == COLORMAP_VIRIDIS);
    TV_CHECK(tv_lookup_name(tv_cmap_names, COLORMAP_NB, "VIRI") == COLORMAP_VIRIDIS);
    TV_CHECK(tv_lookup_name(tv_cmap_names, COLORMAP_NB, "s") == -1); /* spring/summer */
    TV_CHECK(tv_lookup_name(tv_cmap_names, COLORMAP_NB, "nope") == -1);
    TV_CHECK(tv_lookup_name(tv_scale_names, SCALE_NB, "log") == SCALE_LOG); /* exact beats prefix */
    TV_CHECK(tv_lookup_name(tv_range_keys, RANGE_NB, "1") == RANGE_01_99);
    for (int i = 0; i < COLORMAP_NB; i++)
    {
        TV_CHECK(tv_lookup_name(tv_cmap_names, COLORMAP_NB, tv_cmap_names[i]) == i);
    }
    for (int i = 0; i < SCALE_NB; i++)
    {
        TV_CHECK(tv_lookup_name(tv_scale_names, SCALE_NB, tv_scale_names[i]) == i);
    }
    for (int i = 0; i < RANGE_NB; i++)
    {
        TV_CHECK(tv_lookup_name(tv_range_names, RANGE_NB, tv_range_names[i]) == i);
        TV_CHECK(tv_lookup_name(tv_range_keys, RANGE_NB, tv_range_keys[i]) == i);
    }
    for (int i = 0; i < TV_3D_NB; i++)
    {
        TV_CHECK(tv_lookup_name(tv_3d_names, TV_3D_NB, tv_3d_names[i]) == i);
        TV_CHECK(tv_lookup_name(tv_3d_labels, TV_3D_NB, tv_3d_labels[i]) == i);
    }
}

/**
 * test_fmt_len - Verify bounded snprintf return value clamping
 *
 * Checks that tv_fmt_len() maps negative errors to 0 and clamps truncated
 * lengths to `size - 1`.
 *
 * Return: void
 */
static void test_fmt_len(void)
{
    TV_CHECK(tv_fmt_len(-1, 16) == 0);
    TV_CHECK(tv_fmt_len(5, 16) == 5);
    TV_CHECK(tv_fmt_len(16, 16) == 15);
    TV_CHECK(tv_fmt_len(300, 16) == 15);
}

/**
 * test_names_unique - Verify that all entries in option name tables are non-empty and unique
 *
 * Scans colormap, scale, range name, range key, and 3D mode tables to ensure no
 * NULL, empty, or case-insensitively duplicate strings exist.
 *
 * Return: void
 */
static void test_names_unique(void)
{
    const char *const *tables[] = { tv_cmap_names, tv_scale_names, tv_range_names,
                                    tv_range_keys, tv_3d_names,    tv_3d_labels };
    int                sizes[]  = { COLORMAP_NB, SCALE_NB, RANGE_NB, RANGE_NB, TV_3D_NB, TV_3D_NB };
    for (int t = 0; t < 6; t++)
    {
        for (int i = 0; i < sizes[t]; i++)
        {
            TV_CHECK(tables[t][i] != NULL && tables[t][i][0] != '\0');
            for (int j = i + 1; j < sizes[t]; j++)
            {
                TV_CHECK(strcasecmp(tables[t][i], tables[t][j]) != 0);
            }
        }
    }
}

/**
 * test_lut - Verify colormap and intensity scale lookup table generation
 *
 * Checks greyscale endpoints, monotonicity across all scaling curves, and
 * presence of the magenta NaN sentinel color at index TV_LUT_N for every
 * colormap/scale combination.
 *
 * Return: void
 */
static void test_lut(void)
{
    static tv_rgb_t lut[TV_LUT_N + 1];

    tv_lut_build(lut, COLORMAP_GREYSCALE, SCALE_LINEAR);
    TV_CHECK(lut[0].r == 0 && lut[0].g == 0 && lut[0].b == 0);
    TV_CHECK(lut[TV_LUT_N - 1].r == 255 && lut[TV_LUT_N - 1].g == 255);
    for (int sc = 0; sc < SCALE_NB; sc++)
    {
        tv_lut_build(lut, COLORMAP_GREYSCALE, (termview_scale_t) sc);
        int monotonic = 1;
        for (int i = 1; i < TV_LUT_N; i++)
        {
            monotonic &= (lut[i].r >= lut[i - 1].r);
        }
        TV_CHECK(monotonic);
    }
    for (int cm = 0; cm < COLORMAP_NB; cm++)
    {
        for (int sc = 0; sc < SCALE_NB; sc++)
        {
            tv_lut_build(lut, (termview_colormap_t) cm, (termview_scale_t) sc);
            TV_CHECK(lut[TV_LUT_N].r == 255 && lut[TV_LUT_N].g == 0 && lut[TV_LUT_N].b == 255);
        }
    }
}

/** fill_samples() - Image state holding @n samples @v (NaN allowed) on a 1-row grid. */
static void fill_samples(tv_image_t *im, const float *v, int n)
{
    memset(im, 0, sizeof(*im));
    im->samp  = malloc((size_t) n * sizeof(float));
    im->ncols = n;
    im->nrows = 1;
    im->vmin  = 3.4e38f;
    im->vmax  = -3.4e38f;
    for (int k = 0; k < n; k++)
    {
        im->samp[k] = v[k];
        if (tv_finite(v[k]))
        {
            im->vmin = (v[k] < im->vmin) ? v[k] : im->vmin;
            im->vmax = (v[k] > im->vmax) ? v[k] : im->vmax;
            im->nvalid++;
        }
    }
}

/**
 * test_range_smoothing - Verify display range hysteresis and temporal smoothing
 *
 * Tests deadband filtering for small extrema fluctuations, immediate widening
 * on brighter frames, exponential half-step narrowing on dimmer frames, user
 * snap override, and unsmoothed tracking.
 *
 * Return: void
 */
static void test_range_smoothing(void)
{
    termview_options_t opt;
    termview_options_init(&opt);
    tv_image_t im;
    float      v[2] = { 0.0f, 1000.0f };
    fill_samples(&im, v, 2);
    tv_image_range(&im, &opt, 1);
    TV_CHECK(im.lo == 0.0 && im.hi == 1000.0);

    /* noise in the extremes (below span/64) does not move the range at all */
    im.vmin = 3.0f;
    im.vmax = 990.0f;
    tv_image_range(&im, &opt, 0);
    TV_CHECK(im.lo == 0.0 && im.hi == 1000.0);

    /* a brighter frame widens at once, a dimmer one narrows halfway */
    im.vmax = 2000.0f;
    tv_image_range(&im, &opt, 0);
    TV_CHECK(im.hi == 2000.0);
    im.vmax = 1000.0f;
    tv_image_range(&im, &opt, 0);
    TV_CHECK(im.hi == 1500.0);
    TV_CHECK(!im.settled);

    /* a user action snaps; smoothing off follows every frame */
    tv_image_range(&im, &opt, 1);
    TV_CHECK(im.hi == 1000.0 && im.settled);
    opt.range_smooth = false;
    im.vmax          = 1010.0f;
    tv_image_range(&im, &opt, 0);
    TV_CHECK(im.hi == 1010.0);
    free(im.samp);
}

/** mkstream() - Create stream @name in MILK_SHM_DIR (@w keeps it open for writing). */
static int mkstream(IMAGE *w, const char *name, uint32_t xs, uint32_t ys, uint8_t dt)
{
    uint32_t size[2] = { xs, ys };
    memset(w, 0, sizeof(*w));
    return ImageStreamIO_createIm(w, name, 2, size, dt, 1, 0, 0) == IMAGESTREAMIO_SUCCESS ? 0 : -1;
}

static void test_3d_modes(const char *shmdir);

/** Stream lifecycle: removed, created again with another geometry, truncated. */
static void test_stream_lifecycle(void)
{
    char dir[] = "/tmp/tvunit-XXXXXX";
    if (mkdtemp(dir) == NULL)
    {
        TV_CHECK(0);
        return;
    }
    char *saved = getenv("MILK_SHM_DIR") ? strdup(getenv("MILK_SHM_DIR")) : NULL;
    setenv("MILK_SHM_DIR", dir, 1);
    char path[600];
    snprintf(path, sizeof(path), "%s/tvunit_life.im.shm", dir);

    IMAGE       w;
    tv_stream_t s;
    char        err[256];
    TV_CHECK(mkstream(&w, "tvunit_life", 16, 8, _DATATYPE_FLOAT) == 0);
    w.array.F[3] = 7.0f;
    TV_CHECK(tv_stream_open(&s, "tvunit_life", err, sizeof(err)) == 0);
    TV_CHECK(s.xsize == 16 && s.ysize == 8 && s.img.array.F[3] == 7.0f);
    TV_CHECK(tv_stream_check(&s) == TV_STREAM_OK && tv_stream_safe(&s));

    /* removed: no longer the stream, but the mapping stays readable; nothing to reopen yet */
    unlink(path);
    TV_CHECK(tv_stream_check(&s) == TV_STREAM_MOVED && s.img.array.F[3] == 7.0f);
    TV_CHECK(tv_stream_reopen(&s) != 0 && s.map != NULL && s.img.array.F[3] == 7.0f);
    ImageStreamIO_closeIm(&w);

    /* created again with another size and type: the reopen switches over */
    TV_CHECK(mkstream(&w, "tvunit_life", 32, 4, _DATATYPE_UINT16) == 0);
    TV_CHECK(tv_stream_check(&s) == TV_STREAM_MOVED && tv_stream_reopen(&s) == 0);
    TV_CHECK(s.xsize == 32 && s.ysize == 4 && s.datatype == _DATATYPE_UINT16);
    TV_CHECK(tv_stream_check(&s) == TV_STREAM_OK);
    ImageStreamIO_closeIm(&w);

    /* truncated in place: the mapping is dropped before anything reads it */
    TV_CHECK(truncate(path, 0) == 0);
    TV_CHECK(!tv_stream_safe(&s) && s.map == NULL && s.img.md == NULL && s.fd < 0);
    TV_CHECK(tv_stream_check(&s) == TV_STREAM_CLOSED && tv_stream_reopen(&s) != 0);

    unlink(path);
    ImageStreamIO_destroyIm(&w);
    tv_stream_close(&s);

    test_3d_modes(dir);

    rmdir(dir);
    saved ? setenv("MILK_SHM_DIR", saved, 1) : unsetenv("MILK_SHM_DIR");
    free(saved);
}

/**
 * test_3d_modes - Verify 3D layout, virtual geometry, coordinate tables, and sampling
 * @shmdir: Active temporary MILK_SHM_DIR directory.
 *
 * Tests 3D stream opening, slider layout reservation, and sampling across all
 * five 3D viewing modes (slice, gallery, and XY/XZ/YZ 3D->2D remappings).
 *
 * Return: void
 */
static void test_3d_modes(const char *shmdir)
{
    enum
    {
        XS = 4,
        YS = 3,
        ZS = 5
    };
    float cube[XS * YS * ZS];
    for (int z = 0; z < ZS; z++)
    {
        for (int y = 0; y < YS; y++)
        {
            for (int x = 0; x < XS; x++)
            {
                cube[z * (XS * YS) + y * XS + x] = (float) (100 * z + 10 * y + x);
            }
        }
    }

    tv_layout_t l_no, l_sl;
    tv_layout_compute_3d(&l_no, 24, 80, 0);
    tv_layout_compute_3d(&l_sl, 24, 80, 1);
    TV_CHECK(l_no.valid && l_no.slider_row == -1 && l_no.img_rows == 22);
    TV_CHECK(l_sl.valid && l_sl.slider_row == 22 && l_sl.img_rows == 21);

    termview_options_t opt;
    termview_options_init(&opt);
    tv_view_t  view = { .cx = 0.5, .cy = 0.5, .zoom = 1.0 };
    tv_vgeom_t vg;
    tv_xform_t xf;
    tv_image_t im;
    memset(&im, 0, sizeof(im));

    /* 1. Slice mode (slice = 2) */
    opt.mode3d = TV_3D_SLICE;
    opt.slice  = 2;
    tv_vgeom_compute(&vg, &l_sl, XS, YS, ZS, 3, &opt);
    TV_CHECK(vg.width == XS && vg.height == YS && vg.slice == 2);
    tv_xform_compute_3d(&xf, &l_sl, &view, &vg, false, false);
    TV_CHECK(tv_image_geom_3d(&im, &l_sl, &xf, &vg) == 0);
    TV_CHECK(tv_image_sample(&im, cube, _DATATYPE_FLOAT) == 1);
    TV_CHECK(im.nvalid > 0 && im.vmin == 200.0f && im.vmax == 223.0f);

    /* 2. Gallery mode (all 5 slices tiled, with trailing empty slot(s) marked out-of-bounds) */
    opt.mode3d = TV_3D_GALLERY;
    tv_vgeom_compute(&vg, &l_no, XS, YS, ZS, 3, &opt);
    TV_CHECK(vg.gcols * vg.grows >= ZS);
    TV_CHECK(vg.width == vg.gcols * XS && vg.height == vg.grows * YS);
    tv_xform_compute_3d(&xf, &l_no, &view, &vg, false, false);
    TV_CHECK(tv_image_geom_3d(&im, &l_no, &xf, &vg) == 0);
    TV_CHECK(tv_image_sample(&im, cube, _DATATYPE_FLOAT) == 1);
    TV_CHECK(im.nvalid > 0 && im.vmin == 0.0f && im.vmax == 423.0f);

    /* 3. Remap XY -> (X*Y) x Z = 12 x 5 */
    opt.mode3d = TV_3D_REMAP_XY;
    tv_vgeom_compute(&vg, &l_no, XS, YS, ZS, 3, &opt);
    TV_CHECK(vg.width == (uint64_t) XS * YS && vg.height == ZS);
    tv_xform_compute_3d(&xf, &l_no, &view, &vg, false, false);
    TV_CHECK(tv_image_geom_3d(&im, &l_no, &xf, &vg) == 0);
    TV_CHECK(tv_image_sample(&im, cube, _DATATYPE_FLOAT) == 1);
    TV_CHECK(im.nvalid > 0 && im.vmin == 0.0f && im.vmax == 423.0f);

    /* 4. Remap XZ -> (X*Z) x Y = 20 x 3 */
    opt.mode3d = TV_3D_REMAP_XZ;
    tv_vgeom_compute(&vg, &l_no, XS, YS, ZS, 3, &opt);
    TV_CHECK(vg.width == (uint64_t) XS * ZS && vg.height == YS);
    tv_xform_compute_3d(&xf, &l_no, &view, &vg, false, false);
    TV_CHECK(tv_image_geom_3d(&im, &l_no, &xf, &vg) == 0);
    TV_CHECK(tv_image_sample(&im, cube, _DATATYPE_FLOAT) == 1);
    TV_CHECK(im.nvalid > 0 && im.vmin == 0.0f && im.vmax == 423.0f);

    /* 5. Remap YZ -> X x (Y*Z) = 4 x 15 */
    opt.mode3d = TV_3D_REMAP_YZ;
    tv_vgeom_compute(&vg, &l_no, XS, YS, ZS, 3, &opt);
    TV_CHECK(vg.width == XS && vg.height == (uint64_t) YS * ZS);
    tv_xform_compute_3d(&xf, &l_no, &view, &vg, false, false);
    TV_CHECK(tv_image_geom_3d(&im, &l_no, &xf, &vg) == 0);
    TV_CHECK(tv_image_sample(&im, cube, _DATATYPE_FLOAT) == 1);
    TV_CHECK(im.nvalid > 0 && im.vmin == 0.0f && im.vmax == 423.0f);

    /* 5b. Multi-stream 2D sampling (ZS separate 2D buffers sharing gallery & range) */
    const void *srcs[ZS];
    uint8_t     dtypes[ZS];
    for (uint32_t z = 0; z < ZS; z++)
    {
        srcs[z]   = &cube[z * XS * YS];
        dtypes[z] = _DATATYPE_FLOAT;
    }
    opt.mode3d = TV_3D_GALLERY;
    tv_vgeom_compute(&vg, &l_no, XS, YS, ZS, 3, &opt);
    tv_xform_compute_3d(&xf, &l_no, &view, &vg, false, false);
    TV_CHECK(tv_image_geom_3d(&im, &l_no, &xf, &vg) == 0);
    TV_CHECK(tv_image_sample_multi(&im, srcs, dtypes, ZS, (uint64_t) XS * YS) == 1);
    TV_CHECK(im.nvalid > 0 && im.vmin == 0.0f && im.vmax == 423.0f);
    TV_CHECK(tv_image_sample_multi(&im, srcs, dtypes, ZS, (uint64_t) XS * YS) == 0);

    tv_image_free(&im);

    /* 6. 3D stream opening, keyboard/menu actions, and interactive slider mouse controls */
    IMAGE    w3d;
    uint32_t sz3[3] = { XS, YS, ZS };
    memset(&w3d, 0, sizeof(w3d));
    TV_CHECK(ImageStreamIO_createIm(&w3d, "tvunit_cube", 3, sz3, _DATATYPE_FLOAT, 1, 0, 0) ==
             IMAGESTREAMIO_SUCCESS);
    memcpy(w3d.array.F, cube, sizeof(cube));

    tv_stream_t st;
    char        err[256];
    TV_CHECK(tv_stream_open(&st, "tvunit_cube", err, sizeof(err)) == 0);
    TV_CHECK(st.naxis == 3 && st.xsize == XS && st.ysize == YS && st.zsize == ZS);

    tv_state_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.st   = &st;
    ctx.rows = 24;
    ctx.cols = 80;
    ctx.lay  = l_sl;
    termview_options_init(&ctx.opt);
    ctx.screen = calloc((size_t) ctx.rows * ctx.cols, sizeof(tv_cell_t));
    TV_CHECK(ctx.screen != NULL);

    /* Step and clamp slice via tv_do */
    tv_do(&ctx, TV_ACT_SLICE, 1);
    TV_CHECK(ctx.opt.slice == 1);
    tv_do(&ctx, TV_ACT_SET_SLICE, 999);
    TV_CHECK(ctx.opt.slice == ZS - 1);
    tv_do(&ctx, TV_ACT_SET_SLICE, -5);
    TV_CHECK(ctx.opt.slice == 0);

    /* Render slider and test mouse click/drag/wheel on slider row */
    tv_ui_render(&ctx, 0);
    TV_CHECK(ctx.ui.slider_x1 > ctx.ui.slider_x0);

    tv_event_t ev = { .type   = TV_EV_MOUSE,
                      .button = TV_MB_LEFT,
                      .action = TV_MA_PRESS,
                      .x      = ctx.ui.slider_x1 + 1,
                      .y      = ctx.lay.slider_row + 1 };
    TV_CHECK(tv_ui_wants_mouse(&ctx, &ev));
    tv_ui_mouse(&ctx, &ev);
    TV_CHECK(ctx.ui.slider_drag == 1 && ctx.opt.slice == ZS - 1);

    ev.action = TV_MA_DRAG;
    ev.x      = ctx.ui.slider_x0 + 1;
    TV_CHECK(tv_ui_wants_mouse(&ctx, &ev));
    tv_ui_mouse(&ctx, &ev);
    TV_CHECK(ctx.opt.slice == 0);

    ev.action = TV_MA_RELEASE;
    tv_ui_mouse(&ctx, &ev);
    TV_CHECK(ctx.ui.slider_drag == 0);

    /* Wheel up on slider row steps slice forward */
    ev.button = TV_MB_WHEEL_UP;
    ev.action = TV_MA_PRESS;
    TV_CHECK(tv_ui_wants_mouse(&ctx, &ev));
    tv_ui_mouse(&ctx, &ev);
    TV_CHECK(ctx.opt.slice == 1);

    /* Cycle 3D modes via TV_ACT_3D */
    tv_do(&ctx, TV_ACT_3D, 1);
    TV_CHECK(ctx.opt.mode3d == TV_3D_GALLERY);

    free(ctx.screen);
    char path[600];
    snprintf(path, sizeof(path), "%s/tvunit_cube.im.shm", shmdir);
    unlink(path);
    ImageStreamIO_destroyIm(&w3d);
    tv_stream_close(&st);
}

/**
 * main - Run all termview unit tests
 *
 * Executes unit tests for floating-point classification, LUT indexing, number
 * parsing, view sanitization, option name lookup, string length clamping, LUT
 * construction, range smoothing, shared-memory stream lifecycle handling, and
 * 3D viewing modes.
 *
 * Return: 0 if all checks pass, 1 if any check failed
 */
int main(void)
{
    test_finite();
    test_lut_idx();
    test_parse_double();
    test_view_sanitize();
    test_names();
    test_fmt_len();
    test_names_unique();
    test_lut();
    test_range_smoothing();
    test_stream_lifecycle();

    if (tv_failures != 0)
    {
        fprintf(stderr, "termview_unit: %d failure(s)\n", tv_failures);
        return 1;
    }
    return 0;
}
