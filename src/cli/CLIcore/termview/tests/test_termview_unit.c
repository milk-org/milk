// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file test_termview_unit.c
 * @brief Unit tests for termview helpers. Built with the viewer's flags (Release: -Ofast), so a
 *        pass also proves the NaN/Inf guards survived the optimiser.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

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
}

static void test_fmt_len(void)
{
    TV_CHECK(tv_fmt_len(-1, 16) == 0);
    TV_CHECK(tv_fmt_len(5, 16) == 5);
    TV_CHECK(tv_fmt_len(16, 16) == 15);
    TV_CHECK(tv_fmt_len(300, 16) == 15);
}

static void test_names_unique(void)
{
    const char *const *tables[] = { tv_cmap_names, tv_scale_names, tv_range_names, tv_range_keys };
    int                sizes[]  = { COLORMAP_NB, SCALE_NB, RANGE_NB, RANGE_NB };
    for (int t = 0; t < 4; t++)
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

/* ---- tiny terminal model: just what tv_flush_diff() emits ---- */
typedef struct
{
    int        rows, cols, r, c, sync, moves, errors;
    uint32_t   bg, fg;
    tv_cell_t *grid;
} tv_vt_t;

static void vt_feed(tv_vt_t *vt, const char *buf, size_t n)
{
    size_t i = 0;
    while (i < n)
    {
        unsigned char b = (unsigned char) buf[i];
        if (b == 0x1B)
        {
            size_t j    = i + 2;
            int    priv = 0, np = 0, v = 0, par[8];
            if (i + 1 >= n || buf[i + 1] != '[')
            {
                vt->errors++;
                return;
            }
            if (j < n && buf[j] == '?')
            {
                priv = 1;
                j++;
            }
            for (; j < n && ((buf[j] >= '0' && buf[j] <= '9') || buf[j] == ';'); j++)
            {
                if (buf[j] == ';')
                {
                    par[np < 7 ? np++ : np] = v;
                    v                       = 0;
                }
                else
                {
                    v = v * 10 + (buf[j] - '0');
                }
            }
            par[np < 7 ? np++ : np] = v;
            char fin                = (j < n) ? buf[j] : 0;
            if (priv)
            {
                vt->sync += (par[0] == 2026 && fin == 'h')   ? 1
                            : (par[0] == 2026 && fin == 'l') ? -1
                                                             : 0;
                vt->errors += (par[0] != 2026);
            }
            else if (fin == 'H')
            {
                vt->r = par[0] - 1;
                vt->c = (np > 1 ? par[1] : 1) - 1;
                vt->moves++;
            }
            else if (fin == 'm' && np == 5 && par[0] == 48 && par[1] == 2)
            {
                vt->bg = TV_RGB(par[2], par[3], par[4]);
            }
            else if (fin == 'm' && np == 5 && par[0] == 38 && par[1] == 2)
            {
                vt->fg = TV_RGB(par[2], par[3], par[4]);
            }
            else if (!(fin == 'm' && np == 1 && par[0] == 0))
            {
                vt->errors++;
            }
            i = j + 1;
            continue;
        }

        uint8_t g;
        if (b == 0xE2 && i + 2 < n && (unsigned char) buf[i + 1] == 0x96)
        {
            unsigned char t = (unsigned char) buf[i + 2];
            g               = (t == 0x84) ? TV_G_LOWER_HALF : (t == 0x88) ? TV_G_FULL_BLOCK : 0;
            i += 3;
        }
        else
        {
            g = (b >= 0x20 && b < 0x7F) ? b : 0;
            i++;
        }
        if (g == 0 || vt->r < 0 || vt->r >= vt->rows || vt->c < 0 || vt->c >= vt->cols)
        {
            vt->errors++;
            continue;
        }
        vt->grid[vt->r * vt->cols + vt->c] = tv_cell(vt->bg, vt->fg, g);
        vt->c++;
    }
}

static uint64_t tv_rng = 0x2545F4914F6CDD1DULL;

static uint32_t rnd(void)
{
    tv_rng ^= tv_rng << 13;
    tv_rng ^= tv_rng >> 7;
    tv_rng ^= tv_rng << 17;
    return (uint32_t) (tv_rng >> 16);
}

static tv_cell_t random_cell(void)
{
    static const uint8_t glyphs[] = { TV_G_LOWER_HALF, TV_G_FULL_BLOCK, 'a', 'Z', ' ', '9', '%' };
    return tv_cell(rnd() & 0xFFFFFF, rnd() & 0xFFFFFF, glyphs[rnd() % sizeof(glyphs)]);
}

static void test_flush_roundtrip(void)
{
    enum
    {
        R = 23,
        C = 137
    };
    static tv_cell_t cur[R * C], prev[R * C], screen[R * C];
    size_t           cap = tv_framebuf_size(R, C);
    char            *out = malloc(cap);
    tv_vt_t          vt  = { .rows = R, .cols = C, .grid = screen };

    for (int i = 0; i < R * C; i++)
    {
        cur[i]    = random_cell();
        prev[i]   = TV_CELL_UNKNOWN;
        screen[i] = 0;
    }
    size_t n = tv_flush_diff(out, cur, prev, R, C);
    TV_CHECK(n > 0 && n <= cap);
    vt_feed(&vt, out, n);
    TV_CHECK(vt.errors == 0 && vt.sync == 0);
    TV_CHECK(memcmp(screen, cur, sizeof(cur)) == 0);
    TV_CHECK(memcmp(prev, cur, sizeof(cur)) == 0);

    /* nothing changed: nothing to write */
    TV_CHECK(tv_flush_diff(out, cur, prev, R, C) == 0);

    /* one changed cell: one cursor move */
    cur[7 * C + 100] = tv_cell(0x123456, 0x654321, 'x');
    vt.moves         = 0;
    n                = tv_flush_diff(out, cur, prev, R, C);
    vt_feed(&vt, out, n);
    TV_CHECK(vt.errors == 0 && vt.moves == 1);
    TV_CHECK(memcmp(screen, cur, sizeof(cur)) == 0);
    free(out);
}

static void test_flush_worst_case(void)
{
    const int  rows = 120, cols = 1500;
    size_t     cells = (size_t) rows * cols;
    tv_cell_t *cur   = malloc(cells * sizeof(tv_cell_t));
    tv_cell_t *prev  = malloc(cells * sizeof(tv_cell_t));
    size_t     cap   = tv_framebuf_size(rows, cols);
    char      *out   = malloc(cap);

    /* every cell new, all colours distinct and 3-digit */
    for (size_t i = 0; i < cells; i++)
    {
        cur[i] =
            tv_cell(0xC8C8C8 + (uint32_t) (i & 1), 0xFAFAFA - (uint32_t) (i & 1), TV_G_FULL_BLOCK);
        prev[i] = TV_CELL_UNKNOWN;
    }
    size_t n = tv_flush_diff(out, cur, prev, rows, cols);
    TV_CHECK(n > 0 && n <= cap);

    /* checkerboard of changes: a cursor move before every changed cell */
    for (size_t i = 0; i < cells; i += 2)
    {
        cur[i] = tv_cell(0xFFFFFF - (uint32_t) i % 7, 0xC8C8C8 + (uint32_t) i % 5, TV_G_LOWER_HALF);
    }
    n = tv_flush_diff(out, cur, prev, rows, cols);
    TV_CHECK(n > 0 && n <= cap);
    free(cur);
    free(prev);
    free(out);
}

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
    test_flush_roundtrip();
    test_flush_worst_case();
    if (tv_failures > 0)
    {
        fprintf(stderr, "%d check(s) failed\n", tv_failures);
        return 1;
    }
    printf("termview unit tests passed\n");
    return 0;
}
