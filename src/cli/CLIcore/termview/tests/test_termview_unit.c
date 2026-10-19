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

/* ---- tiny terminal model: just what tv_flush_diff() emits, with DECAWM off ---- */
typedef struct
{
    int        rows, cols, r, c, sync, moves, cufs, errors;
    uint32_t   bg, fg; /* codes as sent: RGB at depth 3, palette index otherwise */
    int        set;    /* 1 bg, 2 fg: set since the last SGR 0 (else the terminal's default) */
    tv_cell_t *grid;   /* what is on screen: (bg code, fg code, glyph) */
    uint8_t   *dflt;   /* optional, per cell: drawn in the default bg (1) or fg (2) */
} tv_vt_t;

/** vt_sgr() - Apply SGR parameters @par[0..np). */
static void vt_sgr(tv_vt_t *vt, const int *par, int np)
{
    for (int k = 0; k < np; k++)
    {
        int p = par[k];
        if (p == 0)
        {
            vt->bg = vt->fg = 0;
            vt->set         = 0;
        }
        else if ((p == 48 || p == 38) && k + 4 < np + 0 && par[k + 1] == 2)
        {
            uint32_t rgb                   = TV_RGB(par[k + 2], par[k + 3], par[k + 4]);
            *(p == 48 ? &vt->bg : &vt->fg) = rgb;
            vt->set |= (p == 48) ? 1 : 2;
            k += 4;
        }
        else if ((p == 48 || p == 38) && k + 2 < np && par[k + 1] == 5)
        {
            *(p == 48 ? &vt->bg : &vt->fg) = (uint32_t) par[k + 2];
            vt->set |= (p == 48) ? 1 : 2;
            k += 2;
        }
        else if ((p >= 40 && p <= 47) || (p >= 100 && p <= 107))
        {
            vt->bg = (uint32_t) ((p >= 100) ? p - 100 + 8 : p - 40);
            vt->set |= 1;
        }
        else if ((p >= 30 && p <= 37) || (p >= 90 && p <= 97))
        {
            vt->fg = (uint32_t) ((p >= 90) ? p - 90 + 8 : p - 30);
            vt->set |= 2;
        }
        else
        {
            vt->errors++;
        }
    }
}

static void vt_feed(tv_vt_t *vt, const char *buf, size_t n)
{
    size_t i = 0;
    while (i < n)
    {
        unsigned char b = (unsigned char) buf[i];
        if (b == 0x1B)
        {
            size_t j    = i + 2;
            int    priv = 0, np = 0, v = 0, par[16];
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
                    par[np < 15 ? np++ : np] = v;
                    v                        = 0;
                }
                else
                {
                    v = v * 10 + (buf[j] - '0');
                }
            }
            par[np < 15 ? np++ : np] = v;
            char fin                 = (j < n) ? buf[j] : 0;
            if (priv)
            {
                vt->sync += (par[0] == 2026 && fin == 'h')   ? 1
                            : (par[0] == 2026 && fin == 'l') ? -1
                                                             : 0;
                vt->errors += (par[0] != 2026);
            }
            else if (fin == 'H')
            {
                int r = (par[0] > 0) ? par[0] : 1, c = (np > 1 && par[1] > 0) ? par[1] : 1;
                vt->r = ((r < vt->rows) ? r : vt->rows) - 1; /* clamped to the screen */
                vt->c = ((c < vt->cols) ? c : vt->cols) - 1;
                vt->moves++;
            }
            else if (fin == 'C')
            {
                vt->c += (par[0] > 0) ? par[0] : 1;
                vt->c = (vt->c < vt->cols) ? vt->c : vt->cols - 1; /* stops at the margin */
                vt->cufs++;
            }
            else if (fin == 'm')
            {
                vt_sgr(vt, par, np);
            }
            else
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
        if (g == 0)
        {
            vt->errors++;
            continue;
        }
        vt->grid[vt->r * vt->cols + vt->c] = tv_cell(vt->bg, vt->fg, g);
        if (vt->dflt)
        {
            vt->dflt[vt->r * vt->cols + vt->c] = (uint8_t) (~vt->set & 3);
        }
        if (vt->c < vt->cols - 1)
        {
            vt->c++; /* no autowrap: the last column is overwritten */
        }
    }
}

/**
 * vt_matches() - Every cell of @cur looks as drawn on @vt (undrawn colours ignored, unknown cells
 * skipped); a drawn colour must not be the terminal's default.
 */
static int vt_matches(tv_emit_t *e, const tv_vt_t *vt, const tv_cell_t *cur)
{
    for (int i = 0; i < vt->rows * vt->cols; i++)
    {
        if (cur[i] == TV_CELL_UNKNOWN)
        {
            continue;
        }
        uint32_t bg = tv_emit_code(e, tv_cell_bg(cur[i]));
        uint32_t fg = tv_emit_code(e, tv_cell_fg(cur[i]));
        uint8_t  g  = tv_cell_glyph(cur[i]);
        if (g == TV_G_LOWER_HALF && bg == fg)
        {
            g = ' ';
        }
        tv_cell_t on = vt->grid[i];
        if (tv_cell_glyph(on) != g)
        {
            return 0;
        }
        if (g != TV_G_FULL_BLOCK && tv_cell_bg(on) != (bg & 0xFFFFFF))
        {
            return 0;
        }
        if (g != ' ' && tv_cell_fg(on) != (fg & 0xFFFFFF))
        {
            return 0;
        }
        int d = vt->dflt ? vt->dflt[i] : 0;
        if ((g != TV_G_FULL_BLOCK && (d & 1)) || (g != ' ' && (d & 2)))
        {
            return 0;
        }
    }
    return 1;
}

static uint64_t tv_rng = 0x2545F4914F6CDD1DULL;

static int tv_test_dist2(uint32_t a, uint32_t b)
{
    int dr = (int) ((a >> 16) & 255) - (int) ((b >> 16) & 255);
    int dg = (int) ((a >> 8) & 255) - (int) ((b >> 8) & 255);
    int db = (int) (a & 255) - (int) (b & 255);
    return dr * dr + dg * dg + db * db;
}

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
    uint32_t             bg       = rnd() & 0xFFFFFF;
    uint32_t             fg       = (rnd() % 4 == 0) ? bg : rnd() & 0xFFFFFF;
    return tv_cell(bg, fg, glyphs[rnd() % sizeof(glyphs)]);
}

static void test_flush_roundtrip(void)
{
    enum
    {
        R = 23,
        C = 137
    };
    static tv_cell_t cur[R * C], prev[R * C], screen[R * C];
    static uint8_t   dflt[R * C];
    static tv_emit_t e;
    size_t           cap = tv_framebuf_size(R, C);
    char            *out = malloc(cap);

    for (int level = 3; level >= 1; level--)
    {
        tv_emit_init(&e, level);
        tv_vt_t vt = { .rows = R, .cols = C, .grid = screen, .dflt = dflt };
        for (int i = 0; i < R * C; i++)
        {
            cur[i]    = random_cell();
            prev[i]   = TV_CELL_UNKNOWN;
            screen[i] = 0;
        }
        size_t n = tv_flush_diff(&e, out, cur, prev, R, C);
        TV_CHECK(n > 0 && n <= cap);
        vt_feed(&vt, out, n);
        TV_CHECK(vt.errors == 0 && vt.sync == 0);
        TV_CHECK(vt_matches(&e, &vt, cur));
        TV_CHECK(memcmp(prev, cur, sizeof(cur)) == 0);

        /* nothing changed: nothing to write */
        TV_CHECK(tv_flush_diff(&e, out, cur, prev, R, C) == 0);

        /* scattered changes, same row and across rows: still reproduced exactly */
        for (int k = 0; k < 200; k++)
        {
            cur[rnd() % (R * C)] = random_cell();
        }
        n = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
        TV_CHECK(vt.errors == 0 && vt_matches(&e, &vt, cur));

        /* one changed cell: one cursor move */
        cur[7 * C + 100] = tv_cell(0x123456, 0x654321, 'x');
        vt.moves         = 0;
        n                = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
        TV_CHECK(vt.errors == 0 && vt.moves == 1 && vt_matches(&e, &vt, cur));

        /* white text after the reset that ends a flush: white is sent, not the default colour */
        cur[3 * C + 5] = tv_cell(0x000000, 0xFFFFFF, 'w');
        n              = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
        cur[3 * C + 6] = tv_cell(0x000000, 0xFFFFFF, 'y');
        n              = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
        TV_CHECK(vt.errors == 0 && vt_matches(&e, &vt, cur));

        /* the last column, with autowrap off: drawn in place */
        cur[4 * C + C - 1] = tv_cell(0x445566, 0x112233, TV_G_LOWER_HALF);
        cur[5 * C]         = tv_cell(0x665544, 0x332211, TV_G_LOWER_HALF);
        n                  = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
        TV_CHECK(vt.errors == 0 && vt_matches(&e, &vt, cur));
    }

    /* a different colour that looks the same at this depth: nothing to send */
    tv_emit_init(&e, 2);
    for (int i = 0; i < R * C; i++)
    {
        cur[i] = prev[i] = tv_cell(TV_RGB(0, 0, 200), TV_RGB(0, 0, 200), ' ');
    }
    cur[5] = tv_cell(TV_RGB(0, 0, 201), TV_RGB(255, 0, 0), ' '); /* same blue, fg not drawn */
    TV_CHECK(tv_flush_diff(&e, out, cur, prev, R, C) == 0);

    /* short gaps whose cells need no colour change are re-sent instead of a cursor move */
    tv_emit_init(&e, 3);
    tv_vt_t vt = { .rows = R, .cols = C, .grid = screen };
    for (int i = 0; i < R * C; i++)
    {
        cur[i] = prev[i] = screen[i] = tv_cell(0x101010, 0x202020, TV_G_LOWER_HALF);
    }
    cur[10]  = tv_cell(0x101010, 0x202020, 'a');
    cur[12]  = tv_cell(0x101010, 0x202020, 'b');
    cur[40]  = tv_cell(0x303030, 0x202020, 'c');
    size_t n = tv_flush_diff(&e, out, cur, prev, R, C);
    vt_feed(&vt, out, n);
    TV_CHECK(vt.errors == 0 && vt_matches(&e, &vt, cur) && vt.moves == 1 && vt.cufs == 1);

    /* a 2-cell gap: re-sent when both cells are free to draw, skipped when the second is not */
    for (int pass = 0; pass < 2; pass++)
    {
        cur[C + 12] = prev[C + 12] = screen[C + 12] =
            tv_cell(pass ? 0x505050 : 0x101010, 0x202020, TV_G_LOWER_HALF);
        cur[C + 10] = tv_cell(0x101010, 0x202020, 'd' + pass);
        cur[C + 13] = tv_cell(0x101010, 0x202020, 'f' + pass);
        vt.moves = vt.cufs = 0;
        n                  = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
        TV_CHECK(vt.errors == 0 && vt_matches(&e, &vt, cur) && vt.moves == 1);
        TV_CHECK(vt.cufs == pass);
    }
    free(out);
}

/** A gap re-sent by the interface pass leaves @prev equal to the screen, even under a budget. */
static void test_flush_gap_budget(void)
{
    enum
    {
        R = 2,
        C = 4
    };
    static tv_cell_t cur[R * C], prev[R * C], screen[R * C];
    static tv_emit_t e;
    char            *out = malloc(tv_framebuf_size(R, C));
    tv_emit_init(&e, 3);
    tv_vt_t        vt  = { .rows = R, .cols = C, .grid = screen };
    const uint32_t red = 0xFF0000, green = 0x00FF00, blue = 0x0000FF;
    for (int c = 0; c < C; c++)
    {
        cur[c]  = tv_cell(0x111111 * (uint32_t) (c + 1), 0, TV_G_LOWER_HALF);
        prev[c] = prev[C + c] = TV_CELL_UNKNOWN;
    }
    cur[C + 0] = tv_cell(red, red, ' ') | TV_ATTR_CHROME;
    cur[C + 1] = tv_cell(green, green, ' ');
    cur[C + 2] = tv_cell(red, red, ' ') | TV_ATTR_CHROME;
    cur[C + 3] = tv_cell(green, green, ' ');
    vt_feed(&vt, out, tv_flush_diff(&e, out, cur, prev, R, C));

    /* row 0 uses up the budget; in row 1 the image cell between two interface cells changes */
    for (int c = 0; c < C; c++)
    {
        cur[c] = tv_cell(0x0F0F0F * (uint32_t) (c + 3), 0x123456, TV_G_LOWER_HALF);
    }
    cur[C + 0] = tv_cell(blue, blue, ' ') | TV_ATTR_CHROME;
    cur[C + 1] = tv_cell(blue, blue, ' ');
    cur[C + 2] = tv_cell(blue, blue, ' ') | TV_ATTR_CHROME;
    e.budget   = 16;
    vt_feed(&vt, out, tv_flush_diff(&e, out, cur, prev, R, C));
    TV_CHECK(e.pending && vt.errors == 0 && vt_matches(&e, &vt, prev));

    /* ... and changes back before the image pass reaches it */
    cur[C + 1] = tv_cell(green, green, ' ');
    e.budget   = 0;
    vt_feed(&vt, out, tv_flush_diff(&e, out, cur, prev, R, C));
    TV_CHECK(!e.pending && vt.errors == 0 && vt_matches(&e, &vt, cur));
    free(out);
}

/** Random interface/image changes, reverts, forgotten cells and budgets: the screen is @prev. */
static void test_flush_fuzz(void)
{
    enum
    {
        R = 12,
        C = 40
    };
    static tv_cell_t      cur[R * C], prev[R * C], screen[R * C];
    static uint8_t        dflt[R * C];
    static tv_emit_t      e;
    static const uint32_t pal[4]    = { 0x000000, 0xFF0000, 0x00FF00, 0x0000FF };
    static const uint8_t  glyphs[4] = { TV_G_LOWER_HALF, TV_G_FULL_BLOCK, ' ', 'x' };
    char                 *out       = malloc(tv_framebuf_size(R, C));
    for (int level = 3; level >= 1; level--)
    {
        tv_emit_init(&e, level);
        tv_vt_t vt = { .rows = R, .cols = C, .grid = screen, .dflt = dflt };
        for (int i = 0; i < R * C; i++)
        {
            cur[i]  = tv_cell(pal[rnd() % 4], pal[rnd() % 4], glyphs[rnd() % 4]);
            prev[i] = TV_CELL_UNKNOWN;
        }
        int ok = 1;
        for (int it = 0; it < 2000; it++)
        {
            for (int k = rnd() % 24; k > 0; k--)
            {
                size_t i = rnd() % (R * C);
                if (rnd() % 3 == 0 && prev[i] != TV_CELL_UNKNOWN)
                {
                    cur[i] = prev[i]; /* back to what was sent */
                    continue;
                }
                cur[i] = tv_cell(pal[rnd() % 4], pal[rnd() % 4], glyphs[rnd() % 4]) |
                         ((rnd() % 3 == 0) ? TV_ATTR_CHROME : 0);
            }
            if (rnd() % 16 == 0)
            {
                size_t i  = rnd() % (R * C);
                prev[i]   = TV_CELL_UNKNOWN; /* the terminal lost it */
                screen[i] = tv_cell(0x123456, 0x654321, '?');
            }
            e.budget = (rnd() % 3 == 0) ? 0 : 40 + rnd() % 400;
            size_t n = tv_flush_diff(&e, out, cur, prev, R, C);
            vt_feed(&vt, out, n);
            ok &= vt.errors == 0 && vt.sync == 0 && vt_matches(&e, &vt, prev);
            ok &= e.pending || memcmp(prev, cur, sizeof(cur)) == 0;
        }
        e.budget = 0;
        vt_feed(&vt, out, tv_flush_diff(&e, out, cur, prev, R, C));
        TV_CHECK(ok && vt.errors == 0 && vt_matches(&e, &vt, cur));
    }
    free(out);
}

/** The 256- and 16-colour codes are the nearest colours of xterm's palette (brute force). */
static void test_quantiser(void)
{
    static const uint32_t pal16[16] = {
        0x000000, 0xCD0000, 0x00CD00, 0xCDCD00, 0x0000EE, 0xCD00CD, 0x00CDCD, 0xE5E5E5,
        0x7F7F7F, 0xFF0000, 0x00FF00, 0xFFFF00, 0x5C5CFF, 0xFF00FF, 0x00FFFF, 0xFFFFFF,
    };
    static const int lv[6] = { 0, 95, 135, 175, 215, 255 };
    uint32_t         pal256[256];
    for (int k = 16; k < 232; k++)
    {
        pal256[k] = TV_RGB(lv[(k - 16) / 36], lv[(k - 16) / 6 % 6], lv[(k - 16) % 6]);
    }
    for (int k = 232; k < 256; k++)
    {
        pal256[k] = TV_RGB(8 + 10 * (k - 232), 8 + 10 * (k - 232), 8 + 10 * (k - 232));
    }
    static tv_emit_t e2, e1;
    tv_emit_init(&e2, 2);
    tv_emit_init(&e1, 1);
    TV_CHECK(tv_emit_code(&e2, 0xFFFFFF) == 231 && tv_emit_code(&e2, 0x080808) == 232);
    TV_CHECK(tv_emit_code(&e2, 0xEEEEEE) == 255 && tv_emit_code(&e2, 0x000000) == 16);
    TV_CHECK(tv_emit_code(&e1, 0xFFFFFF) == 15 && tv_emit_code(&e1, 0xCD0000) == 1);

    /* all greys and more random colours than the memo holds, twice: collisions and hits */
    enum
    {
        N = 256 + 4 * TV_QCACHE
    };
    static uint32_t rgb[N];
    for (int i = 0; i < N; i++)
    {
        rgb[i] = (i < 256) ? (uint32_t) i * 0x010101u : rnd() & 0xFFFFFF;
    }
    int bad = 0;
    for (int pass = 0; pass < 2; pass++)
    {
        for (int i = 0; i < N; i++)
        {
            int best2 = INT32_MAX, best1 = INT32_MAX;
            for (int k = 16; k < 256; k++)
            {
                int d = tv_test_dist2(rgb[i], pal256[k]);
                best2 = (d < best2) ? d : best2;
            }
            for (int k = 0; k < 16; k++)
            {
                int d = tv_test_dist2(rgb[i], pal16[k]);
                best1 = (d < best1) ? d : best1;
            }
            uint32_t c2 = tv_emit_code(&e2, rgb[i]), c1 = tv_emit_code(&e1, rgb[i]);
            bad += (c2 < 16 || c2 > 255 || tv_test_dist2(rgb[i], pal256[c2]) != best2);
            bad += (c1 > 15 || tv_test_dist2(rgb[i], pal16[c1]) != best1);
        }
    }
    TV_CHECK(bad == 0);
}

/** Colour depth from the environment: COLORTERM, then 24-bit and xterm-like TERM names. */
static void test_color_level(void)
{
    static const struct
    {
        const char *colorterm, *term;
        int         level;
    } cases[] = {
        { "truecolor", "xterm", 3 },
        { "24bit", NULL, 3 },
        { NULL, "xterm-256color", 2 },
        { NULL, "xterm-kitty", 3 },
        { NULL, "foot", 3 },
        { NULL, "xterm-direct", 3 },
        { NULL, "tmux-256color", 2 },
        { NULL, "screen", 2 },
        { NULL, "xterm", 2 },
        { NULL, "rxvt-unicode", 1 },
        { NULL, "rxvt-unicode-256color", 2 },
        { NULL, "linux", 1 },
        { NULL, "vt100", 1 },
        { NULL, NULL, 1 },
        { "yes", "xterm-256color", 2 },
        { NULL, "dumb", 1 },
        { NULL, "xterm-16color", 1 },
        { NULL, "rxvt-88color", 1 },
        { NULL, "xterm-mono", 1 },
        { NULL, "iTerm2.app", 2 },
        { NULL, "st", 2 },
        { NULL, "mlterm", 2 },
        { NULL, "sun", 1 },
    };
    char *saved_ct = getenv("COLORTERM") ? strdup(getenv("COLORTERM")) : NULL;
    char *saved_t  = getenv("TERM") ? strdup(getenv("TERM")) : NULL;
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++)
    {
        cases[k].colorterm ? setenv("COLORTERM", cases[k].colorterm, 1) : unsetenv("COLORTERM");
        cases[k].term ? setenv("TERM", cases[k].term, 1) : unsetenv("TERM");
        int got = tv_term_color_level();
        if (got != cases[k].level)
        {
            fprintf(stderr, "color level %s/%s: %d\n",
                    cases[k].colorterm ? cases[k].colorterm : "-",
                    cases[k].term ? cases[k].term : "-", got);
        }
        TV_CHECK(got == cases[k].level);
    }
    saved_ct ? setenv("COLORTERM", saved_ct, 1) : unsetenv("COLORTERM");
    saved_t ? setenv("TERM", saved_t, 1) : unsetenv("TERM");
    free(saved_ct);
    free(saved_t);
}

static void test_flush_cost(void)
{
    enum
    {
        R = 40,
        C = 120
    };
    static tv_cell_t cur[R * C], prev[R * C];
    static tv_emit_t e;
    char            *out = malloc(tv_framebuf_size(R, C));

    /* noisy image, every cell new: combined colour escape, about 37 bytes per cell or less */
    tv_emit_init(&e, 3);
    for (int i = 0; i < R * C; i++)
    {
        cur[i]  = tv_cell(rnd() & 0xFFFFFF, rnd() & 0xFFFFFF, TV_G_LOWER_HALF);
        prev[i] = TV_CELL_UNKNOWN;
    }
    double per_cell = (double) tv_flush_diff(&e, out, cur, prev, R, C) / (R * C);
    TV_CHECK(per_cell <= 37.5);

    /* the same at 256 colours: about 23 bytes per cell */
    tv_emit_init(&e, 2);
    for (int i = 0; i < R * C; i++)
    {
        prev[i] = TV_CELL_UNKNOWN;
    }
    TV_CHECK((double) tv_flush_diff(&e, out, cur, prev, R, C) / (R * C) < 0.7 * per_cell);

    /* flat regions (both halves alike) cost a space and the occasional background change */
    tv_emit_init(&e, 3);
    for (int i = 0; i < R * C; i++)
    {
        uint32_t v = (i / 8) * 0x010101u & 0xFFFFFF;
        cur[i]     = tv_cell(v, v, TV_G_LOWER_HALF);
        prev[i]    = TV_CELL_UNKNOWN;
    }
    TV_CHECK((double) tv_flush_diff(&e, out, cur, prev, R, C) / (R * C) < 4.0);
    free(out);
}

/** A byte budget: interface first, image over several calls, no row starved. */
static void test_flush_budget(void)
{
    enum
    {
        R = 30,
        C = 100
    };
    static tv_cell_t cur[R * C], prev[R * C], screen[R * C];
    static tv_emit_t e;
    char            *out = malloc(tv_framebuf_size(R, C));
    tv_emit_init(&e, 3);
    tv_vt_t vt = { .rows = R, .cols = C, .grid = screen };
    for (int i = 0; i < R * C; i++)
    {
        cur[i]  = tv_cell(rnd() & 0xFFFFFF, rnd() & 0xFFFFFF, TV_G_LOWER_HALF);
        prev[i] = TV_CELL_UNKNOWN;
    }
    for (int c = 0; c < C; c++)
    {
        cur[(R - 1) * C + c] =
            tv_cell(0x141620, 0xC8CDD6, (uint8_t) ('a' + c % 26)) | TV_ATTR_CHROME;
    }
    e.budget = 4000;
    size_t n = tv_flush_diff(&e, out, cur, prev, R, C);
    vt_feed(&vt, out, n);
    TV_CHECK(e.pending && n < 4000 + 6000);
    int chrome_ok = 1;
    for (int c = 0; c < C; c++)
    {
        chrome_ok &= (tv_cell_glyph(screen[(R - 1) * C + c]) == 'a' + c % 26);
    }
    TV_CHECK(chrome_ok); /* the whole status row went out with the first budgeted flush */

    int calls = 1;
    while (e.pending && calls < 100)
    {
        n = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
        calls++;
    }
    TV_CHECK(!e.pending && calls > 5 && vt.errors == 0 && vt_matches(&e, &vt, cur));

    /* the top rows change before every call: the bottom rows still get their turn */
    for (int i = 0; i < R * C; i++)
    {
        cur[i] = tv_cell(rnd() & 0xFFFFFF, rnd() & 0xFFFFFF, TV_G_LOWER_HALF);
    }
    for (calls = 0; calls < 200; calls++)
    {
        for (int i = 0; i < 3 * C; i++)
        {
            cur[i] = tv_cell(rnd() & 0xFFFFFF, rnd() & 0xFFFFFF, TV_G_LOWER_HALF);
        }
        n = tv_flush_diff(&e, out, cur, prev, R, C);
        vt_feed(&vt, out, n);
    }
    int bottom_ok = 1;
    for (int i = 3 * C; i < R * C; i++)
    {
        bottom_ok &= (prev[i] == cur[i]);
    }
    TV_CHECK(bottom_ok);

    /* a popup closes while the whole image changes: it is gone after the next budgeted flush */
    e.budget = 0;
    for (int i = 0; i < R * C; i++)
    {
        int in = (i / C >= 5 && i / C < 10 && i % C >= 10 && i % C < 50);
        cur[i] = in ? (tv_cell(0x1E2028, 0xC8CDD6, 'p') | TV_ATTR_CHROME) : cur[i];
    }
    vt_feed(&vt, out, tv_flush_diff(&e, out, cur, prev, R, C));
    for (int i = 0; i < R * C; i++)
    {
        cur[i] = tv_cell(rnd() & 0xFFFFFF, rnd() & 0xFFFFFF, TV_G_LOWER_HALF);
    }
    e.budget = 4000;
    vt_feed(&vt, out, tv_flush_diff(&e, out, cur, prev, R, C));
    int closed = 1;
    for (int i = 0; i < R * C; i++)
    {
        int in = (i / C >= 5 && i / C < 10 && i % C >= 10 && i % C < 50);
        closed &= !in || prev[i] == cur[i];
    }
    TV_CHECK(e.pending && closed && vt.errors == 0 && vt_matches(&e, &vt, prev));

    /* held (flow control): interface cells go, the image waits where the rotation stopped */
    int resume       = e.resume_row;
    e.hold           = 1;
    cur[(R - 1) * C] = tv_cell(0x141620, 0xC8CDD6, 'Z') | TV_ATTR_CHROME;
    n                = tv_flush_diff(&e, out, cur, prev, R, C);
    vt_feed(&vt, out, n);
    TV_CHECK(n > 0 && n < 200 && prev[(R - 1) * C] == cur[(R - 1) * C]);
    TV_CHECK(e.pending && e.resume_row == resume);
    TV_CHECK(tv_flush_diff(&e, out, cur, prev, R, C) == 0 && e.pending && e.resume_row == resume);
    e.hold = 0;
    free(out);
}

static void test_flush_worst_case(void)
{
    const int        rows = 120, cols = 1500;
    size_t           cells = (size_t) rows * cols;
    tv_cell_t       *cur   = malloc(cells * sizeof(tv_cell_t));
    tv_cell_t       *prev  = malloc(cells * sizeof(tv_cell_t));
    size_t           cap   = tv_framebuf_size(rows, cols);
    char            *out   = malloc(cap);
    static tv_emit_t e;

    for (int level = 3; level >= 1; level--)
    {
        tv_emit_init(&e, level);
        /* every cell new, both colours changing from cell to cell */
        for (size_t i = 0; i < cells; i++)
        {
            cur[i]  = tv_cell(0xC8C8C8 + (uint32_t) (i & 1), 0xFAFAFA - (uint32_t) (i & 1),
                              TV_G_LOWER_HALF);
            prev[i] = TV_CELL_UNKNOWN;
        }
        size_t n = tv_flush_diff(&e, out, cur, prev, rows, cols);
        TV_CHECK(n > 0 && n <= cap);

        /* checkerboard of changes: a cursor move before every changed cell */
        for (size_t i = 0; i < cells; i += 2)
        {
            cur[i] =
                tv_cell(0xFFFFFF - (uint32_t) i % 7, 0xC8C8C8 + (uint32_t) i % 5, TV_G_LOWER_HALF);
        }
        n = tv_flush_diff(&e, out, cur, prev, rows, cols);
        TV_CHECK(n > 0 && n <= cap);
    }
    free(cur);
    free(prev);
    free(out);
}

/* ---- input decoder ---- */

/** tv_ev_str() - Compact text form of an event, appended to @out. */
static void tv_ev_str(const tv_event_t *ev, char *out, size_t cap)
{
    size_t len = strlen(out);
    if (ev->type == TV_EV_KEY)
    {
        snprintf(out + len, cap - len, "K%u/%u ", ev->key, ev->mods);
    }
    else if (ev->type == TV_EV_CPR)
    {
        snprintf(out + len, cap - len, "P%d,%d ", ev->y, ev->x);
    }
    else
    {
        snprintf(out + len, cap - len, "M%u/%u/%d,%d/%u ", ev->button, ev->action, ev->x, ev->y,
                 ev->mods);
    }
}

/**
 * decode_chunked() - Feed @bytes in chunks (sizes from @chunk, cycling; NULL = all at once) with
 * @gap_us between chunks, also polling just before each next chunk; then let the ESC timeout
 * expire. @early receives the events produced before that final expiry.
 */
static void decode_chunked(const char *bytes,
                           size_t      n,
                           const int  *chunk,
                           int         nchunk,
                           int         gap_us,
                           char       *out,
                           size_t      cap,
                           char       *early)
{
    tv_input_t in;
    tv_event_t ev;
    int64_t    t = 1000000;
    memset(&in, 0, sizeof(in));
    out[0] = '\0';
    for (size_t off = 0, k = 0; off < n; k++)
    {
        size_t c = (nchunk == 0) ? n - off : (size_t) chunk[k % nchunk];
        c        = (c > n - off) ? n - off : c;
        c        = (c > tv_input_room(&in)) ? tv_input_room(&in) : c;
        tv_input_feed(&in, (const unsigned char *) bytes + off, c, t);
        off += c;
        while (tv_input_next(&in, t, &ev))
        {
            tv_ev_str(&ev, out, cap);
        }
        TV_CHECK(tv_input_room(&in) > 0);
        if (gap_us > 1)
        {
            while (tv_input_next(&in, t + gap_us - 1, &ev))
            {
                tv_ev_str(&ev, out, cap);
            }
        }
        t += gap_us;
    }
    if (early != NULL)
    {
        snprintf(early, cap, "%s", out);
    }
    t += TV_ESC_TIMEOUT_US;
    while (tv_input_next(&in, t, &ev))
    {
        tv_ev_str(&ev, out, cap);
    }
    TV_CHECK(in.len == 0);
}

static void expect_events(const char *what, const char *in, size_t n, const char *want, int now)
{
    char out[4096], early[4096];
    decode_chunked(in, n, NULL, 0, 0, out, sizeof(out), early);
    if (strcmp(out, want) != 0 || (now && strcmp(early, want) != 0))
    {
        fprintf(stderr, "%s: got '%s' (before timeout '%s') want '%s'\n", what, out, early, want);
        tv_failures++;
    }
}

static void test_input_vectors(void)
{
    static const struct
    {
        const char *in;
        const char *want; /* "" means: no event at all */
    } plain[] = {
        { "\x1b[<128;1;1M", "" },
        { "\x1b[<64;10;5m", "" },
        { "\x1b[M !!", "" },
        { "\x1b[M#q!", "" },
        { "\x1b[I", "" },
        { "\x1b[O", "" },
        { "\x1b[200~", "" },
        { "\xc3\xa9", "" },
        { "\xe2\x96\x84", "" },
        { "\x1b[?62;22c", "" },
        { "\x1b[?2026;2$y", "" },
        { "\x1bOz", "" },
        { "\x1b]11;rgb:0000/0000/0000\x07q", "K113/0 " },
        { "\x1bP1$r0m\x1b\\q", "K113/0 " },
        { "\x1b[5$q", "K113/0 " },
        { "q", "K113/0 " },
        { "\x03", "K3/0 " },
        { "\r", "K13/0 " },
        { "\x7f", "K127/0 " },
        { "\x1b"
          "c",
          "K99/2 " },
        { "\x1b[<0;67;12M", "M0/0/67,12/0 " },
        { "\x1b[<0;67;12m", "M0/1/67,12/0 " },
        { "\x1b[<32;300;80M", "M0/2/300,80/0 " },
        { "\x1b[<64;10;5M", "M64/0/10,5/0 " },
        { "\x1b[<65;10;5M", "M65/0/10,5/0 " },
        { "\x1b[<80;10;5M", "M64/0/10,5/4 " },
        { "\x1b[<66;3;3M", "M66/0/3,3/0 " },
        { "\x1b[<67;3;3M", "M67/0/3,3/0 " },
        { "\x1b[<4;2;2M", "M0/0/2,2/1 " },
        { "\x1b[<8;2;2M", "M0/0/2,2/2 " },
        { "\x1b[<35;5;5M", "M3/3/5,5/0 " },
        { "\x1b[<2;0;0m", "M2/1/1,1/0 " },
        { "\x1b[90;300R", "P90,300 " }, /* answers to cursor position requests */
        { "\x1b[2;1R", "P2,1 " },
        { "\x1b[999;9999R", "P999,9999 " },
        { "\x1b[1;120R", "P1,120 " }, /* the far terminal 1 row high: still past column 16 */
        { "\x1b[1;17R", "P1,17 " },
        { "\x1b[2;99999R", "P2,32767 " },
    };
    for (size_t i = 0; i < sizeof(plain) / sizeof(plain[0]); i++)
    {
        char what[32];
        snprintf(what, sizeof(what), "vector %zu", i);
        expect_events(what, plain[i].in, strlen(plain[i].in), plain[i].want, 1);
    }

    /* special keys, emitted as soon as complete */
    struct
    {
        const char *in;
        uint32_t    key;
        unsigned    mods;
    } keys[] = {
        { "\x1b[A", TV_K_UP, 0 },
        { "\x1b[B", TV_K_DOWN, 0 },
        { "\x1b[C", TV_K_RIGHT, 0 },
        { "\x1b[D", TV_K_LEFT, 0 },
        { "\x1bOA", TV_K_UP, 0 },
        { "\x1bOB", TV_K_DOWN, 0 },
        { "\x1bOC", TV_K_RIGHT, 0 },
        { "\x1bOD", TV_K_LEFT, 0 },
        { "\x1b[1;5A", TV_K_UP, 4 },
        { "\x1b[1;3A", TV_K_UP, 2 },
        { "\x1b[1;6C", TV_K_RIGHT, 5 },
        { "\x1b[5~", TV_K_PGUP, 0 },
        { "\x1b[6~", TV_K_PGDN, 0 },
        { "\x1b[3~", TV_K_DEL, 0 },
        { "\x1b[2~", TV_K_INS, 0 },
        { "\x1b[1~", TV_K_HOME, 0 },
        { "\x1b[7~", TV_K_HOME, 0 },
        { "\x1b[4~", TV_K_END, 0 },
        { "\x1b[8~", TV_K_END, 0 },
        { "\x1b[H", TV_K_HOME, 0 },
        { "\x1b[F", TV_K_END, 0 },
        { "\x1bOP", TV_K_F1, 0 },
        { "\x1b[1;2P", TV_K_F1, 1 },
        { "\x1b[R", TV_K_F1 + 2, 0 }, /* F3, not a cursor position report: row 1 */
        { "\x1b[1;2R", TV_K_F1 + 2, 1 },
        { "\x1b[1;5R", TV_K_F1 + 2, 4 },
        { "\x1b[1;3R", TV_K_F1 + 2, 2 },
        { "\x1b[1;16R", TV_K_F1 + 2, 7 },
        { "\x1b[2R", TV_K_F1 + 2, 1 }, /* xterm modifyFunctionKeys:1 */
        { "\x1b[5R", TV_K_F1 + 2, 4 },
        { "\x1b[3P", TV_K_F1, 2 },
        { "\x1b[[A", TV_K_F1, 0 },
        { "\x1b[11~", TV_K_F1, 0 },
        { "\x1b[15~", TV_K_F1 + 4, 0 },
        { "\x1b[15;2~", TV_K_F1 + 4, 1 },
        { "\x1b[17~", TV_K_F1 + 5, 0 },
        { "\x1b[18~", TV_K_F1 + 6, 0 },
        { "\x1b[19~", TV_K_F1 + 7, 0 },
        { "\x1b[20~", TV_K_F1 + 8, 0 },
        { "\x1b[21~", TV_K_F1 + 9, 0 },
        { "\x1b[23~", TV_K_F1 + 10, 0 },
        { "\x1b[24~", TV_K_F1 + 11, 0 },
        { "\x1b[Z", '\t', 1 },
        { "\x1bOk", '+', 0 },
        { "\x1bOM", '\r', 0 },
        { "\x1bO2P", TV_K_F1, 1 },
        { "\x1bO5A", TV_K_UP, 4 },
        { "\x1bO\x1b[A", TV_K_UP, 0 },
        { "\x1b[[\x1b[A", TV_K_UP, 0 },
    };
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++)
    {
        char want[32], what[32];
        snprintf(want, sizeof(want), "K%u/%u ", keys[i].key, keys[i].mods);
        snprintf(what, sizeof(what), "key %zu", i);
        expect_events(what, keys[i].in, strlen(keys[i].in), want, 1);
    }

    /* lone ESC: only once the timeout expires; ESC ESC [A is ESC then Up */
    char esc[16], esc_up[48];
    snprintf(esc, sizeof(esc), "K%u/0 ", TV_K_ESC);
    snprintf(esc_up, sizeof(esc_up), "K%u/0 K%u/0 ", TV_K_ESC, TV_K_UP);
    expect_events("lone ESC", "\x1b", 1, esc, 0);
    expect_events("ESC ESC [A", "\x1b\x1b[A", 4, esc_up, 1);
}

/** Timing rules that depend on when bytes arrive and when the loop looks at them. */
static void test_input_timing(void)
{
    tv_input_t in;
    tv_event_t ev;
    char       out[1024];

    /* deadline: none when idle, set while a prefix waits, cleared once resolved */
    memset(&in, 0, sizeof(in));
    TV_CHECK(tv_input_deadline(&in) == -1);
    tv_input_feed(&in, (const unsigned char *) "\x1b[", 2, 1000);
    TV_CHECK(tv_input_deadline(&in) == 1000 + TV_ESC_TIMEOUT_US);
    TV_CHECK(!tv_input_next(&in, 2000, &ev));
    tv_input_feed(&in, (const unsigned char *) "A", 1, 3000);
    TV_CHECK(tv_input_next(&in, 3000, &ev) && ev.key == TV_K_UP);
    TV_CHECK(tv_input_deadline(&in) == -1);

    /* Esc, then a key read 80 ms later without the loop looking in between: two keys */
    memset(&in, 0, sizeof(in));
    out[0] = '\0';
    tv_input_feed(&in, (const unsigned char *) "\x1b", 1, 0);
    tv_input_feed(&in, (const unsigned char *) "q", 1, 80000);
    while (tv_input_next(&in, 80000, &ev))
    {
        tv_ev_str(&ev, out, sizeof(out));
    }
    char want[64];
    snprintf(want, sizeof(want), "K%u/0 K113/0 ", TV_K_ESC);
    TV_CHECK(strcmp(out, want) == 0);

    /* ... but ESC + q within the timeout is Alt-q */
    memset(&in, 0, sizeof(in));
    out[0] = '\0';
    tv_input_feed(&in, (const unsigned char *) "\x1b", 1, 0);
    tv_input_feed(&in, (const unsigned char *) "q", 1, 10000);
    while (tv_input_next(&in, 10000, &ev))
    {
        tv_ev_str(&ev, out, sizeof(out));
    }
    TV_CHECK(strcmp(out, "K113/2 ") == 0);

    /* a mouse report whose tail is read late still joins into one event */
    memset(&in, 0, sizeof(in));
    out[0] = '\0';
    tv_input_feed(&in, (const unsigned char *) "\x1b[<0;12", 7, 0);
    tv_input_feed(&in, (const unsigned char *) ";5M", 3, 80000);
    while (tv_input_next(&in, 80000, &ev))
    {
        tv_ev_str(&ev, out, sizeof(out));
    }
    TV_CHECK(strcmp(out, "M0/0/12,5/0 ") == 0);

    /* over-long CSI cut by a read: the skip ends at the next sequence, which is not delayed */
    char cut[80] = "\x1b[";
    memset(cut + 2, '1', 70);
    memset(&in, 0, sizeof(in));
    out[0] = '\0';
    tv_input_feed(&in, (const unsigned char *) cut, 72, 0);
    TV_CHECK(!tv_input_next(&in, 0, &ev) && tv_input_deadline(&in) == TV_ESC_TIMEOUT_US);
    tv_input_feed(&in, (const unsigned char *) "\x1b[A", 3, 1000);
    TV_CHECK(tv_input_next(&in, 1000, &ev) && ev.key == TV_K_UP);

    /* ... a full chunk after it never leaves the buffer full */
    unsigned char full[256];
    for (size_t k = 0; k < sizeof(full); k++)
    {
        full[k] = (k % 4 == 0) ? 0x1B : (k % 4 == 1) ? '[' : (k % 4 == 2) ? 'A' : 'x';
    }
    memset(&in, 0, sizeof(in));
    tv_input_feed(&in, (const unsigned char *) cut, 72, 0);
    tv_input_next(&in, 0, &ev);
    tv_input_feed(&in, full, sizeof(full), 1000);
    int nev = 0;
    while (tv_input_next(&in, 1000, &ev))
    {
        nev++;
    }
    TV_CHECK(nev == 128 && tv_input_room(&in) > 0);

    /* ... and when the loop wakes at its deadline, the skip expires and later keys are keys */
    memset(&in, 0, sizeof(in));
    tv_input_feed(&in, (const unsigned char *) cut, 72, 0);
    tv_input_next(&in, 0, &ev);
    TV_CHECK(!tv_input_next(&in, TV_ESC_TIMEOUT_US, &ev) && tv_input_deadline(&in) == -1);
    tv_input_feed(&in, (const unsigned char *) "+q", 2, 5000000);
    TV_CHECK(tv_input_next(&in, 5000000, &ev) && ev.key == '+');

    /* a 300-byte CSI fed as fast as the buffer allows: bounded, dropped, next key intact */
    static char huge[320];
    huge[0] = 0x1B;
    huge[1] = '[';
    memset(huge + 2, '9', 300);
    huge[302]   = 'A';
    huge[303]   = 'q';
    int ones[1] = { 256 };
    decode_chunked(huge, 304, ones, 1, 0, out, sizeof(out), NULL);
    TV_CHECK(strcmp(out, "K113/0 ") == 0);

    /* an unterminated 200-byte OSC string, then BEL and a key, in small chunks */
    static char osc[210];
    osc[0] = 0x1B;
    osc[1] = ']';
    memset(osc + 2, 'x', 200);
    osc[202]      = 0x07;
    osc[203]      = 'q';
    int sevens[1] = { 7 };
    decode_chunked(osc, 204, sevens, 1, 1000, out, sizeof(out), NULL);
    TV_CHECK(strcmp(out, "K113/0 ") == 0);
}

/** All vectors back to back: any split with gaps below the ESC timeout decodes the same. */
static void test_input_splits(void)
{
    static const char stream[] = "ab\x1b[A\x1b[<0;67;12M\x1b[<32;70;12M\x1b[<0;70;12m"
                                 "\x1b[M#q!\x1b[15;2~\x1bOP\x1b[<64;1;1Mz\xc3\xa9\x1b[1;5Dq"
                                 "\x1b[?62;22c\x1b[[B\x1b[200~x\x1b[201~\x03"
                                 "\x1b[40;120R\x1b[1;5R";
    size_t            n        = sizeof(stream) - 1;
    char              want[512], got[4096], early[4096];

    snprintf(want, sizeof(want),
             "K97/0 K98/0 K%u/0 M0/0/67,12/0 M0/2/70,12/0 M0/1/70,12/0 K%u/1 K%u/0 "
             "M64/0/1,1/0 K122/0 K%u/4 K113/0 K%u/0 K120/0 K3/0 P40,120 K%u/4 ",
             TV_K_UP, TV_K_F1 + 4, TV_K_F1, TV_K_LEFT, TV_K_F1 + 1, TV_K_F1 + 2);
    decode_chunked(stream, n, NULL, 0, 0, got, sizeof(got), early);
    TV_CHECK(strcmp(got, want) == 0 && strcmp(early, want) == 0);

    for (int c = 1; c <= 9; c++)
    {
        int chunk[1] = { c };
        decode_chunked(stream, n, chunk, 1, 49000, got, sizeof(got), early);
        if (strcmp(got, want) != 0 || strcmp(early, want) != 0)
        {
            fprintf(stderr, "split %d differs:\n  %s\n  %s\n", c, early, want);
            tv_failures++;
        }
    }
    for (int trial = 0; trial < 200; trial++)
    {
        int chunk[16];
        for (int k = 0; k < 16; k++)
        {
            chunk[k] = 1 + (int) (rnd() % 12);
        }
        decode_chunked(stream, n, chunk, 16, (int) (rnd() % 40000), got, sizeof(got), early);
        if (strcmp(got, want) != 0 || strcmp(early, want) != 0)
        {
            fprintf(stderr, "random split %d differs\n", trial);
            tv_failures++;
            break;
        }
    }
}

/** Structured fuzz: fragments of real sequences, truncated and mixed with noise. */
static void test_input_fuzz(void)
{
    static const char *const frags[] = {
        "\x1b[<0;10;5M",
        "\x1b[<32;11;5M",
        "\x1b[<0;11;5m",
        "\x1b[<64;3;3M",
        "\x1b[M #!",
        "\x1bOA",
        "\x1bO5P",
        "\x1b[15;2~",
        "\x1b[1;5C",
        "\x1b]0;title\x07",
        "\x1bP1$r\x1b\\",
        "\xe2\x96\x84",
        "q",
        "\x1b",
        "\x1b[",
        "\x1b[?1;2c",
        "\x1b[5$",
        "\x07",
        "\x1b[[C",
    };
    tv_input_t in;
    tv_event_t ev;
    char       buf[512];
    int64_t    t    = 0;
    long       mice = 0, keys = 0, skips = 0;
    memset(&in, 0, sizeof(in));
    for (long total = 0; total < 1000000;)
    {
        size_t c = 0;
        while (c < 256)
        {
            uint32_t r = rnd() % 10;
            if (r < 6)
            {
                const char *f  = frags[rnd() % (sizeof(frags) / sizeof(frags[0]))];
                size_t      fl = strlen(f);
                fl             = (rnd() % 5 == 0) ? rnd() % (fl + 1) : fl; /* truncate some */
                memcpy(buf + c, f, fl);
                c += fl;
            }
            else if (r < 7)
            {
                size_t run = 60 + rnd() % 240; /* over-long parameter runs */
                buf[c++]   = 0x1B;
                buf[c++]   = (rnd() & 1) ? '[' : ']';
                for (size_t k = 0; k < run && c < sizeof(buf); k++)
                {
                    buf[c++] = (char) ('0' + rnd() % 10);
                }
            }
            else
            {
                buf[c++] = (char) rnd();
            }
        }
        for (size_t off = 0; off < c;)
        {
            size_t part = 1 + rnd() % 300;
            part        = (part > c - off) ? c - off : part;
            part        = (part > tv_input_room(&in)) ? tv_input_room(&in) : part;
            tv_input_feed(&in, (const unsigned char *) buf + off, part, t);
            off += part;
            total += (long) part;
            t += (rnd() % 4 == 0) ? 60000 : rnd() % 20000;
            while (tv_input_next(&in, t, &ev))
            {
                if (ev.type == TV_EV_MOUSE)
                {
                    mice++;
                    TV_CHECK(ev.x >= 1 && ev.y >= 1);
                }
                else
                {
                    keys++;
                    TV_CHECK(ev.type == TV_EV_KEY);
                }
            }
            skips += (in.skip != 0);
            TV_CHECK(tv_input_room(&in) > 0);
        }
    }
    TV_CHECK(mice > 1000 && keys > 1000 && skips > 100);
}

/* ---- actions and input handlers on a viewer state ---- */

static tv_stream_t tv_fake_stream;

static void state_init(tv_state_t *s)
{
    memset(s, 0, sizeof(*s));
    memset(&tv_fake_stream, 0, sizeof(tv_fake_stream));
    tv_fake_stream.xsize = 200;
    tv_fake_stream.ysize = 100;
    s->st                = &tv_fake_stream;
    termview_options_init(&s->opt);
    s->view.zoom  = 1.0;
    s->view.cx    = 0.5;
    s->view.cy    = 0.5;
    s->rows       = 40;
    s->cols       = 120;
    s->running    = 1;
    s->target_fps = 20;
    s->cur_min    = 0.0;
    s->cur_max    = 1000.0;
    tv_layout_compute(&s->lay, s->rows, s->cols);
}

static void key(tv_state_t *s, uint32_t k, uint8_t mods)
{
    tv_event_t ev = { .type = TV_EV_KEY, .key = k, .mods = mods };
    if (!tv_ui_key(s, &ev)) /* as tv_dispatch_input() does */
    {
        tv_handle_key(s, &ev);
    }
}

static void mouse(tv_state_t *s, int button, int action, int x, int y)
{
    tv_event_t ev = { .type   = TV_EV_MOUSE,
                      .button = (uint8_t) button,
                      .action = (uint8_t) action,
                      .x      = (int16_t) x,
                      .y      = (int16_t) y };
    tv_handle_mouse(s, &ev);
}

/** feed_dispatch() - Decode @bytes as one read and dispatch, then let the ESC timeout pass. */
static void feed_dispatch(tv_state_t *s, const char *bytes, size_t n)
{
    tv_input_t in;
    memset(&in, 0, sizeof(in));
    tv_input_feed(&in, (const unsigned char *) bytes, n, 0);
    tv_dispatch_input(s, &in, 0, 512);
    tv_dispatch_input(s, &in, TV_ESC_TIMEOUT_US, 512);
}

/** feed_each() - Same events, one dispatch per mouse report (the uncoalesced reference). */
static void feed_each(tv_state_t *s, const char *bytes)
{
    const char *p = bytes;
    while (*p != '\0')
    {
        const char *e = strpbrk(p + 1, "\x1b");
        size_t      n = (e != NULL) ? (size_t) (e - p) : strlen(p);
        feed_dispatch(s, p, n);
        p += n;
    }
}

static void test_actions(void)
{
    tv_state_t s, r;

    /* c/C, s/S, r/R step both ways and open nothing; r also gives up a locked range */
    state_init(&s);
    key(&s, 'c', 0);
    TV_CHECK(s.opt.colormap == 1 && !s.ui.open);
    key(&s, 'C', 0);
    key(&s, 'C', 0);
    TV_CHECK(s.opt.colormap == COLORMAP_NB - 1 && s.view.cy == 0.5);
    s.opt.range_locked = true;
    key(&s, 'R', 0);
    TV_CHECK(s.opt.range == RANGE_NB - 1 && !s.opt.range_locked);

    /* a menu previews the highlighted choice; Esc undoes it, Enter keeps it */
    state_init(&s);
    key(&s, TV_K_F1 + 1, 0); /* F2: Cmap */
    TV_CHECK(s.ui.open && s.ui.menu == TV_M_CMAP && s.ui.sel == 0);
    key(&s, TV_K_DOWN, 0);
    key(&s, TV_K_DOWN, 0);
    TV_CHECK(s.opt.colormap == 2 && (s.dirty & TV_D_MAP));
    key(&s, TV_K_ESC, 0);
    TV_CHECK(!s.ui.open && s.opt.colormap == 0);
    key(&s, '\t', 0); /* Tab opens the last menu */
    key(&s, TV_K_END, 0);
    key(&s, '\r', 0);
    TV_CHECK(!s.ui.open && s.opt.colormap == COLORMAP_NB - 1);
    key(&s, TV_K_F1 + 1, 0);
    key(&s, TV_K_UP, 0);
    key(&s, TV_K_RIGHT, 0); /* to the Scale menu: the colormap preview is undone */
    TV_CHECK(s.ui.open && s.ui.menu == TV_M_SCALE && s.opt.colormap == COLORMAP_NB - 1);
    key(&s, 'c', 0); /* other keys do nothing while a menu is open */
    TV_CHECK(s.opt.colormap == COLORMAP_NB - 1);
    key(&s, TV_K_ESC, 0);

    /* Range: previewing a mode unlocks, Esc locks again with the same limits */
    state_init(&s);
    tv_set_range(&s, 3.0, 4.0);
    key(&s, TV_K_F1 + 3, 0);
    key(&s, TV_K_DOWN, 0);
    TV_CHECK(!s.opt.range_locked && s.opt.range == 1);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(s.opt.range_locked && s.opt.range == 0 && s.opt.manual_min == 3.0);

    /* a toggle chosen with Space stays applied and the menu stays open; Esc keeps it */
    state_init(&s);
    key(&s, TV_K_F1 + 4, 0); /* View */
    for (int k = 0; k < 4; k++)
    {
        key(&s, TV_K_DOWN, 0);
    }
    key(&s, ' ', 0);
    TV_CHECK(s.opt.flip_h && s.ui.open);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(s.opt.flip_h && !s.ui.open);

    /* Stream: the fps ladder previews; q still quits from a menu */
    state_init(&s);
    key(&s, TV_K_F1 + 5, 0);
    key(&s, TV_K_DOWN, 0);
    TV_CHECK(s.target_fps == 1);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(s.target_fps == 20);
    key(&s, TV_K_F1 + 5, 0);
    key(&s, 'q', 0);
    TV_CHECK(!s.running);

    /* help: ? opens it, keys scroll it, Esc closes it */
    state_init(&s);
    key(&s, '?', 0);
    TV_CHECK(s.ui.help && !s.ui.open);
    key(&s, TV_K_DOWN, 0);
    key(&s, '+', 0);
    TV_CHECK(s.ui.help_top == 1 && s.view.zoom == 1.0);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(!s.ui.help);

    /* Esc cancels a drag; Enter leaves a box zoom alone */
    state_init(&s);
    mouse(&s, TV_MB_LEFT, TV_MA_PRESS, 50, 20);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(!s.mouse_is_dragging);
    mouse(&s, TV_MB_RIGHT, TV_MA_PRESS, 40, 10);
    key(&s, '\r', 0);
    TV_CHECK(s.roi_is_dragging);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(!s.roi_is_dragging);

    /* arrows move the viewport in screen space, flips included; Alt+arrow still pans */
    state_init(&s);
    key(&s, TV_K_LEFT, 0);
    TV_CHECK(s.view.cx < 0.5);
    state_init(&s);
    s.opt.flip_h = true;
    key(&s, TV_K_LEFT, 0);
    TV_CHECK(s.view.cx > 0.5);
    state_init(&s);
    key(&s, TV_K_UP, 0);
    TV_CHECK(s.view.cy < 0.5);
    state_init(&s);
    s.opt.flip_v = true;
    key(&s, TV_K_UP, 0);
    TV_CHECK(s.view.cy > 0.5);
    state_init(&s);
    key(&s, TV_K_LEFT, TV_MOD_ALT);
    TV_CHECK(s.view.cx < 0.5);
    state_init(&s);
    key(&s, 'c', TV_MOD_ALT);
    TV_CHECK(s.opt.colormap == 0 && !s.ui.open);

    /* left drag grabs the image: dragging right shows what was on the left */
    state_init(&s);
    s.view.zoom = 4.0;
    mouse(&s, TV_MB_LEFT, TV_MA_PRESS, 50, 20);
    mouse(&s, TV_MB_LEFT, TV_MA_DRAG, 60, 20);
    TV_CHECK(s.view.cx < 0.5 && s.view.cy == 0.5);
    mouse(&s, TV_MB_LEFT, TV_MA_RELEASE, 60, 20);
    double cx = s.view.cx;
    mouse(&s, TV_MB_LEFT, TV_MA_DRAG, 70, 20);
    TV_CHECK(s.view.cx == cx);
    state_init(&s);
    s.opt.flip_h = true;
    s.view.zoom  = 4.0;
    mouse(&s, TV_MB_LEFT, TV_MA_PRESS, 50, 20);
    mouse(&s, TV_MB_LEFT, TV_MA_DRAG, 60, 20);
    TV_CHECK(s.view.cx > 0.5);
    state_init(&s);
    s.view.zoom = 4.0;
    mouse(&s, TV_MB_LEFT, TV_MA_PRESS, 50, 20);
    key(&s, TV_K_ESC, 0);
    mouse(&s, TV_MB_LEFT, TV_MA_DRAG, 60, 20);
    TV_CHECK(s.view.cx == 0.5);

    /* box zoom goes towards the box, mirrored when flipped */
    for (int flip = 0; flip < 2; flip++)
    {
        state_init(&s);
        s.opt.flip_h = flip;
        mouse(&s, TV_MB_RIGHT, TV_MA_PRESS, 70, 10);
        mouse(&s, TV_MB_RIGHT, TV_MA_DRAG, 100, 30);
        mouse(&s, TV_MB_RIGHT, TV_MA_RELEASE, 100, 30);
        TV_CHECK(s.view.zoom > 1.0 && (flip ? s.view.cx < 0.5 : s.view.cx > 0.5));
    }

    /* minibuffer: pre-filled, the first key replaces a field, every key validates, Enter locks */
    state_init(&s);
    feed_dispatch(&s, "m10\r", 4);
    TV_CHECK(s.opt.range_locked && s.opt.manual_min == 10.0 && s.opt.manual_max == 1000.0);
    TV_CHECK((s.dirty & TV_D_MAP) && !s.mini.active);
    state_init(&s);
    feed_dispatch(&s, "mnan\r", 5);
    TV_CHECK(s.mini.active && !s.opt.range_locked && s.mini.err != NULL);
    feed_dispatch(&s,
                  "\x7f\x7f\x7f"
                  "2000",
                  7);
    TV_CHECK(s.mini.err != NULL && strcmp(s.mini.err, "min must be < max") == 0);
    feed_dispatch(&s, "\t100000\r", 8); /* the first key replaces the pre-filled max */
    TV_CHECK(!s.mini.active && s.opt.manual_min == 2000.0 && s.opt.manual_max == 100000.0);
    state_init(&s);
    feed_dispatch(&s, "M5q \x1b", 5);
    TV_CHECK(!s.mini.active && !s.opt.range_locked && s.running && !s.paused);
    feed_dispatch(&s, "\x03", 1);
    TV_CHECK(!s.running);

    /* every wheel notch counts */
    char buf[1024] = "";
    for (int k = 0; k < 10; k++)
    {
        strcat(buf, "\x1b[<64;40;20M");
    }
    state_init(&s);
    feed_dispatch(&s, buf, strlen(buf));
    TV_CHECK(fabs(s.view.zoom - pow(1.2, 10)) < 1e-9);

    /* coalesced drags and wheel notches end where one-at-a-time handling ends */
    snprintf(buf, sizeof(buf),
             "\x1b[<0;40;20M\x1b[<32;45;20M\x1b[<32;50;21M\x1b[<64;50;21M\x1b[<32;55;22M"
             "\x1b[<64;55;22M\x1b[<65;55;22M\x1b[<32;60;23M\x1b[<0;60;23m");
    state_init(&s);
    state_init(&r);
    s.view.zoom = 2.0;
    r.view.zoom = 2.0;
    feed_dispatch(&s, buf, strlen(buf));
    feed_each(&r, buf);
    TV_CHECK(fabs(s.view.cx - r.view.cx) < 1e-12 && fabs(s.view.cy - r.view.cy) < 1e-12);
    TV_CHECK(fabs(s.view.zoom - r.view.zoom) < 1e-12 && s.view.cx != 0.5);

    /* zoom stays within its limits however far it is pushed */
    state_init(&s);
    for (int k = 0; k < 200; k++)
    {
        key(&s, '+', 0);
    }
    TV_CHECK(s.view.zoom <= 64.0 * 200 && tv_finite(s.view.zoom));
    for (int k = 0; k < 400; k++)
    {
        key(&s, '-', 0);
    }
    TV_CHECK(s.view.zoom >= 0.1);

    /* refresh rate: clamped to 1..120 */
    state_init(&s);
    tv_do(&s, TV_ACT_FPS, 500);
    TV_CHECK(s.target_fps == 120);
    tv_do(&s, TV_ACT_FPS, -500);
    TV_CHECK(s.target_fps == 1);
}

/* ---- interface ---- */

/** ui_screen() - Size @s to @rows x @cols with a fresh screen grid, and draw the interface. */
static void ui_draw(tv_state_t *s, tv_cell_t *grid, int rows, int cols)
{
    s->rows   = rows;
    s->cols   = cols;
    s->screen = grid;
    tv_layout_compute(&s->lay, rows, cols);
    for (int i = 0; i < rows * cols; i++)
    {
        grid[i] = tv_cell(0, 0, 'x');
    }
    tv_ui_render(s, 0);
}

/** row_text() - Glyphs of screen row @r as text; non-ASCII glyphs become '#'. */
static void row_text(const tv_state_t *s, int r, char *out)
{
    for (int c = 0; c < s->cols; c++)
    {
        uint8_t g = tv_cell_glyph(s->screen[(size_t) r * s->cols + c]);
        out[c]    = (g >= 0x20 && g < 0x7F) ? (char) g : '#';
    }
    out[s->cols] = '\0';
}

/** Every state draws inside the screen at every size, interface cells marked, the pill shown. */
static void test_ui_render(void)
{
    static tv_cell_t grid[300 * 999];
    static char      txt[1000];
    static const int sizes[][2] = { { 4, 17 },  { 4, 40 },   { 5, 17 },    { 6, 25 },  { 9, 30 },
                                    { 24, 80 }, { 40, 120 }, { 300, 999 }, { 60, 17 }, { 4, 999 } };
    tv_state_t       s;
    int              ok = 1;
    for (size_t z = 0; z < sizeof(sizes) / sizeof(sizes[0]); z++)
    {
        for (int mode = 0; mode < TV_M_COUNT + 3; mode++)
        {
            state_init(&s);
            snprintf(tv_fake_stream.name, sizeof(tv_fake_stream.name), "tvunit_stream");
            s.rows = sizes[z][0];
            s.cols = sizes[z][1];
            tv_layout_compute(&s.lay, s.rows, s.cols);
            if (mode < TV_M_COUNT)
            {
                tv_ui_open(&s, mode);
                key(&s, TV_K_END, 0); /* the last item: scrolled into view */
            }
            else if (mode == TV_M_COUNT)
            {
                key(&s, '?', 0);
                key(&s, TV_K_END, 0);
            }
            else if (mode == TV_M_COUNT + 1)
            {
                key(&s, 'm', 0);
            }
            ui_draw(&s, grid, sizes[z][0], sizes[z][1]);
            for (int k = 0; k < s.ui.nhits; k++)
            {
                const tv_hit_t *h = &s.ui.hits[k];
                ok &= h->x0 >= 0 && h->y0 >= 0 && h->x1 < s.cols && h->y1 < s.rows;
                /* dropdowns and the help box over the image are interface: sent first */
                for (int y = h->y0; h->kind <= TV_HIT_PANEL && y <= h->y1; y++)
                {
                    for (int x = h->x0; h->kind != TV_HIT_BUTTON && x <= h->x1; x++)
                    {
                        ok &= tv_cell_is_chrome(grid[(size_t) y * s.cols + x]);
                    }
                }
            }
            /* the menubar and the status row are interface everywhere */
            for (int c = 0; c < s.cols; c++)
            {
                ok &= tv_cell_is_chrome(grid[c]);
                ok &= tv_cell_is_chrome(grid[(size_t) s.lay.status_row * s.cols + c]);
            }
            row_text(&s, s.lay.status_row, txt);
            if (mode != TV_M_COUNT + 1)
            {
                ok &= (strncmp(txt, " # ", 3) == 0); /* the pill's dot */
            }
            row_text(&s, 0, txt);
            ok &= (txt[s.cols - 2] == 'q' || txt[s.cols - 2] == 't'); /* "q" or "q Quit" */
        }
    }
    TV_CHECK(ok);
}

/** The status bar drops segments from the end as the width shrinks; the pill always shows. */
static void test_ui_status(void)
{
    static tv_cell_t grid[10 * 200];
    static char      txt[201];
    tv_state_t       s;
    int              last = 1000, ok = 1;
    state_init(&s);
    snprintf(tv_fake_stream.name, sizeof(tv_fake_stream.name), "tvunit_stream");
    s.opt.flip_h = true;
    for (int cols = 200; cols >= 17; cols--)
    {
        ui_draw(&s, grid, 10, cols);
        row_text(&s, s.lay.status_row, txt);
        static const char *const order[] = { "LIVE",       "tvunit_stream", " Hz",
                                             "[0 : 1000]", "200x100",       "CPU",
                                             "fps",        "zoom",          "flipH" };
        int                      shown   = 0;
        while (shown < 9 && strstr(txt, order[shown]) != NULL)
        {
            shown++;
        }
        for (int k = shown; k < 9; k++)
        {
            ok &= (strstr(txt, order[k]) == NULL); /* nothing after the first one missing */
        }
        ok &= (shown >= 1 || cols < 8) && shown <= last;
        last = shown;
        if (cols == 200)
        {
            TV_CHECK(shown == 9);
        }
    }
    TV_CHECK(ok && last <= 2);
}

/** A dropdown over the image takes the click (topmost); with the menu closed the image does. */
static void test_ui_hits(void)
{
    static tv_cell_t grid[40 * 120];
    tv_state_t       s;
    state_init(&s);
    snprintf(tv_fake_stream.name, sizeof(tv_fake_stream.name), "tvunit_stream");
    tv_ui_open(&s, TV_M_CMAP);
    ui_draw(&s, grid, 40, 120);
    int x = -1, y = -1;
    for (int k = 0; k < s.ui.nhits; k++)
    {
        if (s.ui.hits[k].kind == TV_HIT_ITEM && s.ui.hits[k].id == 3)
        {
            x = s.ui.hits[k].x0 + 2;
            y = s.ui.hits[k].y0;
        }
    }
    TV_CHECK(x >= 0 && y > s.lay.img_top);
    tv_event_t ev = { .type   = TV_EV_MOUSE,
                      .button = TV_MB_LEFT,
                      .action = TV_MA_MOVE,
                      .x      = (int16_t) (x + 1),
                      .y      = (int16_t) (y + 1) };
    ev.button     = TV_MB_NONE;
    TV_CHECK(tv_ui_wants_mouse(&s, &ev));
    tv_ui_mouse(&s, &ev); /* hovering previews */
    TV_CHECK(s.opt.colormap == 3 && s.ui.open);
    ev.button = TV_MB_LEFT;
    ev.action = TV_MA_PRESS;
    tv_ui_mouse(&s, &ev);
    TV_CHECK(s.opt.colormap == 3 && !s.ui.open && s.ui.saved == 3);
    ui_draw(&s, grid, 40, 120);
    TV_CHECK(!tv_ui_wants_mouse(&s, &ev)); /* the same cell is image now */

    /* a click on a menubar button opens its menu; on another, switches; outside, closes */
    int bx = s.ui.btn_x[TV_M_SCALE] + 1;
    ev     = (tv_event_t) { .type = TV_EV_MOUSE, .button = TV_MB_LEFT, .action = TV_MA_PRESS };
    ev.x   = (int16_t) (bx + 1);
    ev.y   = 1;
    TV_CHECK(tv_ui_wants_mouse(&s, &ev));
    tv_ui_mouse(&s, &ev);
    TV_CHECK(s.ui.open && s.ui.menu == TV_M_SCALE);
    key(&s, TV_K_DOWN, 0);
    TV_CHECK(s.opt.scale == 1);
    ui_draw(&s, grid, 40, 120);
    ev.x = 100;
    ev.y = 30;
    tv_ui_mouse(&s, &ev);
    TV_CHECK(!s.ui.open && s.opt.scale == 0 && s.view.zoom == 1.0 && !s.mouse_is_dragging);

    /* help: the box takes clicks; the wheel scrolls it; outside closes it */
    key(&s, '?', 0);
    ui_draw(&s, grid, 12, 120);
    ev.button = TV_MB_WHEEL_DOWN;
    ev.x      = 60;
    ev.y      = 6;
    for (int k = 0; k < 20; k++)
    {
        tv_ui_mouse(&s, &ev);
        ui_draw(&s, grid, 12, 120);
    }
    TV_CHECK(s.ui.help && s.ui.help_top == tv_help_nkeys - (12 - 5));
    ev.button = TV_MB_LEFT;
    ev.x      = 2; /* left of the box and its shadow */
    tv_ui_mouse(&s, &ev);
    TV_CHECK(!s.ui.help);
}

/** hit_cell() - A cell inside the first hit rect of @kind and @id, 1-based for mouse events. */
static int hit_cell(const tv_state_t *s, int kind, int id, int *x, int *y)
{
    for (int k = 0; k < s->ui.nhits; k++)
    {
        if (s->ui.hits[k].kind == kind && s->ui.hits[k].id == id)
        {
            *x = s->ui.hits[k].x0 + 2;
            *y = s->ui.hits[k].y0 + 1;
            return 1;
        }
    }
    return 0;
}

/** A click handled before the next draw finds no stale rect; too small, the interface is inert;
 * a click outside a menu closes it and does nothing to the image. */
static void test_ui_stale(void)
{
    static tv_cell_t grid[40 * 120];
    tv_state_t       s;
    char             buf[64];
    int              x = 0, y = 0;

    for (int target = 0; target < 2; target++) /* the old Quit item, then old item 4 */
    {
        state_init(&s);
        tv_ui_open(&s, TV_M_STREAM);
        ui_draw(&s, grid, 40, 120);
        TV_CHECK(hit_cell(&s, TV_HIT_ITEM, target ? 4 : 14, &x, &y));
        snprintf(buf, sizeof(buf), "\x1b[D\x1b[<0;%d;%dM\x1b[<0;%d;%dm", x, y, x, y);
        feed_dispatch(&s, buf, strlen(buf)); /* Left, then a click, in one read */
        TV_CHECK(s.running && !s.opt.flip_h && s.target_fps == 20);
    }

    state_init(&s);
    ui_draw(&s, grid, 40, 120);
    TV_CHECK(hit_cell(&s, TV_HIT_BUTTON, TV_M_COUNT + 1, &x, &y)); /* Quit */
    s.rows = 3;
    tv_layout_compute(&s.lay, 3, 120);
    snprintf(buf, sizeof(buf), "\x1b[<0;%d;%dM\x1b[<0;%d;%dm\x1bOQ", x, y, x, y);
    feed_dispatch(&s, buf, strlen(buf));
    TV_CHECK(s.running && !s.ui.open);

    state_init(&s);
    tv_ui_open(&s, TV_M_SCALE);
    key(&s, TV_K_DOWN, 0);
    ui_draw(&s, grid, 40, 120);
    feed_dispatch(&s, "\x1b[<0;60;30M\x1b[<32;70;32M\x1b[<0;70;32m", 33);
    TV_CHECK(!s.ui.open && s.opt.scale == 0 && !s.mouse_is_dragging);
    TV_CHECK(s.view.zoom == 1.0 && s.view.cx == 0.5 && s.view.cy == 0.5 && !s.roi_is_dragging);
}

/** Toggles and actions do not keep a range previewed on the way to them; help closes the editor. */
static void test_ui_range_revert(void)
{
    tv_state_t s;
    state_init(&s);
    key(&s, TV_K_F1 + 3, 0);
    for (int k = 0; k < 8; k++)
    {
        key(&s, TV_K_DOWN, 0); /* through the eight modes to Lock range */
    }
    s.cur_min = 200.0; /* what a preview render would have shown */
    s.cur_max = 800.0;
    key(&s, '\r', 0);
    TV_CHECK(s.opt.range_locked && s.opt.range == 0);
    TV_CHECK(s.opt.manual_min == 0.0 && s.opt.manual_max == 1000.0);

    state_init(&s);
    tv_set_range(&s, 3.0, 4.0);
    s.cur_min = 3.0;
    s.cur_max = 4.0;
    key(&s, TV_K_F1 + 3, 0);
    key(&s, TV_K_DOWN, 0);
    s.cur_min = 200.0;
    s.cur_max = 800.0;
    key(&s, TV_K_END, 0);
    key(&s, '\r', 0); /* Set min / max... */
    TV_CHECK(s.mini.active && strcmp(s.mini.text[0], "3") == 0 && strcmp(s.mini.text[1], "4") == 0);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(!s.mini.active && s.opt.range_locked && s.opt.manual_min == 3.0);

    /* the same with Space, which keeps the menu open */
    state_init(&s);
    key(&s, TV_K_F1 + 3, 0);
    for (int k = 0; k < 8; k++)
    {
        key(&s, TV_K_DOWN, 0);
    }
    s.cur_min = 200.0;
    s.cur_max = 800.0;
    key(&s, ' ', 0);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(s.opt.range_locked && s.opt.range == 0 && s.opt.manual_max == 1000.0);

    /* locked before: the box shows unlocked after the previews, and asking to lock keeps the
     * typed limits */
    state_init(&s);
    tv_set_range(&s, 3.0, 4.0);
    s.cur_min = 3.0;
    s.cur_max = 4.0;
    key(&s, TV_K_F1 + 3, 0);
    for (int k = 0; k < 8; k++)
    {
        key(&s, TV_K_DOWN, 0);
    }
    s.cur_min = 200.0;
    s.cur_max = 800.0;
    key(&s, '\r', 0);
    TV_CHECK(s.opt.range_locked && s.opt.manual_min == 3.0 && s.opt.manual_max == 4.0);

    /* by mouse: the pointer crosses the choices on its way to Lock range */
    static tv_cell_t grid[40 * 120];
    char             buf[96];
    int              x = 0, y = 0;
    state_init(&s);
    tv_ui_open(&s, TV_M_RANGE);
    ui_draw(&s, grid, 40, 120);
    for (int id = 1; id < 8; id++)
    {
        hit_cell(&s, TV_HIT_ITEM, id, &x, &y);
        snprintf(buf, sizeof(buf), "\x1b[<35;%d;%dM", x, y);
        feed_dispatch(&s, buf, strlen(buf));
        ui_draw(&s, grid, 40, 120);
    }
    TV_CHECK(s.opt.range == 7);
    s.cur_min = 200.0;
    s.cur_max = 800.0;
    TV_CHECK(hit_cell(&s, TV_HIT_ITEM, 9, &x, &y));
    snprintf(buf, sizeof(buf), "\x1b[<35;%d;%dM\x1b[<0;%d;%dM\x1b[<0;%d;%dm", x, y, x, y, x, y);
    feed_dispatch(&s, buf, strlen(buf));
    key(&s, TV_K_ESC, 0);
    TV_CHECK(s.opt.range_locked && s.opt.range == 0 && s.opt.manual_max == 1000.0);

    /* Stream: Space on a toggle does not keep the refresh rate passed over */
    state_init(&s);
    key(&s, TV_K_F1 + 5, 0);
    for (int k = 0; k < 10; k++)
    {
        key(&s, TV_K_DOWN, 0);
    }
    key(&s, ' ', 0);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(s.target_fps == 20 && s.opt.lowbw);

    state_init(&s);
    key(&s, 'm', 0);
    key(&s, TV_K_F1, 0);
    TV_CHECK(s.ui.help && !s.mini.active);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(!s.ui.help);
    key(&s, 'q', 0);
    TV_CHECK(!s.running);
}

/** The hover highlight shows only while motion reports are on (a menu or help open). */
static void test_ui_hover(void)
{
    static tv_cell_t grid[40 * 120];
    tv_state_t       s;
    int              x = 0, y = 0;
    state_init(&s);
    tv_ui_open(&s, TV_M_CMAP);
    ui_draw(&s, grid, 40, 120);
    TV_CHECK(hit_cell(&s, TV_HIT_BUTTON, TV_M_COUNT, &x, &y)); /* Help */
    tv_event_t ev = { .type   = TV_EV_MOUSE,
                      .button = TV_MB_NONE,
                      .action = TV_MA_MOVE,
                      .x      = (int16_t) x,
                      .y      = (int16_t) y };
    tv_ui_mouse(&s, &ev);
    ui_draw(&s, grid, 40, 120);
    uint32_t hov = tv_cell_bg(grid[(size_t) (y - 1) * 120 + x - 1]);
    TV_CHECK(s.ui.hover == TV_M_COUNT + 1 && hov == TV_RGB(45, 52, 72));
    key(&s, TV_K_ESC, 0);
    ui_draw(&s, grid, 40, 120);
    TV_CHECK(s.ui.hover == 0 && tv_cell_bg(grid[(size_t) (y - 1) * 120 + x - 1]) != hov);

    /* nothing open: a report over a button (a drag passing by) highlights nothing */
    TV_CHECK(hit_cell(&s, TV_HIT_BUTTON, TV_M_SCALE, &x, &y));
    ev.button = TV_MB_LEFT;
    ev.action = TV_MA_DRAG;
    ev.x      = (int16_t) x;
    ev.y      = (int16_t) y;
    tv_ui_mouse(&s, &ev);
    ui_draw(&s, grid, 40, 120);
    TV_CHECK(tv_cell_bg(grid[(size_t) (y - 1) * 120 + x - 1]) != hov);
}

/** Colorbar labels at extreme ranges: each one parses back whole, close to its value. */
static void test_ui_labels(void)
{
    static tv_cell_t    grid[40 * 120];
    static const double r[][2] = { { -1e308, 1e308 },
                                   { 0.0, 1.7e308 },
                                   { -8e307, 9e307 },
                                   { -2e100, -1e100 },
                                   { -1.23456e-5, -1.23455e-5 },
                                   { 1234567.0, 1234567.5 },
                                   { 0.0, 1.0 },
                                   { -3.0, 1000.0 },
                                   { 12345678.0, 12345682.0 },
                                   { -0.1, 0.3 },
                                   { -0.0004, 3.9996 } };
    tv_state_t          s;
    int                 ok = 1, seen = 0;
    for (size_t k = 0; k < sizeof(r) / sizeof(r[0]); k++)
    {
        state_init(&s);
        s.cur_min = r[k][0];
        s.cur_max = r[k][1];
        ui_draw(&s, grid, 40, 120);
        double span = r[k][1] / 4 - r[k][0] / 4;
        char   seen_txt[8][16];
        int    nseen = 0;
        for (int row = s.lay.img_top; row < s.lay.img_top + s.lay.img_rows; row++)
        {
            char txt[16];
            int  n = 0;
            for (int c = s.lay.bar_col + 4; c < s.cols && n < 15; c++)
            {
                txt[n++] = (char) tv_cell_glyph(grid[(size_t) row * s.cols + c]);
            }
            while (n > 0 && txt[n - 1] == ' ')
            {
                n--;
            }
            txt[n] = '\0';
            if (n == 0)
            {
                continue;
            }
            char  *end;
            double v = strtod(txt, &end);
            ok &= (txt[0] != '-' || v != 0.0); /* no negative zero */
            for (int j = 0; j < nseen; j++)
            {
                ok &= (k < 5 || strcmp(seen_txt[j], txt) != 0); /* distinct where they fit */
            }
            if (nseen < 8)
            {
                snprintf(seen_txt[nseen++], sizeof(seen_txt[0]), "%s", txt);
            }
            double f    = 1.0 - (double) (row - s.lay.img_top) / (s.lay.img_rows - 1);
            double want = r[k][1] * f + r[k][0] * (1 - f);
            ok &= (*end == '\0' && tv_finite(v));
            ok &= fabs(v - want) <= fmax(span, 1e-3 * fabs(want));
            seen++;
        }
    }
    TV_CHECK(ok && seen >= 5 * 8);
}

/* ---- image pipeline ---- */

static void test_view_transform(void)
{
    tv_layout_t l;
    tv_layout_compute(&l, 40, 120);
    TV_CHECK(l.valid && l.img_cols == 107 && l.img_rows == 38 && l.img_top == 1);
    double zooms[5] = { 0.3, 1.0, 2.5, 17.0, 400.0 };
    for (int f = 0; f < 4; f++)
    {
        for (int z = 0; z < 5; z++)
        {
            tv_view_t  v = { .zoom = zooms[z], .cx = 0.3, .cy = 0.7 };
            tv_xform_t x;
            tv_xform_compute(&x, &l, &v, 640, 480, f & 1, f & 2);
            for (double d = -3.0; d < 120.0; d += 7.25)
            {
                TV_CHECK(fabs(tv_xf_disp_x(&x, tv_xf_img_x(&x, d)) - d) < 1e-9);
                TV_CHECK(fabs(tv_xf_disp_y(&x, tv_xf_img_y(&x, d)) - d) < 1e-9);
            }
            /* absolute values: step from the layout, edges from the centre */
            double step = fmax(640.0 / 107, 480.0 / 76) / zooms[z];
            TV_CHECK(fabs(x.step - step) < 1e-12 && x.cdx == 53.5 && x.cdy == 38.0);
            TV_CHECK(fabs(tv_xf_img_x(&x, 0.0) - (0.3 * 640 - x.sx * 53.5 * step)) < 1e-9);
            TV_CHECK(fabs(tv_xf_img_y(&x, 0.0) - (0.7 * 480 - x.sy * 38.0 * step)) < 1e-9);
            /* the display centre shows the view centre */
            TV_CHECK(fabs(tv_xf_img_x(&x, x.cdx) - 0.3 * 640) < 1e-9);
            TV_CHECK(fabs(tv_xf_img_y(&x, x.cdy) - 0.7 * 480) < 1e-9);
        }
    }
}

/** img_point_at() - Image point under 1-based screen cell (@cx, @cy) for state @s. */
static void img_point_at(const tv_state_t *s, int cx, int cy, double *px, double *py)
{
    tv_layout_t l;
    tv_xform_t  x;
    tv_layout_compute(&l, s->rows, s->cols);
    tv_xform_compute(&x, &l, &s->view, s->st->xsize, s->st->ysize, s->opt.flip_h, s->opt.flip_v);
    *px = tv_xf_img_x(&x, cx - 0.5);
    *py = tv_xf_img_y(&x, 2.0 * (cy - l.img_top - 1) + 1);
}

static void test_zoom_and_box(void)
{
    tv_state_t s;
    for (int f = 0; f < 4; f++)
    {
        /* drag grabs: the image point under the pointer follows it, on both axes */
        state_init(&s);
        s.opt.flip_h = f & 1;
        s.opt.flip_v = f & 2;
        s.view.zoom  = 4.0;
        double gx0, gy0, gx1, gy1;
        img_point_at(&s, 40, 15, &gx0, &gy0);
        mouse(&s, TV_MB_LEFT, TV_MA_PRESS, 40, 15);
        mouse(&s, TV_MB_LEFT, TV_MA_DRAG, 47, 18);
        img_point_at(&s, 47, 18, &gx1, &gy1);
        TV_CHECK(fabs(gx1 - gx0) < 1e-9 && fabs(gy1 - gy0) < 1e-9);
    }

    /* box zoom: the box fills the screen exactly */
    state_init(&s);
    mouse(&s, TV_MB_RIGHT, TV_MA_PRESS, 70, 10);
    mouse(&s, TV_MB_RIGHT, TV_MA_RELEASE, 100, 30);
    TV_CHECK(fabs(s.view.zoom - fmin(107.0 / 31, 76.0 / 42)) < 1e-9);

    /* the wheel stops at the zoom limit and keeps anchoring there */
    state_init(&s);
    tv_layout_t l;
    tv_layout_compute(&l, s.rows, s.cols);
    double zmax = 64.0 * tv_view_base(&l, 200, 100);
    s.view.zoom = zmax / 1.1;
    double ax0, ay0, ax1, ay1;
    img_point_at(&s, 30, 12, &ax0, &ay0);
    mouse(&s, TV_MB_WHEEL_UP, TV_MA_PRESS, 30, 12);
    mouse(&s, TV_MB_WHEEL_UP, TV_MA_PRESS, 30, 12);
    img_point_at(&s, 30, 12, &ax1, &ay1);
    TV_CHECK(fabs(s.view.zoom - zmax) < 1e-9 && fabs(ax1 - ax0) < 1e-9 && fabs(ay1 - ay0) < 1e-9);

    /* a 1x1 image: the wheel never zooms out below 1 */
    state_init(&s);
    tv_fake_stream.xsize = 1;
    tv_fake_stream.ysize = 1;
    mouse(&s, TV_MB_WHEEL_UP, TV_MA_PRESS, 30, 12);
    TV_CHECK(s.view.zoom >= 1.0);
    for (int f = 0; f < 4; f++)
    {
        /* wheel zoom keeps the image point under the pointer */
        state_init(&s);
        s.opt.flip_h = f & 1;
        s.opt.flip_v = f & 2;
        s.view.zoom  = 2.0;
        double px0, py0, px1, py1;
        img_point_at(&s, 30, 12, &px0, &py0);
        mouse(&s, TV_MB_WHEEL_UP, TV_MA_PRESS, 30, 12);
        mouse(&s, TV_MB_WHEEL_UP, TV_MA_PRESS, 30, 12);
        img_point_at(&s, 30, 12, &px1, &py1);
        TV_CHECK(fabs(s.view.zoom - 2.88) < 1e-9);
        TV_CHECK(fabs(px1 - px0) < 1e-6 && fabs(py1 - py0) < 1e-6);

        /* box zoom: the box centre becomes the view centre, under every flip */
        state_init(&s);
        s.opt.flip_h = f & 1;
        s.opt.flip_v = f & 2;
        double bx0, by0, bx1, by1;
        img_point_at(&s, 70, 10, &bx0, &by0);
        img_point_at(&s, 100, 30, &bx1, &by1);
        mouse(&s, TV_MB_RIGHT, TV_MA_PRESS, 70, 10);
        mouse(&s, TV_MB_RIGHT, TV_MA_DRAG, 100, 30);
        mouse(&s, TV_MB_RIGHT, TV_MA_RELEASE, 100, 30);
        double cx = s.view.cx * 200, cy = s.view.cy * 100;
        TV_CHECK(s.view.zoom > 1.0);
        TV_CHECK(fabs(cx - 0.5 * (bx0 + bx1)) < 1.0 && fabs(cy - 0.5 * (by0 + by1)) < 1.0);
    }
}

/** tables_identity() - Tables mapping a 7x5 image one-to-one onto 7x6 samples (row 5 outside). */
static void tables_identity(tv_image_t *im)
{
    tv_layout_t l;
    tv_layout_compute(&l, 3 + TV_CHROME_ROWS, 7 + TV_BAR_WIDTH);
    tv_view_t  v = { .zoom = 1.0, .cx = 0.5, .cy = 0.5 };
    tv_xform_t x;
    tv_xform_compute(&x, &l, &v, 7, 5, false, false);
    TV_CHECK(tv_image_geom(im, &l, &x, 7, 5) == 0);
}

static void test_samplers(void)
{
    static const uint8_t types[] = { _DATATYPE_UINT8,  _DATATYPE_INT8,   _DATATYPE_UINT16,
                                     _DATATYPE_INT16,  _DATATYPE_UINT32, _DATATYPE_INT32,
                                     _DATATYPE_UINT64, _DATATYPE_INT64,  _DATATYPE_FLOAT,
                                     _DATATYPE_DOUBLE };
    for (size_t t = 0; t < sizeof(types); t++)
    {
        union
        {
            uint8_t  u8[35];
            int8_t   i8[35];
            uint16_t u16[35];
            int16_t  i16[35];
            uint32_t u32[35];
            int32_t  i32[35];
            uint64_t u64[35];
            int64_t  i64[35];
            float    f[35];
            double   d[35];
        } a;
        double base = 0.0;
        switch (types[t])
        {
        case _DATATYPE_UINT32:
            base = 4.0e9;
            break;
        case _DATATYPE_INT32:
            base = -2.0e9;
            break;
        case _DATATYPE_UINT64:
        case _DATATYPE_INT64:
            base = 1.0e15; /* far beyond float precision: needs the offset */
            break;
        case _DATATYPE_DOUBLE:
            base = 1.0e12 + 0.25;
            break;
        case _DATATYPE_INT8:
        case _DATATYPE_INT16:
            base = -20.0;
            break;
        default:
            break;
        }
        for (int k = 0; k < 35; k++)
        {
            double v = base + k;
            switch (types[t])
            {
            case _DATATYPE_UINT8:
                a.u8[k] = (uint8_t) v;
                break;
            case _DATATYPE_INT8:
                a.i8[k] = (int8_t) v;
                break;
            case _DATATYPE_UINT16:
                a.u16[k] = (uint16_t) v;
                break;
            case _DATATYPE_INT16:
                a.i16[k] = (int16_t) v;
                break;
            case _DATATYPE_UINT32:
                a.u32[k] = (uint32_t) v;
                break;
            case _DATATYPE_INT32:
                a.i32[k] = (int32_t) v;
                break;
            case _DATATYPE_UINT64:
                a.u64[k] = (uint64_t) v;
                break;
            case _DATATYPE_INT64:
                a.i64[k] = (int64_t) v;
                break;
            case _DATATYPE_FLOAT:
                a.f[k] = (float) v;
                break;
            default:
                a.d[k] = v;
                break;
            }
        }
        tv_image_t im;
        memset(&im, 0, sizeof(im));
        tables_identity(&im);
        TV_CHECK(tv_image_sample(&im, &a, types[t]) == 1);
        int ok = (im.nvalid == 35);
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 7; j++)
            {
                float  sv   = im.samp[i * 7 + j];
                double want = base + i * 7 + j;
                ok &= (i == 5) ? !tv_finite(sv) : ((double) sv + im.offset == want);
            }
        }
        if (!ok)
        {
            fprintf(stderr, "sampler for datatype %d: wrong values\n", types[t]);
            tv_failures++;
        }
        /* same data again: reported unchanged; one changed element: reported changed */
        TV_CHECK(tv_image_sample(&im, &a, types[t]) == 0);
        switch (types[t])
        {
        case _DATATYPE_FLOAT:
            a.f[34] += 1.0f;
            break;
        case _DATATYPE_DOUBLE:
            a.d[34] += 1.0;
            break;
        default:
            a.u8[0] ^= 1; /* low byte of element 0 for every integer width */
            break;
        }
        TV_CHECK(tv_image_sample(&im, &a, types[t]) == 1);

        termview_options_t opt;
        termview_options_init(&opt);
        tv_image_range(&im, &opt, 1);
        TV_CHECK(tv_finite(im.lo) && tv_finite(im.hi) && im.hi > im.lo);
        tv_image_free(&im);
    }

    /* NaN and Inf in float and double data: excluded from the statistics, not from the image */
    double nan_v = tv_zero / tv_zero, inf_v = 1.0 / tv_zero;
    float  f[35];
    double d[35];
    for (int k = 0; k < 35; k++)
    {
        f[k] = (float) k;
        d[k] = 1e12 + k;
    }
    f[3]  = (float) nan_v;
    f[20] = (float) inf_v;
    d[3]  = nan_v;
    d[20] = -inf_v;
    for (int pass = 0; pass < 2; pass++)
    {
        tv_image_t im;
        memset(&im, 0, sizeof(im));
        tables_identity(&im);
        tv_image_sample(&im, pass ? (void *) d : (void *) f,
                        pass ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
        double base = pass ? 1e12 : 0.0;
        TV_CHECK(im.nvalid == 33);
        TV_CHECK(im.vmin + im.offset == base && im.vmax + im.offset == base + 34);
        tv_image_free(&im);
    }

    /* wide types: an atypical first pixel must not cost the rest of the image its precision */
    uint32_t u[35];
    for (int k = 0; k < 35; k++)
    {
        u[k] = 4000000000u + (uint32_t) k;
        d[k] = 1e12 + 1e-3 * k;
    }
    u[0] = 0;
    d[0] = nan_v;
    u[8] = 0; /* also on a point of the offset grid (row 1, column 1) */
    d[8] = 1e30;
    for (int pass = 0; pass < 2; pass++)
    {
        tv_image_t im;
        memset(&im, 0, sizeof(im));
        tables_identity(&im);
        tv_image_sample(&im, pass ? (void *) d : (void *) u,
                        pass ? _DATATYPE_DOUBLE : _DATATYPE_UINT32);
        int distinct = 1;
        for (int k = 2; k < 35; k++)
        {
            if (k != 8 && k != 9)
            {
                distinct &= (im.samp[k] > im.samp[k - 1]);
            }
        }
        TV_CHECK(distinct);
        tv_image_free(&im);
    }
}

/** Sampling tables: exact mirrors under flips, -1 outside the image on both sides. */
static void test_tables(void)
{
    tv_image_t  im;
    tv_layout_t l;
    tv_layout_compute(&l, 3 + TV_CHROME_ROWS, 7 + TV_BAR_WIDTH);
    for (int f = 0; f < 4; f++)
    {
        memset(&im, 0, sizeof(im));
        tv_view_t  v = { .zoom = 1.0, .cx = 0.5, .cy = 0.5 };
        tv_xform_t x;
        tv_xform_compute(&x, &l, &v, 7, 5, f & 1, f & 2);
        TV_CHECK(tv_image_geom(&im, &l, &x, 7, 5) == 0);
        for (int j = 0; j < 7; j++)
        {
            TV_CHECK(im.col_src[j] == ((f & 1) ? 6 - j : j));
        }
        for (int i = 0; i < 6; i++)
        {
            int64_t want =
                (f & 2) ? ((i == 0) ? -1 : (int64_t) (5 - i) * 7) : ((i == 5) ? -1 : i * 7);
            TV_CHECK(im.row_off[i] == want);
        }
        tv_image_free(&im);
    }

    /* zoomed out on a wide layout: columns outside on both sides, swapped by a horizontal flip */
    tv_layout_compute(&l, 40, 120);
    int left[2], right[2];
    for (int f = 0; f < 2; f++)
    {
        memset(&im, 0, sizeof(im));
        tv_view_t  v = { .zoom = 0.5, .cx = 0.3, .cy = 0.5 };
        tv_xform_t x;
        tv_xform_compute(&x, &l, &v, 64, 64, f, false);
        TV_CHECK(tv_image_geom(&im, &l, &x, 64, 64) == 0);
        left[f] = right[f] = 0;
        for (int j = 0; j < im.ncols && im.col_src[j] < 0; j++)
        {
            left[f]++;
        }
        for (int j = im.ncols - 1; j >= 0 && im.col_src[j] < 0; j--)
        {
            right[f]++;
        }
        tv_image_free(&im);
    }
    TV_CHECK(left[0] > 0 && right[0] > 0 && left[0] != right[0]);
    TV_CHECK(left[0] == right[1] && right[0] == left[1]);
}

/** Image layer: top sample on the background, bottom on the foreground, border, blank. */
static void test_render(void)
{
    static tv_rgb_t lut[TV_LUT_N + 1];
    tv_lut_build(lut, COLORMAP_GREYSCALE, SCALE_LINEAR);
    float data[35];
    for (int k = 0; k < 35; k++)
    {
        data[k] = (float) k;
    }
    tv_image_t im;
    memset(&im, 0, sizeof(im));
    tables_identity(&im);
    tv_image_sample(&im, data, _DATATYPE_FLOAT);
    termview_options_t opt;
    termview_options_init(&opt);
    tv_image_range(&im, &opt, 1);
    tv_image_render(&im, lut, 0);
    double    scale = (TV_LUT_N - 1) / 34.0;
    tv_cell_t c00   = im.cells[0];
    TV_CHECK(tv_cell_glyph(c00) == TV_G_LOWER_HALF);
    TV_CHECK(tv_cell_bg(c00) == tv_rgb(lut[tv_lut_idx(0.0, 0.0, scale)]));
    TV_CHECK(tv_cell_fg(c00) == tv_rgb(lut[tv_lut_idx(7.0, 0.0, scale)]));
    /* character row 2 shows image row 4 on top and the border row below it */
    tv_cell_t c20 = im.cells[2 * 7 + 3];
    TV_CHECK(tv_cell_bg(c20) == tv_rgb(lut[tv_lut_idx(31.0, 0.0, scale)]));
    TV_CHECK(tv_cell_fg(c20) == TV_RGB(64, 64, 64));
    tv_image_free(&im);

    /* far outside the image: a blank space */
    tv_layout_t l;
    tv_layout_compute(&l, 40, 120);
    tv_view_t  v = { .zoom = 0.5, .cx = 0.5, .cy = 0.5 };
    tv_xform_t x;
    tv_xform_compute(&x, &l, &v, 16, 16, false, false);
    memset(&im, 0, sizeof(im));
    TV_CHECK(tv_image_geom(&im, &l, &x, 16, 16) == 0);
    static float big[256];
    tv_image_sample(&im, big, _DATATYPE_FLOAT);
    tv_image_range(&im, &opt, 1);
    tv_image_render(&im, lut, 0);
    TV_CHECK(tv_cell_glyph(im.cells[0]) == ' ' && tv_cell_bg(im.cells[0]) == 0);
    tv_image_free(&im);
}

/** cells_close() - Both halves of every cell within @d2 (squared RGB distance) of @ref. */
static int cells_close(const tv_cell_t *a, const tv_cell_t *ref, int n, int d2)
{
    for (int i = 0; i < n; i++)
    {
        if (tv_test_dist2(tv_cell_bg(a[i]), tv_cell_bg(ref[i])) > d2 ||
            tv_test_dist2(tv_cell_fg(a[i]), tv_cell_fg(ref[i])) > d2)
        {
            return 0;
        }
    }
    return 1;
}

/** lb_frame() - Sample @data into both images and render @lb held, @ex exact. */
static void lb_frame(tv_image_t               *lb,
                     tv_image_t               *ex,
                     const float              *data,
                     const tv_rgb_t           *lut,
                     const termview_options_t *opt)
{
    tv_image_t *im[2] = { lb, ex };
    for (int k = 0; k < 2; k++)
    {
        tv_image_sample(im[k], data, _DATATYPE_FLOAT);
        tv_image_range(im[k], opt, 1);
        tv_image_render(im[k], lut, k == 0);
    }
}

/** Low-bandwidth rendering: held colours stay close, rows converge in a cycle, big changes show. */
static void test_render_lowbw(void)
{
    enum
    {
        W = 32,
        H = 40
    };
    static tv_rgb_t lut[TV_LUT_N + 1];
    static float    ramp[W * H], data[W * H];
    for (int k = 0; k < W * H; k++)
    {
        ramp[k] = (float) k * 1000.0f / (W * H);
    }
    termview_options_t opt;
    termview_options_init(&opt);
    opt.range_locked = true;
    opt.manual_min   = 0.0;
    opt.manual_max   = 1000.0;
    tv_layout_t l;
    tv_layout_compute(&l, H / 2 + TV_CHROME_ROWS, W + TV_BAR_WIDTH);
    tv_view_t  v = { .zoom = 1.0, .cx = 0.5, .cy = 0.5 };
    tv_xform_t x;
    tv_xform_compute(&x, &l, &v, W, H, false, false);

    static const termview_scale_t scales[] = { SCALE_LINEAR, SCALE_LOG_EXTREME, SCALE_ASINH };
    for (size_t sc = 0; sc < sizeof(scales) / sizeof(scales[0]); sc++)
    {
        tv_lut_build(lut, COLORMAP_GREYSCALE, scales[sc]);
        tv_image_t lb, ex;
        memset(&lb, 0, sizeof(lb));
        memset(&ex, 0, sizeof(ex));
        TV_CHECK(tv_image_geom(&lb, &l, &x, W, H) == 0 && tv_image_geom(&ex, &l, &x, W, H) == 0);
        int    n  = lb.ncols * (lb.nrows / 2);
        size_t sz = (size_t) n * sizeof(tv_cell_t);

        /* the first render on new tables is exact */
        lb_frame(&lb, &ex, ramp, lut, &opt);
        TV_CHECK(memcmp(lb.cells, ex.cells, sz) == 0 && lb.held == 0);

        /* sensor noise: colours are held, never further than the hold limit from exact */
        int close = 1, held = 0;
        for (int f = 0; f < 12; f++)
        {
            for (int k = 0; k < W * H; k++)
            {
                data[k] = ramp[k] + (float) ((int) (rnd() % 17) - 8);
            }
            lb_frame(&lb, &ex, data, lut, &opt);
            close &= cells_close(lb.cells, ex.cells, n, TV_LOWBW_DIST2);
            held += (lb.held > 0);
        }
        TV_CHECK(close && held == 12);

        /* the scene settles: each row is redrawn exactly within one cycle, then nothing is held */
        lb_frame(&lb, &ex, ramp, lut, &opt);
        TV_CHECK(lb.held > 0);
        for (int k = 1; k < TV_LOWBW_CYCLE; k++)
        {
            tv_image_render(&lb, lut, 1);
        }
        TV_CHECK(memcmp(lb.cells, ex.cells, sz) == 0 && lb.held == 0);

        /* new tables: exact again whatever was on screen */
        lb_frame(&lb, &ex, data, lut, &opt);
        TV_CHECK(tv_image_geom(&lb, &l, &x, W, H) == 0);
        lb_frame(&lb, &ex, ramp, lut, &opt);
        TV_CHECK(memcmp(lb.cells, ex.cells, sz) == 0);
        tv_image_free(&lb);
        tv_image_free(&ex);
    }

    /* linear: a change well beyond the hold limit shows on every row at once */
    tv_lut_build(lut, COLORMAP_GREYSCALE, SCALE_LINEAR);
    tv_image_t lb, ex;
    memset(&lb, 0, sizeof(lb));
    memset(&ex, 0, sizeof(ex));
    TV_CHECK(tv_image_geom(&lb, &l, &x, W, H) == 0 && tv_image_geom(&ex, &l, &x, W, H) == 0);
    lb_frame(&lb, &ex, ramp, lut, &opt);
    for (int k = 0; k < W * H; k++)
    {
        data[k] = fmodf(ramp[k] + 500.0f, 1000.0f);
    }
    lb_frame(&lb, &ex, data, lut, &opt);
    TV_CHECK(memcmp(lb.cells, ex.cells, (size_t) lb.ncols * (lb.nrows / 2) * sizeof(tv_cell_t)) ==
             0);
    tv_image_free(&lb);
    tv_image_free(&ex);
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *) a, y = *(const float *) b;
    return (x > y) - (x < y);
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

static void test_percentiles(void)
{
    enum
    {
        N = 5000
    };
    static float v[N], sorted[N];
    double       nan_v = tv_zero / tv_zero;
    int          nv    = 0;
    for (int k = 0; k < N; k++)
    {
        v[k] = (k % 50 == 0) ? (float) nan_v : (float) ((rnd() % 100000) / 7.0 + (k % 7) * 3.0);
        if (tv_finite(v[k]))
        {
            sorted[nv++] = v[k];
        }
    }
    qsort(sorted, (size_t) nv, sizeof(float), cmp_float);
    termview_options_t opt;
    termview_options_init(&opt);
    for (int r = 1; r < RANGE_NB; r++)
    {
        static const double plo[RANGE_NB] = { 0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20 };
        tv_image_t          im;
        fill_samples(&im, v, N);
        opt.range = (termview_range_t) r;
        tv_image_range(&im, &opt, 1);
        /* exact: the same samples a full sort would pick */
        TV_CHECK(im.lo == sorted[(long) (plo[r] * (nv - 1))]);
        TV_CHECK(im.hi == sorted[(long) ((1.0 - plo[r]) * (nv - 1))]);
        free(im.samp);
    }

    /* all equal, a single sample, nothing valid: always a finite range with lo < hi */
    float      same[10] = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
    float      one[3]   = { (float) nan_v, 7.0f, (float) nan_v };
    float      none[3]  = { (float) nan_v, (float) nan_v, (float) nan_v };
    tv_image_t im;
    for (int r = 0; r < RANGE_NB; r++)
    {
        opt.range = (termview_range_t) r;
        fill_samples(&im, same, 10);
        tv_image_range(&im, &opt, 1);
        TV_CHECK(im.lo < 5.0 && im.hi > 5.0);
        free(im.samp);
        fill_samples(&im, one, 3);
        tv_image_range(&im, &opt, 1);
        TV_CHECK(im.lo < 7.0 && im.hi > 7.0);
        free(im.samp);
        fill_samples(&im, none, 3);
        tv_image_range(&im, &opt, 1);
        TV_CHECK(im.lo == 0.0 && im.hi == 1.0);
        free(im.samp);
    }
}

/** Outliers far from the bulk (hot pixels, sentinels) must not move the percentile range. */
static void test_percentile_outliers(void)
{
    enum
    {
        N = 10001
    };
    static float       v[N], sorted[N];
    static const float outliers[4] = { 1e9f, -1e30f, 65535.0f, 3.0e38f };
    termview_options_t opt;
    termview_options_init(&opt);
    opt.range = RANGE_01_99;
    for (int o = 0; o < 4; o++)
    {
        for (int k = 0; k < N - 1; k++)
        {
            v[k] = (float) ((rnd() % 1000000) / 10000.0); /* 0..100 */
        }
        v[N - 1] = outliers[o];
        memcpy(sorted, v, sizeof(v));
        qsort(sorted, N, sizeof(float), cmp_float);
        tv_image_t im;
        fill_samples(&im, v, N);
        tv_image_range(&im, &opt, 1);
        TV_CHECK(im.lo == sorted[(long) (0.01 * (N - 1))]);
        TV_CHECK(im.hi == sorted[(long) (0.99 * (N - 1))]);
        TV_CHECK(im.lo >= 0.0 && im.hi <= 100.0);
        free(im.samp);
    }
}

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

    TV_CHECK(!im.settled); /* still on its way down */

    /* a darker minimum widens at once too; a small dip does nothing */
    im.vmin = -500.0f;
    tv_image_range(&im, &opt, 0);
    TV_CHECK(im.lo == -500.0);
    im.vmin = -490.0f;
    tv_image_range(&im, &opt, 0);
    TV_CHECK(im.lo == -500.0);

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

/** Stream lifecycle: removed, created again with another geometry, truncated, replaced by junk. */
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
    TV_CHECK(s.xsize == 32 && strcmp(s.name, "tvunit_life") == 0); /* still described */

    /* junk under the name: rejected, the stream stays closed */
    FILE *f = fopen(path, "w");
    for (int k = 0; f != NULL && k < 100000; k++)
    {
        fputc(k * 7 & 255, f);
    }
    if (f != NULL)
    {
        fclose(f);
    }
    TV_CHECK(tv_stream_reopen(&s) != 0 && s.map == NULL);

    /* and a real stream again */
    unlink(path);
    TV_CHECK(mkstream(&w, "tvunit_life", 16, 8, _DATATYPE_FLOAT) == 0);
    TV_CHECK(tv_stream_reopen(&s) == 0 && s.xsize == 16 && tv_stream_check(&s) == TV_STREAM_OK);
    ImageStreamIO_destroyIm(&w);
    tv_stream_close(&s);
    tv_stream_close(&s); /* twice is harmless */
    rmdir(dir);
    saved ? setenv("MILK_SHM_DIR", saved, 1) : unsetenv("MILK_SHM_DIR");
    free(saved);
}

/* ---- flow control ---- */

/** fc_req() - Record request @col written after @at bytes, @inflight of them unanswered, at @us. */
static void fc_req(tv_fc_t *fc, uint64_t at, uint64_t inflight, int64_t us, int col)
{
    tv_fc_req_t *r = &fc->req[(fc->head + fc->n++) % TV_FC_REQS];
    r->at          = at;
    r->inflight    = inflight;
    r->us          = us;
    r->col         = col;
    fc->sent       = at;
}

static void test_fc_window(void)
{
    const int64_t sl = 50000;
    tv_fc_t       fc;
    tv_fc_req_t   r;
    memset(&fc, 0, sizeof(fc));

    /* an answer behind (almost) nothing times the round trip; while timings agree, less often */
    fc.window  = 64 << 10;
    fc.base_us = 20000;
    fc.gap_us  = TV_FC_REBASE_US;
    r          = (tv_fc_req_t) { .at = 1000, .inflight = 500, .us = 1000000, .quiet = 1 };
    tv_fc_window(&fc, 1021000, &r, 500, sl);
    TV_CHECK(fc.base_us == 21000 && fc.quiet_us == 1021000 && fc.gap_us == 2 * TV_FC_REBASE_US);
    r.us = 2000000;
    tv_fc_window(&fc, 2080000, &r, 0, sl); /* the round trip grew */
    TV_CHECK(fc.base_us == 80000 && fc.quiet_us == 2080000 && fc.gap_us == TV_FC_REBASE_US);
    /* behind a queue an answer times nothing, unless it is faster than the base */
    r = (tv_fc_req_t) { .at = 1000, .inflight = 100000, .us = 3000000 };
    tv_fc_window(&fc, 3150000, &r, 0, sl);
    TV_CHECK(fc.base_us == 80000 && fc.quiet_us == 2080000);
    r.us = 3200000;
    tv_fc_window(&fc, 3230000, &r, 0, sl);
    TV_CHECK(fc.base_us == 30000 && fc.quiet_us == 2080000);

    /* the quietness of a request is decided when it is written, at the rate known then */
    r = (tv_fc_req_t) { .at = 1000, .inflight = 4000, .us = 4000000, .quiet = 1 };
    tv_fc_window(&fc, 4090000, &r, 0, sl);
    TV_CHECK(fc.base_us == 90000 && fc.quiet_us == 4090000);
    r = (tv_fc_req_t) { .at = 1000, .inflight = TV_FC_QUIET + 1, .us = 5000000 };
    tv_fc_window(&fc, 5100000, &r, 0, sl);
    TV_CHECK(fc.base_us == 90000 && fc.quiet_us == 4090000);

    /* delivery rate: bytes shown between answers far enough apart, over the longer of the
     * times between the answers and between their requests; the window is that rate times
     * (round trip + slice). A sample at most doubles the rate within a round trip. */
    memset(&fc, 0, sizeof(fc));
    fc.window  = 64 << 10;
    fc.base_us = 20000;
    r          = (tv_fc_req_t) { .at = 100000, .inflight = 60000, .us = 1000000 };
    tv_fc_window(&fc, 1030000, &r, 60000, sl); /* the first answer: where counting starts */
    TV_CHECK(fc.rate == 0.0 && fc.ref_us == 1030000 && fc.window == 64 << 10);
    r = (tv_fc_req_t) { .at = 110000, .inflight = 60000, .us = 1010000 };
    tv_fc_window(&fc, 1040000, &r, 10000, sl); /* 10 ms later: too close (ssh bunches) */
    TV_CHECK(fc.rate == 0.0 && fc.ref_us == 1030000);
    r = (tv_fc_req_t) { .at = 400000, .inflight = 60000, .us = 1300000 };
    tv_fc_window(&fc, 1330000, &r, 290000, sl); /* 300000 bytes in 300 ms */
    TV_CHECK(fabs(fc.rate - 1.0) < 1e-9 && fc.window == 70000);
    r = (tv_fc_req_t) { .at = 450000, .inflight = 60000, .us = 1400000 };
    tv_fc_window(&fc, 1430000, &r, 50000, sl); /* little to send: the maximum stays */
    TV_CHECK(fabs(fc.rate - 1.0) < 1e-9 && fc.window == 70000);
    r = (tv_fc_req_t) { .at = 1650000, .inflight = 60000, .us = 1500000 };
    tv_fc_window(&fc, 1530000, &r, 1200000, sl); /* 12 B/us: twice the rate, no more */
    TV_CHECK(fabs(fc.rate - 2.0) < 1e-9 && fc.window == 140000);
    tv_fc_window(&fc, 1530000, &r, 0, sl);
    TV_CHECK(fc.window == 140000);
    r = (tv_fc_req_t) { .at = 2850000, .inflight = 60000, .us = 1580000 };
    tv_fc_window(&fc, 1610000, &r, 1200000, sl); /* a round trip later: twice again */
    TV_CHECK(fabs(fc.rate - 4.0) < 1e-9 && fc.window == 280000);
    /* answers that come together after a delay count over the time between their requests:
     * here the reference answer came 1 s late, the next one 30 ms after it */
    tv_fc_t f2   = fc;
    f2.rate      = 0.5;
    f2.ref_us    = 3000000;
    f2.ref_sus   = 2000000;
    f2.ref_at    = 1000000;
    f2.grow_us   = 3029999;
    f2.grow_rate = 0.0;
    r            = (tv_fc_req_t) { .at = 1300000, .inflight = 60000, .us = 2700000 };
    tv_fc_window(&f2, 3030000, &r, 300000, sl); /* 300000 bytes: over 700 ms, not 30 */
    TV_CHECK(fabs(f2.rate - 0.5) < 1e-9);

    /* a queue of over 3 slices, for a slice of time: the link is slower than the maximum,
     * which goes toward what the answers show, by half at most (one late answer is not
     * enough, and delays vary) */
    r = (tv_fc_req_t) { .at = 3000000, .inflight = 200000, .us = 1610000 };
    tv_fc_window(&fc, 1810000, &r, 150000, sl); /* rtt 200 ms */
    TV_CHECK(fabs(fc.rate - 4.0) < 1e-9 && fc.high_us == 1810000);
    r = (tv_fc_req_t) { .at = 3060000, .inflight = 200000, .us = 1660000 };
    tv_fc_window(&fc, 1870000, &r, 60000, sl); /* 60 ms on, 60000 bytes in 60 ms */
    TV_CHECK(fabs(fc.rate - 2.0) < 1e-9 && fabs(fc.cut_rate - 4.0) < 1e-9 && fc.window == 140000);
    r = (tv_fc_req_t) { .at = 3100000, .inflight = 20000, .us = 1890000 };
    tv_fc_window(&fc, 1930000, &r, 40000, sl); /* no queue any more */
    TV_CHECK(fc.high_us == 0);
    /* a quiet answer: the round trip grew, so that queue was its doing: the rate comes back */
    f2 = fc;
    r  = (tv_fc_req_t) { .at = 3100100, .inflight = 100, .us = 2475000, .quiet = 1 };
    tv_fc_window(&f2, 2500000, &r, 100, sl);
    TV_CHECK(fabs(f2.rate - 2.0) < 1e-9 && f2.cut_rate == 0.0 && f2.base_us == 25000);
    /* ... unless a queue seen since the cut was longer than that growth: the link got slower */
    f2            = fc;
    f2.cut_rtt_us = 400000;
    r             = (tv_fc_req_t) { .at = 3100100, .inflight = 100, .us = 2300000, .quiet = 1 };
    tv_fc_window(&f2, 2500000, &r, 100, sl);
    TV_CHECK(fabs(f2.rate - 2.0) < 1e-9 && f2.cut_rate == 0.0 && f2.base_us == 200000);
    TV_CHECK(fc.cut_rtt_us == 210000);
    tv_fc_window(&fc, 2500000, &r, 100, sl);
    TV_CHECK(fabs(fc.rate - 4.0) < 1e-9 && fc.cut_rate == 0.0 && fc.base_us == 200000);

    /* long with little to send, the window starts below the estimate (the link may have got
     * slower); a stale maximum is replaced only by a sample taken while the window held image
     * back behind a queue (without one, it measures the window, or the idle time before) */
    r = (tv_fc_req_t) { .at = 3200000, .inflight = 50000, .us = 9800000 };
    tv_fc_window(&fc, 10000000, &r, 99900, sl);
    TV_CHECK(fabs(fc.rate - 4.0) < 1e-9 && fc.window == 250000);
    f2         = fc;
    fc.limited = 1;
    r          = (tv_fc_req_t) { .at = 3201000, .inflight = 50000, .us = 9900000 };
    tv_fc_window(&fc, 10100000, &r, 1000, sl);
    TV_CHECK(fabs(fc.rate - 4.0) < 1e-9 && fc.full_us == 10100000);
    f2.limited = 1;
    r.us       = 9800000; /* 100 ms of queue */
    tv_fc_window(&f2, 10100000, &r, 1000, sl);
    TV_CHECK(fabs(f2.rate - 0.01) < 1e-9 && f2.full_us == 10100000);
    /* image left over for another limit (writes blocking) is not idle either */
    f2.busy    = 1;
    f2.full_us = 0;
    tv_fc_window(&f2, 10200000, &r, 0, sl);
    TV_CHECK(f2.full_us == 10200000 && !f2.busy);

    /* the window held image back and no queue shows: it grows by what was answered (a window
     * more per round trip), not only to the measured rate */
    memset(&fc, 0, sizeof(fc));
    fc.window  = TV_FC_MIN;
    fc.base_us = 20000;
    fc.limited = 1;
    r          = (tv_fc_req_t) { .at = 20000, .inflight = 20000, .us = 4040000 };
    tv_fc_window(&fc, 4070000, &r, 20000, sl);
    TV_CHECK(fc.window == TV_FC_MIN + 20000 && fc.limited == 0);
    fc.limited = 1;
    fc.ref_us  = 4099000; /* (no rate sample here) */
    r          = (tv_fc_req_t) { .at = 70000, .inflight = 20000, .us = 4000000 };
    tv_fc_window(&fc, 4100000, &r, 50000, sl); /* with a queue it does not */
    TV_CHECK(fc.window == TV_FC_MIN + 20000);

    /* limits */
    fc.rate    = 1e6;
    fc.rate_us = 4100000;
    fc.full_us = 4100000;
    fc.window  = TV_FC_MAX;
    r          = (tv_fc_req_t) { .at = 80000, .inflight = 20000, .us = 4100000 };
    tv_fc_window(&fc, 4120000, &r, 0, sl);
    TV_CHECK(fc.window == TV_FC_MAX);
}

static void test_fc_requests(void)
{
    tv_state_t s;
    state_init(&s); /* 40 x 120 */
    tv_fc_t *fc = &s.fc;
    char     buf[32];

    /* the startup request; the first answer turns flow control on */
    size_t n = tv_fc_request(&s, buf, 1000);
    TV_CHECK(n > 0 && strcmp(buf, "\x1b[40;17H\x1b[6n") == 0 && fc->n == 1);
    tv_fc_wrote(fc, n, n, 0);
    TV_CHECK(fc->req[0].at == n && fc->req[0].inflight == n && fc->req[0].col == 17);
    TV_CHECK(fc->req[0].quiet);
    tv_fc_answer(&s, 31000, 17);
    TV_CHECK(fc->state == TV_FC_ON && fc->n == 0 && fc->acked == n && fc->base_us == 30000);

    /* then up to TV_FC_REQS, each in a column of its own, from 17 on (a narrower terminal does
     * not move them) */
    int cols[TV_FC_REQS], distinct = 1;
    for (int k = 0; k < TV_FC_REQS; k++)
    {
        n = tv_fc_request(&s, buf, 40000 + k);
        tv_fc_wrote(fc, 1000 + n, n, 0);
        cols[k] = fc->req[(fc->head + fc->n - 1) % TV_FC_REQS].col;
        distinct &= (cols[k] >= 17 && cols[k] <= 16 + TV_FC_IDS);
        for (int j = 0; j < k; j++)
        {
            distinct &= (cols[j] != cols[k]);
        }
    }
    TV_CHECK(distinct && fc->n == TV_FC_REQS && tv_fc_request(&s, buf, 50000) == 0);
    unsigned seq = fc->seq;
    for (int k = 0; k < 40; k++)
    {
        tv_fc_answer(&s, 60000, fc->req[fc->head].col);
        tv_fc_wrote(fc, 100, tv_fc_request(&s, buf, 60000), 0);
    }
    TV_CHECK(fc->seq == seq + 40 && fc->n == TV_FC_REQS);

    /* a failed write takes its request back */
    tv_fc_answer(&s, 70000, fc->req[fc->head].col);
    n = tv_fc_request(&s, buf, 70000);
    tv_fc_wrote(fc, n, n, 1);
    TV_CHECK(fc->n == TV_FC_REQS - 1);

    /* none while closing, on a 1-row terminal, or with flow control off */
    fc->closing = 1;
    TV_CHECK(tv_fc_request(&s, buf, 70000) == 0);
    fc->closing = 0;
    s.rows      = 1;
    TV_CHECK(tv_fc_request(&s, buf, 70000) == 0);
    s.rows    = 40;
    fc->state = TV_FC_OFF;
    TV_CHECK(tv_fc_request(&s, buf, 70000) == 0);

    /* narrow terminals: columns past 16 while there are any */
    state_init(&s);
    s.cols = 20;
    for (int k = 0; k < 3; k++)
    {
        s.fc.state = TV_FC_ON;
        tv_fc_wrote(&s.fc, 10, tv_fc_request(&s, buf, k), 0);
    }
    TV_CHECK(s.fc.req[0].col == 17 && s.fc.req[1].col == 18 && s.fc.req[2].col == 19);
    /* narrower: those are answered in its last column, so it is in use until they are */
    s.cols = 10;
    TV_CHECK(tv_fc_request(&s, buf, 5) == 0);
    tv_fc_answer(&s, 6, 10);
    TV_CHECK(s.fc.n == 2 && s.fc.req[s.fc.head].col == 18);
    tv_fc_answer(&s, 7, 10);
    tv_fc_answer(&s, 8, 10);
    tv_fc_wrote(&s.fc, 10, tv_fc_request(&s, buf, 9), 0);
    TV_CHECK(s.fc.n == 1 && s.fc.req[s.fc.head].col == 10 && tv_fc_request(&s, buf, 10) == 0);

    /* 41 columns, 24 of them: no more outstanding than that */
    state_init(&s);
    s.cols     = 41;
    s.fc.state = TV_FC_ON;
    for (int k = 0; k < 30; k++)
    {
        tv_fc_wrote(&s.fc, 10, tv_fc_request(&s, buf, k), 0);
    }
    TV_CHECK(s.fc.n == 24);
    tv_fc_answer(&s, 40, 17);
    tv_fc_wrote(&s.fc, 10, tv_fc_request(&s, buf, 41), 0);
    TV_CHECK(s.fc.n == 24 && s.fc.req[(s.fc.head + 23) % TV_FC_REQS].col == 17);
}

static void test_fc_answers(void)
{
    tv_state_t s;
    state_init(&s);
    tv_fc_t *fc = &s.fc;

    /* an answer nobody asked for changes nothing */
    fc->sent = 5000;
    tv_fc_answer(&s, 1000, 120);
    TV_CHECK(fc->state == TV_FC_PROBE && fc->acked == 0 && fc->n == 0);

    /* the startup request: on, timed, counting starts; what went out behind it gets a request */
    fc->window = TV_FC_START;
    fc_req(fc, 5000, 5000, 1000, 120);
    fc->sent = 9000;
    tv_fc_answer(&s, 31000, 120);
    TV_CHECK(fc->state == TV_FC_ON && fc->acked == 5000 && fc->base_us == 30000);
    TV_CHECK(fc->quiet_us == 31000 && fc->ref_us == 31000 && fc->n == 0 && fc->ask);
    fc->ask = 0;

    /* answers name their request; ones before it never came, an unknown column is dropped */
    fc_req(fc, 10000, 5000, 40000, 119);
    fc_req(fc, 20000, 15000, 50000, 118);
    fc_req(fc, 30000, 25000, 60000, 117);
    tv_fc_answer(&s, 80000, 100);
    TV_CHECK(fc->n == 3 && fc->acked == 5000);
    tv_fc_answer(&s, 90000, 118);
    TV_CHECK(fc->n == 1 && fc->acked == 20000 && fc->req[fc->head].col == 117);
    tv_fc_answer(&s, 95000, 119); /* the one given up on comes after all: too late */
    TV_CHECK(fc->n == 1 && fc->acked == 20000);
    fc->sent  = 40000;
    fc->acked = 35000; /* trickled past it (an overdue answer): never back */
    tv_fc_answer(&s, 96000, 117);
    TV_CHECK(fc->n == 0 && fc->acked == 35000);
    fc->acked = 30000;
    fc->sent  = 30000;
    fc->ask   = 0;

    /* across the end of the ring */
    fc->n    = 0;
    fc->head = TV_FC_REQS - 1;
    fc_req(fc, 40000, 10000, 100000, 116);
    fc_req(fc, 50000, 10000, 110000, 115);
    fc->wait = 1;
    s.dirty  = 0;
    tv_fc_answer(&s, 140000, 116);
    TV_CHECK(fc->acked == 40000 && fc->n == 1 && fc->head == 0);
    TV_CHECK(!fc->wait && fc->release && (s.dirty & TV_D_CHROME)); /* held image goes now */
    fc->release = 0;
    tv_fc_answer(&s, 150000, 115);
    TV_CHECK(fc->acked == 50000 && fc->n == 0 && !fc->release && !fc->ask);
    s.emit.pending = 1; /* image left over by the window goes now too */
    fc_req(fc, 60000, 10000, 150000, 114);
    tv_fc_answer(&s, 180000, 114);
    TV_CHECK(fc->release);
    /* no room, but what went after the last request needs one: held image goes with it */
    fc->release = 0;
    fc->wait    = 1;
    fc_req(fc, 70000, 10000, 190000, 113);
    fc->sent = 70000 + 2 * fc->window;
    tv_fc_answer(&s, 220000, 113);
    TV_CHECK(fc->ask && fc->release && !fc->wait);
    /* a drain: once nothing is unanswered, its quiet request goes now, not at the next frame */
    fc->release  = 0;
    fc->ask      = 0;
    fc->wait     = 1;
    fc->drain_us = 230000;
    fc_req(fc, fc->sent, 10000, 230000, 112);
    tv_fc_answer(&s, 260000, 112);
    TV_CHECK(fc->release && fc->drain_us != 0);

    /* a startup request answered after it timed out: on again, and timed again soon */
    state_init(&s);
    fc_req(fc, 100, 100, 0, 120);
    tv_fc_expire(&s, TV_FC_PROBE_US);
    TV_CHECK(fc->state == TV_FC_OFF && fc->n == 1);
    TV_CHECK(tv_fc_room(&s, TV_FC_PROBE_US) == SIZE_MAX);
    fc->sent = 900000;
    tv_fc_answer(&s, 3000000, 120);
    TV_CHECK(fc->state == TV_FC_ON && fc->ask && fc->quiet_us + fc->gap_us <= 3000000);
}

static void test_fc_room(void)
{
    tv_state_t s;
    state_init(&s);
    tv_fc_t *fc = &s.fc;
    char     buf[32];
    s.target_fps = 20;

    /* nothing asked yet, or a terminal that does not answer: no limit */
    TV_CHECK(tv_fc_room(&s, 0) == SIZE_MAX);
    fc->state = TV_FC_OFF;
    TV_CHECK(tv_fc_room(&s, 0) == SIZE_MAX);

    /* while the startup request is out: the first window (a slow link with a long round trip
     * would get a flood otherwise) */
    fc->state  = TV_FC_PROBE;
    fc->window = TV_FC_START;
    fc_req(fc, 20000, 20000, 0, 120);
    TV_CHECK(tv_fc_room(&s, 0) == TV_FC_START - 20000);
    TV_CHECK(tv_fc_room(&s, TV_FC_PROBE_US - 1) == TV_FC_START - 20000);

    /* on: a slice of the window per flush, so that answers come back a slice apart */
    fc->state    = TV_FC_ON;
    fc->n        = 0;
    fc->acked    = fc->sent;
    fc->window   = 300 << 10;
    fc->base_us  = 100000;
    fc->gap_us   = TV_FC_REBASE_US;
    fc->quiet_us = 0;
    fc->ramp_us  = -10000000;
    TV_CHECK(tv_fc_room(&s, 1000) == 100 << 10 && !fc->wbound);
    fc->base_us = 0; /* no round trip: the whole window */
    TV_CHECK(tv_fc_room(&s, 1000) == 300 << 10);
    fc->base_us = 5000000; /* a long one: at most TV_FC_REQS - 2 slices in flight */
    TV_CHECK(tv_fc_room(&s, 1000) == (300 << 10) / (TV_FC_REQS - 2));
    fc->base_us = 100000;
    fc->sent += 250 << 10; /* most of the window unanswered */
    TV_CHECK(tv_fc_room(&s, 1000) == 50 << 10 && fc->wbound);
    fc->sent += 100 << 10;
    TV_CHECK(tv_fc_room(&s, 1000) == 0);
    TV_CHECK(tv_fc_lead(&s, buf, 1000, 0) > 0 && fc->n == 1); /* held: something must answer */
    tv_fc_wrote(fc, strlen(buf), strlen(buf), 0);
    TV_CHECK(tv_fc_lead(&s, buf, 1000, 0) == 0);
    fc->credit = 100 << 10; /* trickled out while answers were late: not counted as answered */
    TV_CHECK(tv_fc_room(&s, 1000) == (50 << 10) - strlen(buf) &&
             fc->acked + (350 << 10) < fc->sent);
    fc->credit = 0;
    fc->n      = 0;
    fc->acked  = fc->sent;

    /* not in crumbs, but a chunk and what it leaves of the window both go while no queue
     * shows: a short round trip would idle the link per answer */
    fc->window  = TV_FC_MIN;
    fc->base_us = 20000;
    fc->sent += 5851 + 12;
    TV_CHECK(tv_fc_room(&s, 1000) == TV_FC_MIN - 5863);
    fc->high_us = 1;
    TV_CHECK(tv_fc_room(&s, 1000) == 0);
    fc->high_us = 0;
    fc->window  = 300 << 10;
    fc->base_us = 100000;
    fc->acked   = fc->sent;

    /* long without a quiet answer: the image waits until the queue has drained, then one
     * request goes out behind nothing and the image follows it at once */
    TV_CHECK(tv_fc_room(&s, TV_FC_REBASE_US) > 0 && fc->drain_us == 0);
    fc->sent += 5000;
    TV_CHECK(tv_fc_room(&s, TV_FC_REBASE_US + 1) == 0 && fc->drain_us == TV_FC_REBASE_US + 1);
    int64_t t = TV_FC_REBASE_US + 2;
    size_t  n = tv_fc_lead(&s, buf, t, 0); /* 5000 bytes nobody asked about */
    tv_fc_wrote(fc, n, n, 0);
    TV_CHECK(n > 0 && fc->drain_us != 0 && fc->n == 1);
    TV_CHECK(tv_fc_lead(&s, buf, t, 0) == 0); /* ... until that is answered */
    t += 100000;
    tv_fc_answer(&s, t, fc->req[fc->head].col);
    n = tv_fc_lead(&s, buf, t, 0);
    TV_CHECK(n > 0 && fc->drain_us == 0 && fc->quiet_us == t && fc->base_us == 100000);
    tv_fc_wrote(fc, n, n, 0);
    TV_CHECK(fc->req[(fc->head + fc->n - 1) % TV_FC_REQS].quiet);
    TV_CHECK(tv_fc_room(&s, t) > 0 && !fc->wbound);
    /* ... and the window opens over a round trip: keys would wait behind all of it at once */
    fc->sent += 150000;
    TV_CHECK(tv_fc_room(&s, t) == 0);
    TV_CHECK(tv_fc_room(&s, t + 50000) == 204800 - (fc->sent - fc->acked));
    TV_CHECK(tv_fc_room(&s, t + 100000) == 100 << 10);
    fc->n     = 0;
    fc->acked = fc->sent;

    /* a drain that gets nowhere ends, and the next one waits */
    t = fc->quiet_us + fc->gap_us + 1;
    fc->sent += 5000;
    TV_CHECK(tv_fc_room(&s, t) == 0);
    int64_t limit = 2 * fc->base_us + 4 * 50000 + 200000;
    TV_CHECK(tv_fc_room(&s, t + limit) == 0);
    TV_CHECK(tv_fc_room(&s, t + limit + 1) > 0 && fc->drain_us == 0 &&
             fc->quiet_us == t + limit + 1);

    TV_CHECK(fc->gap_us == 2 * TV_FC_REBASE_US); /* timings to come less often */

    /* answers that keep showing a queue start a drain early: the round trip may have grown.
     * It lasts as long as the round trip they showed, which is the base if it times out. */
    fc->acked       = fc->sent;
    t               = fc->quiet_us + 1000;
    fc->lift_us     = t;
    fc->lift_rtt_us = 400000;
    TV_CHECK(tv_fc_room(&s, t + 1999999) > 0);
    TV_CHECK(tv_fc_room(&s, t + 2000001) == 0 && fc->drain_us == t + 2000001 && fc->lift_us == 0);
    t += 2000001;
    fc->sent += 5000;
    limit = 2 * 400000 + 4 * 50000 + 200000;
    TV_CHECK(tv_fc_room(&s, t + limit) == 0);
    tv_fc_room(&s, t + limit + 1);
    TV_CHECK(fc->drain_us == 0 && fc->base_us == 400000 && fc->timed_us == 400000);

    /* the round trip shrank by more than two slices: timed again at once */
    tv_fc_req_t r = { .at = fc->sent, .inflight = 50000, .us = t + limit };
    tv_fc_window(fc, t + limit + 200000, &r, 0, 50000);
    TV_CHECK(fc->base_us == 200000 && fc->quiet_us + fc->gap_us <= t + limit + 200000);
}

static void test_fc_expire(void)
{
    tv_state_t s;
    state_init(&s);
    tv_fc_t *fc = &s.fc;

    /* the startup request unanswered: flow control is off, but its answer still counts */
    fc_req(fc, 100, 100, 0, 120);
    TV_CHECK(tv_fc_due_us(&s) == TV_FC_PROBE_US);
    tv_fc_expire(&s, TV_FC_PROBE_US - 1);
    TV_CHECK(fc->state == TV_FC_PROBE);
    TV_CHECK(tv_fc_settle_end(&s, 300000) == TV_FC_PROBE_US); /* leaving early */
    fc_req(fc, 200, 100, 1900000, 121);                       /* more requests may follow it */
    TV_CHECK(tv_fc_settle_end(&s, 1950000) == TV_FC_SETTLE_MAX_US); /* 3 s from the start */
    tv_fc_expire(&s, TV_FC_PROBE_US);
    TV_CHECK(fc->state == TV_FC_OFF && fc->n == 2 && fc->req[fc->head].col == 120);
    TV_CHECK(tv_fc_due_us(&s) == 0 && fc->release);
    TV_CHECK(tv_fc_settle_end(&s, TV_FC_PROBE_US) == TV_FC_SETTLE_MAX_US); /* a late answer */
    TV_CHECK(tv_fc_settle_end(&s, TV_FC_SETTLE_MAX_US + 1) == TV_FC_SETTLE_MAX_US + 1);
    tv_fc_answer(&s, 2500000, 120); /* it came: the others are on their way behind it */
    TV_CHECK(fc->state == TV_FC_ON && fc->n == 1 && tv_fc_settle_end(&s, 2500000) > 2500000);

    /* on: an overdue answer is late, not lost. A little image goes now and then meanwhile, and
     * the answer still counts when it comes. */
    state_init(&s);
    fc->state   = TV_FC_ON;
    fc->base_us = 100000;
    fc_req(fc, 64 << 10, 64 << 10, 0, 120);
    fc_req(fc, 100 << 10, 100 << 10, 10, 119);
    int64_t due = TV_FC_LOST_US + 4 * 100000 + (int64_t) ((64 << 10) / (TV_FC_FLOOR / 1e6));
    TV_CHECK(llabs(tv_fc_due_us(&s) - due) <= 1);
    fc->wait = 1;
    tv_fc_expire(&s, due + 1);
    TV_CHECK(fc->n == 2 && fc->credit == TV_FC_MIN && fc->acked == 0 && !fc->wait);
    TV_CHECK(fc->state == TV_FC_ON && tv_fc_due_us(&s) == due + 1 + TV_FC_LOST_US);
    fc_req(fc, 101 << 10, 100, due + 2, 118); /* not quiet: answers are late */
    TV_CHECK(fc->n == 3);
    fc->req[(fc->head + 2) % TV_FC_REQS].quiet = 1;
    fc->n                                      = 2;
    fc->sent                                   = 100 << 10;
    tv_fc_wrote(fc, 0, 0, 0);
    tv_fc_expire(&s, due + 1 + TV_FC_LOST_US); /* less and less often */
    TV_CHECK(fc->credit == 2 * TV_FC_MIN && fc->late_us == due + 1 + 3 * TV_FC_LOST_US);
    tv_fc_answer(&s, due + 2 + TV_FC_LOST_US, 120);
    TV_CHECK(fc->acked == 64 << 10 && fc->n == 1 && fc->late_us == 0 && fc->credit == 0);

    /* with the rate measured, the queue is allowed for at a quarter of it */
    fc->rate = 1.0;
    TV_CHECK(tv_fc_due_us(&s) == 10 + TV_FC_LOST_US + 400000 + 4 * (100 << 10));

    /* leaving: wait for the newest answer, as long as its queue takes while answers come on
     * time, 3 s at most once one is overdue */
    TV_CHECK(tv_fc_settle_end(&s, 0) == 10 + TV_FC_LOST_US + 400000 + 4 * (100 << 10));
    fc->rate = 0.0;
    TV_CHECK(tv_fc_settle_end(&s, 0) ==
             10 + TV_FC_LOST_US + 400000 + (int64_t) ((100 << 10) / (TV_FC_FLOOR / 1e6)));
    fc->late_us = 1;
    TV_CHECK(tv_fc_settle_end(&s, 0) == TV_FC_SETTLE_MAX_US);
    fc->late_us = 0;
    fc->rate    = 1000.0;
    TV_CHECK(tv_fc_settle_end(&s, 0) == 10 + TV_FC_LOST_US + 400000 + 409);
    TV_CHECK(tv_fc_settle_end(&s, 5000000) == 5000000 + TV_FC_SETTLE_MIN_US);
    fc->state = TV_FC_OFF; /* the startup request kept: 3 s from it, for a late answer */
    TV_CHECK(tv_fc_settle_end(&s, 0) == 10 + TV_FC_SETTLE_MAX_US - 10);
    fc->n = 0;
    TV_CHECK(tv_fc_settle_end(&s, 0) == 0);

    /* a full ring of overdue requests: the oldest goes, so that a request can go out, but not
     * in its column: its answer may still come, and would name a younger request */
    state_init(&s);
    char buf[32];
    fc->state = TV_FC_ON;
    for (int k = 0; k < TV_FC_REQS; k++)
    {
        fc_req(fc, 1000 * (uint64_t) (k + 1), 1000, k, 17 + k);
    }
    tv_fc_expire(&s, 100000000);
    TV_CHECK(fc->n == TV_FC_REQS - 1 && fc->req[fc->head].col == 18 && fc->state == TV_FC_ON);
    fc->seq = TV_FC_IDS; /* column 17 next */
    tv_fc_wrote(fc, 10, tv_fc_request(&s, buf, 100000001), 0);
    TV_CHECK(fc->n == TV_FC_REQS &&
             fc->req[(fc->head + fc->n - 1) % TV_FC_REQS].col == 17 + TV_FC_REQS);
    tv_fc_answer(&s, 100000002, 17); /* the evicted one: dropped */
    TV_CHECK(fc->n == TV_FC_REQS && fc->gone[0] != 0);
    tv_fc_answer(&s, 100000003, 18);
    TV_CHECK(fc->n == TV_FC_REQS - 1 && fc->gone[0] == 0 && fc->gone[1] == 0);

    /* a stop: nothing asked before it is waited for, and a request goes before the repaint */
    fc->state = TV_FC_ON;
    fc->sent  = 500000;
    tv_fc_resume(fc);
    TV_CHECK(fc->n == 0 && fc->acked == 500000 && fc->ask && tv_fc_due_us(&s) == 0);
    TV_CHECK(((fc->gone[0] >> (19 - 1)) & 1) && fc->credit == 0); /* their columns wait */
}

static void test_wake_at(void)
{
    tv_state_t s;
    state_init(&s);
    s.target_fps   = 20;
    tv_timers_t tm = { .next_data = 50000, .next_tick = 1000000, .last_render = 0 };
    TV_CHECK(tv_wake_at(&s, &tm, 0, -1) == 50000); /* the next data poll */
    s.paused = 1;
    TV_CHECK(tv_wake_at(&s, &tm, 0, -1) == 1000000); /* paused: the tick */
    /* an overdue answer: the trickle is not left to the tick */
    s.fc.state   = TV_FC_ON;
    s.fc.base_us = 100000;
    fc_req(&s.fc, 1000, 1000, 0, 17);
    s.fc.late_us = 300000;
    TV_CHECK(tv_wake_at(&s, &tm, 0, -1) == 300000);
    TV_CHECK(tv_wake_at(&s, &tm, 0, 20000) == 20000); /* an ESC sequence times out first */
    /* image left over: the interactive cap; held image an answer let go: now */
    s.paused       = 0;
    s.fc.late_us   = 0;
    s.emit.pending = 1;
    TV_CHECK(tv_wake_at(&s, &tm, 0, -1) == TV_INTERACTIVE_US);
    s.dirty      = TV_D_CHROME;
    s.fc.release = 1;
    TV_CHECK(tv_wake_at(&s, &tm, 5000, -1) == 5000);
    /* a new size waits while the image does, for a round trip at most */
    s.fc.release   = 0;
    s.emit.pending = 0;
    s.fc.wait      = 1;
    tm.resized_us  = 1000;
    TV_CHECK(tv_resize_waits(&s, &tm, 2000) && !tv_resize_waits(&s, &tm, 201000));
    tm.next_data = 900000;
    TV_CHECK(tv_wake_at(&s, &tm, 2000, -1) == 201000);
    s.fc.n = 0; /* nothing will answer: no wait */
    TV_CHECK(!tv_resize_waits(&s, &tm, 2000));
}

static void test_budget_update(void)
{
    tv_state_t s;
    state_init(&s);
    s.target_fps = 20;

    /* fast writes leave the budget off */
    tv_budget_update(&s, 500000, 1000);
    TV_CHECK(s.budget == 0);
    /* a write that blocked: what drains in a slice, at most twice the write, at least 16 KB */
    tv_budget_update(&s, 100000, 100000); /* 1 B/us */
    TV_CHECK(s.budget == 50000);
    tv_budget_update(&s, 10000, 1000000); /* 0.01 B/us */
    TV_CHECK(s.budget == 16384);
    tv_budget_update(&s, 20000, 10000); /* 2 B/us: believed halfway, capped at twice the write */
    TV_CHECK(s.budget == 40000);
    /* the terminal caught up: doubling, then off */
    s.budget = 1 << 20;
    tv_budget_update(&s, 600000, 100);
    TV_CHECK(s.budget == 2 << 20);
    tv_budget_update(&s, 1100000, 100);
    TV_CHECK(s.budget == 4 << 20);
    tv_budget_update(&s, 2100000, 100);
    TV_CHECK(s.budget == 0 && s.drain_rate == 0.0);
}

/*
 * A deterministic model of a viewer behind ssh, to test the flow control end to end. The
 * viewer's writes reach the far terminal fwd_us later, which shows them at a rate (0: the link
 * stalls) and answers each request when it shows it; answers (and keys) take ret_us back, in
 * order. sshd takes everything at once. The viewer gets a new frame of image bytes every 50 ms
 * (or keeps a still image), a key every 0.7 s (300 interface bytes) and a status update every
 * second (200), and flushes as termview_flush_ansi() does, through the same tv_fc_* calls. The
 * image goes out in exact budgets: the emitter stops within a cell of them.
 */
#define SIM_STEP 250 /* us */
#define SIM_SEGS 16384
#define SIM_ANS 4096

typedef struct
{
    int64_t t;    /* from this time on */
    double  rate; /* bytes/us shown */
    int64_t fwd_us, ret_us;
} sim_link_t;

typedef struct
{
    const sim_link_t *link;
    int               nlink;
    uint64_t          frame;       /* image bytes changed every 50 ms; 0: a still image */
    double            secs;        /* run time */
    double            from;        /* measured from */
    double            quit;        /* > 0: q then: the settle, and answers it misses */
    double            repaint;     /* > 0: the still image changes whole then */
    int               drop;        /* every drop-th answer is lost */
    int64_t           jitter_us;   /* answers take up to this much longer back */
    int               spike_every; /* every n-th answer is spike_us late, the rest behind it */
    int64_t           spike_us;
    int64_t           freeze_every_us; /* the far terminal shows nothing for freeze_us ... */
    int64_t           freeze_us;       /* ... then all that came meanwhile at once */
    int64_t           iface_us;        /* interface bytes every iface_us (a key held down) */
    int               iface_bytes;
    int               cols; /* the terminal's width (300) */
} sim_cfg_t;

typedef struct
{
    double   busy;    /* bytes shown over what the link could show */
    double   lat_med; /* keys: from pressing to seeing, s */
    double   lat_max;
    double   stall;   /* longest time without 8 KB of image shown, s (link up, live image) */
    double   first;   /* first image complete, s */
    double   repaint; /* the repaint complete, s after it */
    double   settle;  /* how long the settle waited, s */
    int      leaks;   /* answers that came after the settle */
    int      mism;    /* answers taken for another request than their own */
    int      state;
    uint64_t queue_max; /* bytes written but not shown */
} sim_result_t;

static sim_result_t sim_run(const sim_cfg_t *c)
{
    static uint64_t seg_end[SIM_SEGS];
    static int64_t  seg_at[SIM_SEGS];
    static int      seg_col[SIM_SEGS];
    static int64_t  ans_t[SIM_ANS];
    static int      ans_col[SIM_ANS];
    static uint64_t ans_at[SIM_ANS]; /* the request it answers: bytes up to it */
    static double   lat[512];
    static int64_t  key_t[64];  /* keys pressed, in order: when */
    static uint64_t key_at[64]; /* the byte that shows them, 0 until flushed */
    sim_result_t    res;
    tv_state_t      s;
    memset(&res, 0, sizeof(res));
    state_init(&s);
    s.rows         = 90;
    s.cols         = c->cols ? c->cols : 300;
    s.target_fps   = 20;
    tv_fc_t *fc    = &s.fc;
    fc->window     = TV_FC_START;
    fc->ask        = 1;
    int         sh = 0, sn = 0, ah = 0, an = 0, nlat = 0, nans = 0, li = 0, kh = 0, kn = 0;
    uint64_t    shown = 0, pending = 0, iface = 0, shown0 = 0, last_img = 0, repaint_at = 0;
    int64_t     last_flush = -1000000, next_frame = 0, next_key = 350000, next_tick = 1000000;
    int64_t     next_iface = c->iface_us, last_prog = 0, last_ans = 0;
    int64_t     end = (int64_t) (c->secs * 1e6), t0 = (int64_t) (c->from * 1e6);
    int64_t     tq = (int64_t) (c->quit * 1e6), tr = (int64_t) (c->repaint * 1e6);
    double      credit = 0.0, cap = 0.0;
    uint64_t    rng      = 88172645463325252ull;
    int         quitting = 0, gone = 0;
    tv_settle_t st;
    char        buf[32];
    res.first      = -1.0;
    res.repaint    = -1.0;
    uint64_t still = 560000 * (uint64_t) s.cols / 300; /* a whole screen */
    pending        = c->frame ? 0 : still;             /* a still image: one paint */

    for (int64_t t = 0; t < end; t += SIM_STEP)
    {
        while (li + 1 < c->nlink && c->link[li + 1].t <= t)
        {
            li++;
        }
        const sim_link_t *L = &c->link[li];

        /* the far terminal shows what has arrived, answering requests as it reaches them */
        int frozen = c->freeze_every_us && t % c->freeze_every_us < c->freeze_us;
        credit += L->rate * SIM_STEP;
        while (!frozen && sn > 0 && seg_at[sh] + L->fwd_us <= t && credit >= 1.0)
        {
            uint64_t take = seg_end[sh] - shown;
            take          = (take > (uint64_t) credit) ? (uint64_t) credit : take;
            shown += take;
            credit -= (double) take;
            if (shown == seg_end[sh])
            {
                if (seg_col[sh] && an < SIM_ANS && !(c->drop && ++nans % c->drop == 0))
                {
                    int64_t late =
                        (c->jitter_us ? (int64_t) ((rng >> 11) % (uint64_t) c->jitter_us) : 0);
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    if (c->spike_every && ++nans % c->spike_every == 0)
                    {
                        late += c->spike_us;
                    }
                    int64_t at                   = t + L->ret_us + late;
                    at                           = (at < last_ans) ? last_ans : at; /* in order */
                    last_ans                     = at;
                    ans_t[(ah + an) % SIM_ANS]   = at;
                    ans_col[(ah + an) % SIM_ANS] = seg_col[sh];
                    ans_at[(ah + an) % SIM_ANS]  = seg_end[sh];
                    an++;
                }
                sh = (sh + 1) % SIM_SEGS;
                sn--;
            }
        }
        if (!frozen && (sn == 0 || seg_at[sh] + L->fwd_us > t))
        {
            credit = 0.0; /* an idle link saves nothing */
        }
        if (t >= t0 && !quitting)
        {
            cap += L->rate * SIM_STEP;
            uint64_t q    = fc->sent - shown;
            res.queue_max = (q > res.queue_max) ? q : res.queue_max;
        }
        else if (t < t0)
        {
            shown0 = shown;
        }
        while (kn > 0 && key_at[kh] != 0 && shown >= key_at[kh])
        {
            if (key_t[kh] >= t0 && nlat < 512)
            {
                lat[nlat++] = (double) (t - key_t[kh] + L->ret_us) / 1e6;
            }
            kh = (kh + 1) % 64;
            kn--;
        }
        if (res.first < 0.0 && shown >= (c->frame ? c->frame : still))
        {
            res.first = (double) t / 1e6;
        }
        if (repaint_at && res.repaint < 0.0 && shown >= repaint_at)
        {
            res.repaint = (double) (t - tr) / 1e6;
        }
        if (shown - last_img >= 8192 || L->rate == 0.0 || !c->frame)
        {
            if (t >= t0 && !quitting && (double) (t - last_prog) / 1e6 > res.stall)
            {
                res.stall = (double) (t - last_prog) / 1e6;
            }
            last_prog = t;
            last_img  = shown;
        }

        /* the viewer */
        while (an > 0 && ans_t[ah] <= t)
        {
            /* its own request, if still outstanding, is the one it must be taken for */
            int k = tv_fc_match(&s, ans_col[ah]), own = fc->n;
            for (int j = 0; j < fc->n && own == fc->n; j++)
            {
                own = (fc->req[(fc->head + j) % TV_FC_REQS].at == ans_at[ah]) ? j : own;
            }
            res.mism += !gone && k != own && (k < fc->n || own < fc->n);
            if (gone)
            {
                res.leaks++; /* the shell gets it */
            }
            else if (quitting)
            {
                tv_settle_answer(&s, &st, t, ans_col[ah]);
            }
            else
            {
                tv_fc_answer(&s, t, ans_col[ah]);
            }
            ah = (ah + 1) % SIM_ANS;
            an--;
        }
        if (quitting)
        {
            /* tv_fc_settle(): answers only, until none is owed or the time is up */
            if (!gone && (fc->n == 0 || t >= st.end))
            {
                gone       = 1;
                res.settle = (double) (t - tq) / 1e6;
            }
            if (gone && sn == 0 && an == 0)
            {
                break;
            }
            continue;
        }
        if (tq && t >= tq)
        {
            quitting = 1;
            tv_settle_begin(&s, &st, t);
            end = t + 60000000; /* until all is shown and answered */
            continue;
        }
        tv_fc_expire(&s, t);
        int flush = 0;
        if (c->frame && t >= next_frame)
        {
            pending = c->frame; /* noise: every cell changes again */
            next_frame += 50000;
            flush = 1;
        }
        if (tr && t >= tr && !repaint_at)
        {
            pending    = still;
            repaint_at = fc->sent + still + 300;
            flush      = 1;
        }
        if (t >= next_key && kn < 64)
        {
            iface += 300;
            key_t[(kh + kn) % 64]  = t;
            key_at[(kh + kn) % 64] = 0;
            kn++;
            next_key += 700000;
        }
        if (t >= next_tick)
        {
            iface += 200;
            next_tick += 1000000;
        }
        if (c->iface_us && t >= next_iface)
        {
            iface += (uint64_t) c->iface_bytes;
            next_iface += c->iface_us;
        }
        flush = flush || fc->release || (t - last_flush >= 16000 && (iface > 0 || tv_fc_more(&s)));
        if (!flush)
        {
            continue;
        }
        size_t room = tv_fc_room(&s, t);
        size_t nl   = tv_fc_lead(&s, buf, t, room);
        if (nl > 0 && sn < SIM_SEGS)
        {
            tv_fc_wrote(fc, nl, nl, 0);
            seg_end[(sh + sn) % SIM_SEGS] = fc->sent;
            seg_at[(sh + sn) % SIM_SEGS]  = t;
            seg_col[(sh + sn) % SIM_SEGS] = fc->req[(fc->head + fc->n - 1) % TV_FC_REQS].col;
            sn++;
            room = tv_fc_room(&s, t);
        }
        tv_fc_plan(&s, room);
        size_t   budget = s.emit.hold ? 0 : s.emit.budget;
        uint64_t img    = s.emit.hold ? 0 : (budget == 0 || pending < budget) ? pending : budget;
        pending -= img;
        s.emit.pending = pending > 0 || s.emit.hold; /* as tv_flush_diff() leaves it */
        tv_fc_flushed(&s, room);
        fc->release = 0;
        last_flush  = t;
        uint64_t n  = iface + img;
        if (n == 0 || sn == SIM_SEGS)
        {
            continue;
        }
        iface      = 0;
        size_t req = tv_fc_request(&s, buf, t);
        tv_fc_wrote(fc, n + req, req, 0);
        seg_end[(sh + sn) % SIM_SEGS] = fc->sent;
        seg_at[(sh + sn) % SIM_SEGS]  = t;
        seg_col[(sh + sn) % SIM_SEGS] = req ? fc->req[(fc->head + fc->n - 1) % TV_FC_REQS].col : 0;
        sn++;
        for (int k = 0; k < kn; k++)
        {
            key_at[(kh + k) % 64] = key_at[(kh + k) % 64] ? key_at[(kh + k) % 64] : fc->sent;
        }
    }
    res.busy  = (cap > 0.0) ? (double) (shown - shown0) / cap : 0.0;
    res.state = fc->state;
    for (; kn > 0 && nlat < 512 && !tq; kh = (kh + 1) % 64, kn--)
    {
        if (key_t[kh] >= t0 && end - key_t[kh] > 2000000)
        {
            lat[nlat++] = (double) (end - key_t[kh]) / 1e6; /* never shown: what it waited so far */
        }
    }
    for (int i = 1; i < nlat; i++)
    {
        for (int j = i; j > 0 && lat[j - 1] > lat[j]; j--)
        {
            double x   = lat[j];
            lat[j]     = lat[j - 1];
            lat[j - 1] = x;
        }
    }
    res.lat_med = nlat ? lat[nlat / 2] : 0.0;
    res.lat_max = nlat ? lat[nlat - 1] : 0.0;
    return res;
}

static void sim_report(const char *what, sim_result_t r)
{
    if (getenv("TV_SIM_VERBOSE"))
    {
        fprintf(stderr,
                "%-30s busy %3.0f%%  keys %5.0f / %5.0f ms  stall %5.0f ms  queue %5.0f KB  first "
                "%.3f s  repaint %.3f s  settle %.3f s  leaks %d\n",
                what, r.busy * 100, r.lat_med * 1000, r.lat_max * 1000, r.stall * 1000,
                r.queue_max / 1024.0, r.first, r.repaint, r.settle, r.leaks);
    }
}

/** sim_check() - Report @r; fail when it is worse than @want (today's values with margin). */
static void sim_check(const char *what, sim_result_t r, sim_result_t want)
{
    sim_report(what, r);
    if (r.busy < want.busy || r.lat_med > want.lat_med || r.lat_max > want.lat_max ||
        r.stall > want.stall || (want.queue_max && r.queue_max > want.queue_max) || r.mism ||
        (want.first > 0.0 && r.first > want.first) ||
        (want.repaint > 0.0 && (r.repaint < 0.0 || r.repaint > want.repaint)) ||
        r.state != TV_FC_ON)
    {
        fprintf(stderr,
                "sim %s: busy %.2f (%.2f), keys %.3f/%.3f s (%.3f/%.3f), stall %.3f (%.3f), queue "
                "%llu (%llu), first %.3f (%.3f), repaint %.3f (%.3f), state %d, %d answer(s) "
                "taken for another request\n",
                what, r.busy, want.busy, r.lat_med, r.lat_max, want.lat_med, want.lat_max, r.stall,
                want.stall, (unsigned long long) r.queue_max, (unsigned long long) want.queue_max,
                r.first, want.first, r.repaint, want.repaint, r.state, r.mism);
        tv_failures++;
    }
}

#define SIM_LINK(mbs, rtt_ms) { 0, (mbs), (rtt_ms) * 500, (rtt_ms) * 500 }

/** sim_pin() - Limits from today's values: busy 3 points less, the rest 1.3x + 5 ms; 0: not
 * checked. Re-pin with TV_SIM_VERBOSE=1 when the controller is changed on purpose. */
static sim_result_t sim_pin(double busy,
                            double med_ms,
                            double max_ms,
                            double stall_ms,
                            double queue_kb)
{
    sim_result_t w = { 0 };
    w.busy         = busy - 0.03;
    w.lat_med      = med_ms ? 1.3 * med_ms / 1000 + 0.005 : 99;
    w.lat_max      = max_ms ? 1.3 * max_ms / 1000 + 0.005 : 99;
    w.stall        = stall_ms ? 1.3 * stall_ms / 1000 + 0.005 : 99;
    w.queue_max    = (uint64_t) (1.3 * queue_kb * 1024);
    return w;
}

static void test_fc_sim(void)
{
    char       what[64];
    sim_cfg_t  c;
    sim_link_t L[3];

    /* steady links, a noisy 300x90 image: keys within the round trip and a little more, the
     * link busy */
    static const struct
    {
        double mbs;
        int    rtt_ms;
        double busy, med, max, stall, queue; /* today: fraction, ms, KB */
    } steady[] = {
        { 1, 20, 1.0, 54, 68, 28, 126 },
        { 1, 100, 0.99, 137, 150, 108, 129 },
        { 1, 300, 0.98, 338, 350, 308, 292 },
        { 1, 1000, 0.93, 1045, 1055, 960, 600 },
        { 10, 50, 1.0, 82, 117, 51, 905 },
        { 0.1, 50, 1.0, 78, 103, 132, 8 },
        { 0.03, 50, 0.98, 207, 276, 323, 7 },
        { 0.03, 600, 0.96, 648, 690, 883, 12 },
        { 0.05, 1000, 0.93, 1036, 1094, 1167, 29 },
        { 0.1, 1000, 0.93, 1032, 1061, 1074, 58 },
        { 0.1, 20, 1.0, 68, 85, 102, 7 }, /* short round trips: a few chunks, all in flight */
        { 0.15, 20, 1.0, 43, 72, 74, 9 },
        { 0.05, 10, 0.96, 93, 153, 184, 8 },
        { 0.2, 10, 1.0, 49, 62, 51, 19 },
    };
    for (size_t i = 0; i < sizeof(steady) / sizeof(steady[0]); i++)
    {
        L[0] = (sim_link_t) SIM_LINK(steady[i].mbs, steady[i].rtt_ms);
        c    = (sim_cfg_t) { .link = L, .nlink = 1, .frame = 560000, .secs = 30, .from = 15 };
        snprintf(what, sizeof(what), "%.2f MB/s, %d ms", steady[i].mbs, steady[i].rtt_ms);
        sim_check(what, sim_run(&c),
                  sim_pin(steady[i].busy, steady[i].med, steady[i].max, steady[i].stall,
                          steady[i].queue));
    }

    /* startup on slow links with long round trips: the first keys, and what they wait behind */
    static const struct
    {
        double mbs;
        int    rtt_ms;
        double secs, max, queue;
    } start[] = { { 0.1, 800, 6, 1353, 83 },    { 0.06, 800, 6, 1547, 64 },
                  { 0.2, 1000, 6, 1627, 205 },  { 1, 1000, 6, 1366, 1026 },
                  { 0.05, 1500, 12, 2677, 75 }, { 0.2, 1500, 12, 2426, 303 },
                  { 0.1, 1900, 12, 3250, 190 } };
    for (size_t i = 0; i < sizeof(start) / sizeof(start[0]); i++)
    {
        L[0] = (sim_link_t) SIM_LINK(start[i].mbs, start[i].rtt_ms);
        c    = (sim_cfg_t) { .link = L, .nlink = 1, .frame = 560000, .secs = start[i].secs };
        snprintf(what, sizeof(what), "start %.2f MB/s, %d ms", start[i].mbs, start[i].rtt_ms);
        sim_check(what, sim_run(&c), sim_pin(0, 0, start[i].max, 0, start[i].queue));
    }

    /* a local terminal: the first paint goes out at once */
    L[0]           = (sim_link_t) { 0, 100, 50, 50 };
    c              = (sim_cfg_t) { .link = L, .nlink = 1, .frame = 1000000, .secs = 2, .from = 1 };
    sim_result_t w = sim_pin(0, 10, 10, 40, 0);
    w.first        = 0.02;
    sim_check("local, 100 MB/s", sim_run(&c), w);

    /* the link becomes 10x slower: keys are fast again within a second */
    L[0] = (sim_link_t) SIM_LINK(1, 50);
    L[1] = (sim_link_t) { 5000000, 0.1, 25000, 25000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 560000, .secs = 12, .from = 6.5 };
    sim_check("1 -> 0.1 MB/s", sim_run(&c), sim_pin(1.0, 79, 87, 85, 14));

    /* the link stalls for 8 s: afterwards it is as before */
    L[0] = (sim_link_t) SIM_LINK(1, 50);
    L[1] = (sim_link_t) { 4000000, 0, 25000, 25000 };
    L[2] = (sim_link_t) { 12000000, 1, 25000, 25000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 3, .frame = 560000, .secs = 20, .from = 14 };
    sim_check("8 s stall", sim_run(&c), sim_pin(0.99, 92, 100, 58, 104));

    /* the round trip grows tenfold, both ways or the way back only (up to 1.5 s: an uplink
     * full of something else): the link is busy again within seconds */
    L[0] = (sim_link_t) SIM_LINK(1, 20);
    L[1] = (sim_link_t) { 5000000, 1, 100000, 100000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 560000, .secs = 20, .from = 12 };
    sim_check("rtt 20 -> 200 ms", sim_run(&c), sim_pin(1.0, 244, 251, 14, 174));
    L[1] = (sim_link_t) { 5000000, 1, 10000, 1500000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 560000, .secs = 25, .from = 15 };
    sim_check("way back 10 -> 1500 ms", sim_run(&c), sim_pin(0.98, 1568, 1714, 101, 211));

    /* answers held up and let through together (1 in 33 is 200 ms late, the rest behind it),
     * and the far terminal pausing 200 ms every 2 s: no burst of output follows */
    L[0] = (sim_link_t) SIM_LINK(1, 200);
    c    = (sim_cfg_t) { .link        = L,
                         .nlink       = 1,
                         .frame       = 560000,
                         .secs        = 25,
                         .from        = 8,
                         .spike_every = 33,
                         .spike_us    = 200000 };
    sim_check("late answers", sim_run(&c), sim_pin(0.87, 245, 301, 208, 388));
    c = (sim_cfg_t) { .link            = L,
                      .nlink           = 1,
                      .frame           = 560000,
                      .secs            = 25,
                      .from            = 8,
                      .freeze_every_us = 2000000,
                      .freeze_us       = 200000 };
    sim_check("far end pausing", sim_run(&c), sim_pin(0.88, 204, 401, 354, 332));

    /* answers coming back up to 100 ms late at random */
    L[0] = (sim_link_t) SIM_LINK(1, 50);
    c    = (sim_cfg_t) {
           .link = L, .nlink = 1, .frame = 560000, .secs = 25, .from = 8, .jitter_us = 100000
    };
    sim_check("jitter 0-100 ms", sim_run(&c), sim_pin(0.87, 120, 214, 140, 257));

    /* every 10th answer lost */
    c = (sim_cfg_t) { .link = L, .nlink = 1, .frame = 560000, .secs = 20, .from = 8, .drop = 10 };
    sim_check("1 answer in 10 lost", sim_run(&c), sim_pin(0.92, 71, 117, 58, 107));

    /* a key held down (150 bytes at 30 Hz): the round trip is still timed, no long holds */
    L[0] = (sim_link_t) SIM_LINK(1, 300);
    c    = (sim_cfg_t) { .link        = L,
                         .nlink       = 1,
                         .frame       = 560000,
                         .secs        = 30,
                         .from        = 8,
                         .iface_us    = 33000,
                         .iface_bytes = 150 };
    sim_check("key held, 300 ms", sim_run(&c), sim_pin(0.99, 343, 352, 306, 401));

    /* a still image for 10 s, then a repaint: it goes out about as fast as the link allows */
    static const struct
    {
        int    rtt_ms;
        double repaint;
    } still[] = { { 100, 0.685 }, { 300, 0.935 }, { 600, 1.147 } };
    for (size_t i = 0; i < sizeof(still) / sizeof(still[0]); i++)
    {
        L[0] = (sim_link_t) SIM_LINK(1, still[i].rtt_ms);
        c    = (sim_cfg_t) {
               .link = L, .nlink = 1, .frame = 0, .secs = 16, .from = 10, .repaint = 10
        };
        snprintf(what, sizeof(what), "repaint after 10 s, %d ms", still[i].rtt_ms);
        w         = sim_pin(0, 0, 0, 0, 0);
        w.repaint = 1.3 * still[i].repaint + 0.005;
        sim_check(what, sim_run(&c), w);
    }
    /* ... and if the link got 10x slower meanwhile, keys still show soon after */
    L[0] = (sim_link_t) SIM_LINK(1, 300);
    L[1] = (sim_link_t) { 5000000, 0.1, 150000, 150000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 0, .secs = 20, .from = 10, .repaint = 10 };
    w    = sim_pin(0, 323, 1032, 0, 0);
    w.repaint = 1.3 * 5.888;
    sim_check("repaint, link 10x slower", sim_run(&c), w);

    /* a repaint after 10 s on fast links: the window opens again within a round trip or two */
    static const struct
    {
        double mbs;
        int    rtt_ms;
        double repaint;
    } fast[] = { { 10, 100, 0.200 }, { 3, 300, 0.561 } };
    for (size_t i = 0; i < sizeof(fast) / sizeof(fast[0]); i++)
    {
        L[0] = (sim_link_t) SIM_LINK(fast[i].mbs, fast[i].rtt_ms);
        c    = (sim_cfg_t) {
               .link = L, .nlink = 1, .frame = 0, .secs = 16, .from = 10, .repaint = 10
        };
        snprintf(what, sizeof(what), "repaint after 10 s, %.0f MB/s %d ms", fast[i].mbs,
                 fast[i].rtt_ms);
        w         = sim_pin(0, 0, 0, 0, 0);
        w.repaint = fast[i].repaint ? 1.3 * fast[i].repaint + 0.005 : 99;
        sim_check(what, sim_run(&c), w);
    }
    /* ... and a slow link's still image: no idle link while drains time the round trip */
    L[0] = (sim_link_t) SIM_LINK(0.03, 50);
    c    = (sim_cfg_t) { .link = L, .nlink = 1, .frame = 0, .secs = 40, .from = 10, .repaint = 10 };
    w    = sim_pin(0, 0, 0, 0, 0);
    w.repaint = 1.03 * 19.253; /* the link's own pace: 18.7 s */
    sim_check("repaint after 10 s, 0.03 MB/s 50 ms", sim_run(&c), w);

    /* the link gets slower as its round trip grows (a bulk upload sharing it): no flood */
    L[0] = (sim_link_t) SIM_LINK(1, 50);
    L[1] = (sim_link_t) { 20000000, 0.1, 125000, 125000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 560000, .secs = 40, .from = 20 };
    sim_check("1 -> 0.1 MB/s, rtt 50 -> 250 ms", sim_run(&c), sim_pin(0.9, 289, 580, 372, 88));
    L[1] = (sim_link_t) { 20000000, 0.05, 300000, 300000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 560000, .secs = 40, .from = 20 };
    sim_check("1 -> 0.05 MB/s, rtt 50 -> 600 ms", sim_run(&c), sim_pin(0.89, 638, 1940, 765, 83));

    /* the round trip shrinks (the way back clears) or grows a little: timed again soon */
    L[0] = (sim_link_t) { 0, 1, 5000, 1500000 };
    L[1] = (sim_link_t) { 20000000, 1, 5000, 5000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 560000, .secs = 40, .from = 22 };
    sim_check("way back 1500 -> 10 ms", sim_run(&c), sim_pin(0.99, 41, 108, 18, 112));
    L[0] = (sim_link_t) SIM_LINK(1, 20);
    L[1] = (sim_link_t) { 40000000, 1, 55000, 55000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 2, .frame = 560000, .secs = 80, .from = 40 };
    sim_check("rtt 20 -> 110 ms after 40 s", sim_run(&c), sim_pin(0.95, 143, 220, 118, 236));

    /* answers coming back up to 200 ms late at random */
    L[0] = (sim_link_t) SIM_LINK(1, 50);
    c    = (sim_cfg_t) {
           .link = L, .nlink = 1, .frame = 560000, .secs = 25, .from = 8, .jitter_us = 200000
    };
    sim_check("jitter 0-200 ms", sim_run(&c), sim_pin(0.56, 52, 291, 250, 379));

    /* stalls: a slow link 30 s, a long one 20 s; keys and the link afterwards */
    L[0] = (sim_link_t) SIM_LINK(0.03, 100);
    L[1] = (sim_link_t) { 10000000, 0, 50000, 50000 };
    L[2] = (sim_link_t) { 40000000, 0.03, 50000, 50000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 3, .frame = 560000, .secs = 70, .from = 40 };
    sim_check("30 s stall, 0.03 MB/s 100 ms", sim_run(&c), sim_pin(0.96, 245, 1386, 373, 46));
    L[0] = (sim_link_t) SIM_LINK(1, 600);
    L[1] = (sim_link_t) { 20000000, 0, 300000, 300000 };
    L[2] = (sim_link_t) { 40000000, 1, 300000, 300000 };
    c    = (sim_cfg_t) { .link = L, .nlink = 3, .frame = 560000, .secs = 50, .from = 40 };
    sim_check("20 s stall, 1 MB/s 600 ms", sim_run(&c), sim_pin(0.99, 645, 766, 128, 624));

    /* outages past what the ring holds: each answer still names its own request (checked in
     * every case), and leaving soon after the link is back sends none to the shell */
    static const struct
    {
        double mbs;
        int    rtt_ms, cols, secs;
        double quit, busy, med, max, stall, queue;
    } outage[] = { { 1, 100, 120, 90, 0, 0.99, 144, 160, 108, 130 },
                   { 0.1, 200, 120, 90, 0, 0.98, 226, 282, 277, 22 },
                   { 0.1, 600, 41, 20, 0, 1.0, 620, 650, 90, 42 },
                   { 0.1, 200, 120, 100, 2, 0, 0, 0, 0, 0 },
                   { 0.1, 600, 41, 45, 2.5, 0, 0, 0, 0, 0 } };
    for (size_t i = 0; i < sizeof(outage) / sizeof(outage[0]); i++)
    {
        int back = 20 + outage[i].secs;
        L[0]     = (sim_link_t) SIM_LINK(outage[i].mbs, outage[i].rtt_ms);
        L[1]     = (sim_link_t) { 20000000, 0, L[0].fwd_us, L[0].ret_us };
        L[2] = (sim_link_t) { (int64_t) back * 1000000, outage[i].mbs, L[0].fwd_us, L[0].ret_us };
        c    = (sim_cfg_t) { .link  = L,
                             .nlink = 3,
                             .frame = 560000 * (uint64_t) outage[i].cols / 300,
                             .secs  = back + (outage[i].quit ? outage[i].quit : 20),
                             .from  = back + 5,
                             .cols  = outage[i].cols,
                             .quit  = outage[i].quit ? back + outage[i].quit : 0 };
        snprintf(what, sizeof(what), "%d s outage, %d cols, %.1f MB/s %d ms%s", outage[i].secs,
                 outage[i].cols, outage[i].mbs, outage[i].rtt_ms, outage[i].quit ? ", q" : "");
        sim_result_t r = sim_run(&c);
        if (outage[i].quit)
        {
            sim_report(what, r);
            TV_CHECK(r.leaks == 0 && r.mism == 0);
        }
        else
        {
            sim_check(what, r,
                      sim_pin(outage[i].busy, outage[i].med, outage[i].max, outage[i].stall,
                              outage[i].queue));
        }
    }

    /* leaving: the settle waits for every answer still owed, early and late, on slow and long
     * links too, and no longer than they take */
    static const double rates[] = { 0.02, 0.03, 0.06, 0.1, 0.3, 1 };
    static const int    rtts[]  = { 50, 300, 600, 1000 };
    static const double quits[] = { 0.3, 1, 2, 4, 9 };
    int                 leaks = 0, slow = 0;
    double              longest = 0.0;
    for (size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); i++)
    {
        for (size_t j = 0; j < sizeof(rtts) / sizeof(rtts[0]); j++)
        {
            for (size_t k = 0; k < sizeof(quits) / sizeof(quits[0]); k++)
            {
                L[0]           = (sim_link_t) SIM_LINK(rates[i], rtts[j]);
                c              = (sim_cfg_t) { .link  = L,
                                               .nlink = 1,
                                               .frame = 560000,
                                               .secs  = quits[k] + 1,
                                               .from  = 0,
                                               .quit  = quits[k] };
                sim_result_t r = sim_run(&c);
                leaks += r.leaks;
                slow += (r.settle > TV_FC_SETTLE_LONG_US / 1e6);
                longest = (r.settle > longest) ? r.settle : longest;
                if (r.leaks && getenv("TV_SIM_VERBOSE"))
                {
                    fprintf(stderr, "quit %.2f MB/s %d ms at %.1f s: %d leak(s), settle %.2f s\n",
                            rates[i], rtts[j], quits[k], r.leaks, r.settle);
                }
            }
        }
    }
    if (getenv("TV_SIM_VERBOSE"))
    {
        fprintf(stderr, "quits: %zu runs, %d answer(s) reach the shell, longest settle %.2f s\n",
                sizeof(rates) / sizeof(rates[0]) * sizeof(rtts) / sizeof(rtts[0]) * sizeof(quits) /
                    sizeof(quits[0]),
                leaks, longest);
    }
    if (leaks || slow)
    {
        fprintf(stderr, "sim quits: %d answer(s) reach the shell, %d settle(s) too long\n", leaks,
                slow);
        tv_failures++;
    }
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
    test_flush_gap_budget();
    test_flush_fuzz();
    test_quantiser();
    test_color_level();
    test_flush_cost();
    test_flush_budget();
    test_flush_worst_case();
    test_input_vectors();
    test_input_timing();
    test_input_splits();
    test_input_fuzz();
    test_actions();
    test_ui_render();
    test_ui_status();
    test_ui_hits();
    test_ui_stale();
    test_ui_range_revert();
    test_ui_labels();
    test_ui_hover();
    test_view_transform();
    test_zoom_and_box();
    test_samplers();
    test_percentiles();
    test_percentile_outliers();
    test_range_smoothing();
    test_tables();
    test_render();
    test_render_lowbw();
    test_stream_lifecycle();
    test_fc_window();
    test_fc_requests();
    test_fc_answers();
    test_fc_room();
    test_fc_expire();
    test_wake_at();
    test_budget_update();
    test_fc_sim();
    if (tv_failures > 0)
    {
        fprintf(stderr, "%d check(s) failed\n", tv_failures);
        return 1;
    }
    printf("termview unit tests passed\n");
    return 0;
}
