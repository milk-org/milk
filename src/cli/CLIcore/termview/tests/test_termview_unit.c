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

/* ---- input decoder ---- */

/** tv_ev_str() - Compact text form of an event, appended to @out. */
static void tv_ev_str(const tv_event_t *ev, char *out, size_t cap)
{
    size_t len = strlen(out);
    if (ev->type == TV_EV_KEY)
    {
        snprintf(out + len, cap - len, "K%u/%u ", ev->key, ev->mods);
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
                                 "\x1b[?62;22c\x1b[[B\x1b[200~x\x1b[201~\x03";
    size_t            n        = sizeof(stream) - 1;
    char              want[512], got[4096], early[4096];

    snprintf(want, sizeof(want),
             "K97/0 K98/0 K%u/0 M0/0/67,12/0 M0/2/70,12/0 M0/1/70,12/0 K%u/1 K%u/0 "
             "M64/0/1,1/0 K122/0 K%u/4 K113/0 K%u/0 K120/0 K3/0 ",
             TV_K_UP, TV_K_F1 + 4, TV_K_F1, TV_K_LEFT, TV_K_F1 + 1);
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
}

static void key(tv_state_t *s, uint32_t k, uint8_t mods)
{
    tv_event_t ev = { .type = TV_EV_KEY, .key = k, .mods = mods };
    tv_handle_key(s, &ev);
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

    /* quick-pick popup: Up/Down step it, Enter closes it, the view does not move */
    state_init(&s);
    key(&s, 'c', 0);
    TV_CHECK(s.opt.colormap == 1 && s.popup_type == 1);
    key(&s, TV_K_DOWN, 0);
    TV_CHECK(s.opt.colormap == 2);
    key(&s, TV_K_UP, 0);
    key(&s, TV_K_UP, 0);
    key(&s, TV_K_UP, 0);
    TV_CHECK(s.opt.colormap == COLORMAP_NB - 1 && s.view.cy == 0.5);
    key(&s, '\r', 0);
    TV_CHECK(s.popup_type == 0);

    /* Esc cancels a drag and closes the popup in one press; Enter leaves a box zoom alone */
    key(&s, 's', 0);
    mouse(&s, TV_MB_LEFT, TV_MA_PRESS, 50, 20);
    key(&s, TV_K_ESC, 0);
    TV_CHECK(s.popup_type == 0 && !s.mouse_is_dragging);
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
    TV_CHECK(s.opt.colormap == 0 && s.popup_type == 0);

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

    /* typed limits lock the range and recolour; invalid input keeps the prompt open */
    state_init(&s);
    feed_dispatch(&s, "m10\r", 4);
    TV_CHECK(s.opt.range_locked && s.opt.manual_min == 10.0 && s.opt.manual_max == 1000.0);
    TV_CHECK((s.dirty & TV_D_MAP) && s.input_mode == 0);
    state_init(&s);
    feed_dispatch(&s, "mnan\r", 5);
    TV_CHECK(s.input_mode == 1 && !s.opt.range_locked && s.input_err != NULL);
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
    test_input_vectors();
    test_input_timing();
    test_input_splits();
    test_input_fuzz();
    test_actions();
    if (tv_failures > 0)
    {
        fprintf(stderr, "%d check(s) failed\n", tv_failures);
        return 1;
    }
    printf("termview unit tests passed\n");
    return 0;
}
