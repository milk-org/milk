// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_flush.c
 * @brief Diff renderer: emits only cells whose visible appearance changed, with the fewest bytes,
 *        inside one synchronized update. Truecolor, xterm-256 or 16-colour output.
 */

#include "termview_internal.h"

static const char *const tv_glyph_utf8[TV_G_COUNT - 0x80] = {
    [TV_G_LOWER_HALF - 0x80] = "\xE2\x96\x84",
    [TV_G_FULL_BLOCK - 0x80] = "\xE2\x96\x88",
};

/* xterm's default 16-colour palette */
static const uint32_t tv_pal16[16] = {
    0x000000, 0xCD0000, 0x00CD00, 0xCDCD00, 0x0000EE, 0xCD00CD, 0x00CDCD, 0xE5E5E5,
    0x7F7F7F, 0xFF0000, 0x00FF00, 0xFFFF00, 0x5C5CFF, 0xFF00FF, 0x00FFFF, 0xFFFFFF,
};

static const uint8_t tv_cube[6] = { 0, 95, 135, 175, 215, 255 };

size_t tv_framebuf_size(int rows, int cols)
{
    /* worst case per cell: move (14) + combined colours (38) + glyph (3) */
    return (size_t) rows * (size_t) cols * 56 + (size_t) rows * 16 + 4096;
}

static inline int tv_dist2(uint32_t a, uint32_t b)
{
    int dr = (int) ((a >> 16) & 255) - (int) ((b >> 16) & 255);
    int dg = (int) ((a >> 8) & 255) - (int) ((b >> 8) & 255);
    int db = (int) (a & 255) - (int) (b & 255);
    return dr * dr + dg * dg + db * db;
}

/** tv_nearest256() - xterm-256 index nearest to @rgb among the 6x6x6 cube and the grey ramp. */
static uint8_t tv_nearest256(uint32_t rgb)
{
    int ch[3] = { (int) ((rgb >> 16) & 255), (int) ((rgb >> 8) & 255), (int) (rgb & 255) };
    int ci[3];
    for (int k = 0; k < 3; k++)
    {
        ci[k] = (ch[k] < 48) ? 0 : (ch[k] < 115) ? 1 : (ch[k] - 35) / 40;
    }
    uint32_t cube = TV_RGB(tv_cube[ci[0]], tv_cube[ci[1]], tv_cube[ci[2]]);
    int      avg  = (ch[0] + ch[1] + ch[2]) / 3;
    int      gi   = (avg < 8) ? 0 : (avg - 3) / 10;
    gi            = (gi > 23) ? 23 : gi;
    int      gv   = 8 + 10 * gi;
    uint32_t grey = TV_RGB(gv, gv, gv);
    return (tv_dist2(rgb, grey) < tv_dist2(rgb, cube))
               ? (uint8_t) (232 + gi)
               : (uint8_t) (16 + 36 * ci[0] + 6 * ci[1] + ci[2]);
}

static uint8_t tv_nearest16(uint32_t rgb)
{
    int best = 0, bd = tv_dist2(rgb, tv_pal16[0]);
    for (int k = 1; k < 16; k++)
    {
        int d = tv_dist2(rgb, tv_pal16[k]);
        if (d < bd)
        {
            bd   = d;
            best = k;
        }
    }
    return (uint8_t) best;
}

void tv_emit_init(tv_emit_t *e, int level)
{
    e->level      = (level >= 1 && level <= 3) ? level : 3;
    e->budget     = 0;
    e->resume_row = 0;
    e->pending    = 0;
    for (int k = 0; k < TV_QCACHE; k++)
    {
        e->qkey[k] = UINT32_MAX; /* no 24-bit colour has this key */
    }
}

/** tv_quant() - The colour as emitted at the current level: 24-bit RGB or a palette index. */
static inline uint32_t tv_quant(tv_emit_t *e, uint32_t rgb)
{
    if (e->level == 3)
    {
        return rgb;
    }
    uint32_t h = (rgb * 2654435761u) >> (32 - TV_QCACHE_BITS);
    if (e->qkey[h] != rgb)
    {
        e->qkey[h] = rgb;
        e->qval[h] = (e->level == 2) ? tv_nearest256(rgb) : tv_nearest16(rgb);
    }
    return e->qval[h];
}

uint32_t tv_emit_code(tv_emit_t *e, uint32_t rgb)
{
    return tv_quant(e, rgb);
}

/* What a cell looks like: colours that are not drawn (no ink, or full ink) do not count. */
typedef struct
{
    uint32_t bg, fg; /* emitted colour codes; TV_ANY when invisible */
    uint8_t  glyph;  /* after the half-block -> space reduction */
} tv_look_t;

#define TV_ANY UINT32_MAX

static inline tv_look_t tv_look(tv_emit_t *e, tv_cell_t c)
{
    tv_look_t l;
    l.bg    = tv_quant(e, tv_cell_bg(c));
    l.fg    = tv_quant(e, tv_cell_fg(c));
    l.glyph = tv_cell_glyph(c);
    if (l.glyph == TV_G_LOWER_HALF && l.bg == l.fg)
    {
        l.glyph = ' '; /* both halves alike: a space shows it with one colour */
    }
    if (l.glyph == ' ')
    {
        l.fg = TV_ANY;
    }
    else if (l.glyph == TV_G_FULL_BLOCK)
    {
        l.bg = TV_ANY;
    }
    return l;
}

static inline int tv_look_eq(tv_look_t a, tv_look_t b)
{
    return a.glyph == b.glyph && a.bg == b.bg && a.fg == b.fg;
}

/** tv_put_dec() - Decimal digits of @v; 0..255 come from a table (no division). */
static inline void tv_put_dec(char *out, size_t *pos, unsigned v)
{
    static const char d2[] = "000102030405060708091011121314151617181920212223242526272829"
                             "303132333435363738394041424344454647484950515253545556575859"
                             "606162636465666768697071727374757677787980818283848586878889"
                             "90919293949596979899";
    if (v < 10)
    {
        out[(*pos)++] = (char) ('0' + v);
        return;
    }
    if (v < 256)
    {
        if (v >= 100)
        {
            out[(*pos)++] = (v >= 200) ? '2' : '1';
            v -= (v >= 200) ? 200 : 100;
        }
        out[(*pos)++] = d2[2 * v];
        out[(*pos)++] = d2[2 * v + 1];
        return;
    }
    char   tmp[10];
    size_t n = 0;
    while (v > 0 && n < sizeof(tmp))
    {
        tmp[n++] = (char) ('0' + v % 10);
        v /= 10;
    }
    while (n > 0)
    {
        out[(*pos)++] = tmp[--n];
    }
}

static inline void tv_put_str(char *out, size_t *pos, const char *s, size_t n)
{
    memcpy(out + *pos, s, n);
    *pos += n;
}

/** tv_put_colour() - The SGR parameters of one colour: "48;2;r;g;b", "38;5;n", "41", ... */
static void tv_put_colour(const tv_emit_t *e, char *out, size_t *pos, uint32_t code, int is_bg)
{
    if (e->level == 3)
    {
        tv_put_str(out, pos, is_bg ? "48;2;" : "38;2;", 5);
        tv_put_dec(out, pos, (code >> 16) & 255);
        out[(*pos)++] = ';';
        tv_put_dec(out, pos, (code >> 8) & 255);
        out[(*pos)++] = ';';
        tv_put_dec(out, pos, code & 255);
    }
    else if (e->level == 2)
    {
        tv_put_str(out, pos, is_bg ? "48;5;" : "38;5;", 5);
        tv_put_dec(out, pos, code);
    }
    else
    {
        unsigned base = (code < 8) ? (is_bg ? 40 : 30) : (is_bg ? 100 - 8 : 90 - 8);
        tv_put_dec(out, pos, base + code);
    }
}

/** tv_put_glyph() - The glyph's bytes. */
static inline void tv_put_glyph(char *out, size_t *pos, uint8_t g)
{
    if (g >= 0x80 && g < TV_G_COUNT && tv_glyph_utf8[g - 0x80] != NULL)
    {
        const char *u = tv_glyph_utf8[g - 0x80];
        tv_put_str(out, pos, u, strlen(u));
    }
    else
    {
        out[(*pos)++] = (g >= 0x20 && g < 0x7F) ? (char) g : ' ';
    }
}

/** tv_free_to_draw() - @l can be drawn with the colours in effect, without any escape. */
static inline int tv_free_to_draw(tv_look_t l, uint32_t bg, uint32_t fg)
{
    return (l.bg == TV_ANY || l.bg == bg) && (l.fg == TV_ANY || l.fg == fg);
}

/* Emission state shared by the cells of one flush. */
typedef struct
{
    tv_emit_t *e;
    char      *out;
    size_t     pos;
    int        r, c;   /* terminal cursor, -1 when unknown */
    uint32_t   bg, fg; /* colours in effect */
} tv_out_t;

/**
 * tv_emit_cell() - Send cell (@r, @c) of @cur, reaching it from the current cursor cheaply.
 * Skipped cells re-sent on the way are drawn exactly as in @cur, so @prev takes them too.
 */
static void tv_emit_cell(tv_out_t        *o,
                         const tv_cell_t *cur,
                         tv_cell_t       *prev,
                         int              cols,
                         int              r,
                         int              c,
                         tv_look_t        l)
{
    char *out = o->out;
    if (o->r == r && c > o->c)
    {
        /* same row: re-send up to 2 skipped cells when that needs no escape, else CUF */
        const tv_cell_t *row  = cur + (size_t) r * cols;
        int              gap  = c - o->c;
        int              free = (gap <= 2);
        for (int k = o->c; k < c && free; k++)
        {
            free = tv_free_to_draw(tv_look(o->e, row[k]), o->bg, o->fg);
        }
        if (free)
        {
            for (int k = o->c; k < c; k++)
            {
                tv_put_glyph(out, &o->pos, tv_look(o->e, row[k]).glyph);
                prev[(size_t) r * cols + k] = row[k];
            }
        }
        else
        {
            tv_put_str(out, &o->pos, "\033[", 2);
            tv_put_dec(out, &o->pos, (unsigned) gap);
            out[o->pos++] = 'C';
        }
    }
    else if (o->r != r || o->c != c)
    {
        tv_put_str(out, &o->pos, "\033[", 2);
        tv_put_dec(out, &o->pos, (unsigned) r + 1);
        if (c > 0)
        {
            out[o->pos++] = ';';
            tv_put_dec(out, &o->pos, (unsigned) c + 1);
        }
        out[o->pos++] = 'H';
    }

    int set_bg = (l.bg != TV_ANY && l.bg != o->bg);
    int set_fg = (l.fg != TV_ANY && l.fg != o->fg);
    if (set_bg || set_fg)
    {
        tv_put_str(out, &o->pos, "\033[", 2);
        if (set_bg)
        {
            tv_put_colour(o->e, out, &o->pos, l.bg, 1);
            o->bg = l.bg;
        }
        if (set_fg)
        {
            if (set_bg)
            {
                out[o->pos++] = ';';
            }
            tv_put_colour(o->e, out, &o->pos, l.fg, 0);
            o->fg = l.fg;
        }
        out[o->pos++] = 'm';
    }
    tv_put_glyph(out, &o->pos, l.glyph);
    o->r = r;
    o->c = c + 1;
}

/**
 * tv_flush_row() - Send the changed cells of row @r: interface cells only (@chrome) or the rest.
 * Returns the number of cells sent.
 */
static int tv_flush_row(tv_out_t        *o,
                        const tv_cell_t *cur,
                        tv_cell_t       *prev,
                        int              cols,
                        int              r,
                        int              chrome)
{
    int sent = 0;
    for (int c = 0; c < cols; c++)
    {
        size_t    i    = (size_t) r * cols + c;
        tv_cell_t cell = cur[i];
        if (cell == prev[i])
        {
            continue;
        }
        /* interface cells, and cells that were interface a moment ago (a closing popup) */
        int is_chrome =
            tv_cell_is_chrome(cell) || (prev[i] != TV_CELL_UNKNOWN && tv_cell_is_chrome(prev[i]));
        if (is_chrome != chrome)
        {
            continue;
        }
        tv_look_t l = tv_look(o->e, cell);
        if (prev[i] != TV_CELL_UNKNOWN && tv_look_eq(l, tv_look(o->e, prev[i])))
        {
            prev[i] = cell; /* different cell, same appearance: nothing to send */
            continue;
        }
        prev[i] = cell;
        tv_emit_cell(o, cur, prev, cols, r, c, l);
        sent++;
    }
    return sent;
}

size_t tv_flush_diff(tv_emit_t       *e,
                     char            *out,
                     const tv_cell_t *cur,
                     tv_cell_t       *prev,
                     int              rows,
                     int              cols)
{
    static const char sync_on[]  = "\033[?2026h";
    static const char sync_off[] = "\033[0m\033[?2026l";

    tv_out_t o = {
        .e = e, .out = out, .pos = sizeof(sync_on) - 1, .r = -1, .c = -1, .bg = TV_ANY, .fg = TV_ANY
    };
    int sent = 0;
    memcpy(out, sync_on, sizeof(sync_on) - 1);

    /* the interface first: keystroke feedback never waits behind image bytes */
    for (int r = 0; r < rows; r++)
    {
        sent += tv_flush_row(&o, cur, prev, cols, r, 1);
    }

    /* then the image, within the budget, starting where the last budgeted flush stopped */
    size_t img_start = o.pos;
    int    start     = (e->budget > 0 && e->resume_row < rows) ? e->resume_row : 0;
    e->pending       = 0;
    for (int k = 0; k < rows; k++)
    {
        int r = (start + k) % rows;
        if (e->budget > 0 && o.pos - img_start >= e->budget)
        {
            e->resume_row = r;
            e->pending    = 1;
            break;
        }
        sent += tv_flush_row(&o, cur, prev, cols, r, 0);
    }
    if (!e->pending)
    {
        e->resume_row = 0;
    }
    if (sent == 0)
    {
        return 0;
    }
    tv_put_str(out, &o.pos, sync_off, sizeof(sync_off) - 1);
    return o.pos;
}
