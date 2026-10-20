// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"
uint32_t ov__mono_reverse_bg[2] = { OV_COLOR_NONE, OV_COLOR_NONE };

/* xterm default 16-colour palette */
static const uint8_t ov_pal16[16][3] = { { 0, 0, 0 },      { 205, 0, 0 },     { 0, 205, 0 },
                                         { 205, 205, 0 },  { 0, 0, 238 },     { 205, 0, 205 },
                                         { 0, 205, 205 },  { 229, 229, 229 }, { 127, 127, 127 },
                                         { 255, 0, 0 },    { 0, 255, 0 },     { 255, 255, 0 },
                                         { 92, 92, 255 },  { 255, 0, 255 },   { 0, 255, 255 },
                                         { 255, 255, 255 } };
static const uint8_t ov_cube[6]      = { 0, 95, 135, 175, 215, 255 };

/** ov_dist2 - squared RGB distance. */
static int ov_dist2(int r, int g, int b, int r2, int g2, int b2)
{
    return (r - r2) * (r - r2) + (g - g2) * (g - g2) + (b - b2) * (b - b2);
}

/** ov_rgb_to_256 - nearest xterm-256 index: 6x6x6 cube entry or 232-255 grey. */
static int ov_rgb_to_256(int r, int g, int b)
{
    int ri = (r < 48) ? 0 : (r < 115) ? 1 : (r - 35) / 40;
    int gi = (g < 48) ? 0 : (g < 115) ? 1 : (g - 35) / 40;
    int bi = (b < 48) ? 0 : (b < 115) ? 1 : (b - 35) / 40;
    int gr = ((r + g + b) / 3 - 3) / 10;
    gr     = (gr < 0) ? 0 : (gr > 23) ? 23 : gr;
    int gv = 8 + 10 * gr;
    return (ov_dist2(r, g, b, gv, gv, gv) <
            ov_dist2(r, g, b, ov_cube[ri], ov_cube[gi], ov_cube[bi]))
               ? 232 + gr
               : 16 + 36 * ri + 6 * gi + bi;
}

/** ov_rgb_to_16 - nearest entry of the xterm default 16-colour palette. */
static int ov_rgb_to_16(int r, int g, int b)
{
    int best = 0, bd = 1 << 30;
    for (int i = 0; i < 16; i++)
    {
        int d = ov_dist2(r, g, b, ov_pal16[i][0], ov_pal16[i][1], ov_pal16[i][2]);
        if (d < bd)
        {
            bd   = d;
            best = i;
        }
    }
    return best;
}

/**
 * ov_color_downgrade - colour @col as the terminal can show it at @level (see ov_detect_color_level):
 * unchanged at 3, a 256 index at 2, a 16 index at 1, OV_COLOR_NONE at 0.
 */
uint32_t ov_color_downgrade(uint32_t col, int level)
{
    if (col == OV_COLOR_NONE || level >= 3)
    {
        return col;
    }
    if (level <= 0)
    {
        return OV_COLOR_NONE;
    }
    int r, g, b;
    if (col & OV_COLOR_TRUE)
    {
        r = (col >> 16) & 0xFF;
        g = (col >> 8) & 0xFF;
        b = col & 0xFF;
        if (level == 2)
        {
            return OV_COLOR_256 | (uint32_t) ov_rgb_to_256(r, g, b);
        }
    }
    else if (col & OV_COLOR_256)
    {
        int i = (int) (col & 0xFF);
        if (level == 2 || i < 16)
        {
            return (level == 2) ? col : (OV_COLOR_16 | (uint32_t) i);
        }
        if (i >= 232)
        {
            r = g = b = 8 + 10 * (i - 232);
        }
        else
        {
            r = ov_cube[(i - 16) / 36];
            g = ov_cube[((i - 16) / 6) % 6];
            b = ov_cube[(i - 16) % 6];
        }
    }
    else
    {
        return col; /* already a 16 index */
    }
    return OV_COLOR_16 | (uint32_t) ov_rgb_to_16(r, g, b);
}

/**
 * ov_buf_emit_sgr_delta - Emit ANSI SGR codes to transition between styling states
 * @sc:        Target cell style to transition to
 * @emit_attr: Current terminal attribute state pointer
 * @emit_fg:   Current terminal foreground color pointer
 * @emit_bg:   Current terminal background color pointer
 * @emit_ul:   Current terminal underline color pointer
 */
static void ov_buf_emit_sgr_delta(const OV_CELL *sc,
                                  uint8_t       *emit_attr,
                                  uint32_t      *emit_fg,
                                  uint32_t      *emit_bg,
                                  uint32_t      *emit_ul)
{
    int need_reset =
        ((*emit_attr & ~sc->attr) != 0 ||
         (sc->fg != *emit_fg && *emit_fg != OV_COLOR_NONE && sc->fg == OV_COLOR_NONE) ||
         (sc->bg != *emit_bg && *emit_bg != OV_COLOR_NONE && sc->bg == OV_COLOR_NONE) ||
         (sc->ul != *emit_ul && *emit_ul != OV_COLOR_NONE && sc->ul == OV_COLOR_NONE));

    if (!need_reset && sc->attr == *emit_attr && sc->fg == *emit_fg && sc->bg == *emit_bg &&
        sc->ul == *emit_ul)
    {
        return;
    }

    if (need_reset)
    {
        *emit_attr = 0;
        *emit_fg   = OV_COLOR_NONE;
        *emit_bg   = OV_COLOR_NONE;
        *emit_ul   = OV_COLOR_NONE;
    }

    char tmp[128];
    int  len   = 0;
    tmp[len++] = '\033';
    tmp[len++] = '[';
    int first  = 1;

    if (need_reset)
    {
        tmp[len++] = '0';
        first      = 0;
    }

    if (sc->attr != *emit_attr)
    {
        if ((sc->attr & OV_ATTR_BOLD) && !(*emit_attr & OV_ATTR_BOLD))
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            tmp[len++] = '1';
            first      = 0;
        }
        if ((sc->attr & OV_ATTR_DIM) && !(*emit_attr & OV_ATTR_DIM))
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            tmp[len++] = '2';
            first      = 0;
        }
        if ((sc->attr & OV_ATTR_ITALIC) && !(*emit_attr & OV_ATTR_ITALIC))
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            tmp[len++] = '3';
            first      = 0;
        }
        if ((sc->attr & OV_ATTR_UNDERLINE) && !(*emit_attr & OV_ATTR_UNDERLINE))
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            tmp[len++] = '4';
            first      = 0;
        }
        if ((sc->attr & OV_ATTR_REVERSE) && !(*emit_attr & OV_ATTR_REVERSE))
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            tmp[len++] = '7';
            first      = 0;
        }
        if ((sc->attr & OV_ATTR_BLINK) && !(*emit_attr & OV_ATTR_BLINK))
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            tmp[len++] = '5';
            first      = 0;
        }
        *emit_attr = sc->attr;
    }

    if (sc->fg != *emit_fg)
    {
        if (sc->fg != OV_COLOR_NONE)
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            if (sc->fg & OV_COLOR_TRUE)
            {
                len += snprintf(tmp + len, sizeof(tmp) - len, "38;2;%u;%u;%u",
                                (sc->fg >> 16) & 0xFF, (sc->fg >> 8) & 0xFF, sc->fg & 0xFF);
            }
            else if (sc->fg & OV_COLOR_256)
            {
                len += snprintf(tmp + len, sizeof(tmp) - len, "38;5;%u", sc->fg & 0xFF);
            }
            else if (sc->fg & OV_COLOR_16)
            {
                unsigned i = sc->fg & 0xF;
                len += snprintf(tmp + len, sizeof(tmp) - len, "%u", (i < 8) ? 30 + i : 90 + i - 8);
            }
            first = 0;
        }
        *emit_fg = sc->fg;
    }

    if (sc->bg != *emit_bg)
    {
        if (sc->bg != OV_COLOR_NONE)
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            if (sc->bg & OV_COLOR_TRUE)
            {
                len += snprintf(tmp + len, sizeof(tmp) - len, "48;2;%u;%u;%u",
                                (sc->bg >> 16) & 0xFF, (sc->bg >> 8) & 0xFF, sc->bg & 0xFF);
            }
            else if (sc->bg & OV_COLOR_256)
            {
                len += snprintf(tmp + len, sizeof(tmp) - len, "48;5;%u", sc->bg & 0xFF);
            }
            else if (sc->bg & OV_COLOR_16)
            {
                unsigned i = sc->bg & 0xF;
                len += snprintf(tmp + len, sizeof(tmp) - len, "%u", (i < 8) ? 40 + i : 100 + i - 8);
            }
            first = 0;
        }
        *emit_bg = sc->bg;
    }

    if (sc->ul != *emit_ul)
    {
        if (sc->ul != OV_COLOR_NONE)
        {
            if (!first)
            {
                tmp[len++] = ';';
            }
            if (sc->ul & OV_COLOR_TRUE)
            {
                len += snprintf(tmp + len, sizeof(tmp) - len, "58;2;%u;%u;%u",
                                (sc->ul >> 16) & 0xFF, (sc->ul >> 8) & 0xFF, sc->ul & 0xFF);
            }
            else if (sc->ul & OV_COLOR_256)
            {
                len += snprintf(tmp + len, sizeof(tmp) - len, "58;5;%u", sc->ul & 0xFF);
            }
            first = 0;
        }
        *emit_ul = sc->ul;
    }

    if (!first)
    {
        tmp[len++] = 'm';
        ov_buf_append(tmp, len);
    }
}

/**
 * ov_cell_width_trusted - 0 when terminals may disagree on the cell's width (emoji-like ranges,
 * multi-codepoint clusters, wide cells); the next changed cell then gets an explicit cursor move.
 */
static int ov_cell_width_trusted(const OV_CELL *sc)
{
    uint32_t cp = 0;
    if ((unsigned char) sc->ch[0] < 0x80)
    {
        return 1;
    }
    int n = ov_utf8_decode(sc->ch, (int) strlen(sc->ch), &cp);
    if (sc->width == 2 || sc->ch[n] != '\0')
    {
        return 0;
    }
    return !((cp >= 0x2300 && cp <= 0x23FF) || (cp >= 0x2600 && cp <= 0x27BF) ||
             (cp >= 0x2B00 && cp <= 0x2BFF) || cp >= 0x1F000);
}

/**
 * ov_buf_flush_delta - Diff shadow buffer against front buffer and render changes
 * @term_rows: Active terminal rows
 * @term_cols: Active terminal columns
 *
 * Emits cursor movements and minimal SGR styling deltas to stdout, utilizing
 * synchronized update escapes (mode 2026) to prevent screen tearing.
 */
void ov_buf_flush_delta(int term_rows, int term_cols)
{
    int      emit_cursor_r = -1;
    int      emit_cursor_c = -1;
    uint32_t emit_fg       = OV_COLOR_NONE;
    uint32_t emit_bg       = OV_COLOR_NONE;
    uint32_t emit_ul       = OV_COLOR_NONE;
    uint8_t  emit_attr     = 0;
    char     tmp[128];

    if (term_rows > OV_MAX_ROWS)
    {
        term_rows = OV_MAX_ROWS;
    }
    if (term_cols > OV_MAX_COLS)
    {
        term_cols = OV_MAX_COLS;
    }

    int level = (ov__color_level < 0) ? 3 : ov__color_level;

    // Start synchronized output
    ov_buf_append("\033[?2026h", 8);

    for (int r = 0; r < term_rows; r++)
    {
        for (int c = 0; c < term_cols; c++)
        {
            OV_CELL *sc = &ov__shadow[r][c];
            OV_CELL *fc = &ov__front[r][c];

            /* Continuation cell of a wide character: already drawn by column c-1 */
            if (sc->width == 0)
            {
                *fc = *sc;
                continue;
            }

            if (sc->ch[0] == '\0')
            {
                sc->ch[0] = ' ';
                sc->ch[1] = '\0'; // ensure valid char
                sc->width = 1;
            }

            if (sc->attr != fc->attr || sc->fg != fc->fg || sc->bg != fc->bg || sc->ul != fc->ul ||
                sc->width != fc->width || memcmp(sc->ch, fc->ch, sizeof(sc->ch)) != 0)
            {
                // Pos
                if (emit_cursor_r != r + 1 || emit_cursor_c != c + 1)
                {
                    int n = snprintf(tmp, sizeof(tmp), "\033[%d;%dH", r + 1, c + 1);
                    ov_buf_append(tmp, n);
                    emit_cursor_r = r + 1;
                    emit_cursor_c = c + 1;
                }

                // Batch SGR attributes and colors; below truecolor a converted copy is emitted (the
                // front buffer keeps the theme colours, so the diff is unchanged)
                if (level >= 3)
                {
                    ov_buf_emit_sgr_delta(sc, &emit_attr, &emit_fg, &emit_bg, &emit_ul);
                }
                else
                {
                    OV_CELL dc = *sc;
                    dc.fg      = ov_color_downgrade(sc->fg, level);
                    dc.bg      = ov_color_downgrade(sc->bg, level);
                    dc.ul = OV_COLOR_NONE; /* SGR 58 reads as blink without truecolor support */
                    if (level == 0 && sc->bg != OV_COLOR_NONE &&
                        (sc->bg == ov__mono_reverse_bg[0] || sc->bg == ov__mono_reverse_bg[1]))
                    {
                        dc.attr |= OV_ATTR_REVERSE; /* selection and freeze stay visible */
                    }
                    ov_buf_emit_sgr_delta(&dc, &emit_attr, &emit_fg, &emit_bg, &emit_ul);
                }

                // Char
                size_t chlen = strlen(sc->ch);
                ov_buf_append(sc->ch, (int) chlen);
                emit_cursor_c += sc->width;
                if (!ov_cell_width_trusted(sc))
                {
                    emit_cursor_c = -1;
                }

                *fc = *sc;
                if (sc->width == 2 && c + 1 < term_cols)
                {
                    ov__front[r][c + 1] = ov__shadow[r][c + 1];
                }
            }
        }
    }

    // Reset terminal state if we left it dirty so the next frame starts clean
    if (emit_attr != 0 || emit_fg != OV_COLOR_NONE || emit_bg != OV_COLOR_NONE ||
        emit_ul != OV_COLOR_NONE)
    {
        ov_buf_append("\033[0m", 4);
    }

    // End synchronized output
    ov_buf_append("\033[?2026l", 8);

    ov_buf_flush_internal();
}
