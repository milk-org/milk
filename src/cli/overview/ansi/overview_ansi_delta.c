// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"
/**
 * ov_buf_emit_sgr_delta - Emit ANSI SGR codes to transition between styling states
 * @sc:        Target cell style to transition to
 * @emit_attr: Current terminal attribute state pointer
 * @emit_fg:   Current terminal foreground color pointer
 * @emit_bg:   Current terminal background color pointer
 * @emit_ul:   Current terminal underline color pointer
 */
static void ov_buf_emit_sgr_delta(
    const OV_CELL *sc,
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
 * ov_buf_flush_delta - Diff shadow buffer against front buffer and render changes
 * @term_rows: Active terminal rows
 * @term_cols: Active terminal columns
 *
 * Emits cursor movements and minimal SGR styling deltas to stdout, utilizing
 * synchronized update escapes (mode 2026) to prevent screen tearing.
 */
void ov_buf_flush_delta(
    int term_rows,
    int term_cols)
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

                // Batch SGR attributes and colors
                ov_buf_emit_sgr_delta(sc, &emit_attr, &emit_fg, &emit_bg, &emit_ul);

                // Char
                size_t chlen = strlen(sc->ch);
                ov_buf_append(sc->ch, (int) chlen);
                emit_cursor_c += sc->width;

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
