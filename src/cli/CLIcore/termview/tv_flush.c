// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_flush.c
 * @brief Diff renderer: emits only the cells that changed, inside one synchronized update.
 */

#include "termview_ansi.h"
#include "termview_internal.h"

static const char *const tv_glyph_utf8[TV_G_COUNT - 0x80] = {
    [TV_G_LOWER_HALF - 0x80] = "\xE2\x96\x84",
    [TV_G_FULL_BLOCK - 0x80] = "\xE2\x96\x88",
};

size_t tv_framebuf_size(int rows, int cols)
{
    /* move (14) + bg (19) + fg (19) + glyph (3) per cell, plus sync markers and reset */
    return (size_t) rows * (size_t) cols * 56 + (size_t) rows * 16 + 4096;
}

size_t tv_flush_diff(char *out, const tv_cell_t *cur, tv_cell_t *prev, int rows, int cols)
{
    static const char sync_on[]  = "\033[?2026h";
    static const char sync_off[] = "\033[?2026l";

    size_t   pos   = sizeof(sync_on) - 1;
    int      cur_r = -1, cur_c = -1;
    uint32_t last_bg = UINT32_MAX, last_fg = UINT32_MAX;
    int      changed = 0;

    memcpy(out, sync_on, sizeof(sync_on) - 1);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            size_t    i    = (size_t) r * cols + c;
            tv_cell_t cell = cur[i];
            if (cell == prev[i])
            {
                continue;
            }
            changed = 1;
            prev[i] = cell;

            if (cur_r != r || cur_c != c)
            {
                tv_move(out, &pos, r + 1, c + 1);
            }
            uint32_t bg = tv_cell_bg(cell);
            if (bg != last_bg)
            {
                tv_bg(out, &pos, (uint8_t) (bg >> 16), (uint8_t) (bg >> 8), (uint8_t) bg);
                last_bg = bg;
            }
            uint32_t fg = tv_cell_fg(cell);
            if (fg != last_fg)
            {
                tv_fg(out, &pos, (uint8_t) (fg >> 16), (uint8_t) (fg >> 8), (uint8_t) fg);
                last_fg = fg;
            }

            uint8_t g = tv_cell_glyph(cell);
            if (g >= 0x80 && g < TV_G_COUNT && tv_glyph_utf8[g - 0x80] != NULL)
            {
                for (const char *u = tv_glyph_utf8[g - 0x80]; *u != '\0'; u++)
                {
                    out[pos++] = *u;
                }
            }
            else
            {
                out[pos++] = (g >= 0x20 && g < 0x7F) ? (char) g : ' ';
            }
            cur_r = r;
            cur_c = c + 1;
        }
    }
    if (!changed)
    {
        return 0;
    }

    tv_reset(out, &pos);
    memcpy(out + pos, sync_off, sizeof(sync_off) - 1);
    return pos + sizeof(sync_off) - 1;
}
