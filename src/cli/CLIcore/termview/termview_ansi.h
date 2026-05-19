/**
 * @file termview_ansi.h
 * @brief Inline helpers for ANSI TrueColor frame-buffer rendering.
 *
 * All functions write directly into a caller-supplied byte buffer and
 * advance *pos.  No heap allocation, no external dependencies.
 *
 * Typical usage:
 *
 *   char buf[BIG_ENOUGH];
 *   size_t pos = 0;
 *   tv_move(buf, &pos, 1, 1);
 *   tv_bg(buf, &pos, 0xFF, 0x80, 0x00);
 *   tv_halfblock(buf, &pos);
 *   tv_reset(buf, &pos);
 *   write(STDOUT_FILENO, buf, pos);
 */

#ifndef TERMVIEW_ANSI_H
#define TERMVIEW_ANSI_H

#include <stdint.h>
#include <stddef.h>

/* -----------------------------------------------------------------------
 * Cursor movement: \e[<row>;<col>H  (1-indexed)
 * Worst case: "\e[999;999H" = 10 bytes
 * ----------------------------------------------------------------------- */
static inline void tv_move(
    char   *buf,
    size_t *pos,
    int    row,
    int    col)
{
    /* Hand-rolled integer serialization avoids printf overhead */
    buf[(*pos)++] = '\033';
    buf[(*pos)++] = '[';

    /* row */
    if(row >= 100)
    {
        buf[(*pos)++] = (char)('0' + row / 100);
    }
    if(row >= 10)
    {
        buf[(*pos)++] = (char)('0' + (row / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + row % 10);

    buf[(*pos)++] = ';';

    /* col */
    if(col >= 100)
    {
        buf[(*pos)++] = (char)('0' + col / 100);
    }
    if(col >= 10)
    {
        buf[(*pos)++] = (char)('0' + (col / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + col % 10);

    buf[(*pos)++] = 'H';
}

/* -----------------------------------------------------------------------
 * TrueColor background: \e[48;2;<R>;<G>;<B>m
 * Worst case: "\e[48;2;255;255;255m" = 19 bytes
 * ----------------------------------------------------------------------- */
static inline void tv_bg(
    char    *buf,
    size_t  *pos,
    uint8_t r,
    uint8_t g,
    uint8_t b)
{
    /* Static prefix bytes */
    buf[(*pos)++] = '\033';
    buf[(*pos)++] = '[';
    buf[(*pos)++] = '4';
    buf[(*pos)++] = '8';
    buf[(*pos)++] = ';';
    buf[(*pos)++] = '2';
    buf[(*pos)++] = ';';

    /* R */
    if(r >= 100)
    {
        buf[(*pos)++] = (char)('0' + r / 100);
    }
    if(r >= 10)
    {
        buf[(*pos)++] = (char)('0' + (r / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + r % 10);
    buf[(*pos)++] = ';';

    /* G */
    if(g >= 100)
    {
        buf[(*pos)++] = (char)('0' + g / 100);
    }
    if(g >= 10)
    {
        buf[(*pos)++] = (char)('0' + (g / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + g % 10);
    buf[(*pos)++] = ';';

    /* B */
    if(b >= 100)
    {
        buf[(*pos)++] = (char)('0' + b / 100);
    }
    if(b >= 10)
    {
        buf[(*pos)++] = (char)('0' + (b / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + b % 10);

    buf[(*pos)++] = 'm';
}

/* -----------------------------------------------------------------------
 * TrueColor foreground: \e[38;2;<R>;<G>;<B>m
 * Worst case: "\e[38;2;255;255;255m" = 19 bytes
 * ----------------------------------------------------------------------- */
static inline void tv_fg(
    char    *buf,
    size_t  *pos,
    uint8_t r,
    uint8_t g,
    uint8_t b)
{
    buf[(*pos)++] = '\033';
    buf[(*pos)++] = '[';
    buf[(*pos)++] = '3';
    buf[(*pos)++] = '8';
    buf[(*pos)++] = ';';
    buf[(*pos)++] = '2';
    buf[(*pos)++] = ';';

    if(r >= 100)
    {
        buf[(*pos)++] = (char)('0' + r / 100);
    }
    if(r >= 10)
    {
        buf[(*pos)++] = (char)('0' + (r / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + r % 10);
    buf[(*pos)++] = ';';

    if(g >= 100)
    {
        buf[(*pos)++] = (char)('0' + g / 100);
    }
    if(g >= 10)
    {
        buf[(*pos)++] = (char)('0' + (g / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + g % 10);
    buf[(*pos)++] = ';';

    if(b >= 100)
    {
        buf[(*pos)++] = (char)('0' + b / 100);
    }
    if(b >= 10)
    {
        buf[(*pos)++] = (char)('0' + (b / 10) % 10);
    }
    buf[(*pos)++] = (char)('0' + b % 10);

    buf[(*pos)++] = 'm';
}

/* -----------------------------------------------------------------------
 * Lower-half block ▄ (U+2584): UTF-8 = 0xE2 0x96 0x84  (3 bytes)
 *
 * The terminal renders this character's top half in the BACKGROUND color
 * and the bottom half in the FOREGROUND color, giving 2 image rows per
 * character row.
 * ----------------------------------------------------------------------- */
static inline void tv_halfblock(
    char   *buf,
    size_t *pos)
{
    buf[(*pos)++] = (char)0xE2;
    buf[(*pos)++] = (char)0x96;
    buf[(*pos)++] = (char)0x84;
}

/* -----------------------------------------------------------------------
 * Full block █ (U+2588): UTF-8 = 0xE2 0x96 0x88  (3 bytes)
 * Used for the colorbar and single-row fallback.
 * ----------------------------------------------------------------------- */
static inline void tv_fullblock(
    char   *buf,
    size_t *pos)
{
    buf[(*pos)++] = (char)0xE2;
    buf[(*pos)++] = (char)0x96;
    buf[(*pos)++] = (char)0x88;
}

/* -----------------------------------------------------------------------
 * Reset all attributes: \e[0m  (4 bytes)
 * ----------------------------------------------------------------------- */
static inline void tv_reset(
    char   *buf,
    size_t *pos)
{
    buf[(*pos)++] = '\033';
    buf[(*pos)++] = '[';
    buf[(*pos)++] = '0';
    buf[(*pos)++] = 'm';
}

/* -----------------------------------------------------------------------
 * Emit a plain ASCII space with the current attribute state (1 byte).
 * ----------------------------------------------------------------------- */
static inline void tv_space(
    char   *buf,
    size_t *pos)
{
    buf[(*pos)++] = ' ';
}

/* -----------------------------------------------------------------------
 * Emit a newline (1 byte).
 * ----------------------------------------------------------------------- */
static inline void tv_newline(
    char   *buf,
    size_t *pos)
{
    buf[(*pos)++] = '\n';
}

/* -----------------------------------------------------------------------
 * Compute required frame-buffer size for a given terminal size.
 *
 * Each cell can emit at most:
 *   tv_bg:        19 bytes
 *   tv_fg:        19 bytes
 *   tv_halfblock:  3 bytes
 *   tv_reset:      4 bytes
 * = 45 bytes per cell.  Multiply by cell count and add headroom for
 * cursor-positioning escapes and the info bar.
 * ----------------------------------------------------------------------- */
static inline size_t tv_framebuf_size(
    int rows,
    int cols)
{
    return (size_t)(rows * cols * 48 + rows * 16 + 4096);
}

#endif /* TERMVIEW_ANSI_H */
