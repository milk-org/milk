// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"

char     ov__screenbuf[OV_SCREENBUF_SIZE];
int      ov__screenbuf_len        = 0;
uint64_t ov__total_bytes_rendered = 0;

OV_CELL  ov__shadow[OV_MAX_ROWS][OV_MAX_COLS];
OV_CELL  ov__front[OV_MAX_ROWS][OV_MAX_COLS];
int      ov__cursor_row   = 1;
int      ov__cursor_col   = 1;
uint32_t ov__current_fg   = OV_COLOR_NONE;
uint32_t ov__current_bg   = OV_COLOR_NONE;
uint32_t ov__default_bg   = OV_COLOR_NONE;
uint32_t ov__current_ul   = OV_COLOR_NONE;
uint8_t  ov__current_attr = 0;
/**
 * ov_buf_force_clear - Invalidate the front buffer to force full screen repaint
 */
void ov_buf_force_clear(void)
{
    memset(ov__front, 0, sizeof(ov__front));
}

/**
 * ov_buf_reset_size - Reset shadow buffer dimensions and clear cells
 * @rows: Target terminal rows
 * @cols: Target terminal columns
 */
void ov_buf_reset_size(
    int rows,
    int cols)
{
    ov__screenbuf_len = 0;
    ov__cursor_row    = 1;
    ov__cursor_col    = 1;
    ov__current_fg    = OV_COLOR_NONE;
    ov__current_bg    = ov__default_bg;
    ov__current_ul    = OV_COLOR_NONE;
    ov__current_attr  = 0;

    if (rows <= 0 || rows > OV_MAX_ROWS)
    {
        rows = OV_MAX_ROWS;
    }
    if (cols <= 0 || cols > OV_MAX_COLS)
    {
        cols = OV_MAX_COLS;
    }

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            memset(&ov__shadow[r][c], 0, sizeof(OV_CELL));
            ov__shadow[r][c].ch[0] = ' ';
            ov__shadow[r][c].ch[1] = '\0';
            ov__shadow[r][c].width = 1;
            ov__shadow[r][c].fg    = OV_COLOR_NONE;
            ov__shadow[r][c].bg    = ov__default_bg;
            ov__shadow[r][c].ul    = OV_COLOR_NONE;
            ov__shadow[r][c].attr  = 0;
        }
    }
}

/**
 * ov_buf_reset - Reset shadow screen buffer to max terminal dimensions
 */
void ov_buf_reset(void)
{
    ov_buf_reset_size(OV_MAX_ROWS, OV_MAX_COLS);
}

/**
 * ov_buf_append - Append raw ANSI sequence bytes to the screen output buffer
 * @data: Byte buffer containing ANSI sequences or text
 * @len:  Number of bytes to append
 */
void ov_buf_append(
    const char *data,
    int         len)
{
    if (ov__screenbuf_len + len < OV_SCREENBUF_SIZE)
    {
        memcpy(ov__screenbuf + ov__screenbuf_len, data, (size_t) len);
        ov__screenbuf_len += len;
    }
}

/**
 * ov_buf_flush_internal - Flush accumulated screen output buffer to stdout
 */
void ov_buf_flush_internal(void)
{
    if (ov__screenbuf_len > 0)
    {
        int written = 0;
        while (written < ov__screenbuf_len)
        {
            ssize_t ret = write(STDOUT_FILENO, ov__screenbuf + written,
                                (size_t) (ov__screenbuf_len - written));
            if (ret < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    struct pollfd pfd;
                    pfd.fd     = STDOUT_FILENO;
                    pfd.events = POLLOUT;
                    poll(&pfd, 1, 100);
                    continue;
                }
                break;
            }
            if (ret == 0)
            {
                break;
            }
            written += ret;
            ov__total_bytes_rendered += ret;
        }
        ov__screenbuf_len = 0;
    }
}

/**
 * ov_buf_append_cluster - Append a single grapheme cluster to the shadow buffer
 * @utf8_seq: Grapheme cluster bytes
 * @bytes:    Length of sequence in bytes
 * @width:    Display column width (1 or 2)
 */
void ov_buf_append_cluster(
    const char *utf8_seq,
    int         bytes,
    int         width)
{
    if (width <= 0)
    {
        return;
    }
    if (ov__cursor_row >= 1 && ov__cursor_row <= OV_MAX_ROWS && ov__cursor_col >= 1 &&
        ov__cursor_col <= OV_MAX_COLS)
    {
        OV_CELL *cell = &ov__shadow[ov__cursor_row - 1][ov__cursor_col - 1];
        if (bytes >= (int) sizeof(cell->ch))
        {
            bytes = (int) sizeof(cell->ch) - 1;
        }
        memset(cell->ch, 0, sizeof(cell->ch));
        memcpy(cell->ch, utf8_seq, (size_t) bytes);
        cell->width = (uint8_t) width;
        cell->fg    = ov__current_fg;
        cell->bg    = ov__current_bg;
        cell->ul    = ov__current_ul;
        cell->attr  = ov__current_attr;

        if (width == 2 && ov__cursor_col < OV_MAX_COLS)
        {
            OV_CELL *cont = &ov__shadow[ov__cursor_row - 1][ov__cursor_col];
            memset(cont->ch, 0, sizeof(cont->ch));
            cont->width = 0;
            cont->fg    = ov__current_fg;
            cont->bg    = ov__current_bg;
            cont->ul    = ov__current_ul;
            cont->attr  = ov__current_attr;
        }
    }
    ov__cursor_col += width;
}

/**
 * ov_buf_append_char - Append a UTF-8 character sequence to the shadow buffer
 * @utf8_seq: Character bytes
 * @bytes:    Length in bytes
 */
void ov_buf_append_char(
    const char *utf8_seq,
    int         bytes)
{
    int b = 0, w = 1;
    ov_utf8_next_cluster(utf8_seq, bytes, &b, &w);
    ov_buf_append_cluster(utf8_seq, bytes, w);
}

/**
 * ov_buf_printf - Format and print string into shadow buffer at current cursor
 * @fmt: Printf format string
 * @...: Variable arguments
 */
void ov_buf_printf(
    const char *fmt,
    ...)
{
    char    tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    if (n > 0)
    {
        if (n >= (int) sizeof(tmp))
        {
            n = (int) sizeof(tmp) - 1;
        }
        int i = 0;
        while (i < n)
        {
            int b = 0, w = 1;
            ov_utf8_next_cluster(&tmp[i], n - i, &b, &w);
            if (b <= 0)
            {
                break;
            }
            ov_buf_append_cluster(&tmp[i], b, w);
            i += b;
        }
    }
}

/**
 * ov_buf_hline - Draw horizontal line of ASCII characters in shadow buffer
 * @ch:  Character to draw
 * @len: Number of columns to fill
 */
void ov_buf_hline(
    char ch,
    int  len)
{
    for (int i = 0; i < len; i++)
    {
        ov_buf_append_char(&ch, 1);
    }
}

/**
 * ov_buf_hline_utf8 - Draw horizontal line of UTF-8 glyphs in shadow buffer
 * @s:   UTF-8 character string
 * @len: Number of repetitions
 */
void ov_buf_hline_utf8(
    const char *s,
    int         len)
{
    int slen = (int) strlen(s);
    for (int i = 0; i < len; i++)
    {
        ov_buf_append_char(s, slen);
    }
}
