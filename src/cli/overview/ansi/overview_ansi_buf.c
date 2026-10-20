// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"
#include <time.h>

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
 * ov_cell_blank - turn a cell into a single-width space, keeping its colours.
 */
static void ov_cell_blank(OV_CELL *c)
{
    memset(c->ch, 0, sizeof(c->ch));
    c->ch[0] = ' ';
    c->width = 1;
}

static int s_drawn_rows, s_drawn_cols; /* largest area drawn so far: the part of the grids in use */

/**
 * ov_buf_force_clear - Invalidate the front buffer to force full screen repaint
 *
 * Only the area drawn so far is cleared: untouched cells are zero already, and clearing the whole
 * 256x512 grid made all 4 MiB resident.
 */
void ov_buf_force_clear(void)
{
    for (int r = 0; r < s_drawn_rows; r++)
    {
        memset(ov__front[r], 0, (size_t) s_drawn_cols * sizeof(OV_CELL));
    }
}

/**
 * ov_buf_reset_size - Reset shadow buffer dimensions and clear cells
 * @rows: Target terminal rows
 * @cols: Target terminal columns
 */
void ov_buf_reset_size(int rows, int cols)
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

    s_drawn_rows = (rows > s_drawn_rows) ? rows : s_drawn_rows;
    s_drawn_cols = (cols > s_drawn_cols) ? cols : s_drawn_cols;
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
void ov_buf_append(const char *data, int len)
{
    if (ov__screenbuf_len + len >= OV_SCREENBUF_SIZE)
    {
        ov_buf_flush_internal(); /* still inside ?2026, so the frame stays atomic */
    }
    if (ov__screenbuf_len + len >= OV_SCREENBUF_SIZE)
    {
        ov_buf_force_clear(); /* cannot be sent: repaint everything next frame */
        return;
    }
    memcpy(ov__screenbuf + ov__screenbuf_len, data, (size_t) len);
    ov__screenbuf_len += len;
}

/**
 * ov_buf_flush_internal - Flush accumulated screen output buffer to the terminal
 */
void ov_buf_flush_internal(void)
{
    if (ov__screenbuf_len > 0)
    {
        int written = 0;
        while (written < ov__screenbuf_len)
        {
            ssize_t ret =
                write(ov__tty_fd, ov__screenbuf + written, (size_t) (ov__screenbuf_len - written));
            if (ret < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    struct pollfd pfd;
                    pfd.fd     = ov__tty_fd;
                    pfd.events = POLLOUT;
                    poll(&pfd, 1, 100);
                    continue;
                }
                ov_buf_force_clear(); /* the front buffer no longer matches the terminal */
                break;
            }
            if (ret == 0)
            {
                ov_buf_force_clear();
                break;
            }
            written += ret;
            ov__total_bytes_rendered += ret;
        }
        ov__screenbuf_len = 0;
    }
}

/**
 * ov_sys_get_bandwidth_usage - compute terminal delta rendering bandwidth.
 *
 * Return: Smoothed bandwidth in kB/s.
 */
double ov_sys_get_bandwidth_usage(void)
{
    static struct timespec last_time;
    static uint64_t        last_bytes  = 0;
    static int             initialized = 0;
    static double          smoothed_bw = 0.0;

    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    if (!initialized)
    {
        last_time   = current_time;
        last_bytes  = ov__total_bytes_rendered;
        initialized = 1;
        return 0.0;
    }

    double dt =
        (current_time.tv_sec - last_time.tv_sec) + (current_time.tv_nsec - last_time.tv_nsec) / 1e9;

    if (dt >= 0.5) /* update every 0.5s */
    {
        uint64_t d_bytes = ov__total_bytes_rendered - last_bytes;
        double   inst_bw = (double) d_bytes / 1024.0 / dt;
        smoothed_bw      = inst_bw;
        last_bytes       = ov__total_bytes_rendered;
        last_time        = current_time;
    }
    return smoothed_bw;
}

/**
 * ov_buf_append_cluster - Append a single grapheme cluster to the shadow buffer
 * @utf8_seq: Grapheme cluster bytes
 * @bytes:    Length of sequence in bytes
 * @width:    Display column width (1 or 2)
 */
void ov_buf_append_cluster(const char *utf8_seq, int bytes, int width)
{
    if (width <= 0)
    {
        return;
    }
    if (ov__cursor_row >= 1 && ov__cursor_row <= OV_MAX_ROWS && ov__cursor_col >= 1 &&
        ov__cursor_col <= OV_MAX_COLS)
    {
        OV_CELL *row  = ov__shadow[ov__cursor_row - 1];
        int      c    = ov__cursor_col - 1;
        OV_CELL *cell = &row[c];

        /* Keep the invariant: a width-2 cell is always followed by its width-0 continuation */
        if (cell->width == 0 && c > 0)
        {
            ov_cell_blank(&row[c - 1]); /* overwriting the right half of a wide glyph */
        }
        if (cell->width == 2 && width == 1 && c + 1 < OV_MAX_COLS)
        {
            ov_cell_blank(&row[c + 1]); /* orphaned continuation */
        }
        if (width == 2 && c + 2 < OV_MAX_COLS && row[c + 1].width == 2)
        {
            ov_cell_blank(&row[c + 2]); /* our right half lands on another wide glyph */
        }
        if (bytes >= (int) sizeof(cell->ch))
        {
            bytes = (int) sizeof(cell->ch) - 1;
        }
        unsigned char b0 = (unsigned char) utf8_seq[0];
        if ((b0 < 0x20 || b0 >= 0x7F) && !ov_utf8_cluster_printable(utf8_seq, bytes))
        {
            utf8_seq = "?"; /* never emit control bytes or broken UTF-8 taken from shm strings */
            bytes    = 1;
            width    = 1;
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
void ov_buf_append_char(const char *utf8_seq, int bytes)
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
void ov_buf_printf(const char *fmt, ...)
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
void ov_buf_hline(char ch, int len)
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
void ov_buf_hline_utf8(const char *s, int len)
{
    int slen = (int) strlen(s);
    for (int i = 0; i < len; i++)
    {
        ov_buf_append_char(s, slen);
    }
}
