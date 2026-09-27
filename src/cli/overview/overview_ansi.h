// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_ANSI_H
#define OVERVIEW_ANSI_H

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif
#ifndef _XOPEN_SOURCE
#    define _XOPEN_SOURCE 700
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <poll.h>
#include <wchar.h>

extern int wcwidth(wchar_t c);

/* =========================================================
 * Key code constants
 * ========================================================= */

#define OV_KEY_NONE 0
#define OV_KEY_EOF (-1)
#define OV_KEY_UP 256
#define OV_KEY_DOWN 257
#define OV_KEY_LEFT 258
#define OV_KEY_RIGHT 259
#define OV_KEY_PGUP 260
#define OV_KEY_PGDN 261
#define OV_KEY_HOME 262
#define OV_KEY_END 263
#define OV_KEY_DEL 264
#define OV_KEY_F1 265
#define OV_KEY_F2 266
#define OV_KEY_F3 267
#define OV_KEY_F4 268
#define OV_KEY_F5 269
#define OV_KEY_F6 270
#define OV_KEY_F7 271
#define OV_KEY_F8 272
#define OV_KEY_TAB 9
#define OV_KEY_ENTER 10
#define OV_KEY_ESC 27
#define OV_KEY_CTRL_LEFT 277
#define OV_KEY_CTRL_RIGHT 278
#define OV_KEY_SHIFT_LEFT 279
#define OV_KEY_SHIFT_RIGHT 280
#define OV_KEY_SHIFT_UP 284
#define OV_KEY_SHIFT_DOWN 285
#define OV_KEY_BTAB 286

#define OV_KEY_MOUSE_CLICK 281
#define OV_KEY_MOUSE_UP 282
#define OV_KEY_MOUSE_DOWN 283
#define OV_KEY_MOUSE_DRAG 287
#define OV_KEY_MOUSE_RELEASE 288
#define OV_KEY_CTRL_SCROLL_UP 289
#define OV_KEY_CTRL_SCROLL_DOWN 290
#define OV_KEY_MOUSE_MOVE 291

extern int ov_mouse_row;
extern int ov_mouse_col;
extern int ov_mouse_btn;
extern int ov_hover_row;
extern int ov_hover_col;

#ifndef ctrl
#    define ctrl(x) ((x) & 0x1f)
#endif

/* =========================================================
 * Terminal state
 * ========================================================= */

extern struct termios ov__orig_termios;
extern int            ov__raw_active;

static inline void ov_raw_mode_enter(void)
{
    struct termios raw;
    if (ov__raw_active)
    {
        return;
    }
    if (tcgetattr(STDIN_FILENO, &ov__orig_termios) == -1)
    {
        return;
    }
    raw = ov__orig_termios;
    raw.c_iflag &= ~(unsigned int) (IXON | ICRNL | BRKINT | INPCK | ISTRIP);
    raw.c_oflag &= ~(unsigned int) (OPOST);
    raw.c_cflag |= (unsigned int) (CS8);
    raw.c_lflag &= ~(unsigned int) (ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    const char seq[] = "\033[?1049h\033[?25l\033[?7l\033[?1002h\033[?1006h";
    if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0)
    {
    }
    ov__raw_active = 1;
}

static inline void ov_raw_mode_exit(void)
{
    if (!ov__raw_active)
    {
        return;
    }
    const char seq[] = "\033[?1003l\033[?1006l\033[?1002l\033[?25h\033[?7h\033[0m\033[?1049l";
    if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0)
    {
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &ov__orig_termios);
    ov__raw_active = 0;
}

static inline void ov_set_mouse_hover(int enable)
{
    if (enable)
    {
        const char seq[] = "\033[?1002l\033[?1003h";
        if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0)
        {
        }
    }
    else
    {
        const char seq[] = "\033[?1003l\033[?1002h";
        if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0)
        {
        }
    }
}

static inline void ov_get_terminal_size(int *rows, int *cols)
{
    struct winsize ws;
    *rows = 24;
    *cols = 80;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0)
    {
        if (ws.ws_row > 0)
        {
            *rows = (int) ws.ws_row;
        }
        if (ws.ws_col > 0)
        {
            *cols = (int) ws.ws_col;
        }
    }
}

/* =========================================================
 * Buffered screen writer (Delta Rendering Shadow Buffer)
 * ========================================================= */

#define OV_SCREENBUF_SIZE (2 * 1024 * 1024)

#define OV_MAX_ROWS 256
#define OV_MAX_COLS 512

#define OV_COLOR_NONE 0xFFFFFFFF // Unset color

#define OV_COLOR_256 0x01000000  // Flag for 256-color
#define OV_COLOR_TRUE 0x02000000 // Flag for TrueColor

#define OV_ATTR_BOLD (1 << 0)
#define OV_ATTR_DIM (1 << 1)
#define OV_ATTR_ITALIC (1 << 2)
#define OV_ATTR_UNDERLINE (1 << 3)
#define OV_ATTR_REVERSE (1 << 4)
#define OV_ATTR_BLINK (1 << 5)

typedef struct
{
    char     ch[16]; // UTF-8 char/cluster up to 15 bytes + null terminator
    uint8_t  width;  // visual display width: 1 or 2 (0 for continuation cell)
    uint8_t  attr;   // bitmask for BOLD, DIM, REVERSE, etc.
    uint32_t fg;     // Color code + flag
    uint32_t bg;     // Color code + flag
    uint32_t ul;     // Underline color
} OV_CELL;

extern char     ov__screenbuf[OV_SCREENBUF_SIZE];
extern int      ov__screenbuf_len;
extern uint64_t ov__total_bytes_rendered;

extern OV_CELL ov__shadow[OV_MAX_ROWS][OV_MAX_COLS];
extern OV_CELL ov__front[OV_MAX_ROWS][OV_MAX_COLS];

extern int      ov__cursor_row; // 1-based
extern int      ov__cursor_col; // 1-based
extern uint32_t ov__current_fg;
extern uint32_t ov__current_bg;
extern uint32_t ov__current_ul;
extern uint8_t  ov__current_attr;

static inline void ov_buf_force_clear(void)
{
    memset(ov__front, 0, sizeof(ov__front));
}

static inline void ov_buf_reset_size(int rows, int cols)
{
    ov__screenbuf_len = 0;
    ov__cursor_row    = 1;
    ov__cursor_col    = 1;
    ov__current_fg    = OV_COLOR_NONE;
    ov__current_bg    = OV_COLOR_NONE;
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
            ov__shadow[r][c].bg    = OV_COLOR_NONE;
            ov__shadow[r][c].ul    = OV_COLOR_NONE;
            ov__shadow[r][c].attr  = 0;
        }
    }
}

static inline void ov_buf_reset(void)
{
    ov_buf_reset_size(OV_MAX_ROWS, OV_MAX_COLS);
}

static inline void ov_buf_append(const char *data, int len)
{
    if (ov__screenbuf_len + len < OV_SCREENBUF_SIZE)
    {
        memcpy(ov__screenbuf + ov__screenbuf_len, data, (size_t) len);
        ov__screenbuf_len += len;
    }
}

static inline void ov_buf_flush_internal(void)
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

static inline void ov_buf_emit_sgr_delta(
    const OV_CELL *sc,
    uint8_t       *emit_attr,
    uint32_t      *emit_fg,
    uint32_t      *emit_bg,
    uint32_t      *emit_ul)
{
    int need_reset = ((*emit_attr & ~sc->attr) != 0 ||
                      (sc->fg != *emit_fg && *emit_fg != OV_COLOR_NONE &&
                       sc->fg == OV_COLOR_NONE) ||
                      (sc->bg != *emit_bg && *emit_bg != OV_COLOR_NONE &&
                       sc->bg == OV_COLOR_NONE) ||
                      (sc->ul != *emit_ul && *emit_ul != OV_COLOR_NONE &&
                       sc->ul == OV_COLOR_NONE));

    if (!need_reset && sc->attr == *emit_attr && sc->fg == *emit_fg &&
        sc->bg == *emit_bg && sc->ul == *emit_ul)
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

static inline void ov_buf_flush_delta(int term_rows, int term_cols)
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

            if (sc->attr != fc->attr || sc->fg != fc->fg || sc->bg != fc->bg ||
                sc->ul != fc->ul || sc->width != fc->width ||
                memcmp(sc->ch, fc->ch, sizeof(sc->ch)) != 0)
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

static inline int utf8_char_length(unsigned char c)
{
    if ((c & 0x80) == 0)
    {
        return 1;
    }
    if ((c & 0xE0) == 0xC0)
    {
        return 2;
    }
    if ((c & 0xF0) == 0xE0)
    {
        return 3;
    }
    if ((c & 0xF8) == 0xF0)
    {
        return 4;
    }
    return 1;
}

static inline int ov_utf8_decode(
    const char *s,
    int         len,
    uint32_t   *cp)
{
    if (len <= 0)
    {
        *cp = 0;
        return 0;
    }
    unsigned char c = (unsigned char) s[0];
    if (c < 0x80)
    {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && len >= 2)
    {
        *cp = (uint32_t) (((c & 0x1F) << 6) | (s[1] & 0x3F));
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && len >= 3)
    {
        *cp = (uint32_t) (((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F));
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && len >= 4)
    {
        *cp = (uint32_t) (((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) |
                          ((s[2] & 0x3F) << 6) | (s[3] & 0x3F));
        return 4;
    }
    *cp = c;
    return 1;
}

static inline int ov_utf8_next_cluster(
    const char *s,
    int         max_len,
    int        *bytes_out,
    int        *width_out)
{
    if (max_len <= 0 || s[0] == '\0')
    {
        *bytes_out = 0;
        *width_out = 0;
        return 0;
    }

    uint32_t cp0         = 0;
    int      b0          = ov_utf8_decode(s, max_len, &cp0);
    int      total_bytes = b0;
    int      has_vs16    = 0;

    /* Consume trailing modifiers: VS16 (U+FE0F), Keycap (U+20E3), etc. */
    while (total_bytes < max_len && total_bytes < 15)
    {
        uint32_t next_cp = 0;
        int      nb      = ov_utf8_decode(s + total_bytes, max_len - total_bytes, &next_cp);
        if (next_cp == 0xFE0F)
        {
            has_vs16 = 1;
            total_bytes += nb;
        }
        else if (next_cp == 0x20E3 || next_cp == 0xFE0E)
        {
            total_bytes += nb;
        }
        else
        {
            break;
        }
    }

    int w = 1;
    if (has_vs16)
    {
        w = 2;
    }
    else if (cp0 >= 0x1F000 && cp0 <= 0x1FAFF)
    {
        /* Standard emoji symbols / pictographs */
        w = 2;
    }
    else if (cp0 >= 0x2600 && cp0 <= 0x27BF)
    {
        if (cp0 == 0x2705 || cp0 == 0x274C || cp0 == 0x274E ||
            (cp0 >= 0x2753 && cp0 <= 0x2755) || cp0 == 0x2757 ||
            cp0 == 0x2728 || cp0 == 0x26A0 || cp0 == 0x26A1 ||
            cp0 == 0x26BD || cp0 == 0x26BE || cp0 == 0x26C4 ||
            cp0 == 0x26C5 || cp0 == 0x26D4 || cp0 == 0x26EA ||
            cp0 == 0x26F2 || cp0 == 0x26F3 || cp0 == 0x26F5 ||
            cp0 == 0x26FA || cp0 == 0x26FD)
        {
            w = 2;
        }
        else
        {
            int sys_w = wcwidth((wchar_t) cp0);
            w         = (sys_w == 2) ? 2 : 1;
        }
    }
    else
    {
        int sys_w = wcwidth((wchar_t) cp0);
        w         = (sys_w == 2) ? 2 : 1;
    }

    *bytes_out = total_bytes;
    *width_out = w;
    return 1;
}

static inline int ov_str_display_width(const char *s)
{
    if (s == NULL)
    {
        return 0;
    }
    int len     = (int) strlen(s);
    int total_w = 0;
    int pos     = 0;
    while (pos < len)
    {
        int b = 0, w = 0;
        if (!ov_utf8_next_cluster(s + pos, len - pos, &b, &w))
        {
            break;
        }
        total_w += w;
        pos += b;
    }
    return total_w;
}

static inline void ov_buf_append_cluster(
    const char *utf8_seq,
    int         bytes,
    int         width)
{
    if (width <= 0)
    {
        return;
    }
    if (ov__cursor_row >= 1 && ov__cursor_row <= OV_MAX_ROWS &&
        ov__cursor_col >= 1 && ov__cursor_col <= OV_MAX_COLS)
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
            cont->width   = 0;
            cont->fg      = ov__current_fg;
            cont->bg      = ov__current_bg;
            cont->ul      = ov__current_ul;
            cont->attr    = ov__current_attr;
        }
    }
    ov__cursor_col += width;
}

static inline void ov_buf_append_char(
    const char *utf8_seq,
    int         bytes)
{
    int b = 0, w = 1;
    ov_utf8_next_cluster(utf8_seq, bytes, &b, &w);
    ov_buf_append_cluster(utf8_seq, bytes, w);
}

static inline void ov_buf_printf(
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

/* =========================================================
 * Buffered color / attribute helpers
 * ========================================================= */

static inline void ov_buf_fg(int r, int g, int b)
{
    ov__current_fg = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

static inline void ov_buf_bg(int r, int g, int b)
{
    ov__current_bg = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

static inline void ov_buf_fg_256(int code)
{
    ov__current_fg = OV_COLOR_256 | (code & 0xFF);
}

static inline void ov_buf_bg_256(int code)
{
    ov__current_bg = OV_COLOR_256 | (code & 0xFF);
}

static inline void ov_buf_ul_color(int r, int g, int b)
{
    ov__current_ul = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

static inline void ov_buf_ul_color_256(int code)
{
    ov__current_ul = OV_COLOR_256 | (code & 0xFF);
}

static inline void ov_buf_pos(int row, int col)
{
    ov__cursor_row = row;
    ov__cursor_col = col;
}

static inline void ov_buf_reset_attr(void)
{
    ov__current_fg   = OV_COLOR_NONE;
    ov__current_bg   = OV_COLOR_NONE;
    ov__current_ul   = OV_COLOR_NONE;
    ov__current_attr = 0;
}

static inline void ov_buf_bold(void)
{
    ov__current_attr |= OV_ATTR_BOLD;
}
static inline void ov_buf_dim(void)
{
    ov__current_attr |= OV_ATTR_DIM;
}
static inline void ov_buf_italic(void)
{
    ov__current_attr |= OV_ATTR_ITALIC;
}
static inline void ov_buf_underline(void)
{
    ov__current_attr |= OV_ATTR_UNDERLINE;
}
static inline void ov_buf_reverse(void)
{
    ov__current_attr |= OV_ATTR_REVERSE;
}
static inline void ov_buf_blink(void)
{
    ov__current_attr |= OV_ATTR_BLINK;
}

static inline void ov_buf_cls(void)
{
    ov_buf_force_clear();
}

static inline void ov_buf_hline(char ch, int len)
{
    for (int i = 0; i < len; i++)
    {
        ov_buf_append_char(&ch, 1);
    }
}

static inline void ov_buf_hline_utf8(const char *s, int len)
{
    int slen = (int) strlen(s);
    for (int i = 0; i < len; i++)
    {
        ov_buf_append_char(s, slen);
    }
}

/* =========================================================
 * Color level detection (TrueColor / 256 / 16)
 * ========================================================= */

extern int ov__color_level;

static inline void ov_detect_color_level(void)
{
    if (ov__color_level > 0)
    {
        return;
    }
    const char *colorterm = getenv("COLORTERM");
    const char *term      = getenv("TERM");
    if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
    {
        ov__color_level = 3; /* TrueColor */
    }
    else if (term && strstr(term, "256color"))
    {
        ov__color_level = 2; /* 256-color */
    }
    else
    {
        ov__color_level = 1; /* 16-color */
    }
}

/* =========================================================
 * Keyboard input (non-blocking)
 * ========================================================= */

static inline int ov_get_key(void)
{
    static unsigned char buf[256];
    static int           buf_len = 0;
    ssize_t              n;

    n = read(STDIN_FILENO, buf + buf_len, sizeof(buf) - (size_t) buf_len);
    if (n > 0)
    {
        buf_len += (int) n;
    }
    if (buf_len == 0)
    {
        if (n == 0)
        {
            return OV_KEY_EOF;
        }
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            return OV_KEY_EOF;
        }
        return OV_KEY_NONE;
    }

    /* single ASCII byte, not ESC */
    if (buf[0] != 0x1b)
    {
        int key = (int) buf[0];
        memmove(buf, buf + 1, (size_t) (buf_len - 1));
        buf_len--;
        return key;
    }

    /* Check if trailing bytes follow ESC for an escape sequence */
    if (buf_len == 1)
    {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
        if (poll(&pfd, 1, 50) > 0 && (pfd.revents & POLLIN))
        {
            n = read(STDIN_FILENO, buf + buf_len, sizeof(buf) - (size_t) buf_len);
            if (n > 0)
            {
                buf_len += (int) n;
            }
        }
    }

    /* Solitary ESC with no subsequent bytes */
    if (buf_len == 1)
    {
        buf_len = 0;
        return OV_KEY_ESC;
    }

    /* ESC with '[' or 'O' waiting for a 3rd byte */
    if (buf_len == 2 && (buf[1] == '[' || buf[1] == 'O'))
    {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
        if (poll(&pfd, 1, 50) > 0 && (pfd.revents & POLLIN))
        {
            n = read(STDIN_FILENO, buf + buf_len, sizeof(buf) - (size_t) buf_len);
            if (n > 0)
            {
                buf_len += (int) n;
            }
        }
    }

    /* Escape sequence */
    if (buf_len >= 2)
    {
        if (buf[1] == '[')
        {
            if (buf_len >= 3)
            {
                int consumed = 0;
                int key      = 0;
                switch (buf[2])
                {
                case 'A':
                    key      = OV_KEY_UP;
                    consumed = 3;
                    break;
                case 'B':
                    key      = OV_KEY_DOWN;
                    consumed = 3;
                    break;
                case 'C':
                    key      = OV_KEY_RIGHT;
                    consumed = 3;
                    break;
                case 'D':
                    key      = OV_KEY_LEFT;
                    consumed = 3;
                    break;
                case 'H':
                    key      = OV_KEY_HOME;
                    consumed = 3;
                    break;
                case 'F':
                    key      = OV_KEY_END;
                    consumed = 3;
                    break;
                case 'Z':
                    key      = OV_KEY_BTAB;
                    consumed = 3;
                    break;
                default:
                    break;
                }
                if (key)
                {
                    memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                    buf_len -= consumed;
                    return key;
                }

                /* ESC [ <digits> ~ */
                int tilde_idx = -1;
                for (int i = 2; i < buf_len && i < 10; i++)
                {
                    if (buf[i] == '~')
                    {
                        tilde_idx = i;
                        break;
                    }
                    if (buf[i] >= 0x40 && buf[i] <= 0x7E)
                    {
                        break;
                    }
                }

                if (tilde_idx != -1)
                {
                    int code = atoi((char *) buf + 2);
                    consumed = tilde_idx + 1;
                    switch (code)
                    {
                    case 1:
                        key = OV_KEY_HOME;
                        break;
                    case 3:
                        key = OV_KEY_DEL;
                        break;
                    case 4:
                        key = OV_KEY_END;
                        break;
                    case 5:
                        key = OV_KEY_PGUP;
                        break;
                    case 6:
                        key = OV_KEY_PGDN;
                        break;
                    case 15:
                        key = OV_KEY_F5;
                        break;
                    case 17:
                        key = OV_KEY_F6;
                        break;
                    case 18:
                        key = OV_KEY_F7;
                        break;
                    case 19:
                        key = OV_KEY_F8;
                        break;
                    default:
                        break;
                    }
                    if (key)
                    {
                        memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* CTRL+Arrow: ESC [ 1 ; 5 C/D
                 * SHIFT+Arrow: ESC [ 1 ; 2 A/B/C/D */
                if (buf_len >= 6 && buf[2] == '1' && buf[3] == ';')
                {
                    if (buf[4] == '5')
                    {
                        if (buf[5] == 'D')
                        {
                            key      = OV_KEY_CTRL_LEFT;
                            consumed = 6;
                        }
                        else if (buf[5] == 'C')
                        {
                            key      = OV_KEY_CTRL_RIGHT;
                            consumed = 6;
                        }
                    }
                    else if (buf[4] == '2')
                    {
                        if (buf[5] == 'D')
                        {
                            key      = OV_KEY_SHIFT_LEFT;
                            consumed = 6;
                        }
                        else if (buf[5] == 'C')
                        {
                            key      = OV_KEY_SHIFT_RIGHT;
                            consumed = 6;
                        }
                        else if (buf[5] == 'A')
                        {
                            key      = OV_KEY_SHIFT_UP;
                            consumed = 6;
                        }
                        else if (buf[5] == 'B')
                        {
                            key      = OV_KEY_SHIFT_DOWN;
                            consumed = 6;
                        }
                    }
                    if (key)
                    {
                        memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* SGR mouse: ESC [ < btn;col;row M/m */
                if (buf[2] == '<')
                {
                    int end_idx = -1;
                    for (int i = 3; i < buf_len && i < 32; i++)
                    {
                        if (buf[i] == 'M' || buf[i] == 'm')
                        {
                            end_idx = i;
                            break;
                        }
                    }
                    if (end_idx > 0)
                    {
                        int  mb = 0, mc = 0, mr = 0;
                        char tmp[64];
                        int  tlen = end_idx - 3;
                        if (tlen > 0 && tlen < (int) sizeof(tmp))
                        {
                            memcpy(tmp, buf + 3, (size_t) tlen);
                            tmp[tlen] = '\0';
                            sscanf(tmp, "%d;%d;%d", &mb, &mc, &mr);
                        }
                        int press = (buf[end_idx] == 'M');
                        consumed  = end_idx + 1;
                        memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                        buf_len -= consumed;

                        ov_mouse_btn = mb;
                        ov_mouse_col = mc;
                        ov_mouse_row = mr;

                        if (mb == 64)
                        {
                            return OV_KEY_MOUSE_UP;
                        }
                        if (mb == 65)
                        {
                            return OV_KEY_MOUSE_DOWN;
                        }
                        /* Ctrl+scroll (#12) */
                        if (mb == 80)
                        {
                            return OV_KEY_CTRL_SCROLL_UP;
                        }
                        if (mb == 81)
                        {
                            return OV_KEY_CTRL_SCROLL_DOWN;
                        }

                        /* Mouse move (passive) */
                        if (mb == 35 && press)
                        {
                            ov_hover_col = mc;
                            ov_hover_row = mr;
                            return OV_KEY_MOUSE_MOVE;
                        }

                        if ((mb & 32) && press)
                        {
                            return OV_KEY_MOUSE_DRAG;
                        }
                        if (mb == 0 && press)
                        {
                            return OV_KEY_MOUSE_CLICK;
                        }
                        if (!press)
                        {
                            return OV_KEY_MOUSE_RELEASE;
                        }
                        return OV_KEY_NONE;
                    }
                    return OV_KEY_NONE;
                }

                for (int i = 2; i < buf_len; i++)
                {
                    if (buf[i] >= 0x40 && buf[i] <= 0x7E)
                    {
                        memmove(buf, buf + i + 1, (size_t) (buf_len - (i + 1)));
                        buf_len -= (i + 1);
                        return OV_KEY_NONE;
                    }
                }
            }
            return OV_KEY_NONE;
        }

        /* SS3: ESC O ... (application cursor keys and xterm F1-F4) */
        if (buf[1] == 'O')
        {
            if (buf_len >= 3)
            {
                int key      = 0;
                int consumed = 3;
                switch (buf[2])
                {
                case 'A':
                    key = OV_KEY_UP;
                    break;
                case 'B':
                    key = OV_KEY_DOWN;
                    break;
                case 'C':
                    key = OV_KEY_RIGHT;
                    break;
                case 'D':
                    key = OV_KEY_LEFT;
                    break;
                case 'H':
                    key = OV_KEY_HOME;
                    break;
                case 'F':
                    key = OV_KEY_END;
                    break;
                case 'P':
                    key = OV_KEY_F1;
                    break;
                case 'Q':
                    key = OV_KEY_F2;
                    break;
                case 'R':
                    key = OV_KEY_F3;
                    break;
                case 'S':
                    key = OV_KEY_F4;
                    break;
                default:
                    break;
                }
                memmove(buf, buf + consumed, (size_t) (buf_len - consumed));
                buf_len -= consumed;
                return key ? key : OV_KEY_NONE;
            }
            return OV_KEY_NONE;
        }

        /* If this is an incomplete CSI or SS3 sequence, do NOT split into ESC + char */
        if (buf_len >= 2 && (buf[1] == '[' || buf[1] == 'O'))
        {
            return OV_KEY_NONE;
        }

        memmove(buf, buf + 1, (size_t) (buf_len - 1));
        buf_len--;
        return OV_KEY_ESC;
    }
    return OV_KEY_NONE;
}

#endif /* OVERVIEW_ANSI_H */
