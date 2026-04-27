/**
 * @file streamCTRL_TUIcompat.h
 * @brief Drop-in replacement for TUItools.h for standalone streamCTRL build
 *
 * Provides static inline implementations of all TUItools functions using
 * raw TrueColor ANSI escape sequences instead of ncurses.
 *
 * Include this INSTEAD of TUItools.h when building milk-streamCTRL standalone.
 * All output is written to a global line-buffer that is flushed atomically
 * to stdout at the end of each frame to avoid tearing.
 */

#ifndef _STREAMCTRL_TUITOOLS_COMPAT_H
#define _STREAMCTRL_TUITOOLS_COMPAT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include "streamCTRL_ansi.h"

/* =========================================================
 * SCREENPRINT constants (legacy compat)
 * ========================================================= */

#define SCREENPRINT_STDIO   0
#define SCREENPRINT_NCURSES 1
#define SCREENPRINT_NONE    2

/* =========================================================
 * Frame buffer
 *
 * All output is accumulated into sc_framebuf to allow atomic
 * screen updates (write entire frame in one syscall).
 * ========================================================= */

#define SC_FRAMEBUF_SIZE (1024 * 256)

extern char  sc_framebuf[SC_FRAMEBUF_SIZE];
extern int   sc_framebuf_pos;
extern int   sc_cursor_row;   /* current row being written, 1-indexed */
extern int   sc_cursor_col;   /* current column being written, 0-indexed */
extern int   sc_term_rows;
extern int   sc_term_cols;

/* Append formatted text to frame buffer */
#define SC_APPEND(fmt, ...) \
    do { \
        int _sc_n = snprintf(sc_framebuf + sc_framebuf_pos, \
                             SC_FRAMEBUF_SIZE - sc_framebuf_pos, \
                             fmt, ##__VA_ARGS__); \
        if(_sc_n > 0) { \
            sc_framebuf_pos += _sc_n; \
            if(sc_framebuf_pos >= SC_FRAMEBUF_SIZE) { \
                sc_framebuf_pos = SC_FRAMEBUF_SIZE - 1; \
            } \
        } \
    } while(0)

/** sc_frame_flush - write the accumulated frame buffer to stdout. */
static inline void sc_frame_flush(void)
{
    if(sc_framebuf_pos > 0)
    {
        if(write(STDOUT_FILENO, sc_framebuf, (size_t) sc_framebuf_pos) < 0) {}
        sc_framebuf_pos = 0;
    }
}

/** sc_frame_clear - discard current frame buffer without writing. */
static inline void sc_frame_clear(void)
{
    sc_framebuf_pos = 0;
}

/* =========================================================
 * TUI_newline — advance to the next terminal row
 * ========================================================= */

static inline void TUI_newline(void)
{
    SC_APPEND("\033[K\r\n");
    sc_cursor_row++;
    sc_cursor_col = 0;
}

/* =========================================================
 * TUI_printfw — print formatted text to the frame buffer
 * ========================================================= */

__attribute__((format(printf, 1, 2)))
static inline void TUI_printfw(const char *fmt, ...)
{
    char tmpbuf[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmpbuf, sizeof(tmpbuf), fmt, ap);
    va_end(ap);

    if(n <= 0)
    {
        return;
    }
    if (n >= (int)sizeof(tmpbuf))
    {
        n = sizeof(tmpbuf) - 1;
    }

    for (int i = 0; i < n; i++)
    {
        if (tmpbuf[i] == '\n')
        {
            TUI_newline();
        }
        else if (sc_cursor_col < sc_term_cols && sc_framebuf_pos < SC_FRAMEBUF_SIZE - 1)
        {
            sc_framebuf[sc_framebuf_pos++] = tmpbuf[i];
            sc_cursor_col++;
        }
    }
}

static inline void TUI_cleartobottom(void)
{
    SC_APPEND("\033[J");
}

/* =========================================================
 * screenprint_* — attribute helpers
 *
 * These emit raw ANSI codes into the frame buffer.
 * ========================================================= */

static inline void screenprint_setbold(void)
{
    SC_APPEND("\033[1m");
}

static inline void screenprint_unsetbold(void)
{
    SC_APPEND("\033[22m");
}

static inline void screenprint_setblink(void)
{
    SC_APPEND("\033[5m");
}

static inline void screenprint_unsetblink(void)
{
    SC_APPEND("\033[25m");
}

static inline void screenprint_setdim(void)
{
    SC_APPEND("\033[2m");
}

static inline void screenprint_unsetdim(void)
{
    SC_APPEND("\033[22m");
}

static inline void screenprint_setreverse(void)
{
    SC_APPEND("\033[7m");
}

static inline void screenprint_unsetreverse(void)
{
    SC_APPEND("\033[27m");
}

static inline void screenprint_setnormal(void)
{
    SC_APPEND("\033[0m");
}

/**
 * screenprint_setcolor - apply legacy color index to frame buffer.
 *
 * Color index mapping (matches TUItools legacy palette):
 *   2  = green  (active PID / update)
 *   3  = yellow (upstream level 0)
 *   4  = red    (error)
 *   5  = magenta (symlink)
 *   7  = cyan   (upstream level > 0)
 *   9  = orange (filter ongoing)
 *  12  = bright-green (upstream active)
 */
static inline void screenprint_setcolor(int idx)
{
    switch(idx)
    {
    case 2:  SC_APPEND("\033[38;2;80;220;80m");    break; /* green        */
    case 3:  SC_APPEND("\033[38;2;220;200;0m");    break; /* yellow       */
    case 4:  SC_APPEND("\033[38;2;240;60;60m");    break; /* red          */
    case 5:  SC_APPEND("\033[38;2;200;80;220m");   break; /* magenta      */
    case 7:  SC_APPEND("\033[38;2;0;200;220m");    break; /* cyan         */
    case 9:  SC_APPEND("\033[38;2;255;140;0m");    break; /* orange       */
    case 12: SC_APPEND("\033[38;2;100;255;100m");  break; /* bright-green */
    default: SC_APPEND("\033[0m"); break;
    }
}

static inline void screenprint_unsetcolor(int idx)
{
    (void) idx;
    SC_APPEND("\033[0m");
}

/* =========================================================
 * TUI_print_header — print a header line separated by dashes
 * ========================================================= */

static inline int TUI_print_header(const char *str, char c)
{
    int col = sc_term_cols > 0 ? sc_term_cols : 80;
    int len = (int) strlen(str);
    int pad = col - len - 2;

    TUI_printfw("%s", str);
    for(int i = 0; i < pad; i++)
    {
        SC_APPEND("%c", c);
    }
    TUI_newline();
    return 0;
}

/* =========================================================
 * TUI_clearscreen / TUI_init_terminal compat stubs
 * ========================================================= */

static inline void TUI_set_screenprintmode(int mode)
{
    (void) mode; /* single mode: ANSI */
}

static inline int TUI_get_screenprintmode(void)
{
    return SCREENPRINT_STDIO;
}

static inline int TUI_init_terminal(
    short unsigned int *wrow,
    short unsigned int *wcol)
{
    int r, c;
    ansi_get_terminal_size(&r, &c);
    sc_term_rows = r;
    sc_term_cols = c;
    if(wrow) *wrow = (short unsigned int) r;
    if(wcol) *wcol = (short unsigned int) c;
    return 0;
}

static inline void TUI_clearscreen(
    short unsigned int *wrow,
    short unsigned int *wcol)
{
    int r, c;
    ansi_get_terminal_size(&r, &c);
    sc_term_rows = r;
    sc_term_cols = c;
    if(wrow) *wrow = (short unsigned int) r;
    if(wcol) *wcol = (short unsigned int) c;
    /* Move to top-left of screen without erasing — we overwrite line by line */
    SC_APPEND("\033[H");
    sc_cursor_row = 1;
    sc_cursor_col = 0;
}

/* ncurses stubs */
static inline int TUI_ncurses_refresh(void)  { return 0; }
static inline int TUI_ncurses_erase(void)    { return 0; }
static inline int TUI_stdio_clear(void)      { return 0; }

static inline int get_singlechar_nonblock(void)
{
    return ansi_get_key();
}

static inline int get_singlechar_block(void)
{
    /* block until a key arrives */
    int ch = ANSI_KEY_NONE;
    while(ch == ANSI_KEY_NONE)
    {
        usleep(10000);
        ch = ansi_get_key();
    }
    return ch;
}

/* =========================================================
 * print_help_entry — keep binary-compatible with TUItools.h
 * ========================================================= */

static inline void print_help_entry(char *key, char *descr)
{
    screenprint_setbold();
    TUI_printfw("    %10s", key);
    screenprint_unsetbold();
    TUI_printfw("   %s", descr);
    TUI_newline();
}

/* TUISCREEN — keep the struct so old code still compiles */
typedef struct
{
    int  index;
    int  keych;
    char name[16];
} TUISCREEN;

#endif /* _STREAMCTRL_TUITOOLS_COMPAT_H */
