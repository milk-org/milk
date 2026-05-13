/**
 * @file milkTUI_compat.h
 * @brief Shared ANSI frame-buffer TUI rendering primitives
 *
 * Drop-in replacement for ncurses TUItools.h used by all
 * standalone TUI tools (milk-fpsCTRL, milk-procCTRL,
 * milk-streamCTRL).
 *
 * Prerequisites: include your tool-specific *_ansi.h
 * header BEFORE this file. That header must provide:
 *   - ansi_get_key()
 *   - ansi_detect_color_level()
 *   - ansi_get_terminal_size()
 *   - ansi_raw_mode_enter()
 *   - ansi_raw_mode_exit()
 *   - ansi__color_level  (extern int)
 *   - ANSI_KEY_NONE
 */

#ifndef MILK_TUI_COMPAT_H
#define MILK_TUI_COMPAT_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Compile-time prerequisite check */
#ifndef ANSI_KEY_NONE
#error "Include your *_ansi.h before milkTUI_compat.h"
#endif

/* =========================================================
 * SCREENPRINT constants (legacy compat)
 * ========================================================= */

/* Semantic color names */
#define COLOR_OK              2
#define COLOR_WARNING         3
#define COLOR_ERROR           4
#define COLOR_DIRECTORY       7
#define COLOR_BLACK_ON_WHITE  1
#define COLOR_TRIGGER_BG     13

#define SCREENPRINT_STDIO   0
#define SCREENPRINT_NCURSES 1
#define SCREENPRINT_NONE    2

/* =========================================================
 * Frame buffer
 *
 * All output is accumulated into sc_framebuf to allow
 * atomic screen updates (write entire frame in one
 * syscall).
 * ========================================================= */

#define SC_FRAMEBUF_SIZE (1024 * 256)

extern char sc_framebuf[SC_FRAMEBUF_SIZE];
extern int  sc_framebuf_pos;
extern int  sc_cursor_row;
extern int  sc_cursor_col;
extern int  sc_term_rows;
extern int  sc_term_cols;

/* Append formatted text to frame buffer */
#define SC_APPEND(fmt, ...) \
    do { \
        int _sc_n = snprintf( \
            sc_framebuf + sc_framebuf_pos, \
            SC_FRAMEBUF_SIZE - sc_framebuf_pos, \
            fmt, ##__VA_ARGS__); \
        if(_sc_n > 0) { \
            sc_framebuf_pos += _sc_n; \
            if(sc_framebuf_pos >= SC_FRAMEBUF_SIZE) \
            { \
                sc_framebuf_pos = \
                    SC_FRAMEBUF_SIZE - 1; \
            } \
        } \
    } while(0)

/**
 * sc_frame_flush - write frame buffer to stdout.
 *
 * Appends clear-to-end-of-screen before flushing to
 * remove ghost lines from the previous frame.
 */
static inline void sc_frame_flush(void)
{
    SC_APPEND("\033[J");
    if(sc_framebuf_pos > 0)
    {
        if(write(STDOUT_FILENO,
                 sc_framebuf,
                 (size_t) sc_framebuf_pos) < 0)
        {
        }
        sc_framebuf_pos = 0;
    }
}

/** sc_frame_clear - discard frame buffer. */
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
 * TUI_printfw — print formatted text to frame buffer
 *
 * Handles embedded newlines and enforces column limits.
 * ========================================================= */

__attribute__((format(printf, 1, 2)))
static inline void TUI_printfw(
    const char *fmt, ...)
{
    char    tmpbuf[2048];
    va_list ap;

    va_start(ap, fmt);
    int n = vsnprintf(
        tmpbuf, sizeof(tmpbuf), fmt, ap);
    va_end(ap);

    if(n <= 0)
    {
        return;
    }
    if(n >= (int) sizeof(tmpbuf))
    {
        n = sizeof(tmpbuf) - 1;
    }

    for(int i = 0; i < n; i++)
    {
        if(tmpbuf[i] == '\n')
        {
            TUI_newline();
        }
        else if(sc_cursor_col < sc_term_cols
                && sc_framebuf_pos
                       < SC_FRAMEBUF_SIZE - 1)
        {
            sc_framebuf[sc_framebuf_pos++] =
                tmpbuf[i];
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
 * screenprint_setcolor - apply legacy color index.
 *
 * Color index mapping (TUItools legacy palette):
 *   1  = black on white
 *   2  = green   (active PID / update)
 *   3  = yellow  (upstream level 0)
 *   4  = red     (error)
 *   5  = magenta (symlink)
 *   6  = green   (alias)
 *   7  = cyan    (upstream level > 0)
 *   8  = red     (alias)
 *   9  = orange  (filter ongoing)
 *  10  = black on cyan
 *  12  = bright-green (upstream active)
 *  13  = trig-stream background
 */
static inline void screenprint_setcolor(int idx)
{
    ansi_detect_color_level();

    if(ansi__color_level >= 3)
    {
        switch(idx)
        {
        case 1:
            SC_APPEND("\033[30;47m");
            break;
        case 2:
            SC_APPEND("\033[38;2;80;220;80m");
            break;
        case 3:
            SC_APPEND("\033[38;2;220;200;0m");
            break;
        case 4:
            SC_APPEND("\033[38;2;240;60;60m");
            break;
        case 5:
            SC_APPEND("\033[38;2;200;80;220m");
            break;
        case 6:
            SC_APPEND("\033[38;2;80;220;80m");
            break;
        case 7:
            SC_APPEND("\033[38;2;0;200;220m");
            break;
        case 8:
            SC_APPEND("\033[38;2;240;60;60m");
            break;
        case 9:
            SC_APPEND("\033[38;2;255;140;0m");
            break;
        case 10:
            SC_APPEND("\033[30;46m");
            break;
        case 12:
            SC_APPEND("\033[38;2;100;255;100m");
            break;
        case 13:
            SC_APPEND("\033[48;2;20;50;50m");
            break;
        default:
            SC_APPEND("\033[0m");
            break;
        }
    }
    else if(ansi__color_level == 2)
    {
        switch(idx)
        {
        case 1:
            SC_APPEND("\033[30;47m");
            break;
        case 2:
            SC_APPEND("\033[38;5;114m");
            break;
        case 3:
            SC_APPEND("\033[38;5;220m");
            break;
        case 4:
            SC_APPEND("\033[38;5;203m");
            break;
        case 5:
            SC_APPEND("\033[38;5;176m");
            break;
        case 6:
            SC_APPEND("\033[38;5;114m");
            break;
        case 7:
            SC_APPEND("\033[38;5;44m");
            break;
        case 8:
            SC_APPEND("\033[38;5;203m");
            break;
        case 9:
            SC_APPEND("\033[38;5;208m");
            break;
        case 10:
            SC_APPEND("\033[30;46m");
            break;
        case 12:
            SC_APPEND("\033[38;5;119m");
            break;
        case 13:
            SC_APPEND("\033[48;5;23m");
            break;
        default:
            SC_APPEND("\033[0m");
            break;
        }
    }
    else
    {
        switch(idx)
        {
        case 1:
            SC_APPEND("\033[30;47m");
            break;
        case 2:
            SC_APPEND("\033[32m");
            break;
        case 3:
            SC_APPEND("\033[33m");
            break;
        case 4:
            SC_APPEND("\033[31m");
            break;
        case 5:
            SC_APPEND("\033[35m");
            break;
        case 6:
            SC_APPEND("\033[32m");
            break;
        case 7:
            SC_APPEND("\033[36m");
            break;
        case 8:
            SC_APPEND("\033[31m");
            break;
        case 9:
            SC_APPEND("\033[33m");
            break;
        case 10:
            SC_APPEND("\033[30;46m");
            break;
        case 12:
            SC_APPEND("\033[92m");
            break;
        case 13:
            SC_APPEND("\033[100m");
            break;
        default:
            SC_APPEND("\033[0m");
            break;
        }
    }
}

/* =========================================================
 * SLEEK UI COLORS
 * ========================================================= */

static inline void
screenprint_setbgcolor_highlight(void)
{
    ansi_detect_color_level();
    if(ansi__color_level >= 3)
    {
        SC_APPEND("\033[48;2;70;130;180m");
    }
    else if(ansi__color_level == 2)
    {
        SC_APPEND("\033[48;5;67m");
    }
    else
    {
        SC_APPEND("\033[44m");
    }
}

static inline void
screenprint_set_status_bar(void)
{
    ansi_detect_color_level();
    if(ansi__color_level >= 3)
    {
        SC_APPEND(
            "\033[38;2;255;255;255;"
            "48;2;60;60;80m");
    }
    else if(ansi__color_level == 2)
    {
        SC_APPEND("\033[38;5;255;48;5;60m");
    }
    else
    {
        SC_APPEND("\033[37;44m");
    }
}

static inline void screenprint_color_string(void)
{
    ansi_detect_color_level();
    if(ansi__color_level >= 3)
    {
        SC_APPEND("\033[38;2;0;255;255m");
    }
    else if(ansi__color_level == 2)
    {
        SC_APPEND("\033[38;5;51m");
    }
    else
    {
        SC_APPEND("\033[36m");
    }
}

static inline void screenprint_color_number(void)
{
    ansi_detect_color_level();
    if(ansi__color_level >= 3)
    {
        SC_APPEND("\033[38;2;255;220;50m");
    }
    else if(ansi__color_level == 2)
    {
        SC_APPEND("\033[38;5;220m");
    }
    else
    {
        SC_APPEND("\033[33m");
    }
}

static inline void screenprint_color_flag(void)
{
    ansi_detect_color_level();
    if(ansi__color_level >= 3)
    {
        SC_APPEND("\033[38;2;50;255;100m");
    }
    else if(ansi__color_level == 2)
    {
        SC_APPEND("\033[38;5;46m");
    }
    else
    {
        SC_APPEND("\033[32m");
    }
}

static inline void screenprint_color_dim(void)
{
    ansi_detect_color_level();
    if(ansi__color_level >= 3)
    {
        SC_APPEND("\033[38;2;120;120;120m");
    }
    else if(ansi__color_level == 2)
    {
        SC_APPEND("\033[38;5;243m");
    }
    else
    {
        SC_APPEND("\033[90m");
    }
}

/**
 * Reset fg+bg colors to default, preserving other
 * attrs (bold, dim, reverse, blink).
 * Use screenprint_setnormal() for a full reset.
 */
static inline void screenprint_unsetcolor(
    int idx)
{
    (void) idx;
    SC_APPEND("\033[39;49m");
}

/** Reset only the background color. */
static inline void screenprint_unsetbgcolor(void)
{
    SC_APPEND("\033[49m");
}

/* =========================================================
 * TUI_print_header — header line padded with char
 * ========================================================= */

static inline int TUI_print_header(
    const char *str,
    char        c)
{
    int col =
        sc_term_cols > 0 ? sc_term_cols : 80;
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
 * TUI_clearscreen / TUI_init_terminal compat
 * ========================================================= */

static inline void TUI_set_screenprintmode(
    int mode)
{
    (void) mode;
}

static inline int TUI_get_screenprintmode(void)
{
    return SCREENPRINT_STDIO;
}

static inline int TUI_init_terminal(
    short unsigned int *wrow,
    short unsigned int *wcol)
{
    ansi_raw_mode_enter();
    int r, c;
    ansi_get_terminal_size(&r, &c);
    sc_term_rows = r;
    sc_term_cols = c;
    if(wrow)
    {
        *wrow = (short unsigned int) r;
    }
    if(wcol)
    {
        *wcol = (short unsigned int) c;
    }
    return 0;
}

static inline int TUI_get_terminal_size(
    short unsigned int *wrow,
    short unsigned int *wcol)
{
    return TUI_init_terminal(wrow, wcol);
}

static inline void TUI_exit(void)
{
    ansi_raw_mode_exit();
}

static inline void TUI_clearscreen(
    short unsigned int *wrow,
    short unsigned int *wcol)
{
    int r, c;
    ansi_get_terminal_size(&r, &c);
    sc_term_rows = r;
    sc_term_cols = c;
    if(wrow)
    {
        *wrow = (short unsigned int) r;
    }
    if(wcol)
    {
        *wcol = (short unsigned int) c;
    }
    sc_frame_clear();
    SC_APPEND("\033[H");
    sc_cursor_row = 1;
    sc_cursor_col = 0;
}

/* ncurses stubs */
static inline int TUI_ncurses_refresh(void)
{
    return 0;
}

static inline int TUI_ncurses_erase(void)
{
    return 0;
}

static inline int TUI_stdio_clear(void)
{
    sc_frame_clear();
    SC_APPEND("\033[H");
    sc_cursor_row = 1;
    sc_cursor_col = 0;
    return 0;
}

static inline int get_singlechar_nonblock(void)
{
    return ansi_get_key();
}

static inline int get_singlechar_block(void)
{
    int ch = ANSI_KEY_NONE;
    while(ch == ANSI_KEY_NONE)
    {
        usleep(10000);
        ch = ansi_get_key();
    }
    return ch;
}

/* =========================================================
 * print_help_entry — formatted help key line
 * ========================================================= */

static inline void print_help_entry(
    char *key,
    char *descr)
{
    screenprint_setbold();
    TUI_printfw("    %10s", key);
    screenprint_unsetbold();
    TUI_printfw("   %s", descr);
    TUI_newline();
}

/* TUISCREEN — struct for legacy compat */
typedef struct
{
    int  index;
    int  keych;
    char name[16];
} TUISCREEN;

#endif /* MILK_TUI_COMPAT_H */
