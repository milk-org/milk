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
#define OV_KEY_PASTE_BEGIN 292 /* ESC [ 200 ~ (bracketed paste) */
#define OV_KEY_PASTE_END 293   /* ESC [ 201 ~ */
#define OV_KEY_PASTED 0x10000  /* flag on printable bytes inside a bracketed paste */

extern int      ov_mouse_row;
extern int      ov_mouse_col;
extern uint32_t ov__default_bg;
extern uint32_t ov__mono_reverse_bg[2]; /* backgrounds drawn as reverse video in monochrome */
extern int      ov__color_level;        /* 0 mono, 1 16, 2 256, 3 truecolor; -1 not detected */
extern int      ov_mouse_btn;
extern int      ov_hover_row;
extern int      ov_hover_col;

#ifndef ctrl
#    define ctrl(x) ((x) & 0x1f)
#endif

/* =========================================================
 * Terminal state
 * ========================================================= */

extern struct termios        ov__orig_termios;
extern volatile sig_atomic_t ov__raw_active;
extern int                   ov__tty_fd;


/* =========================================================
 * Terminal and input API prototypes
 * ========================================================= */

void ov_raw_mode_enter(void);
void ov_raw_mode_exit(void);
void ov_term_restore_async(void);
int  ov_term_capture_fd(void);
int  ov_term_init_locale(void);
void ov_set_mouse_hover(int enable);
void ov_get_terminal_size(int *rows, int *cols);

void ov_buf_force_clear(void);
void ov_buf_reset_size(int rows, int cols);
void ov_buf_append(const char *data, int len);
void ov_buf_flush_internal(void);
void ov_buf_flush_delta(int term_rows, int term_cols);

int  utf8_char_length(unsigned char c);
int  ov_utf8_decode(const char *s, int len, uint32_t *cp);
int  ov_utf8_next_cluster(const char *s, int max_len, int *bytes_out, int *width_out);
int  ov_str_display_width(const char *s);
int  ov_utf8_cluster_printable(const char *s, int bytes);
void ov_str_sanitize(char *s);
int  ov_utf8_fit(const char *s, int len, int max_cols, int *cols_out);
void ov_buf_append_cluster(const char *utf8_seq, int bytes, int width);
void ov_buf_append_char(const char *utf8_seq, int bytes);
void ov_buf_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

void     ov_buf_hline(char ch, int len);
void     ov_buf_hline_utf8(const char *s, int len);
void     ov_detect_color_level(void);
uint32_t ov_color_downgrade(uint32_t col, int level);
int      ov_get_key(void);
/* =========================================================
 * Buffered screen writer (Delta Rendering Shadow Buffer)
 * ========================================================= */

#define OV_SCREENBUF_SIZE (2 * 1024 * 1024)

#define OV_MAX_ROWS 256
#define OV_MAX_COLS 512

#define OV_COLOR_NONE 0xFFFFFFFF // Unset color

#define OV_COLOR_256 0x01000000  // Flag for 256-color
#define OV_COLOR_TRUE 0x02000000 // Flag for TrueColor
#define OV_COLOR_16 0x04000000   // Flag for 16-colour index (flush-time downgrade only)

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

double ov_sys_get_bandwidth_usage(void);

extern OV_CELL ov__shadow[OV_MAX_ROWS][OV_MAX_COLS];
extern OV_CELL ov__front[OV_MAX_ROWS][OV_MAX_COLS];

extern int      ov__cursor_row; // 1-based
extern int      ov__cursor_col; // 1-based
extern uint32_t ov__current_fg;
extern uint32_t ov__current_bg;
extern uint32_t ov__current_ul;
extern uint8_t  ov__current_attr;


/**
 * ov_buf_fg - Set active 24-bit truecolor foreground RGB color for subsequent draws
 * @r: Red channel (`0..255`)
 * @g: Green channel (`0..255`)
 * @b: Blue channel (`0..255`)
 */
static inline void ov_buf_fg(int r, int g, int b)
{
    ov__current_fg = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

/**
 * ov_buf_bg - Set active 24-bit truecolor background RGB color for subsequent draws
 * @r: Red channel (`0..255`)
 * @g: Green channel (`0..255`)
 * @b: Blue channel (`0..255`)
 */
static inline void ov_buf_bg(int r, int g, int b)
{
    ov__current_bg = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

/**
 * ov_buf_ul_color - Set active 24-bit truecolor underline RGB color for subsequent draws
 * @r: Red channel (`0..255`)
 * @g: Green channel (`0..255`)
 * @b: Blue channel (`0..255`)
 */
static inline void ov_buf_ul_color(int r, int g, int b)
{
    ov__current_ul = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

/**
 * ov_buf_pos - Move virtual cell-buffer cursor to 1-based (`@row`, `@col`)
 * @row: 1-based terminal row
 * @col: 1-based terminal column
 */
static inline void ov_buf_pos(int row, int col)
{
    ov__cursor_row = row;
    ov__cursor_col = col;
}

/**
 * ov_buf_reset_attr - Reset foreground, background, underline colors, and style attributes
 */
static inline void ov_buf_reset_attr(void)
{
    ov__current_fg   = OV_COLOR_NONE;
    ov__current_bg   = OV_COLOR_NONE;
    ov__current_ul   = OV_COLOR_NONE;
    ov__current_attr = 0;
}

/**
 * ov_buf_bold - Enable bold attribute bit (`OV_ATTR_BOLD`) for subsequent draws
 */
static inline void ov_buf_bold(void)
{
    ov__current_attr |= OV_ATTR_BOLD;
}

/**
 * ov_buf_underline - Enable underline attribute bit (`OV_ATTR_UNDERLINE`) for subsequent draws
 */
static inline void ov_buf_underline(void)
{
    ov__current_attr |= OV_ATTR_UNDERLINE;
}

#endif /* OVERVIEW_ANSI_H */
