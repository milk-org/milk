#ifndef OVERVIEW_ANSI_H
#define OVERVIEW_ANSI_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <poll.h>

/* =========================================================
 * Key code constants
 * ========================================================= */

#define OV_KEY_NONE        0
#define OV_KEY_UP        256
#define OV_KEY_DOWN      257
#define OV_KEY_LEFT      258
#define OV_KEY_RIGHT     259
#define OV_KEY_PGUP      260
#define OV_KEY_PGDN      261
#define OV_KEY_HOME      262
#define OV_KEY_END       263
#define OV_KEY_DEL       264
#define OV_KEY_F1        265
#define OV_KEY_F2        266
#define OV_KEY_F3        267
#define OV_KEY_F4        268
#define OV_KEY_F5        269
#define OV_KEY_F6        270
#define OV_KEY_F7        271
#define OV_KEY_F8        272
#define OV_KEY_TAB         9
#define OV_KEY_ENTER      10
#define OV_KEY_ESC        27
#define OV_KEY_CTRL_LEFT  277
#define OV_KEY_CTRL_RIGHT 278
#define OV_KEY_SHIFT_LEFT 279
#define OV_KEY_SHIFT_RIGHT 280
#define OV_KEY_SHIFT_UP 284
#define OV_KEY_SHIFT_DOWN 285
#define OV_KEY_BTAB 286

#define OV_KEY_MOUSE_CLICK  281
#define OV_KEY_MOUSE_UP     282
#define OV_KEY_MOUSE_DOWN   283

extern int ov_mouse_row;
extern int ov_mouse_col;
extern int ov_mouse_btn;

#ifndef ctrl
#define ctrl(x) ((x) & 0x1f)
#endif

/* =========================================================
 * Terminal state
 * ========================================================= */

extern struct termios ov__orig_termios;
extern int            ov__raw_active;

static inline void ov_raw_mode_enter(void)
{
    struct termios raw;
    if (ov__raw_active) return;
    if (tcgetattr(STDIN_FILENO, &ov__orig_termios) == -1) return;
    raw = ov__orig_termios;
    raw.c_iflag &= ~(unsigned int)(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
    raw.c_oflag &= ~(unsigned int)(OPOST);
    raw.c_cflag |=  (unsigned int)(CS8);
    raw.c_lflag &= ~(unsigned int)(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    const char seq[] = "\033[?1049h\033[?25l\033[?7l\033[?1000h\033[?1006h";
    if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0) {}
    ov__raw_active = 1;
}

static inline void ov_raw_mode_exit(void)
{
    if (!ov__raw_active) return;
    const char seq[] = "\033[?1006l\033[?1000l\033[?25h\033[?7h\033[0m\033[?1049l";
    if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0) {}
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &ov__orig_termios);
    ov__raw_active = 0;
}

static inline void ov_get_terminal_size(int *rows, int *cols)
{
    struct winsize ws;
    *rows = 24;
    *cols = 80;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        if (ws.ws_row > 0) *rows = (int) ws.ws_row;
        if (ws.ws_col > 0) *cols = (int) ws.ws_col;
    }
}

/* =========================================================
 * Buffered screen writer (Delta Rendering Shadow Buffer)
 * ========================================================= */

#define OV_SCREENBUF_SIZE (2 * 1024 * 1024)

#define OV_MAX_ROWS 256
#define OV_MAX_COLS 512

#define OV_COLOR_NONE  0xFFFFFFFF // Unset color

#define OV_COLOR_256   0x01000000 // Flag for 256-color
#define OV_COLOR_TRUE  0x02000000 // Flag for TrueColor

#define OV_ATTR_BOLD      (1 << 0)
#define OV_ATTR_DIM       (1 << 1)
#define OV_ATTR_ITALIC    (1 << 2)
#define OV_ATTR_UNDERLINE (1 << 3)
#define OV_ATTR_REVERSE   (1 << 4)
#define OV_ATTR_BLINK     (1 << 5)

typedef struct {
    char     ch[5];    // UTF-8 char up to 4 bytes + null terminator
    uint32_t fg;       // Color code + flag
    uint32_t bg;       // Color code + flag
    uint32_t ul;       // Underline color
    uint8_t  attr;     // bitmask for BOLD, DIM, REVERSE, etc.
} OV_CELL;

extern char ov__screenbuf[OV_SCREENBUF_SIZE];
extern int  ov__screenbuf_len;

extern OV_CELL ov__shadow[OV_MAX_ROWS][OV_MAX_COLS];
extern OV_CELL ov__front[OV_MAX_ROWS][OV_MAX_COLS];

extern int ov__cursor_row; // 1-based
extern int ov__cursor_col; // 1-based
extern uint32_t ov__current_fg;
extern uint32_t ov__current_bg;
extern uint32_t ov__current_ul;
extern uint8_t  ov__current_attr;

static inline void ov_buf_force_clear(void)
{
    memset(ov__front, 0, sizeof(ov__front));
}

static inline void ov_buf_reset(void)
{
    ov__screenbuf_len = 0;
    ov__cursor_row = 1;
    ov__cursor_col = 1;
    ov__current_fg = OV_COLOR_NONE;
    ov__current_bg = OV_COLOR_NONE;
    ov__current_ul = OV_COLOR_NONE;
    ov__current_attr = 0;

    for (int r = 0; r < OV_MAX_ROWS; r++) {
        for (int c = 0; c < OV_MAX_COLS; c++) {
            ov__shadow[r][c].ch[0] = ' ';
            ov__shadow[r][c].ch[1] = '\0';
            ov__shadow[r][c].fg = OV_COLOR_NONE;
            ov__shadow[r][c].bg = OV_COLOR_NONE;
            ov__shadow[r][c].ul = OV_COLOR_NONE;
            ov__shadow[r][c].attr = 0;
        }
    }
}

static inline void ov_buf_append(const char *data, int len)
{
    if (ov__screenbuf_len + len < OV_SCREENBUF_SIZE) {
        memcpy(ov__screenbuf + ov__screenbuf_len, data, (size_t) len);
        ov__screenbuf_len += len;
    }
}

static inline void ov_buf_flush_internal(void)
{
    if (ov__screenbuf_len > 0) {
        int written = 0;
        while (written < ov__screenbuf_len) {
            ssize_t ret = write(STDOUT_FILENO, ov__screenbuf + written, (size_t) (ov__screenbuf_len - written));
            if (ret < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    struct pollfd pfd; pfd.fd = STDOUT_FILENO; pfd.events = POLLOUT;
                    poll(&pfd, 1, 100);
                    continue;
                }
                break;
            }
            if (ret == 0) break;
            written += ret;
        }
        ov__screenbuf_len = 0;
    }
}

static inline void ov_buf_flush_delta(int term_rows, int term_cols)
{
    int emit_cursor_r = -1;
    int emit_cursor_c = -1;
    uint32_t emit_fg = OV_COLOR_NONE;
    uint32_t emit_bg = OV_COLOR_NONE;
    uint32_t emit_ul = OV_COLOR_NONE;
    uint8_t emit_attr = 0;
    char tmp[128];

    if (term_rows > OV_MAX_ROWS) term_rows = OV_MAX_ROWS;
    if (term_cols > OV_MAX_COLS) term_cols = OV_MAX_COLS;

    // Start synchronized output
    ov_buf_append("\033[?2026h", 8);

    for (int r = 0; r < term_rows; r++) {
        for (int c = 0; c < term_cols; c++) {
            OV_CELL *sc = &ov__shadow[r][c];
            OV_CELL *fc = &ov__front[r][c];

            if (sc->ch[0] == '\0') {
                sc->ch[0] = ' '; sc->ch[1] = '\0'; // ensure valid char
            }

            if (memcmp(sc, fc, sizeof(OV_CELL)) != 0) {
                // Pos
                if (emit_cursor_r != r + 1 || emit_cursor_c != c + 1) {
                    int n = snprintf(tmp, sizeof(tmp), "\033[%d;%dH", r + 1, c + 1);
                    ov_buf_append(tmp, n);
                    emit_cursor_r = r + 1;
                    emit_cursor_c = c + 1;
                }

                // Attr reset if missing
                if ((emit_attr & ~sc->attr) != 0 || 
                    (sc->fg != emit_fg && emit_fg != OV_COLOR_NONE && sc->fg == OV_COLOR_NONE) ||
                    (sc->bg != emit_bg && emit_bg != OV_COLOR_NONE && sc->bg == OV_COLOR_NONE) ||
                    (sc->ul != emit_ul && emit_ul != OV_COLOR_NONE && sc->ul == OV_COLOR_NONE)) {
                    ov_buf_append("\033[0m", 4);
                    emit_attr = 0;
                    emit_fg = OV_COLOR_NONE;
                    emit_bg = OV_COLOR_NONE;
                    emit_ul = OV_COLOR_NONE;
                }

                // Add attrs
                if (sc->attr != emit_attr) {
                    if ((sc->attr & OV_ATTR_BOLD)      && !(emit_attr & OV_ATTR_BOLD))      ov_buf_append("\033[1m", 4);
                    if ((sc->attr & OV_ATTR_DIM)       && !(emit_attr & OV_ATTR_DIM))       ov_buf_append("\033[2m", 4);
                    if ((sc->attr & OV_ATTR_ITALIC)    && !(emit_attr & OV_ATTR_ITALIC))    ov_buf_append("\033[3m", 4);
                    if ((sc->attr & OV_ATTR_UNDERLINE) && !(emit_attr & OV_ATTR_UNDERLINE)) ov_buf_append("\033[4m", 4);
                    if ((sc->attr & OV_ATTR_REVERSE)   && !(emit_attr & OV_ATTR_REVERSE))   ov_buf_append("\033[7m", 4);
                    if ((sc->attr & OV_ATTR_BLINK)     && !(emit_attr & OV_ATTR_BLINK))     ov_buf_append("\033[5m", 4);
                    emit_attr = sc->attr;
                }

                // Colors
                if (sc->fg != emit_fg) {
                    if (sc->fg != OV_COLOR_NONE) {
                        if (sc->fg & OV_COLOR_TRUE) {
                            int n = snprintf(tmp, sizeof(tmp), "\033[38;2;%u;%u;%um", (sc->fg >> 16) & 0xFF, (sc->fg >> 8) & 0xFF, sc->fg & 0xFF);
                            ov_buf_append(tmp, n);
                        } else if (sc->fg & OV_COLOR_256) {
                            int n = snprintf(tmp, sizeof(tmp), "\033[38;5;%um", sc->fg & 0xFF);
                            ov_buf_append(tmp, n);
                        }
                    }
                    emit_fg = sc->fg;
                }
                if (sc->bg != emit_bg) {
                    if (sc->bg != OV_COLOR_NONE) {
                        if (sc->bg & OV_COLOR_TRUE) {
                            int n = snprintf(tmp, sizeof(tmp), "\033[48;2;%u;%u;%um", (sc->bg >> 16) & 0xFF, (sc->bg >> 8) & 0xFF, sc->bg & 0xFF);
                            ov_buf_append(tmp, n);
                        } else if (sc->bg & OV_COLOR_256) {
                            int n = snprintf(tmp, sizeof(tmp), "\033[48;5;%um", sc->bg & 0xFF);
                            ov_buf_append(tmp, n);
                        }
                    }
                    emit_bg = sc->bg;
                }
                if (sc->ul != emit_ul) {
                    if (sc->ul != OV_COLOR_NONE) {
                        if (sc->ul & OV_COLOR_TRUE) {
                            int n = snprintf(tmp, sizeof(tmp), "\033[58;2;%u;%u;%um", (sc->ul >> 16) & 0xFF, (sc->ul >> 8) & 0xFF, sc->ul & 0xFF);
                            ov_buf_append(tmp, n);
                        } else if (sc->ul & OV_COLOR_256) {
                            int n = snprintf(tmp, sizeof(tmp), "\033[58;5;%um", sc->ul & 0xFF);
                            ov_buf_append(tmp, n);
                        }
                    }
                    emit_ul = sc->ul;
                }

                // Char
                size_t chlen = strlen(sc->ch);
                ov_buf_append(sc->ch, chlen);
                emit_cursor_c++;

                *fc = *sc;
            }
        }
    }

    // Reset terminal state if we left it dirty so the next frame starts clean
    if (emit_attr != 0 || emit_fg != OV_COLOR_NONE || emit_bg != OV_COLOR_NONE || emit_ul != OV_COLOR_NONE) {
        ov_buf_append("\033[0m", 4);
    }

    // End synchronized output
    ov_buf_append("\033[?2026l", 8);

    ov_buf_flush_internal();
}

static inline void ov_buf_append_char(const char *utf8_seq, int bytes) {
    if (ov__cursor_row >= 1 && ov__cursor_row <= OV_MAX_ROWS &&
        ov__cursor_col >= 1 && ov__cursor_col <= OV_MAX_COLS) {
        
        OV_CELL *cell = &ov__shadow[ov__cursor_row - 1][ov__cursor_col - 1];
        memcpy(cell->ch, utf8_seq, bytes);
        cell->ch[bytes] = '\0';
        cell->fg = ov__current_fg;
        cell->bg = ov__current_bg;
        cell->ul = ov__current_ul;
        cell->attr = ov__current_attr;
    }
    ov__cursor_col++;
}

static inline int utf8_char_length(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static inline void ov_buf_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static inline void ov_buf_printf(const char *fmt, ...)
{
    char tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    if (n > 0) {
        if (n >= (int)sizeof(tmp)) n = (int)sizeof(tmp) - 1;
        int i = 0;
        while (i < n) {
            int char_len = utf8_char_length((unsigned char)tmp[i]);
            if (i + char_len > n) char_len = n - i;
            ov_buf_append_char(&tmp[i], char_len);
            i += char_len;
        }
    }
}

/* =========================================================
 * Buffered color / attribute helpers
 * ========================================================= */

static inline void ov_buf_fg(int r, int g, int b) {
    ov__current_fg = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

static inline void ov_buf_bg(int r, int g, int b) {
    ov__current_bg = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

static inline void ov_buf_fg_256(int code) {
    ov__current_fg = OV_COLOR_256 | (code & 0xFF);
}

static inline void ov_buf_bg_256(int code) {
    ov__current_bg = OV_COLOR_256 | (code & 0xFF);
}

static inline void ov_buf_ul_color(int r, int g, int b) {
    ov__current_ul = OV_COLOR_TRUE | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}

static inline void ov_buf_ul_color_256(int code) {
    ov__current_ul = OV_COLOR_256 | (code & 0xFF);
}

static inline void ov_buf_pos(int row, int col) {
    ov__cursor_row = row;
    ov__cursor_col = col;
}

static inline void ov_buf_reset_attr(void) {
    ov__current_fg = OV_COLOR_NONE;
    ov__current_bg = OV_COLOR_NONE;
    ov__current_ul = OV_COLOR_NONE;
    ov__current_attr = 0;
}

static inline void ov_buf_bold(void)      { ov__current_attr |= OV_ATTR_BOLD; }
static inline void ov_buf_dim(void)       { ov__current_attr |= OV_ATTR_DIM; }
static inline void ov_buf_italic(void)    { ov__current_attr |= OV_ATTR_ITALIC; }
static inline void ov_buf_underline(void) { ov__current_attr |= OV_ATTR_UNDERLINE; }
static inline void ov_buf_reverse(void)   { ov__current_attr |= OV_ATTR_REVERSE; }
static inline void ov_buf_blink(void)     { ov__current_attr |= OV_ATTR_BLINK; }

static inline void ov_buf_cls(void) {
    ov_buf_force_clear();
}

static inline void ov_buf_hline(char ch, int len) {
    for (int i = 0; i < len; i++) {
        ov_buf_append_char(&ch, 1);
    }
}

static inline void ov_buf_hline_utf8(const char *s, int len) {
    int slen = (int) strlen(s);
    for (int i = 0; i < len; i++) {
        ov_buf_append_char(s, slen);
    }
}

/* =========================================================
 * Color level detection (TrueColor / 256 / 16)
 * ========================================================= */

extern int ov__color_level;

static inline void ov_detect_color_level(void)
{
    if (ov__color_level > 0) return;
    const char *colorterm = getenv("COLORTERM");
    const char *term      = getenv("TERM");
    if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit"))) {
        ov__color_level = 3; /* TrueColor */
    } else if (term && strstr(term, "256color")) {
        ov__color_level = 2; /* 256-color */
    } else {
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
    if (n > 0) buf_len += (int) n;
    if (buf_len == 0) return OV_KEY_NONE;

    /* single ASCII byte, not ESC */
    if (buf[0] != 0x1b) {
        int key = (int) buf[0];
        memmove(buf, buf + 1, (size_t)(buf_len - 1));
        buf_len--;
        return key;
    }

    /* Escape sequence */
    if (buf_len >= 2) {
        if (buf[1] == '[') {
            if (buf_len >= 3) {
                int consumed = 0;
                int key      = 0;
                switch (buf[2]) {
                case 'A': key = OV_KEY_UP; consumed = 3; break;
                case 'B': key = OV_KEY_DOWN; consumed = 3; break;
                case 'C': key = OV_KEY_RIGHT; consumed = 3; break;
                case 'D': key = OV_KEY_LEFT; consumed = 3; break;
                case 'H': key = OV_KEY_HOME; consumed = 3; break;
                case 'F': key = OV_KEY_END; consumed = 3; break;
                case 'Z': key = OV_KEY_BTAB; consumed = 3; break;
                default: break;
                }
                if (key) {
                    memmove(buf, buf + consumed, (size_t)(buf_len - consumed));
                    buf_len -= consumed;
                    return key;
                }

                /* ESC [ <digits> ~ */
                int tilde_idx = -1;
                for (int i = 2; i < buf_len && i < 10; i++) {
                    if (buf[i] == '~') { tilde_idx = i; break; }
                    if (buf[i] >= 0x40 && buf[i] <= 0x7E) break;
                }

                if (tilde_idx != -1) {
                    int code = atoi((char *) buf + 2);
                    consumed = tilde_idx + 1;
                    switch (code) {
                    case 1: key = OV_KEY_HOME; break;
                    case 3: key = OV_KEY_DEL; break;
                    case 4: key = OV_KEY_END; break;
                    case 5: key = OV_KEY_PGUP; break;
                    case 6: key = OV_KEY_PGDN; break;
                    case 15: key = OV_KEY_F5; break;
                    case 17: key = OV_KEY_F6; break;
                    case 18: key = OV_KEY_F7; break;
                    case 19: key = OV_KEY_F8; break;
                    default: break;
                    }
                    if (key) {
                        memmove(buf, buf + consumed, (size_t)(buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* CTRL+Arrow: ESC [ 1 ; 5 C/D
                 * SHIFT+Arrow: ESC [ 1 ; 2 A/B/C/D */
                if (buf_len >= 6 && buf[2] == '1' && buf[3] == ';') {
                    if (buf[4] == '5') {
                        if (buf[5] == 'D') { key = OV_KEY_CTRL_LEFT; consumed = 6; }
                        else if (buf[5] == 'C') { key = OV_KEY_CTRL_RIGHT; consumed = 6; }
                    }
                    else if (buf[4] == '2') {
                        if (buf[5] == 'D') { key = OV_KEY_SHIFT_LEFT; consumed = 6; }
                        else if (buf[5] == 'C') { key = OV_KEY_SHIFT_RIGHT; consumed = 6; }
                        else if (buf[5] == 'A') { key = OV_KEY_SHIFT_UP; consumed = 6; }
                        else if (buf[5] == 'B') { key = OV_KEY_SHIFT_DOWN; consumed = 6; }
                    }
                    if (key) {
                        memmove(buf, buf + consumed, (size_t)(buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* SGR mouse: ESC [ < btn;col;row M/m */
                if (buf[2] == '<') {
                    int end_idx = -1;
                    for (int i = 3; i < buf_len && i < 32; i++) {
                        if (buf[i] == 'M' || buf[i] == 'm') { end_idx = i; break; }
                    }
                    if (end_idx > 0) {
                        int mb = 0, mc = 0, mr = 0;
                        char tmp[64];
                        int tlen = end_idx - 3;
                        if (tlen > 0 && tlen < (int) sizeof(tmp)) {
                            memcpy(tmp, buf + 3, (size_t) tlen);
                            tmp[tlen] = '\0';
                            sscanf(tmp, "%d;%d;%d", &mb, &mc, &mr);
                        }
                        int press = (buf[end_idx] == 'M');
                        consumed = end_idx + 1;
                        memmove(buf, buf + consumed, (size_t)(buf_len - consumed));
                        buf_len -= consumed;

                        ov_mouse_btn = mb;
                        ov_mouse_col = mc;
                        ov_mouse_row = mr;

                        if (mb == 64) return OV_KEY_MOUSE_UP;
                        if (mb == 65) return OV_KEY_MOUSE_DOWN;
                        if (mb == 0 && press) return OV_KEY_MOUSE_CLICK;
                        return OV_KEY_NONE;
                    }
                    return OV_KEY_NONE;
                }

                for (int i = 2; i < buf_len; i++) {
                    if (buf[i] >= 0x40 && buf[i] <= 0x7E) {
                        memmove(buf, buf + i + 1, (size_t)(buf_len - (i + 1)));
                        buf_len -= (i + 1);
                        return OV_KEY_NONE;
                    }
                }
            }
            return OV_KEY_NONE;
        }

        /* SS3: ESC O ... (xterm F1-F4) */
        if (buf[1] == 'O') {
            if (buf_len >= 3) {
                int key      = 0;
                int consumed = 3;
                switch (buf[2]) {
                case 'P': key = OV_KEY_F1; break;
                case 'Q': key = OV_KEY_F2; break;
                case 'R': key = OV_KEY_F3; break;
                case 'S': key = OV_KEY_F4; break;
                default: break;
                }
                memmove(buf, buf + consumed, (size_t)(buf_len - consumed));
                buf_len -= consumed;
                return key ? key : OV_KEY_NONE;
            }
            return OV_KEY_NONE;
        }

        memmove(buf, buf + 1, (size_t)(buf_len - 1));
        buf_len--;
        return OV_KEY_ESC;
    }
    return OV_KEY_NONE;
}

#endif /* OVERVIEW_ANSI_H */
