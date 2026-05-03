/**
 * @file overview_ansi.h
 * @brief Buffered ANSI terminal primitives for milkCTRL
 *
 * Self-contained raw-terminal helpers with a buffered-write
 * system for flicker-free rendering. All output is
 * accumulated into a screen buffer and flushed in a single
 * write() call per frame.
 *
 * Also provides: TrueColor fg/bg, cursor movement,
 * non-blocking keyboard input, and terminal size query.
 * No ncurses dependency.
 */

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

/* Mouse events — encoded as OV_KEY_MOUSE_xxx.
 * The coordinates are stored in a global that the
 * input handler reads after receiving the key. */
#define OV_KEY_MOUSE_CLICK  280
#define OV_KEY_MOUSE_UP     281  /* scroll wheel up   */
#define OV_KEY_MOUSE_DOWN   282  /* scroll wheel down */

extern int ov_mouse_row;   /* 1-based row of event  */
extern int ov_mouse_col;   /* 1-based col of event  */
extern int ov_mouse_btn;   /* raw button code       */

#ifndef ctrl
#define ctrl(x) ((x) & 0x1f)
#endif

/* =========================================================
 * Terminal state
 * ========================================================= */

extern struct termios ov__orig_termios;
extern int            ov__raw_active;

/**
 * ov_raw_mode_enter - switch stdin to raw mode.
 */
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
    raw.c_iflag &=
        ~(unsigned int)(IXON | ICRNL | BRKINT
                        | INPCK | ISTRIP);
    raw.c_oflag &= ~(unsigned int)(OPOST);
    raw.c_cflag |=  (unsigned int)(CS8);
    raw.c_lflag &=
        ~(unsigned int)(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    /* non-blocking stdin */
    {
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }

    /* hide cursor, disable line wrap, alt screen,
     * enable SGR mouse tracking (btn+motion+scroll) */
    {
        const char seq[] =
            "\033[?1049h\033[?25l\033[?7l"
            "\033[?1000h\033[?1006h";
        if (write(STDOUT_FILENO,
                  seq, sizeof(seq) - 1) < 0) {}
    }
    ov__raw_active = 1;
}

/**
 * ov_raw_mode_exit - restore original terminal settings.
 */
static inline void ov_raw_mode_exit(void)
{
    if (!ov__raw_active)
    {
        return;
    }
    /* disable mouse, show cursor, enable line wrap,
     * leave alt screen */
    {
        const char seq[] =
            "\033[?1006l\033[?1000l"
            "\033[?25h\033[?7h\033[0m\033[?1049l";
        if (write(STDOUT_FILENO,
                  seq, sizeof(seq) - 1) < 0) {}
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH,
              &ov__orig_termios);
    ov__raw_active = 0;
}

/* =========================================================
 * Terminal size
 * ========================================================= */

static inline void ov_get_terminal_size(
    int *rows,
    int *cols)
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
 * Buffered screen writer
 *
 * All TUI output goes through this buffer. At the end of
 * each frame, ov_buf_flush() does a single write() call,
 * eliminating flicker entirely.
 * ========================================================= */

#define OV_SCREENBUF_SIZE (2 * 1024 * 1024)

extern char ov__screenbuf[OV_SCREENBUF_SIZE];
extern int  ov__screenbuf_len;

static inline void ov_buf_reset(void)
{
    ov__screenbuf_len = 0;
}

static inline void ov_buf_flush(void)
{
    if (ov__screenbuf_len > 0)
    {
        int written = 0;
        while (written < ov__screenbuf_len)
        {
            ssize_t ret = write(STDOUT_FILENO,
                                ov__screenbuf + written,
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
                break; /* Should not happen for pipe/pty, but safe */
            }
            written += ret;
        }
        ov__screenbuf_len = 0;
    }
}

static inline void ov_buf_append(
    const char *data,
    int         len)
{
    if (ov__screenbuf_len + len
            < OV_SCREENBUF_SIZE)
    {
        memcpy(ov__screenbuf + ov__screenbuf_len,
               data, (size_t) len);
        ov__screenbuf_len += len;
    }
}

/**
 * ov_buf_printf - buffered printf to screen buffer.
 */
static inline void ov_buf_printf(
    const char *fmt,
    ...) __attribute__((format(printf, 1, 2)));

static inline void ov_buf_printf(
    const char *fmt,
    ...)
{
    char tmp[1024];
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
        ov_buf_append(tmp, n);
    }
}

/* =========================================================
 * Buffered color / attribute helpers
 * ========================================================= */

static inline void ov_buf_fg(int r, int g, int b)
{
    ov_buf_printf("\033[38;2;%d;%d;%dm", r, g, b);
}

static inline void ov_buf_bg(int r, int g, int b)
{
    ov_buf_printf("\033[48;2;%d;%d;%dm", r, g, b);
}

static inline void ov_buf_fg_256(int code)
{
    ov_buf_printf("\033[38;5;%dm", code);
}

static inline void ov_buf_bg_256(int code)
{
    ov_buf_printf("\033[48;5;%dm", code);
}

static inline void ov_buf_pos(int row, int col)
{
    ov_buf_printf("\033[%d;%dH", row, col);
}

static inline void ov_buf_reset_attr(void)
{
    ov_buf_append("\033[0m", 4);
}

static inline void ov_buf_bold(void)
{
    ov_buf_append("\033[1m", 4);
}

static inline void ov_buf_dim(void)
{
    ov_buf_append("\033[2m", 4);
}

static inline void ov_buf_italic(void)
{
    ov_buf_append("\033[3m", 4);
}

static inline void ov_buf_underline(void)
{
    ov_buf_append("\033[4m", 4);
}

static inline void ov_buf_reverse(void)
{
    ov_buf_append("\033[7m", 4);
}

static inline void ov_buf_blink(void)
{
    ov_buf_append("\033[5m", 4);
}

static inline void ov_buf_cls(void)
{
    ov_buf_append("\033[2J\033[H", 7);
}

/**
 * ov_buf_hline - draw a horizontal line of a character.
 * @ch:  the character to repeat (single byte)
 * @len: how many times to repeat
 */
static inline void ov_buf_hline(char ch, int len)
{
    for (int i = 0; i < len; i++)
    {
        ov_buf_append(&ch, 1);
    }
}

/**
 * ov_buf_hline_utf8 - draw a horizontal line of a
 *                     multi-byte UTF-8 character.
 * @s:   UTF-8 string of one character
 * @len: how many times to repeat
 */
static inline void ov_buf_hline_utf8(
    const char *s,
    int len)
{
    int slen = (int) strlen(s);
    for (int i = 0; i < len; i++)
    {
        ov_buf_append(s, slen);
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

    if (colorterm &&
        (strstr(colorterm, "truecolor")
         || strstr(colorterm, "24bit")))
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

/**
 * ov_get_key - read one key event from stdin.
 *
 * Returns ASCII code for printable chars, or an
 * OV_KEY_* constant for special keys.
 * Returns OV_KEY_NONE (0) if no input available.
 */
static inline int ov_get_key(void)
{
    static unsigned char buf[256];
    static int           buf_len = 0;
    ssize_t              n;

    n = read(STDIN_FILENO,
             buf + buf_len,
             sizeof(buf) - (size_t) buf_len);
    if (n > 0)
    {
        buf_len += (int) n;
    }

    if (buf_len == 0)
    {
        return OV_KEY_NONE;
    }

    /* single ASCII byte, not ESC */
    if (buf[0] != 0x1b)
    {
        int key = (int) buf[0];
        memmove(buf, buf + 1, (size_t)(buf_len - 1));
        buf_len--;
        return key;
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
                    key = OV_KEY_UP;
                    consumed = 3;
                    break;
                case 'B':
                    key = OV_KEY_DOWN;
                    consumed = 3;
                    break;
                case 'C':
                    key = OV_KEY_RIGHT;
                    consumed = 3;
                    break;
                case 'D':
                    key = OV_KEY_LEFT;
                    consumed = 3;
                    break;
                case 'H':
                    key = OV_KEY_HOME;
                    consumed = 3;
                    break;
                case 'F':
                    key = OV_KEY_END;
                    consumed = 3;
                    break;
                case 'Z':
                    key = OV_KEY_TAB; /* Shift+Tab */
                    consumed = 3;
                    break;
                default:
                    break;
                }

                if (key)
                {
                    memmove(buf, buf + consumed,
                            (size_t)(buf_len - consumed));
                    buf_len -= consumed;
                    return key;
                }

                /* ESC [ <digits> ~ */
                int tilde_idx = -1;
                for (int i = 2;
                     i < buf_len && i < 10; i++)
                {
                    if (buf[i] == '~')
                    {
                        tilde_idx = i;
                        break;
                    }
                    if (buf[i] >= 0x40
                            && buf[i] <= 0x7E)
                    {
                        break;
                    }
                }

                if (tilde_idx != -1)
                {
                    int code = atoi(
                                   (char *) buf + 2);
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
                        memmove(
                            buf,
                            buf + consumed,
                            (size_t)(buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* CTRL+Arrow: ESC [ 1 ; 5 C/D */
                if (buf_len >= 6
                        && buf[2] == '1'
                        && buf[3] == ';'
                        && buf[4] == '5')
                {
                    if (buf[5] == 'D')
                    {
                        key = OV_KEY_CTRL_LEFT;
                        consumed = 6;
                    }
                    else if (buf[5] == 'C')
                    {
                        key = OV_KEY_CTRL_RIGHT;
                        consumed = 6;
                    }
                    if (key)
                    {
                        memmove(
                            buf,
                            buf + consumed,
                            (size_t)(buf_len - consumed));
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* SGR mouse: ESC [ < btn;col;row M/m */
                if (buf[2] == '<')
                {
                    int end_idx = -1;
                    for (int i = 3;
                         i < buf_len && i < 32; i++)
                    {
                        if (buf[i] == 'M'
                            || buf[i] == 'm')
                        {
                            end_idx = i;
                            break;
                        }
                    }
                    if (end_idx > 0)
                    {
                        int mb = 0, mc = 0, mr = 0;
                        char tmp[64];
                        int tlen = end_idx - 3;
                        if (tlen > 0
                            && tlen < (int) sizeof(tmp))
                        {
                            memcpy(tmp, buf + 3,
                                   (size_t) tlen);
                            tmp[tlen] = '\0';
                            sscanf(tmp, "%d;%d;%d",
                                   &mb, &mc, &mr);
                        }
                        /* press = 'M', release = 'm' */
                        int press =
                            (buf[end_idx] == 'M');
                        consumed = end_idx + 1;
                        memmove(
                            buf,
                            buf + consumed,
                            (size_t)(buf_len
                                     - consumed));
                        buf_len -= consumed;

                        ov_mouse_btn = mb;
                        ov_mouse_col = mc;
                        ov_mouse_row = mr;

                        /* scroll wheel */
                        if (mb == 64)
                        {
                            return OV_KEY_MOUSE_UP;
                        }
                        if (mb == 65)
                        {
                            return OV_KEY_MOUSE_DOWN;
                        }
                        /* left click press */
                        if (mb == 0 && press)
                        {
                            return OV_KEY_MOUSE_CLICK;
                        }
                        return OV_KEY_NONE;
                    }
                    /* incomplete, wait for more */
                    return OV_KEY_NONE;
                }

                /* Consume unrecognized CSI */
                for (int i = 2; i < buf_len; i++)
                {
                    if (buf[i] >= 0x40
                            && buf[i] <= 0x7E)
                    {
                        memmove(
                            buf,
                            buf + i + 1,
                            (size_t)(buf_len
                                     - (i + 1)));
                        buf_len -= (i + 1);
                        return OV_KEY_NONE;
                    }
                }
            }
            return OV_KEY_NONE;
        }

        /* SS3: ESC O ... (xterm F1-F4) */
        if (buf[1] == 'O')
        {
            if (buf_len >= 3)
            {
                int key      = 0;
                int consumed = 3;
                switch (buf[2])
                {
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
                memmove(buf, buf + consumed,
                        (size_t)(buf_len - consumed));
                buf_len -= consumed;
                return key ? key : OV_KEY_NONE;
            }
            return OV_KEY_NONE;
        }

        /* bare ESC */
        memmove(buf, buf + 1,
                (size_t)(buf_len - 1));
        buf_len--;
        return OV_KEY_ESC;
    }

    return OV_KEY_NONE;
}

#endif /* OVERVIEW_ANSI_H */
