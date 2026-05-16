/**
 * @file streamCTRL_ansi.h
 * @brief TrueColor ANSI terminal primitives for milk-streamCTRL
 *
 * Self-contained raw-terminal helpers: TrueColor fg/bg, cursor movement,
 * non-blocking keyboard input with ANSI escape sequence decoding, and
 * terminal size query.  No ncurses dependency.
 *
 * Usage:
 *   - Call ansi_raw_mode_enter() at startup (stores original termios).
 *   - Call ansi_raw_mode_exit() at cleanup (restores original termios).
 *   - Use ansi_get_key() in the main loop for non-blocking input.
 *   - Use ANSI_KEY_* constants to compare returned key codes.
 */

#ifndef _STREAMCTRL_ANSI_H
#define _STREAMCTRL_ANSI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

/* =========================================================
 * Key code constants
 * ========================================================= */

#define ANSI_KEY_NONE        0
#define ANSI_KEY_UP        256
#define ANSI_KEY_DOWN      257
#define ANSI_KEY_LEFT      258
#define ANSI_KEY_RIGHT     259
#define ANSI_KEY_PGUP      260
#define ANSI_KEY_PGDN      261
#define ANSI_KEY_HOME      262
#define ANSI_KEY_END       263
#define ANSI_KEY_DEL       264
#define ANSI_KEY_F1        265
#define ANSI_KEY_F2        266
#define ANSI_KEY_F3        267
#define ANSI_KEY_F4        268
#define ANSI_KEY_F5        269
#define ANSI_KEY_F6        270
#define ANSI_KEY_F7        271
#define ANSI_KEY_F8        272
#define ANSI_KEY_F9        273
#define ANSI_KEY_F10       274
#define ANSI_KEY_F11       275
#define ANSI_KEY_F12       276
#define ANSI_KEY_CTRL_LEFT  277
#define ANSI_KEY_CTRL_RIGHT 278
#define ANSI_KEY_MOUSE      279
#define ANSI_KEY_SCROLL_UP  280
#define ANSI_KEY_SCROLL_DN  281

/* Mouse event data — valid when ansi_get_key() returns
 * ANSI_KEY_MOUSE, ANSI_KEY_SCROLL_UP, or ANSI_KEY_SCROLL_DN. */
struct ansi_mouse_event
{
    int x;   /* 1-based column */
    int y;   /* 1-based row    */
    int btn; /* 0=left, 1=mid, 2=right, 64=scrollup, 65=scrolldn */
};

extern struct ansi_mouse_event ansi__last_mouse;

/* ctrl(x) helper — same as CLIcore convention */
#ifndef ctrl
#define ctrl(x) ((x) & 0x1f)
#endif

/* =========================================================
 * Terminal state
 * ========================================================= */

extern struct termios ansi__orig_termios;
extern int            ansi__raw_active;

/**
 * ansi_raw_mode_enter - switch stdin to raw non-canonical mode.
 *
 * Saves the original termios so it can be restored on exit.
 */
static inline void ansi_raw_mode_enter(void)
{
    struct termios raw;

    if(ansi__raw_active)
    {
        return;
    }

    if(tcgetattr(STDIN_FILENO, &ansi__orig_termios) == -1)
    {
        return;
    }
    raw = ansi__orig_termios;
    raw.c_iflag &= ~(unsigned int)(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
    raw.c_oflag &= ~(unsigned int)(OPOST);
    raw.c_cflag |=  (unsigned int)(CS8);
    raw.c_lflag &= ~(unsigned int)(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    /* non-blocking stdin */
    {
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }

    /* hide cursor, disable line wrap, enable SGR mouse tracking */
    if(write(STDOUT_FILENO,
             "\033[?25l\033[?7l"
             "\033[?1000h\033[?1006h",
             11 + 10 + 10) < 0)
    {
    }
    ansi__raw_active = 1;
}

/**
 * ansi_raw_mode_exit - restore the original terminal settings.
 */
static inline void ansi_raw_mode_exit(void)
{
    if(!ansi__raw_active)
    {
        return;
    }
    /* disable mouse tracking, show cursor, enable line wrap */
    if(write(STDOUT_FILENO,
             "\033[?1006l\033[?1000l"
             "\033[?25h\033[?7h",
             10 + 10 + 11) < 0)
    {
    }
    /* clear screen, reset attributes, home cursor */
    if(write(STDOUT_FILENO, "\033[0m\033[2J\033[H", 11) < 0)
    {
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &ansi__orig_termios);
    ansi__raw_active = 0;
}

/* =========================================================
 * Terminal size
 * ========================================================= */

/**
 * ansi_get_terminal_size - query terminal dimensions via ioctl.
 * @rows: pointer to store row count
 * @cols: pointer to store column count
 */
static inline void ansi_get_terminal_size(int *rows, int *cols)
{
    struct winsize ws;

    *rows = 24;
    *cols = 80;
    if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0)
    {
        if(ws.ws_row > 0)
        {
            *rows = (int) ws.ws_row;
        }
        if(ws.ws_col > 0)
        {
            *cols = (int) ws.ws_col;
        }
    }
}

/* =========================================================
 * Screen control
 * ========================================================= */

/** ansi_cls - clear entire screen and move cursor to top-left. */
static inline void ansi_cls(void)
{
    if(write(STDOUT_FILENO, "\033[2J\033[H", 7) < 0) {}
}

/**
 * ansi_pos - move cursor to (row, col), 1-indexed.
 * @row: target row (1 = top)
 * @col: target column (1 = left)
 */
static inline void ansi_pos(int row, int col)
{
    char buf[32];
    int  n = snprintf(buf, sizeof(buf), "\033[%d;%dH", row, col);
    if(n > 0)
    {
        if(write(STDOUT_FILENO, buf, (size_t) n) < 0) {}
    }
}

/* =========================================================
 * Terminal capabilities
 * ========================================================= */

static int ansi__color_level = 0; // 0=uninit, 1=16-color, 2=256-color, 3=TrueColor

static inline void ansi_detect_color_level(void)
{
    if (ansi__color_level > 0)
    {
        return;
    }

    const char *term = getenv("TERM");
    const char *colorterm = getenv("COLORTERM");

    if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
    {
        ansi__color_level = 3;
    }
    else if (term && strstr(term, "256color"))
    {
        ansi__color_level = 2;
    }
    else
    {
        ansi__color_level = 1;
    }
}

/* =========================================================
 * Color / attribute helpers
 * ========================================================= */

/**
 * ansi_fg - set TrueColor foreground (RGB, 0–255 each).
 * @r: red component
 * @g: green component
 * @b: blue component
 */
static inline void ansi_fg(int r, int g, int b)
{
    char buf[32];
    int  n = snprintf(buf, sizeof(buf), "\033[38;2;%d;%d;%dm", r, g, b);
    if(n > 0)
    {
        if(write(STDOUT_FILENO, buf, (size_t) n) < 0) {}
    }
}

/**
 * ansi_bg - set TrueColor background (RGB, 0–255 each).
 * @r: red component
 * @g: green component
 * @b: blue component
 */
static inline void ansi_bg(int r, int g, int b)
{
    char buf[32];
    int  n = snprintf(buf, sizeof(buf), "\033[48;2;%d;%d;%dm", r, g, b);
    if(n > 0)
    {
        if(write(STDOUT_FILENO, buf, (size_t) n) < 0) {}
    }
}

/**
 * ansi_fg_256 - set 256-color foreground.
 * @code: 8-bit color code (0-255)
 */
static inline void ansi_fg_256(int code)
{
    char buf[32];
    int  n = snprintf(buf, sizeof(buf), "\033[38;5;%dm", code);
    if(n > 0)
    {
        if(write(STDOUT_FILENO, buf, (size_t) n) < 0) {}
    }
}

/**
 * ansi_fg_16 - set 16-color standard foreground.
 * @code: ANSI standard color code (e.g., 32 for green)
 */
static inline void ansi_fg_16(int code)
{
    char buf[32];
    int  n = snprintf(buf, sizeof(buf), "\033[%dm", code);
    if(n > 0)
    {
        if(write(STDOUT_FILENO, buf, (size_t) n) < 0) {}
    }
}

/** ansi_bold_on - enable bold/bright text attribute. */
static inline void ansi_bold_on(void)
{
    if(write(STDOUT_FILENO, "\033[1m", 4) < 0) {}
}

/** ansi_bold_off - disable bold/bright text attribute. */
static inline void ansi_bold_off(void)
{
    if(write(STDOUT_FILENO, "\033[22m", 5) < 0) {}
}

/** ansi_reverse_on - enable reverse-video attribute. */
static inline void ansi_reverse_on(void)
{
    if(write(STDOUT_FILENO, "\033[7m", 4) < 0) {}
}

/** ansi_reverse_off - disable reverse-video attribute. */
static inline void ansi_reverse_off(void)
{
    if(write(STDOUT_FILENO, "\033[27m", 5) < 0) {}
}

/** ansi_reset - reset all attributes (color + style) to default. */
static inline void ansi_reset(void)
{
    if(write(STDOUT_FILENO, "\033[0m", 4) < 0) {}
}

/* =========================================================
 * Colour palette — maps old screenprint_setcolor() indices
 * to RGB triples for backward compatibility.
 *
 * Index mapping (matches TUItools legacy palette):
 *   0  default/white
 *   2  green  (active process / received trigger)
 *   3  yellow (upstream level 0)
 *   4  red    (error)
 *   5  magenta (symlink)
 *   7  cyan   (upstream level > 0)
 *   9  orange (filter / fuser ongoing)
 *  12  bright-green (upstream active)
 * ========================================================= */

static inline void ansi_setcolor(int idx)
{
    ansi_detect_color_level();

    if (ansi__color_level >= 3)
    {
        /* TrueColor (24-bit) */
        switch(idx)
        {
        case 2:  ansi_fg(80,  220, 80);  break; /* green        */
        case 3:  ansi_fg(220, 200, 0);   break; /* yellow       */
        case 4:  ansi_fg(240, 60,  60);  break; /* red          */
        case 5:  ansi_fg(200, 80,  220); break; /* magenta      */
        case 7:  ansi_fg(0,   200, 220); break; /* cyan         */
        case 9:  ansi_fg(255, 140, 0);   break; /* orange       */
        case 12: ansi_fg(100, 255, 100); break; /* bright-green */
        default: ansi_reset(); break;
        }
    }
    else if (ansi__color_level == 2)
    {
        /* 256-color fallback */
        switch(idx)
        {
        case 2:  ansi_fg_256(114); break; /* green        */
        case 3:  ansi_fg_256(220); break; /* yellow       */
        case 4:  ansi_fg_256(203); break; /* red          */
        case 5:  ansi_fg_256(176); break; /* magenta      */
        case 7:  ansi_fg_256(44);  break; /* cyan         */
        case 9:  ansi_fg_256(208); break; /* orange       */
        case 12: ansi_fg_256(119); break; /* bright-green */
        default: ansi_reset();     break;
        }
    }
    else
    {
        /* 16-color (standard ANSI) fallback */
        switch(idx)
        {
        case 2:  ansi_fg_16(32); break; /* green        */
        case 3:  ansi_fg_16(33); break; /* yellow       */
        case 4:  ansi_fg_16(31); break; /* red          */
        case 5:  ansi_fg_16(35); break; /* magenta      */
        case 7:  ansi_fg_16(36); break; /* cyan         */
        case 9:  ansi_fg_16(33); break; /* orange -> yellow */
        case 12: ansi_fg_16(92); break; /* bright-green */
        default: ansi_reset();   break;
        }
    }
}

static inline void ansi_unsetcolor(int idx)
{
    (void) idx;
    ansi_reset();
}

/* =========================================================
 * Keyboard input
 * ========================================================= */

/**
 * ansi_get_key - read one key event from stdin (non-blocking).
 *
 * Returns a plain ASCII code for printable characters and
 * ctrl sequences, or one of the ANSI_KEY_* constants for
 * special keys.  Returns ANSI_KEY_NONE (0) if no input is
 * available.
 */
static inline int ansi_get_key(void)
{
    static unsigned char buf[256];
    static int           buf_len = 0;
    ssize_t              n;

    /* Read whatever is available into the remaining buffer space */
    n = read(STDIN_FILENO, buf + buf_len, sizeof(buf) - buf_len);
    if(n > 0)
    {
        buf_len += n;
    }

    if(buf_len == 0)
    {
        return ANSI_KEY_NONE;
    }

    /* single ASCII byte, not ESC */
    if(buf[0] != 0x1b)
    {
        int key = (int) buf[0];
        memmove(buf, buf + 1, buf_len - 1);
        buf_len--;
        return key;
    }

    /* Escape sequence */
    if(buf_len >= 2)
    {
        /* CSI sequences: ESC [ ... */
        if(buf[1] == '[')
        {
            if(buf_len >= 3)
            {
                int consumed = 0;
                int key      = 0;

                /* ---- SGR mouse: ESC [ < Btn;X;Y M/m ---- */
                if(buf[2] == '<')
                {
                    /* Find terminator: 'M' (press) or 'm' (release) */
                    int term_idx = -1;
                    for(int ii = 3; ii < buf_len && ii < 32; ii++)
                    {
                        if(buf[ii] == 'M' || buf[ii] == 'm')
                        {
                            term_idx = ii;
                            break;
                        }
                    }
                    if(term_idx == -1)
                    {
                        /* Incomplete — wait for more bytes */
                        return ANSI_KEY_NONE;
                    }

                    int mbtn = 0, mx = 1, my = 1;
                    {
                        /* Parse "Btn;X;Y" between buf[3..term_idx-1] */
                        char tmp[64];
                        int  tlen = term_idx - 3;
                        if(tlen > 0 && tlen < (int) sizeof(tmp))
                        {
                            memcpy(tmp, buf + 3, tlen);
                            tmp[tlen] = '\0';
                            sscanf(tmp, "%d;%d;%d", &mbtn, &mx, &my);
                        }
                    }

                    /* Save terminator before consume */
                    unsigned char term_ch = buf[term_idx];

                    consumed = term_idx + 1;
                    memmove(buf, buf + consumed, buf_len - consumed);
                    buf_len -= consumed;

                    ansi__last_mouse.btn = mbtn;
                    ansi__last_mouse.x   = mx;
                    ansi__last_mouse.y   = my;

                    /* Only act on press ('M'), ignore release ('m') */
                    if(term_ch == 'm')
                    {
                        return ANSI_KEY_NONE;
                    }

                    /* Scroll wheel: btn 64 = up, 65 = down */
                    if(mbtn == 64)
                    {
                        return ANSI_KEY_SCROLL_UP;
                    }
                    if(mbtn == 65)
                    {
                        return ANSI_KEY_SCROLL_DN;
                    }

                    /* Left-click press (btn 0) */
                    if(mbtn == 0)
                    {
                        return ANSI_KEY_MOUSE;
                    }

                    /* Other buttons — consume but ignore */
                    return ANSI_KEY_NONE;
                } /* end SGR mouse */

                switch(buf[2])
                {
                case 'A': key = ANSI_KEY_UP; consumed = 3; break;
                case 'B': key = ANSI_KEY_DOWN; consumed = 3; break;
                case 'C': key = ANSI_KEY_RIGHT; consumed = 3; break;
                case 'D': key = ANSI_KEY_LEFT; consumed = 3; break;
                case 'H': key = ANSI_KEY_HOME; consumed = 3; break;
                case 'F': key = ANSI_KEY_END; consumed = 3; break;
                default:  break;
                }

                if(key)
                {
                    memmove(buf, buf + consumed, buf_len - consumed);
                    buf_len -= consumed;
                    return key;
                }

                /* Extended: ESC [ <digits> ~ */
                int tilde_idx = -1;
                for(int i = 2; i < buf_len && i < 10; i++)
                {
                    if(buf[i] == '~')
                    {
                        tilde_idx = i;
                        break;
                    }
                    if(buf[i] >= 0x40 && buf[i] <= 0x7E)
                    {
                        break; /* Other terminator */
                    }
                }

                if(tilde_idx != -1)
                {
                    int code = atoi((char *) buf + 2);
                    consumed = tilde_idx + 1;
                    switch(code)
                    {
                    case 1:  key = ANSI_KEY_HOME; break;
                    case 3:  key = ANSI_KEY_DEL; break;
                    case 4:  key = ANSI_KEY_END; break;
                    case 5:  key = ANSI_KEY_PGUP; break;
                    case 6:  key = ANSI_KEY_PGDN; break;
                    case 11: key = ANSI_KEY_F1; break;
                    case 12: key = ANSI_KEY_F2; break;
                    case 13: key = ANSI_KEY_F3; break;
                    case 14: key = ANSI_KEY_F4; break;
                    case 15: key = ANSI_KEY_F5; break;
                    case 17: key = ANSI_KEY_F6; break;
                    case 18: key = ANSI_KEY_F7; break;
                    case 19: key = ANSI_KEY_F8; break;
                    case 20: key = ANSI_KEY_F9; break;
                    case 21: key = ANSI_KEY_F10; break;
                    case 23: key = ANSI_KEY_F11; break;
                    case 24: key = ANSI_KEY_F12; break;
                    default: break;
                    }
                    if(key)
                    {
                        memmove(buf, buf + consumed, buf_len - consumed);
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* CTRL+Arrow: ESC [ 1 ; 5 C/D */
                if(buf_len >= 6 && buf[2] == '1' && buf[3] == ';' && buf[4] == '5')
                {
                    if(buf[5] == 'D')
                    {
                        key = ANSI_KEY_CTRL_LEFT;
                        consumed = 6;
                    }
                    else if(buf[5] == 'C')
                    {
                        key = ANSI_KEY_CTRL_RIGHT;
                        consumed = 6;
                    }

                    if(key)
                    {
                        memmove(buf, buf + consumed, buf_len - consumed);
                        buf_len -= consumed;
                        return key;
                    }
                }

                /* Unrecognized or partial CSI sequence */
                /* If it has a terminator letter, consume it */
                for(int i = 2; i < buf_len; i++)
                {
                    if(buf[i] >= 0x40 && buf[i] <= 0x7E)
                    {
                        memmove(buf, buf + i + 1, buf_len - (i + 1));
                        buf_len -= (i + 1);
                        return ANSI_KEY_NONE;
                    }
                }
            }
            /* Need more bytes for CSI */
            return ANSI_KEY_NONE;
        }

        /* SS3 sequences: ESC O ... (xterm F1-F4) */
        if(buf[1] == 'O')
        {
            if(buf_len >= 3)
            {
                int key      = 0;
                int consumed = 3;
                switch(buf[2])
                {
                case 'P': key = ANSI_KEY_F1; break;
                case 'Q': key = ANSI_KEY_F2; break;
                case 'R': key = ANSI_KEY_F3; break;
                case 'S': key = ANSI_KEY_F4; break;
                default:  break;
                }
                memmove(buf, buf + consumed, buf_len - consumed);
                buf_len -= consumed;
                return key ? key : ANSI_KEY_NONE;
            }
            return ANSI_KEY_NONE;
        }

        /* bare ESC or alt+key */
        /* For now, just return ESC and consume 1 byte */
        memmove(buf, buf + 1, buf_len - 1);
        buf_len--;
        return 0x1b;
    }

    /* Wait for more bytes for escape sequence */
    return ANSI_KEY_NONE;
}

#endif /* _STREAMCTRL_ANSI_H */
