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

#ifndef _FPSCTRL_ANSI_H
#define _FPSCTRL_ANSI_H

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
#define ANSI_KEY_RESIZE    299
#define ANSI_KEY_CTRL_LEFT  277
#define ANSI_KEY_CTRL_RIGHT 278

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
    raw.c_cflag |= (unsigned int)(CS8);
    raw.c_lflag &= ~(unsigned int)(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    /* non-blocking stdin */
    {
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }

    /* hide cursor and disable line wrap */
    if(write(STDOUT_FILENO, "\033[?25l\033[?7l", 11) < 0) {}
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
    /* show cursor and enable line wrap */
    if(write(STDOUT_FILENO, "\033[?25h\033[?7h", 11) < 0) {}
    /* clear screen, home cursor */
    if(write(STDOUT_FILENO, "\033[2J\033[H", 7) < 0) {}
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
    if(ansi__color_level > 0)
    {
        return;
    }

    const char *term = getenv("TERM");
    const char *colorterm = getenv("COLORTERM");

    if(colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
    {
        ansi__color_level = 3;
    }
    else if(term && strstr(term, "256color"))
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

    if(ansi__color_level >= 3)
    {
        /* TrueColor (24-bit) */
        switch(idx)
        {
        case 2:
            ansi_fg(80,  220, 80);
            break; /* green        */
        case 3:
            ansi_fg(220, 200, 0);
            break; /* yellow       */
        case 4:
            ansi_fg(240, 60,  60);
            break; /* red          */
        case 5:
            ansi_fg(200, 80,  220);
            break; /* magenta      */
        case 7:
            ansi_fg(0,   200, 220);
            break; /* cyan         */
        case 9:
            ansi_fg(255, 140, 0);
            break; /* orange       */
        case 12:
            ansi_fg(100, 255, 100);
            break; /* bright-green */
        default:
            ansi_reset();
            break;
        }
    }
    else if(ansi__color_level == 2)
    {
        /* 256-color fallback */
        switch(idx)
        {
        case 2:
            ansi_fg_256(114);
            break; /* green        */
        case 3:
            ansi_fg_256(220);
            break; /* yellow       */
        case 4:
            ansi_fg_256(203);
            break; /* red          */
        case 5:
            ansi_fg_256(176);
            break; /* magenta      */
        case 7:
            ansi_fg_256(44);
            break; /* cyan         */
        case 9:
            ansi_fg_256(208);
            break; /* orange       */
        case 12:
            ansi_fg_256(119);
            break; /* bright-green */
        default:
            ansi_reset();
            break;
        }
    }
    else
    {
        /* 16-color (standard ANSI) fallback */
        switch(idx)
        {
        case 2:
            ansi_fg_16(32);
            break; /* green        */
        case 3:
            ansi_fg_16(33);
            break; /* yellow       */
        case 4:
            ansi_fg_16(31);
            break; /* red          */
        case 5:
            ansi_fg_16(35);
            break; /* magenta      */
        case 7:
            ansi_fg_16(36);
            break; /* cyan         */
        case 9:
            ansi_fg_16(33);
            break; /* orange -> yellow */
        case 12:
            ansi_fg_16(92);
            break; /* bright-green */
        default:
            ansi_reset();
            break;
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
    unsigned char buf[16];
    ssize_t       n;

    n = read(STDIN_FILENO, buf, sizeof(buf));
    if(n <= 0)
    {
        return ANSI_KEY_NONE;
    }

    /* single ASCII byte */
    if(n == 1)
    {
        return (int) buf[0];
    }

    /* Escape sequence */
    if(buf[0] == 0x1b && n >= 2)
    {
        /* CSI sequences: ESC [ ... */
        if(buf[1] == '[' && n >= 3)
        {
            switch(buf[2])
            {
            case 'A':
                return ANSI_KEY_UP;
            case 'B':
                return ANSI_KEY_DOWN;
            case 'C':
                return ANSI_KEY_RIGHT;
            case 'D':
                return ANSI_KEY_LEFT;
            case 'H':
                return ANSI_KEY_HOME;
            case 'F':
                return ANSI_KEY_END;
            default:
                break;
            }

            /* Extended: ESC [ <digits> ~ */
            if(n >= 4 && buf[n - 1] == '~')
            {
                int code = atoi((char *) buf + 2);
                switch(code)
                {
                case 1:
                    return ANSI_KEY_HOME;
                case 3:
                    return ANSI_KEY_DEL;
                case 4:
                    return ANSI_KEY_END;
                case 5:
                    return ANSI_KEY_PGUP;
                case 6:
                    return ANSI_KEY_PGDN;
                case 11:
                    return ANSI_KEY_F1;
                case 12:
                    return ANSI_KEY_F2;
                case 13:
                    return ANSI_KEY_F3;
                case 14:
                    return ANSI_KEY_F4;
                case 15:
                    return ANSI_KEY_F5;
                case 17:
                    return ANSI_KEY_F6;
                case 18:
                    return ANSI_KEY_F7;
                case 19:
                    return ANSI_KEY_F8;
                case 20:
                    return ANSI_KEY_F9;
                case 21:
                    return ANSI_KEY_F10;
                case 23:
                    return ANSI_KEY_F11;
                case 24:
                    return ANSI_KEY_F12;
                default:
                    break;
                }
            }

            /* CTRL+Arrow: ESC [ 1 ; 5 C/D */
            if(n >= 6 && buf[2] == '1'
                    && buf[3] == ';'
                    && buf[4] == '5')
            {
                if(buf[5] == 'D')
                {
                    return ANSI_KEY_CTRL_LEFT;
                }
                if(buf[5] == 'C')
                {
                    return ANSI_KEY_CTRL_RIGHT;
                }
            }
        }

        /* SS3 sequences: ESC O ... (xterm F1-F4) */
        if(buf[1] == 'O' && n >= 3)
        {
            switch(buf[2])
            {
            case 'P':
                return ANSI_KEY_F1;
            case 'Q':
                return ANSI_KEY_F2;
            case 'R':
                return ANSI_KEY_F3;
            case 'S':
                return ANSI_KEY_F4;
            default:
                break;
            }
        }

        /* bare ESC */
        if(n == 2 && buf[1] == 0x1b)
        {
            return 0x1b;
        }
    }

    return (int) buf[0];
}

#endif /* _PROCCTRL_ANSI_H */
