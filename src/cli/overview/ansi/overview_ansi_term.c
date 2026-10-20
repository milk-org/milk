// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"

#include <langinfo.h>
#include <locale.h>

struct termios        ov__orig_termios;
volatile sig_atomic_t ov__raw_active  = 0;
int                   ov__color_level = 0;
int                   ov__tty_fd      = STDOUT_FILENO;
static int            ov__saved_out   = -1;
static int            ov__saved_err   = -1;
static int            ov__cap[2]      = { -1, -1 };

static const char ov__restore_seq[] =
    "\033[?2004l\033[?2026l\033[?1003l\033[?1006l\033[?1002l\033[?25h\033[?7h\033[0m\033[?1049l";

/**
 * ov_capture_stdio - point fds 1 and 2 at a pipe while the TUI owns the terminal.
 *
 * Library printf/perror output then reaches the command log (ov_cmdlog_drain_fd) instead of
 * corrupting the delta-rendered screen. The TUI itself writes to ov__tty_fd.
 */
static void ov_capture_stdio(void)
{
    fflush(stdout);
    fflush(stderr);
    if (ov__cap[0] < 0)
    {
        if (pipe2(ov__cap, O_CLOEXEC) != 0)
        {
            ov__cap[0] = ov__cap[1] = -1;
            return;
        }
        fcntl(ov__cap[0], F_SETFL, fcntl(ov__cap[0], F_GETFL) | O_NONBLOCK);
    }
    ov__saved_out = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
    ov__saved_err = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
    if (ov__saved_out < 0 || ov__saved_err < 0)
    {
        return;
    }
    ov__tty_fd = ov__saved_out;
    dup2(ov__cap[1], STDOUT_FILENO);
    dup2(ov__cap[1], STDERR_FILENO);
}

/**
 * ov_release_stdio - give fds 1 and 2 back to the terminal.
 */
static void ov_release_stdio(void)
{
    fflush(stdout);
    fflush(stderr);
    if (ov__saved_out >= 0)
    {
        dup2(ov__saved_out, STDOUT_FILENO);
        close(ov__saved_out);
        ov__saved_out = -1;
    }
    if (ov__saved_err >= 0)
    {
        dup2(ov__saved_err, STDERR_FILENO);
        close(ov__saved_err);
        ov__saved_err = -1;
    }
    clearerr(stdout);
    clearerr(stderr);
    ov__tty_fd = STDOUT_FILENO;
}

/**
 * ov_raw_mode_enter - Configure terminal in non-canonical raw mode
 *
 * Saves original termios settings, disables ECHO/ICANON/signals, captures stdout/stderr,
 * switches to alternate screen buffer, and enables SGR mouse reporting. stdin stays blocking:
 * VMIN=0/VTIME=0 already makes read() return at once.
 */
void ov_raw_mode_enter(void)
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
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    ov_capture_stdio();

    const char seq[] = "\033[?1049h\033[?25l\033[?7l\033[?1002h\033[?1006h\033[?2004h";
    if (write(ov__tty_fd, seq, sizeof(seq) - 1) < 0)
    {
    }
    ov__raw_active = 1;
}

/**
 * ov_raw_mode_exit - Restore original terminal settings and exit raw mode
 *
 * Restores original termios, re-enables cursor, disables mouse reporting, gives stdout/stderr
 * back to the terminal and switches back to the primary terminal screen buffer.
 */
void ov_raw_mode_exit(void)
{
    if (!ov__raw_active)
    {
        return;
    }
    if (write(ov__tty_fd, ov__restore_seq, sizeof(ov__restore_seq) - 1) < 0)
    {
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &ov__orig_termios);
    ov_release_stdio();
    ov__raw_active = 0;
}

/**
 * ov_term_restore_async - restore the terminal from a fatal-signal handler (async-signal-safe).
 */
void ov_term_restore_async(void)
{
    if (!ov__raw_active)
    {
        return;
    }
    if (write(ov__tty_fd, ov__restore_seq, sizeof(ov__restore_seq) - 1) < 0)
    {
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &ov__orig_termios);
    if (ov__saved_err >= 0)
    {
        dup2(ov__saved_err, STDERR_FILENO);
    }
    if (ov__saved_out >= 0)
    {
        dup2(ov__saved_out, STDOUT_FILENO);
    }
}

/**
 * ov_term_capture_fd - read end of the stdout/stderr capture pipe, or -1.
 */
int ov_term_capture_fd(void)
{
    return ov__cap[0];
}

/**
 * ov_term_init_locale - select the user's LC_CTYPE so wcwidth() knows non-ASCII glyphs.
 *
 * Falls back to C.UTF-8 when the user's locale is not UTF-8; LC_NUMERIC is left alone.
 * Return: 1 if the user's locale is UTF-8, 0 otherwise.
 */
int ov_term_init_locale(void)
{
    if (setlocale(LC_CTYPE, "") != NULL && strcmp(nl_langinfo(CODESET), "UTF-8") == 0)
    {
        return 1;
    }
    setlocale(LC_CTYPE, "C.UTF-8");
    return 0;
}

/**
 * ov_set_mouse_hover - Toggle mouse movement / hover tracking
 * @enable: 1 to enable all-motion mouse tracking (1003h), 0 for click-only (1002h)
 */
void ov_set_mouse_hover(int enable)
{
    if (enable)
    {
        const char seq[] = "\033[?1002l\033[?1003h";
        if (write(ov__tty_fd, seq, sizeof(seq) - 1) < 0)
        {
        }
    }
    else
    {
        const char seq[] = "\033[?1003l\033[?1002h";
        if (write(ov__tty_fd, seq, sizeof(seq) - 1) < 0)
        {
        }
    }
}

/**
 * ov_get_terminal_size - Query current terminal dimensions via ioctl
 * @rows: Output pointer for number of terminal rows
 * @cols: Output pointer for number of terminal columns
 */
void ov_get_terminal_size(int *rows, int *cols)
{
    struct winsize ws;
    *rows = 24;
    *cols = 80;
    if (ioctl(ov__tty_fd, TIOCGWINSZ, &ws) == 0)
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

/**
 * ov_detect_color_level - Detect terminal color capabilities from environment variables
 *
 * Sets ov__color_level to 3 (TrueColor), 2 (256-color), or 1 (16-color) based on
 * COLORTERM and TERM environment variables.
 */
void ov_detect_color_level(void)
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
