// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_ansi.h"

struct termios ov__orig_termios;
int            ov__orig_flags  = -1;
int            ov__raw_active  = 0;
int            ov__color_level = 0;
/**
 * ov_raw_mode_enter - Configure terminal in non-canonical raw mode
 *
 * Saves original termios settings, disables ECHO/ICANON/signals, sets non-blocking
 * standard input, switches to alternate screen buffer, and enables SGR mouse reporting.
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

    ov__orig_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (ov__orig_flags >= 0)
    {
        fcntl(STDIN_FILENO, F_SETFL, ov__orig_flags | O_NONBLOCK);
    }

    const char seq[] = "\033[?1049h\033[?25l\033[?7l\033[?1002h\033[?1006h";
    if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0)
    {
    }
    ov__raw_active = 1;
}

/**
 * ov_raw_mode_exit - Restore original terminal settings and exit raw mode
 *
 * Restores original termios, re-enables cursor, disables mouse reporting,
 * and switches back to the primary terminal screen buffer.
 */
void ov_raw_mode_exit(void)
{
    if (!ov__raw_active)
    {
        return;
    }
    const char seq[] = "\033[?1003l\033[?1006l\033[?1002l\033[?25h\033[?7h\033[0m\033[?1049l";
    if (write(STDOUT_FILENO, seq, sizeof(seq) - 1) < 0)
    {
    }
    if (ov__orig_flags >= 0)
    {
        fcntl(STDIN_FILENO, F_SETFL, ov__orig_flags);
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &ov__orig_termios);
    ov__raw_active = 0;
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

/**
 * ov_get_terminal_size - Query current terminal dimensions via ioctl
 * @rows: Output pointer for number of terminal rows
 * @cols: Output pointer for number of terminal columns
 */
void ov_get_terminal_size(
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
