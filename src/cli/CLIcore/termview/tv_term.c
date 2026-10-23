// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_term.c
 * @brief Terminal lifecycle: raw mode, alternate screen, signals, restore on every exit path.
 */

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <termios.h>
#include <unistd.h>

#include "termview_internal.h"

volatile sig_atomic_t tv_sig_quit;
volatile sig_atomic_t tv_sig_winch;
volatile sig_atomic_t tv_sig_cont;
volatile sig_atomic_t tv_term_hup;
volatile sig_atomic_t tv_sig_nquit;
volatile sig_atomic_t tv_sig_tstp;

static struct termios        tv_tio_orig;
static volatile sig_atomic_t tv_term_active;
static int                   tv_err_fd  = -1; /* the original stderr, for exit and crash messages */
static int                   tv_null_fd = -1; /* /dev/null: stderr while the screen is ours */
static volatile sig_atomic_t tv_err_quiet;
static int                   tv_tstp_ignored; /* SIGTSTP was ignored at startup: no stops */
static char                  tv_altstack[65536];

/* Alt screen, hide cursor, no autowrap, button-event mouse tracking, SGR encoding. */
static const char TV_SEQ_ENTER[] = "\033[?1049h\033[?25l\033[?7l\033[?1002h\033[?1006h";

/* End sync, mouse off before its encoding, reset SGR, autowrap and cursor on, leave alt screen. */
static const char TV_SEQ_LEAVE[] = "\033[?2026l\033[?1003l\033[?1002l\033[?1000l\033[?1006l"
                                   "\033[0m\033[?7h\033[?25h\033[?1049l";

#ifndef TV_KEEP_STDERR
/** tv_err_on_screen() - stderr is the terminal the image is drawn on (or /dev/tty). */
static int tv_err_on_screen(void)
{
    struct stat o, e;
    if (fstat(STDOUT_FILENO, &o) != 0 || fstat(STDERR_FILENO, &e) != 0 || !S_ISCHR(o.st_mode) ||
        !S_ISCHR(e.st_mode))
    {
        return 0;
    }
    return e.st_rdev == o.st_rdev || e.st_rdev == makedev(5, 0);
}
#endif

/** tv_term_restore() - Idempotent, async-signal-safe: leave sequence, then original termios. */
void tv_term_restore(void)
{
    if (!tv_term_active)
    {
        return;
    }
    tv_term_active = 0;

    size_t off = 0;
    while (off < sizeof(TV_SEQ_LEAVE) - 1)
    {
        ssize_t w = write(STDOUT_FILENO, TV_SEQ_LEAVE + off, sizeof(TV_SEQ_LEAVE) - 1 - off);
        if (w > 0)
        {
            off += (size_t) w;
        }
        else if (!(w < 0 && errno == EINTR))
        {
            break;
        }
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &tv_tio_orig);
    tcflush(STDIN_FILENO, TCIFLUSH);
    if (tv_err_quiet)
    {
        dup2(tv_err_fd, STDERR_FILENO);
        tv_err_quiet = 0;
    }
}

/**
 * tv_on_quit - Signal handler for clean termination signals.
 * @sig: Signal number received (SIGINT, SIGTERM, SIGHUP, or SIGQUIT).
 *
 * Records the signal number in @tv_sig_quit and increments the quit signal
 * counter @tv_sig_nquit so the event loop and flow-control settle loop wake
 * up and exit cleanly.
 *
 * Return: Nothing.
 */
static void tv_on_quit(int sig)
{
    tv_sig_quit = sig;
    tv_sig_nquit++;
}

/**
 * tv_on_winch - Signal handler for terminal window size changes (SIGWINCH).
 * @sig: Signal number received (unused).
 *
 * Sets @tv_sig_winch so the main event loop queries the updated terminal
 * geometry and resizes the screen buffers on its next iteration.
 *
 * Return: Nothing.
 */
static void tv_on_winch(int sig)
{
    (void) sig;
    tv_sig_winch = 1;
}

/**
 * tv_on_cont - Signal handler for process continuation after stop (SIGCONT).
 * @sig: Signal number received (unused).
 *
 * Sets @tv_sig_cont to notify the main event loop that the process has
 * resumed and must re-enter raw terminal mode and repaint the screen.
 *
 * Return: Nothing.
 */
static void tv_on_cont(int sig)
{
    (void) sig;
    tv_sig_cont = 1;
}

/**
 * tv_on_tstp - Signal handler for interactive terminal stop (SIGTSTP).
 * @sig: Signal number received (unused).
 *
 * Flags @tv_sig_tstp so the main loop can drain outstanding terminal CPR
 * replies and restore normal terminal state before suspending the process.
 *
 * Return: Nothing.
 */
static void tv_on_tstp(int sig)
{
    (void) sig;
    tv_sig_tstp = 1; /* the loop reads the terminal's last answers, then tv_term_suspend() */
}

/**
 * tv_on_fatal - Signal handler for fatal crash signals on the alternate stack.
 * @sig: Fatal signal number received (SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL).
 *
 * Immediately restores the terminal state via tv_term_restore(), writes a
 * diagnostic notice to the saved original stderr descriptor, and re-raises
 * @sig with `SA_RESETHAND` so default kernel termination proceeds.
 *
 * Return: Nothing (does not return once @sig is re-raised).
 */
static void tv_on_fatal(int sig)
{
    static const char msg[] = "milk-termview: fatal signal, terminal restored\n";

    tv_term_restore();
    if (tv_err_fd >= 0)
    {
        ssize_t w = write(tv_err_fd, msg, sizeof(msg) - 1);
        (void) w;
    }
    raise(sig); /* SA_RESETHAND: default action now */
}

/**
 * tv_set_handler - Register a POSIX signal action without SA_RESTART.
 * @sig:   Signal number to configure.
 * @fn:    Signal handler callback function.
 * @flags: Additional sigaction flags (e.g., SA_RESETHAND | SA_ONSTACK).
 *
 * Installs @fn for @sig with an empty signal mask and without `SA_RESTART`
 * so blocking `poll()` and `write()` calls return `EINTR` immediately when
 * a signal arrives.
 *
 * Return: Nothing.
 */
static void tv_set_handler(int sig, void (*fn)(int), int flags)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = fn;
    sa.sa_flags   = flags; /* no SA_RESTART: poll() and write() see EINTR */
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, NULL);
}

/**
 * tv_term_init - Initialize terminal state, signal handlers, and exit hooks.
 * @err:   Destination buffer for an error message on failure.
 * @errsz: Size of @err in bytes.
 *
 * Verifies that stdin and stdout are TTYs, saves the original `termios`
 * attributes, duplicates stderr so it can be silenced while the alternate
 * screen is active, sets up an alternate signal stack, installs handlers
 * for quit, resize, job-control, and fatal signals, and registers
 * tv_term_restore() with `atexit()`.
 *
 * Return: 0 on success, or -1 if stdin/stdout are not terminals or termios fails.
 */
int tv_term_init(char *err, size_t errsz)
{
    static int registered = 0;

    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO))
    {
        snprintf(err, errsz, "stdin and stdout must be a terminal");
        return -1;
    }
    if (tcgetattr(STDIN_FILENO, &tv_tio_orig) != 0)
    {
        snprintf(err, errsz, "cannot read terminal attributes: %s", strerror(errno));
        return -1;
    }
    tv_err_fd = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
#ifndef TV_KEEP_STDERR /* sanitizer builds: keep their reports visible */
    if (tv_err_fd >= 0 && tv_null_fd < 0 && tv_err_on_screen())
    {
        tv_null_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
    }
#endif

    stack_t ss;
    memset(&ss, 0, sizeof(ss));
    ss.ss_sp   = tv_altstack;
    ss.ss_size = sizeof(tv_altstack);
    sigaltstack(&ss, NULL);

    tv_set_handler(SIGINT, tv_on_quit, 0);
    tv_set_handler(SIGTERM, tv_on_quit, 0);
    tv_set_handler(SIGHUP, tv_on_quit, 0);
    tv_set_handler(SIGQUIT, tv_on_quit, 0);
    tv_set_handler(SIGWINCH, tv_on_winch, 0);
    tv_set_handler(SIGCONT, tv_on_cont, 0);
    struct sigaction old;
    tv_tstp_ignored = sigaction(SIGTSTP, NULL, &old) == 0 && old.sa_handler == SIG_IGN;
    if (!tv_tstp_ignored)
    {
        tv_set_handler(SIGTSTP, tv_on_tstp, 0);
    }
    tv_set_handler(SIGPIPE, SIG_IGN, 0);

    int fatal_flags = SA_RESETHAND | SA_NODEFER | SA_ONSTACK;
    tv_set_handler(SIGSEGV, tv_on_fatal, fatal_flags);
    tv_set_handler(SIGBUS, tv_on_fatal, fatal_flags);
    tv_set_handler(SIGABRT, tv_on_fatal, fatal_flags);
    tv_set_handler(SIGFPE, tv_on_fatal, fatal_flags);
    tv_set_handler(SIGILL, tv_on_fatal, fatal_flags);

    if (!registered)
    {
        atexit(tv_term_restore);
        registered = 1;
    }
    return 0;
}

/**
 * tv_term_enter - Switch the terminal into raw mode and enter the alternate screen.
 *
 * Configures non-blocking raw `termios` on stdin, redirects stderr to
 * `/dev/null` if it shares the display terminal so background library
 * messages do not corrupt the TUI, and writes the ANSI enter sequence to
 * enable the alternate screen, hide the cursor, disable autowrap, and
 * activate SGR mouse reporting.
 *
 * Return: 0 on success, or -1 on termios or write failure.
 */
int tv_term_enter(void)
{
    struct termios raw = tv_tio_orig;
    raw.c_iflag &= ~(tcflag_t) (BRKINT | ICRNL | INPCK | ISTRIP | IXON | INLCR | IGNCR | PARMRK);
    raw.c_oflag &= ~(tcflag_t) OPOST;
    raw.c_cflag |= CS8;
    raw.c_lflag &= ~(tcflag_t) (ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;

    tv_term_active = 1; /* set first: a signal mid-way still restores */
    /* EINTR: stopped by SIGTTOU while in the background and resumed by fg; just retry */
    int rc;
    while ((rc = tcsetattr(STDIN_FILENO, TCSANOW, &raw)) != 0 && errno == EINTR && !tv_sig_quit)
    {
    }
    if (rc != 0)
    {
        return -1;
    }
    tv_sig_cont = 0; /* the resume is handled: do not let the enter sequence's write abort */
    /* library messages written to a terminal stderr would land in the middle of the image */
    if (tv_null_fd >= 0 && dup2(tv_null_fd, STDERR_FILENO) >= 0)
    {
        tv_err_quiet = 1;
    }
    return tv_write_all(TV_SEQ_ENTER, sizeof(TV_SEQ_ENTER) - 1);
}

/**
 * tv_term_can_suspend - Check whether interactive job suspension (SIGTSTP) is allowed.
 *
 * Verifies that `SIGTSTP` was not ignored when the process started and is
 * not currently blocked in the process signal mask.
 *
 * Return: 1 if the process can be suspended via SIGTSTP, or 0 otherwise.
 */
int tv_term_can_suspend(void)
{
    sigset_t mask;
    return !tv_tstp_ignored && sigprocmask(SIG_BLOCK, NULL, &mask) == 0 &&
           !sigismember(&mask, SIGTSTP);
}

/**
 * tv_term_is_active - Query whether the terminal is currently in raw/TUI mode.
 *
 * Returns the state flag set by tv_term_enter() and cleared by
 * tv_term_restore().
 *
 * Return: Non-zero if raw terminal mode is active, or 0 if restored.
 */
int tv_term_is_active(void)
{
    return tv_term_active;
}

/**
 * tv_term_suspend - Restore the terminal and suspend the process with SIGTSTP.
 *
 * Leaves the alternate screen and restores original `termios` settings,
 * resets `SIGTSTP` to `SIG_DFL`, raises `SIGTSTP` to stop the process, and
 * upon resumption reinstalls `tv_on_tstp` and sets @tv_sig_cont.
 *
 * Return: Nothing.
 */
void tv_term_suspend(void)
{
    tv_sig_tstp = 0;
    if (!tv_term_can_suspend())
    {
        return;
    }
    tv_term_restore();
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTSTP, &sa, NULL);
    raise(SIGTSTP); /* stops here until SIGCONT */
    tv_set_handler(SIGTSTP, tv_on_tstp, 0);
    tv_sig_cont = 1;
}

/**
 * tv_term_size - Query the current terminal dimensions in character cells.
 * @rows: Output pointer receiving the terminal height in rows.
 * @cols: Output pointer receiving the terminal width in columns.
 *
 * Queries `TIOCGWINSZ` on stdout, falling back to 24x80 if the ioctl fails
 * or reports zero dimensions, and clamps the result to `TV_MAX_ROWS` and
 * `TV_MAX_COLS`.
 *
 * Return: Nothing.
 */
void tv_term_size(int *rows, int *cols)
{
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == -1 || ws.ws_row == 0 || ws.ws_col == 0)
    {
        *rows = 24;
        *cols = 80;
        return;
    }
    *rows = (ws.ws_row > TV_MAX_ROWS) ? TV_MAX_ROWS : ws.ws_row;
    *cols = (ws.ws_col > TV_MAX_COLS) ? TV_MAX_COLS : ws.ws_col;
}

/**
 * tv_write_all - Write an entire buffer to stdout, handling partial writes and EAGAIN.
 * @buf: Pointer to the byte buffer to write.
 * @n:   Number of bytes to write.
 *
 * Loops over `write(STDOUT_FILENO, ...)` until all @n bytes are sent. Aborts
 * early returning -1 if a quit, stop, or continue signal arrives mid-frame,
 * or sets @tv_term_hup and returns -1 on unrecoverable I/O errors.
 *
 * Return: 0 when all bytes are written, or -1 if interrupted by a signal or I/O error.
 */
int tv_write_all(const char *buf, size_t n)
{
    while (n > 0)
    {
        ssize_t w = write(STDOUT_FILENO, buf, n);
        if (w > 0)
        {
            buf += w;
            n -= (size_t) w;
            if (n > 0 && (tv_sig_quit || tv_sig_cont || tv_sig_tstp))
            {
                return -1; /* a signal cut a blocked write short (not EINTR: bytes went) */
            }
            continue;
        }
        if (w < 0 && errno == EINTR)
        {
            if (tv_sig_quit || tv_sig_cont || tv_sig_tstp)
            {
                return -1; /* abandon the frame; caller repaints */
            }
            continue;
        }
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            struct pollfd p = { .fd = STDOUT_FILENO, .events = POLLOUT, .revents = 0 };
            if (poll(&p, 1, 100) < 0 && errno != EINTR)
            {
                tv_term_hup = 1;
                return -1;
            }
            if (tv_sig_quit || tv_sig_tstp)
            {
                return -1;
            }
            continue;
        }
        tv_term_hup = 1; /* EIO, EPIPE, ... */
        return -1;
    }
    return 0;
}

/**
 * tv_term_color_level - Detect the terminal's ANSI color capability from environment variables.
 *
 * Inspects `COLORTERM` and `TERM` to classify the terminal into 24-bit
 * truecolor (3), 256-color palette (2), or 16-color basic ANSI (1),
 * recognizing modern direct-color emulators even over SSH where `COLORTERM`
 * is often not forwarded.
 *
 * Return: 3 for 24-bit truecolor, 2 for 256-color, or 1 for 16-color.
 */
int tv_term_color_level(void)
{
    /* ssh forwards TERM but not COLORTERM: known 24-bit terminals are recognised by name, and
     * xterm-like emulators have the 256-colour palette unless TERM says fewer */
    static const char *const direct[]  = { "-direct", "kitty",   "alacritty", "foot",
                                           "wezterm", "ghostty", "contour" };
    static const char *const fewer[]   = { "16color", "88color", "mono" };
    static const char *const xterms[]  = { "xterm", "screen", "tmux",  "rxvt",  "konsole",
                                           "gnome", "vte",    "putty", "iterm", "mlterm" };
    const char              *colorterm = getenv("COLORTERM");
    const char              *term      = getenv("TERM");
    if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
    {
        return 3;
    }
    if (term == NULL)
    {
        return 1;
    }
    for (size_t k = 0; k < sizeof(direct) / sizeof(direct[0]); k++)
    {
        if (strstr(term, direct[k]))
        {
            return 3;
        }
    }
    if (strstr(term, "256color"))
    {
        return 2;
    }
    /* 88 colours (rxvt-unicode's default) have no 16..255 cube either */
    if (strcmp(term, "rxvt-unicode") == 0)
    {
        return 1;
    }
    for (size_t k = 0; k < sizeof(fewer) / sizeof(fewer[0]); k++)
    {
        if (strstr(term, fewer[k]))
        {
            return 1;
        }
    }
    for (size_t k = 0; k < sizeof(xterms) / sizeof(xterms[0]); k++)
    {
        if (strncasecmp(term, xterms[k], strlen(xterms[k])) == 0)
        {
            return 2;
        }
    }
    return (strcmp(term, "st") == 0 || strncmp(term, "st-", 3) == 0) ? 2 : 1;
}

/**
 * tv_term_reraise - Reset a signal handler to default and re-raise the signal.
 * @sig: Signal number to re-raise.
 *
 * Restores `SIG_DFL` for @sig and raises it so the calling process exits
 * with the authentic wait status for that signal.
 *
 * Return: Nothing.
 */
void tv_term_reraise(int sig)
{
    signal(sig, SIG_DFL);
    raise(sig);
}
