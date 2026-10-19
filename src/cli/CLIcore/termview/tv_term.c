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

static struct termios        tv_tio_orig;
static volatile sig_atomic_t tv_term_active;
static int                   tv_err_fd  = -1; /* the original stderr, for exit and crash messages */
static int                   tv_null_fd = -1; /* /dev/null: stderr while the screen is ours */
static volatile sig_atomic_t tv_err_quiet;
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

static void tv_on_quit(int sig)
{
    tv_sig_quit = sig;
}

static void tv_on_winch(int sig)
{
    (void) sig;
    tv_sig_winch = 1;
}

static void tv_on_cont(int sig)
{
    (void) sig;
    tv_sig_cont = 1;
}

static void tv_on_tstp(int sig)
{
    int              saved = errno;
    struct sigaction sa;

    tv_term_restore();
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTSTP, &sa, NULL);

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGTSTP);
    sigprocmask(SIG_UNBLOCK, &mask, NULL);
    raise(sig); /* stops here until SIGCONT */

    sa.sa_handler = tv_on_tstp;
    sigaction(SIGTSTP, &sa, NULL);
    tv_sig_cont = 1;
    errno       = saved;
}

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

static void tv_set_handler(int sig, void (*fn)(int), int flags)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = fn;
    sa.sa_flags   = flags; /* no SA_RESTART: poll() and write() see EINTR */
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, NULL);
}

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
    tv_set_handler(SIGTSTP, tv_on_tstp, 0);
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

void tv_term_suspend(void)
{
    raise(SIGTSTP); /* tv_on_tstp restores, stops, and flags the resume */
}

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

int tv_write_all(const char *buf, size_t n)
{
    while (n > 0)
    {
        ssize_t w = write(STDOUT_FILENO, buf, n);
        if (w > 0)
        {
            buf += w;
            n -= (size_t) w;
            continue;
        }
        if (w < 0 && errno == EINTR)
        {
            if (tv_sig_quit || tv_sig_cont)
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
            if (tv_sig_quit)
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

void tv_term_reraise(int sig)
{
    signal(sig, SIG_DFL);
    raise(sig);
}
