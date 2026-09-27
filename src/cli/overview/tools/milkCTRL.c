// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkCTRL.c
 * @brief Main entry point for milk-CTRL TUI dashboard
 *
 * Standalone binary providing a unified dashboard of all
 * milk shared-memory components (streams, FPS, processes)
 * and their connections.
 *
 * Links: ImageStreamIO + milkprocessinfo + milkfps
 *        + m + rt + pthread
 * No CLIcore dependency.
 */

#include "milkCTRL_opts.h"
#include "milk_config.h"
#include "overview_ansi.h"
#include "overview_data.h"
#include "overview_defs.h"
#include "overview_layout.h"
#include "overview_theme.h"
#include "processinfo_shm_list_create.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* =========================================================
 * Global state (defined here, declared extern elsewhere)
 * ========================================================= */

volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;
/* Global ANSI and terminal state is owned and defined in overview_ansi.c */

/* =========================================================
 * Signal handlers
 * ========================================================= */

/**
 * handle_sigint - Signal handler for SIGINT (Ctrl-C)
 * @sig: Signal number
 */
static void handle_sigint(int sig)
{
    (void) sig;
    ov_sigINT = 1;
}

/**
 * handle_sigterm - SIGTERM handler for milkCTRL exit
 * @sig: Signal number
 */
static void handle_sigterm(int sig)
{
    (void) sig;
    ov_sigTERM = 1;
}

/**
 * crash_handler - Crash signal handler for milkCTRL
 * @sig: Signal number
 *
 * Captures SIGSEGV/SIGABRT, restores terminal,
 * and prints a diagnostic message.
 */
static void crash_handler(int sig)
{
    static const char reset[] = "\033[?1049l\033[?25h\033[0m\n";
    if (write(STDERR_FILENO, reset, sizeof(reset) - 1) < 0)
    {
    }
    if (ov__raw_active)
    {
        if (ov__orig_flags >= 0)
        {
            fcntl(STDIN_FILENO, F_SETFL, ov__orig_flags);
        }
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &ov__orig_termios);
    }
    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(sig, &sa, NULL);
    raise(sig);
}

/* =========================================================
 * External API declarations
 * ========================================================= */

extern int             ov_scan_start(void);
extern void            ov_scan_stop(void);
extern const OV_MODEL *ov_scan_get_model(void);
extern void            ov_render_frame(const OV_LAYOUT *lay, const OV_MODEL *m);
extern void            ov_render__sync_selection(OV_LAYOUT *lay, const OV_MODEL *m);
extern int             ov_handle_key(int key, OV_LAYOUT *lay, const OV_MODEL *m);

/* =========================================================
 * main
 * ========================================================= */

/**
 * main - Entry point for milk-CTRL standalone TUI dashboard
 * @argc: Command-line argument count
 * @argv: Command-line argument vector
 *
 * Return: 0 on clean exit, non-zero on error.
 */
int main(int argc, char *argv[])
{
    const char *cli_theme = NULL;
    int         opt_rc    = milkctrl_parse_options(argc, argv, &cli_theme);
    if (opt_rc > 0)
    {
        return 0;
    }
    if (opt_rc < 0)
    {
        return 1;
    }

    /* --- Require interactive terminal --- */
    if (!isatty(STDIN_FILENO))
    {
        fprintf(stderr, "%s: interactive terminal required on stdin.\n", argv[0]);
        return 1;
    }

    /* --- Install signal handlers --- */
    {
        struct sigaction sa;
        sa.sa_handler = handle_sigint;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);

        sa.sa_handler = handle_sigterm;
        sigaction(SIGTERM, &sa, NULL);

        sa.sa_handler = crash_handler;
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGBUS, &sa, NULL);
        sigaction(SIGABRT, &sa, NULL);
    }

    /* --- Detect color level --- */
    ov_detect_color_level();

    /* --- Initialize color theme --- */
    ov_theme_init(cli_theme);

    /* --- Enter raw mode --- */
    ov_raw_mode_enter();

    /* --- Connect to process list shared memory --- */
    {
        long pindex_unused;
        if (processinfo_shm_list_create(&pindex_unused) != RETURN_SUCCESS)
        {
            ov_raw_mode_exit();
            PRINT_ERROR("failed to connect to process list shared memory");
            return 1;
        }
    }

    /* --- Start background scanner --- */
    if (ov_scan_start() != 0)
    {
        ov_raw_mode_exit();
        PRINT_ERROR("failed to start scan thread");
        return 1;
    }

    /* --- Wait for first scan to complete to avoid blank startup --- */
    {
        struct timespec wts;
        wts.tv_sec  = 0;
        wts.tv_nsec = 10000000L; /* 10 ms */
        int w_iters = 0;
        while (!OV_SIG_ANY_SET() && w_iters < 100) /* max 1 sec wait */
        {
            if (ov_scan_has_new_data())
            {
                break;
            }
            nanosleep(&wts, NULL);
            w_iters++;
        }
    }

    /* --- Initialize layout --- */
    OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    lay.view                  = OV_VIEW_DASHBOARD;
    lay.focus                 = OV_FOCUS_STREAMS;
    lay.cmdlog_rows           = 4;
    lay.param_sel             = -1;
    lay.fps_split_ratio       = 0.4f;
    lay.fps_split_dragging    = 0;
    lay.dash_split_v_ratio    = 0.5f;
    lay.dash_split_h_ratio    = 0.5f;
    lay.dash_split_v_dragging = 0;
    lay.dash_split_h_dragging = 0;

    ov_cmdlog_push(&lay.cmdlog, OV_CMDLOG_INFO, "Shared memory directory: %s", ov_get_shmdir());

    /* --- Main TUI loop (~10 fps) --- */
    /* Clear screen once on startup */
    {
        const char cls[] = "\033[2J\033[H";
        if (write(STDOUT_FILENO, cls, sizeof(cls) - 1) < 0)
        {
        }
    }

    int             last_rows        = -1;
    int             last_cols        = -1;
    int             last_cmdlog_rows = lay.cmdlog_rows;
    const OV_MODEL *m                = NULL;
    int             need_render      = 1; /* force first frame */

    while (!OV_SIG_ANY_SET())
    {
        /* Recompute layout (handles resize) */
        ov_layout_compute(&lay);

        if (lay.term_rows != last_rows || lay.term_cols != last_cols ||
            lay.cmdlog_rows != last_cmdlog_rows)
        {
            if (lay.term_rows != last_rows || lay.term_cols != last_cols)
            {
                /* Size changed, force clear */
                const char cls[] = "\033[2J\033[H";
                if (write(STDOUT_FILENO, cls, sizeof(cls) - 1) < 0)
                {
                }
            }
            ov_buf_force_clear();
            last_rows        = lay.term_rows;
            last_cols        = lay.term_cols;
            last_cmdlog_rows = lay.cmdlog_rows;
            need_render      = 1;
        }

        /* Pick up new model if available */
        if (!lay.paused || m == NULL)
        {
            const OV_MODEL *prev = m;
            m                    = ov_scan_get_model();
            if (m != prev)
            {
                ov_render__sync_selection(&lay, m);
                need_render = 1;
            }
        }
        else
        {
            /* Drain eventfd to prevent waking up poll loop while paused */
            ov_scan_drain_event_fd();
        }

        /* Drain all pending input */
        {
            int quit = 0;
            int key;
            while ((key = ov_get_key()) != OV_KEY_NONE)
            {
                if (key == OV_KEY_EOF)
                {
                    quit = 1;
                    break;
                }
                need_render = 1;
                if (ov_handle_key(key, &lay, m))
                {
                    quit = 1;
                    break;
                }
            }
            if (quit)
            {
                break;
            }
        }

        /* Check if theme selector popup timed out (1s inactivity) */
        if (lay.theme_popup_active)
        {
            struct timespec now_ts;
            clock_gettime(CLOCK_MONOTONIC, &now_ts);
            double elapsed = (now_ts.tv_sec - lay.theme_popup_ts.tv_sec) +
                             (now_ts.tv_nsec - lay.theme_popup_ts.tv_nsec) * 1e-9;
            if (elapsed >= 1.0)
            {
                lay.theme_popup_active = 0;
                need_render            = 1;
            }
            else
            {
                /* Force frame redraw to update auto-close countdown in popup */
                need_render = 1;
            }
        }

        /* Render only when something changed */
        if (need_render)
        {
            ov_render_frame(&lay, m);
            need_render = 0;
        }

        /* Frame delay: poll stdin and scan eventfd, wake on
         * new data, keypress, or 100ms timeout (~10 Hz frame tick) */
        {
            struct pollfd pfds[2];
            int           npfd = 1;

            pfds[0].fd     = STDIN_FILENO;
            pfds[0].events = POLLIN;

            int scan_efd = ov_scan_get_event_fd();
            if (scan_efd >= 0 && !lay.paused)
            {
                pfds[1].fd     = scan_efd;
                pfds[1].events = POLLIN;
                npfd           = 2;
            }

            int pr = poll(pfds, npfd, 100);
            if (pr > 0)
            {
                if (pfds[0].revents & (POLLHUP | POLLERR | POLLNVAL))
                {
                    break;
                }
                if (npfd > 1 && (pfds[1].revents & POLLIN))
                {
                    ov_scan_drain_event_fd();
                }
            }
            else if (pr < 0 && errno != EINTR)
            {
                break;
            }
        }
    }

    /* --- Cleanup --- */
    ov_scan_stop();
    ov_raw_mode_exit();

    return 0;
}
