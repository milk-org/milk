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
 * crash_handler - restore the terminal on a fatal signal, then re-raise it with the default action.
 * @sig: Signal number
 */
static void crash_handler(int sig)
{
    ov_term_restore_async();
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
    OV_OPTS opts;
    int     opt_rc = milkctrl_parse_options(argc, argv, &opts);
    if (opt_rc > 0)
    {
        return 0;
    }
    if (opt_rc < 0)
    {
        return 2;
    }
    if (opts.shm_dir != NULL)
    {
        setenv("MILK_SHM_DIR", opts.shm_dir, 1);
        setenv("MILK_PROC_DIR", opts.shm_dir, 1);
    }
    FILE *logf = NULL;
    if (opts.log != NULL && (logf = fopen(opts.log, "a")) == NULL)
    {
        fprintf(stderr, "%s: --log %s: %s\n", argv[0], opts.log, strerror(errno));
        return 1;
    }
    ov_cmdlog_mirror(logf, opts.debug);

    /* --- Require interactive terminal --- */
    if (!isatty(STDIN_FILENO))
    {
        fprintf(stderr, "%s: interactive terminal required on stdin.\n", argv[0]);
        return 1;
    }
    int utf8_locale = ov_term_init_locale();

    /* --- Install signal handlers --- */
    {
        struct sigaction sa;
        sa.sa_handler = handle_sigint;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);

        sa.sa_handler = handle_sigterm;
        sigaction(SIGTERM, &sa, NULL);
        sigaction(SIGHUP, &sa, NULL);

        static const int fatal_sigs[] = {
            SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL, SIGQUIT, SIGPIPE
        };
        sa.sa_handler = crash_handler;
        for (size_t i = 0; i < sizeof(fatal_sigs) / sizeof(fatal_sigs[0]); i++)
        {
            sigaction(fatal_sigs[i], &sa, NULL);
        }
    }

    /* --- Detect color level --- */
    ov_detect_color_level();

    /* --- Initialize color theme --- */
    int theme_rc = ov_theme_init(opts.theme);

    /* --- Resolve directories while the terminal is still cooked (resolvers may print) --- */
    ov_dirs_init();

    /* --- Enter raw mode --- */
    ov_raw_mode_enter();

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
    lay.view                    = OV_VIEW_DASHBOARD;
    lay.focus                   = OV_FOCUS_STREAMS;
    lay.cmdlog_rows             = 4;
    lay.sub[OV_SUB_DETAILS].sel = -1;
    lay.fps_split_ratio         = 0.4f;
    lay.fps_split_dragging      = 0;
    lay.dash_split_v_ratio      = 0.5f;
    lay.dash_split_h_ratio      = 0.5f;
    lay.dash_split_v_dragging   = 0;
    lay.dash_split_h_dragging   = 0;

    if (opts.view >= 0)
    {
        ov_set_view(&lay, (ov_view_t) opts.view);
    }
    if (opts.pattern != NULL) /* the view's list; DASH, CONN and LOOPS filter streams */
    {
        int             f  = ov_views[lay.view].default_focus;
        OV_PANEL_STATE *ps = &lay.panel[(f >= 0 && f < OV_NB_LIST_PANELS) ? f : OV_FOCUS_STREAMS];
        snprintf(ps->filter, sizeof(ps->filter), "%s", opts.pattern);
        ps->filter_on     = 1;
        lay.filter_active = 1;
    }
    if (opts.interval > 0)
    {
        ov_scan_set_interval((float) opts.interval);
    }

    ov_cmdlog_push(&lay.cmdlog, OV_CMDLOG_INFO, "Shared memory directory: %s", ov_get_shmdir());
    ov_cmdlog_push(&lay.cmdlog, OV_CMDLOG_INFO,
                   "h help \xc2\xb7 F2\xe2\x80\x93"
                   "F7 views \xc2\xb7 / filter \xc2\xb7 c control \xc2\xb7 q quit");
    if (strcmp(ov_get_procdir(), ov_get_shmdir()) != 0)
    {
        ov_cmdlog_push(&lay.cmdlog, OV_CMDLOG_WARN,
                       "Process directory: %s (differs from shm directory)", ov_get_procdir());
    }
    if (!utf8_locale)
    {
        ov_cmdlog_push(&lay.cmdlog, OV_CMDLOG_WARN,
                       "Locale is not UTF-8: glyphs may be garbled; widths use C.UTF-8");
    }
    if (theme_rc < 0)
    {
        ov_cmdlog_push(&lay.cmdlog, OV_CMDLOG_WARN,
                       "MILK_CTRL_THEME=%s is not a theme, using %s (milk-CTRL -h lists them)",
                       getenv("MILK_CTRL_THEME"), ov_theme_get(ov_theme_get_active_index())->id);
    }

    /* --- Main TUI loop (~10 fps) --- */
    /* Clear screen once on startup */
    {
        const char cls[] = "\033[2J\033[H";
        if (write(ov__tty_fd, cls, sizeof(cls) - 1) < 0)
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
                if (write(ov__tty_fd, cls, sizeof(cls) - 1) < 0)
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
                if (m != NULL)
                {
                    ov_cmdlog_debug("scan %.1f ms: %d streams, %d processes, %d FPS, %u tmux polls",
                                    m->scan_time_ms, m->nb_streams, m->nb_procs, m->nb_fps,
                                    m->tmux_polls);
                }
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
                ov_cmdlog_debug("key %d", key);
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

        /* Tell the scan thread what is on screen: pause stops scanning, tmux only for FPS panels */
        ov_scan_set_hints((lay.paused ? OV_HINT_PAUSED : 0U) |
                          (ov_view_has_panel(lay.view, OV_FOCUS_FPS) ? OV_HINT_FPS_VISIBLE : 0U));

        /* Library output captured from fds 1/2 goes to the command log */
        fflush(stdout);
        if (ov_cmdlog_drain_fd(&lay.cmdlog, ov_term_capture_fd()) > 0)
        {
            need_render = 1;
        }

        /* Check if theme selector popup timed out (1s inactivity) */
        if (lay.theme_popup.active)
        {
            struct timespec now_ts;
            clock_gettime(CLOCK_MONOTONIC, &now_ts);
            double elapsed = (now_ts.tv_sec - lay.theme_popup.ts.tv_sec) +
                             (now_ts.tv_nsec - lay.theme_popup.ts.tv_nsec) * 1e-9;
            if (elapsed >= 1.0)
            {
                lay.theme_popup.active = 0;
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
            ov_layout_compute(&lay); /* keys may have switched view: draw with its rectangles */
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

    /* --- Cleanup: restore the tty first; a second Ctrl-C then kills a stuck scan join --- */
    ov_raw_mode_exit();
    signal(SIGINT, SIG_DFL);
    ov_scan_stop();
    pid_perf_close();
    ov_input_history_free(&lay);
    ov_cmdlog_mirror(NULL, 0);
    if (logf != NULL)
    {
        fclose(logf);
    }

    return 0;
}
