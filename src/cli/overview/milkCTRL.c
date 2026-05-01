/**
 * @file milkCTRL.c
 * @brief Main entry point for milkCTRL TUI
 *
 * Standalone binary providing a unified dashboard of all
 * milk shared-memory components (streams, FPS, processes)
 * and their connections.
 *
 * Links: ImageStreamIO + milkprocessinfo + milkfps
 *        + m + rt + pthread
 * No CLIcore dependency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <poll.h>

#include "overview_defs.h"
#include "overview_ansi.h"
#include "overview_theme.h"
#include "overview_data.h"
#include "overview_layout.h"

/* =========================================================
 * Global state (defined here, declared extern elsewhere)
 * ========================================================= */

volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;

struct termios ov__orig_termios;
int            ov__raw_active  = 0;
int            ov__color_level = 0;

/* mouse event coordinates (set by ov_get_key) */
int ov_mouse_row = 0;
int ov_mouse_col = 0;
int ov_mouse_btn = 0;

char ov__screenbuf[OV_SCREENBUF_SIZE];
int  ov__screenbuf_len = 0;

/* =========================================================
 * Signal handlers
 * ========================================================= */

static void handle_sigint(int sig)
{
    (void) sig;
    ov_sigINT = 1;
}

static void handle_sigterm(int sig)
{
    (void) sig;
    ov_sigTERM = 1;
}

static void crash_handler(int sig)
{
    static const char reset[] =
        "\033[?1049l\033[?25h\033[0m\n";
    if (write(STDERR_FILENO,
              reset, sizeof(reset) - 1) < 0) {}
    if (ov__raw_active)
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH,
                  &ov__orig_termios);
    }
    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(sig, &sa, NULL);
    raise(sig);
}

/* =========================================================
 * External API from overview_scan.c
 * ========================================================= */

extern int             ov_scan_start(void);
extern void            ov_scan_stop(void);
extern const OV_MODEL *ov_scan_get_model(void);

/* External API from overview_render.c */
extern void ov_render_frame(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);

/* External API from overview_input.c */
extern int ov_handle_key(
    int              key,
    OV_LAYOUT       *lay,
    const OV_MODEL  *m);


#define MILKCTRL_VERSION "1.0.1"

/* =========================================================
 * Usage / help
 * ========================================================= */

static void print_usage(const char *prog)
{
    printf("milkCTRL version %s\n\n", MILKCTRL_VERSION);
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("  milkCTRL: unified system dashboard\n");
    printf("  Shows streams, FPS entries, processes,\n");
    printf("  and their connections in a single TUI.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help     Show this help\n");
    printf("  -h1            One-line description\n");
    printf("  -d DIR         Override SHM directory\n");
    printf("\n");
    printf("Keys:\n");
    printf("  F2-F6, ^Left/^Right Switch views\n"
           "         (dashboard/graph/streams/procs/fps)\n");
    printf("  TAB    Cycle panel focus\n");
    printf("  UP/DN  Navigate\n");
    printf("  +/-    Adjust scan rate\n");
    printf("  h      Help overlay\n");
    printf("  q/x    Exit\n");
}


/* =========================================================
 * main
 * ========================================================= */

int main(int argc, char *argv[])
{
    /* --- Pre-getopt: handle -h1 ---*/
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h1") == 0
            || strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("unified system dashboard TUI (milkCTRL) "
                   "for streams, FPS, and processes\n");
            return 0;
        }
        if (strcmp(argv[i], "-h") == 0
            || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-d") == 0
            && (i + 1 < argc))
        {
            setenv("MILK_SHM_DIR", argv[++i], 1);
        }
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
        sigaction(SIGBUS,  &sa, NULL);
        sigaction(SIGABRT, &sa, NULL);
    }

    /* --- Detect color level --- */
    ov_detect_color_level();

    /* --- Enter raw mode --- */
    ov_raw_mode_enter();

    /* --- Start background scanner --- */
    if (ov_scan_start() != 0)
    {
        ov_raw_mode_exit();
        fprintf(stderr,
                "ERROR: failed to start scan thread\n");
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
    lay.view  = OV_VIEW_DASHBOARD;
    lay.focus = OV_FOCUS_STREAMS;

    /* --- Main TUI loop (~10 fps) --- */
    struct timespec frame_ts;
    frame_ts.tv_sec  = 0;
    frame_ts.tv_nsec = 100000000L; /* 100 ms */

    /* Clear screen once on startup */
    {
        const char cls[] = "\033[2J\033[H";
        if (write(STDOUT_FILENO,
                  cls, sizeof(cls) - 1) < 0) {}
    }

    int last_rows = -1;
    int last_cols = -1;

    while (!OV_SIG_ANY_SET())
    {
        /* Recompute layout (handles resize) */
        ov_layout_compute(&lay);

        if (lay.term_rows != last_rows || lay.term_cols != last_cols)
        {
            /* Size changed, force clear to remove layout ghosts */
            const char cls[] = "\033[2J\033[H";
            if (write(STDOUT_FILENO, cls, sizeof(cls) - 1) < 0) {}
            last_rows = lay.term_rows;
            last_cols = lay.term_cols;
        }

        /* Get current model snapshot */
        static const OV_MODEL *last_m = NULL;
        const OV_MODEL *m = last_m;
        if (!lay.paused || m == NULL)
        {
            m = ov_scan_get_model();
            last_m = m;
        }
        /* Drain all pending input for snappy response */
        {
            int quit = 0;
            int key;
            while ((key = ov_get_key()) != OV_KEY_NONE)
            {

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

        /* Render */
        ov_render_frame(&lay, m);

        /* Frame delay: poll stdin for fast response or wait for scan */
        {
            struct pollfd pfd;
            pfd.fd = STDIN_FILENO;
            pfd.events = POLLIN;
            int quit_now = 0;

            /* Check up to 10 times with 10ms timeout = 100ms max */
            for (int i = 0; i < 10; i++)
            {
                if (poll(&pfd, 1, 10) > 0)
                {
                    if (pfd.revents & POLLIN)
                    {
                        /* Read key and handle it now to
                         * avoid losing the keypress. */
                        int pk = ov_get_key();
                        if (pk == 'q' || pk == 'x')
                        {
                            quit_now = 1;
                        }
                        else if (pk != OV_KEY_NONE)
                        {
                            ov_handle_key(pk, &lay, m);
                        }
                        break;
                    }
                }
                if (ov_scan_has_new_data())
                {
                    /* New background scan arrived, wake up */
                    break;
                }
            }
            if (quit_now)
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
