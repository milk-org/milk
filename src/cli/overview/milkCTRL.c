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

OV_CELL  ov__shadow[OV_MAX_ROWS][OV_MAX_COLS];
OV_CELL  ov__front[OV_MAX_ROWS][OV_MAX_COLS];
int      ov__cursor_row = 1;
int      ov__cursor_col = 1;
uint32_t ov__current_fg = OV_COLOR_NONE;
uint32_t ov__current_bg = OV_COLOR_NONE;
uint32_t ov__current_ul = OV_COLOR_NONE;
uint8_t  ov__current_attr = 0;

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


#define MILKCTRL_VERSION "2.0.0"

/* =========================================================
 * Usage / help
 * ========================================================= */

static void print_usage(const char *prog)
{
    printf("milkCTRL version %s\n\n", MILKCTRL_VERSION);
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("  Unified system dashboard TUI.\n");
    printf("  Shows streams, FPS entries, processes,\n");
    printf("  and their connections in a single view.\n");
    printf("\nOptions:\n");
    printf("  -h, --help     Show this help\n");
    printf("  -h1            One-line description\n");
    printf("  -d DIR         Override SHM directory\n");
    printf("\nNavigation:\n");
    printf("  F2-F6, ^Left/^Right  Switch views\n");
    printf("  TAB        Cycle panel focus\n");
    printf("  UP/DN      Navigate list\n");
    printf("  Left/Right Panel focus / scroll\n");
    printf("  PgUp/Dn    Scroll page\n");
    printf("  Home/End   Jump to top/bottom\n");
    printf("\nSorting:\n");
    printf("  </>, S/s   Change sort column / mode\n");
    printf("  [          Toggle sort direction\n");
    printf("\nDisplay:\n");
    printf("  +/-        Adjust scan rate\n");
    printf("  D          Toggle detail pane\n");
    printf("  L          Toggle lineage mode\n");
    printf("  p          Pause/resume display\n");
    printf("  SPACE      Freeze selection highlight\n");
    printf("  /          Filter (regex search)\n");
    printf("  W          Export snapshot to file\n");
    printf("  G          Toggle command log panel\n");
    printf("  h          Help overlay\n");
    printf("\nControl mode (c to toggle):\n");
    printf("  FPS:   r=run  s=conf  e=remove\n");
    printf("  PROCS: e=remove selected entry\n");
    printf("  STRM:  d=delete stream\n");
    printf("\nProcess signals (PROCS & FPS panels):\n");
    printf("  k    Graceful kill (SIGTERM)\n");
    printf("  K    Immediate kill (SIGKILL)\n");
    printf("  x    Pause/resume (SIGSTOP/SIGCONT)\n");
    printf("  C    Cleanup dead/stopped procs (PROCS panel)\n");
    printf("\nDetail View (ENTER or D) and Inspection (i):\n");
    printf("  ENTER/D  Toggles detail pane for selected item\n");
    printf("  i        Spawn interactive CLI diagnostic tool\n");
    printf("\nColumns:\n");
    printf("  STREAMS: MB/s throughput,"
           " total in panel footer\n");
    printf("  PROCS:   DUTY%% (exec/iter),"
           " CPU%%, MEM (RSS)\n");
    printf("  FPS:     MEM (RSS)\n");
    printf("\nMouse:\n");
    printf("  Click=select  DblClick=detail\n");
    printf("  Scroll wheel=navigate list\n");
    printf("\n  q    Quit\n");
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
        else if (strcmp(argv[i], "-d") == 0
            && (i + 1 < argc))
        {
            setenv("MILK_SHM_DIR", argv[++i], 1);
        }
        else if (argv[i][0] == '-')
        {
            printf("\n\033[1;31mERROR\033[0m: Invalid option: %s\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
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
    lay.cmdlog_rows = 4;

    /* --- Main TUI loop (~10 fps) --- */
    /* Clear screen once on startup */
    {
        const char cls[] = "\033[2J\033[H";
        if (write(STDOUT_FILENO,
                  cls, sizeof(cls) - 1) < 0) {}
    }

    int last_rows = -1;
    int last_cols = -1;
    const OV_MODEL *m = NULL;
    int need_render = 1; /* force first frame */

    while (!OV_SIG_ANY_SET())
    {
        /* Recompute layout (handles resize) */
        ov_layout_compute(&lay);

        if (lay.term_rows != last_rows
            || lay.term_cols != last_cols)
        {
            /* Size changed, force clear */
            const char cls[] = "\033[2J\033[H";
            if (write(STDOUT_FILENO,
                      cls, sizeof(cls) - 1) < 0) {}
            ov_buf_force_clear();
            last_rows = lay.term_rows;
            last_cols = lay.term_cols;
            need_render = 1;
        }

        /* Pick up new model if available */
        if (!lay.paused || m == NULL)
        {
            const OV_MODEL *prev = m;
            m = ov_scan_get_model();
            if (m != prev)
            {
                need_render = 1;
            }
        }

        /* Drain all pending input */
        {
            int quit = 0;
            int key;
            while ((key = ov_get_key())
                   != OV_KEY_NONE)
            {
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

        /* Render only when something changed */
        if (need_render)
        {
            ov_render_frame(&lay, m);
            need_render = 0;
        }

        /* Frame delay: poll stdin, wake on
         * new data or keypress */
        {
            struct pollfd pfd;
            pfd.fd = STDIN_FILENO;
            pfd.events = POLLIN;
            int quit_now = 0;

            for (int i = 0; i < 10; i++)
            {
                if (poll(&pfd, 1, 10) > 0)
                {
                    if (pfd.revents & POLLIN)
                    {
                        int pk = ov_get_key();
                        if (pk == 'q')
                        {
                            quit_now = 1;
                        }
                        else if (pk
                                 != OV_KEY_NONE)
                        {
                            ov_handle_key(
                                pk, &lay, m);
                            need_render = 1;
                        }
                        break;
                    }
                }
                if (ov_scan_has_new_data())
                {
                    need_render = 1;
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
