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


#include "milk_help.h"

#define MILKCTRL_VERSION "2.0.0"

/* =========================================================
 * Usage / help
 * ========================================================= */

static void print_help(const char *prog, int mh_color)
{
    milk_help_banner(prog, "unified system dashboard TUI (milkCTRL) for streams, FPS, and processes", mh_color);

    milk_help_section("Usage", mh_color);
    printf("  $ %s [%s %s]\n\n", prog, MH(MH_OPT, "-d"), MH(MH_ARG, "DIR"));

    milk_help_section("Description", mh_color);
    printf("  milkCTRL is the unified system dashboard TUI for the milk framework.\n"
           "  It provides a consolidated view of all shared-memory components,\n"
           "  including data streams, Function Processing Systems (FPS), and\n"
           "  active processes.\n"
           "\n"
           "  Key capabilities include:\n"
           "  - %s: Visualizes dataflow lineage between streams\n"
           "    and the processes reading/writing them.\n"
           "  - %s: Monitors cache misses (L1D, LLC, dTLB) per\n"
           "    process iteration using Linux perf counters.\n"
           "  - %s: Control execution states (pause, step, kill)\n"
           "    and clean up zombie streams or stale memory segments safely.\n"
           "  - %s: Real-time metrics on process CPU usage, memory\n"
           "    (RSS), context switching, and pipeline throughput.\n\n",
           MH(MH_BOLD, "Live Pipeline Topology"),
           MH(MH_BOLD, "Hardware Diagnostics"),
           MH(MH_BOLD, "Process Orchestration"),
           MH(MH_BOLD, "System Telemetry"));

    milk_help_section("Options", mh_color);
    printf("  %s                      Show this help\n", MH(MH_OPT, "-h, --help"));
    printf("  %s             One-line description\n", MH(MH_OPT, "-h1, --help-oneline"));
    printf("  %s                Full help, forced monochrome\n", MH(MH_OPT, "-hm, --help-mono"));
    printf("  %s %s                         Override SHM directory\n\n", MH(MH_OPT, "-d"), MH(MH_ARG, "DIR"));

    milk_help_section("Navigation", mh_color);
    printf("  %s, %s             Switch views\n", MH(MH_OPT, "F2-F6"), MH(MH_OPT, "^Left/^Right"));
    printf("  %s                             Cycle panel focus\n", MH(MH_OPT, "TAB"));
    printf("  %s                           Navigate list\n", MH(MH_OPT, "UP/DN"));
    printf("  %s                      Panel focus / scroll\n", MH(MH_OPT, "Left/Right"));
    printf("  %s                         Scroll page\n", MH(MH_OPT, "PgUp/Dn"));
    printf("  %s                        Jump to top/bottom\n\n", MH(MH_OPT, "Home/End"));

    milk_help_section("Sorting", mh_color);
    printf("  %s, %s                        Change sort column / mode\n", MH(MH_OPT, "</>"), MH(MH_OPT, "S/s"));
    printf("  %s                               Toggle sort direction\n\n", MH(MH_OPT, "["));

    milk_help_section("Display", mh_color);
    printf("  %s                             Adjust scan rate\n", MH(MH_OPT, "+/-"));
    printf("  %s                               Toggle detail pane\n", MH(MH_OPT, "D"));
    printf("  %s                               Toggle lineage mode\n", MH(MH_OPT, "L"));
    printf("  %s                               Pause/resume display\n", MH(MH_OPT, "p"));
    printf("  %s                           Freeze selection highlight\n", MH(MH_OPT, "SPACE"));
    printf("  %s                               Filter (regex search)\n", MH(MH_OPT, "/"));
    printf("  %s                               Export snapshot to file\n", MH(MH_OPT, "W"));
    printf("  %s                               Toggle command log panel\n", MH(MH_OPT, "G"));
    printf("  %s                               Help overlay\n\n", MH(MH_OPT, "h"));

    milk_help_section("Control mode (c to toggle)", mh_color);
    printf("  FPS:   %s=run  %s=conf  %s=remove\n", MH(MH_OPT, "r"), MH(MH_OPT, "s"), MH(MH_OPT, "e"));
    printf("  PROCS: %s=remove selected entry\n", MH(MH_OPT, "e"));
    printf("  STRM:  %s=delete stream\n\n", MH(MH_OPT, "d"));

    milk_help_section("Process signals (PROCS & FPS panels)", mh_color);
    printf("  %s                               Graceful kill (SIGTERM)\n", MH(MH_OPT, "k"));
    printf("  %s                               Immediate kill (SIGKILL)\n", MH(MH_OPT, "K"));
    printf("  %s                               Pause/resume (SIGSTOP/SIGCONT)\n", MH(MH_OPT, "x"));
    printf("  %s                               Cleanup dead/stopped procs (PROCS panel)\n\n", MH(MH_OPT, "C"));

    milk_help_section("Detail View (ENTER or D) and Inspection (i)", mh_color);
    printf("  %s                         Toggles detail pane for selected item\n", MH(MH_OPT, "ENTER/D"));
    printf("  %s                               Spawn interactive CLI diagnostic tool\n\n", MH(MH_OPT, "i"));

    milk_help_section("Columns", mh_color);
    printf("  STREAMS: MB/s throughput, total in panel footer\n");
    printf("  PROCS:   DUTY%% (exec/iter), CPU%%, MEM (RSS)\n");
    printf("  FPS:     MEM (RSS)\n\n");

    milk_help_section("Mouse", mh_color);
    printf("  Click=select  DblClick=detail\n");
    printf("  Scroll wheel=navigate list\n\n");
    printf("  %s                               Quit\n", MH(MH_OPT, "q"));
}


/* =========================================================
 * main
 * ========================================================= */

int main(int argc, char *argv[])
{
    /* --- Help handling --- */
    int action = milk_help_init(argc, argv,
        "unified system dashboard TUI (milkCTRL) for streams, FPS, and processes",
        NULL);

    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);

    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    /* --- Custom options parsing ---*/
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-d") == 0 && (i + 1 < argc))
        {
            setenv("MILK_SHM_DIR", argv[++i], 1);
        }
        else if (argv[i][0] == '-')
        {
            fprintf(stderr, "%s Invalid option: %s%s%s\n"
                    "Run %s%s%s %s for usage.\n",
                    MH(MH_ERR, "Error:"),
                    mh_color ? MH_OPT : "", argv[i], mh_color ? MH_RST : "",
                    mh_color ? MH_CMD : "", argv[0], mh_color ? MH_RST : "",
                    MH(MH_OPT, "-h"));
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
