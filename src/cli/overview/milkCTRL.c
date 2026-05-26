/**
 * @file milk-CTRL.c
 * @brief Main entry point for milk-CTRL TUI
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
#include "processinfo_shm_list_create.h"

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
int ov_hover_row = 0;
int ov_hover_col = 0;

char     ov__screenbuf[OV_SCREENBUF_SIZE];
int      ov__screenbuf_len        = 0;
uint64_t ov__total_bytes_rendered = 0;

OV_CELL  ov__shadow[OV_MAX_ROWS][OV_MAX_COLS];
OV_CELL  ov__front[OV_MAX_ROWS][OV_MAX_COLS];
int      ov__cursor_row   = 1;
int      ov__cursor_col   = 1;
uint32_t ov__current_fg   = OV_COLOR_NONE;
uint32_t ov__current_bg   = OV_COLOR_NONE;
uint32_t ov__current_ul   = OV_COLOR_NONE;
uint8_t  ov__current_attr = 0;

/* =========================================================
 * Signal handlers
 * ========================================================= */

static void handle_sigint(int sig)
{
    (void) sig;
    ov_sigINT = 1;
}

/**
 * @brief SIGTERM handler for milkCTRL exit.
 */
static void handle_sigterm(int sig)
{
    (void) sig;
    ov_sigTERM = 1;
}

/**
 * @brief Crash signal handler for milkCTRL.
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
 * External API from overview_scan.c
 * ========================================================= */

extern int             ov_scan_start(void);
extern void            ov_scan_stop(void);
extern const OV_MODEL *ov_scan_get_model(void);

/* External API from overview_render.c */
extern void ov_render_frame(const OV_LAYOUT *lay, const OV_MODEL *m);

/* External API from overview_input.c */
extern int ov_handle_key(int key, OV_LAYOUT *lay, const OV_MODEL *m);


#include "milk_help.h"

#define MILKCTRL_VERSION "2.0.0"

/* =========================================================
 * Usage / help
 * ========================================================= */

static void print_help(const char *prog, int mh_color)
{
    milk_help_banner(
        prog, "unified system dashboard TUI (milk-CTRL) for streams, FPS, and processes", mh_color);

    milk_help_section("Usage", mh_color);
    printf("  $ %s [%s %s]\n\n", prog, MH(MH_OPT, "-d"), MH(MH_ARG, "DIR"));

    milk_help_section("Description", mh_color);
    printf("  milk-CTRL is the unified real-time dashboard for the milk framework.\n"
           "  It monitors and controls three core pillars of the shared-memory architecture:\n\n"
           "  1. %s (Streams): Zero-copy n-dimensional data passing between processes.\n"
           "     Files are located in /milk/shm/ (override via MILK_SHM_DIR or -d).\n"
           "  2. %s (Function Processing System): Real-time parameter sync and state control\n"
           "     (conf/run loop). Processes run in isolated tmux sessions for fault tolerance.\n"
           "  3. %s (Process Info): Heartbeat telemetry, loop rate, and CPU profiling.\n\n"
           "  Designed for low-latency adaptive optics, operators can trace pipeline topology,\n"
           "  diagnose CPU/dTLB bottlenecks, and orchestrate compute loops dynamically.\n\n",
           MH(MH_BOLD, "ImageStreamIO"), MH(MH_BOLD, "FPS"), MH(MH_BOLD, "processinfo"));

    milk_help_section("Dashboard Layout (F2 - F6)", mh_color);
    printf(
        "  - %s (F2): Grid overview of Streams, Processes, and FPS panels.\n"
        "  - %s (F3): Full-screen Streams panel with detailed dimensions, semaphores, & IO rates.\n"
        "  - %s (F4): Full-screen Process monitor with status (RUN/STOP/CRSH), CPU, & loop "
        "counts.\n"
        "  - %s  (F5): Full-screen FPS list (left) and interactive parameter tree (right).\n"
        "  - %s (F6): Visual dataflow node graph tracing upstream/downstream lineage.\n\n",
        MH(MH_BOLD, "DASH"), MH(MH_BOLD, "STRM"), MH(MH_BOLD, "PROC"), MH(MH_BOLD, "FPS"),
        MH(MH_BOLD, "CONN"));

    milk_help_section("Options", mh_color);
    printf("  %-30s Show this help and exit\n", MH(MH_OPT, "-h, --help"));
    printf("  %-30s One-line description and exit\n", MH(MH_OPT, "-h1, --help-oneline"));
    printf("  %-30s Full help, forced monochrome\n", MH(MH_OPT, "-hm, --help-mono"));
    printf("  %-30s Override SHM/process directory\n\n", MH(MH_OPT, "-d <DIR>"));

    milk_help_section("Navigation & View Controls", mh_color);
    printf("  %-30s Switch active panel focus (Dashboard / FPS view)\n", MH(MH_OPT, "TAB"));
    printf("  %-30s Navigate rows in the currently focused list\n", MH(MH_OPT, "UP / DOWN"));
    printf("  %-30s Scroll page up / down\n", MH(MH_OPT, "PgUp / PgDn"));
    printf("  %-30s Jump to top / bottom of the list\n", MH(MH_OPT, "Home / End"));
    printf("  %-30s Scroll list/table horizontally\n", MH(MH_OPT, "LEFT / RIGHT"));
    printf("  %-30s Toggle detailed inspection pane / parameter edit mode\n", MH(MH_OPT, "ENTER"));
    printf("  %-30s Toggle details tab on selected item / Graph details\n", MH(MH_OPT, "D"));
    printf("  %-30s Filter items in the focused list (regex search)\n", MH(MH_OPT, "/"));
    printf("  %-30s Freeze selection highlight (prevents jumping during updates)\n",
           MH(MH_OPT, "SPACE"));
    printf("  %-30s Export current dashboard state snapshot to file\n", MH(MH_OPT, "W"));
    printf("  %-30s Toggle command log ring-buffer visibility\n", MH(MH_OPT, "G"));
    printf("  %-30s Cycle graph lineage mode on F6 view (Trigger / Input)\n", MH(MH_OPT, "L"));
    printf("  %-30s Pause/resume real-time UI data updates\n", MH(MH_OPT, "F"));
    printf("  %-30s Increase / decrease scan updates speed (interval)\n", MH(MH_OPT, "+ / -"));
    printf("  %-30s Quit milk-CTRL\n\n", MH(MH_OPT, "q / x"));

    milk_help_section("Column Hiding & Layout Management", mh_color);
    printf("  %-30s Move highlighted column cursor backward / forward\n",
           MH(MH_OPT, "SHIFT + LEFT/RIGHT"));
    printf("  %-30s Toggle visibility (hide/show) of the highlighted column\n",
           MH(MH_OPT, "t / T"));
    printf("  %-30s Toggle compact layout mode (hides secondary columns to fit terminal)\n",
           MH(MH_OPT, "d"));
    printf("  %-30s Adjust F5:FPS or F2:Dashboard vertical split panel ratio\n",
           MH(MH_OPT, "{ / }"));
    printf("  %-30s Adjust F2:Dashboard horizontal split panel ratio\n\n", MH(MH_OPT, "( / )"));

    milk_help_section("Sorting", mh_color);
    printf("  %-30s Sort list by Name (alphabetical)\n", MH(MH_OPT, "s"));
    printf("  %-30s Sort list by Frequency (Hz) or process execution status\n", MH(MH_OPT, "S"));
    printf("  %-30s Sort list by Ancestry / pipeline dataflow topology\n", MH(MH_OPT, "A"));
    printf("  %-30s Cycle active sort column backward / forward\n", MH(MH_OPT, "< / > or ]"));
    printf("  %-30s Toggle sort direction (Ascending / Descending)\n\n", MH(MH_OPT, "["));

    milk_help_section("Control Mode Actions (press 'c' to toggle Control Mode ON/OFF)", mh_color);
    printf("  Global:\n"
           "    %-28s Deletes selected stream, FPS config, or process registry entry.\n"
           "                 For processes, this deactivates stale/crashed process slots in\n"
           "                 processinfo.list.shm, removing them from the dashboard.\n\n"
           "  Streams (STRM view):\n"
           "    %-28s Delete stream shared-memory file on disk\n\n"
           "  Processes (PROC view):\n"
           "    %-28s Send SIGTERM signal to process\n"
           "    %-28s Send SIGKILL signal to process\n"
           "    %-28s Toggle Pause/Resume (sends SIGSTOP/SIGCONT to process PID)\n"
           "    %-28s Send Step execution command (CTRLval=2)\n"
           "    %-28s Send Exit execution command (CTRLval=3)\n"
           "    %-28s Reset performance counter metrics to zero\n"
           "    %-28s Perform cleanup / release allocations\n\n"
           "  FPS Modules (FPS view):\n"
           "    %-28s Send SIGTERM to FPS tmux session\n"
           "    %-28s Send SIGKILL to FPS tmux session\n"
           "    %-28s Toggle Run loop state (runstart / runstop)\n"
           "    %-28s Toggle Configuration loop state (confstart / confstop)\n"
           "    %-28s Cycle through run states / pause\n\n",
           MH(MH_OPT, "CTRL + e"), MH(MH_OPT, "DEL / CTRL+e"), MH(MH_OPT, "k"), MH(MH_OPT, "K"),
           MH(MH_OPT, "p"), MH(MH_OPT, "^s"), MH(MH_OPT, "e"), MH(MH_OPT, "z"), MH(MH_OPT, "C"),
           MH(MH_OPT, "k"), MH(MH_OPT, "K"), MH(MH_OPT, "r"), MH(MH_OPT, "s"), MH(MH_OPT, "x"));

    milk_help_section("Mouse Interactions", mh_color);
    printf("  - Click anywhere on a row to select it.\n"
           "  - Double-click a row to open the detailed inspector pane.\n"
           "  - Scroll the mouse wheel to navigate lists vertically.\n"
           "  - Click column headers to sort the table by that column.\n"
           "  - Click dashboard tabs (DASH, STRM, etc.) to switch views.\n"
           "  - Drag panel borders/separators to resize split panels.\n\n");
}


/* =========================================================
 * main
 * ========================================================= */

int main(int argc, char *argv[])
{
    /* --- Help handling --- */
    int action = milk_help_init(
        argc, argv, "unified system dashboard TUI (milk-CTRL) for streams, FPS, and processes",
        "milk-CTRL is the unified, real-time diagnostic and control dashboard for the\n"
        "milk shared-memory micro-service framework. It connects directly to zero-copy\n"
        "ImageStreamIO streams, maps the FPS parameters, and tracks managed processes.\n"
        "Operators can trace dataflow lineage, configure running nodes, diagnose CPU\n"
        "and hardware bottlenecks, and manage execution loops interactively.");

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
            fprintf(stderr,
                    "%s Invalid option: %s%s%s\n"
                    "Run %s%s%s %s for usage.\n",
                    MH(MH_ERR, "Error:"), mh_color ? MH_OPT : "", argv[i], mh_color ? MH_RST : "",
                    mh_color ? MH_CMD : "", argv[0], mh_color ? MH_RST : "", MH(MH_OPT, "-h"));
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
        sigaction(SIGBUS, &sa, NULL);
        sigaction(SIGABRT, &sa, NULL);
    }

    /* --- Detect color level --- */
    ov_detect_color_level();

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

    /* --- Main TUI loop (~10 fps) --- */
    /* Clear screen once on startup */
    {
        const char cls[] = "\033[2J\033[H";
        if (write(STDOUT_FILENO, cls, sizeof(cls) - 1) < 0)
        {
        }
    }

    int             last_rows   = -1;
    int             last_cols   = -1;
    const OV_MODEL *m           = NULL;
    int             need_render = 1; /* force first frame */

    while (!OV_SIG_ANY_SET())
    {
        /* Recompute layout (handles resize) */
        ov_layout_compute(&lay);

        if (lay.term_rows != last_rows || lay.term_cols != last_cols)
        {
            /* Size changed, force clear */
            const char cls[] = "\033[2J\033[H";
            if (write(STDOUT_FILENO, cls, sizeof(cls) - 1) < 0)
            {
            }
            ov_buf_force_clear();
            last_rows   = lay.term_rows;
            last_cols   = lay.term_cols;
            need_render = 1;
        }

        /* Pick up new model if available */
        if (!lay.paused || m == NULL)
        {
            const OV_MODEL *prev = m;
            m                    = ov_scan_get_model();
            if (m != prev)
            {
                need_render = 1;
            }
        }

        /* Drain all pending input */
        {
            int quit = 0;
            int key;
            while ((key = ov_get_key()) != OV_KEY_NONE)
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
            pfd.fd       = STDIN_FILENO;
            pfd.events   = POLLIN;
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
                        else if (pk != OV_KEY_NONE)
                        {
                            ov_handle_key(pk, &lay, m);
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
