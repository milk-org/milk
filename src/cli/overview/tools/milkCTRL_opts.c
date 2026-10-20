// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkCTRL_opts.c
 * @brief   Command-line option parsing and help banner for milk-CTRL.
 */

#include "milkCTRL_opts.h"
#include "milk_config.h"
#include "milk_help.h"
#include "overview_data_internal.h"
#include "overview_theme.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/**
 * milkctrl_print_help - render colored command reference and CLI manual for milk-CTRL.
 * @prog:     Program invocation name.
 * @mh_color: Color formatting mode flag.
 */
void milkctrl_print_help(const char *prog, int mh_color)
{
    milk_help_banner(
        prog, "unified system dashboard TUI (milk-CTRL) for streams, FPS, and processes", mh_color);

    milk_help_section("Usage", mh_color);
    printf("  $ %s [%s %s]  (commit %s, shm %s)\n\n", prog, MH(MH_OPT, "-d"), MH(MH_ARG, "DIR"),
           MILK_GIT_COMMIT, ov_get_shmdir());

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

    milk_help_section("Dashboard Layout (F2 - F7)", mh_color);
    printf(
        "  - %s (F2): Grid overview of Streams, Processes, and FPS panels.\n"
        "  - %s (F3): Full-screen Streams panel with detailed dimensions, semaphores, & IO "
        "rates.\n"
        "  - %s (F4): Full-screen Process monitor with status (RUN/STOP/CRSH), CPU, & loop "
        "counts.\n"
        "  - %s  (F5): Full-screen FPS list (left) and interactive parameter tree (right).\n"
        "  - %s (F6): Visual dataflow node graph tracing upstream/downstream lineage.\n"
        "  - %s (F7): Closed feedback loops detection, circuit breakdown, & overlap analysis.\n\n",
        MH(MH_BOLD, "DASH"), MH(MH_BOLD, "STRM"), MH(MH_BOLD, "PROC"), MH(MH_BOLD, "FPS"),
        MH(MH_BOLD, "CONN"), MH(MH_BOLD, "LOOP"));

    milk_help_section("Feedback Loops (LOOPS tab / F7 view)", mh_color);
    printf("  %-30s Cycle Graph sub-tabs (CONNECTIONS, LOOPS, DETAILS, RESOURCES)\n",
           MH(MH_OPT, "SHIFT + TAB"));
    printf("  %-30s Rename selected feedback loop (persisted across sessions)\n", MH(MH_OPT, "r"));
    printf("  %-30s Toggle loop isolation filter (isolate loop streams, procs, & FPS)\n",
           MH(MH_OPT, "f / ENTER"));
    printf("  %-30s Switch to graph CONNECTIONS tab to inspect dataflow circuit tree\n\n",
           MH(MH_OPT, "g"));

    milk_help_section("Options", mh_color);
    printf("  %-30s Show this help and exit\n", MH(MH_OPT, "-h, --help"));
    printf("  %-30s One-line description and exit\n", MH(MH_OPT, "-h1, --help-oneline"));
    printf("  %-30s Full help, forced monochrome\n", MH(MH_OPT, "-hm, --help-mono"));
    printf("  %-30s Set color theme: dark, night, accessible, light, nordic,\n"
           "  %-30s   dracula, solarized-dark, solarized-light, monokai, matrix\n",
           MH(MH_OPT, "-T, --theme <NAME>"), "");
    printf("  %-30s Override SHM/process directory (current: %s)\n\n", MH(MH_OPT, "-d <DIR>"),
           ov_get_shmdir());

    milk_help_section("Navigation & View Controls", mh_color);
    printf("  %-30s Switch active panel focus (Dashboard / FPS view)\n", MH(MH_OPT, "TAB"));
    printf("  %-30s Navigate rows in the currently focused list\n", MH(MH_OPT, "UP / DOWN"));
    printf("  %-30s Scroll page up / down\n", MH(MH_OPT, "PgUp / PgDn"));
    printf("  %-30s Jump to top / bottom of the list\n", MH(MH_OPT, "Home / End"));
    printf("  %-30s Scroll list/table horizontally\n", MH(MH_OPT, "LEFT / RIGHT"));
    printf("  %-30s Open theme selector popup (↑/↓ to choose, ESC/1s to close)\n",
           MH(MH_OPT, "F8 / CTRL+T"));
    printf("  %-30s Toggle detailed inspection pane / parameter edit mode\n", MH(MH_OPT, "ENTER"));
    printf("  %-30s Toggle details tab on selected item / Graph details\n", MH(MH_OPT, "D"));
    printf("  %-30s Filter items in the focused list (regex search)\n", MH(MH_OPT, "/"));
    printf("  %-30s Toggle regex filter ON/OFF (preserves query string)\n", MH(MH_OPT, "f"));
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
           "    %-28s Toggle Configuration loop state (confstart / confstop)\n\n",
           MH(MH_OPT, "CTRL + e"), MH(MH_OPT, "DEL / CTRL+e"), MH(MH_OPT, "k"), MH(MH_OPT, "K"),
           MH(MH_OPT, "p"), MH(MH_OPT, "^s"), MH(MH_OPT, "e"), MH(MH_OPT, "z"), MH(MH_OPT, "C"),
           MH(MH_OPT, "k"), MH(MH_OPT, "K"), MH(MH_OPT, "r"), MH(MH_OPT, "s"));

    milk_help_section("Mouse Interactions", mh_color);
    printf("  - Click anywhere on a row to select it.\n"
           "  - Double-click a row to open the detailed inspector pane.\n"
           "  - Scroll the mouse wheel to navigate lists vertically.\n"
           "  - Click column headers to sort the table by that column.\n"
           "  - Click dashboard tabs (DASH, STRM, etc.) to switch views.\n"
           "  - Drag panel borders/separators to resize split panels.\n\n");
}

/**
 * usage_error - print a monochrome usage error to stderr.
 * Return: -1, so callers can return it directly.
 */
static int usage_error(const char *prog, const char *what, const char *arg)
{
    fprintf(stderr, "Error: %s: %s\nRun %s -h for usage.\n", what, arg, prog);
    return -1;
}

/**
 * milkctrl_parse_options - parse command line flags for milk-CTRL.
 * @argc:          Argument count.
 * @argv:          Argument strings.
 * @out_cli_theme: Output pointer to theme name if -T specified.
 *
 * Return: 0 to run, 1 when help was printed, -1 on a usage error (caller exits 2).
 */
int milkctrl_parse_options(int argc, char *argv[], const char **out_cli_theme)
{
    int action = milk_help_init(
        argc, argv, "unified system dashboard TUI (milk-CTRL) for streams, FPS, and processes",
        "milk-CTRL is the unified, real-time diagnostic and control dashboard for the\n"
        "milk shared-memory micro-service framework. It connects directly to zero-copy\n"
        "ImageStreamIO streams, maps the FPS parameters, and tracks managed processes.\n"
        "Operators can trace dataflow lineage, configure running nodes, diagnose CPU\n"
        "and hardware bottlenecks, and manage execution loops interactively.");

    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 1;
    }
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        milkctrl_print_help(argv[0], action == MH_ACTION_HELP);
        return 1;
    }

    static const struct option lopts[]   = { { "theme", required_argument, NULL, 'T' },
                                             { NULL, 0, NULL, 0 } };
    const char                *cli_theme = NULL;
    int                        c;

    opterr = 0;
    while ((c = getopt_long(argc, argv, ":d:T:", lopts, NULL)) != -1)
    {
        switch (c)
        {
        case 'd':
        {
            struct stat st;
            if (stat(optarg, &st) != 0 || !S_ISDIR(st.st_mode))
            {
                return usage_error(argv[0], "not a directory", optarg);
            }
            setenv("MILK_SHM_DIR", optarg, 1);
            setenv("MILK_PROC_DIR", optarg, 1);
            break;
        }
        case 'T':
            if (ov_theme_find_by_id(optarg) < 0)
            {
                usage_error(argv[0], "unknown theme", optarg);
                fprintf(stderr, "Valid themes:");
                for (int i = 0; i < ov_theme_count(); i++)
                {
                    fprintf(stderr, "%s %s", i ? "," : "", ov_theme_get(i)->id);
                }
                fprintf(stderr, "\n");
                return -1;
            }
            cli_theme = optarg;
            break;
        case ':':
            return usage_error(argv[0], "option requires an argument", argv[optind - 1]);
        default:
            return usage_error(argv[0], "unknown option", argv[optind - 1]);
        }
    }
    if (optind < argc)
    {
        return usage_error(argv[0], "unexpected argument", argv[optind]);
    }

    if (out_cli_theme != NULL)
    {
        *out_cli_theme = cli_theme;
    }
    return 0;
}
