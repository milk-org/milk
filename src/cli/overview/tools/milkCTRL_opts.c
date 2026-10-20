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

/** help_rows - print @n "key  description" rows, the key padded before it is coloured. */
static void help_rows(const char *const rows[][2], int n, int mh_color)
{
    for (int i = 0; i < n; i++)
    {
        printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", rows[i][0], mh_color ? MH_RST : "",
               rows[i][1]);
    }
    printf("\n");
}

#define HELP_ROWS(rows) help_rows(rows, (int) (sizeof(rows) / sizeof(rows[0])), mh_color)

/**
 * milkctrl_print_help - command reference for milk-CTRL, in the help-message-standard order.
 * @prog:     Program invocation name.
 * @mh_color: Color formatting mode flag.
 */
void milkctrl_print_help(const char *prog, int mh_color)
{
    milk_help_banner(
        prog, "unified system dashboard TUI (milk-CTRL) for streams, FPS, and processes", mh_color);

    milk_help_section("Usage", mh_color);
    printf("  $ %s [%s %s] [%s %s]  (commit %s, shm %s)\n\n", prog, MH(MH_OPT, "-d"),
           MH(MH_ARG, "DIR"), MH(MH_OPT, "-T"), MH(MH_ARG, "THEME"), MILK_GIT_COMMIT,
           ov_get_shmdir());

    milk_help_section("Description", mh_color);
    printf("  Real-time dashboard for milk streams (ImageStreamIO), function parameter\n"
           "  structures (FPS) and processinfo loops, with the links between them.\n"
           "  Viewing never writes shared memory; CONTROL mode ('c') enables actions.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %-30s Show this help and exit\n", MH(MH_OPT, "-h, --help"));
    printf("  %-30s One-line description and exit\n", MH(MH_OPT, "-h1, --help-oneline"));
    printf("  %-30s Full help, forced monochrome\n", MH(MH_OPT, "-hm, --help-mono"));
    printf("  %-30s Streams and processinfo directory (current: %s)\n", MH(MH_OPT, "-d <DIR>"),
           ov_get_shmdir());
    printf("  %-30s Colour theme, one of:\n", MH(MH_OPT, "-T, --theme <NAME>"));
    for (int i = 0; i < ov_theme_count(); i++)
    {
        printf("      %-18s %s\n", ov_theme_get(i)->id, ov_theme_get(i)->desc);
    }
    printf("  Exit status: 0 ok, 1 runtime error, 2 usage error\n\n");

    milk_help_section("Views (F2-F7; Ctrl+Left/Right or Ctrl+wheel cycle them)", mh_color);
    static const char *const views[][2] = {
        { "F2  DASH", "Streams, processes and FPS panels plus the graph panel" },
        { "F3  STRM", "Streams: size, rate, semaphores, writers" },
        { "F4  PROC", "Processes: state (RUN/STOP/CRSH/STAL), rate, CPU" },
        { "F5  FPS", "FPS list (left) and parameter tree (right)" },
        { "F6  CONN", "Connection tree of the entry selected last" },
        { "F7  LOOPS", "Feedback loops: members, rate, status (ERR/STAL/PAUS/RUN/IDLE)" },
    };
    HELP_ROWS(views);

    milk_help_section("Navigation", mh_color);
    static const char *const nav[][2] = {
        { "UP / DOWN, PgUp / PgDn", "Move in the focused list" },
        { "Home / End", "First / last row" },
        { "TAB", "Cycle panel focus (DASH); list / parameter tree (F5)" },
        { "LEFT / RIGHT", "Adjacent panel (DASH); tree up / down (F5)" },
        { "Shift+TAB", "Cycle graph sub-tab: CONNECTIONS, LOOPS, DETAILS, RESOURCES" },
        { "ENTER", "DETAILS sub-tab (DASH); select the row's stream (CONNECTIONS);" },
        { "", "enter tree / edit parameter (F5); loop filter (LOOPS)" },
        { "D", "Toggle DETAILS / CONNECTIONS (DASH, F6)" },
        { "Shift+UP / DOWN", "Nearest upstream / downstream entity" },
        { "L", "Lineage mode: Trigger / Input / Full" },
        { "SPACE", "Freeze the selected entry (its relations stay highlighted)" },
        { "/  ?", "Filter the focused list (regex) / jump to the first match" },
        { "f / ^F", "Toggle the filter; ESC clears it" },
        { "ESC", "Back out: parameter tree, DETAILS cursor, filters, FPS marks" },
        { "i", "Inspect: milk-stream-info / milk-procinfo-info / milk-fps-info" },
        { "W", "Snapshot to $TMPDIR (default /tmp), path shown in the log" },
        { "G, v / V", "Command log on/off, smaller / larger" },
        { "F", "Pause / resume display updates" },
        { "+ / -", "Faster / slower scan" },
        { "m", "Mouse hover highlights" },
        { "F8 / ^T", "Next theme (selector: UP/DOWN, ESC)" },
        { "h", "Interactive help" },
        { "q / x", "Quit (also ^C ^D ^Q ^X)" },
    };
    HELP_ROWS(nav);

    milk_help_section("Columns and sorting", mh_color);
    static const char *const cols[][2] = {
        { "Shift+LEFT / RIGHT", "Move the column cursor" },
        { "t / T", "Hide / show the column under the cursor" },
        { "d", "Compact columns" },
        { "{ }  ( )", "Resize vertical / horizontal splits" },
        { "s / S / A", "Sort by name / rate / ancestry (PROC, FPS: 's' outside CONTROL)" },
        { "< / > or ]", "Previous / next sort column" },
        { "[", "Reverse the sort" },
    };
    HELP_ROWS(cols);

    milk_help_section("Feedback loops (F7, or LOOPS sub-tab)", mh_color);
    static const char *const loops[][2] = {
        { "f / ENTER", "Isolate the loop: lists show its members only" },
        { "r / R", "Rename the loop (saved)" },
        { "g", "Show its connection tree" },
    };
    HELP_ROWS(loops);

    milk_help_section("Control mode actions ('c' toggles; y/N asks before destructive ones)",
                      mh_color);
    static const char *const ctl[][2] = {
        { "k", "PROC/FPS: SIGTERM (FPS: conf and run processes)" },
        { "K", "PROC/FPS: SIGKILL, after y/N" },
        { "p / ^S / e", "PROC: pause toggle (CTRLval 1/0) / step / exit" },
        { "z", "PROC: reset the loop counter" },
        { "C", "PROC: remove all STOPPED/CRASHED entries, after y/N" },
        { "r / s", "FPS: start or stop RUN / CONF" },
        { "o", "F5: toggle an ON/OFF parameter" },
        { "b / a", "FPS: mark / mark all visible; k K r s then act on the marks (y/N)" },
        { "DEL, ^e", "Delete stream / remove dead process / erase FPS, after y/N" },
    };
    HELP_ROWS(ctl);

    milk_help_section("Mouse", mh_color);
    printf("  Click selects, double-click acts as ENTER, the wheel scrolls, Ctrl+wheel cycles\n"
           "  views. Click tabs and sortable column headers; drag split borders.\n"
           "  Hover action buttons appear in CONTROL mode.\n\n");

    milk_help_section("Examples", mh_color);
    printf("  $ milk-CTRL\n  $ milk-CTRL -T night\n  $ milk-CTRL -d /tmp/mysim\n\n");

    milk_help_section("Environment", mh_color);
    static const char *const env[][2] = {
        { "MILK_SHM_DIR", "Streams and FPS directory (else /milk/shm, /dev/shm, /tmp)" },
        { "MILK_PROC_DIR", "processinfo directory (-d sets both)" },
        { "MILK_CTRL_THEME", "Initial theme (-T wins)" },
        { "COLORTERM, TERM", "truecolor/24bit: 24-bit; *256color: 256; else 16 colours" },
        { "NO_COLOR, TERM=dumb", "Monochrome; the selection is shown in reverse video" },
    };
    HELP_ROWS(env);

    const char *see[] = { "milk-stream-info:stream details",
                          "milk-procinfo-info:processinfo details",
                          "milk-fps-info:FPS details",
                          "milk-stream-graph:stream dataflow graph",
                          "milk-setup-caps:grant perf counter capabilities",
                          "milk-fpsCTRL:fpslist workflows" };
    milk_help_see_also(see, (int) (sizeof(see) / sizeof(see[0])), mh_color);
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
