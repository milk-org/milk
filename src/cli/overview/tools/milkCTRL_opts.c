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
#include "overview_keymap.h"
#include "overview_layout.h"
#include "overview_theme.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/** help_row - print one "key  description" row, the key padded before it is coloured. */
static void help_row(const char *key, const char *what, int mh_color)
{
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", key, mh_color ? MH_RST : "", what);
}

/** help_rows - print @n "key  description" rows and a blank line. */
static void help_rows(const char *const rows[][2], int n, int mh_color)
{
    for (int i = 0; i < n; i++)
    {
        help_row(rows[i][0], rows[i][1], mh_color);
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

    for (int g = 0; g < OV_KG_COUNT; g++) /* key sections: the views, then ov_keys rows by group */
    {
        milk_help_section(ov_keygroup_title[g], mh_color);
        for (int v = 0; g == OV_KG_VIEWS && v < OV_VIEW_COUNT; v++)
        {
            char key[16];
            snprintf(key, sizeof(key), "F%d  %s", ov_views[v].fkey, ov_views[v].label);
            help_row(key, ov_views[v].desc, mh_color);
        }
        for (int i = 0; i < ov_nb_keys; i++)
        {
            const OV_KEYDEF *k     = &ov_keys[i];
            const char      *panel = (k->section == HS_STREAMS) ? "STRM: "
                                     : (k->section == HS_PROCS) ? "PROC: "
                                     : (k->section == HS_FPS)   ? "FPS: "
                                                                : "";
            char             what[160];
            snprintf(what, sizeof(what), "%s%s", (g == OV_KG_CTRL) ? panel : "", k->label);
            if (k->group == g)
            {
                help_row(k->display, what, mh_color);
            }
        }
        printf("\n");
    }

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
