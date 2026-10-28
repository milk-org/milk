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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** help_row - print one "key  description" row, the key padded before it is coloured. */
static void help_row(const char *key, const char *what, int mh_color)
{
    printf("  %s%-24s%s %s\n", mh_color ? MH_OPT : "", key, mh_color ? MH_RST : "", what);
}

/** key_what - the one-line description of key row @k in -h and the docs (CONTROL rows name their panel). */
static const char *key_what(const OV_KEYDEF *k, char *buf, size_t n)
{
    const char *panel = (k->section == HS_STREAMS) ? "STRM: "
                        : (k->section == HS_PROCS) ? "PROC: "
                        : (k->section == HS_FPS)   ? "FPS: "
                                                   : "";
    snprintf(buf, n, "%s%s", (k->group == OV_KG_CTRL) ? panel : "", k->label);
    return buf;
}

/** milkctrl_print_keys_markdown - the key sections of -h as Markdown tables (docs/guide/milk-CTRL.md). */
static void milkctrl_print_keys_markdown(void)
{
    for (int g = 0; g < OV_KG_COUNT; g++)
    {
        printf("%s### %s\n\n| Keys | Action |\n|------|--------|\n", g ? "\n" : "",
               ov_keygroup_title[g]);
        for (int v = 0; g == OV_KG_VIEWS && v < OV_VIEW_COUNT; v++)
        {
            printf("| `F%d` %s | %s |\n", ov_views[v].fkey, ov_views[v].label, ov_views[v].desc);
        }
        for (int i = 0; i < ov_nb_keys; i++)
        {
            char what[160];
            if (ov_keys[i].group == g)
            {
                printf("| `%s` | %s |\n", ov_keys[i].display,
                       key_what(&ov_keys[i], what, sizeof(what)));
            }
        }
    }
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
    printf("  $ %s [%s] [%s]  (commit %s, shm %s)\n\n", prog, MH(MH_OPT, "OPTIONS"),
           MH(MH_ARG, "PATTERN"), MILK_GIT_COMMIT, ov_get_shmdir());

    milk_help_section("Description", mh_color);
    printf("  Real-time dashboard for milk streams (ImageStreamIO), function parameter\n"
           "  structures (FPS) and processinfo loops, with the links between them.\n"
           "  Viewing never writes shared memory; CONTROL mode ('c') enables actions.\n\n");

    const char *exe = (strrchr(prog, '/') != NULL) ? strrchr(prog, '/') + 1 : prog;
    if (strcmp(exe, "milk-streamCTRL") == 0)
    {
        static const char *const legacy[][2] = {
            { "F2 semaphores", "SEMS column; inspector SEMS page (Right, then Right)" },
            { "F3/F4 write/read PIDs", "WPID and RPID columns; inspector SEMS page" },
            { "F5 process trace", "inspector TRACE page" },
            { "F6 holders (fuser)", "inspector HOLDERS page (/proc maps, every 2 s)" },
            { "RIGHT detail", "Right or ENTER: inspector INFO page" },
            { "1 2 3 4 ] [ sort", "s name, S rate, > < columns (CPID, OWNER, HOLD), [ reverse" },
            { "Ctrl+e erase", "c (CONTROL), then Ctrl+e or DEL, confirmed with y" },
            { "{ } scan rate", "+ / -" },
            { "-w", "accepted, does nothing (as before)" },
        };
        milk_help_section("milk-streamCTRL", mh_color);
        printf("  milk-streamCTRL is milk-CTRL opened on F3:STRM. Its old keys map to:\n");
        HELP_ROWS(legacy);
    }

    milk_help_section("Options", mh_color);
    printf("  %-30s Show this help and exit\n", MH(MH_OPT, "-h, --help"));
    printf("  %-30s One-line description and exit\n", MH(MH_OPT, "-h1, --help-oneline"));
    printf("  %-30s Full help, forced monochrome\n", MH(MH_OPT, "-hm, --help-mono"));
    printf("  %-30s Streams and processinfo directory (current: %s)\n",
           MH(MH_OPT, "-d, --shm-dir <DIR>"), ov_get_shmdir());
    printf("  %-30s Open this view: dash streams procs fps conn loops\n",
           MH(MH_OPT, "--view <NAME>"));
    printf("  %-30s Scan period, 0.1 to 10 seconds (default 1; +/- change it)\n",
           MH(MH_OPT, "--interval <SEC>"));
    printf("  %-30s Append the command log to FILE\n", MH(MH_OPT, "--log <FILE>"));
    printf("  %-30s Also log key events and scan timing to the --log file\n",
           MH(MH_OPT, "--debug"));
    printf("  %-30s Initial regex filter of the view's list (DASH, CONN, LOOPS: streams)\n",
           MH(MH_ARG, "PATTERN"));
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
            char what[160];
            if (ov_keys[i].group == g)
            {
                help_row(ov_keys[i].display, key_what(&ov_keys[i], what, sizeof(what)), mh_color);
            }
        }
        printf("\n");
    }

    milk_help_section("Mouse", mh_color);
    printf("  Click selects, double-click acts as ENTER, the wheel scrolls, Ctrl+wheel cycles\n"
           "  views. Click tabs and sortable column headers; drag split borders.\n"
           "  Hover action buttons appear in CONTROL mode.\n\n");

    milk_help_section("Examples", mh_color);
    printf("  $ milk-CTRL\n  $ milk-CTRL -T night\n  $ milk-CTRL -d /tmp/mysim\n"
           "  $ milk-CTRL --view procs 'wfs|dm'\n  $ milk-CTRL --interval 0.2 --log "
           "/tmp/ctrl.log\n\n");

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
 * milkctrl_parse_options - help options, then ov_opts_parse(); usage errors go to stderr.
 * Return: 0 to run, 1 when help was printed, -1 on a usage error (caller exits 2).
 */
int milkctrl_parse_options(int argc, char *argv[], OV_OPTS *o)
{
    if (argc == 2 &&
        strcmp(argv[1], "--keys-markdown") == 0) /* hidden: regenerates the docs key tables */
    {
        milkctrl_print_keys_markdown();
        return 1;
    }
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
    if (ov_opts_parse(argc, argv, o) == 0)
    {
        return 0;
    }
    fprintf(stderr, "Error: %s\nRun %s -h for usage.\n", o->err, argv[0]);
    if (strncmp(o->err, "unknown theme", 13) == 0)
    {
        fprintf(stderr, "Valid themes:");
        for (int i = 0; i < ov_theme_count(); i++)
        {
            fprintf(stderr, "%s %s", i ? "," : "", ov_theme_get(i)->id);
        }
        fprintf(stderr, "\n");
    }
    return -1;
}
