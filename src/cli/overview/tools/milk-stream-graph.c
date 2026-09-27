// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-stream-graph.c
 * @brief Standalone stream dependency graph tool entry point.
 *
 * Computes and displays ancestor/descendant lineage
 * for a given shared-memory stream.
 */

#include "milk-stream-graph.h"
#include "milk_help.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;

/**
 * print_usage - Print command-line usage information for milk-stream-graph
 * @prog:     Program invocation name
 * @mh_color: Color mode flag
 */
static void print_usage(
    const char *prog,
    int         mh_color)
{
    milk_help_banner(prog, SG_ONELINE, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<stream>%s\n\n", mh_color ? MH_CMD : "", prog,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SG_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-m, --mode MODE", mh_color ? MH_RST : "",
           "Traversal mode: trigger|input|full (default: trigger)");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-p, --pretty", mh_color ? MH_RST : "",
           "Force TrueColor ANSI output");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-t, --text", mh_color ? MH_RST : "",
           "Plain text machine-readable output");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-j, --json", mh_color ? MH_RST : "",
           "JSON machine-readable output");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-i, --interactive", mh_color ? MH_RST : "",
           "Interactive navigation");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-d DIR", mh_color ? MH_RST : "",
           "Override SHM directory");
    printf("  %s%-25s%s %s (default: %d)\n", mh_color ? MH_OPT : "", "--depth N",
           mh_color ? MH_RST : "", "Max traversal depth", SG_MAX_DEPTH);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Interactive keys", mh_color);
    printf("  UP/DOWN   Navigate list\n");
    printf("  ENTER     Re-root on selected stream\n");
    printf("  r         Rescan graph\n");
    printf("  t/i/f     Switch mode (trigger/input/full)\n");
    printf("  g         Go to stream by name\n");
    printf("  q         Quit\n\n");
    const char *see_also[] = { "milk-stream-info:inspect stream metadata and data",
                               "milk-stream-list:list active shared memory streams" };
    milk_help_see_also(see_also, 2, mh_color);
}

/**
 * parse_mode - Parse graph traversal mode string
 * @s: Mode name ("input", "full", or "trigger")
 *
 * Return: Corresponding sg_mode_t enum.
 */
static sg_mode_t parse_mode(
    const char *s)
{
    if (strcmp(s, "input") == 0)
    {
        return SG_MODE_INPUT;
    }
    if (strcmp(s, "full") == 0)
    {
        return SG_MODE_FULL;
    }
    return SG_MODE_TRIGGER;
}

/**
 * sg_scan_model - Scan system state and build graph model
 * @model: Target data model to populate
 */
void sg_scan_model(
    OV_MODEL *model)
{
    memset(model, 0, sizeof(*model));
    ov_scan_streams(model);
    ov_scan_fps(model);
    ov_scan_procs(model);
    ov_build_graph(model);
}

/**
 * main - Entry point for milk-stream-graph utility
 * @argc: Argument count
 * @argv: Argument vector
 *
 * Return: 0 on success, non-zero on error.
 */
int main(
    int   argc,
    char *argv[])
{
    int action = milk_help_init(argc, argv, SG_ONELINE, SG_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_usage(argv[0], mh_color);
        return 0;
    }

    sg_mode_t   mode        = SG_MODE_TRIGGER;
    sg_output_t output      = isatty(STDOUT_FILENO) ? OUT_PRETTY : OUT_TEXT;
    int         interactive = 0;
    const char *stream_name = NULL;

    /* Parse arguments */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            break; /* handled above */
        }
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mode") == 0) && i + 1 < argc)
        {
            mode = parse_mode(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--pretty") == 0)
        {
            output = OUT_PRETTY;
            continue;
        }
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--text") == 0)
        {
            output = OUT_TEXT;
            continue;
        }
        if (strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--json") == 0)
        {
            output = OUT_JSON;
            continue;
        }
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0)
        {
            interactive = 1;
            continue;
        }
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
        {
            setenv("MILK_SHM_DIR", argv[++i], 1);
            continue;
        }
        if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc)
        {
            /* depth is compile-time constant;
             * accepted for compat, ignored */
            i++;
            continue;
        }
        /* positional: stream name */
        if (argv[i][0] != '-')
        {
            stream_name = argv[i];
            continue;
        }
        printf("\n\033[1;31mERROR\033[0m: Invalid option: %s\n\n", argv[i]);
        print_usage(argv[0], mh_color);
        return 1;
    }

    if (stream_name == NULL)
    {
        printf("\n\033[1;31mERROR\033[0m: stream name required.\n\n");
        print_usage(argv[0], mh_color);
        return 1;
    }

    /* Scan system */
    OV_MODEL *model = calloc(1, sizeof(OV_MODEL));
    if (model == NULL)
    {
        PRINT_ERROR("memory allocation failed");
        return 1;
    }

    if (interactive)
    {
        sg_interactive(model, stream_name, mode);
        free(model);
        return 0;
    }

    /* One-shot mode */
    sg_scan_model(model);

    int si = ov_find_stream_by_name(model, stream_name);
    if (si < 0)
    {
        PRINT_ERROR("stream '%s' not found", stream_name);
        free(model);
        return 1;
    }

    SG_LINEAGE lin;
    sg_compute_lineage(model, si, mode, &lin);

    switch (output)
    {
    case OUT_TEXT:
        sg_print_text(model, stream_name, mode, &lin);
        break;
    case OUT_PRETTY:
        sg_print_pretty(model, stream_name, mode, &lin);
        break;
    case OUT_JSON:
        sg_print_json(model, stream_name, mode, &lin);
        break;
    }

    free(model);
    return 0;
}
