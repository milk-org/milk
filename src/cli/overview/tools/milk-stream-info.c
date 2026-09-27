// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-stream-info.c
 * @brief Standalone CLI tool to print detailed info for a single SHM stream.
 */

#include "milk-stream-info.h"

#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by overview_defs.h (extern) */
volatile sig_atomic_t ov_sigINT  = 0;
volatile sig_atomic_t ov_sigTERM = 0;

/**
 * print_help - Print command-line help message for milk-stream-info
 * @progname: Name of executable
 * @mh_color: Flag indicating whether color is enabled
 */
static void print_help(
    const char *progname,
    int         mh_color)
{
    milk_help_banner(progname, SI_ONELINE, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<stream_name>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", SI_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-stream-info%s %sdm00disp%s\n\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-stream-list:list active shared memory streams",
                               "milk-stream-rm:remove shared memory streams",
                               "milk-procinfo-info:inspect processinfo memory contents" };
    milk_help_see_also(see_also, 3, mh_color);
}

/**
 * main - Entry point for milk-stream-info utility
 * @argc: Argument count
 * @argv: Argument vector
 *
 * Return: 0 on success, non-zero on error.
 */
int main(
    int   argc,
    char *argv[])
{
    int action = milk_help_init(argc, argv, SI_ONELINE, SI_DESC_LONG);
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

    static struct option long_opts[] = { { "help", no_argument, 0, 'h' }, { 0, 0, 0, 0 } };

    int opt;
    while ((opt = getopt_long(argc, argv, "h", long_opts, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            break; /* handled above */
        default:
            printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    if (optind >= argc)
    {
        printf("\n\033[1;31mERROR\033[0m stream name required\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    const char *stream_name = argv[optind];

    /* Build the system model */
    OV_MODEL model;
    memset(&model, 0, sizeof(model));
    ov_model_full_scan(&model);

    /* Find the requested stream */
    int si = ov_find_stream_by_name(&model, stream_name);
    if (si < 0)
    {
        PRINT_ERROR("stream '%s' not found in shared memory", stream_name);
        ov_scan_cache_cleanup();
        return 1;
    }

    print_stream_info(&model, si);

    ov_scan_cache_cleanup();
    return 0;
}
