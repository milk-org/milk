// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    perfbench-mkstream.c
 * @brief   Create a dummy SHM stream for benchmarking
 *
 * Usage: milk-perfbench-mkstream <name> <xsize> <ysize>
 *
 * Creates a 2D float shared-memory stream.
 * Engine-tier only (no CLI dependency).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libgen.h>

#include "milk_help.h"
#include "ImageStruct.h"
#include "ImageStreamIO.h"

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, "create a dummy shared-memory stream for FPS benchmarking",
                     mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s<name> <xsize> <ysize>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  Creates a 2D float shared-memory stream used for benchmarking the\n"
           "  FPS compute pipeline. Engine-tier only.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
}

int main(int argc, char *argv[])
{
    const char *progname = basename(argv[0]);

    int action =
        milk_help_init(argc, argv, "create a dummy shared-memory stream for FPS benchmarking",
                       "Creates a 2D float shared-memory stream used for benchmarking.");
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        return 0;
    }

    if (argc < 4)
    {
        fprintf(stderr,
                "Usage: %s <name>"
                " <xsize> <ysize>\n",
                progname);
        return 1;
    }

    const char *name  = argv[1];
    uint32_t    xsize = (uint32_t) atoi(argv[2]);
    uint32_t    ysize = (uint32_t) atoi(argv[3]);

    IMAGE    img;
    uint32_t dims[2] = { xsize, ysize };

    if (ImageStreamIO_createIm(&img, name, 2, dims, _DATATYPE_FLOAT, 1, 10, 0) != EXIT_SUCCESS)
    {
        fprintf(stderr, "Error creating stream '%s'\n", name);
        return 1;
    }

    printf("Created stream '%s' %ux%u float\n", name, xsize, ysize);

    return 0;
}
