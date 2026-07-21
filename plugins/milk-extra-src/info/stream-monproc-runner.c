// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream-monproc-runner.c
 * @brief   Standalone runner for stream_monproc
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "stream_monproc.h"

// External initialization functions from CLIcore
extern errno_t CLI_startup();
extern errno_t setSHMdir();
extern errno_t CLI_data_init();
extern errno_t memory_re_alloc();

int main(int argc, char *argv[])
{
    char    *stream_name = NULL;
    uint64_t binflag     = 63;   // Default
    uint32_t cbsize      = 1024; // Default
    int      procinfo    = 0;
    int      fps         = 0;

    static struct option long_options[] = {
        { "help", no_argument, 0, 'h' },         { "binflag", required_argument, 0, 'b' },
        { "cbsize", required_argument, 0, 'c' }, { "procinfo", no_argument, 0, 'p' },
        { "fps", no_argument, 0, 'f' },          { 0, 0, 0, 0 }
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "hb:c:pf", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'h':
            printf("Usage: %s [options] <stream_name>\n", argv[0]);
            printf("Options:\n");
            printf("  -b, --binflag <int>   Time binning flag (default: %lu)\n", binflag);
            printf("  -c, --cbsize <int>    Circular buffer size (default: %u)\n", cbsize);
            printf("  -p, --procinfo        Enable process info\n");
            printf("  -f, --fps             Enable FPS (Function Parameter Structure)\n");
            printf("  -h, --help            Show this help message\n");
            printf("\n");
            stream_monitor_help();
            return EXIT_SUCCESS;
        case 'b':
            binflag = strtoull(optarg, NULL, 10);
            break;
        case 'c':
            cbsize = (uint32_t) strtoul(optarg, NULL, 10);
            break;
        case 'p':
            procinfo = 1;
            break;
        case 'f':
            fps = 1;
            break;
        default:
            fprintf(stderr, "Usage: %s [options] <stream_name>\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (optind < argc)
    {
        stream_name = argv[optind];
    }
    else
    {
        fprintf(stderr, "Error: Stream name argument is required.\n");
        fprintf(stderr, "Usage: %s [options] <stream_name>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Initialization
    strncpy(data.processname, argv[0], STRINGMAXLEN_PROCESSNAME - 1);

    if (CLI_startup() != RETURN_SUCCESS)
    {
        fprintf(stderr, "Error: CLI_startup failed\n");
        return EXIT_FAILURE;
    }

    setSHMdir();

    if (CLI_data_init() != RETURN_SUCCESS)
    {
        fprintf(stderr, "Error: CLI_data_init failed\n");
        return EXIT_FAILURE;
    }

    if (memory_re_alloc() != RETURN_SUCCESS)
    {
        fprintf(stderr, "Error: memory_re_alloc failed\n");
        return EXIT_FAILURE;
    }

    printf("Starting stream monitor for stream: %s\n", stream_name);
    printf("Binflag: %lu, CBsize: %u\n", binflag, cbsize);

    // Run
    if (stream_monitor_run(stream_name, binflag, cbsize, procinfo, fps) != RETURN_SUCCESS)
    {
        fprintf(stderr, "Stream monitor exited with error\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
