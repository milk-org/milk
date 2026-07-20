// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include "ImageStreamIO/ImageStreamIO.h"

void print_help()
{
    printf("Usage: milk-stream-cnt2push <stream_name> [options]\n");
    printf("\n");
    printf("Purpose:\n");
    printf("  Updates the 'cnt2' counter of a shared memory image stream. This counter is "
           "typically\n");
    printf("  used for demand-driven synchronization (flow control), where a writer waits until\n");
    printf(
        "  cnt0 < cnt2 before proceeding. By incrementing cnt2, a reader requests more frames.\n");
    printf("\n");
    printf("counters:\n");
    printf("  cnt0: Current write counter (increments when stream is written)\n");
    printf("  cnt2: Control counter (used to gate execution in CNT2 trigger mode)\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --value <val>   Value to add or set (default: 1)\n");
    printf("  -a, --abs           Set cnt2 to an absolute value <val>.\n");
    printf("                      (default is relative to cnt0: cnt2 = cnt0 + <val>)\n");
    printf("  -i, --inc           Increment from current cnt2 value.\n");
    printf("                      (cnt2 = cnt2 + <val>)\n");
    printf("  -h, --help          Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  milk-stream-cnt2push mystream\n");
    printf("      Sets cnt2 = cnt0 + 1. Writer will process 1 more frame.\n");
    printf("\n");
    printf("  milk-stream-cnt2push mystream -v 10\n");
    printf("      Sets cnt2 = cnt0 + 10. Writer will process 10 more frames.\n");
    printf("\n");
    printf("  milk-stream-cnt2push mystream -i -v 5\n");
    printf("      Sets cnt2 = cnt2 + 5. Adds 5 more frames to the current quota.\n");
    printf("\n");
    printf("  milk-stream-cnt2push mystream -a -v 100\n");
    printf("      Sets cnt2 = 100. Writer will stop when cnt0 >= 100.\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_help();
        return 1;
    }

    char   *streamname = NULL;
    int64_t val        = 1;
    int     mode_abs   = 0;
    int     mode_inc   = 0;

    int opt;
    // Simple argument parsing loop (since getopt might not handle non-option arg first easily if strict posix)
    // We assume first arg is streamname if it doesn't start with -
    int arg_idx = 1;
    if (argv[1][0] != '-')
    {
        streamname = argv[1];
        arg_idx++;
    }

    for (int i = arg_idx; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_help();
            return 0;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--value") == 0)
        {
            if (i + 1 < argc)
            {
                val = atoll(argv[++i]);
            }
            else
            {
                fprintf(stderr, "Error: -v requires an argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--abs") == 0)
        {
            mode_abs = 1;
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--inc") == 0)
        {
            mode_inc = 1;
        }
        else
        {
            if (streamname == NULL && argv[i][0] != '-')
            {
                streamname = argv[i];
            }
            else
            {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                return 1;
            }
        }
    }

    if (streamname == NULL)
    {
        fprintf(stderr, "Error: stream name required\n");
        return 1;
    }

    IMAGE   image;
    errno_t res = ImageStreamIO_read_sharedmem_image_toIMAGE(streamname, &image);
    if (res != 0)
    {
        fprintf(stderr, "Error: could not read shared memory image %s\n", streamname);
        return 1;
    }

    uint64_t cnt0   = image.md->cnt0;
    uint64_t cnt2   = image.md->cnt2;
    uint64_t target = 0;

    if (mode_abs)
    {
        target = val;
        printf("Stream %s: cnt0=%lu, cnt2=%lu. Setting cnt2 to absolute value %lu\n", streamname,
               cnt0, cnt2, target);
    }
    else if (mode_inc)
    {
        target = cnt2 + val;
        printf("Stream %s: cnt0=%lu, cnt2=%lu. Incrementing cnt2 by %ld -> %lu\n", streamname, cnt0,
               cnt2, val, target);
    }
    else
    {
        target = cnt0 + val;
        printf("Stream %s: cnt0=%lu, cnt2=%lu. Setting cnt2 to cnt0 + %ld -> %lu\n", streamname,
               cnt0, cnt2, val, target);
    }

    image.md->cnt2 = target;

    // Optional: Post semaphore if needed? Usually writing cnt2 is enough for polling readers.
    // If waiting on semaphore, one might need to post.
    // But cnt2 logic usually implies polling or waiting on condition.
    // The previous implementation of PROCESSINFO_TRIGGERMODE_CNT2 waits on condition.

    return 0;
}
