// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-stream-help.c
 * @brief Help utility for ImageStreamIO shared memory streams.
 */

#include <stdio.h>
#include <string.h>

#define C_TITLE "\033[1;36m"
#define C_HDR "\033[1;34m"
#define C_CMD "\033[1;32m"
#define C_NOTE "\033[1;33m"
#define C_BOLD "\033[1m"
#define C_STRM "\033[1;35m"
#define C_RST "\033[0m"

int main(int argc, char *argv[])
{
    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h1") == 0 || strcmp(argv[ii], "--help-oneline") == 0)
        {
            printf("ImageStreamIO shared memory stream concepts and management guide\n");
            return 0;
        }
    }

    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "             ImageStreamIO Shared Memory Streams\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");

    printf("ImageStreamIO (Streams) is a real-time, zero-copy shared memory\n"
           "data-passing mechanism. It allows high-performance, n-dimensional\n"
           "data (such as images, matrices, or tensors) to be written by one\n"
           "process and read by multiple downstream processes with sub-microsecond\n"
           "latency.\n\n"
           "Each stream is represented by a shared-memory file on disk at:\n"
           "  " C_BOLD "/dev/shm/<streamname>.im.shm" C_RST "\n\n");

    printf(C_HDR "Inspecting and Monitoring Streams\n" C_RST);
    printf("  " C_CMD "milk-streamCTRL" C_RST
           "         TUI stream monitor (like top/htop for streams)\n"
           "  " C_CMD "milk-stream-info " C_STRM "<name>" C_RST
           " Detailed configuration and connection graph\n"
           "  " C_CMD "milk-stream-graph" C_RST
           "       Output a DOT graph of stream-process topologies\n"
           "  " C_CMD "milk-termview " C_STRM "<name>" C_RST
           "    Interactive terminal viewer for 2D stream pixels\n"
           "\n");

    printf(C_HDR "File I/O and Conversion\n" C_RST);
    printf("  " C_CMD "milk-FITS2shm " C_STRM "<file.fits> <stream>" C_RST
           " Load a FITS file into a new stream\n"
           "  " C_CMD "milk-shm2FITS " C_STRM "<stream> <file.fits>" C_RST
           " Save a stream snapshot to a FITS file\n"
           "\n");

    printf(C_HDR "Synchronization and Semaphores\n" C_RST);
    printf("  Each stream includes a configurable number of semaphores (default 10).\n"
           "  Downstream processes can register to block on a specific semaphore index.\n"
           "  When the upstream process writes a new frame, it posts to all registered\n"
           "  semaphores, instantly waking up the waiting processes.\n"
           "\n");

    printf(C_HDR "Interactive Session & Bash Stream Access\n" C_RST);
    printf("  Within Bash (after 'source milk.bash') or milk-cli:\n"
           "    " C_CMD "@get s.name.xsize" C_RST "        Read stream width (or " C_CMD
           "@s.name.xsize" C_RST ")\n"
           "    " C_CMD "@get s.name.cnt0" C_RST "         Read total frame counter\n"
           "    " C_CMD "@get s.name.summary" C_RST "      Read compact dimensions/dtype summary\n"
           "    " C_CMD "@set s.name.kw.KEY=val" C_RST "   Create or update a stream FITS keyword\n"
           "    " C_CMD "@waitfor name [N]" C_RST "        Block for N new frames (-t <sec>)\n"
           "    " C_CMD "@grab name local_im" C_RST
           "      Snapshot next SHM frame to local memory\n"
           "\n");

    printf(C_HDR "Stream Modifiers & Slicing in CLI\n" C_RST);
    printf("  Use prefix modifiers and bracket slices on stream names:\n"
           "    " C_STRM "@S:mystream" C_RST "          Force shared memory (/dev/shm)\n"
           "    " C_STRM "@L:mystream" C_RST "          Force session-local memory\n"
           "    " C_STRM "@E:mystream / @N:..." C_RST " Require existing / require new stream\n"
           "    " C_STRM "@F:mystream" C_RST "          Auto-load mystream.fits from disk\n"
           "    " C_STRM "@TF32:mystream" C_RST "       Explicit datatype (F32, F64, U16, etc.)\n"
           "    " C_STRM "mystream[0:10, 0:10]" C_RST " Sub-region / slice view (axes 0 and 1)\n"
           "\n");

    printf(C_NOTE "Refer to docs/guide/streams.md in the milk source tree for full details.\n" C_RST
                  "\n");

    return 0;
}
