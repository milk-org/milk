/**
 * @file milk-termview.c
 * @brief Standalone executable for TrueColor terminal image viewer.
 *
 * This is the entry point for the milk-termview command line utility.
 * It does NOT link to CLIcore. It relies solely on ImageStreamIO and
 * standard POSIX libraries.
 */

#include "termview.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_help() {
    printf("Usage: milk-termview <stream_name>\n\n");
    printf("milk-termview is a high-performance TrueColor terminal image viewer.\n");
    printf("It maps a shared memory stream to ANSI terminal graphics.\n\n");
    printf("Interactive Controls:\n");
    printf("  Arrow Keys : Pan the image\n");
    printf("  + / =      : Zoom in\n");
    printf("  - / _      : Zoom out\n");
    printf("  0          : Reset zoom and pan\n");
    printf("  < / ,      : Decrease framerate\n");
    printf("  > / .      : Increase framerate\n");
    printf("  c          : Cycle colormaps (Greyscale, Heat, Cold, Jet, Inferno, Viridis)\n");
    printf("  s          : Cycle scaling (Linear, Sqrt, Log)\n");
    printf("  r          : Cycle dynamic ranges (MinMax, 1-99%%, 5-95%%, 10-90%%)\n");
    printf("  l          : Lock/Unlock current intensity range\n");
    printf("  q          : Quit\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Error: Missing stream name.\n");
        print_help();
        return 1;
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_help();
        return 0;
    }

    if (strcmp(argv[1], "-h1") == 0 || strcmp(argv[1], "--help-oneline") == 0) {
        printf("TrueColor terminal image viewer (standalone)\n");
        return 0;
    }

    termview_options_t options;
    options.colormap = COLORMAP_GREYSCALE;
    options.scale = SCALE_LINEAR;
    options.range = RANGE_MINMAX;
    options.range_locked = false;
    options.manual_min = 0.0;
    options.manual_max = 1.0;

    int ret = termview_screen(argv[1], options);
    
    return ret;
}
