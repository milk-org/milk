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
#include <libgen.h>

#include "milk_help.h"

/**
 * @brief Print help message for milk-termview.
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, "TrueColor terminal image viewer", mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s<stream_name>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  A high-performance TrueColor terminal image viewer. It maps a shared memory\n"
           "  stream to ANSI terminal graphics.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");

    milk_help_section("Interactive Commands", mh_color);
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "Arrows", mh_color ? MH_RST : "",
           "Pan the image");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "+ / =", mh_color ? MH_RST : "", "Zoom in");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "- / _", mh_color ? MH_RST : "", "Zoom out");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "0", mh_color ? MH_RST : "",
           "Reset zoom and pan");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "< / ,", mh_color ? MH_RST : "",
           "Decrease framerate");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "> / .", mh_color ? MH_RST : "",
           "Increase framerate");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "c", mh_color ? MH_RST : "",
           "Cycle colormaps (Greyscale, Heat, Cold, Jet...)");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "s", mh_color ? MH_RST : "",
           "Cycle scaling (Linear, Sqrt, Log)");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "r", mh_color ? MH_RST : "",
           "Cycle dynamic ranges (MinMax, 1-99%, 5-95%...)");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "l", mh_color ? MH_RST : "",
           "Lock/Unlock current intensity range");
    printf("  %s%-12s%s %s\n\n", mh_color ? MH_CMD : "", "q", mh_color ? MH_RST : "", "Quit");
}

int main(int argc, char **argv)
{
    const char *progname = basename(argv[0]);

    int action = milk_help_init(
        argc, argv, "TrueColor terminal image viewer",
        "A high-performance TrueColor terminal image viewer. It maps a shared memory\n"
        "stream to ANSI terminal graphics.");
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

    if (argc < 2)
    {
        printf("\n\033[1;31mERROR\033[0m: Missing stream name.\n\n");
        print_help(progname, 1);
        return 1;
    }

    termview_options_t options;
    options.colormap     = COLORMAP_GREYSCALE;
    options.scale        = SCALE_LINEAR;
    options.range        = RANGE_MINMAX;
    options.range_locked = false;
    options.manual_min   = 0.0;
    options.manual_max   = 1.0;

    int ret = termview_screen(argv[1], options);

    return ret;
}
