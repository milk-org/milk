// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_config.h"
#include <getopt.h>
#include <CommandLineInterface/CLIcore.h>
#include <CommandLineInterface/CLIcore/CLIcore_datainit.h>
#include <CommandLineInterface/CLIcore/CLIcore_setSHMdir.h>
#include "termview.h"

int main(int argc, char *argv[])
{
    // Initialize data
    if (getenv("MILK_QUIET"))
    {
        data.quiet = 1;
    }
    else
    {
        data.quiet = 0;
    }

    strncpy(data.processname, "termview", STRINGMAXLEN_PROCESSNAME - 1);

    // Default options
    termview_options_t options;
    options.colormap     = COLORMAP_GREYSCALE;
    options.scale        = SCALE_LINEAR;
    options.range        = RANGE_MINMAX;
    options.range_locked = false;
    options.manual_min   = 0.0;
    options.manual_max   = 1.0;

    // Parse arguments
    static struct option long_options[] = { { "ascii", no_argument, 0, 'a' }, { 0, 0, 0, 0 } };

    int opt;
    int long_index = 0;
    while ((opt = getopt_long(argc, argv, "a", long_options, &long_index)) != -1)
    {
        switch (opt)
        {
        case 'a':
            options.colormap = COLORMAP_GREYSCALE;
            break;
        default:
            printf("Usage: %s [-a|--ascii] <image_name>\n", argv[0]);
            return 1;
        }
    }

    if (optind >= argc)
    {
        printf("Usage: %s [-a|--ascii] <image_name>\n", argv[0]);
        return 0;
    }

    // Core initialization
    CLI_startup();
    setSHMdir();
    CLI_data_init();

    // Run the tool
    termview_screen(argv[optind], options);

    return 0;
}
