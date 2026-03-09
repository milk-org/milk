/**
 * @file milk-fps-info.c
 * @brief Milk fps info module
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>

#include "fps.h"
#include "fps_globals.h"
#include "fps_scan.h"
#include "fps_printparameter_valuestring.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] <fpsname>\n", progname);
    printf("Print content of a Function Parameter Structure (FPS).\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode\n");
    printf("  -i, --info      Show detailed stream information on separate line\n");
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    int show_info = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument,       0, 'v'},
        {"info",    no_argument,       0, 'i'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "vih", long_options, NULL)) != -1) {
        switch (opt) {
            case 'v':
                verbose = 1;
                break;
            case 'i':
                show_info = 1;
                break;
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Error: missing FPS name.\n");
        print_help(argv[0]);
        return 1;
    }

    const char *fpsname = argv[optind];

    FUNCTION_PARAMETER_STRUCT fps;
    fps.SMfd = -1;

    if (function_parameter_struct_connect(fpsname, &fps, 0) == -1) {
        fprintf(stderr, "Error: cannot connect to FPS '%s'.\n", fpsname);
        return 1;
    }

    function_parameter_print_info(&fps, verbose, show_info);

    function_parameter_struct_disconnect(&fps);

    return 0;
}
