#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "fpsCTRL_globals.h"

// Standalone main for milk-fpsCTRL

void print_usage(const char *progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Options:\n");
    printf("  -m, --match       Force match with fpscmd/fpslist.txt (default: 0)\n");
    printf("  -n, --name NAME   Filter FPS by name mask (default: \"_ALL\")\n");
    printf("  -f, --fifo FIFO   Input FIFO name (default: \"\")\n");
    printf("  -q, --quiet       Quiet mode (suppress TUI output)\n");
    printf("  -s, --stdio       Use stdio instead of ncurses\n");
    printf("  -h, --help        Show this help message\n");
    printf("\n");
    printf("Environment Variables:\n");
    printf("  MILK_FPS_LOGFILE            output logfile for milk-fpsCTRL\n");
    printf("  FPS_FILTSTRING_NAME         filter by name\n");
    printf("  FPS_FILTSTRING_KEYWORD      filter by keyword\n");
    printf("  FPS_FILTSTRING_CALLFUNC     filter by call function in source code\n");
    printf("  FPS_FILTSTRING_MODULE       filter by source code module\n");
}

int main(int argc, char *argv[]) {
    int opt;
    int matchmode = 0;
    char fpsnamemask[256] = "_ALL";
    char fifoname[512] = "";

    // Silence ImageStreamIO library (suppress stderr warnings/errors in TUI)
    ImageStreamIO_set_verbosity(0);

    // Allocate global fpsarray
    fpsarray = (FUNCTION_PARAMETER_STRUCT *) calloc(NB_FPS_MAX, sizeof(FUNCTION_PARAMETER_STRUCT));
    if(fpsarray == NULL) {
        fprintf(stderr, "Error: cannot allocate fpsarray\n");
        return 1;
    }
    for(int i=0; i<NB_FPS_MAX; i++) {
        fpsarray[i].SMfd = -1;
    }

    struct option long_options[] = {
        {"match", no_argument, 0, 'm'},
        {"name", required_argument, 0, 'n'},
        {"fifo", required_argument, 0, 'f'},
        {"quiet", no_argument, 0, 'q'},
        {"stdio", no_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "mn:f:qsh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm':
                matchmode = 1;
                break;
            case 'n':
                strncpy(fpsnamemask, optarg, sizeof(fpsnamemask) - 1);
                break;
            case 'f':
                strncpy(fifoname, optarg, sizeof(fifoname) - 1);
                break;
            case 'q':
                setenv("MILK_TUIPRINT_NONE", "1", 1);
                break;
            case 's':
                setenv("MILK_TUIPRINT_STDIO", "1", 1);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Handle positional argument for name mask
    if (optind < argc) {
        strncpy(fpsnamemask, argv[optind], sizeof(fpsnamemask) - 1);
    }

    // Call the main TUI function
    // functionparameter_CTRLscreen(uint32_t mode, char *fpsnamemask, char *fpsCTRLfifoname);
    functionparameter_CTRLscreen((uint32_t)matchmode, fpsnamemask, fifoname, 0.0);

    return 0;
}
