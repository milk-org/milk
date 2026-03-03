#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>

#include "fps.h"
#include "fps_FPSremove.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] <fpsname>\n", progname);
    printf("Remove a Function Parameter Structure (FPS) and associated shared memory.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode\n");
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument,       0, 'v'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "vh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'v':
                verbose = 1;
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
        fprintf(stderr, "Error: cannot connect to FPS '%s'. It may not exist.\n", fpsname);
        return 1;
    }


    // Safety check: ensure no active processes are using this FPS
    int running = 0;
    if (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CONF) {
        if (kill(fps.md->confpid, 0) == 0) {
            fprintf(stderr, "Error: Configuration process (PID %d) is still running for FPS '%s'.\n", 
                    (int)fps.md->confpid, fpsname);
            running = 1;
        }
    }
    if (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_RUN) {
        if (kill(fps.md->runpid, 0) == 0) {
            fprintf(stderr, "Error: Run process (PID %d) is still running for FPS '%s'.\n", 
                    (int)fps.md->runpid, fpsname);
            running = 1;
        }
    }

    if (running) {
        fprintf(stderr, "Abort: Please stop these processes before removing the FPS.\n");
        function_parameter_struct_disconnect(&fps);
        return 1;
    }

    if (verbose) {
        printf("Removing FPS '%s'...\n", fpsname);
    }

    functionparameter_FPSremove(&fps);

    function_parameter_struct_disconnect(&fps);

    printf("FPS '%s' removed.\n", fpsname);

    return 0;
}
