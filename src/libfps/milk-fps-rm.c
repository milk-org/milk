#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <glob.h>

#include "fps.h"
#include "fps_FPSremove.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] [fpsname]\n",
           progname);
    printf("Remove a Function Parameter Structure"
           " (FPS).\n\n");
    printf("If no FPS name is given, lists existing"
           " FPS instances\nand prompts for"
           " selection.\n\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode\n");
    printf("  -h, --help      Show this help\n");
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

    while ((opt = getopt_long(argc, argv,
                              "vh",
                              long_options,
                              NULL)) != -1)
    {
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

    char fpsname_buf[256];
    const char *fpsname;

    if (optind >= argc) {
        /* No name given — interactive selection */
        char shmdir[200];
        function_parameter_struct_shmdirname(
            shmdir);

        char pat[300];
        snprintf(pat, sizeof(pat),
                 "%s/*.fps.shm", shmdir);

        glob_t gl;
        int ret = glob(pat, 0, NULL, &gl);
        if (ret != 0 || gl.gl_pathc == 0) {
            printf("No FPS instances found.\n");
            if (ret == 0) {
                globfree(&gl);
            }
            return 0;
        }

        /* Build name list from filenames */
        int count = (int)gl.gl_pathc;
        char **names = calloc(count,
                              sizeof(char *));

        printf("\n  FPS instances:\n\n");
        for (int i = 0; i < count; i++) {
            char *base =
                strrchr(gl.gl_pathv[i], '/');
            base = base ? base + 1 : gl.gl_pathv[i];
            /* strip .fps.shm suffix */
            char *dot = strstr(base, ".fps.shm");
            int len = dot ? (int)(dot - base)
                          : (int)strlen(base);
            names[i] = strndup(base, len);
            printf("  %3d  %s\n", i + 1, names[i]);
        }
        globfree(&gl);

        printf("\n  Enter number to remove"
               " (0 to cancel): ");
        fflush(stdout);

        char linebuf[64];
        if (fgets(linebuf, sizeof(linebuf),
                  stdin) == NULL)
        {
            printf("Cancelled.\n");
            for (int i = 0; i < count; i++) {
                free(names[i]);
            }
            free(names);
            return 0;
        }

        int sel = atoi(linebuf);
        if (sel < 1 || sel > count) {
            printf("Cancelled.\n");
            for (int i = 0; i < count; i++) {
                free(names[i]);
            }
            free(names);
            return 0;
        }

        strncpy(fpsname_buf, names[sel - 1],
                sizeof(fpsname_buf) - 1);
        fpsname_buf[sizeof(fpsname_buf) - 1] = '\0';
        fpsname = fpsname_buf;

        for (int i = 0; i < count; i++) {
            free(names[i]);
        }
        free(names);
    } else {
        fpsname = argv[optind];
    }

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
