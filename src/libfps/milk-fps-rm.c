#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <glob.h>
#include <termios.h>
#include <regex.h>

#include "fps.h"
#include "fps_FPSremove.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] [fpsname | regex pattern]\n", progname);
    printf("Remove a Function Parameter Structure (FPS).\n");
    printf("\n");
    printf("If no FPS name is given, lists existing FPS instances\n");
    printf("and prompts for selection. A regex can be provided to\n");
    printf("filter the list.\n");
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
    const char *fpsname = NULL;

    const char *pattern = NULL;
    regex_t regex;
    int use_regex = 0;

    if (optind < argc) {
        pattern = argv[optind];
        int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
        if (ret == 0) {
            use_regex = 1;
        } else if (verbose) {
            printf("Supplied argument could not be compiled as regex. Assuming exact literal.\n");
        }
    }

    if (1) { /* Always scan directory unless we specifically only want to target exactly one existing pattern but let's scan anyway */
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
        int count = (int)gl.gl_pathc;
        int matched_count = 0;
        char **names = calloc(count,
                              sizeof(char *));

        for (int i = 0; i < count; i++) {
            char *base =
                strrchr(gl.gl_pathv[i], '/');
            base = base ? base + 1 : gl.gl_pathv[i];
            /* strip .fps.shm suffix */
            char *dot = strstr(base, ".fps.shm");
            int len = dot ? (int)(dot - base)
                          : (int)strlen(base);
            
            char tmp_name[256];
            snprintf(tmp_name, sizeof(tmp_name), "%.*s", len, base);

            if (use_regex) {
                if (regexec(&regex, tmp_name, 0, NULL, 0) != 0) {
                    continue; // Skip if regex doesn't match
                }
            } else if (pattern != NULL) {
                // Not a valid regex, check exact match if possible
                if (strcmp(tmp_name, pattern) != 0) {
                    continue;
                }
            }
            
            names[matched_count] = strdup(tmp_name);
            matched_count++;
        }
        globfree(&gl);

        if (matched_count == 0) {
            if (pattern) {
                fprintf(stderr, "Error: cannot connect to FPS '%s'. It may not exist.\n", pattern);
            } else {
                printf("No FPS instances found.\n");
            }
            free(names);
            if(use_regex) regfree(&regex);
            return 1;
        }

        if (pattern != NULL && matched_count == 1) {
            // Exactly one match and an argument was passed, remove it without interactive prompt
            strncpy(fpsname_buf, names[0], sizeof(fpsname_buf) - 1);
            fpsname_buf[sizeof(fpsname_buf) - 1] = '\0';
            fpsname = fpsname_buf;
        } else {
            // Interactive Selection
            printf("\n  FPS instances:\n\n");
            for (int i = 0; i < matched_count; i++) {
                printf("  %3d  %s\n", i + 1, names[i]);
            }

            printf("\n  Enter number to remove"
                   " (0 to cancel): ");
        fflush(stdout);

        /* Ensure terminal is in canonical mode with CR->NL
         * translation so ENTER works correctly in all
         * contexts (tmux, pty, etc.).
         */
        struct termios old_term;
        int is_tty = isatty(STDIN_FILENO);
        if (is_tty) {
            tcgetattr(STDIN_FILENO, &old_term);
            struct termios t = old_term;
            t.c_lflag |= (ICANON | ECHO);
            t.c_iflag |= ICRNL;
            tcsetattr(STDIN_FILENO, TCSANOW, &t);
        }

        char linebuf[64];
        int fgets_ok =
            (fgets(linebuf, sizeof(linebuf),
                   stdin) != NULL);

        if (is_tty) {
            tcsetattr(STDIN_FILENO, TCSANOW,
                      &old_term);
        }

        if (!fgets_ok) {
            printf("Cancelled.\n");
            for (int i = 0; i < matched_count; i++) {
                free(names[i]);
            }
            free(names);
            if(use_regex) regfree(&regex);
            return 0;
        }

        /* Strip trailing \r and \n */
        {
            char *p = linebuf + strlen(linebuf);
            while (p > linebuf &&
                   (*(p-1) == '\n' || *(p-1) == '\r'))
            {
                *(--p) = '\0';
            }
        }

        int sel = atoi(linebuf);
        if (sel < 1 || sel > matched_count) {
            printf("Cancelled.\n");
            for (int i = 0; i < matched_count; i++) {
                free(names[i]);
            }
            free(names);
            if(use_regex) regfree(&regex);
            return 0;
        }

        strncpy(fpsname_buf, names[sel - 1],
                sizeof(fpsname_buf) - 1);
        fpsname_buf[sizeof(fpsname_buf) - 1] = '\0';
        fpsname = fpsname_buf;

        }

        for (int i = 0; i < matched_count; i++) {
            free(names[i]);
        }
        free(names);
    } 

    if (use_regex) regfree(&regex);

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
