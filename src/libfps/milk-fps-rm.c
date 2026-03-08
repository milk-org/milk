#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <glob.h>
#include <termios.h>
#include <regex.h>

#include "CLIcore/multiselect_parse.h"

#include "fps.h"
#include "fps_FPSremove.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] [fpsname | regex pattern]\n", progname);
    printf("Remove a Function Parameter Structure (FPS).\n");
    printf("\n");
    printf("If no FPS name is given, lists existing FPS instances\n");
    printf("and prompts for selection. Multiple items can be\n");
    printf("selected using numbers, ranges, or 'all' (e.g. 1 3 5-7).\n");
    printf("A regex can be provided to filter the list.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode\n");
    printf("  -h, --help      Show this help message\n");
}

/**
 * remove_fps() - Remove a single FPS by name
 * @name:    FPS name
 * @verbose: extra output if true
 *
 * Connects, checks for running processes,
 * removes, disconnects.  Returns 0 on success.
 */
static int remove_fps(
    const char *name,
    int         verbose)
{
    FUNCTION_PARAMETER_STRUCT fps;

    fps.SMfd = -1;

    if (function_parameter_struct_connect(
            name, &fps, 0) == -1)
    {
        fprintf(stderr,
                "Error: cannot connect to"
                " FPS '%s'.\n", name);
        return 1;
    }

    int running = 0;

    if (fps.md->status
        & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
    {
        if (kill(fps.md->confpid, 0) == 0) {
            fprintf(stderr,
                    "Error: conf process"
                    " (PID %d) running"
                    " for '%s'.\n",
                    (int) fps.md->confpid,
                    name);
            running = 1;
        }
    }
    if (fps.md->status
        & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
    {
        if (kill(fps.md->runpid, 0) == 0) {
            fprintf(stderr,
                    "Error: run process"
                    " (PID %d) running"
                    " for '%s'.\n",
                    (int) fps.md->runpid,
                    name);
            running = 1;
        }
    }

    if (running) {
        fprintf(stderr,
                "Abort: stop processes"
                " before removing '%s'.\n",
                name);
        function_parameter_struct_disconnect(
            &fps);
        return 1;
    }

    if (verbose) {
        printf("Removing FPS '%s'...\n",
               name);
    }

    functionparameter_FPSremove(&fps);
    function_parameter_struct_disconnect(&fps);
    printf("FPS '%s' removed.\n", name);

    return 0;
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
            /* Exactly one match for a CLI arg:
             * remove without interactive prompt */
            int rc = remove_fps(
                names[0], verbose);
            for (int i = 0;
                 i < matched_count; i++)
            {
                free(names[i]);
            }
            free(names);
            if (use_regex) {
                regfree(&regex);
            }
            return rc;
        }

        /* Interactive multi-select */
        printf("\n  FPS instances:\n\n");
        for (int i = 0;
             i < matched_count; i++)
        {
            printf("  %3d  %s\n",
                   i + 1, names[i]);
        }

        printf("\n  Enter number(s) to remove"
               " (e.g. 1 3 5-7, 'all',"
               " 0 to cancel): ");
        fflush(stdout);

        struct termios old_term;
        int is_tty = isatty(STDIN_FILENO);

        if (is_tty) {
            tcgetattr(STDIN_FILENO,
                      &old_term);
            struct termios t = old_term;

            t.c_lflag |= (ICANON | ECHO);
            t.c_iflag |= ICRNL;
            tcsetattr(STDIN_FILENO,
                      TCSANOW, &t);
        }

        char linebuf[512];
        int fgets_ok =
            (fgets(linebuf, sizeof(linebuf),
                   stdin) != NULL);

        if (is_tty) {
            tcsetattr(STDIN_FILENO,
                      TCSANOW, &old_term);
        }

        if (!fgets_ok) {
            printf("Cancelled.\n");
            for (int i = 0;
                 i < matched_count; i++)
            {
                free(names[i]);
            }
            free(names);
            if (use_regex) {
                regfree(&regex);
            }
            return 0;
        }

        /* Strip trailing \r \n */
        {
            char *p =
                linebuf + strlen(linebuf);

            while (p > linebuf
                   && (*(p - 1) == '\n'
                       || *(p - 1) == '\r'))
            {
                *(--p) = '\0';
            }
        }

        int *selected =
            calloc(matched_count, sizeof(int));
        int nsel = parse_multiselect(
            linebuf, selected, matched_count);

        if (nsel <= 0) {
            printf("Cancelled.\n");
            free(selected);
            for (int i = 0;
                 i < matched_count; i++)
            {
                free(names[i]);
            }
            free(names);
            if (use_regex) {
                regfree(&regex);
            }
            return 0;
        }

        int errors = 0;

        for (int i = 0;
             i < matched_count; i++)
        {
            if (selected[i]) {
                errors += remove_fps(
                    names[i], verbose);
            }
        }

        free(selected);
        for (int i = 0;
             i < matched_count; i++)
        {
            free(names[i]);
        }
        free(names);

        if (use_regex) {
            regfree(&regex);
        }

        if (errors > 0) {
            fprintf(stderr,
                    "%d FPS(es) failed"
                    " to remove.\n",
                    errors);
            return 1;
        }
        return 0;
    }

    return 0;
}
