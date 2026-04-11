#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <glob.h>
#include <termios.h>
#include <regex.h>

#include "libmilkcommon/multiselect_parse.h"

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
    printf("  -f, --force     Force removal by killing running processes\n");
    printf("  -v, --verbose   Verbose mode\n");
    printf("  -h, --help      Show this help message\n");
}

/**
 * kill_proc() - Terminate a process with SIGTERM→SIGKILL escalation
 * @pid:     Process ID to terminate; must be > 0
 * @label:   Human-readable label used in diagnostic messages
 * @name:    FPS name used in diagnostic messages
 * @verbose: Print progress messages when non-zero
 *
 * Validates that @pid is positive and the process group exists
 * before signaling.  Sends SIGTERM, waits 200 ms, then escalates
 * to SIGKILL if the process is still alive.  Waits an additional
 * 50 ms after SIGKILL and confirms the process has exited.
 *
 * Returns 0 if the process is gone, 1 on error or if the process
 * is still running after SIGKILL.
 */
static int kill_proc(
    pid_t       pid,
    const char *label,
    const char *name,
    int         verbose)
{
    if (pid <= 0 || getpgid(pid) < 0)
        return 0;

    if (verbose)
        printf("Terminating %s process"
               " (PID %d) for '%s'...\n",
               label, (int)pid, name);

    if (kill(pid, SIGTERM) == -1)
    {
        int saved_errno = errno;

        if (saved_errno == ESRCH)
            return 0; /* already gone */

        fprintf(stderr,
                "Error: cannot send SIGTERM"
                " to %s (PID %d): %s\n",
                label, (int)pid,
                strerror(saved_errno));
        return 1;
    }

    usleep(200000); /* 200 ms */

    if (kill(pid, 0) != 0)
        return 0; /* gone after SIGTERM */

    if (verbose)
        printf("Process did not exit cleanly,"
               " sending SIGKILL to %s"
               " (PID %d)...\n",
               label, (int)pid);

    if (kill(pid, SIGKILL) == -1)
    {
        int saved_errno = errno;

        if (saved_errno == ESRCH)
            return 0; /* gone between checks */

        fprintf(stderr,
                "Error: cannot send SIGKILL"
                " to %s (PID %d): %s\n",
                label, (int)pid,
                strerror(saved_errno));
        return 1;
    }

    usleep(50000); /* 50 ms */

    if (kill(pid, 0) == 0)
    {
        fprintf(stderr,
                "Error: %s (PID %d) still"
                " running after SIGKILL.\n",
                label, (int)pid);
        return 1;
    }

    return 0;
}

/**
 * remove_fps() - Remove a single FPS by name
 * @name:    FPS name
 * @verbose: extra output if true
 * @force:   kill running processes instead of aborting
 *
 * Connects, checks for running processes,
 * removes, disconnects.  Returns 0 on success.
 */
static int remove_fps(
    const char *name,
    int         verbose,
    int         force)
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
        pid_t cpid = fps.md->confpid;

        if (cpid > 0 && getpgid(cpid) >= 0
            && kill(cpid, 0) == 0)
        {
            if (force)
            {
                if (kill_proc(cpid, "conf",
                              name, verbose))
                    running = 1;
            }
            else
            {
                fprintf(stderr,
                        "Error: conf process"
                        " (PID %d) running"
                        " for '%s'.\n",
                        (int)cpid, name);
                running = 1;
            }
        }
    }

    if (fps.md->status
        & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
    {
        pid_t rpid = fps.md->runpid;

        if (rpid > 0 && getpgid(rpid) >= 0
            && kill(rpid, 0) == 0)
        {
            if (force)
            {
                if (kill_proc(rpid, "run",
                              name, verbose))
                    running = 1;
            }
            else
            {
                fprintf(stderr,
                        "Error: run process"
                        " (PID %d) running"
                        " for '%s'.\n",
                        (int)rpid, name);
                running = 1;
            }
        }
    }

    if (running) {
        fprintf(stderr,
                "Abort: stop processes"
                " before removing '%s' (or use -f/--force).\n",
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
    int force = 0;
    int opt;

    static struct option long_options[] = {
        {"force",   no_argument,       0, 'f'},
        {"verbose", no_argument,       0, 'v'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv,
                              "fvh",
                              long_options,
                              NULL)) != -1)
    {
        switch (opt) {
            case 'f':
                force = 1;
                break;
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
                names[0], verbose, force);
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
                    names[i], verbose, force);
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
