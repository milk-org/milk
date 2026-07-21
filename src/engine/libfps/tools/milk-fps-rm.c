// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <getopt.h>
#include <signal.h>
#include <glob.h>
#include <termios.h>
#include <regex.h>

#include "libmilkcommon/multiselect_parse.h"

#include "fps_FPSremove.h"

#define FR_DESC "remove Function Parameter Structure (FPS) from shared memory"
#define FR_DESC_LONG                                                      \
    "Remove one or more FPS instances from /dev/shm.\n"                   \
    "Without a name argument, scans for all FPS instances and presents\n" \
    "an interactive selection menu. With a name or regex, removes\n"      \
    "matching instances directly. Use -f to kill running processes\n"     \
    "before removal (otherwise running FPSes are left untouched)."

/**
 * @brief Print help message for milk-fps-rm.
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FR_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sfpsname%s | %sregex%s]\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FR_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-f, --force", mh_color ? MH_RST : "",
           "Kill running processes before removal");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-v, --verbose", mh_color ? MH_RST : "",
           "Verbose output");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-fps-rm%s              %s# interactive%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_DIM : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fps-rm%s %smyfps00%s\n", mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fps-rm%s -f %smyfps00%s\n\n", mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-fps-list:list active FPS instances",
                               "milk-fps-info:inspect FPS directory contents",
                               "milk-fps-confstop:stop an FPS configuration process",
                               "milk-fps-runstop:stop an FPS run process" };
    milk_help_see_also(see_also, 4, mh_color);
}

/**
 * kill_proc() - Terminate a process with SIGTERM->SIGKILL escalation
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
static int kill_proc(pid_t pid, const char *label, const char *name, int verbose)
{
    /*
     * An invalid or non-existent PID means there is nothing to
     * terminate; treat as success so removal can proceed.
     */
    if (pid <= 0 || getpgid(pid) < 0)
    {
        return 0;
    }

    if (verbose)
    {
        printf("Terminating %s process"
               " (PID %d) for '%s'...\n",
               label, (int) pid, name);
    }

    if (kill(pid, SIGTERM) == -1)
    {
        int saved_errno = errno;

        if (saved_errno == ESRCH)
        {
            return 0; /* already gone */
        }

        PRINT_ERROR("cannot send SIGTERM to %s (PID %d): %s", label, (int) pid,
                    strerror(saved_errno));
        return 1;
    }

    usleep(200000); /* 200 ms */

    {
        int rc        = kill(pid, 0);
        int chk_errno = errno;

        if (rc == -1 && chk_errno == ESRCH)
        {
            return 0; /* gone after SIGTERM */
        }
    }

    if (verbose)
    {
        printf("Process did not exit cleanly,"
               " sending SIGKILL to %s"
               " (PID %d)...\n",
               label, (int) pid);
    }

    if (kill(pid, SIGKILL) == -1)
    {
        int saved_errno = errno;

        if (saved_errno == ESRCH)
        {
            return 0; /* gone between checks */
        }

        PRINT_ERROR("cannot send SIGKILL to %s (PID %d): %s", label, (int) pid,
                    strerror(saved_errno));
        return 1;
    }

    usleep(50000); /* 50 ms */

    {
        int rc        = kill(pid, 0);
        int chk_errno = errno;

        if (rc == -1 && chk_errno == ESRCH)
        {
            return 0; /* confirmed gone */
        }

        /* rc == 0 (alive) or rc == -1 with EPERM (still exists) */
        PRINT_ERROR("Error: %s (PID %d) still"
                    " running after SIGKILL.",
                    label, (int) pid);
        return 1;
    }
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
static int remove_fps(const char *name, int verbose, int force)
{
    FPS fps;

    fps.SMfd = -1;

    if (fps_connect(name, &fps, 0) == -1)
    {
        PRINT_ERROR("Error: cannot connect to"
                    " FPS '%s'.",
                    name);
        return 1;
    }

    int running = 0;

    if (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
    {
        pid_t cpid = (pid_t) fps.md->confpid;
        if (cpid > 0 && getpgid(cpid) >= 0)
        {
            if (force)
            {
                if (kill_proc(cpid, "conf", name, verbose) != 0)
                {
                    running = 1;
                }
            }
            else
            {
                PRINT_ERROR("conf process (PID %d) running for '%s'.", (int) cpid, name);
                running = 1;
            }
        }
    }

    if (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
    {
        pid_t rpid = (pid_t) fps.md->runpid;
        if (rpid > 0 && getpgid(rpid) >= 0)
        {
            if (force)
            {
                if (kill_proc(rpid, "run", name, verbose) != 0)
                {
                    running = 1;
                }
            }
            else
            {
                PRINT_ERROR("run process (PID %d) running for '%s'.", (int) rpid, name);
                running = 1;
            }
        }
    }

    if (running)
    {
        PRINT_ERROR("Abort: stop processes"
                    " before removing '%s' (or use -f/--force).",
                    name);
        fps_disconnect(&fps);
        return 1;
    }

    if (verbose)
    {
        printf("Removing FPS '%s'...\n", name);
    }

    functionparameter_FPSremove(&fps);
    fps_disconnect(&fps);
    printf("FPS '%s' removed.\n", name);

    return 0;
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv, FR_DESC, FR_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int verbose = 0;
    int force   = 0;
    int opt;

    static struct option long_options[] = { { "force", no_argument, 0, 'f' },
                                            { "verbose", no_argument, 0, 'v' },
                                            { "help", no_argument, 0, 'h' },
                                            { "help-oneline", no_argument, 0, '1' },
                                            { 0, 0, 0, 0 } };

    while ((opt = getopt_long(argc, argv, "fvh1", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'f':
            force = 1;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'h':
        case '1':
            break; /* handled above */
        default:
            printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    const char *pattern = NULL;
    regex_t     regex;
    int         use_regex = 0;

    if (optind < argc)
    {
        pattern = argv[optind];
        int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
        if (ret == 0)
        {
            use_regex = 1;
        }
        else if (verbose)
        {
            printf("Supplied argument could not be compiled as regex. Assuming exact literal.\n");
        }
    }

    if (1) /* Always scan directory unless we specifically only want to target exactly one existing pattern but let's scan anyway */
    {
        char shmdir[200];
        function_parameter_struct_shmdirname(shmdir);

        char pat[300];
        snprintf(pat, sizeof(pat), "%s/*.fps.shm", shmdir);

        glob_t gl;
        int    ret = glob(pat, 0, NULL, &gl);
        if (ret != 0 || gl.gl_pathc == 0)
        {
            printf("No FPS instances found.\n");
            if (ret == 0)
            {
                globfree(&gl);
            }
            return 0;
        }
        int    count         = (int) gl.gl_pathc;
        int    matched_count = 0;
        char **names         = calloc(count, sizeof(char *));

        for (int ii = 0; ii < count; ii++)
        {
            char *base = strrchr(gl.gl_pathv[ii], '/');
            base       = base ? base + 1 : gl.gl_pathv[ii];
            /* strip .fps.shm suffix */
            char *dot = strstr(base, ".fps.shm");
            int   len = dot ? (int) (dot - base) : (int) strlen(base);

            char tmp_name[256];
            snprintf(tmp_name, sizeof(tmp_name), "%.*s", len, base);

            if (use_regex)
            {
                if (regexec(&regex, tmp_name, 0, NULL, 0) != 0)
                {
                    continue; // Skip if regex doesn't match
                }
            }
            else if (pattern != NULL)
            {
                // Not a valid regex, check exact match if possible
                if (strcmp(tmp_name, pattern) != 0)
                {
                    continue;
                }
            }

            names[matched_count] = strdup(tmp_name);
            matched_count++;
        }
        globfree(&gl);

        if (matched_count == 0)
        {
            if (pattern)
            {
                PRINT_ERROR("Error: cannot connect to FPS '%s'. It may not exist.", pattern);
            }
            else
            {
                printf("No FPS instances found.\n");
            }
            free(names);
            if (use_regex)
            {
                regfree(&regex);
            }
            return 1;
        }

        if (pattern != NULL && matched_count == 1)
        {
            /* Exactly one match for a CLI arg:
             * remove without interactive prompt */
            int rc = remove_fps(names[0], verbose, force);
            for (int ii = 0; ii < matched_count; ii++)
            {
                free(names[ii]);
            }
            free(names);
            if (use_regex)
            {
                regfree(&regex);
            }
            return rc;
        }

        /* Interactive multi-select */
        printf("\n  FPS instances:\n\n");
        for (int ii = 0; ii < matched_count; ii++)
        {
            printf("  %3d  %s\n", ii + 1, names[ii]);
        }

        printf("\n  Enter number(s) to remove"
               " (e.g. 1 3 5-7, 'all',"
               " 0 to cancel): ");
        fflush(stdout);

        struct termios old_term;
        int            is_tty = isatty(STDIN_FILENO);

        if (is_tty)
        {
            tcgetattr(STDIN_FILENO, &old_term);
            struct termios t = old_term;

            t.c_lflag |= (ICANON | ECHO);
            t.c_iflag |= ICRNL;
            tcsetattr(STDIN_FILENO, TCSANOW, &t);
        }

        char linebuf[512];
        int  fgets_ok = (fgets(linebuf, sizeof(linebuf), stdin) != NULL);

        if (is_tty)
        {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
        }

        if (!fgets_ok)
        {
            printf("Cancelled.\n");
            for (int ii = 0; ii < matched_count; ii++)
            {
                free(names[ii]);
            }
            free(names);
            if (use_regex)
            {
                regfree(&regex);
            }
            return 0;
        }

        /* Strip trailing \r \n */
        {
            char *p = linebuf + strlen(linebuf);

            while (p > linebuf && (*(p - 1) == '\n' || *(p - 1) == '\r'))
            {
                *(--p) = '\0';
            }
        }

        int *selected = calloc(matched_count, sizeof(int));
        int  nsel     = parse_multiselect(linebuf, selected, matched_count);

        if (nsel <= 0)
        {
            printf("Cancelled.\n");
            free(selected);
            for (int ii = 0; ii < matched_count; ii++)
            {
                free(names[ii]);
            }
            free(names);
            if (use_regex)
            {
                regfree(&regex);
            }
            return 0;
        }

        int errors = 0;

        for (int ii = 0; ii < matched_count; ii++)
        {
            if (selected[ii])
            {
                errors += remove_fps(names[ii], verbose, force);
            }
        }

        free(selected);
        for (int ii = 0; ii < matched_count; ii++)
        {
            free(names[ii]);
        }
        free(names);

        if (use_regex)
        {
            regfree(&regex);
        }

        if (errors > 0)
        {
            PRINT_ERROR("%d FPS(es) failed"
                        " to remove.",
                        errors);
            return 1;
        }
        return 0;
    }

    return 0;
}
