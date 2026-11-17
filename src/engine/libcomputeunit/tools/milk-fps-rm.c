// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <getopt.h>
#include <glob.h>
#include <termios.h>
#include <regex.h>

#include "libmilkcommon/multiselect_parse.h"

#include "computeunit.h"
#include "fps_FPSremove.h"

#define FR_DESC "remove Function Parameter Structure (FPS) from shared memory"
#define FR_DESC_LONG                                                      \
    "Remove one or more FPS instances from /dev/shm.\n"                   \
    "Without a name argument, scans for all FPS instances and presents\n" \
    "an interactive selection menu. With a name, removes the matching\n"  \
    "instance directly (use -r/--regex to match a regular expression).\n" \
    "Use -f to stop running processes before removal (otherwise\n"        \
    "running FPSes are left untouched). The unit directory and the\n"     \
    "processinfo files of the FPS go with it."

/**
 * @brief Print help message for milk-fps-rm.
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FR_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sfpsname%s ... | -r %sregex%s]\n\n", mh_color ? MH_CMD : "",
           progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FR_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-f, --force", mh_color ? MH_RST : "",
           "Stop running processes before removal");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-r, --regex", mh_color ? MH_RST : "",
           "Treat argument as an extended regular expression");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-x, --exact", mh_color ? MH_RST : "",
           "Arguments are exact FPS names (default)");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "--ignore-missing", mh_color ? MH_RST : "",
           "Succeed without error when named FPS does not exist");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-q, --quiet", mh_color ? MH_RST : "",
           "Suppress informational stdout messages");
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
 * remove_fps() - Remove a single FPS by name
 * @name:    FPS name
 * @verbose: extra output if true
 * @force:   stop running processes (fps_unit_stop()) instead of aborting
 *
 * Removes the FPS file, its unit directory, and the processinfo files of its dead processes.
 * Returns 0 on success.
 */
static int remove_fps(const char *name, int verbose, int quiet, int force)
{
    FPS_UNIT_STATUS st[2];
    int             running = 0;

    if (fps_unit_status(name, FPS_UNIT_RUN, &st[FPS_UNIT_RUN]) != 0)
    {
        PRINT_ERROR("Error: cannot connect to"
                    " FPS '%s'.",
                    name);
        return 1;
    }
    fps_unit_status(name, FPS_UNIT_CONF, &st[FPS_UNIT_CONF]);

    for (int r = FPS_UNIT_CONF; r >= FPS_UNIT_RUN; r--)
    {
        if (!fps_unit_state_alive(st[r].state))
        {
            continue;
        }
        if (!force)
        {
            PRINT_ERROR("%s process (PID %d) running for '%s'.",
                        fps_unit_role_name((FPS_UNIT_ROLE) r), (int) st[r].pid, name);
            running = 1;
        }
        else if (fps_unit_stop(name, (FPS_UNIT_ROLE) r, verbose ? 0 : FPS_UNIT_QUIET) != 0)
        {
            running = 1;
        }
    }

    if (running)
    {
        PRINT_ERROR("Abort: stop processes"
                    " before removing '%s' (or use -f/--force).",
                    name);
        return 1;
    }

    if (verbose)
    {
        printf("Removing FPS '%s'...\n", name);
    }

    /* under the unit lock, so that no start slips in between */
    int lockfd = fps_unit_lock(name, 1);
    FPS fps    = { 0 };
    fps.SMfd   = -1;
    if (lockfd < 0 || fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        PRINT_ERROR("Error: cannot connect to FPS '%s'.", name);
        fps_unit_unlock(lockfd);
        return 1;
    }
    for (int r = FPS_UNIT_RUN; r <= FPS_UNIT_CONF; r++)
    {
        fps_unit_status_fps(&fps, (FPS_UNIT_ROLE) r, &st[r], FPS_UNIT_LOCKED);
        if (fps_unit_state_alive(st[r].state))
        {
            PRINT_ERROR("%s process (PID %d) started again for '%s'.",
                        fps_unit_role_name((FPS_UNIT_ROLE) r), (int) st[r].pid, name);
            fps_disconnect(&fps);
            fps_unit_unlock(lockfd);
            return 1;
        }
    }
    fps_unit_procinfo_leftovers(name, 0);
    functionparameter_FPSremove(&fps);
    fps_disconnect(&fps);
    fps_unit_unlock(lockfd);
    if (!quiet)
    {
        printf("FPS '%s' removed.\n", name);
    }

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

    enum
    {
        OPT_IGNORE_MISSING = 1001
    };

    int verbose        = 0;
    int quiet          = 0;
    int force          = 0;
    int regex_mode     = 0;
    int ignore_missing = 0;
    int opt;

    static struct option long_options[] = { { "force", no_argument, 0, 'f' },
                                            { "regex", no_argument, 0, 'r' },
                                            { "exact", no_argument, 0, 'x' },
                                            { "ignore-missing", no_argument, 0,
                                              OPT_IGNORE_MISSING },
                                            { "quiet", no_argument, 0, 'q' },
                                            { "verbose", no_argument, 0, 'v' },
                                            { "help", no_argument, 0, 'h' },
                                            { "help-oneline", no_argument, 0, '1' },
                                            { 0, 0, 0, 0 } };

    while ((opt = getopt_long(argc, argv, "frxqvh1", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'f':
            force = 1;
            break;
        case 'r':
            regex_mode = 1;
            break;
        case 'x':
            regex_mode = 0;
            break;
        case OPT_IGNORE_MISSING:
            ignore_missing = 1;
            break;
        case 'q':
            quiet = 1;
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

    /* Fast path: exact FPS name(s) without scanning the entire SHM directory */
    if (!regex_mode && optind < argc)
    {
        char shmdir[200];
        function_parameter_struct_shmdirname(shmdir);

        int errors = 0;
        for (int arg_idx = optind; arg_idx < argc; arg_idx++)
        {
            const char *name = argv[arg_idx];
            if (name[0] == '\0' || strchr(name, '/') != NULL)
            {
                PRINT_ERROR("Error: invalid FPS name '%s'.", name);
                errors++;
                continue;
            }

            char shmpath[512];
            snprintf(shmpath, sizeof(shmpath), "%s/%s.fps.shm", shmdir, name);
            if (access(shmpath, F_OK) != 0)
            {
                if (ignore_missing)
                {
                    continue;
                }
                PRINT_ERROR("Error: cannot connect to FPS '%s'. It may not exist.", name);
                errors++;
                continue;
            }

            errors += remove_fps(name, verbose, quiet, force);
        } // for arg_idx

        return (errors > 0) ? 1 : 0;
    }

    const char *pattern = NULL;
    regex_t     regex;
    int         use_regex = 0;

    if (optind < argc)
    {
        pattern = argv[optind];
        if (regex_mode)
        {
            int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
            if (ret != 0)
            {
                char regerr[128];
                regerror(ret, &regex, regerr, sizeof(regerr));
                PRINT_ERROR("Invalid regex '%s': %s", pattern, regerr);
                return 1;
            }
            use_regex = 1;
        }
    }

    if (1) /* Scan directory for regex or interactive selection */
    {
        char shmdir[200];
        function_parameter_struct_shmdirname(shmdir);

        char pat[300];
        snprintf(pat, sizeof(pat), "%s/*.fps.shm", shmdir);

        glob_t gl;
        int    ret = glob(pat, 0, NULL, &gl);
        if (ret != 0 || gl.gl_pathc == 0)
        {
            if (pattern != NULL && !ignore_missing)
            {
                PRINT_ERROR("Error: cannot connect to FPS '%s'. It may not exist.", pattern);
            }
            else if (pattern == NULL && !quiet)
            {
                printf("No FPS instances found.\n");
            }
            if (ret == 0)
            {
                globfree(&gl);
            }
            if (use_regex)
            {
                regfree(&regex);
            }
            return (pattern != NULL && !ignore_missing) ? 1 : 0;
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
            if (pattern && !ignore_missing)
            {
                PRINT_ERROR("Error: cannot connect to FPS '%s'. It may not exist.", pattern);
            }
            else if (!pattern && !quiet)
            {
                printf("No FPS instances found.\n");
            }
            free(names);
            if (use_regex)
            {
                regfree(&regex);
            }
            return (pattern != NULL && ignore_missing) ? 0 : 1;
        }

        if (pattern != NULL && matched_count == 1)
        {
            /* Exactly one match for a CLI arg:
             * remove without interactive prompt */
            int rc = remove_fps(names[0], verbose, quiet, force);
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
            return (pattern != NULL || !is_tty) ? 1 : 0;
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
            return (pattern != NULL || !is_tty) ? 1 : 0;
        }

        int errors = 0;

        for (int ii = 0; ii < matched_count; ii++)
        {
            if (selected[ii])
            {
                errors += remove_fps(names[ii], verbose, quiet, force);
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
