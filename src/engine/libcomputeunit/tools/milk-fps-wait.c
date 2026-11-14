// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk-fps-wait.c
 * @brief   Wait for one or more FPS unit processes to finish
 */

#include <errno.h>
#include <libgen.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "computeunit.h"
#include "fps_unit_state.h"
#include "milk_help.h"
#include "milk_syscalls.h"

#define MFW_DESC "wait for FPS unit process completion"
#define MFW_DESC_LONG                                                                 \
    "Block until the run (or --conf) process of each named FPS unit is no longer\n"   \
    "running, using pidfd notifications without busy polling.\n\n"                    \
    "If a unit already finished cleanly before milk-fps-wait started, returns 0\n"    \
    "immediately. If any unit exited with a non-zero status or fatal signal,\n"       \
    "prints its failure state and exits with status 1.\n\n"                           \
    "Exit status: 0 all finished cleanly, 1 unit failed or missing, 2 usage error,\n" \
    "124 timeout, 130 interrupted (SIGINT/SIGTERM)."

static volatile sig_atomic_t g_interrupted = 0;

static void on_signal(int sig)
{
    (void) sig;
    g_interrupted = 1;
}

static double monotonic_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, MFW_DESC, mh_color);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s-t SEC%s] [%s--conf%s] [%s-q%s] %s<fpsname>%s [%s<fpsname> ...%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", MFW_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_ARG, "<fpsname>", 24),
           "One or more target FPS unit names");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-t, --timeout SEC", 24),
           "Overall timeout in seconds, 0 = infinite (default: 0)");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "--conf", 24),
           "Wait on the configuration loop instead of the run loop");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-q, --quiet", 24),
           "Suppress diagnostic messages on stderr");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h, --help", 24), "Show this help and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h1, --help-oneline", 24),
           "Print one-line description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h2, --help-description", 24),
           "Print verbose description and exit");
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "-hm, --help-mono", 24),
           "Full help without ANSI color");

    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s%s %smlat-7%s\n", mh_color ? MH_DIM : "", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ %s%s%s %s-t 60%s %smeaslinresp-7 compstrcm-7%s\n\n", mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    const char *see_also[] = { "milk-fps-runstart:start FPS run loop",
                               "milk-fps-runstop:stop FPS run loop",
                               "milk-fps-log:show the output of a unit",
                               "milk-stream-wait:wait for shared memory streams" };
    milk_help_see_also(see_also, 4, mh_color);
} // print_help

static int wait_one_unit(const char   *name,
                         FPS_UNIT_ROLE role,
                         double        t_start,
                         double        timeout_sec,
                         int           quiet,
                         const char   *prog)
{
    FPS_UNIT_STATUS st;

    if (fps_unit_status(name, role, &st) != 0)
    {
        if (!quiet)
        {
            fprintf(stderr, "%s: cannot connect to FPS '%s'\n", prog, name);
        }
        return 1;
    }

    while (fps_unit_state_alive(st.state))
    {
        if (g_interrupted)
        {
            return 130;
        }
        if (timeout_sec > 0.0 && monotonic_now() - t_start >= timeout_sec)
        {
            if (!quiet)
            {
                fprintf(stderr, "%s: timed out after %.3gs waiting for %s (%s)\n", prog,
                        timeout_sec, name, fps_unit_role_name(role));
            }
            return 124;
        }

        pid_t cur_pid = st.pid;
        int   pidfd   = (cur_pid > 0) ? milkos_pidfd_open(cur_pid) : -1;

        /* Re-check state after opening pidfd to avoid waiting on a stale handle */
        if (fps_unit_status(name, role, &st) != 0)
        {
            if (pidfd >= 0)
            {
                close(pidfd);
            }
            if (!quiet)
            {
                fprintf(stderr, "%s: FPS '%s' disappeared while waiting\n", prog, name);
            }
            return 1;
        }
        if (!fps_unit_state_alive(st.state) || st.pid != cur_pid)
        {
            if (pidfd >= 0)
            {
                close(pidfd);
            }
            continue;
        }

        int step_ms = 100;
        if (timeout_sec > 0.0)
        {
            double rem_ms = (timeout_sec - (monotonic_now() - t_start)) * 1000.0;
            if (rem_ms < 1.0)
            {
                rem_ms = 1.0;
            }
            if (rem_ms < (double) step_ms)
            {
                step_ms = (int) rem_ms + 1;
            }
        }

        if (pidfd >= 0)
        {
            milkos_pidfd_wait_exit(pidfd, step_ms);
            close(pidfd);
        }
        else
        {
            struct timespec slp = { 0, 10000000L };
            nanosleep(&slp, NULL);
        }

        if (fps_unit_status(name, role, &st) != 0)
        {
            if (!quiet)
            {
                fprintf(stderr, "%s: FPS '%s' disappeared while waiting\n", prog, name);
            }
            return 1;
        }
    } // while alive

    /* Synchronize with any concurrent runstop/confstop still holding the unit lock */
    if (st.state == FPS_UNIT_FAILED)
    {
        int lfd = fps_unit_lock(name, 1);
        fps_unit_unlock(lfd);
        fps_unit_status(name, role, &st);
    }

    if (st.state == FPS_UNIT_FAILED)
    {
        if (!quiet)
        {
            fprintf(stderr, "%s: %s (%s) failed: %s (see milk-fps-log %s)\n", prog, name,
                    fps_unit_role_name(role), st.last[0] ? st.last : "non-zero exit", name);
        }
        return 1;
    }
    return 0;
} // wait_one_unit

int main(int argc, char *argv[])
{
    char       *progbuf  = strdup(argv[0]);
    const char *progname = (progbuf != NULL) ? basename(progbuf) : argv[0];

    int action = milk_help_init(argc, argv, MFW_DESC, MFW_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        free(progbuf);
        return 0;
    }
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, action == MH_ACTION_HELP);
        free(progbuf);
        return 0;
    }

    double        timeout_sec = 0.0;
    FPS_UNIT_ROLE role        = FPS_UNIT_RUN;
    int           quiet       = 0;
    int           opts_done   = 0;

    const char **names  = (const char **) calloc((size_t) argc, sizeof(*names));
    int          nnames = 0;
    if (names == NULL)
    {
        fprintf(stderr, "%s: out of memory\n", progname);
        free(progbuf);
        return 1;
    }

    for (int ii = 1; ii < argc; ii++)
    {
        const char *arg = argv[ii];
        if (!opts_done && strcmp(arg, "--") == 0)
        {
            opts_done = 1;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-t") == 0 || strcmp(arg, "--timeout") == 0))
        {
            if (ii + 1 >= argc)
            {
                fprintf(stderr, "%s: option '%s' requires an argument\n", progname, arg);
                free(names);
                free(progbuf);
                return 2;
            }
            char  *end = NULL;
            double val = strtod(argv[++ii], &end);
            if (end == argv[ii] || *end != '\0' || val < 0.0)
            {
                fprintf(stderr, "%s: invalid timeout '%s'\n", progname, argv[ii]);
                free(names);
                free(progbuf);
                return 2;
            }
            timeout_sec = val;
            continue;
        }
        if (!opts_done && strcmp(arg, "--conf") == 0)
        {
            role = FPS_UNIT_CONF;
            continue;
        }
        if (!opts_done && (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0))
        {
            quiet = 1;
            continue;
        }
        if (!opts_done && arg[0] == '-')
        {
            fprintf(stderr, "%s: unknown option '%s'\n", progname, arg);
            free(names);
            free(progbuf);
            return 2;
        }
        names[nnames++] = arg;
    } // for ii

    if (nnames == 0)
    {
        fprintf(stderr, "%s: missing <fpsname>\n", progname);
        free(names);
        free(progbuf);
        return 2;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    double t_start   = monotonic_now();
    int    hard_fail = 0;
    int    final_rc  = 0;

    for (int ii = 0; ii < nnames; ii++)
    {
        int rc = wait_one_unit(names[ii], role, t_start, timeout_sec, quiet, progname);
        if (rc == 130 || rc == 124)
        {
            final_rc = rc;
            break;
        }
        if (rc != 0)
        {
            hard_fail = 1;
        }
    }

    if (final_rc == 0 && hard_fail)
    {
        final_rc = 1;
    }

    free(names);
    free(progbuf);
    return final_rc;
} // main
