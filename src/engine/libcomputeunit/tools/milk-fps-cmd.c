// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-fps-cmd.c
 * @brief FPS lifecycle command dispatcher (single-unit direct or multi-unit parallel fan-out).
 *
 * Compiled 6x with different binary names:
 *   milk-fps-confstart, milk-fps-confstop, milk-fps-confstep,
 *   milk-fps-runstart, milk-fps-runstop, and milk-fps-run (for service managers, U17).
 */

#include <errno.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "computeunit.h"
#include "fps_globals.h"
#include "milk_spawn.h"

/* ---------------------------------------------------------
 * Per-variant descriptions
 * --------------------------------------------------------- */

static const char *cmd_desc(const char *prog)
{
    if (strcmp(prog, "milk-fps-confstart") == 0)
    {
        return "start FPS configuration loop";
    }
    if (strcmp(prog, "milk-fps-confstop") == 0)
    {
        return "stop FPS configuration loop";
    }
    if (strcmp(prog, "milk-fps-confstep") == 0)
    {
        return "run one FPS configuration step";
    }
    if (strcmp(prog, "milk-fps-runstart") == 0)
    {
        return "start FPS run loop";
    }
    if (strcmp(prog, "milk-fps-runstop") == 0)
    {
        return "stop FPS run loop";
    }
    if (strcmp(prog, "milk-fps-run") == 0)
    {
        return "run a unit in the foreground, for a service manager";
    }
    return "send lifecycle command to FPS";
} // cmd_desc

static const char *cmd_desc_long(const char *prog)
{
    if (strcmp(prog, "milk-fps-confstart") == 0)
    {
        return "Start the configuration loop of one or more units in the background, and\n"
               "return once all run. Multiple units are started concurrently. Exit status:\n"
               "0 running (or already running), 1 failed (the loop's messages are printed),\n"
               "124 not running yet after the timeout. Output goes to each unit's conf.out.";
    }
    if (strcmp(prog, "milk-fps-confstop") == 0)
    {
        return "Stop the configuration loop of one or more units concurrently, and return\n"
               "once all are gone.";
    }
    if (strcmp(prog, "milk-fps-confstep") == 0)
    {
        return "Run a single configuration step for one or more Function Parameter\n"
               "Structures (FPS). Applies and validates the current parameter values once\n"
               "without entering a continuous conf loop.";
    }
    if (strcmp(prog, "milk-fps-runstart") == 0)
    {
        return "Start the run loop of one or more units in the background, and return once\n"
               "active. Multiple units are started concurrently. Each unit's fpsexec runs\n"
               "detached with the environment captured at fpsinit; output goes to run.out\n"
               "(milk-fps-log). Exit status: 0 running, 1 failed, 124 timeout.";
    }
    if (strcmp(prog, "milk-fps-runstop") == 0)
    {
        return "Stop the run loop of one or more units concurrently, and return once all\n"
               "processes are gone: each loop is asked to exit, then signalled (SIGINT,\n"
               "SIGTERM, SIGKILL) if it does not.";
    }
    if (strcmp(prog, "milk-fps-run") == 0)
    {
        return "Check a unit (an fpsexec's FPS, owned by the caller, not running), then run\n"
               "its run (or conf) loop in the foreground: the ExecStart= of a systemd service\n"
               "(share/milk/systemd/milk-unit@.service). Under systemd the unit records its\n"
               "service, and milk-fps-runstart, milk-fps-runstop and the TUIs then go through\n"
               "systemctl; --release returns it to milk's own start and stop.";
    }
    return "Send a lifecycle command (confstart/confstop/confstep/runstart/\n"
           "runstop) to one or more Function Parameter Structures (FPS).";
} // cmd_desc_long

static const char *cmd_op(const char *prog)
{
    if (strncmp(prog, "milk-fps-", 9) == 0)
    {
        return prog + 9;
    }
    return NULL;
}

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, cmd_desc(progname), mh_color);

    int run = (strcmp(progname, "milk-fps-run") == 0);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s%s%s] %s<fpsname>%s%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", run ? "--release" : "--timeout S",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", run ? " [run|conf]" : " [<fpsname> ...]",
           mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", cmd_desc_long(progname));

    milk_help_section("Arguments", mh_color);
    printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_ARG, "<fpsname>", 25),
           run ? "Name of the target FPS" : "One or more target FPS names");

    milk_help_section("Options", mh_color);
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h, --help", 25), "Show this help and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h1, --help-oneline", 25),
           "Print one-line description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-h2, --help-description", 25),
           "Print verbose description and exit");
    printf("  " MH_PAD_FMT " %s\n", MH_PAD(MH_OPT, "-hm, --help-mono", 25),
           "Full help, no ANSI color");
    if (run)
    {
        printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "--release", 25),
               "Forget the systemd services that run the unit");
    }
    else
    {
        printf("  " MH_PAD_FMT " %s\n\n", MH_PAD(MH_OPT, "-t, --timeout S", 25),
               "Start: readiness timeout (MILK_UNIT_START_TIMEOUT, default 10)");
    }

    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s%s %smyfps00%s\n", mh_color ? MH_DIM : "", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    if (!run)
    {
        printf("  %s$ %s%s%s %swfs2cmval-7 mfilt-7 mvalC2dm-7%s\n", mh_color ? MH_DIM : "",
               mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
               mh_color ? MH_RST : "");
    }
    printf("\n");

    const char *see_also[] = { "milk-fps-list:list active FPS instances",
                               "milk-fps-log:show the output of a unit",
                               "milk-fps-wait:wait for FPS unit process completion",
                               "milk-fps-set:set FPS parameter values",
                               "milk-fpsCTRL:launch the FPS dashboard TUI",
                               "milk-fpsexec-help:print detailed FPS framework info" };
    milk_help_see_also(see_also, 6, mh_color);
} // print_help

static int run_single_confstep(const char *fpsname)
{
    FPS fps  = { 0 };
    fps.SMfd = -1;
    if (fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        fprintf(stderr, "Error: cannot connect to FPS \"%s\".\n", fpsname);
        return 1;
    }
    char exe[sizeof(fps.md->execfullpath)];
    char target[STRINGMAXLEN_FPS_NAME + 16];
    snprintf(exe, sizeof(exe), "%s", fps.md->execfullpath);
    fps_disconnect(&fps);
    if (exe[0] != '/')
    {
        fprintf(stderr, "Error: execfullpath not set for FPS \"%s\".\n", fpsname);
        return 1;
    }
    snprintf(target, sizeof(target), "%s:confstep", fpsname);
    const char *cargv[] = { exe, target, NULL };
    int         rc      = milkspawn_run(cargv, STDOUT_FILENO, STDERR_FILENO);
    if (rc < 0)
    {
        fprintf(stderr, "Error: cannot run %s: %s\n", exe, strerror(errno));
        return 1;
    }
    return rc;
} // run_single_confstep

static int run_multi_units(const char        *progname,
                           const char        *op,
                           const char *const *names,
                           int                nnames,
                           int                has_timeout,
                           double             timeout)
{
    if (has_timeout)
    {
        char tbuf[64];
        snprintf(tbuf, sizeof(tbuf), "%.6f", timeout);
        setenv("MILK_UNIT_START_TIMEOUT", tbuf, 1);
    }

    pid_t *pids = (pid_t *) calloc((size_t) nnames, sizeof(*pids));
    if (pids == NULL)
    {
        fprintf(stderr, "%s: out of memory\n", progname);
        return 1;
    }

    int hard_fail = 0;
    int timed_out = 0;

    for (int ii = 0; ii < nnames; ii++)
    {
        pids[ii] = fps_unit_spawn(names[ii], op, NULL, STDOUT_FILENO, STDERR_FILENO);
        if (pids[ii] < 0)
        {
            fprintf(stderr, "%s: cannot spawn '%s' for '%s': %s\n", progname, op, names[ii],
                    strerror(errno));
            hard_fail = 1;
        }
    }

    for (int ii = 0; ii < nnames; ii++)
    {
        if (pids[ii] <= 0)
        {
            continue;
        }
        int status = milkspawn_wait(pids[ii]);
        if (status == FPS_UNIT_START_TIMEOUT)
        {
            timed_out = 1;
        }
        else if (status != 0)
        {
            hard_fail = 1;
        }
    }

    free(pids);
    if (hard_fail)
    {
        return 1;
    }
    if (timed_out)
    {
        return FPS_UNIT_START_TIMEOUT;
    }
    return 0;
} // run_multi_units

int main(int argc, char *argv[])
{
    char       *progbuf  = strdup(argv[0]);
    const char *progname = (progbuf != NULL) ? basename(progbuf) : argv[0];

    int action = milk_help_init(argc, argv, cmd_desc(progname), cmd_desc_long(progname));
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        free(progbuf);
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        free(progbuf);
        return 0;
    }

    if (argc < 2)
    {
        fprintf(stderr, "%s: missing <fpsname>\n", progname);
        print_help(progname, 1);
        free(progbuf);
        return 2;
    }

    double       timeout     = 0.0;
    int          has_timeout = 0;
    const char  *role        = NULL;
    int          release     = 0;
    int          opts_done   = 0;
    int          run         = (strcmp(progname, "milk-fps-run") == 0);
    const char **names       = (const char **) calloc((size_t) argc, sizeof(*names));
    int          nnames      = 0;

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
        if (!opts_done && strcmp(arg, "-tmux") == 0)
        {
            fprintf(stderr, "Error: -tmux is gone: units start in the background (fpsexec run or "
                            "--fg keeps one in the foreground)\n");
            free(names);
            free(progbuf);
            return 2;
        }
        if (!opts_done && (strcmp(arg, "--timeout") == 0 || strcmp(arg, "-t") == 0) && !run)
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
            timeout     = val;
            has_timeout = 1;
            continue;
        }
        if (!opts_done && strcmp(arg, "--release") == 0 && run)
        {
            release = 1;
            continue;
        }
        if (!opts_done && arg[0] == '-')
        {
            fprintf(stderr, "%s: unknown option '%s'\n", progname, arg);
            free(names);
            free(progbuf);
            return 2;
        }

        if (run && nnames == 1 && role == NULL &&
            (strcmp(arg, "run") == 0 || strcmp(arg, "conf") == 0))
        {
            role = arg;
            continue;
        }
        if (run && nnames >= 1)
        {
            fprintf(stderr, "%s: unexpected argument '%s'\n", progname, arg);
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

    int rc = 1;
    if (run)
    {
        rc = release ? fps_unit_release(names[0])
                     : fps_unit_exec_service(names[0], (role != NULL && strcmp(role, "conf") == 0)
                                                           ? FPS_UNIT_CONF
                                                           : FPS_UNIT_RUN);
    }
    else if (nnames > 1)
    {
        const char *op = cmd_op(progname);
        if (op == NULL)
        {
            fprintf(stderr, "Error: unknown command \"%s\".\n", progname);
            rc = 1;
        }
        else
        {
            rc = run_multi_units(progname, op, names, nnames, has_timeout, timeout);
        }
    }
    else if (strcmp(progname, "milk-fps-confstart") == 0)
    {
        rc = fps_unit_start(names[0], FPS_UNIT_CONF, timeout, 0);
    }
    else if (strcmp(progname, "milk-fps-confstop") == 0)
    {
        rc = fps_unit_stop(names[0], FPS_UNIT_CONF, 0);
    }
    else if (strcmp(progname, "milk-fps-runstart") == 0)
    {
        rc = fps_unit_start(names[0], FPS_UNIT_RUN, timeout, 0);
    }
    else if (strcmp(progname, "milk-fps-runstop") == 0)
    {
        rc = fps_unit_stop(names[0], FPS_UNIT_RUN, 0);
    }
    else if (strcmp(progname, "milk-fps-confstep") == 0)
    {
        rc = run_single_confstep(names[0]);
    }
    else
    {
        fprintf(stderr, "Error: unknown command \"%s\".\n", progname);
        rc = 1;
    }

    free(names);
    free(progbuf);
    return rc;
} // main
