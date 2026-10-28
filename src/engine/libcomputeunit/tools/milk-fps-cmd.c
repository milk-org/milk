// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-fps-cmd.c
 * @brief FPS lifecycle command dispatcher.
 *
 * Compiled 6x with different binary names:
 *   milk-fps-confstart, milk-fps-confstop, milk-fps-confstep,
 *   milk-fps-runstart, milk-fps-runstop, and milk-fps-run (for service managers, U17).
 */

#include <libgen.h>

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
}

/**
 * @brief Print the extended command description.
 *
 * Called by -h2 to display detailed usage,
 * examples, and environment variables.
 */
static const char *cmd_desc_long(const char *prog)
{
    if (strcmp(prog, "milk-fps-confstart") == 0)
    {
        return "Start the configuration loop of a unit in the background, and return once\n"
               "it runs. The conf loop checks the FPS parameters as they change. Exit status:\n"
               "0 running (or already running), 1 failed (the loop's messages are printed),\n"
               "124 not running yet after the timeout. Its output goes to the unit's conf.out.";
    }
    if (strcmp(prog, "milk-fps-confstop") == 0)
    {
        return "Stop the configuration loop of a unit, and return once it is gone.";
    }
    if (strcmp(prog, "milk-fps-confstep") == 0)
    {
        return "Run a single configuration step for a Function Parameter\n"
               "Structure (FPS). Applies the current parameter values once\n"
               "without entering a continuous conf loop. Useful for one-shot\n"
               "parameter updates during testing or scripted calibration.";
    }
    if (strcmp(prog, "milk-fps-runstart") == 0)
    {
        return "Start the run loop of a unit in the background, and return once the loop is\n"
               "active. The unit's fpsexec runs detached, with the environment captured at\n"
               "fpsinit; its output goes to the unit's run.out (milk-fps-log). Exit status:\n"
               "0 running (or already running), 1 failed (the worker's messages are printed),\n"
               "124 not active yet after the timeout.";
    }
    if (strcmp(prog, "milk-fps-runstop") == 0)
    {
        return "Stop the run loop of a unit, and return once the process is gone: the loop\n"
               "is asked to exit, then signalled (SIGINT, SIGTERM, SIGKILL) if it does not.";
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
           "runstop) to a Function Parameter Structure (FPS).";
}

/* ---------------------------------------------------------
 * print_help() - Full formatted help output.
 * @progname:  Executable name.
 * @mh_color:  Non-zero to emit ANSI color.
 * --------------------------------------------------------- */

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, cmd_desc(progname), mh_color);

    int run = (strcmp(progname, "milk-fps-run") == 0);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s%s%s] %s<fpsname>%s%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", run ? "--release" : "--timeout S",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", run ? " [run|conf]" : "",
           mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", cmd_desc_long(progname));

    milk_help_section("Arguments", mh_color);
    printf("  %s%-14s%s %s\n\n", mh_color ? MH_ARG : "", "<fpsname>", mh_color ? MH_RST : "",
           "Name of the target FPS");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "Print one-line description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Print verbose description and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    if (run)
    {
        printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "--release", mh_color ? MH_RST : "",
               "Forget the systemd services that run the unit");
    }
    else
    {
        printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "--timeout S", mh_color ? MH_RST : "",
               "Start: readiness timeout (MILK_UNIT_START_TIMEOUT, default 10)");
    }

    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s%s %smyfps00%s\n\n", mh_color ? MH_DIM : "", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    const char *see_also[] = { "milk-fps-list:list active FPS instances",
                               "milk-fps-log:show the output of a unit",
                               "milk-fps-info:inspect FPS directory contents",
                               "milk-fps-set:set an FPS parameter value",
                               "milk-fpsCTRL:launch the FPS dashboard TUI",
                               "milk-fpsexec-help:print detailed FPS framework info" };
    milk_help_see_also(see_also, 6, mh_color);
}

/* ---------------------------------------------------------
 * main()
 * --------------------------------------------------------- */

int main(int argc, char *argv[])
{
    const char *progname = basename(argv[0]);

    int action = milk_help_init(argc, argv, cmd_desc(progname), cmd_desc_long(progname));
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);

    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(progname, mh_color);
        return 0;
    }

    if (argc < 2)
    {
        printf("\n\033[1;31mERROR\033[0m missing <fpsname>\n\n");
        print_help(progname, 1);
        return 1;
    }

    double      timeout = 0.0;
    const char *fpsname = NULL;
    const char *role    = NULL;
    int         release = 0;
    int         run     = (strcmp(progname, "milk-fps-run") == 0);

    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-tmux") == 0)
        {
            fprintf(stderr, "Error: -tmux is gone: units start in the background (fpsexec run or "
                            "--fg keeps one in the foreground)\n");
            return 2;
        }
        else if (strcmp(argv[ii], "--timeout") == 0 && ii + 1 < argc && !run)
        {
            timeout = atof(argv[++ii]);
        }
        else if (strcmp(argv[ii], "--release") == 0 && run)
        {
            release = 1;
        }
        else if (fpsname == NULL)
        {
            fpsname = argv[ii];
        }
        else if (run && role == NULL &&
                 (strcmp(argv[ii], "run") == 0 || strcmp(argv[ii], "conf") == 0))
        {
            role = argv[ii];
        }
        else
        {
            printf("\n\033[1;31mERROR\033[0m unexpected argument '%s'\n\n", argv[ii]);
            print_help(progname, 1);
            return 1;
        }
    } // for ii

    if (fpsname == NULL)
    {
        printf("\n\033[1;31mERROR\033[0m missing <fpsname>\n\n");
        print_help(progname, 1);
        return 1;
    }

    if (run)
    {
        return release ? fps_unit_release(fpsname)
                       : fps_unit_exec_service(fpsname, (role != NULL && strcmp(role, "conf") == 0)
                                                            ? FPS_UNIT_CONF
                                                            : FPS_UNIT_RUN);
    }
    if (strcmp(progname, "milk-fps-confstart") == 0)
    {
        return fps_unit_start(fpsname, FPS_UNIT_CONF, timeout, 0);
    }
    if (strcmp(progname, "milk-fps-confstop") == 0)
    {
        return fps_unit_stop(fpsname, FPS_UNIT_CONF, 0);
    }
    if (strcmp(progname, "milk-fps-runstart") == 0)
    {
        return fps_unit_start(fpsname, FPS_UNIT_RUN, timeout, 0);
    }
    if (strcmp(progname, "milk-fps-runstop") == 0)
    {
        return fps_unit_stop(fpsname, FPS_UNIT_RUN, 0);
    }
    if (strcmp(progname, "milk-fps-confstep") != 0)
    {
        fprintf(stderr, "Error: unknown command \"%s\".\n", progname);
        return 1;
    }

    /* confstep: one step by the unit's own fpsexec, in the foreground */
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
}
