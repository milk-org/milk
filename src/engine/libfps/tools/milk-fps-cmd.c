/**
 * @file milk-fps-cmd.c
 * @brief FPS lifecycle command dispatcher.
 *
 * Compiled 5x with different binary names:
 *   milk-fps-confstart, milk-fps-confstop, milk-fps-confstep,
 *   milk-fps-runstart, milk-fps-runstop.
 */

#include <libgen.h>

#include "fps_globals.h"

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
        return "Start the configuration loop for a Function Parameter Structure\n"
               "(FPS). The conf loop reads FPS parameters from shared memory and\n"
               "applies them to the running compute unit, allowing live parameter\n"
               "updates without restarting the process. Use -tmux to dispatch\n"
               "into the FPS tmux 'conf' window so it runs in the background.";
    }
    if (strcmp(prog, "milk-fps-confstop") == 0)
    {
        return "Stop the configuration loop for a Function Parameter Structure\n"
               "(FPS). Sends the confstop signal to the running compute unit.\n"
               "Use -tmux to dispatch the stop command via the tmux session.";
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
        return "Start the run loop for a Function Parameter Structure (FPS).\n"
               "The run loop executes the compute function continuously (or\n"
               "semaphore-triggered), using the current FPS parameters. Use\n"
               "-tmux to dispatch into the FPS tmux 'run' window so the loop\n"
               "runs in the background independently of the terminal.";
    }
    if (strcmp(prog, "milk-fps-runstop") == 0)
    {
        return "Stop the run loop for a Function Parameter Structure (FPS).\n"
               "Sends the runstop signal to the running compute unit, causing\n"
               "it to exit its main loop cleanly. Use -tmux to dispatch via\n"
               "the tmux session.";
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

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s-tmux%s] %s<fpsname>%s\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

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
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-tmux", mh_color ? MH_RST : "",
           "Dispatch via FPS tmux session (runs in background)");

    milk_help_section("Examples", mh_color);
    printf("  %s$ %s%s%s %smyfps00%s\n", mh_color ? MH_DIM : "", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ %s%s%s -tmux %smyfps00%s\n\n", mh_color ? MH_DIM : "", mh_color ? MH_CMD : "",
           progname, mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    const char *see_also[] = { "milk-fps-list:list active FPS instances",
                               "milk-fps-info:inspect FPS directory contents",
                               "milk-fps-set:set an FPS parameter value",
                               "milk-fpsCTRL:launch the FPS dashboard TUI",
                               "milk-fpsexec-help:print detailed FPS framework info" };
    milk_help_see_also(see_also, 5, mh_color);
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

    int         use_tmux = 0;
    const char *fpsname  = NULL;

    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-tmux") == 0)
        {
            use_tmux = 1;
        }
        else if (fpsname == NULL)
        {
            fpsname = argv[ii];
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

    FPS fps;
    fps.SMfd = -1;

    if (fps_connect(fpsname, &fps, 0) == -1)
    {
        fprintf(stderr, "Error: cannot connect to FPS \"%s\".\n", fpsname);
        return 1;
    }

    char *command = NULL;
    if (strcmp(progname, "milk-fps-confstart") == 0)
    {
        command = "confstart";
    }
    else if (strcmp(progname, "milk-fps-confstop") == 0)
    {
        command = "confstop";
    }
    else if (strcmp(progname, "milk-fps-runstart") == 0)
    {
        command = "runstart";
    }
    else if (strcmp(progname, "milk-fps-runstop") == 0)
    {
        command = "runstop";
    }
    else if (strcmp(progname, "milk-fps-confstep") == 0)
    {
        command = "confstep";
    }

    if (command == NULL)
    {
        fprintf(stderr, "Error: unknown command \"%s\".\n", progname);
        fps_disconnect(&fps);
        return 1;
    }

    if (strlen(fps.md->execfullpath) == 0 || strcmp(fps.md->execfullpath, "unknown") == 0)
    {
        fprintf(stderr, "Error: execfullpath not set for FPS \"%s\".\n", fpsname);
        fps_disconnect(&fps);
        return 1;
    }

    if (use_tmux)
    {
        functionparameter_FPS_tmux_ensure(&fps);

        char extra_args[256];
        snprintf(extra_args, sizeof(extra_args), " %s:%s", fpsname, command);

        if (functionparameter_FPS_tmux_send_dispatch(fpsname, command, fps.md->execfullpath,
                                                     extra_args) != 0)
        {
            fprintf(stderr,
                    "Warning: '%s' not recognized for tmux dispatch,"
                    " running locally...\n",
                    command);
            char cmdline[1024];
            snprintf(cmdline, sizeof(cmdline), "%s %s:%s", fps.md->execfullpath, fpsname, command);
            printf("Executing locally: %s\n", cmdline);
            int ret = system(cmdline);
            fps_disconnect(&fps);
            return WEXITSTATUS(ret);
        }
    }
    else
    {
        char cmdline[1024];
        snprintf(cmdline, sizeof(cmdline), "%s %s:%s", fps.md->execfullpath, fpsname, command);
        printf("Executing locally: %s\n", cmdline);
        int ret = system(cmdline);
        fps_disconnect(&fps);
        return WEXITSTATUS(ret);
    }

    fps_disconnect(&fps);
    return 0;
}
