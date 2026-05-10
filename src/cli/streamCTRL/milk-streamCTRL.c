/**
 * @file milk-streamCTRL-standalone.c
 * @brief Standalone entry point for milk-streamCTRL (no CLIcore, no ncurses)
 *
 * Links: ImageStreamIO + libprocessinfo + libm + libpthread + librt
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <libgen.h>

#include "milk_help.h"
#include "streamCTRL_defs.h"
#include "streamCTRL_ansi.h"
#include "streamCTRL_TUI.h"

/* Global signal flags referenced by streamCTRL_TUI.c */
volatile sig_atomic_t sc_sigINT  = 0;
volatile sig_atomic_t sc_sigTERM = 0;

static void handle_sigint(int sig)
{
    (void) sig;
    sc_sigINT = 1;
}

static void handle_sigterm(int sig)
{
    (void) sig;
    sc_sigTERM = 1;
}


static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, "interactive stream monitor TUI", mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  Launches the interactive Terminal User Interface (TUI) for monitoring and\n"
           "  managing shared memory streams in real-time.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-d DIR",
           mh_color ? MH_RST : "", "Override SHM directory (default: /dev/shm)");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-w, --wave",
           mh_color ? MH_RST : "", "Enable wave background animation");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");

    milk_help_section("Interactive Commands", mh_color);
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "x", mh_color ? MH_RST : "", "Exit");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "h", mh_color ? MH_RST : "", "Help screen");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "F2-F6", mh_color ? MH_RST : "", "Switch tabs");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "UP/DOWN", mh_color ? MH_RST : "", "Navigate streams");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "CTRL+e", mh_color ? MH_RST : "", "Erase selected stream");
    printf("  %s%-12s%s %s\n", mh_color ? MH_CMD : "", "+/-", mh_color ? MH_RST : "", "Increase/decrease display rate");
    printf("  %s%-12s%s %s\n\n", mh_color ? MH_CMD : "", "{/}", mh_color ? MH_RST : "", "Decrease/increase scan rate");

    const char *see_also[] = {
        "milk-stream-info", "milk-stream-list", "milk-streamCTRL-cli"
    };
    milk_help_see_also(see_also, 3, mh_color);
}


int main(int argc, char *argv[])
{
    const char *progname = basename(argv[0]);

    int action = milk_help_init(argc, argv,
                                "interactive stream monitor TUI",
                                "Launches the interactive Terminal User Interface (TUI) for monitoring and\n"
                                "managing shared memory streams in real-time.");
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

    /* Parse options */
    for(int i = 1; i < argc; i++)
    {
        if((strcmp(argv[i], "-w") == 0) ||
           (strcmp(argv[i], "--wave") == 0))
        {
            setenv("MILK_STREAMCTRL_WAVE", "1", 1);
        }
        else if((strcmp(argv[i], "-d") == 0) && (i + 1 < argc))
        {
            setenv("MILK_SHM_DIR", argv[++i], 1);
        }
    }

    /* Install signal handlers */
    signal(SIGINT,  handle_sigint);
    signal(SIGTERM, handle_sigterm);

    /* Enter raw terminal mode */
    ansi_raw_mode_enter();

    /* Run the TUI */
    streamCTRL_CTRLscreen();

    /* Restore terminal */
    ansi_raw_mode_exit();

    return 0;
}
