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
    printf("  $ %s [%s]\n\n", progname, MH(MH_OPT, "options"));

    milk_help_section("Description", mh_color);
    printf("  Launches the interactive Terminal User Interface (TUI) for monitoring and\n"
           "  managing shared memory streams in real-time.\n\n");

    milk_help_section("Options", mh_color);
    printf("  %s %s                   Override SHM directory (default: /dev/shm)\n", MH(MH_OPT, "-d"), MH(MH_ARG, "DIR"));
    printf("  %s             Enable wave background animation\n", MH(MH_OPT, "-w, --wave"));
    printf("  %s             Show this help and exit\n", MH(MH_OPT, "-h, --help"));
    printf("  %s             One-line description and exit\n", MH(MH_OPT, "-h1, --help-oneline"));
    printf("  %s             Verbose description and exit\n", MH(MH_OPT, "-h2, --help-description"));
    printf("  %s             Full help, no ANSI color\n\n", MH(MH_OPT, "-hm, --help-mono"));

    milk_help_section("Interactive Commands", mh_color);
    printf("  %s                               Exit\n", MH(MH_CMD, "x"));
    printf("  %s                               Help screen\n", MH(MH_CMD, "h"));
    printf("  %s                           Switch tabs\n", MH(MH_CMD, "F2-F6"));
    printf("  %s                         Navigate streams\n", MH(MH_CMD, "UP/DOWN"));
    printf("  %s                          Erase selected stream\n", MH(MH_CMD, "CTRL+e"));
    printf("  %s                             Increase/decrease display rate\n", MH(MH_CMD, "+/-"));
    printf("  %s                             Decrease/increase scan rate\n\n", MH(MH_CMD, "{/}"));

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
        else
        {
            printf("\n\033[1;31mERROR\033[0m: Invalid option or argument '%s'.\n\n", argv[i]);
            print_help(progname, 1);
            return 1;
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
