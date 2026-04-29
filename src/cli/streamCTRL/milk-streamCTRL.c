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


int main(int argc, char *argv[])
{
    /* Parse optional -h1 / -h / --help */
    for(int i = 1; i < argc; i++)
    {
        if((strcmp(argv[i], "-h1") == 0) ||
                (strcmp(argv[i], "--help-oneline") == 0))
        {
            printf("interactive stream monitor TUI\n");
            return 0;
        }

        if((strcmp(argv[i], "-h") == 0) ||
                (strcmp(argv[i], "--help") == 0))
        {
            printf("Usage: milk-streamCTRL [options]\n");
            printf("  -h, --help     Show this help\n");
            printf("  -h1            One-line description and exit\n");
            printf("  -d DIR         Override SHM directory (default: /dev/shm)\n");
            printf("  -w, --wave     Enable wave background animation\n");
            printf("\nKeys:\n");
            printf("  x         Exit\n");
            printf("  h         Help screen\n");
            printf("  F2-F6     Switch tabs\n");
            printf("  UP/DOWN   Navigate streams\n");
            printf("  CTRL+e    Erase selected stream\n");
            printf("  +/-       Increase/decrease display rate\n");
            printf("  {/}       Decrease/increase scan rate\n");
            return 0;
        }

        if((strcmp(argv[i], "-w") == 0) ||
           (strcmp(argv[i], "--wave") == 0))
        {
            setenv("MILK_STREAMCTRL_WAVE", "1", 1);
        }

        if((strcmp(argv[i], "-d") == 0) && (i + 1 < argc))
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
