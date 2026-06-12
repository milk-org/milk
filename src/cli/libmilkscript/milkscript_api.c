/**
 * @file milkscript_api.c
 * @brief Public API implementation for the minimal scripting engine
 *
 * Implements milkscript_init, milkscript_execute, milkscript_run,
 * and milkscript_cleanup. This layer acts as the foundation for
 * both the standalone milk-script engine and the interactive milk-cli,
 * with ZERO dependencies on readline or ncurses.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/types.h>

#include "CLIcore.h"
#include "milkscript.h"

#include "CLIcore_UI_execute.h"
#include "CLIcore_datainit.h"
#include "CLIcore_modules.h"
#include "CLIcore_setSHMdir.h"

/**
 * @brief Initialize the milk scripting engine.
 *
 * Sets up the interpreter state, loads modules,
 * and prepares the command dispatch table.
 */
errno_t milkscript_init(int argc, char **argv)
{
    if (argc > 0 && argv && argv[0])
    {
        strncpy(data.processname, argv[0], STRINGMAXLEN_PROCESSNAME - 1);
    }
    else
    {
        strncpy(data.processname, "milk-script", STRINGMAXLEN_PROCESSNAME - 1);
    }

    // Initialize struct data defaults specific to scripting
    data.CLIlogON             = 0;
    data.fifoON               = 0;
    data.fifofd               = -1;
    data.autocomplete         = 0;
    data.autocomplete_history = 0;
    data.autocomplete_arghint = 0;
    data.autocomplete_fuzzy   = 0;

    // Use libmilkdata globals via macros defined in CLIcore.h
    dcdebug     = 0;
    dcoverwrite = 0;
    dcprecision = 0;
    dcshareddft = 0;
    snprintf(dcsavedir, STRINGMAXLEN_DIRNAME, ".");

    dcprocinfo    = 1;
    dcprocinfoact = 0;

    // Setup SHM dir
    setSHMdir();

    // Initialize Data Arrays (Variables, FPS, etc) and RNG
    CLI_data_init();

    // Pre-load milkfpsCLI
    {
        load_sharedobj("libmilkfpsCLI.so");
    }

    // Explicitly call constructors of core dependencies
    load_module_shared("milkCOREMODmemory");
#ifdef USE_CFITSIO
    load_module_shared("milkCOREMODiofits");
#endif
    load_module_shared("milkCOREMODarith");
    load_module_shared("milkCOREMODtools");

    // Auto-load local modules if configured
    load_module_shared_local();

    return RETURN_SUCCESS;
}

/**
 * @brief Execute a single milk script command string.
 */
errno_t milkscript_execute(const char *cmdline)
{
    if (!cmdline)
    {
        return -1;
    }
    strncpy(data.CLIcmdline, cmdline, STRINGMAXLEN_CLICMDLINE - 1);
    return CLI_execute_line();
}

/**
 * @brief Run a milk script file.
 */
errno_t milkscript_run(FILE *fp)
{
    if (!fp)
    {
        return -1;
    }
    char line[STRINGMAXLEN_CLICMDLINE];
    data.CLIloopON = 1;

    while (data.CLIloopON && fgets(line, sizeof(line), fp))
    {
        /* Strip trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n')
        {
            line[--len] = '\0';
        }

        /* -E: echo each input line verbatim before executing.
         * Skip the shebang line (first line starting with #!). */
        if (data.echo_input && len > 0)
        {
            static int first_line = 1;
            int        is_shebang = (first_line && line[0] == '#' && line[1] == '!');
            first_line            = 0;

            if (!is_shebang)
            {
                printf("\033[2;36m%s\033[0m\n", line);
            }
        }

        /* Execute via engine */
        milkscript_execute(line);
    }

    return 0;
}

void milkscript_cleanup(void)
{
    // Minimal cleanup handled by OS upon process exit.
    // Persistent history/alias cleanup will be done by CLIcore interactive wrapper if needed.
}
