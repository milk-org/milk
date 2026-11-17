// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkscript_main.c
 * @brief Standalone interpreter executable for milk scripts
 *
 * This binary is the non-interactive interpreter invoked either
 * directly or via a shebang line:
 *
 *   #!/usr/bin/env milk-script
 *   #!/usr/bin/env -S milk-script -e
 *
 * (The env -S flag is required on Linux when passing options in
 * a shebang line, as the kernel passes everything after the
 * interpreter name as a single token otherwise.)
 *
 * SYNOPSIS
 *   milk-script [OPTIONS] [SCRIPT_FILE [ARG...]]
 *   echo "cmds" | milk-script [OPTIONS]
 *
 * OPTIONS
 *   -h, --help       Print this help and exit
 *   -e, --errexit    Exit immediately if a command fails (set -e)
 *   -x, --xtrace     Print each command before executing (set -x)
 *   -E, --echo       Echo each input line verbatim (colored)
 *   -d N             Set debug level to N
 *   -n NAME          Set process name to NAME
 *   -q, --quiet      Accepted for old shebang lines (no banner is ever printed)
 *
 * ARGUMENTS
 *   SCRIPT_FILE      Path to the script to execute.
 *                    If omitted, reads from stdin.
 *   ARG...           Arguments passed to the script as $1 $2 ...
 *
 * Links ONLY against libmilkscript.so (and core dependencies)
 * with zero dependencies on readline or ncurses.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore_script.h"
#include "milk_bash.h"
#include "milk_syscalls.h"
#include "milkscript.h"

/* milkdata_macros.h provides dcquiet, dcdebug, data, etc. */
#include "milkdata_macros.h"
#include "milkdata.h"

/**
 * milkscript_pre_quiet - no startup banner
 *
 * A script's stdout is its data: milk-script never prints the MILK_SHM_DIR line, the '.' progress
 * dots or the LOADED lines (milk.bash does the same in libmilkbash). Constructors in the main
 * executable run before those in shared libraries; priority 101 runs this after the C runtime
 * (0-100) and before any library constructor.
 */
__attribute__((constructor(101))) static void milkscript_pre_quiet(void)
{
    milk_data.quiet = 1;
}


/**
 * milkscript_main_usage - print usage and exit
 * @prog: argv[0], used as program name in output
 */
static void milkscript_main_usage(const char *prog)
{
    printf("Usage: %s [OPTIONS] [SCRIPT_FILE [ARG...]]\n", prog);
    printf("       echo cmds | %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  -h, --help      Print this help and exit\n");
    printf("  -e, --errexit   Exit immediately if a command fails (set -e)\n");
    printf("  -x, --xtrace    Print each command before executing (set -x)\n");
    printf("  -E, --echo      Echo each input line verbatim\n");
    printf("  -d N            Set debug level to N\n");
    printf("  -n NAME         Set process name\n\n");
    printf("Script arguments are available as $1, $2, ... inside the script.\n");
    printf("\nShebang usage:\n");
    printf("  #!/usr/bin/env milk-script\n");
    printf("  #!/usr/bin/env -S milk-script -e\n");
}

/**
 * main - Entry point for the `milk-script` standalone script interpreter
 * @argc: Number of command-line arguments
 * @argv: Command-line argument vector
 *
 * Parses interpreter options (`-e`, `-x`, `-E`, `-d`, `-n`; `-q` is accepted), initializes
 * `libmilkscript`, forwards positional arguments into the bash co-process, and executes
 * the target script file or standard input.
 *
 * Return: Exit code of the executed script
 */
int main(int argc, char **argv)
{
    /* Handle -h1/--help-oneline before getopt so "-h1" is not
     * parsed as "-h" (flag) + "1" (unknown). */
    if (argc >= 2 && (strcmp(argv[1], "-h1") == 0 || strcmp(argv[1], "--help-oneline") == 0))
    {
        printf("execute a milk-cli script file\n");
        return 0;
    }

    setenv("_MILK_FRONTEND", "script", 1); /* this front end: picks the keyword set */
    milkos_malloc_keep(); /* temporaries freed and allocated again are not faulted in each time */

    int         opt_errexit = 0;
    int         opt_xtrace  = 0;
    int         opt_echo    = 0;
    int         opt_debug   = 0;
    const char *opt_name    = NULL;

    {
        static const struct option long_options[] = {
            { "help", no_argument, NULL, 'h' },   { "errexit", no_argument, NULL, 'e' },
            { "xtrace", no_argument, NULL, 'x' }, { "echo", no_argument, NULL, 'E' },
            { "quiet", no_argument, NULL, 'q' },  { NULL, 0, NULL, 0 }
        };

        int option_index = 0;
        int c;
        while ((c = getopt_long(argc, argv, "+hexEqd:n:", long_options, &option_index)) != -1)
        {
            switch (c)
            {
            case 'h':
                milkscript_main_usage(argv[0]);
                return 0;

            case 'e':
                opt_errexit = 1;
                break;

            case 'x':
                opt_xtrace = 1;
                break;

            case 'E':
                opt_echo = 1;
                break;

            case 'q': /* no banner is ever printed: kept for old shebang lines */
                break;

            case 'd':
                opt_debug = (int) strtol(optarg, NULL, 10);
                break;

            case 'n':
                opt_name = optarg;
                break;

            default:
                /* getopt already printed an error */
                return 1;
            }
        } // while options
    }

    /* Remaining args: optind → script file (optional), then $1.. */
    int    script_argc = argc - optind; /* remaining arg count  */
    char **script_argv = argv + optind; /* remaining arg vector */

    /* Build a synthetic argv for milkscript_init */
    {
        char *init_argv[2] = { opt_name ? (char *) opt_name : argv[0], NULL };
        if (milkscript_init(1, init_argv) != 0)
        {
            PRINT_ERROR("milk-script: engine initialization failed");
            return 1;
        }
    }

    /* Apply flags after init (init resets most globals to defaults) */
    if (opt_errexit)
    {
        dcerrorexit = 1;
    }
    if (opt_echo)
    {
        data.echo_input = 1;
    }
    if (opt_debug)
    {
        dcdebug = opt_debug;
    }

    /* Expose script arguments as $0 $1 ... $# */
    int sa = (script_argc > 0); /* the script file (or "-") is $0 */
    if (milk_bash_set_args(sa ? script_argv[0] : "milk-script", script_argc - sa, script_argv + sa,
                           opt_xtrace) != 0)
    {
        PRINT_ERROR("milk-script: cannot set script arguments");
        return 1;
    }

    /* Run from file or stdin.
     * A leading "-" as the first non-option argument means stdin
     * explicitly (POSIX convention), allowing:
     *   milk-script -q - arg1 arg2
     * The trailing args are still passed as $1 $2. */
    {
        int use_stdin = (script_argc == 0) || (strcmp(script_argv[0], "-") == 0);

        if (!use_stdin)
        {
            FILE *fp = fopen(script_argv[0], "r");
            if (!fp)
            {
                PRINT_ERROR("milk-script: cannot open '%s': %s", script_argv[0], strerror(errno));
                return 1;
            }
            milkscript_run(fp);
            fclose(fp);
        }
        else
        {
            /* When stdin is the source, $0 is either the explicit "-"
             * placeholder or the program name, not a real file path. */
            milkscript_run(stdin);
        }
    }

    milkscript_cleanup();
    return dcexitcode;
}
