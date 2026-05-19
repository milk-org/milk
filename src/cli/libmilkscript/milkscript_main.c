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
 *   -q, --quiet      Suppress startup banner
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
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore_script.h"
#include "milkscript.h"

/* milkdata_macros.h provides dcquiet, dcdebug, data, etc. */
#include "milkdata_macros.h"
#include "milkdata.h"

/* Exposed from libmilkscript for flag control */
extern int cli_flag_errexit;
extern int cli_flag_xtrace;


/**
 * milkscript_pre_quiet - early quiet detection constructor
 *
 * Reads /proc/self/cmdline to detect -q/--quiet before any
 * shared-library constructor (COREMOD RegisterModule) runs.
 * Sets milk_data.quiet directly — the same global that dcquiet
 * expands to — so the '.' progress dots and banner messages
 * are suppressed from the very first RegisterModule call.
 *
 * Constructors in the main executable run before those in
 * shared libraries.  Priority 101 ensures this runs after the
 * C runtime (0-100) and before any library constructor.
 */
__attribute__((constructor(101)))
/**
 * @brief Set up quiet mode for script execution.
 *
 * Suppresses banner and verbose output.
 */
static void milkscript_pre_quiet(void)
{
    FILE *fp = fopen("/proc/self/cmdline", "r");
    if(!fp)
    {
        return;
    }

    /* /proc/self/cmdline is NUL-delimited; read up to 4 kB */
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    if(n == 0)
    {
        return;
    }
    buf[n] = '\0';

    /* Walk each NUL-terminated token (skip argv[0]) */
    size_t i = 0;
    while(i < n && buf[i] != '\0')
    {
        i++;
    }
    i++; /* skip argv[0] NUL */

    while(i < n)
    {
        const char *tok = buf + i;
        if(tok[0] == '-' && tok[1] == '\0')
        {
            break;    /* bare '-' = stdin, stop */
        }
        if(tok[0] != '-')
        {
            break;    /* non-option = script file, stop */
        }

        if(strcmp(tok, "-q") == 0
                || strcmp(tok, "--quiet") == 0)
        {
            /* Set both the struct field and the env var so
             * both runtime paths (dcquiet check and
             * getenv check) see the quiet flag. */
            milk_data.quiet = 1;
            setenv("MILK_QUIET", "1", 0);
            break;
        }

        /* advance to next NUL-terminated token */
        while(i < n && buf[i] != '\0')
        {
            i++;
        }
        i++;
    }
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
    printf("  -n NAME         Set process name\n");
    printf("  -q, --quiet     Suppress startup banner\n\n");
    printf("Script arguments are available as $1, $2, ... inside the script.\n");
    printf("\nShebang usage:\n");
    printf("  #!/usr/bin/env milk-script\n");
    printf("  #!/usr/bin/env -S milk-script -e\n");
}

/**
 * set_positional_args - expose script arguments as $1..$N
 * @argc:  number of remaining arguments
 * @argv:  array of argument strings
 *
 * Sets $0 (script name), $1..$N (arguments), and $# (count).
 * Unsets any leftover positional variables beyond $N.
 */
static void set_positional_args(
    int argc,
    char **argv)
{
    char nbuf[32];

    /* $0 = script name (argv[0] here is the script file or "") */
    cli_var_set("0", argc > 0 ? argv[0] : "");

    /* $1..$N */
    for(int i = 1; i < argc; i++)
    {
        snprintf(nbuf, sizeof(nbuf), "%d", i);
        cli_var_set(nbuf, argv[i]);
    }

    /* Unset any extras from a previous invocation */
    for(int i = argc; i <= 9; i++)
    {
        snprintf(nbuf, sizeof(nbuf), "%d", i);
        cli_var_unset(nbuf);
    }

    /* $# = number of script arguments (not counting $0) */
    snprintf(nbuf, sizeof(nbuf), "%d", argc > 0 ? argc - 1 : 0);
    cli_var_set("#", nbuf);
}

int main(
    int argc,
    char **argv)
{
    /* Handle -h1/--help-oneline before getopt so "-h1" is not
     * parsed as "-h" (flag) + "1" (unknown). */
    if(argc >= 2 &&
            (strcmp(argv[1], "-h1") == 0 ||
             strcmp(argv[1], "--help-oneline") == 0))
    {
        printf("execute a milk-cli script file\n");
        return 0;
    }


    int opt_errexit = 0;
    int opt_xtrace  = 0;
    int opt_echo    = 0;
    int opt_quiet   = 0;
    int opt_debug   = 0;

    /* Pre-scan for -q/--quiet BEFORE getopt so that MILK_QUIET is set
     * before any module constructors run (they check getenv("MILK_QUIET"))
     * and before milkscript_init() prints its startup banner. */
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-q") == 0
                || strcmp(argv[i], "--quiet") == 0)
        {
            setenv("MILK_QUIET", "1", 0);
            break;
        }
        /* Stop scanning at first non-option argument */
        if(argv[i][0] != '-' || argv[i][1] == '\0')
        {
            break;
        }
    }
    const char *opt_name = NULL;

    static const struct option long_options[] =
    {
        { "help",    no_argument,       NULL, 'h' },
        { "errexit", no_argument,       NULL, 'e' },
        { "xtrace",  no_argument,       NULL, 'x' },
        { "echo",    no_argument,       NULL, 'E' },
        { "quiet",   no_argument,       NULL, 'q' },
        { NULL,      0,                 NULL,  0  }
    };

    int option_index = 0;
    int c;
    while((c = getopt_long(argc, argv,
                           "+hexEqd:n:",
                           long_options,
                           &option_index)) != -1)
    {
        switch(c)
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

        case 'q':
            opt_quiet = 1;
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

    /* Apply quiet flag before milkscript_init so the banner,
     * SHM dir messages, and module-load lines are suppressed */
    if(opt_quiet)
    {
        dcquiet = 1;
    }

    /* Remaining args: optind → script file (optional), then $1.. */
    int script_argc  = argc - optind;   /* remaining arg count  */
    char **script_argv = argv + optind; /* remaining arg vector */

    /* Build a synthetic argv for milkscript_init */
    char *init_argv[2] =
    {
        opt_name ? (char *) opt_name : argv[0],
        NULL
    };
    if(milkscript_init(1, init_argv) != 0)
    {
        PRINT_ERROR("milk-script: engine initialization failed");
        return 1;
    }

    /* Apply flags after init (init resets most globals to defaults) */
    if(opt_errexit)
    {
        cli_flag_errexit = 1;
    }
    if(opt_xtrace)
    {
        cli_flag_xtrace = 1;
    }
    if(opt_echo)
    {
        data.echo_input = 1;
    }
    if(opt_debug)
    {
        dcdebug = opt_debug;
    }

    /* Expose script arguments as $0 $1 ... $# */
    set_positional_args(script_argc, script_argv);

    /* Run from file or stdin.
     * A leading "-" as the first non-option argument means stdin
     * explicitly (POSIX convention), allowing:
     *   milk-script -q - arg1 arg2
     * The trailing args are still passed as $1 $2. */
    int use_stdin = (script_argc == 0)
                    || (strcmp(script_argv[0], "-") == 0);

    if(!use_stdin)
    {
        FILE *fp = fopen(script_argv[0], "r");
        if(!fp)
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

    milkscript_cleanup();
    return 0;
}
