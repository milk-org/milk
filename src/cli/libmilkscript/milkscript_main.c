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
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore_script.h"
#include "milk_bash_engine.h"
#include "milkscript.h"

/* milkdata_macros.h provides dcquiet, dcdebug, data, etc. */
#include "milkdata_macros.h"
#include "milkdata.h"

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
static void
milkscript_pre_quiet(void)
{
    FILE *fp = fopen("/proc/self/cmdline", "r");
    if (!fp)
    {
        return;
    }

    /* /proc/self/cmdline is NUL-delimited; read up to 4 kB */
    char   buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    if (n == 0)
    {
        return;
    }
    buf[n] = '\0';

    /* Walk each NUL-terminated token (skip argv[0]) */
    size_t i = 0;
    while (i < n && buf[i] != '\0')
    {
        i++;
    }
    i++; /* skip argv[0] NUL */

    while (i < n)
    {
        const char *tok = buf + i;
        if (tok[0] == '-' && tok[1] == '\0')
        {
            break; /* bare '-' = stdin, stop */
        }
        if (tok[0] != '-')
        {
            break; /* non-option = script file, stop */
        }

        if (strcmp(tok, "-q") == 0 || strcmp(tok, "--quiet") == 0)
        {
            /* Set both the struct field and the env var so
             * both runtime paths (dcquiet check and
             * getenv check) see the quiet flag. */
            milk_data.quiet = 1;
            setenv("MILK_QUIET", "1", 0);
            break;
        }

        /* advance to next NUL-terminated token */
        while (i < n && buf[i] != '\0')
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
    printf("  -q, --quiet     Suppress startup banner\n");
    printf("  --check FILE..  Report removed spellings (status 1 if any); does not run\n");
    printf("  --modernize FILE..  Rewrite removed spellings in place ('-': stdin to stdout)\n\n");
    printf("Script arguments are available as $1, $2, ... inside the script.\n");
    printf("\nShebang usage:\n");
    printf("  #!/usr/bin/env milk-script\n");
    printf("  #!/usr/bin/env -S milk-script -e\n");
}

/** struct sbuf - growable string for building the bash prologue. */
struct sbuf
{
    char  *s;
    size_t len;
    size_t cap;
};

/** sbuf_add() - append @n bytes of @src; returns -1 on allocation failure. */
static int sbuf_add(struct sbuf *b, const char *src, size_t n)
{
    if (b->len + n + 1 > b->cap)
    {
        size_t ncap = (b->len + n + 1) * 2;
        char  *ns   = (char *) realloc(b->s, ncap);
        if (ns == NULL)
        {
            return -1;
        }
        b->s   = ns;
        b->cap = ncap;
    }
    memcpy(b->s + b->len, src, n);
    b->len += n;
    b->s[b->len] = '\0';
    return 0;
}

/** sbuf_add_quoted() - append @arg as one single-quoted bash word. */
static int sbuf_add_quoted(struct sbuf *b, const char *arg)
{
    int rc = sbuf_add(b, "'", 1);
    for (const char *c = arg; *c && rc == 0; c++)
    {
        rc = (*c == '\'') ? sbuf_add(b, "'\\''", 4) : sbuf_add(b, c, 1);
    }
    return rc | sbuf_add(b, "'", 1);
}

/**
 * script_prologue() - set $0, $1..$N (and set -x) in the bash engine before the script runs.
 */
static int script_prologue(int argc, char **argv, int xtrace)
{
    struct sbuf b  = { NULL, 0, 0 };
    int         rc = sbuf_add(&b, "BASH_ARGV0=", 11);
    rc |= sbuf_add_quoted(&b, argc > 0 ? argv[0] : "milk-script");
    rc |= sbuf_add(&b, "; set --", 8);
    for (int i = 1; i < argc; i++)
    {
        rc |= sbuf_add(&b, " ", 1);
        rc |= sbuf_add_quoted(&b, argv[i]);
    }
    if (xtrace)
    {
        rc |= sbuf_add(&b, "; set -x", 8);
    }
    if (rc == 0)
    {
        rc = milk_bash_engine_exec(b.s, NULL);
    }
    free(b.s);
    return rc;
}

/** read_all() - contents of @path ("-": stdin) as a heap string, or NULL. */
static char *read_all(const char *path)
{
    FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "milk-script: cannot open '%s'\n", path);
        return NULL;
    }
    size_t cap = 4096;
    size_t len = 0;
    char  *buf = malloc(cap);
    size_t nr;
    while (buf != NULL && (nr = fread(buf + len, 1, cap - len - 1, fp)) > 0)
    {
        len += nr;
        if (len + 1 == cap)
        {
            char *nb = realloc(buf, cap * 2);
            if (nb == NULL)
            {
                free(buf);
                buf = NULL;
                break;
            }
            buf = nb;
            cap *= 2;
        }
    }
    if (fp != stdin)
    {
        fclose(fp);
    }
    if (buf != NULL)
    {
        buf[len] = '\0';
    }
    return buf;
}

/**
 * modernize_files() - `--check`: report the removed spellings of each file (status 1 if any);
 * `--modernize` (@rewrite): rewrite them in place ("-": stdin to stdout), reporting to stderr.
 */
static int modernize_files(int nf, char **files, int rewrite)
{
    int total = 0;
    int err   = 0;
    for (int f = 0; f < nf; f++)
    {
        char *src = read_all(files[f]);
        int   n   = 0;
        char *out = (src != NULL)
                        ? milk_bash_modernize(src, files[f], rewrite ? stderr : stdout, &n)
                        : NULL;
        if (out == NULL)
        {
            err = 1;
        }
        else if (rewrite && strcmp(files[f], "-") == 0)
        {
            fputs(out, stdout);
        }
        else if (rewrite && n > 0)
        {
            FILE *wf = fopen(files[f], "w");
            if (wf == NULL || fputs(out, wf) < 0 || fclose(wf) != 0)
            {
                fprintf(stderr, "milk-script: cannot write '%s'\n", files[f]);
                err = 1;
            }
        }
        total += n;
        free(out);
        free(src);
    }
    if (nf == 0)
    {
        fprintf(stderr, "milk-script: --check/--modernize need files ('-' for stdin)\n");
        return 2;
    }
    return err ? 2 : ((!rewrite && total > 0) ? 1 : 0);
}

int main(int argc, char **argv)
{
    /* Handle -h1/--help-oneline before getopt so "-h1" is not
     * parsed as "-h" (flag) + "1" (unknown). */
    if (argc >= 2 && (strcmp(argv[1], "-h1") == 0 || strcmp(argv[1], "--help-oneline") == 0))
    {
        printf("execute a milk-cli script file\n");
        return 0;
    }


    setenv("_MILK_ENGINE", "script", 1); /* front-end of the bash engine: picks the keyword set */

    int         opt_check     = 0;
    int         opt_modernize = 0;
    int         opt_errexit   = 0;
    int         opt_xtrace    = 0;
    int         opt_echo      = 0;
    int         opt_quiet     = 0;
    int         opt_debug     = 0;
    const char *opt_name      = NULL;

    {
        static const struct option long_options[] = {
            { "help", no_argument, NULL, 'h' },      { "errexit", no_argument, NULL, 'e' },
            { "xtrace", no_argument, NULL, 'x' },    { "echo", no_argument, NULL, 'E' },
            { "quiet", no_argument, NULL, 'q' },     { "check", no_argument, NULL, 'C' },
            { "modernize", no_argument, NULL, 'M' }, { NULL, 0, NULL, 0 }
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

            case 'q':
                opt_quiet = 1;
                break;

            case 'd':
                opt_debug = (int) strtol(optarg, NULL, 10);
                break;

            case 'n':
                opt_name = optarg;
                break;

            case 'C':
                opt_check = 1;
                break;

            case 'M':
                opt_modernize = 1;
                break;

            default:
                /* getopt already printed an error */
                return 1;
            }
        } // while options
    }

    if (opt_check || opt_modernize) /* no engine needed */
    {
        return modernize_files(argc - optind, argv + optind, opt_modernize);
    }

    /* Apply quiet flag before milkscript_init so the banner,
     * SHM dir messages, and module-load lines are suppressed */
    if (opt_quiet)
    {
        dcquiet = 1;
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
    if (script_prologue(script_argc, script_argv, opt_xtrace) != 0)
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
