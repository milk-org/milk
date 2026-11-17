// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_call_main.c
 * @brief `milk-call`: the client as a program, for tools that run milk commands outside bash.
 *
 * Purpose:     external tools (`xargs`, `find -exec`, `timeout`, `parallel`) run module commands
 *              and keywords without a shell that loaded libmilkbash.so.
 * Does:        runs a subcommand (call, complete, translate, shutdown) or a callable keyword
 *              named without its `@` (get, set, calc, waitfor, ...): the client command of
 *              that name (milk_client_cmd()), which the bash builtins run too; run through a
 *              link, takes the command or keyword from the link's name;
 *              any other word is a module command, sent to the session.
 * Used by:     the links in libexec/milk/bin (`<module.cmd>`, `@<keyword>`) that
 *              `milk-cli --catalog` writes (milk_catalog_build.c); python/tests/cli/conftest.py
 *              (`milk-call translate`).
 * Uses:        milk_client.h (the client calls), milk_keywords.h (the keyword list of its
 *              usage), milk_bash.h (milk_translate_shorthand_to_bash()); links libmilkscript.
 * See also:    docs/guide/scripting.md, "Module Commands".
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milk_keywords.h"
#include "milk_bash.h"
#include "milk_client.h"
#include "milkdata.h"

/**
 * milkcall_pre_quiet - Suppress milk library startup messages in `milk-call`
 *
 * Constructor that sets `milk_data.quiet = 1` and `MILK_QUIET=1` before `main()`
 * so standalone client invocations emit only command output.
 *
 * Return: void
 */
__attribute__((constructor(101))) static void milkcall_pre_quiet(void)
{
    milk_data.quiet = 1;
    setenv("MILK_QUIET", "1", 0);
}

/**
 * print_usage - Print `milk-call` command-line usage and subcommand summary
 * @prog: Program name (`argv[0]`) for the usage header
 *
 * Displays available `milk-call` subcommands and their syntax to stdout.
 *
 * Return: void
 */
static void print_usage(const char *prog)
{
    printf("Usage: %s <subcommand> [args...]\n\n", prog);
    printf("Subcommands:\n");
    printf("  call [--async] [--frontend] [--nowait] <cmd> [args...]  Execute a registered milk "
           "command\n");
    printf("                                   (--frontend: the front end's own session; --nowait: "
           "fail\n");
    printf("                                   at once while another shell's command runs)\n");
    printf("  complete <prefix> [cmd|arg]      List completion candidates\n");
    printf("  translate [--legacy] <file>      Translate milk-cli shorthands to milk.bash\n");
    printf("  shutdown                         Shut down session milk-cli --server\n");
    printf("and these @ keywords, named without the @ (libexec/milk/bin/@<kw> runs them):\n");
    milk_lexicon_print_list(stdout, 1, 1);
}

/**
 * dispatch_subcommand - Route a `milk-call` subcommand to its client handler
 * @sub:  Subcommand name (e.g., `call`, `get`, `set`, `calc`, `waitfor`)
 * @argc: Number of arguments following @sub
 * @argv: Argument array following @sub
 *
 * Runs the client command called @sub (milk_client_cmd()), or `args`, `translate` and the
 * removed `server`, which are milk-call's own. Returns -1 if @sub is not a known subcommand so
 * the caller can fall back to treating it as a milk command name.
 *
 * Return: Non-negative exit status of the subcommand, or -1 if @sub is unrecognized
 */
static int dispatch_subcommand(const char *sub, int argc, char **argv)
{
    const milk_client_cmd_t *c = milk_client_cmd(sub);
    if (c != NULL)
    {
        return c->fn(argc, argv, NULL);
    }
    if (strcmp(sub, "args") ==
        0) /* plain assignments: milk-call cannot make locals or end the caller */
    {
        return milk_client_args(argc, argv, "@args", 0, NULL);
    }
    if (strcmp(sub, "translate") == 0)
    {
        int         allow_legacy = 0;
        const char *fname        = NULL;
        for (int i = 0; i < argc; i++)
        {
            if (strcmp(argv[i], "--legacy") == 0 || strcmp(argv[i], "-l") == 0)
            {
                allow_legacy = 1;
            }
            else if (fname == NULL)
            {
                fname = argv[i];
            }
        }
        FILE *fp = (fname == NULL || strcmp(fname, "-") == 0) ? stdin : fopen(fname, "r");
        if (fp == NULL)
        {
            fprintf(stderr, "milk-call translate: cannot open '%s'\n", fname ? fname : "");
            return 1;
        }
        size_t cap = 4096;
        size_t len = 0;
        char  *buf = (char *) malloc(cap);
        if (buf == NULL)
        {
            if (fp != stdin)
            {
                fclose(fp);
            }
            return 1;
        }
        size_t nr;
        while ((nr = fread(buf + len, 1, cap - 1 - len, fp)) > 0)
        {
            len += nr;
            if (len + 1 >= cap)
            {
                cap *= 2;
                char *nb = (char *) realloc(buf, cap);
                if (nb == NULL)
                {
                    free(buf);
                    if (fp != stdin)
                    {
                        fclose(fp);
                    }
                    return 1;
                }
                buf = nb;
            }
        }
        buf[len] = '\0';
        if (fp != stdin)
        {
            fclose(fp);
        }
        char *tr = milk_translate_shorthand_to_bash(buf, allow_legacy);
        free(buf);
        if (tr != NULL)
        {
            fputs(tr, stdout);
            free(tr);
            return 0;
        }
        return 1;
    }
    if (strcmp(sub, "server") == 0)
    {
        fprintf(stderr, "milk-call: server was removed; use: session (@session)\n");
        return 2;
    }

    return -1;
}

/**
 * main - Entry point for the `milk-call` standalone client executable
 * @argc: Number of command-line arguments
 * @argv: Command-line argument vector
 *
 * Inspects `basename(argv[0])` to support busybox-style symlink invocation for
 * `@<keyword>` and `<module.cmd>` links in the bin farm, or dispatches the first
 * argument as a subcommand or a module command.
 *
 * Return: Process exit status from the dispatched subcommand or remote command
 */
int main(int argc, char **argv)
{
    char progbuf[256];
    snprintf(progbuf, sizeof(progbuf), "%s", (argc > 0 && argv[0]) ? argv[0] : "milk-call");
    const char *bname = basename(progbuf);

    /* Invoked through a link: mem.listim, @waitfor, @get, ... */
    if (strcmp(bname, "milk-call") != 0)
    {
        bname += (bname[0] == '@'); /* @keyword links; module commands are linked by name */

        if (bname[0] == '\0')
        {
            return dispatch_subcommand("calc", argc - 1, argv + 1);
        }

        int rc = dispatch_subcommand(bname, argc - 1, argv + 1);
        if (rc != -1)
        {
            return rc;
        }

        argv[0] = (char *) bname;
        return milk_client_call(argc, argv, 0, NULL);
    }

    if (argc >= 2 && (strcmp(argv[1], "-h1") == 0 || strcmp(argv[1], "--help-oneline") == 0))
    {
        printf("client helper for milk-cli --server\n");
        return 0;
    }
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    int rc = dispatch_subcommand(argv[1], argc - 2, argv + 2);
    if (rc != -1)
    {
        return rc;
    }

    /* Treat argv[1] directly as a milk command name */
    return milk_client_call(argc - 1, argv + 1, 0, NULL);
}
