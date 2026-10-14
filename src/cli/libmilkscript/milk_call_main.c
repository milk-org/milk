// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_call_main.c
 * @brief Standalone client executable (`milk-call`) for milk-cli --server
 *
 * Used both as a fallback when `enable -f libmilkbash.so` is unavailable
 * and as the target of `$MILK_INSTALLDIR/libexec/milk-bin/@<cmd>` symlinks
 * so external tools (`xargs`, `find -exec`, `timeout`, `parallel`) can
 * invoke `@<cmd>` directly.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_bash_engine.h"
#include "milk_client.h"
#include "milkdata.h"

__attribute__((constructor(101))) static void milkcall_pre_quiet(void)
{
    milk_data.quiet = 1;
    setenv("MILK_QUIET", "1", 0);
}

static void print_usage(const char *prog)
{
    printf("Usage: %s <subcommand> [args...]\n\n", prog);
    printf("Subcommands:\n");
    printf("  call [--async] <cmd> [args...]   Execute a registered milk command\n");
    printf("  get <entity>                     Query stream/FPS/procinfo/seq property\n");
    printf("  set <entity> [val] [comment]     Set FPS parameter or stream FITS keyword\n");
    printf("  calc <expr...>                   Evaluate a @calc expression\n");
    printf("  test <expr...>                   Evaluate a @test boolean condition\n");
    printf("  exists [-r] <entity>             Check if entity exists (or -r is running)\n");
    printf("  waitfor <target> [N] [-t <sec>]  Wait for stream frames or condition\n");
    printf("  grab <s_in> <N> <s_out> [-t <s>] Capture N consecutive frames into 3D cube\n");
    printf("  complete <prefix> [cmd|arg]      List completion candidates\n");
    printf("  translate [--legacy] <file>      Translate Tier-2 shorthands to Tier-1 bash\n");
    printf("  shutdown                         Shut down session milk-cli --server\n");
}

static char *join_args(int argc, char **argv)
{
    size_t tot = 1;
    for (int i = 0; i < argc; i++)
    {
        tot += strlen(argv[i]) + 1;
    }
    char *out = (char *) malloc(tot);
    if (out == NULL)
    {
        return NULL;
    }
    out[0]     = '\0';
    size_t pos = 0;
    for (int i = 0; i < argc; i++)
    {
        int n = snprintf(out + pos, tot - pos, "%s%s", (i > 0) ? " " : "", argv[i]);
        if (n > 0)
        {
            pos += (size_t) n;
        }
    }
    return out;
}

static int dispatch_subcommand(const char *sub, int argc, char **argv)
{
    if (strcmp(sub, "call") == 0)
    {
        int async_flag = 0;
        int start_idx  = 0;
        if (argc > 0 && strcmp(argv[0], "--async") == 0)
        {
            async_flag = 1;
            start_idx  = 1;
        }
        return milk_client_call(argc - start_idx, argv + start_idx, async_flag, NULL);
    }
    if (strcmp(sub, "get") == 0)
    {
        const char *varname = NULL;
        const char *spec    = NULL;
        for (int i = 0; i < argc; i++)
        {
            if (strcmp(argv[i], "-v") == 0 && i + 1 < argc)
            {
                varname = argv[++i];
            }
            else if (spec == NULL)
            {
                spec = argv[i];
            }
        }
        return milk_client_get(spec, varname, NULL);
    }
    if (strcmp(sub, "set") == 0)
    {
        const char *spec    = (argc >= 1) ? argv[0] : NULL;
        const char *val     = (argc >= 2) ? argv[1] : NULL;
        const char *comment = (argc >= 3) ? argv[2] : NULL;
        return milk_client_set(spec, val, comment, NULL);
    }
    if (strcmp(sub, "calc") == 0)
    {
        const char *varname   = NULL;
        int         start_idx = 0;
        if (argc >= 2 && strcmp(argv[0], "-v") == 0)
        {
            varname   = argv[1];
            start_idx = 2;
        }
        char *expr = join_args(argc - start_idx, argv + start_idx);
        int   rc   = milk_client_calc(expr ? expr : "", varname, NULL);
        free(expr);
        return rc;
    }
    if (strcmp(sub, "test") == 0)
    {
        char *expr = join_args(argc, argv);
        int   rc   = milk_client_test(expr ? expr : "", NULL);
        free(expr);
        return rc;
    }
    if (strcmp(sub, "exists") == 0)
    {
        int         check_running = 0;
        const char *spec          = NULL;
        for (int i = 0; i < argc; i++)
        {
            if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--running") == 0)
            {
                check_running = 1;
            }
            else if (spec == NULL)
            {
                spec = argv[i];
            }
        }
        return milk_client_exists(spec, check_running, NULL);
    }
    if (strcmp(sub, "running") == 0)
    {
        return milk_client_exists((argc >= 1) ? argv[0] : NULL, 1, NULL);
    }
    if (strcmp(sub, "waitfor") == 0)
    {
        return milk_client_waitfor(argc, argv, NULL);
    }
    if (strcmp(sub, "grab") == 0)
    {
        return milk_client_grab(argc, argv, NULL);
    }
    if (strcmp(sub, "complete") == 0)
    {
        const char *prefix    = (argc >= 1) ? argv[0] : "";
        const char *comp_type = (argc >= 2) ? argv[1] : "arg";
        return milk_client_complete(prefix, comp_type, NULL);
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
        char *tr = milk_translate_tier2_to_tier1(buf, allow_legacy);
        free(buf);
        if (tr != NULL)
        {
            fputs(tr, stdout);
            free(tr);
            return 0;
        }
        return 1;
    }
    if (strcmp(sub, "shutdown") == 0 || strcmp(sub, "unload") == 0)
    {
        return milk_client_shutdown(NULL);
    }

    return -1;
}

int main(int argc, char **argv)
{
    char progbuf[256];
    snprintf(progbuf, sizeof(progbuf), "%s", (argc > 0 && argv[0]) ? argv[0] : "milk-call");
    const char *bname = basename(progbuf);

    /* If invoked via a symlink such as @imstats, @waitfor, @get, etc. */
    if (strcmp(bname, "milk-call") != 0)
    {
        const char *sigil = getenv("MILKCLI_SIGIL");
        if (sigil != NULL && sigil[0] != '\0' && strncmp(bname, sigil, strlen(sigil)) == 0)
        {
            bname += strlen(sigil);
        }
        else if (bname[0] == '@' || bname[0] == ',')
        {
            bname++;
        }

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
