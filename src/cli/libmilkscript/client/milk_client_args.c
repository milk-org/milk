// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_args.c
 * @brief `@args`: the parameters of a shell function or script, checked like module-command
 *        arguments.
 *
 * Purpose:     a shell function or script gets the argument rules, checks and help of a module
 *              command from one declaration line.
 * Does:        `@args "name:type[=default][:help]"... -- "$@"`: parses the declarations (types
 *              int uint float bool str stream fits); binds the values as module commands bind
 *              theirs (in order or as `.name=value`, `.` for the default, `--` before values
 *              that start with `.` or `-`); checks each with the typed parsing of fps_args.c;
 *              prints the help after -h; sets the names in the caller in one assignment.
 * Used by:     the `@args` builtin (milkargs_builtin(), milk_bash_hooks.c), which binds locals
 *              and ends the function after -h or a usage error; milk-call's `args` subcommand
 *              (milk_call_main.c), which sets plain variables. Declared in milk_client.h.
 * Uses:        fps_args.h (milk_argspec_parse(), milk_args_bind(), milk_argval_parse()),
 *              milk_ipc_common.h (milk_ipc_eval()), milk_keywords.h (the usage line).
 * See also:    docs/guide/scripting.md, "Your Own Commands".
 * Constraints: also compiled into libmilkbash.so, the thin client every shell loads.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_keywords.h"
#include "fps_types.h"
#include "fps_args.h"
#include "milk_client.h"
#include "milk_ipc_common.h"

#define ARGS_MAX 32

typedef milk_argspec_t arg_spec_t;

/**
 * print_usage - Write a one-line `@args` parameter synopsis to a stream
 * @fp:    Output stream (`stdout` or `stderr`)
 * @fname: Function or script name to display
 * @a:     Array of parsed argument specifications
 * @n:     Number of argument specifications in @a
 *
 * Formats required parameters as `<name>` and optional parameters (those with a
 * default value) as `[name]`.
 *
 * Return: void
 */
static void print_usage(FILE *fp, const char *fname, const arg_spec_t *a, int n)
{
    fprintf(fp, "Usage: %s", fname);
    for (int i = 0; i < n; i++)
    {
        fprintf(fp, (a[i].def != NULL) ? " [%s]" : " <%s>", a[i].name);
    }
    fputc('\n', fp);
}

/**
 * print_help - Print full parameter documentation for a function or script using `@args`
 * @fname: Function or script name
 * @a:     Array of parsed argument specifications
 * @n:     Number of argument specifications in @a
 *
 * Outputs the usage synopsis followed by a formatted table of parameter names,
 * types, defaults, and descriptions when the caller receives `-h` or `--help`.
 *
 * Return: void
 */
static void print_help(const char *fname, const arg_spec_t *a, int n)
{
    print_usage(stdout, fname, a, n);
    for (int i = 0; i < n; i++)
    {
        printf("  %-12s %-6s %s%-10s %s\n", a[i].name, a[i].tname, (a[i].def != NULL) ? "= " : "  ",
               (a[i].def != NULL) ? a[i].def : "", (a[i].help != NULL) ? a[i].help : "");
    }
    printf(
        "Values in order, or as .name=value; '.' keeps the default; -- before values that start\n"
        "with '.' or '-'.\n");
    fflush(stdout);
}

/** check() - @v as a value of @a's type, normalized into @buf (bool: 1 or 0); NULL if invalid. */
static const char *check(const arg_spec_t *a, const char *v, char *buf, size_t sz)
{
    CLICMDARGDATA ad;
    if (a->type == FPTYPE_STRING)
    {
        return v;
    }
    if (a->type == FPTYPE_STREAMNAME && (v[0] == '\0' || strpbrk(v, " \t\n/") != NULL))
    {
        return NULL;
    }
    if (a->type == FPTYPE_FITSFILENAME && v[0] == '\0')
    {
        return NULL;
    }
    memset(&ad, 0, sizeof(ad));
    ad.type = a->type;
    if (milk_argval_parse(a->type, v, &ad) != 0)
    {
        return NULL;
    }
    if (a->type == FPTYPE_ONOFF)
    {
        snprintf(buf, sz, "%d", ad.val.i64 ? 1 : 0);
    }
    else
    {
        milk_argval_format(&ad, buf, sz);
    }
    return buf;
}

/** bad() - report a usage error of @fname; 2. */
static int bad(const char *fname, const arg_spec_t *a, int n, const char *msg, const char *what)
{
    fprintf(stderr, "%s: %s%s\n", fname, msg, what);
    print_usage(stderr, fname, a, n);
    return 2;
}

/**
 * milk_client_args - Parse, validate, and bind `@args` declarations and caller arguments
 * @argc:  Total argument count (`SPEC... -- VALUE...`)
 * @argv:  Argument vector containing specifications before `--` and values after `--`
 * @fname: Name of the calling shell function or script for usage/error messages
 * @local: Non-zero to emit `local` bindings (inside a shell function), 0 for globals
 * @hooks: Shell integration hooks used to evaluate the generated variable bindings
 *
 * Parses parameter specifications (`name:type[=default][:help]`), binds positional and
 * `.name=value` arguments as module commands do (`milk_args_bind`), validates types via
 * `milk_argval_parse`, and evaluates
 * a single-quoted assignment command in the calling shell.
 *
 * Return: 0 on success, 1 after printing `-h`/`--help`, 2 on declaration or argument error
 */
int milk_client_args(int                        argc,
                     char                     **argv,
                     const char                *fname,
                     int                        local,
                     const milk_client_hooks_t *hooks)
{
    arg_spec_t a[ARGS_MAX];
    char      *specs[ARGS_MAX];
    int        n = 0;
    int        i = 0;
    int        rc;

    for (; i < argc && strcmp(argv[i], "--") != 0; i++)
    {
        char *sp = (n < ARGS_MAX) ? strdup(argv[i]) : NULL;
        if (sp == NULL || milk_argspec_parse(sp, &a[n]) != 0)
        {
            fprintf(
                stderr,
                "@args: bad declaration '%s': name:type[=default][:help], type int uint float bool "
                "str stream fits (at most %d)\n",
                argv[i], ARGS_MAX);
            free(sp);
            rc = 2;
            goto out;
        }
        specs[n++] = sp;
    }
    if (i == argc)
    {
        milk_lexicon_usage("args", stderr);
        rc = 2;
        goto out;
    }

    /* -h/--help before a second "--" asks for the help; the values bind as module command arguments
     * do (milk_args_bind()), every parameter taking positional values in order; "." keeps the default */
    char         **vw = argv + i + 1;
    int            nv = argc - i - 1;
    milk_argslot_t slots[ARGS_MAX];
    const char    *val[ARGS_MAX];
    char           msg[192];
    rc = 0;
    for (int t = 0; t < nv && strcmp(vw[t], "--") != 0 && rc == 0; t++)
    {
        if (strcmp(vw[t], "-h") == 0 || strcmp(vw[t], "--help") == 0)
        {
            print_help(fname, a, n);
            rc = 1;
        }
    }
    for (int k = 0; k < n; k++)
    {
        slots[k] = (milk_argslot_t) { a[k].name, 1 };
    }
    if (rc == 0 && milk_args_bind(nv, vw, -1, n, slots, val, NULL, msg, sizeof(msg)) != 0)
    {
        rc = bad(fname, a, n, "", msg);
    }
    for (int k = 0; k < n && rc == 0; k++)
    {
        int keep = (val[k] != NULL && strcmp(val[k], ".") == 0);
        a[k].val = keep ? NULL : val[k];
        if (keep && a[k].def == NULL)
        {
            rc = bad(fname, a, n, "no default for ", a[k].name);
        }
    }
    if (rc != 0)
    {
        goto out;
    }

    size_t sz = 16;
    for (int k = 0; k < n; k++)
    {
        sz += sizeof(a[k].name) +
              4 * strlen((a[k].val != NULL) ? a[k].val : (a[k].def ? a[k].def : "")) + 32;
    }
    char *script = malloc(sz);
    char *p      = script;
    if (script == NULL)
    {
        rc = 1;
        goto out;
    }
    p = stpcpy(p, local ? "local" : ":");
    for (int k = 0; k < n && rc == 0; k++)
    {
        char        buf[256];
        const char *v = (a[k].val != NULL) ? a[k].val : a[k].def;
        const char *c = (v != NULL) ? check(&a[k], v, buf, sizeof(buf)) : NULL;
        if (v == NULL)
        {
            rc = bad(fname, a, n, "missing ", a[k].name);
        }
        else if (c == NULL)
        {
            char what[192];
            snprintf(what, sizeof(what), "%s, not %s %.128s", a[k].name,
                     (a[k].type == FPTYPE_INT64 || a[k].type == FPTYPE_UINT64) ? "an" : "a",
                     a[k].tname);
            snprintf(what + strlen(what), sizeof(what) - strlen(what), ": %.64s", v);
            rc = bad(fname, a, n, "", what);
        }
        else
        {
            p    = stpcpy(stpcpy(p, local ? " " : "; "), a[k].name);
            *p++ = '=';
            *p++ = '\'';
            for (; *c != '\0'; c++)
            {
                p = (*c == '\'') ? stpcpy(p, "'\\''") : (*p = *c, p + 1);
            }
            *p++ = '\'';
            *p   = '\0';
        }
    }
    if (rc == 0)
    {
        milk_ipc_eval(hooks, script);
    }
    free(script);

out:
    for (int k = 0; k < n; k++)
    {
        free(specs[k]);
    }
    return rc;
}
