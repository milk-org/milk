// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_args.c
 * @brief `@args`: the parameters of a shell function or script, checked like module-command arguments.
 *
 * `@args "name:type[=default][:help]"... -- "$@"`: values in order or as `.name=value`, `.` for the
 * default, `--` before values that start with `.` or `-`; types int uint float bool str (the typed
 * parsing of milk_argval.c). The names are bound in the caller: as locals by the builtin, which also
 * ends the function after -h or a usage error.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fps_types.h"
#include "milk_argval.h"
#include "milk_client.h"
#include "milk_ipc_common.h"

#define ARGS_MAX 32

typedef milk_argspec_t arg_spec_t;

static void print_usage(FILE *fp, const char *fname, const arg_spec_t *a, int n)
{
    fprintf(fp, "Usage: %s", fname);
    for (int i = 0; i < n; i++)
    {
        fprintf(fp, (a[i].def != NULL) ? " [%s]" : " <%s>", a[i].name);
    }
    fputc('\n', fp);
}

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
        fprintf(stderr, "Usage: @args \"name:type[=default][:help]\"... -- \"$@\"\n");
        rc = 2;
        goto out;
    }

    int pos = 0;
    int raw = 0;
    rc      = 0;
    for (i++; i < argc && rc == 0; i++)
    {
        const char *v = argv[i];
        if (!raw && (strcmp(v, "-h") == 0 || strcmp(v, "--help") == 0))
        {
            print_help(fname, a, n);
            rc = 1;
        }
        else if (!raw && strcmp(v, "--") == 0)
        {
            raw = 1;
        }
        else if (!raw && v[0] == '.' && v[1] != '\0' && strchr(v, '=') != NULL)
        {
            size_t len = (size_t) (strchr(v, '=') - v - 1);
            int    k   = 0;
            while (k < n && !(strncmp(a[k].name, v + 1, len) == 0 && a[k].name[len] == '\0'))
            {
                k++;
            }
            rc = (k == n) ? bad(fname, a, n, "no parameter ", v) : 0;
            if (k < n)
            {
                a[k].val = strchr(v, '=') + 1;
            }
        }
        else
        {
            while (pos < n && a[pos].val != NULL)
            {
                pos++; /* given by name */
            }
            if (pos == n)
            {
                rc = bad(fname, a, n, "too many values: ", v);
            }
            else if (!raw && strcmp(v, ".") == 0)
            {
                rc = (a[pos].def == NULL) ? bad(fname, a, n, "no default for ", a[pos].name) : 0;
                a[pos].val = a[pos].def;
                pos++;
            }
            else
            {
                a[pos++].val = v;
            }
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
