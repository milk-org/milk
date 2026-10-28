// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_session.c
 * @brief The session keywords served by the engine: @module, @config, @history's subcommands, and
 *        @session save.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_modules.h"
#include "CLIcore_script.h"
#include "cli_calc_parser.h"
#include "milk_catalog.h"
#include "milk_cmd_args.h"
#include "milk_cmd_index.h"
#include "milk_ipc_common.h"
#include "milk_server.h"
#include "milk_server_internal.h"
#include "milk_srvlog.h"

#ifdef USE_CFITSIO
#    include "COREMOD_iofits/COREMOD_iofits.h"
#endif

extern int cli_float_digits;

/**
 * module_list - Print loaded modules and optionally unloaded catalog modules
 * @all: Non-zero (`--all`) to also list catalog modules not yet loaded in this server
 *
 * Formats a table of module names, short prefixes, shared library load names, and
 * registered command counts to stdout.
 *
 * Return: 0
 */
static int module_list(int all)
{
    printf("%-20s %-12s %-28s %s\n", "MODULE", "SHORT", "LOAD NAME", "COMMANDS");
    for (long m = 0; m < data.NBmodule; m++)
    {
        int n = 0;
        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            n += (data.cmd[i].moduleindex == m);
        }
        printf("%-20s %-12s %-28s %d\n", data.module[m].name, data.module[m].shortname,
               (data.module[m].loadname[0] != '\0') ? data.module[m].loadname : "(built-in)", n);
    }
    for (size_t i = 0; all && i < milk_catalog_count(); i++)
    {
        const char *load = milk_catalog_load_at(i);
        int first        = (load != NULL && strcmp(load, "milk") != 0 && !milk_module_loaded(load));
        for (size_t j = 0; first && j < i; j++)
        {
            first = strcmp(load, milk_catalog_load_at(j)) != 0;
        }
        if (first)
        {
            printf("%-20s %-12s %-28s %s\n", "-", "-", load, "(not loaded)");
        }
    }
    return 0;
}

/**
 * milk_server_module - Server handler for the `@module` session keyword
 * @argc: Number of command tokens (including `@module`)
 * @argv: Argument vector (`list [--all]` or `load <name|file.so> [--as <short>]`)
 *
 * Lists loaded/available modules or dynamically loads a shared module into the
 * server process with an optional custom short namespace alias.
 *
 * Return: 0 on success, 1 on load failure, 2 on usage or short-name validation error
 */
int milk_server_module(int argc, char **argv)
{
    const char *sub = (argc >= 2) ? argv[1] : "list";

    if (strcmp(sub, "list") == 0 && argc <= 3)
    {
        return module_list(argc == 3 && strcmp(argv[2], "--all") == 0);
    }
    if (strcmp(sub, "load") == 0 && (argc == 3 || (argc == 5 && strcmp(argv[3], "--as") == 0)))
    {
        if (argc == 3 && milk_module_loaded(argv[2]))
        {
            return 0;
        }
        if (argc == 5)
        {
            if (milk_shortname_check(argv[4], 0) != 0) /* reported; nothing is loaded */
            {
                return 2;
            }
            if (milk_module_loaded(argv[2]))
            {
                fprintf(stderr, "milk: @module load: %s is already loaded\n", argv[2]);
                return 1;
            }
            snprintf(data.moduleshortname, STRINGMAXLEN_MODULE_SHORTNAME, "%s", argv[4]);
        }
        return (load_module_shared(argv[2]) == RETURN_SUCCESS) ? 0 : 1;
    }
    fprintf(stderr, "usage: @module [list [--all]] | @module load <name|file.so> [--as <short>]\n");
    return 2;
}

/**
 * repl_setting - Map a `@config` REPL setting name to its underlying CLI engine command
 * @key: Setting key name (e.g., `synhl`, `indent`, `prompt`)
 *
 * Translates user-facing `@config` keys into the internal `CLIcore` command names
 * registered by the interactive `milk-cli` REPL.
 *
 * Return: Internal engine command name, or NULL if @key is not a REPL setting
 */
static const char *repl_setting(const char *key)
{
    static const char *const map[][2] = { { "synhl", "synhl" },
                                          { "syndiag", "syndiag" },
                                          { "indent", "cliindent" },
                                          { "showmatch", "showmatch" },
                                          { "prompt", "setprompt" },
                                          { "sessionlog", "sessionlog" },
                                          { NULL, NULL } };
    for (int i = 0; map[i][0] != NULL; i++)
    {
        if (strcmp(key, map[i][0]) == 0)
        {
            return map[i][1];
        }
    }
    return NULL;
}

/**
 * redispatch - Dispatch an internal engine command with trailing arguments
 * @cmd:  Engine command name to invoke as `av[0]`
 * @argc: Number of elements in @argv (where `argv[1..argc-1]` are forwarded)
 * @argv: Original argument vector
 *
 * Builds a temporary argument vector headed by @cmd and invokes
 * `milk_server_dispatch_argv`.
 *
 * Return: Exit status returned by `milk_server_dispatch_argv`
 */
static int redispatch(const char *cmd, int argc, char **argv)
{
    char *av[64];
    int   n = 0;

    av[n++] = (char *) cmd;
    for (int i = 1; i < argc && n < 63; i++)
    {
        av[n++] = argv[i];
    }
    av[n] = NULL;
    return milk_server_dispatch_argv(n, av);
}

/**
 * milk_server_config - Server handler for the `@config` session keyword
 * @argc: Number of command tokens (including `@config`)
 * @argv: Argument vector (`@config [<key> [<value>]]`)
 *
 * Queries or updates session settings (`digits`, `precision`, `procinfo`, `calc`)
 * or forwards REPL-specific settings (`synhl`, `syndiag`, `indent`, `showmatch`,
 * `prompt`, `sessionlog`) to their engine commands.
 *
 * Return: 0 on success, 2 on invalid key or value
 */
int milk_server_config(int argc, char **argv)
{
    const char *key = (argc >= 2) ? argv[1] : NULL;
    const char *val = (argc >= 3) ? argv[2] : NULL;
    const char *rc  = NULL;

    if (key == NULL)
    {
        printf("digits    %d\nprecision %s\nprocinfo  %s\n", cli_float_digits,
               dcprecision ? "double" : "single", dcprocinfo ? "on" : "off");
        return 0;
    }
    if (strcmp(key, "digits") == 0 && argc <= 3)
    {
        if (val != NULL)
        {
            char *e;
            long  n = strtol(val, &e, 10);
            if (*e != '\0' || n < 1 || n > 17)
            {
                fprintf(stderr, "milk: @config digits: invalid value '%s' (1 to 17)\n", val);
                return 2;
            }
            cli_float_digits = (int) n;
        }
        printf("digits %d\n", cli_float_digits);
        return 0;
    }
    if (strcmp(key, "precision") == 0 && argc <= 3)
    {
        if (val != NULL && strcmp(val, "single") != 0 && strcmp(val, "double") != 0)
        {
            fprintf(stderr, "milk: @config precision: single or double\n");
            return 2;
        }
        dcprecision = (val != NULL) ? (strcmp(val, "double") == 0) : dcprecision;
        printf("precision %s\n", dcprecision ? "double" : "single");
        return 0;
    }
    if (strcmp(key, "procinfo") == 0 && argc <= 3)
    {
        if (val != NULL && strcmp(val, "on") != 0 && strcmp(val, "off") != 0)
        {
            fprintf(stderr, "milk: @config procinfo: on or off\n");
            return 2;
        }
        dcprocinfo = (val != NULL) ? (strcmp(val, "on") == 0) : dcprocinfo;
        printf("procinfo %s\n", dcprocinfo ? "on" : "off");
        return 0;
    }
    if (strcmp(key, "calc") == 0 && argc <= 3)
    {
        if (val != NULL && strcmp(val, "on") != 0 && strcmp(val, "off") != 0)
        {
            fprintf(stderr, "milk: @config calc: on or off\n");
            return 2;
        }
        cli_calc_mode = (val != NULL) ? (strcmp(val, "on") == 0) : cli_calc_mode;
        printf("calc %s\n", cli_calc_mode ? "on" : "off");
        return 0;
    }
    if ((rc = repl_setting(key)) != NULL)
    {
        if (milk_cmd_lookup(rc) < 0)
        {
            fprintf(stderr, "milk: @config %s: a setting of the milk-cli REPL\n", key);
            return 2;
        }
        return redispatch(rc, argc - 1, argv + 1);
    }
    fprintf(stderr, "milk: @config: unknown setting '%s' (@help @config)\n", key);
    return 2;
}

/*
 * The last @n lines typed at milk-cli prompts in this session (all sessions with @global), read from
 * milk-cli's history log (~/.milk_history_log): `@history` where milk-cli's history commands are not
 * (milk-script).
 */
static int history_from_log(int global, long n)
{
    char        path[PATH_MAX];
    char        line[4096];
    const char *home = getenv("HOME");
    const char *me   = milk_server_session_name();
    long        got  = 0;

    snprintf(path, sizeof(path), "%s/.milk_history_log", (home != NULL) ? home : ".");
    FILE  *fp   = fopen(path, "r");
    char **ring = (fp != NULL) ? calloc((size_t) n, sizeof(char *)) : NULL;
    if (ring == NULL)
    {
        if (fp != NULL)
        {
            fclose(fp);
        }
        return 0;
    }
    while (fgets(line, sizeof(line), fp) != NULL) // <time> <session> <tty> <type> <text>, by tabs
    {
        char *f[5] = { line, NULL, NULL, NULL, NULL };
        for (int k = 1; k < 5 && f[k - 1] != NULL; k++)
        {
            f[k] = strchr(f[k - 1], '\t');
            if (f[k] != NULL)
            {
                *f[k]++ = '\0';
            }
        }
        if (f[4] == NULL || f[3][0] != 'P' || (!global && strcmp(f[1], me) != 0))
        {
            continue;
        }
        f[4][strcspn(f[4], "\n")] = '\0';
        free(ring[got % n]);
        ring[got % n] = NULL;
        if (asprintf(&ring[got % n], global ? "%s  %-12s  %s" : "%s  %s%s", f[0],
                     global ? f[1] : f[4], global ? f[4] : "") < 0)
        {
            ring[got % n] = NULL;
        }
        got++;
    }
    fclose(fp);
    for (long i = (got > n) ? got - n : 0; i < got; i++)
    {
        if (ring[i % n] != NULL)
        {
            puts(ring[i % n]);
        }
        free(ring[i % n]);
    }
    free(ring);
    return 0;
}

/**
 * milk_server_history - Server handler for `@history`
 * @argc:    Number of command tokens (including `@history`)
 * @argv:    Argument vector (`[N]`, `--local [N]`, `--global [N]`, `search`, `save`, `--stats`, `--tui`)
 * @handled: Output flag, set to 1
 *
 * `@history [N]` (or `--local`) shows the last N lines typed in this session, `--global` in all of
 * them, in every front-end (D39). The others are the milk-cli prompt's: its readline history.
 *
 * Return: Exit status of the history command, 2 for a usage error
 */
int milk_server_history(int argc, char **argv, int *handled)
{
    static const char *const repl[][2] = { { "search", "searchhist" },
                                           { "save", "savehistory" },
                                           { "--stats", "cmdstats" },
                                           { "--tui", "fhist" },
                                           { NULL, NULL } };
    const char              *a1        = (argc >= 2) ? argv[1] : "";
    int                      global    = (strcmp(a1, "--global") == 0);
    int                      opt       = global || strcmp(a1, "--local") == 0;

    *handled = 1;
    if (opt || argc == 1 || (argc == 2 && isdigit((unsigned char) a1[0])))
    {
        int  first = opt ? 2 : 1; // [N], then the history command's own options (-t cmd, -g PAT...)
        long n     = 20;
        char *end  = NULL;
        if (first < argc && isdigit((unsigned char) argv[first][0]))
        {
            n = strtol(argv[first++], &end, 10);
        }
        const char *cmd = global ? "ghistory" : "lhistory";
        int has = (milk_cmd_lookup(cmd) >= 0); // milk-cli's; milk-script reads the log itself
        if (n <= 0 || (end != NULL && *end != '\0') || (!has && first < argc))
        {
            fprintf(stderr, "Usage: @history [--local|--global] [N]\n");
            return 2;
        }
        if (!has)
        {
            return history_from_log(global, n);
        }
        char  nb[24];
        char *av[32] = { (char *) cmd, (char *) "-n", nb };
        int   k      = 3;
        snprintf(nb, sizeof(nb), "%ld", n);
        for (int i = first; i < argc && k < 31; i++)
        {
            av[k++] = argv[i];
        }
        av[k] = NULL;
        return milk_server_dispatch_argv(k, av);
    }
    for (int i = 0; repl[i][0] != NULL; i++)
    {
        if (strcmp(a1, repl[i][0]) == 0)
        {
            const char *fe = getenv("_MILK_ENGINE");
            if (g_srvlog_shm == NULL || !(g_srvlog_shm->flags & MILK_SRVFLAG_INPROC) ||
                (fe != NULL && strcmp(fe, "cli") != 0))
            {
                fprintf(stderr, "@history %s: at the milk-cli prompt\n", a1);
                return 2;
            }
            return redispatch(repl[i][1], argc - 1, argv + 1);
        }
    }
    fprintf(stderr,
            "Usage: @history [--local|--global] [N] | search <text> | save <file> | --stats "
            "| --tui\n");
    return 2;
}

/* The calculator variables every session starts with (CLIcore_datainit.c): not saved */
static int predefined_variable(const char *name)
{
    static const char *const names[] = { "_PI", "_e",  "_gamma", "_c", "_h",
                                         "_k",  "_pc", "_ly",    "_AU" };

    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
    {
        if (strcmp(name, names[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/* @s as one single-quoted bash word */
static void put_quoted(FILE *fp, const char *s)
{
    fputc('\'', fp);
    for (; *s != '\0'; s++)
    {
        if (*s == '\'')
        {
            fputs("'\\''", fp);
        }
        else
        {
            fputc(*s, fp);
        }
    }
    fputc('\'', fp);
}

/* The `@calc` line that sets string variable @name to @val, or a comment and a warning for a value a
 * calculator string cannot hold (a double quote, a newline). Return: 1 if saved */
static int put_string_var(FILE *fp, const char *name, const char *val, const char *fails)
{
    char text[CLI_VAR_NAMELEN + CLI_VAR_VALLEN + 8];

    if (strpbrk(val, "\"\n") != NULL)
    {
        fprintf(fp, "# the string variable %s is not saved: it holds a double quote or a newline\n",
                name);
        fprintf(stderr,
                "@session save: the string variable %s is not saved: it holds a double quote or a "
                "newline\n",
                name);
        return 0;
    }
    snprintf(text, sizeof(text), "%s = \"%s\"", name, val);
    fputs("@calc ", fp);
    put_quoted(fp, text);
    fprintf(fp, " >/dev/null%s", fails);
    return 1;
}

/* The FITS files of an earlier save in @dir (made if missing). Return: 0, or -1 */
static int clear_image_dir(const char *dir)
{
    DIR *d;

    if (mkdir(dir, 0755) != 0 && errno != EEXIST)
    {
        return -1;
    }
    if ((d = opendir(dir)) == NULL)
    {
        return -1;
    }
    for (struct dirent *de; (de = readdir(d)) != NULL;)
    {
        size_t len = strlen(de->d_name);
        char   p[PATH_MAX];
        if (len > 5 && strcmp(de->d_name + len - 5, ".fits") == 0)
        {
            snprintf(p, sizeof(p), "%s/%s", dir, de->d_name);
            unlink(p);
        }
    }
    closedir(d);
    return 0;
}

/*
 * @session save PATH: writes PATH, a milk script that rebuilds this session's settings, calculator
 * variables, command defaults and local images (one FITS file each, in PATH.d/), and that lists the
 * units the session started. `@session restore PATH` runs it. Return: 0, or 1
 */
int milk_server_session_save(const char *path)
{
    char        dir[PATH_MAX];
    char        tmp[PATH_MAX];
    char        fits[PATH_MAX + 96]; /* dir, then an image name (80 at most) */
    char        name[64] = "";
    char        when[32];
    unsigned    uid   = 0;
    int         nimg  = 0;
    int         nvar  = 0;
    int         ndef  = 0;
    int         rc    = 0;
    time_t      now   = time(NULL);
    const char *fails = " || __milk_restore_rc=1\n";

    snprintf(dir, sizeof(dir), "%s.d", path);
    snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, (int) getpid());
    milk_srvlog_parse_sock(g_srvlog_sockpath, &uid, name, sizeof(name));
    strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", localtime(&now));
    FILE *fp = fopen(tmp, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "@session save: cannot write %s: %s\n", tmp, strerror(errno));
        return 1;
    }
    fprintf(fp, "# milk session save: %s, %s\n", name, when);
    fprintf(fp,
            "# `@session restore <this file>` (or `source` it) rebuilds the session in the shell's "
            "own; the images are in <this file>.d/\n__milk_restore_rc=0\n");
    fprintf(fp, "@config digits %d >/dev/null%s", cli_float_digits, fails);
    fprintf(fp, "@config precision %s >/dev/null%s", dcprecision ? "double" : "single", fails);
    fprintf(fp, "@config procinfo %s >/dev/null%s", dcprocinfo ? "on" : "off", fails);

    for (long i = 0; data.core.variable != NULL && i < data.core.NB_MAX_VARIABLE; i++)
    {
        const VARIABLE *v = &data.core.variable[i];
        if (!v->used || predefined_variable(v->name))
        {
            continue;
        }
        if (v->type == 0)
        {
            fprintf(fp, "@calc \"%s = %.17g\" >/dev/null%s", v->name, v->value.f, fails);
        }
        else if (v->type == 1)
        {
            fprintf(fp, "@calc \"%s = %ld\" >/dev/null%s", v->name, v->value.l, fails);
        }
        else if (!put_string_var(fp, v->name, v->value.s, fails))
        {
            continue;
        }
        nvar++;
    }
    for (int i = 0; i < CLI_MAX_VARS; i++) /* string variables of the calculator live here only */
    {
        const CLI_VAR *cv = &cli_vars[i];
        if (cv->used && cv->type == 2 && variable_ID(cv->name) < 0 &&
            !predefined_variable(cv->name))
        {
            nvar += put_string_var(fp, cv->name, cv->val, fails);
        }
    }

    for (unsigned int c = 0; c < data.NBcmd; c++)
    {
        int any = 0;
        for (int a = 0; a < data.cmd[c].nbparam && data.cmd[c].argdata != NULL; a++)
        {
            const char *dv = milk_args_session_get((int) c, a);
            if (dv == NULL)
            {
                continue;
            }
            if (!any)
            {
                fprintf(fp, "%s --default", data.cmd[c].key);
            }
            fprintf(fp, " .%s=", milk_args_tag(data.cmd[c].argdata[a].fpstag));
            put_quoted(fp, dv);
            any = 1;
        }
        if (any)
        {
            fprintf(fp, " >/dev/null%s", fails);
            ndef++;
        }
    }

    for (long i = 0; data.core.image != NULL && i < data.core.NB_MAX_IMAGE; i++)
    {
        const IMAGE *im = &data.core.image[i];
        if (!im->used || im->md == NULL || im->md->shared)
        {
            continue;
        }
        if (nimg == 0)
        {
            if (clear_image_dir(dir) != 0)
            {
                fprintf(stderr, "@session save: cannot use %s: %s\n", dir, strerror(errno));
                rc = 1;
                break;
            }
            fprintf(fp,
                    "__milk_oldpwd=$OLDPWD\nif pushd \"${BASH_SOURCE[0]}.d\" >/dev/null; then\n");
        }
#ifdef USE_CFITSIO
        snprintf(fits, sizeof(fits), "%s/%s.fits", dir, im->name);
        if (save_fits(im->name, fits) != RETURN_SUCCESS)
        {
            fprintf(stderr, "@session save: cannot write image %s to %s\n", im->name, fits);
            rc = 1;
            continue;
        }
        fprintf(fp, "    iofits.loadfits %s.fits @L:%s 2 >/dev/null%s", im->name, im->name, fails);
        nimg++;
#else
        (void) fits;
        fprintf(fp, "# the image %s is not saved: milk was built without cfitsio\n", im->name);
        rc = 1;
#endif
    }
    if (nimg > 0)
    {
        fprintf(fp,
                "    popd >/dev/null\nelse\n    __milk_restore_rc=1\nfi\nOLDPWD=$__milk_oldpwd\n");
    }

    for (long i = 0; data.core.image != NULL && i < data.core.NB_MAX_IMAGE; i++)
    {
        const IMAGE *im = &data.core.image[i];
        if (im->used && im->md != NULL && im->md->shared)
        {
            fprintf(fp, "# stream %s: shared memory, not saved\n", im->name);
        }
    }
    int nunit = milk_srvlog_session_list_units(milk_ipc_is_numeric_str(name) ? "" : name, fp);
    fprintf(fp, "# end of milk session save\n");

    if (fclose(fp) != 0 || rename(tmp, path) != 0)
    {
        fprintf(stderr, "@session save: cannot write %s: %s\n", path, strerror(errno));
        unlink(tmp);
        return 1;
    }
    printf(
        "Session %s saved to %s: %d images, %d variables, %d command defaults, %d units listed\n",
        name, path, nimg, nvar, ndef, nunit);
    return rc;
}
