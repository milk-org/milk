// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_session.c
 * @brief The session keywords served by the engine: @module, @config, and @history's subcommands.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_modules.h"
#include "cli_calc_parser.h"
#include "milk_catalog.h"
#include "milk_cmd_index.h"
#include "milk_server_internal.h"

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

/**
 * milk_server_history - Server handler for `@history` subcommands
 * @argc:    Number of command tokens (including `@history`)
 * @argv:    Argument vector (`search`, `--global`, `--local`, `save`, `--stats`, `--tui`)
 * @handled: Output flag set to 1 if `argv[1]` matched a known history subcommand
 *
 * Maps `@history` subcommands and flags to their corresponding `CLIcore` history
 * commands and dispatches them.
 *
 * Return: Exit status of the dispatched command when `*handled == 1`, or 0 otherwise
 */
int milk_server_history(int argc, char **argv, int *handled)
{
    static const char *const map[][2] = { { "search", "searchhist" },
                                          { "--global", "ghistory" },
                                          { "--local", "lhistory" },
                                          { "save", "savehistory" },
                                          { "--stats", "cmdstats" },
                                          { "--tui", "fhist" },
                                          { NULL, NULL } };

    *handled = 0;
    for (int i = 0; argc >= 2 && map[i][0] != NULL; i++)
    {
        if (strcmp(argv[1], map[i][0]) == 0)
        {
            *handled = 1;
            return redispatch(map[i][1], argc - 1, argv + 1);
        }
    }
    return 0;
}
