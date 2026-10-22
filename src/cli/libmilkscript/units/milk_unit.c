// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit.c
 * @brief Command lines of the units without C (milk.calc, milk.script).
 *
 * A unit's program is given with unit options (-e/-c, -t, -p, -f, ...). They are read and removed
 * from argv here, so that fpsexec_main_function() then sees an ordinary fpsexec command line whose
 * values set the declared parameters. The unit creates its FPS itself (its parameters come from
 * -p), then hands the command line to milk_runnable_main().
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "milk_unit.h"

static int milk_unit_fps_exists(const char *name);

/**
 * unit_opt_index - Index in @opts of the unit option @s, or -1
 * @opts: Unit options
 * @nopt: Number of entries in @opts
 * @s:    Command-line word
 *
 * Return: Index of the matching option, -1 if @s is not a unit option.
 */
static int unit_opt_index(const milk_unit_opt *opts, int nopt, const char *s)
{
    for (int k = 0; k < nopt; k++)
    {
        if (strcmp(s, opts[k].shortopt) == 0 || strcmp(s, opts[k].longopt) == 0)
        {
            return k;
        }
    }
    return -1;
}

/**
 * milk_unit_args_read - Read and remove the unit options, then read the fpsexec options and command
 * @argc:      In: argument count; out: count once the unit options are removed
 * @argv:      Argument vector, compacted in place (NULL-terminated)
 * @dflt_name: Default FPS name when neither `-n NAME` nor `NAME:cmd` is given
 * @d:         Unit descriptor (its options)
 * @a:         Output: FPS name, command, help and loop flags, unit option values
 *
 * Unit options take the next word as their value whatever it is. The other words are kept in
 * order, and read with the rules of fpsexec_main_function(): the first word that is not an
 * fpsexec option is the command.
 *
 * Return: 0, or 2 if a unit option has no value.
 */
static int milk_unit_args_read(int                  *argc,
                               char                **argv,
                               const char           *dflt_name,
                               const milk_unit_desc *d,
                               milk_unit_args       *a)
{
    int n   = 1;
    int cmd = -1;

    memset(a, 0, sizeof(*a));
    snprintf(a->name, sizeof(a->name), "%s", dflt_name);

    /* Remove the unit options and their values */
    for (int i = 1; i < *argc; i++)
    {
        int k = unit_opt_index(d->opts, d->nopt, argv[i]);
        if (k < 0)
        {
            argv[n++] = argv[i];
        }
        else if (i + 1 >= *argc)
        {
            fprintf(stderr, "%s: option %s needs a value (run with -h for help)\n", d->tag,
                    argv[i]);
            return 2;
        }
        else
        {
            a->opt[k] = argv[++i];
        }
    }
    *argc   = n;
    argv[n] = NULL;

    /* fpsexec options and command word, as fpsexec_main_function() reads them */
    for (int i = 1; i < *argc; i++)
    {
        const char *s    = argv[i];
        int         span = fps_cli_option_span(*argc, argv, i);
        if (span == 0)
        {
            if (cmd == -1)
            {
                const char *c = strchr(s, ':');
                cmd           = i;
                a->cmd        = (c != NULL) ? c + 1 : s;
                if (c != NULL)
                {
                    snprintf(a->name, sizeof(a->name), "%.*s", (int) (c - s), s);
                }
            }
            continue;
        }
        if (strncmp(s, "-h", 2) == 0 || strncmp(s, "--help", 6) == 0)
        {
            a->help = 1;
        }
        else if (strcmp(s, "-loops") == 0 || strcmp(s, "--loops") == 0)
        {
            a->use_loop = 1;
        }
        else if (strcmp(s, "-loopd") == 0 || strcmp(s, "--loopd") == 0)
        {
            a->use_loop   = 2;
            a->loop_delay = (span == 2) ? atof(argv[i + 1]) : 0.0;
        }
        else if ((strcmp(s, "-n") == 0 || strcmp(s, "--name") == 0) && span == 2)
        {
            snprintf(a->name, sizeof(a->name), "%s", argv[i + 1]);
        }
        i += span - 1;
    } // for (fpsexec options)
    a->cmd = (cmd == -1) ? "run" : a->cmd;
    return 0;
}

/**
 * check_opts - Make file options absolute and check the length of the option values
 * @d: Unit descriptor
 * @a: Parsed arguments; file option values are replaced by their absolute path
 *
 * The parameter declaration may be longer than an FPS string: it is read from the command line,
 * and only shown in the FPS.
 *
 * Return: 0, or 2 if a value does not fit in its FPS string.
 */
static int check_opts(const milk_unit_desc *d, milk_unit_args *a)
{
    for (int k = 0; k < d->nopt; k++)
    {
        if (a->opt[k] == NULL)
        {
            continue;
        }
        if (d->opts[k].is_path && a->opt[k][0] != '\0')
        {
            char path[PATH_MAX];
            if (realpath(a->opt[k], path) != NULL && strlen(path) < FUNCTION_PARAMETER_STRMAXLEN)
            {
                a->opt[k] = strdup(path); /* kept until exit */
            }
        }
        if (k != d->params_opt && strlen(a->opt[k]) >= FUNCTION_PARAMETER_STRMAXLEN)
        {
            fprintf(stderr, "%s: %s is longer than %d characters%s%s\n", d->tag, d->opts[k].longopt,
                    FUNCTION_PARAMETER_STRMAXLEN - 1, (d->opts[k].toolong != NULL) ? ": " : "",
                    (d->opts[k].toolong != NULL) ? d->opts[k].toolong : "");
            return 2;
        }
    }
    return 0;
}

/**
 * opts_to_bindings - Copy the given unit options into their fixed binding variables
 * @d: Unit descriptor
 * @a: Parsed arguments
 *
 * The fixed bindings are FPS strings: their values become the defaults of a new FPS, and the
 * values shown by -h.
 */
static void opts_to_bindings(const milk_unit_desc *d, const milk_unit_args *a)
{
    for (int k = 0; k < d->nopt; k++)
    {
        if (a->opt[k] != NULL)
        {
            snprintf((char *) d->bindings[d->opts[k].slot].ptr, FUNCTION_PARAMETER_STRMAXLEN, "%s",
                     a->opt[k]);
        }
    }
}

/**
 * opts_to_fps - Write the given unit options into an existing FPS
 * @d: Unit descriptor
 * @a: Parsed arguments
 *
 * The declaration (-p) cannot change on an existing FPS: its parameters are created with the FPS.
 * A running unit compiles a new program at its next frame.
 *
 * Return: 0, or 2 if -p differs from the declaration of the FPS.
 */
static int opts_to_fps(const milk_unit_desc *d, const milk_unit_args *a)
{
    FPS fps;
    int rc = 0;

    if (fps_connect(a->name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        return 0;
    }
    for (int k = 0; k < d->nopt && rc == 0; k++)
    {
        const char *kw = d->bindings[d->opts[k].slot].fpskeyword;
        if (a->opt[k] == NULL)
        {
            continue;
        }
        if (k == d->params_opt)
        {
            long pi = functionparameter_GetParamIndex(&fps, kw);
            if (pi >= 0 && strncmp(fps.parray[pi].val.string[0], a->opt[k],
                                   FUNCTION_PARAMETER_STRMAXLEN - 1) != 0)
            {
                fprintf(stderr,
                        "%s: FPS %s declares '%s': run fpsinit to declare other parameters\n",
                        d->tag, a->name, fps.parray[pi].val.string[0]);
                rc = 2;
            }
            continue;
        }
        functionparameter_SetParamValue_STRING(&fps, kw, a->opt[k]);
    }
    fps_disconnect(&fps);
    return rc;
}

/**
 * values_to_new_fps - Write the command-line values into the FPS just created by fpsinit
 * @d: Unit descriptor
 * @a: Parsed arguments
 */
static void values_to_new_fps(const milk_unit_desc *d, const milk_unit_args *a)
{
    FPS fps;

    if (a->name[0] != '_' && fps_connect(a->name, &fps, FPSCONNECT_SIMPLE) != -1)
    {
        fps_process_argv_and_sync(&fps, d->bindings, *d->nbind);
        fps_disconnect(&fps);
    }
}

/**
 * show_fps_values - Load the values of an existing FPS into the bindings, for -h
 * @d: Unit descriptor
 * @a: Parsed arguments; the unit options given still win
 */
static void show_fps_values(const milk_unit_desc *d, const milk_unit_args *a)
{
    FPS fps;

    if (fps_connect(a->name, &fps, FPSCONNECT_SIMPLE) != -1)
    {
        fps_to_modulevars_resync_bindings(&fps, d->bindings, *d->nbind);
        fps_disconnect(&fps);
    }
    opts_to_bindings(d, a);
}

/**
 * milk_unit_main - Run the command line of a unit without C
 * @argc: Argument count
 * @argv: Argument vector (the unit options are removed from it)
 * @d:    Unit descriptor
 *
 * fpsinit, and exec when the FPS does not exist yet, create the FPS from the unit options:
 * -p declares the parameters that the values after the command word then set. On an existing FPS
 * the declared parameters are read back from it, and the other unit options are written into it.
 * -h lists the parameters that -p declares, else those of the existing FPS.
 *
 * Return: Exit status: 0 on success, 2 on a usage error, else that of fpsexec main.
 */
int milk_unit_main(int argc, char **argv, const milk_unit_desc *d)
{
    milk_unit_args a;
    const char    *params = NULL;
    int            exists = 0;
    int            init   = 0;
    int            create = 0;
    int            given  = 0; /* unit options other than -p were given */
    int            values = 0; /* the command takes values */

    if (milk_unit_args_read(&argc, argv, d->unit->app.fps_name, d, &a) != 0 ||
        check_opts(d, &a) != 0)
    {
        return 2;
    }
    params = a.opt[d->params_opt];
    for (int k = 0; k < d->nopt; k++)
    {
        given |= (k != d->params_opt && a.opt[k] != NULL);
    }
    fps_cli_set_standalone_args(argc, argv);
    fps_cli_set_unit_help(NULL, d->help_options);
    exists = milk_unit_fps_exists(a.name);
    init   = (strcmp(a.cmd, "fpsinit") == 0);
    create = !a.help && (init || (strcmp(a.cmd, "exec") == 0 && !exists));
    values = init || strcmp(a.cmd, "exec") == 0 || strcmp(a.cmd, "set") == 0 ||
             strcmp(a.cmd, "runstart") == 0 || strcmp(a.cmd, "run") == 0;
    opts_to_bindings(d, &a);

    if (a.help || create)
    {
        /* The parameters -p declares (else, for -h, those of the existing FPS) */
        if (params != NULL ? d->declare(params) != 0 : (exists && d->find_params(a.name) != 0))
        {
            return 2;
        }
        if (a.help && exists && params == NULL)
        {
            show_fps_values(d, &a);
        }
    }
    else if (exists && d->find_params(a.name) != 0)
    {
        return 1;
    }

    if (!a.help && values && fps_cli_check_values(d->bindings, *d->nbind) != 0)
    {
        return 2;
    }
    if (create)
    {
        if (milk_unit_create(&a, &d->unit->app, d->bindings, *d->nbind) != 0)
        {
            return 1;
        }
        if (init)
        {
            values_to_new_fps(d, &a);
            return 0; /* fpsinit is done here: the parameters come from this command line */
        }
    }
    else if (!a.help && exists && (given || params != NULL))
    {
        if (!values)
        {
            fprintf(stderr, "%s: unit options have no effect with %s\n", d->tag, a.cmd);
        }
        else if (opts_to_fps(d, &a) != 0)
        {
            return 2;
        }
    }
    d->unit->nb_bindings = *d->nbind;
    return milk_runnable_main(argc, argv, d->unit);
}

/**
 * milk_unit_fps_exists - Check whether a shared-memory FPS instance already exists
 * @name: FPS instance name (names starting with `_` are private/ephemeral)
 *
 * Attempts a simple non-creating `fps_connect()` to determine whether @name is
 * already initialized in shared memory.
 *
 * Return: 1 if the FPS exists in shared memory, 0 otherwise.
 */
static int milk_unit_fps_exists(const char *name)
{
    FPS fps;

    if (name[0] == '_' || fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        return 0;
    }
    fps_disconnect(&fps);
    return 1;
}

/**
 * milk_unit_create - Initialize an FPS instance and apply initial bindings and loop overrides
 * @a:   Parsed unit command-line arguments
 * @app: FPS application descriptor
 * @b:   Array of CLI-to-FPS parameter bindings
 * @nb:  Number of entries in @b
 *
 * Creates the FPS shared-memory structure via `fps_generic_init()`, synchronizes
 * default `ON` boolean parameters, and applies `-loops` or `-loopd` trigger overrides.
 *
 * Return: 0 on success, 1 on initialization failure.
 */
int milk_unit_create(const milk_unit_args *a, const FPS_APP_INFO *app, FPS_CLI_BINDING *b, int nb)
{
    FPS fps;

    printf("FPS " COLORCOMMAND "%s" COLORRESET " fpsinit\n", a->name);
    if (fps_generic_init(a->name, (FPS_APP_INFO *) app, b, nb, 1) != 0)
    {
        return 1;
    }
    if (a->name[0] != '_' && fps_connect(a->name, &fps, FPSCONNECT_SIMPLE) != -1)
    {
        for (int i = 0; i < nb; i++) /* function_parameter_add_entry() leaves bools OFF */
        {
            if (b[i].type == FPTYPE_ONOFF && *(const int32_t *) b[i].ptr != 0)
            {
                functionparameter_SetParamValue_ONOFF(&fps, b[i].fpskeyword, 1);
            }
        }
        if (a->use_loop == 1)
        {
            fps_loop_override_trigger(&fps, b, nb);
        }
        else if (a->use_loop == 2)
        {
            fps_loop_override_delay(&fps, a->loop_delay);
        }
        fps_disconnect(&fps);
    }
    return 0;
}
