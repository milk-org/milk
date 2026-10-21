// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit.c
 * @brief Command lines of the units without C. The units create their FPS themselves (their parameters
 * come from the command line), then hand the command line to main_impl(), so both must read it alike.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "milk_unit.h"

static const char *const sync_cmds[] = { "runstart",  "run",      "exec",    "set",
                                         "confstart", "confstep", "fpsinit", NULL };

/**
 * milk_unit_args_read - Parse command-line options, action verb, and positional args for a unit
 * @argc:      Command-line argument count
 * @argv:      Command-line argument vector
 * @dflt_name: Default FPS instance name when `-n`/`--name` or `<name>:<cmd>` is not given
 * @a:         Output structure populated with parsed flags, FPS name, command, and positions
 *
 * Scans @argv using the same rules as `fpsexec_main_function()` and
 * `fps_process_cli_and_sync()` so standalone script/calc compute units can inspect
 * positional arguments and create their dynamic FPS parameters before delegating to
 * the standard compute-unit entry point.
 *
 * Return: void
 */
void milk_unit_args_read(int argc, char **argv, const char *dflt_name, milk_unit_args *a)
{
    int cmd      = -1;
    int sync_pos = -1;

    memset(a, 0, sizeof(*a));
    snprintf(a->name, sizeof(a->name), "%s", dflt_name);
    for (int i = 1; i < argc; i++) /* main_impl()'s options and command word */
    {
        const char *s = argv[i];
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
            a->loop_delay = 0.0;
            if (i + 1 < argc && (isdigit((unsigned char) argv[i + 1][0]) || argv[i + 1][0] == '.'))
            {
                a->loop_delay = atof(argv[++i]);
            }
        }
        else if ((strcmp(s, "-n") == 0 || strcmp(s, "--name") == 0) && i + 1 < argc)
        {
            snprintf(a->name, sizeof(a->name), "%s", argv[++i]);
        }
        else if ((strcmp(s, "-k") == 0 || strcmp(s, "--keywords") == 0 || strcmp(s, "-d") == 0 ||
                  strcmp(s, "--description") == 0) &&
                 i + 1 < argc)
        {
            i++;
        }
        else if (strcmp(s, "-tmux") == 0 || strcmp(s, "-procinfo") == 0 ||
                 strcmp(s, "--procinfo") == 0)
        {
            continue;
        }
        else if (cmd == -1)
        {
            const char *c = strchr(s, ':');
            cmd           = i;
            a->cmd        = (c != NULL) ? c + 1 : s;
            if (c != NULL)
            {
                snprintf(a->name, sizeof(a->name), "%.*s", (int) (c - s), s);
            }
        }
    }
    a->cmd = (cmd == -1) ? "run" : a->cmd;

    for (int i = 1; i < argc && sync_pos == -1; i++) /* fps_process_cli_and_sync()'s positions */
    {
        const char *c = strchr(argv[i], ':');
        for (int k = 0; sync_cmds[k] != NULL; k++)
        {
            sync_pos = (strcmp((c != NULL) ? c + 1 : argv[i], sync_cmds[k]) == 0) ? i : sync_pos;
        }
    }
    for (int i = 1; i < argc && sync_pos == -1; i++)
    {
        sync_pos = (argv[i][0] != '-') ? i : -1;
    }
    a->pos  = calloc((size_t) argc, sizeof(char *));
    a->posi = calloc((size_t) argc, sizeof(int));
    for (int i = sync_pos + 1; sync_pos != -1 && a->pos != NULL && a->posi != NULL && i < argc; i++)
    {
        if (argv[i][0] != '-')
        {
            a->posi[a->npos]  = i;
            a->pos[a->npos++] = argv[i];
        }
    }
}

/**
 * milk_unit_pos - Retrieve the @i-th positional argument, treating "." as omitted
 * @a: Parsed unit command-line arguments
 * @i: Zero-based positional argument index
 *
 * Returns the positional argument at index @i if present and not equal to the
 * placeholder token `"."`.
 *
 * Return: Pointer to the positional argument string, or NULL if absent or `"."`.
 */
const char *milk_unit_pos(const milk_unit_args *a, int i)
{
    return (i < a->npos && strcmp(a->pos[i], ".") != 0) ? a->pos[i] : NULL;
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
int milk_unit_fps_exists(const char *name)
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
