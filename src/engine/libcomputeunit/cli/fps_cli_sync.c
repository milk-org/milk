// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_sync.c
 * @brief   Sync standalone command-line arguments to FPS and local variables
 *
 * Takes the argv of a standalone executable and syncs it into FPS
 * shared memory, then copies FPS values back into module-local C
 * variables via the binding array. The milk-cli counterpart lives in
 * libmilkscript.
 *
 * The values after the command word are read as module commands read theirs:
 * `.name=value` sets a parameter by name, the other values fill the primary
 * parameters not set by name, in order, and `.` keeps the current value.
 */


#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fps_types.h"
#include "fps_internal.h"
#include "fps_GetParamIndex.h"
#include "fps_paramvalue.h"
#include "milk_argval.h"

#include "fps_cli_binding.h"
#include "fps_cli_sync.h"
#include "fps_modulevars_resync.h"


/* Standalone argc/argv captured by main() */
static int    standalone_argc;
static char **standalone_argv;


/**
 * @brief Capture argc/argv from standalone main() for later
 *        CLI-to-FPS argument synchronization.
 *
 * Must be called once before fps_process_argv_and_sync().
 * Never called by milk-cli, where standalone_argv stays NULL.
 *
 * @param argc  Argument count from main()
 * @param argv  Argument vector from main()
 */
void fps_cli_set_standalone_args(int argc, char **argv)
{
    standalone_argc = argc;
    standalone_argv = argv;
}


int fps_cli_is_standalone(void)
{
    return standalone_argv != NULL;
}


/* Own syntax of a unit, for its -h (fps_cli_set_unit_help()) */
static const char *unit_usage;
static const char *unit_options;
static int         unit_first_param;

void fps_cli_set_unit_help(const char *usage, const char *options, int first_param)
{
    unit_usage       = usage;
    unit_options     = options;
    unit_first_param = first_param;
}

const char *fps_cli_unit_usage(void)
{
    return unit_usage;
}

const char *fps_cli_unit_options(void)
{
    return unit_options;
}

int fps_cli_unit_first_param(void)
{
    return unit_first_param;
}


/**
 * fps_cli_option_span - Count the argv words taken by an fpsexec option
 * @argc: Argument count
 * @argv: Argument vector
 * @i:    Index of the word to test
 *
 * Recognizes the options of fpsexec_main_function(): flags take one word,
 * `-n`/`-k`/`-d`/`--timeout` (and long forms) take their value too, and `-loopd`
 * takes the following word only when it starts like a number.
 *
 * Return: 0 if @argv[@i] is not an fpsexec option, else the number of words (1 or 2).
 */
int fps_cli_option_span(int argc, char *const *argv, int i)
{
    static const char *const flags[] = {
        "-h",     "--help",         "-hm",        "--help-mono",
        "-h1",    "--help-oneline", "-h2",        "--help-description",
        "-tmux",  "-procinfo",      "--procinfo", "--no-procinfo",
        "-loops", "--loops",        "--fg",       NULL
    };
    static const char *const withval[] = { "-n", "--name",        "-k",        "--keywords",
                                           "-d", "--description", "--timeout", NULL };
    const char              *s         = argv[i];

    for (int k = 0; flags[k] != NULL; k++)
    {
        if (strcmp(s, flags[k]) == 0)
        {
            return 1;
        }
    }
    for (int k = 0; withval[k] != NULL; k++)
    {
        if (strcmp(s, withval[k]) == 0)
        {
            return (i + 1 < argc) ? 2 : 1;
        }
    }
    if (strcmp(s, "-loopd") == 0 || strcmp(s, "--loopd") == 0)
    {
        const char *n = (i + 1 < argc) ? argv[i + 1] : "";
        return (isdigit((unsigned char) n[0]) || n[0] == '.') ? 2 : 1;
    }
    return 0;
}


/** @brief What a value of parameter type @type must be, for an "invalid value" error. */
static const char *expected_value(uint64_t type)
{
    static char msg[64];

    switch (type)
    {
    case FPTYPE_FLOAT64:
    case FPTYPE_FLOAT32:
    case FPTYPE_TIMESPEC:
        return "a number";
    case FPTYPE_INT64:
    case FPTYPE_INT32:
    case FPTYPE_PID:
        return "an integer";
    case FPTYPE_UINT64:
    case FPTYPE_UINT32:
        return "a non-negative integer";
    case FPTYPE_ONOFF:
        return "on/off, true/false or 1/0";
    }
    snprintf(msg, sizeof(msg), "at most %d characters", FUNCTION_PARAMETER_STRMAXLEN - 1);
    return msg;
}


/**
 * @brief Check one value (milk_argval_parse(): the reading of every argument), and write it into @fps
 *        unless @fps is NULL.
 *
 * Return: 0, or 2 (usage error, reported on stderr) if the value is invalid.
 */
static int assign_value(FPS *fps, const FPS_CLI_BINDING *b, const char *str)
{
    const char   *kw = (b->fpskeyword[0] == '.') ? b->fpskeyword + 1 : b->fpskeyword;
    CLICMDARGDATA ad;

    if (strcmp(str, ".") == 0) /* keep the current value */
    {
        return 0;
    }
    memset(&ad, 0, sizeof(ad));
    ad.type = b->type;
    if (milk_argval_parse(b->type, str, &ad) != 0 ||
        (FPTYPE_IS_STRING(b->type) && strlen(str) >= FUNCTION_PARAMETER_STRMAXLEN))
    {
        fprintf(stderr, "%s: invalid value '%s' for parameter '%s': expected %s\n",
                standalone_argv[0], str, kw, expected_value(b->type));
        return 2;
    }
    if (fps != NULL)
    {
        long pindex = functionparameter_GetParamIndex(fps, b->fpskeyword);
        if (pindex >= 0)
        {
            functionparameter_SetParamValue_arg(fps, (int) pindex, &ad);
        }
    }
    return 0;
}


/**
 * @brief Read the values after the command word; check them, and write them into
 *        @fps unless @fps is NULL.
 *
 * Unit options may appear anywhere before a "--"; another word starting with '-' there is an unknown
 * option unless it reads as a number. The values bind as module command arguments do
 * (milk_args_bind()): `.name=value` by name, the others into the positional parameters in order; a
 * parameter given no value keeps its current one.
 *
 * Return: 0, or 2 (usage error, reported on stderr).
 */
static int walk_values(FPS *fps, const FPS_CLI_BINDING *bindings, int nb_b)
{
    int             argc    = standalone_argc;
    char *const    *argv    = standalone_argv;
    int             cmd_pos = -1;
    int             rc      = 0;
    int             nw      = 0;
    int             dashes  = 0;
    char          **w       = NULL;
    milk_argslot_t *slots   = NULL;
    const char    **val     = NULL;
    char            msg[256];

    if (argv == NULL)
    {
        return 0;
    }

    /* The command word is the first word that is not an option */
    for (int i = 1; i < argc && cmd_pos == -1; i++)
    {
        int span = fps_cli_option_span(argc, argv, i);
        if (span == 0)
        {
            cmd_pos = i;
        }
        i += (span > 0) ? span - 1 : 0;
    }
    if (cmd_pos == -1)
    {
        return 0;
    }

    w     = calloc((size_t) argc, sizeof(*w));
    slots = calloc((size_t) nb_b + 1, sizeof(*slots));
    val   = calloc((size_t) nb_b + 1, sizeof(*val));
    if (w == NULL || slots == NULL || val == NULL)
    {
        rc = 2;
    }
    for (int i = cmd_pos + 1; i < argc && rc == 0; i++)
    {
        CLICMDARGDATA num;
        int           span = dashes ? 0 : fps_cli_option_span(argc, argv, i);
        if (span > 0)
        {
            i += span - 1;
            continue;
        }
        if (!dashes && argv[i][0] == '-' && strcmp(argv[i], "--") != 0 &&
            milk_argval_parse(FPTYPE_FLOAT64, argv[i], &num) != 0)
        {
            fprintf(stderr, "%s: unknown option '%s' (run with -h for help)\n", argv[0], argv[i]);
            rc = 2;
        }
        dashes |= (strcmp(argv[i], "--") == 0);
        w[nw++] = argv[i];
    }
    for (int k = 0; k < nb_b && rc == 0; k++)
    {
        const char *kw      = bindings[k].fpskeyword;
        slots[k].name       = (kw[0] == '.') ? kw + 1 : kw;
        slots[k].positional = bindings[k].is_primary;
    }
    if (rc == 0 && milk_args_bind(nw, w, -1, nb_b, slots, val, NULL, msg, sizeof(msg)) != 0)
    {
        fprintf(stderr, "%s: %s (run with -h for the list)\n", argv[0], msg);
        rc = 2;
    }
    for (int k = 0; k < nb_b && rc == 0; k++)
    {
        rc = (val[k] != NULL) ? assign_value(fps, &bindings[k], val[k]) : 0;
    }

    free(w);
    free(slots);
    free(val);
    return rc;
}


int fps_cli_check_values(const FPS_CLI_BINDING *bindings, int nb_b)
{
    return walk_values(NULL, bindings, nb_b);
}


/**
 * @brief Sync standalone argv into FPS and copy FPS values
 *        back into module-local C variables.
 *
 * Two-step process:
 *   1. argv -> FPS: read the values after the command word
 *      (`.name=value`, then positional values on the primary
 *      bindings not set by name) and write them into the FPS.
 *   2. FPS -> local: for each binding, copy the FPS
 *      parameter value into the local C variable.
 *
 * Invalid values are skipped here: fpsexec_main_function() reports them with
 * fps_cli_check_values() before doing anything.
 *
 * @param fps       Connected FPS
 * @param bindings  Parameter binding array
 * @param nb_b      Number of bindings
 * @return RETURN_SUCCESS
 */
errno_t fps_process_argv_and_sync(FPS *fps, FPS_CLI_BINDING *bindings, int nb_b)
{
    /* ---- Step 1: argv -> FPS ---- */
    (void) walk_values(fps, bindings, nb_b);

    /* ---- Step 2: FPS -> local C variables ---- */
    return fps_to_modulevars_resync_bindings(fps, bindings, nb_b);
}
