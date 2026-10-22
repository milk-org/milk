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
#include <strings.h>

#include "fps_types.h"
#include "fps_internal.h"
#include "fps_GetParamIndex.h"

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
 * `-n`/`-k`/`-d` (and long forms) take their value too, and `-loopd` takes the
 * following word only when it starts like a number.
 *
 * Return: 0 if @argv[@i] is not an fpsexec option, else the number of words (1 or 2).
 */
int fps_cli_option_span(int argc, char *const *argv, int i)
{
    static const char *const flags[]   = { "-h",         "--help",
                                           "-hm",        "--help-mono",
                                           "-h1",        "--help-oneline",
                                           "-h2",        "--help-description",
                                           "-tmux",      "-procinfo",
                                           "--procinfo", "-loops",
                                           "--loops",    NULL };
    static const char *const withval[] = { "-n", "--name",        "-k", "--keywords",
                                           "-d", "--description", NULL };
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


/** @brief 1 if @s (the text after a '-') starts a number: "-0.5" and "-.5" are values. */
static int starts_number(const char *s)
{
    return isdigit((unsigned char) s[0]) || (s[0] == '.' && isdigit((unsigned char) s[1]));
}


/**
 * @brief Length of the ".name" part of a ".name=value" word, 0 if @s is not one.
 *
 * The name starts with a letter or '_' and holds letters, digits, '_' and '.'.
 */
static size_t tag_length(const char *s)
{
    size_t n = 1;

    if (s[0] != '.' || !(isalpha((unsigned char) s[1]) || s[1] == '_'))
    {
        return 0;
    }
    while (isalnum((unsigned char) s[n]) || s[n] == '_' || s[n] == '.')
    {
        n++;
    }
    return (s[n] == '=') ? n : 0;
}


/** @brief Index of the binding whose keyword is the @len first characters of @tag, or -1. */
static int find_binding(const FPS_CLI_BINDING *bindings, int nb_b, const char *tag, size_t len)
{
    for (int k = 0; k < nb_b; k++)
    {
        const char *kw = bindings[k].fpskeyword;
        const char *t  = (kw[0] == '.') ? tag : tag + 1;
        size_t      l  = (kw[0] == '.') ? len : len - 1;
        if (strlen(kw) == l && strncmp(kw, t, l) == 0)
        {
            return k;
        }
    }
    return -1;
}


/** @brief Read an ONOFF value: 1/on/true or 0/off/false (any case); -1 otherwise. */
static int parse_onoff(const char *s)
{
    static const char *const on[]  = { "1", "on", "true", "yes", NULL };
    static const char *const off[] = { "0", "off", "false", "no", NULL };

    for (int k = 0; on[k] != NULL; k++)
    {
        if (strcasecmp(s, on[k]) == 0)
        {
            return 1;
        }
        if (strcasecmp(s, off[k]) == 0)
        {
            return 0;
        }
    }
    return -1;
}


/**
 * @brief Check that @str is a valid value for a parameter of @type.
 *
 * Return: NULL when valid, else a short description of what was expected.
 */
static const char *check_value(uint64_t type, const char *str)
{
    char *end = NULL;

    errno = 0;
    if (type == FPTYPE_FLOAT64 || type == FPTYPE_FLOAT32 || type == FPTYPE_TIMESPEC)
    {
        (void) strtod(str, &end);
        return (end == str || *end != '\0' || errno != 0) ? "a number" : NULL;
    }
    if (type == FPTYPE_INT64 || type == FPTYPE_INT32 || type == FPTYPE_PID)
    {
        (void) strtoll(str, &end, 0);
        return (end == str || *end != '\0' || errno != 0) ? "an integer" : NULL;
    }
    if (type == FPTYPE_UINT64 || type == FPTYPE_UINT32)
    {
        (void) strtoull(str, &end, 0);
        return (end == str || *end != '\0' || errno != 0 || str[0] == '-')
                   ? "a non-negative integer"
                   : NULL;
    }
    if (type == FPTYPE_ONOFF)
    {
        return (parse_onoff(str) < 0) ? "on/off, 1/0 or true/false" : NULL;
    }
    if (FPTYPE_IS_STRING(type) && strlen(str) >= FUNCTION_PARAMETER_STRMAXLEN)
    {
        static char msg[64];
        snprintf(msg, sizeof(msg), "at most %d characters", FUNCTION_PARAMETER_STRMAXLEN - 1);
        return msg;
    }
    return NULL;
}


/**
 * @brief Write a single CLI string value into an FPS
 *        parameter, interpreting it according to the type.
 *
 * Bumps cnt0 and value_cnt, as functionparameter_SetParamValue_*() do, so a
 * running loop that watches the parameter sees the new value.
 */
static void set_fps_value_from_string(FPS *fps, long pindex, uint64_t type, const char *str)
{
    if (type == FPTYPE_FLOAT64)
    {
        fps->parray[pindex].val.f64[0] = strtod(str, NULL);
    }
    else if (type == FPTYPE_FLOAT32)
    {
        fps->parray[pindex].val.f32[0] = strtof(str, NULL);
    }
    else if (type == FPTYPE_INT64)
    {
        fps->parray[pindex].val.i64[0] = strtoll(str, NULL, 0);
    }
    else if (type == FPTYPE_UINT64)
    {
        fps->parray[pindex].val.ui64[0] = strtoull(str, NULL, 0);
    }
    else if (type == FPTYPE_ONOFF)
    {
        fps->parray[pindex].val.i32[0] = (parse_onoff(str) == 1);
    }
    else if (type == FPTYPE_INT32)
    {
        fps->parray[pindex].val.i32[0] = (int32_t) strtol(str, NULL, 0);
    }
    else if (type == FPTYPE_UINT32)
    {
        fps->parray[pindex].val.ui32[0] = (uint32_t) strtoul(str, NULL, 0);
    }
    else if (type == FPTYPE_PID)
    {
        fps->parray[pindex].val.pid[0] = (pid_t) strtol(str, NULL, 0);
    }
    else if (type == FPTYPE_TIMESPEC)
    {
        double val                            = strtod(str, NULL);
        fps->parray[pindex].val.ts[0].tv_sec  = (long) val;
        fps->parray[pindex].val.ts[0].tv_nsec = (long) ((val - (long) val) * 1e9);
    }
    else if (FPTYPE_IS_STRING(type))
    {
        strncpy(fps->parray[pindex].val.string[0], str, FUNCTION_PARAMETER_STRMAXLEN - 1);
        fps->parray[pindex].val.string[0][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
    }
    fps->parray[pindex].cnt0++;
    fps->parray[pindex].value_cnt++;
}


/**
 * @brief Check one value, and write it into @fps unless @fps is NULL.
 *
 * Return: 0, or 2 (usage error, reported on stderr) if the value is invalid.
 */
static int assign_value(FPS *fps, const FPS_CLI_BINDING *b, const char *str)
{
    const char *kw  = (b->fpskeyword[0] == '.') ? b->fpskeyword + 1 : b->fpskeyword;
    const char *bad = NULL;

    if (strcmp(str, ".") == 0) /* keep the current value */
    {
        return 0;
    }
    if ((bad = check_value(b->type, str)) != NULL)
    {
        fprintf(stderr, "%s: invalid value '%s' for parameter '%s': expected %s\n",
                standalone_argv[0], str, kw, bad);
        return 2;
    }
    if (fps != NULL)
    {
        long pindex = functionparameter_GetParamIndex(fps, b->fpskeyword);
        if (pindex >= 0)
        {
            set_fps_value_from_string(fps, pindex, b->type, str);
        }
    }
    return 0;
}


/**
 * @brief Print the parameters that take positional values, for a "too many values" error.
 */
static void report_too_many(const FPS_CLI_BINDING *bindings, int nb_b, const char *extra)
{
    int nprim = 0;

    fprintf(stderr, "%s: too many values, from '%s' on: ", standalone_argv[0], extra);
    for (int k = 0; k < nb_b; k++)
    {
        if (bindings[k].is_primary)
        {
            const char *kw = bindings[k].fpskeyword;
            fprintf(stderr, "%s%s", (nprim++ == 0) ? "" : " ", (kw[0] == '.') ? kw + 1 : kw);
        }
    }
    fputs((nprim == 0) ? "(no parameter takes a positional value)\n"
                       : " take positional values; set others with .name=value\n",
          stderr);
}


/**
 * @brief Read the values after the command word; check them, and write them into
 *        @fps unless @fps is NULL.
 *
 * Return: 0, or 2 (usage error, reported on stderr).
 */
static int walk_values(FPS *fps, const FPS_CLI_BINDING *bindings, int nb_b)
{
    int          argc     = standalone_argc;
    char *const *argv     = standalone_argv;
    int          cmd_pos  = -1;
    int          rc       = 0;
    int          npos     = 0;
    char        *assigned = NULL;
    const char **posv     = NULL;

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

    assigned = calloc((size_t) nb_b + 1, 1);
    posv     = calloc((size_t) argc, sizeof(*posv));
    if (assigned == NULL || posv == NULL)
    {
        free(assigned);
        free(posv);
        return 2;
    }

    /* First pass: skip options, apply .name=value, collect positional values */
    for (int i = cmd_pos + 1; i < argc && rc == 0; i++)
    {
        const char *s    = argv[i];
        int         span = fps_cli_option_span(argc, argv, i);
        size_t      tl   = tag_length(s);
        if (span > 0)
        {
            i += span - 1;
        }
        else if (s[0] == '-' && !starts_number(s + 1))
        {
            fprintf(stderr, "%s: unknown option '%s' (run with -h for help)\n", argv[0], s);
            rc = 2;
        }
        else if (tl > 0)
        {
            int k = find_binding(bindings, nb_b, s, tl);
            if (k < 0)
            {
                fprintf(stderr, "%s: no parameter '%.*s' (run with -h for the list)\n", argv[0],
                        (int) tl - 1, s + 1);
                rc = 2;
            }
            else
            {
                assigned[k] = 1;
                rc          = assign_value(fps, &bindings[k], s + tl + 1);
            }
        }
        else
        {
            posv[npos++] = s;
        }
    } // for (first pass)

    /* Second pass: positional values fill the primary parameters not set by name */
    for (int p = 0, k = 0; p < npos && rc == 0; p++, k++)
    {
        while (k < nb_b && (!bindings[k].is_primary || assigned[k]))
        {
            k++;
        }
        if (k >= nb_b)
        {
            report_too_many(bindings, nb_b, posv[p]);
            rc = 2;
        }
        else
        {
            rc = assign_value(fps, &bindings[k], posv[p]);
        }
    }

    free(assigned);
    free(posv);
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
