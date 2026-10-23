// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_vars.c
 * @brief CLI variable table and environment synchronization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "cli_calc_parser.h"
#include "CLIcore_script.h"
#include "COREMOD_memory/COREMOD_memory.h"

CLI_VAR cli_vars[CLI_MAX_VARS];
int     cli_last_retval  = 0;
int     cli_float_digits = 15;

/**
 * @brief Look up a CLI variable by name
 *
 * @param name  Variable name
 * @return pointer to value string, or NULL
 */
const char *cli_var_get(const char *name)
{
    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (cli_vars[i].used && strcmp(cli_vars[i].name, name) == 0)
        {
            return cli_vars[i].val;
        }
    }
    return NULL;
}

/**
 * @brief Export CLI variables to environment
 */
void cli_export_vars_to_env(void)
{
    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (cli_vars[i].used)
        {
            setenv(cli_vars[i].name, cli_vars[i].val, 1);
        }
    }
}

/* var_set() - set CLI variable @name to @val: a number if it reads as one, else (@eval) the value of
 * the math expression it holds, else text */
static void var_set(const char *name, const char *val, int eval);

/**
 * @brief Set a CLI variable (create or update)
 *
 * @param name  Variable name
 * @param val   Value string
 */
void cli_var_set(const char *name, const char *val)
{
    var_set(name, val, 1);
}

/**
 * cli_var_set_text - Set a CLI variable without evaluating math expressions
 * @name: Variable name to create or update
 * @val:  Value string (parsed as numeric literal if exact, otherwise stored as text)
 *
 * Updates @name in `cli_vars[]` and `dcvar[]` without invoking the calculator parser
 * on non-numeric strings. Used when syncing plain text values from bash.
 *
 * Return: void
 */
void cli_var_set_text(const char *name, const char *val)
{
    var_set(name, val, 0);
}

/**
 * var_set - Create or update a CLI variable in the table and engine variable store
 * @name: Variable name to assign
 * @val:  Input string representation of the value
 * @eval: Non-zero to evaluate @val as a math expression if it is not a numeric literal
 *
 * Classifies @val as an integer, floating-point number, evaluated math expression (when
 * @eval is non-zero), or plain string, then records it in `cli_vars[]` and mirrors
 * numeric values into the engine's `dcvar` table.
 *
 * Return: void
 */
static void var_set(const char *name, const char *val, int eval)
{
    int    type = 2;
    long   numl = 0;
    double numf = 0.0;
    char   valbuf[CLI_VAR_VALLEN];

    if (val != NULL && *val != '\0')
    {
        char *endptr = NULL;
        strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
        valbuf[CLI_VAR_VALLEN - 1] = '\0';
        size_t len                 = strlen(valbuf);

        numl = strtol(valbuf, &endptr, 10);
        if (endptr != valbuf && *endptr == '\0')
        {
            type = 1;
            strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
            valbuf[CLI_VAR_VALLEN - 1] = '\0';
        }
        else
        {
            if (len > 0 && (valbuf[len - 1] == 'f' || valbuf[len - 1] == 'F'))
            {
                valbuf[len - 1] = '\0';
            }
            numf = strtod(valbuf, &endptr);
            if (endptr != valbuf && *endptr == '\0' && len > 0)
            {
                type = 0;
                strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
                valbuf[CLI_VAR_VALLEN - 1] = '\0';
            }
            else
            {
                int    mtype;
                long   mlval;
                double mdval;
                if (eval && cli_calc_eval_math_to_val(valbuf, &mtype, &mlval, &mdval))
                {
                    if (mtype == 1)
                    {
                        type = 1;
                        numl = mlval;
                        snprintf(valbuf, CLI_VAR_VALLEN, "%ld", numl);
                    }
                    else if (mtype == 2)
                    {
                        type = 0;
                        numf = mdval;
                        snprintf(valbuf, CLI_VAR_VALLEN, "%.*g", cli_float_digits, numf);
                    }
                }
                else
                {
                    strncpy(valbuf, val, CLI_VAR_VALLEN - 1);
                    valbuf[CLI_VAR_VALLEN - 1] = '\0';
                }
            }
        }
    }
    else
    {
        valbuf[0] = '\0';
    }

    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (cli_vars[i].used && strcmp(cli_vars[i].name, name) == 0)
        {
            strncpy(cli_vars[i].val, valbuf, CLI_VAR_VALLEN - 1);
            cli_vars[i].val[CLI_VAR_VALLEN - 1] = '\0';
            cli_vars[i].type                    = type;
            if (type == 1)
            {
                cli_vars[i].num.l = numl;
                create_variable_long_ID(name, numl);
            }
            if (type == 0)
            {
                cli_vars[i].num.f = numf;
                create_variable_ID(name, numf);
            }
            return;
        }
    }

    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (!cli_vars[i].used)
        {
            strncpy(cli_vars[i].name, name, CLI_VAR_NAMELEN - 1);
            cli_vars[i].name[CLI_VAR_NAMELEN - 1] = '\0';
            strncpy(cli_vars[i].val, valbuf, CLI_VAR_VALLEN - 1);
            cli_vars[i].val[CLI_VAR_VALLEN - 1] = '\0';
            cli_vars[i].type                    = type;
            if (type == 1)
            {
                cli_vars[i].num.l = numl;
                create_variable_long_ID(name, numl);
            }
            if (type == 0)
            {
                cli_vars[i].num.f = numf;
                create_variable_ID(name, numf);
            }
            cli_vars[i].used = 1;
            return;
        }
    }
    printf("Error: variable table full (max %d)\n", CLI_MAX_VARS);
}
