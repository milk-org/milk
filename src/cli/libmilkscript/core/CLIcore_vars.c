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
#include "CLIcore_script.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "mcalc.h"
#include "mcalc_host.h"

CLI_VAR cli_vars[CLI_MAX_VARS];
int     cli_last_retval  = 0;
int     cli_float_digits = 15;
int     cli_calc_mode    = 0;

__attribute__((constructor)) static void init_mcalc_host(void)
{
    mcalc_host_ops ops = { cli_var_get, cli_var_set };
    mcalc_set_host(&ops);
    mcalc_float_digits = cli_float_digits;
}

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

/**
 * cli_var_set - Create, update or remove a string variable
 * @name: Variable name
 * @val:  Its text, or NULL to remove it
 *
 * The table holds the calculator's string variables only (a number lives in `dcvar`). Registered
 * as the calculator's string-variable setter.
 *
 * Return: void
 */
void cli_var_set(const char *name, const char *val)
{
    int slot = -1;
    for (int i = 0; i < CLI_MAX_VARS; i++)
    {
        if (cli_vars[i].used && strcmp(cli_vars[i].name, name) == 0)
        {
            slot = i;
            break;
        }
        if (!cli_vars[i].used && slot < 0)
        {
            slot = i;
        }
    }
    if (val == NULL)
    {
        if (slot >= 0 && cli_vars[slot].used && strcmp(cli_vars[slot].name, name) == 0)
        {
            cli_vars[slot].used = 0;
        }
        return;
    }
    if (slot < 0)
    {
        printf("Error: variable table full (max %d)\n", CLI_MAX_VARS);
        return;
    }
    snprintf(cli_vars[slot].name, sizeof(cli_vars[slot].name), "%s", name);
    snprintf(cli_vars[slot].val, sizeof(cli_vars[slot].val), "%s", val);
    cli_vars[slot].used = 1;
}
