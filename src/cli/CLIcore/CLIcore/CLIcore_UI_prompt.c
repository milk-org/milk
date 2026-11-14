// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore_UI_prompt.c
 * @brief   Configurable CLI prompt format (`cli_setprompt()`)
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"

/*
 * ============================================================
 *  Configurable Prompt — setprompt command
 * ============================================================
 *
 * Format tokens:
 *   %h = hostname
 *   %u = username
 *   %d = cwd basename
 *   %t = HH:MM:SS
 *   %n = CLI process name (data.processname)
 */

char cli_prompt_format[200] = "";

/**
 * cli_setprompt - Configure or display the interactive CLI prompt format string
 *
 * Prints the active prompt format and supported `%` tokens when called without
 * arguments, or updates `cli_prompt_format` from the first CLI argument.
 *
 * Return: RETURN_SUCCESS
 */
errno_t cli_setprompt(void)
{
    if (data.cmdNBarg < 2)
    {
        if (cli_prompt_format[0] != '\0')
        {
            printf("Current prompt format: "
                   "'%s'\n",
                   cli_prompt_format);
        }
        else
        {
            printf("Using default prompt\n");
        }
        printf("Tokens: %%h=host %%u=user "
               "%%d=dir %%t=time %%n=name\n");
        return RETURN_SUCCESS;
    }
    strncpy(cli_prompt_format, data.cmdargtoken[1].val.string, sizeof(cli_prompt_format) - 1);
    cli_prompt_format[sizeof(cli_prompt_format) - 1] = '\0';
    printf("Prompt set to: '%s'\n", cli_prompt_format);
    return RETURN_SUCCESS;
}
