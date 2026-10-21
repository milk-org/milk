// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif
#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "cli_calc_parser.h"
#include <glob.h>
#include <sys/wait.h>
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"


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

/** prompt_format stored in data struct is TBD;
 *  for now use a file-scope buffer. */
char cli_prompt_format[200] = "";

/**
 * @brief Update the CLI prompt string.
 *
 * Reflects current directory, session name, and
 * script nesting level.
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


// cli_expand_braces moved to CLIcore_script_expand.c
// cli_expand_env moved to CLIcore_script_expand.c
