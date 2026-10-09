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
#include "CLIcore/cli_calc_parser.h"
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

static void append_prompt_str(char *out, int *pos, int maxlen, const char *s)
{
    if (s == NULL)
    {
        return;
    }
    while (*s != '\0' && *pos < maxlen - 1)
    {
        out[(*pos)++] = *s++;
    }
}

/**
 * @brief Build prompt string from format tokens
 */
void cli_build_prompt(const char *fmt, char *out, int maxlen)
{
    int pos = 0;
    for (int i = 0; fmt[i] != '\0' && pos < maxlen - 1; i++)
    {
        if (fmt[i] == '%' && fmt[i + 1] != '\0')
        {
            i++;
            switch (fmt[i])
            {
            case 'h':
            {
                char hn[64];
                gethostname(hn, sizeof(hn));
                append_prompt_str(out, &pos, maxlen, hn);
                break;
            }
            case 'u':
            {
                const char *u = getenv("USER");
                append_prompt_str(out, &pos, maxlen, u ? u : "?");
                break;
            }
            case 'd':
            {
                char cwd[256];
                if (getcwd(cwd, sizeof(cwd)))
                {
                    char *base = strrchr(cwd, '/');
                    append_prompt_str(out, &pos, maxlen, base ? base + 1 : cwd);
                }
                break;
            }
            case 't':
            {
                time_t     now = time(NULL);
                struct tm *tm  = localtime(&now);
                char       tbuf[32];
                strftime(tbuf, sizeof(tbuf), "%H:%M:%S", tm);
                append_prompt_str(out, &pos, maxlen, tbuf);
                break;
            }
            case 'n':
                append_prompt_str(out, &pos, maxlen, data.processname);
                break;
            default:
                if (pos < maxlen - 2)
                {
                    out[pos++] = '%';
                    out[pos++] = fmt[i];
                }
                break;
            }
        }
        else
        {
            out[pos++] = fmt[i];
        }
    }
    out[pos] = '\0';
}

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
