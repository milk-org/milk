// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif
#include "CLIcore.h"

/**
 * @file    CLIcore_UI_history_cmds.c
 * @brief   History expansion, search, save, and
 *          misc history commands.
 *
 * Implements:
 *  - !! / !$ / !prefix expansion
 *  - cli_save_last_argument()
 *  - savehistory command
 *  - history <N> command
 *  - searchhist command
 *
 * @see CLIcore_UI_history.c for core log mgmt.
 * @see CLIcore_UI_history_display.c for ghistory.
 */


/**
 * @brief Expand history references in the
 *        current command line.
 *
 * Called at the very start of CLI_execute_line(),
 * before alias expansion.
 *
 * Supports:
 *   !!        — replay last command
 *   !$        — last argument of previous cmd
 *   !<prefix> — last command starting with prefix
 */
void cli_history_expand(void)
{
#ifdef USE_READLINE
    char *expanded_str = NULL;
    int   result       = history_expand(data.CLIcmdline, &expanded_str);

    if (result < 0 || result == 2)
    {
        /* -1: Error (expanded_str contains error msg)
         *  2: Display only, don't execute
         */
        if (expanded_str != NULL)
        {
            printf("%s\n", expanded_str);
            free(expanded_str);
        }
        data.CLIcmdline[0] = '\0'; /* Prevent execution */
    }
    else if (result == 1)
    {
        /* 1: Expansion took place */
        if (expanded_str != NULL)
        {
            strncpy(data.CLIcmdline, expanded_str, STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
            printf(">> %s\n", data.CLIcmdline);
            free(expanded_str);
        }
    }
    else
    {
        /* 0: No expansion took place */
        if (expanded_str != NULL)
        {
            free(expanded_str);
        }
    }
#endif
}


/**
 * @brief Save last argument after command execution
 */
void cli_save_last_argument(void)
{
    if (data.cmdNBarg > 1)
    {
        long last = data.cmdNBarg - 1;
        strncpy(data.last_argument, data.cmdargtoken[last].val.string,
                sizeof(data.last_argument) - 1);
        data.last_argument[sizeof(data.last_argument) - 1] = '\0';
    }
}


/**
 * @brief Write readline command history to a file.
 *
 * Usage: savehistory <filename>
 */
errno_t cli_savehistory(void)
{
    if (data.cmdNBarg < 2)
    {
        printf("Usage: savehistory "
               "<filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname = data.cmdargtoken[1].val.string;

#ifdef USE_READLINE
    if (write_history(fname) != 0)
    {
        printf("savehistory: failed to write "
               "'%s'\n",
               fname);
        return RETURN_FAILURE;
    }
    printf("History saved to '%s'\n", fname);
    return RETURN_SUCCESS;
#else
    printf("savehistory: readline not "
           "available\n");
    (void) fname;
    return RETURN_FAILURE;
#endif
}


/**
 * @brief Show recent readline history entries.
 *
 * Usage: history [N]  (default 20)
 */
errno_t cli_history_show(void)
{
#ifdef USE_READLINE
    int n = 20; /* default */
    if (data.cmdNBarg >= 2)
    {
        n = atoi(data.cmdargtoken[1].val.string);
        if (n <= 0)
        {
            n = 20;
        }
    }

    HIST_ENTRY **hlist = history_list();
    if (hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int start = total - n;
    if (start < 0)
    {
        start = 0;
    }
    for (int i = start; i < total; i++)
    {
        printf(" %4d  %s\n", i + 1, hlist[i]->line);
    }
#else
    printf("Readline not available\n");
#endif
    return RETURN_SUCCESS;
}


/**
 * @brief Case-insensitive substring search
 *        through readline history.
 *
 * Usage: searchhist <pattern>
 * Highlights matches in bold yellow.
 */
errno_t cli_searchhist(void)
{
#ifdef USE_READLINE
    if (data.cmdNBarg < 2)
    {
        printf("Usage: searchhist <pattern>\n");
        return RETURN_SUCCESS;
    }
    const char *pattern = data.cmdargtoken[1].val.string;

    HIST_ENTRY **hlist = history_list();
    if (hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int found = 0;
    for (int i = 0; i < total; i++)
    {
        if (strcasestr(hlist[i]->line, pattern) != NULL)
        {
            /* Highlight matching substring */
            const char *pos  = strcasestr(hlist[i]->line, pattern);
            int         pre  = (int) (pos - hlist[i]->line);
            int         plen = (int) strlen(pattern);
            printf(" %4d  %.*s"
                   "\033[1;33m%.*s\033[0m"
                   "%s\n",
                   i + 1, pre, hlist[i]->line, plen, pos, pos + plen);
            found++;
        }
    }
    if (found == 0)
    {
        printf("No history entries match"
               " '%s'\n",
               pattern);
    }
    else
    {
        printf("(%d match%s)\n", found, found == 1 ? "" : "es");
    }
#else
    printf("Readline not available\n");
#endif
    return RETURN_SUCCESS;
}
