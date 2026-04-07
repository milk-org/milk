#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
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
    char *line = data.CLIcmdline;

    /* Quick check: must start with '!' */
    if(line[0] != '!')
    {
        return;
    }

    /* !! — replay last command */
    if(line[1] == '!')
    {
        HIST_ENTRY *prev = history_get(
            history_length);
        if(prev != NULL)
        {
            char suffix[
                STRINGMAXLEN_CLICMDLINE];
            suffix[0] = '\0';
            if(line[2] != '\0')
            {
                strncpy(suffix, line + 2,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                suffix[
                    STRINGMAXLEN_CLICMDLINE - 1]
                    = '\0';
            }
            char expanded[
                STRINGMAXLEN_CLICMDLINE * 2];
            snprintf(expanded,
                     sizeof(expanded),
                     "%s%s",
                     prev->line, suffix);
            strncpy(data.CLIcmdline,
                    expanded,
                    STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';
            printf(">> %s\n", data.CLIcmdline);
        }
        return;
    }

    /* !$ — last argument of previous command */
    if(line[1] == '$')
    {
        if(data.last_argument[0] != '\0')
        {
            char rest[
                STRINGMAXLEN_CLICMDLINE];
            strncpy(rest, line + 2,
                    STRINGMAXLEN_CLICMDLINE - 1);
            rest[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';
            char expanded[
                STRINGMAXLEN_CLICMDLINE * 2];
            snprintf(expanded,
                     sizeof(expanded),
                     "%s%s",
                     data.last_argument, rest);
            strncpy(data.CLIcmdline,
                    expanded,
                    STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';
            printf(">> %s\n", data.CLIcmdline);
        }
        return;
    }

    /* !<prefix> — last command starting with it */
    {
        const char *prefix = line + 1;
        size_t plen = strlen(prefix);
        /* Trim trailing spaces from prefix */
        while(plen > 0
                && prefix[plen - 1] == ' ')
        {
            plen--;
        }
        if(plen == 0)
        {
            return;
        }
        HIST_ENTRY **hist = history_list();
        if(hist == NULL)
        {
            return;
        }
        int hlen = history_length;
        for(int i = hlen - 1; i >= 0; i--)
        {
            if(strncmp(hist[i]->line,
                       prefix, plen) == 0)
            {
                strncpy(data.CLIcmdline,
                        hist[i]->line,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                data.CLIcmdline[
                    STRINGMAXLEN_CLICMDLINE - 1]
                    = '\0';
                printf(">> %s\n",
                       data.CLIcmdline);
                return;
            }
        }
        printf("!%.*s: event not found\n",
               (int) plen, prefix);
        data.CLIcmdline[0] = '\0';
    }
#endif
}


/**
 * @brief Save last argument after command execution
 */
void cli_save_last_argument(void)
{
    if(data.cmdNBarg > 1)
    {
        long last = data.cmdNBarg - 1;
        strncpy(data.last_argument,
                data.cmdargtoken[last].val.string,
                sizeof(data.last_argument) - 1);
        data.last_argument[
            sizeof(data.last_argument) - 1]
            = '\0';
    }
}


/**
 * @brief Write readline command history to a file.
 *
 * Usage: savehistory <filename>
 */
errno_t cli_savehistory(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: savehistory "
               "<filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname =
        data.cmdargtoken[1].val.string;

#ifdef USE_READLINE
    if(write_history(fname) != 0)
    {
        printf("savehistory: failed to write "
               "'%s'\n", fname);
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
    int n = 20;  /* default */
    if(data.cmdNBarg >= 2)
    {
        n = atoi(
            data.cmdargtoken[1].val.string);
        if(n <= 0)
        {
            n = 20;
        }
    }

    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int start = total - n;
    if(start < 0)
    {
        start = 0;
    }
    for(int i = start; i < total; i++)
    {
        printf(" %4d  %s\n",
               i + 1, hlist[i]->line);
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
    if(data.cmdNBarg < 2)
    {
        printf("Usage: searchhist <pattern>\n");
        return RETURN_SUCCESS;
    }
    const char *pattern =
        data.cmdargtoken[1].val.string;

    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int found = 0;
    for(int i = 0; i < total; i++)
    {
        if(strcasestr(hlist[i]->line,
                      pattern) != NULL)
        {
            /* Highlight matching substring */
            const char *pos =
                strcasestr(hlist[i]->line,
                           pattern);
            int pre = (int)(pos
                            - hlist[i]->line);
            int plen = (int) strlen(pattern);
            printf(" %4d  %.*s"
                   "\033[1;33m%.*s\033[0m"
                   "%s\n",
                   i + 1,
                   pre, hlist[i]->line,
                   plen, pos,
                   pos + plen);
            found++;
        }
    }
    if(found == 0)
    {
        printf("No history entries match"
               " '%s'\n", pattern);
    }
    else
    {
        printf("(%d match%s)\n",
               found,
               found == 1 ? "" : "es");
    }
#else
    printf("Readline not available\n");
#endif
    return RETURN_SUCCESS;
}
