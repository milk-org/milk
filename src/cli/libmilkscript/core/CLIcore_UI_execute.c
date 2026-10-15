// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_execute.c
 *
 * @brief Core CLI command dispatcher and typo-suggestion helper.
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "milk_bash_engine.h"

/**
 * @brief Central command dispatcher — delegates execution to the bash co-process engine.
 *
 * @return RETURN_SUCCESS on success, or RETURN_FAILURE on error
 */
errno_t CLI_execute_line()
{
    DEBUG_TRACE_FSTART();

    int exit_req = 0;
    int rc       = milk_bash_engine_exec(data.CLIcmdline, &exit_req);
    if (exit_req || (dcerrorexit && rc != 0))
    {
        data.CLIloopON = 0;
    }

    DEBUG_TRACE_FEXIT();
    return (rc == 0) ? RETURN_SUCCESS : RETURN_FAILURE;
}

static int cmd_levenshtein_distance(const char *s1, const char *s2)
{
    size_t l1 = strlen(s1);
    size_t l2 = strlen(s2);
    if (l1 > 63)
    {
        l1 = 63;
    }
    if (l2 > 63)
    {
        l2 = 63;
    }
    int dp[64];
    for (size_t j = 0; j <= l2; j++)
    {
        dp[j] = (int) j;
    }
    for (size_t i = 1; i <= l1; i++)
    {
        int prev = dp[0];
        dp[0]    = (int) i;
        for (size_t j = 1; j <= l2; j++)
        {
            int tmp  = dp[j];
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            int del  = dp[j] + 1;
            int ins  = dp[j - 1] + 1;
            int sub  = prev + cost;
            int m    = (del < ins) ? del : ins;
            dp[j]    = (m < sub) ? m : sub;
            prev     = tmp;
        }
    }
    return dp[l2];
}

/**
 * handle_did_you_mean - print typo suggestions for an unknown command
 * @input_cmd: first token that was not resolved
 */
void handle_did_you_mean(const char *input_cmd)
{
    if (input_cmd != NULL && input_cmd[0] != '\0')
    {
        struct distmatch
        {
            int         dist;
            const char *cmd;
        };
        struct distmatch matches[3] = { { 9999, NULL }, { 9999, NULL }, { 9999, NULL } };

        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            int d = cmd_levenshtein_distance(input_cmd, data.cmd[i].key);
            if (d < matches[2].dist)
            {
                matches[2].dist = d;
                matches[2].cmd  = data.cmd[i].key;
                if (matches[2].dist < matches[1].dist)
                {
                    struct distmatch tmp = matches[1];
                    matches[1]           = matches[2];
                    matches[2]           = tmp;
                }
                if (matches[1].dist < matches[0].dist)
                {
                    struct distmatch tmp = matches[0];
                    matches[0]           = matches[1];
                    matches[1]           = tmp;
                }
            }
        }

        if (matches[0].dist <= 4 && matches[0].cmd != NULL)
        {
            printf("\033[31mCommand '%s' not found. \033[0mDid you mean:\n", input_cmd);
            for (int m = 0; m < 3; m++)
            {
                if (matches[m].cmd && matches[m].dist <= 4 && matches[m].dist < 9999)
                {
                    printf("  - \033[0;96m%s\033[0m\n", matches[m].cmd);
                }
            }
            return;
        }
    }
    printf("\033[31mCommand not found, or command with no effect\n\033[0m");
}
