// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_execute_preproc.c
 *
 * @brief CLI command-line pre-processing transforms.
 *
 * These helpers run on data.CLIcmdline before variable
 * expansion or command dispatch. They handle text-level
 * rewriting that must happen on the raw unmodified line:
 *
 *  - cli_check_unquoted_restricted_symbols()
 *      Shell meta-character gate.
 *  - cli_split_logical_op()
 *      Top-level && / || splitting.
 *  - cli_rewrite_stream_pipe()
 *      |> stream pipeline syntax rewrite.
 *  - cli_split_pipe()
 *      Milk-to-milk | pipe handling.
 *  - cli_split_semicolon()
 *      ; / && / || command chaining.
 *  - cli_rewrite_dot_source()
 *      ". file" → "source file" rewrite.
 *
 * All functions return 1 if they consumed the command
 * line (no further processing needed), or 0 to pass
 * control to the next stage.
 */

#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_UI_execute_internal.h"

extern int cli_last_retval;


/**
 * @brief Detect shell meta-characters outside quotes.
 *
 * Scans @cmdline character by character, tracking
 * single-quote, double-quote, and backslash-escape
 * state. Returns 1 if any restricted symbol
 * (; < > | [ ] ( ) & * $) appears outside quotes.
 *
 * The '=' character is only allowed in valid
 * assignment context (after an identifier).
 *
 * This gate decides whether the line must be
 * dispatched to /bin/sh (restricted symbols present)
 * or can be handled by the milk parser directly.
 *
 * @param cmdline  Command line to scan
 * @return 1 if restricted symbols found, 0 if clean
 */
int cli_check_unquoted_restricted_symbols(const char *cmdline)
{
    int in_squote = 0;
    int in_dquote = 0;
    int esc       = 0;

    /* '[' and ']' removed to allow stream
     * slicing syntax (e.g. im[0:19,10:29]). */
    const char *restricted = ";<>|()*&$";

    int word_start          = 1;
    int valid_assign_prefix = 0;

    for (int i = 0; cmdline[i] != '\0'; i++)
    {
        char c = cmdline[i];

        if (esc)
        {
            esc                 = 0;
            word_start          = 0;
            valid_assign_prefix = 0;
            continue;
        }

        if (c == '\\')
        {
            esc                 = 1;
            word_start          = 0;
            valid_assign_prefix = 0;
            continue;
        }

        if (in_squote)
        {
            if (c == '\'')
            {
                in_squote = 0;
            }
            continue;
        }

        if (in_dquote)
        {
            if (c == '"')
            {
                in_dquote = 0;
            }
            continue;
        }

        if (c == '\'')
        {
            in_squote           = 1;
            word_start          = 0;
            valid_assign_prefix = 0;
            continue;
        }

        if (c == '"')
        {
            in_dquote           = 1;
            word_start          = 0;
            valid_assign_prefix = 0;
            continue;
        }

        if (isspace(c))
        {
            word_start          = 1;
            valid_assign_prefix = 0;
            continue;
        }

        if (strchr(restricted, c) != NULL)
        {
            return 1;
        }

        if (c == '=')
        {
            if (!valid_assign_prefix)
            {
                return 1;
            }
            valid_assign_prefix = 0;
            continue;
        }

        if (word_start)
        {
            if (isalpha(c) || c == '_' || c == '@')
            {
                valid_assign_prefix = 1;
            }
            else
            {
                valid_assign_prefix = 0;
            }
            word_start = 0;
        }
        else
        {
            if (!isalnum(c) && c != '_' && c != '.' && c != '@')
            {
                valid_assign_prefix = 0;
            }
        }
    }

    return 0;
}


/**
 * @brief Split command line at top-level && or ||.
 *
 * Scans for the first unquoted, un-nested && or ||
 * operator. If found, executes the left side, then
 * conditionally executes the right side based on
 * success/failure.
 *
 * @param[out] retval  Set to the return value if
 *                     the line was consumed
 * @return 1 if the line was consumed, 0 otherwise
 */
int cli_split_logical_op(errno_t *retval)
{
    const char *src       = data.CLIcmdline;
    int         split_pos = -1;
    int         op_len    = 0;
    int         op_is_and = 0;

    /* Find first unquoted && and || and pick earliest */
    int pos_and = cli_find_unquoted_op(src, '&', 0, '&');
    int pos_or  = cli_find_unquoted_op(src, '|', 0, '|');

    if (pos_and >= 0 && (pos_or < 0 || pos_and < pos_or))
    {
        split_pos = pos_and;
        op_len    = 2;
        op_is_and = 1;
    }
    else if (pos_or >= 0)
    {
        split_pos = pos_or;
        op_len    = 2;
        op_is_and = 0;
    }
    if (split_pos < 0)
    {
        return 0;
    }

    char        right[STRINGMAXLEN_CLICMDLINE];
    const char *rp = src + split_pos + op_len;
    while (*rp == ' ' || *rp == '\t')
    {
        rp++;
    }
    strncpy(right, rp, STRINGMAXLEN_CLICMDLINE - 1);
    right[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    char left[STRINGMAXLEN_CLICMDLINE];
    strncpy(left, data.CLIcmdline, (size_t) split_pos);
    left[split_pos] = '\0';
    strncpy(data.CLIcmdline, left, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    errno_t lret      = CLI_execute_line();
    int     ok        = (cli_last_retval == 0);
    int     run_right = (op_is_and && ok) || (!op_is_and && !ok);
    if (run_right)
    {
        strncpy(data.CLIcmdline, right, STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        *retval                                      = CLI_execute_line();
        return 1;
    }
    *retval = lret;
    return 1;
}


/**
 * @brief Rewrite stream pipeline |> syntax.
 *
 * Transforms "stream |> cmd args" into
 * "cmd stream args" and re-executes.
 *
 * @param[out] retval  Set to the return value if
 *                     the line was consumed
 * @return 1 if the line was consumed, 0 otherwise
 */
int cli_rewrite_stream_pipe(errno_t *retval)
{
    const char *src   = data.CLIcmdline;
    int         gpipe = cli_find_unquoted_op(src, '|', 0, '>');
    if (gpipe < 0)
    {
        return 0;
    }

    char lhs[STRINGMAXLEN_CLICMDLINE];
    strncpy(lhs, data.CLIcmdline, (size_t) gpipe);
    lhs[gpipe] = '\0';
    {
        int e = gpipe - 1;
        while (e >= 0 && (lhs[e] == ' ' || lhs[e] == '\t'))
        {
            lhs[e--] = '\0';
        }
    }
    const char *rhs = data.CLIcmdline + gpipe + 2;
    while (*rhs == ' ' || *rhs == '\t')
    {
        rhs++;
    }
    const char *sp = strchr(rhs, ' ');
    char        newcmd[STRINGMAXLEN_CLICMDLINE];
    if (sp != NULL)
    {
        snprintf(newcmd, sizeof(newcmd), "%.*s %s %s", (int) (sp - rhs), rhs, lhs, sp + 1);
    }
    else
    {
        snprintf(newcmd, sizeof(newcmd), "%s %s", rhs, lhs);
    }
    strncpy(data.CLIcmdline, newcmd, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    *retval                                      = CLI_execute_line();
    data.CMDexecuted                             = 1;
    return 1;
}


/**
 * @brief Split command at top-level pipe operator.
 *
 * Handles "cmd1 | cmd2" by capturing stdout of the
 * left command into a temp file and feeding it as
 * stdin to the right command. Only matches single
 * '|', not '||' or '|>'.
 *
 * Falls through (returns 0) if the right side is
 * not a registered milk command — the later popen-
 * based handler deals with shell pipes.
 *
 * @param[out] retval  Set to the return value if
 *                     the line was consumed
 * @return 1 if the line was consumed, 0 otherwise
 */
int cli_split_pipe(errno_t *retval)
{
    const char *src      = data.CLIcmdline;
    int         pipe_pos = cli_find_unquoted_op(src, '|', '|', 0);
    /* Also reject |> (stream pipe) */
    if (pipe_pos >= 0 && src[pipe_pos + 1] == '>')
    {
        pipe_pos = -1;
    }
    if (pipe_pos < 0)
    {
        return 0;
    }

    char left[STRINGMAXLEN_CLICMDLINE];
    strncpy(left, data.CLIcmdline, (size_t) pipe_pos);
    left[pipe_pos] = '\0';
    const char *rp = data.CLIcmdline + pipe_pos + 1;
    while (*rp == ' ' || *rp == '\t')
    {
        rp++;
    }
    char right[STRINGMAXLEN_CLICMDLINE];
    strncpy(right, rp, STRINGMAXLEN_CLICMDLINE - 1);
    right[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    /* Check if right side is a milk command;
     * if not, fall through for popen handler */
    {
        char        rword[200];
        int         rw = 0;
        const char *rr = right;
        while (*rr == ' ' || *rr == '\t')
        {
            rr++;
        }
        while (*rr != '\0' && *rr != ' ' && *rr != '\t' && rw < (int) sizeof(rword) - 1)
        {
            rword[rw++] = *rr++;
        }
        rword[rw] = '\0';
        if (!cli_is_command(rword))
        {
            return 0;
        }
    }

    /* Capture left stdout into tmpfile */
    FILE *tmpfp = tmpfile();
    if (tmpfp != NULL)
    {
        fflush(stdout);
        int saved_stdout = dup(STDOUT_FILENO);
        dup2(fileno(tmpfp), STDOUT_FILENO);

        strncpy(data.CLIcmdline, left, STRINGMAXLEN_CLICMDLINE - 1);
        CLI_execute_line();

        fflush(stdout);
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);

        /* Feed tmpfile as stdin to right */
        rewind(tmpfp);
        int saved_stdin = dup(STDIN_FILENO);
        dup2(fileno(tmpfp), STDIN_FILENO);

        strncpy(data.CLIcmdline, right, STRINGMAXLEN_CLICMDLINE - 1);
        *retval = CLI_execute_line();

        dup2(saved_stdin, STDIN_FILENO);
        close(saved_stdin);
        fclose(tmpfp);
        return 1;
    }
    return 0;
}


/**
 * @brief Split at semicolon or chaining ops.
 *
 * Scans for the first unquoted ; or && or ||,
 * splits the line, executes the first part,
 * then conditionally executes the rest.
 *
 * @param[out] retval  Return value if consumed
 * @return 1 if consumed, 0 otherwise
 */
int cli_split_semicolon(errno_t *retval)
{
    char fullline[STRINGMAXLEN_CLICMDLINE];
    strncpy(fullline, data.CLIcmdline, STRINGMAXLEN_CLICMDLINE - 1);
    fullline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    int p_semi = cli_find_unquoted_op(fullline, ';', 0, 0);
    int p_and  = cli_find_unquoted_op(fullline, '&', 0, '&');
    int p_or   = cli_find_unquoted_op(fullline, '|', 0, '|');

    int chain_type = 0; /* 1=; 2=&& 3=|| */
    int chain_off  = -1;
    int chain_len  = 0;

    if (p_semi >= 0 && (chain_off < 0 || p_semi < chain_off))
    {
        chain_off  = p_semi;
        chain_type = 1;
        chain_len  = 1;
    }
    if (p_and >= 0 && (chain_off < 0 || p_and < chain_off))
    {
        chain_off  = p_and;
        chain_type = 2;
        chain_len  = 2;
    }
    if (p_or >= 0 && (chain_off < 0 || p_or < chain_off))
    {
        chain_off  = p_or;
        chain_type = 3;
        chain_len  = 2;
    }
    if (chain_off < 0)
    {
        return 0;
    }

    fullline[chain_off] = '\0';
    strncpy(data.CLIcmdline, fullline, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';

    errno_t ret1 = CLI_execute_line();

    int run_rest = 0;
    if (chain_type == 1)
    {
        run_rest = 1;
    }
    else if (chain_type == 2)
    {
        run_rest = (ret1 == RETURN_SUCCESS) ? 1 : 0;
    }
    else if (chain_type == 3)
    {
        run_rest = (ret1 != RETURN_SUCCESS) ? 1 : 0;
    }
    if (run_rest)
    {
        const char *rest = fullline + chain_off + chain_len;
        while (*rest == ' ' || *rest == '\t')
        {
            rest++;
        }
        if (*rest != '\0')
        {
            strncpy(data.CLIcmdline, rest, STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
            CLI_execute_line();
        }
    }
    *retval = RETURN_SUCCESS;
    return 1;
}


/**
 * @brief Rewrite dot-source syntax to source command.
 *
 * Transforms ". file" into "source file" in
 * data.CLIcmdline so the normal source handler
 * processes it.
 *
 * Always returns 0 (not consumed) — the rewritten
 * line is passed through to subsequent stages.
 */
int cli_rewrite_dot_source(void)
{
    const char *p = data.CLIcmdline;
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }
    if (p[0] == '.' && p[1] == ' ')
    {
        char tmp[STRINGMAXLEN_CLICMDLINE];
        snprintf(tmp, STRINGMAXLEN_CLICMDLINE, "source %s", p + 2);
        strncpy(data.CLIcmdline, tmp, STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    }
    return 0;
}
