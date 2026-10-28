// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_exec.c
 * @brief High-level command/file execution and `@calc` REPL mode dispatch for the bash engine.
 *
 * Role & Goal:
 *   Provides `milk_bash_engine_exec()` and `milk_bash_engine_exec_file()`, handling
 *   re-entrancy guards, bare `@calc` REPL mode toggles, in-process calculator
 *   fast-paths, and multi-line bash block accumulation before delegating raw
 *   shell execution to `milk_bash_engine_exec_raw()`.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "cli_calc_parser.h"
#include "milk_bash_engine.h"
#include "milk_bash_engine_internal.h"
#include "milk_server.h"
#include "milk_shell_lex.h"

/**
 * milk_bash_engine_exec_raw - Translate and execute a raw bash block in the co-process
 * @cmdline:      Command line or multi-line script text
 * @out_exit_req: Optional output set to 1 if script invoked `exit`
 *
 * Return: Exit status of the bash command (0 = success)
 */
int milk_bash_engine_exec_raw(const char *cmdline, int *out_exit_req)
{
    if (out_exit_req != NULL)
    {
        *out_exit_req = 0;
    }
    if (cmdline == NULL)
    {
        return 0;
    }
    fflush(stdout);
    fflush(stderr);
    if (milk_bash_coproc_ensure() != 0)
    {
        return 1;
    }

    int bump_epoch =
        (strcmp(cmdline, ":") != 0 && strncmp(cmdline, "BASH_ARGV0=", 11) != 0 &&
         !(strncmp(cmdline, "@calc --repl \"", 14) == 0 && strchr(cmdline, '$') == NULL &&
           strchr(cmdline, '`') == NULL && strstr(cmdline, "@{") == NULL));
    if (bump_epoch)
    {
        milk_server_bump_bash_epoch();
    }

    msl_map_t chk;
    if (msl_scan(cmdline, strlen(cmdline), &chk) == 0)
    {
        int unterminated = chk.unterminated;
        msl_free(&chk);
        if (unterminated)
        {
            fprintf(stderr, "milk: syntax error: unterminated quote or substitution\n");
            return 2;
        }
    }

    char *translated = milk_translate_shorthand_to_bash(cmdline, 1);
    if (translated == NULL)
    {
        return 1;
    }
    if (dcerrorexit && strcmp(cmdline, ":") != 0)
    {
        size_t old_len = strlen(translated);
        char  *with_e  = (char *) malloc(old_len + 16);
        if (with_e != NULL)
        {
            snprintf(with_e, old_len + 16, "set -e; %s", translated);
            free(translated);
            translated = with_e;
        }
    }

    int lost = 0;
    int rc   = milk_bash_coproc_run(translated, out_exit_req, &lost);
    if (lost)
    {
        return rc;
    }
    if (bump_epoch)
    {
        milk_server_bump_bash_epoch();
    }
    cli_last_retval = rc;
    return rc;
}

/**
 * trim_calc_line - Strip unquoted `#` comment and leading/trailing whitespace in-place
 * @s: Writable line buffer
 *
 * Return: Pointer to the first non-whitespace character in @s
 */
static char *trim_calc_line(char *s)
{
    while (*s == ' ' || *s == '\t' || *s == '\r')
    {
        s++;
    }
    int in_sq = 0;
    int in_dq = 0;
    for (size_t i = 0; s[i] != '\0'; i++)
    {
        if (s[i] == '\\' && !in_sq && s[i + 1] != '\0')
        {
            i++;
            continue;
        }
        if (s[i] == '\'' && !in_dq)
        {
            in_sq = !in_sq;
        }
        else if (s[i] == '"' && !in_sq)
        {
            in_dq = !in_dq;
        }
        else if (s[i] == '#' && !in_sq && !in_dq)
        {
            s[i] = '\0';
            break;
        }
    }
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t' || s[len - 1] == '\r'))
    {
        s[--len] = '\0';
    }
    return s;
}

/**
 * cmdline_has_bare_calc - Check whether any line in @cmdline is a bare `@calc` mode toggle
 * @cmdline: Command line or multi-line script string
 *
 * Return: 1 if @cmdline contains a bare `@calc` line, 0 otherwise
 */
static int cmdline_has_bare_calc(const char *cmdline)
{
    if (cmdline == NULL)
    {
        return 0;
    }
    const char *p = cmdline;
    while (*p != '\0')
    {
        const char *nl   = strchr(p, '\n');
        size_t      llen = (nl != NULL) ? (size_t) (nl - p) : strlen(p);
        const char *s    = p;
        const char *end  = p + llen;
        while (s < end && (*s == ' ' || *s == '\t'))
        {
            s++;
        }
        if ((size_t) (end - s) >= 5 && strncmp(s, "@calc", 5) == 0)
        {
            const char *q = s + 5;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\r'))
            {
                q++;
            }
            if (q == end || *q == '#')
            {
                return 1;
            }
        }
        if (nl == NULL)
        {
            break;
        }
        p = nl + 1;
    }
    return 0;
}

/**
 * exec_calc_repl_expr - Evaluate @expr in-process when eligible, else via `@calc --repl`
 * @expr:         Calculator expression string
 * @out_exit_req: Optional output set if bash requested exit
 *
 * Return: Evaluation status (0 on success)
 */
static int exec_calc_repl_expr(const char *expr, int *out_exit_req)
{
    size_t elen = strlen(expr);
    if (elen >= 2 &&
        ((expr[0] == '"' && expr[elen - 1] == '"') || (expr[0] == '\'' && expr[elen - 1] == '\'')))
    {
        char   q      = expr[0];
        int    closed = 0;
        size_t k      = 1;
        for (; k + 1 < elen; k++)
        {
            if (expr[k] == '\\' && q == '"' && k + 2 < elen)
            {
                k++;
                continue;
            }
            if (expr[k] == q)
            {
                closed = 1;
                break;
            }
        }
        if (!closed)
        {
            char *inner = strndup(expr + 1, elen - 2);
            if (inner != NULL)
            {
                int rc = exec_calc_repl_expr(inner, out_exit_req);
                free(inner);
                return rc;
            }
        }
    }

    if (!dcerrorexit && milk_bash_engine_session()[0] == '\0' &&
        milk_server_can_exec_calc_inproc(expr))
    {
        int rc          = milk_server_exec_calc_repl_inproc(expr);
        cli_last_retval = rc;
        return rc;
    }

    size_t cap = elen * 2 + 32;
    char  *cmd = (char *) malloc(cap);
    if (cmd == NULL)
    {
        return 2;
    }
    size_t pos = (size_t) sprintf(cmd, "@calc --repl \"");
    for (size_t i = 0; i < elen; i++)
    {
        if (expr[i] == '"' || expr[i] == '\\' || expr[i] == '`')
        {
            cmd[pos++] = '\\';
        }
        cmd[pos++] = expr[i];
    }
    cmd[pos++] = '"';
    cmd[pos]   = '\0';
    int rc     = milk_bash_engine_exec_raw(cmd, out_exit_req);
    free(cmd);
    return rc;
}

/**
 * exec_calc_mode_line - Execute a single trimmed line while `@calc` REPL mode is active
 * @trimmed:      Comment-stripped, whitespace-trimmed line
 * @out_exit_req: Output flag set if the command requested shell exit
 *
 * Return: Execution status of the line (0 on success)
 */
static int exec_calc_mode_line(const char *trimmed, int *out_exit_req)
{
    if (strcmp(trimmed, "exit") == 0 || strcmp(trimmed, "quit") == 0)
    {
        cli_calc_mode   = 0;
        cli_last_retval = 0;
        return 0;
    }
    if (trimmed[0] == '@')
    {
        if (strncmp(trimmed, "@calc", 5) == 0 && (trimmed[5] == ' ' || trimmed[5] == '\t'))
        {
            const char *rest = trimmed + 5;
            while (*rest == ' ' || *rest == '\t')
            {
                rest++;
            }
            if (strncmp(rest, "-h", 2) == 0 || strncmp(rest, "--help", 6) == 0 ||
                strncmp(rest, "-v", 2) == 0 || strncmp(rest, "-a", 2) == 0 ||
                strncmp(rest, "-l", 2) == 0 || strncmp(rest, "--lib", 5) == 0 ||
                strncmp(rest, "--list", 6) == 0 || strncmp(rest, "--engines", 9) == 0 ||
                strncmp(rest, "--tokens", 8) == 0 || strncmp(rest, "--repl", 6) == 0 ||
                strncmp(rest, "--explain", 9) == 0)
            {
                return milk_bash_engine_exec_raw(trimmed, out_exit_req);
            }
            return exec_calc_repl_expr(rest, out_exit_req);
        }
        return milk_bash_engine_exec_raw(trimmed, out_exit_req);
    }
    return exec_calc_repl_expr(trimmed, out_exit_req);
}

/**
 * flush_bash_buf - Execute and reset any accumulated non-`@calc` bash lines
 * @bash_buf:     Buffer holding accumulated bash lines
 * @rc:           In/out command status
 * @out_exit_req: Optional output set to 1 if `exit` was requested
 *
 * Return: 1 if execution should stop (exit requested or `dcerrorexit` failure), 0 otherwise
 */
static int flush_bash_buf(sbuf_t *bash_buf, int *rc, int *out_exit_req)
{
    if (bash_buf->data == NULL || bash_buf->len == 0)
    {
        return 0;
    }
    int exreq         = 0;
    *rc               = milk_bash_engine_exec_raw(bash_buf->data, &exreq);
    bash_buf->len     = 0;
    bash_buf->data[0] = '\0';
    if (exreq && out_exit_req != NULL)
    {
        *out_exit_req = 1;
    }
    return exreq || (dcerrorexit && *rc != 0);
}

/**
 * milk_bash_engine_exec - Execute a command line or script block in the bash co-process
 * @cmdline:      Command string or multi-line script to execute
 * @out_exit_req: Optional output flag set to 1 if `exit` or `quit` was requested
 *
 * Guards against recursive shell invocation from inside a milk command handler,
 * dispatches bare `@calc` mode toggles and calculator REPL lines when active,
 * accumulates contiguous non-`@calc` bash lines so multi-line constructs remain
 * intact, and delegates shell execution to `milk_bash_engine_exec_raw()`.
 *
 * Return: Exit status of the executed command (0 on success)
 */
int milk_bash_engine_exec(const char *cmdline, int *out_exit_req)
{
    if (g_milk_bash_exec_depth > 0)
    {
        fprintf(stderr, "milk: cannot run a shell command from inside a milk command "
                        "(use bash constructs, e.g. 'time cmd' or a while loop)\n");
        if (out_exit_req != NULL)
        {
            *out_exit_req = 0;
        }
        return 1;
    }
    if (out_exit_req != NULL)
    {
        *out_exit_req = 0;
    }
    if (cmdline == NULL)
    {
        return 0;
    }

    g_milk_bash_exec_depth++;
    int rc = 0;
    if (!cli_calc_mode && !cmdline_has_bare_calc(cmdline))
    {
        rc = milk_bash_engine_exec_raw(cmdline, out_exit_req);
        g_milk_bash_exec_depth--;
        dcexitcode = rc;
        return rc;
    }

    sbuf_t bash_buf;
    sbuf_init(&bash_buf);
    for (const char *p = cmdline; *p != '\0';)
    {
        const char *nl   = strchr(p, '\n');
        size_t      llen = (nl != NULL) ? (size_t) (nl - p) : strlen(p);
        char       *raw  = strndup(p, llen);
        if (raw == NULL)
        {
            rc = 1;
            break;
        }
        char *trimmed   = trim_calc_line(raw);
        int   is_toggle = (strcmp(trimmed, "@calc") == 0);
        int   exreq     = 0;

        if (is_toggle)
        {
            free(raw);
            if (!cli_calc_mode && flush_bash_buf(&bash_buf, &rc, out_exit_req))
            {
                break;
            }
            cli_calc_mode   = !cli_calc_mode;
            cli_last_retval = 0;
            rc              = 0;
        }
        else if (!cli_calc_mode)
        {
            free(raw);
            sbuf_append_mem(&bash_buf, p, llen);
            sbuf_append_ch(&bash_buf, '\n');
        }
        else
        {
            if (trimmed[0] != '\0')
            {
                rc = exec_calc_mode_line(trimmed, &exreq);
            }
            free(raw);
            if (exreq)
            {
                if (out_exit_req != NULL)
                {
                    *out_exit_req = 1;
                }
                break;
            }
        }
        if (nl == NULL)
        {
            break;
        }
        p = nl + 1;
    }
    flush_bash_buf(&bash_buf, &rc, out_exit_req);
    free(bash_buf.data);
    g_milk_bash_exec_depth--;
    dcexitcode = rc;
    return rc;
}

/**
 * milk_bash_engine_exec_file - Read, translate, and execute a script file in the bash co-process
 * @filename:     Path to script file
 * @out_exit_req: Optional: set to 1 if script invoked `exit`/`quit`
 *
 * Return: Exit status of the script (0 = success)
 */
int milk_bash_engine_exec_file(const char *filename, int *out_exit_req)
{
    if (filename == NULL || filename[0] == '\0')
    {
        fprintf(stderr, "Usage: source <filename>\n");
        return 1;
    }
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "source: %s: %s\n", filename, strerror(errno));
        return 1;
    }
    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fclose(fp);
        return 1;
    }
    long fsz = ftell(fp);
    if (fsz < 0)
    {
        fclose(fp);
        return 1;
    }
    rewind(fp);

    char *buf = (char *) malloc((size_t) fsz + 1);
    if (buf == NULL)
    {
        fclose(fp);
        return 1;
    }
    size_t nread = fread(buf, 1, (size_t) fsz, fp);
    buf[nread]   = '\0';
    fclose(fp);

    const char *base = strrchr(filename, '/');
    milk_bash_engine_set_label((base != NULL) ? base + 1 : filename);
    int rc = milk_bash_engine_exec(buf, out_exit_req);
    milk_bash_engine_set_label("milk");
    free(buf);
    return rc;
}
