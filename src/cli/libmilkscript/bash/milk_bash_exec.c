// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_exec.c
 * @brief Running a command line or a file in the bash co-process, and calc mode (see
 *        milk_bash.h).
 *
 * Text with no bare `@calc` line, outside calc mode, goes to bash in one piece. Otherwise it is
 * taken line by line: a bare `@calc` switches calc mode, lines outside it go to bash together,
 * and in calc mode a line is a calculator statement (a line that starts with `@` runs as a
 * keyword; `exit` or `quit` leaves the mode). A statement is evaluated in this process when it
 * needs nothing from bash (milk_server_can_exec_calc_inproc()), bash uses this process's session
 * and -e is off; else it runs in bash as `@calc --repl "<line>"`. milk_bash_exec_raw() refuses an
 * unterminated quote, translates `@{...}` and adds `set -e` under -e (exit on error); around a
 * command that may change shell variables, it bumps the server's bash epoch, so calc mode reads
 * them again. A shell command started from inside a milk command is refused.
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
#include "milk_bash.h"
#include "milk_bash_internal.h"
#include "milk_server.h"
#include "milk_shell_lex.h"

/**
 * milk_bash_exec_raw - Translate and execute a raw bash block in the co-process
 * @cmdline:      Command line or multi-line script text
 * @out_exit_req: Optional output set to 1 if script invoked `exit`
 *
 * Return: Exit status of the bash command (0 = success)
 */
int milk_bash_exec_raw(const char *cmdline, int *out_exit_req)
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
 * trim_calc_line - Strip the shell comment and the leading/trailing whitespace of @s in-place
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
    size_t    len = strlen(s);
    msl_map_t m;
    if (msl_scan(s, len, &m) == 0)
    {
        len = 0;
        while (s[len] != '\0' && m.ctx[len] != MSL_COMMENT)
        {
            len++;
        }
        msl_free(&m);
    }
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t' || s[len - 1] == '\r'))
    {
        len--;
    }
    s[len] = '\0';
    return s;
}

/**
 * cmdline_has_bare_calc - Check whether any line in @cmdline is a bare `@calc` mode toggle
 * @cmdline: Command line or multi-line script string
 *
 * Return: 1 if @cmdline contains a bare `@calc` line outside quotes/heredocs, 0 otherwise
 */
static int cmdline_has_bare_calc(const char *cmdline)
{
    if (cmdline == NULL)
    {
        return 0;
    }
    msl_map_t   m   = { 0 };
    int         has = (msl_scan(cmdline, strlen(cmdline), &m) == 0);
    int         ret = 0;
    const char *p   = cmdline;
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
        if ((size_t) (end - s) >= 5 && strncmp(s, "@calc", 5) == 0 &&
            (!has || (m.ctx[s - cmdline] == MSL_CODE && m.depth[s - cmdline] == 0)))
        {
            const char *q = s + 5;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\r'))
            {
                q++;
            }
            if (q == end || *q == '#')
            {
                ret = 1;
                break;
            }
        }
        if (nl == NULL)
        {
            break;
        }
        p = nl + 1;
    }
    if (has)
    {
        msl_free(&m);
    }
    return ret;
}

/**
 * is_one_quoted_word - 1 if the @n bytes at @s are one quoted shell word ("a + b" or 'a + b')
 */
static int is_one_quoted_word(const char *s, size_t n)
{
    msl_map_t m;
    int       one = 0;
    if (n >= 2 && (s[0] == '"' || s[0] == '\'') && msl_scan(s, n - 1, &m) == 0)
    {
        one = m.unterminated && m.open_at == 1; /* its first quote is open up to the last byte */
        msl_free(&m);
    }
    if (one)
    {
        one = (msl_scan(s, n, &m) == 0 && !m.unterminated); /* which closes it */
        msl_free(&m);
    }
    return one;
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
    if (is_one_quoted_word(expr, elen))
    {
        char *inner = strndup(expr + 1, elen - 2);
        if (inner != NULL)
        {
            int rc = exec_calc_repl_expr(inner, out_exit_req);
            free(inner);
            return rc;
        }
    }

    if (!dcerrorexit && milk_bash_session()[0] == '\0' && milk_server_can_exec_calc_inproc(expr))
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
    int rc     = milk_bash_exec_raw(cmd, out_exit_req);
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
                strncmp(rest, "--list", 6) == 0 || strncmp(rest, "--tokens", 8) == 0 ||
                strncmp(rest, "--repl", 6) == 0 || strncmp(rest, "--explain", 9) == 0)
            {
                return milk_bash_exec_raw(trimmed, out_exit_req);
            }
            return exec_calc_repl_expr(rest, out_exit_req);
        }
        return milk_bash_exec_raw(trimmed, out_exit_req);
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
    *rc               = milk_bash_exec_raw(bash_buf->data, &exreq);
    bash_buf->len     = 0;
    bash_buf->data[0] = '\0';
    if (exreq && out_exit_req != NULL)
    {
        *out_exit_req = 1;
    }
    return exreq || (dcerrorexit && *rc != 0);
}

/**
 * milk_bash_exec - Execute a command line or script block in the bash co-process
 * @cmdline:      Command string or multi-line script to execute
 * @out_exit_req: Optional output flag set to 1 if `exit` or `quit` was requested
 *
 * Guards against recursive shell invocation from inside a milk command handler,
 * dispatches bare `@calc` mode toggles and calculator REPL lines when active,
 * accumulates contiguous non-`@calc` bash lines so multi-line constructs remain
 * intact, and delegates shell execution to `milk_bash_exec_raw()`.
 *
 * Return: Exit status of the executed command (0 on success)
 */
int milk_bash_exec(const char *cmdline, int *out_exit_req)
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
        rc = milk_bash_exec_raw(cmdline, out_exit_req);
        g_milk_bash_exec_depth--;
        dcexitcode = rc;
        return rc;
    }

    sbuf_t bash_buf;
    sbuf_t calc_buf;
    sbuf_init(&bash_buf);
    sbuf_init(&calc_buf);
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
        if (is_toggle && !cli_calc_mode && bash_buf.len > 0)
        {
            msl_map_t m;
            if (msl_scan(bash_buf.data, bash_buf.len, &m) == 0)
            {
                if (m.unterminated || m.ctx[bash_buf.len] != MSL_CODE)
                {
                    is_toggle = 0;
                }
                msl_free(&m);
            }
        }
        int exreq = 0;

        if (is_toggle)
        {
            free(raw);
            if (cli_calc_mode && calc_buf.len > 0)
            {
                rc               = exec_calc_mode_line(calc_buf.data, &exreq);
                calc_buf.len     = 0;
                calc_buf.data[0] = '\0';
                if (exreq)
                {
                    if (out_exit_req != NULL)
                    {
                        *out_exit_req = 1;
                    }
                    break;
                }
            }
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
                if (calc_buf.len > 0)
                {
                    sbuf_append_ch(&calc_buf, '\n');
                }
                sbuf_append_str(&calc_buf, trimmed);
                if (!milk_calc_incomplete(calc_buf.data))
                {
                    rc               = exec_calc_mode_line(calc_buf.data, &exreq);
                    calc_buf.len     = 0;
                    calc_buf.data[0] = '\0';
                }
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
    if (calc_buf.len > 0)
    {
        int exreq = 0;
        rc        = exec_calc_mode_line(calc_buf.data, &exreq);
        if (exreq && out_exit_req != NULL)
        {
            *out_exit_req = 1;
        }
    }
    flush_bash_buf(&bash_buf, &rc, out_exit_req);
    free(calc_buf.data);
    free(bash_buf.data);
    g_milk_bash_exec_depth--;
    dcexitcode = rc;
    return rc;
}

/**
 * milk_bash_exec_file - Read, translate, and execute a script file in the bash co-process
 * @filename:     Path to script file
 * @out_exit_req: Optional: set to 1 if script invoked `exit`/`quit`
 *
 * Return: Exit status of the script (0 = success)
 */
int milk_bash_exec_file(const char *filename, int *out_exit_req)
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
    milk_bash_set_label((base != NULL) ? base + 1 : filename);
    int rc = milk_bash_exec(buf, out_exit_req);
    milk_bash_set_label("milk");
    free(buf);
    return rc;
}
