// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_engine.c
 * @brief Supervised bash co-process and Tier-2 shorthand translator for milk-cli.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_signals.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "milk_bash_engine.h"
#include "milk_server.h"

/* ============================================================
 *  Dynamic String Buffer Helper
 * ============================================================ */

typedef struct
{
    char  *data;
    size_t len;
    size_t cap;
} sbuf_t;

static void sbuf_init(sbuf_t *b)
{
    b->cap  = 512;
    b->len  = 0;
    b->data = (char *) malloc(b->cap);
    if (b->data != NULL)
    {
        b->data[0] = '\0';
    }
}

static void sbuf_reserve(sbuf_t *b, size_t extra)
{
    if (b->data == NULL)
    {
        return;
    }
    if (b->len + extra + 1 > b->cap)
    {
        size_t ncap = b->cap * 2;
        while (ncap < b->len + extra + 1)
        {
            ncap *= 2;
        }
        char *nd = (char *) realloc(b->data, ncap);
        if (nd == NULL)
        {
            return;
        }
        b->data = nd;
        b->cap  = ncap;
    }
}

static void sbuf_append_mem(sbuf_t *b, const char *s, size_t n)
{
    if (b->data == NULL || s == NULL || n == 0)
    {
        return;
    }
    sbuf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void sbuf_append_str(sbuf_t *b, const char *s)
{
    if (s != NULL)
    {
        sbuf_append_mem(b, s, strlen(s));
    }
}

static void sbuf_append_ch(sbuf_t *b, char c)
{
    sbuf_reserve(b, 1);
    if (b->data != NULL)
    {
        b->data[b->len++] = c;
        b->data[b->len]   = '\0';
    }
}

static void sbuf_append_dq_escaped(sbuf_t *b, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (s[i] == '"' || s[i] == '\\' || s[i] == '`')
        {
            sbuf_append_ch(b, '\\');
        }
        sbuf_append_ch(b, s[i]);
    }
}

/* ============================================================
 *  Engine Mode Selection
 * ============================================================ */

int milk_is_bash_engine_enabled(void)
{
    const char *env = getenv("MILKCLI_ENGINE");
    if (env != NULL &&
        (strcmp(env, "native") == 0 || strcmp(env, "legacy") == 0 || strcmp(env, "0") == 0))
    {
        return 0;
    }
    return 1;
}

static const char *get_sigil(void)
{
    const char *s = getenv("MILKCLI_SIGIL");
    if (s == NULL || s[0] == '\0')
    {
        return "@";
    }
    return s;
}

/* ============================================================
 *  Entity & Command Recognition Helpers
 * ============================================================ */

static int is_entity_prefix(const char *s, size_t *out_len)
{
    static const char *const pfx[] = { "s.",   "f.",      "p.",    "q.", "fps.",
                                       "seq.", "stream.", "proc.", NULL };
    for (int i = 0; pfx[i] != NULL; i++)
    {
        size_t plen = strlen(pfx[i]);
        if (strncmp(s, pfx[i], plen) == 0 && (isalnum((unsigned char) s[plen]) || s[plen] == '_'))
        {
            if (out_len != NULL)
            {
                *out_len = plen;
            }
            return 1;
        }
    }
    return 0;
}

static size_t scan_entity_path(const char *s, int *has_eq)
{
    *has_eq     = 0;
    size_t plen = 0;
    if (!is_entity_prefix(s, &plen))
    {
        return 0;
    }
    size_t i    = plen;
    int    dots = 1;
    while (s[i] != '\0' && (isalnum((unsigned char) s[i]) || s[i] == '_' || s[i] == '-' ||
                            s[i] == '.' || s[i] == '*'))
    {
        if (s[i] == '.')
        {
            dots++;
        }
        i++;
    }
    if (dots < 2)
    {
        return 0;
    }
    if (s[i] == '=')
    {
        *has_eq = 1;
    }
    return i;
}

static int is_bash_reserved_or_builtin(const char *w)
{
    static const char *const kw[] = {
        "if",         "then",     "elif",     "else",     "fi",       "for",      "in",
        "while",      "until",    "do",       "done",     "case",     "esac",     "select",
        "function",   "time",     "coproc",   "{",        "}",        "!",        "[[",
        "]]",         "[",        "]",        "echo",     "printf",   "cd",       "pwd",
        "read",       "export",   "unset",    "shift",    "alias",    "unalias",  "source",
        ".",          "exit",     "quit",     "return",   "break",    "continue", "local",
        "declare",    "typeset",  "readonly", "let",      "eval",     "exec",     "set",
        "shopt",      "trap",     "type",     "command",  "builtin",  "enable",   "wait",
        "kill",       "jobs",     "bg",       "fg",       "true",     "false",    "test",
        "sleep",      "rm",       "ls",       "cat",      "grep",     "awk",      "sed",
        "find",       "xargs",    "sort",     "head",     "tail",     "wc",       "cut",
        "tr",         "mkdir",    "rmdir",    "cp",       "mv",       "ln",       "chmod",
        "touch",      "date",     "uname",    "hostname", "whoami",   "id",       "env",
        "milkcall",   "milkcalc", "milkget",  "milkset",  "milktest", "milkgrab", "milkwaitfor",
        "milkexists", NULL
    };
    for (int i = 0; kw[i] != NULL; i++)
    {
        if (strcmp(w, kw[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static int is_registered_milk_command(const char *w)
{
    if (w == NULL || w[0] == '\0' || is_bash_reserved_or_builtin(w))
    {
        return 0;
    }

    static const char *const extra_cmds[] = { "help",
                                              "imgstat",
                                              "info.imgstat",
                                              "mk2Drnd",
                                              "imgen.mk2Drnd",
                                              "procstat",
                                              "procctl",
                                              "procwait",
                                              "dpdigits",
                                              "assigncheck",
                                              "assert",
                                              "on_update",
                                              "on_fpschange",
                                              "lm?",
                                              "m?",
                                              "cmd?",
                                              "f?",
                                              "?",
                                              "vars",
                                              "savescript",
                                              "savehistory",
                                              NULL };
    for (int i = 0; extra_cmds[i] != NULL; i++)
    {
        if (strcmp(w, extra_cmds[i]) == 0)
        {
            return 1;
        }
    }

    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        const char *k = data.cmd[i].key;
        if (k[0] == '\0')
        {
            continue;
        }
        if (strcmp(k, w) == 0)
        {
            return 1;
        }
        const char *dot = strchr(k, '.');
        if (dot != NULL && strcmp(dot + 1, w) == 0)
        {
            return 1;
        }
        int midx = data.cmd[i].moduleindex;
        if (midx >= 0 && strchr(w, '.') != NULL)
        {
            const char *wdot = strchr(w, '.');
            size_t      mlen = (size_t) (wdot - w);
            const char *ms   = data.module[midx].shortname;
            const char *mn   = data.module[midx].name;
            if (((strlen(ms) == mlen && strncmp(ms, w, mlen) == 0) ||
                 (strlen(mn) == mlen && strncmp(mn, w, mlen) == 0)) &&
                strcmp(k, wdot + 1) == 0)
            {
                return 1;
            }
        }
    }

    /* Also treat <known_module>.<anything> (e.g. mem.lisim typo) as a milk command
     * so milk-cli reports unknown command + Did you mean */
    const char *wdot = strchr(w, '.');
    if (wdot != NULL && wdot > w && strchr(w, '/') == NULL)
    {
        size_t                   mlen       = (size_t) (wdot - w);
        static const char *const std_mods[] = { "mem",    "info",   "imgen",  "imfilt", "imfmt",
                                                "fft",    "linalg", "stat",   "psf",    "zern",
                                                "kdtree", "linopt", "linAR",  "fps",    "seq",
                                                "cacao",  "arith",  "iofits", "tools",  NULL };
        for (int m = 0; std_mods[m] != NULL; m++)
        {
            if (strlen(std_mods[m]) == mlen && strncmp(std_mods[m], w, mlen) == 0)
            {
                return 1;
            }
        }
        for (int m = 0; m < data.NBmodule; m++)
        {
            const char *ms = data.module[m].shortname;
            const char *mn = data.module[m].name;
            if ((ms[0] != '\0' && strlen(ms) == mlen && strncmp(ms, w, mlen) == 0) ||
                (mn[0] != '\0' && strlen(mn) == mlen && strncmp(mn, w, mlen) == 0))
            {
                return 1;
            }
        }
        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            const char *kdot = strchr(data.cmd[i].key, '.');
            if (kdot != NULL && (size_t) (kdot - data.cmd[i].key) == mlen &&
                strncmp(data.cmd[i].key, w, mlen) == 0)
            {
                return 1;
            }
        }
    }

    return 0;
}

/* ============================================================
 *  Inline Expansion Helpers (@{...} and [ -S/-F/-P ... ])
 * ============================================================ */

static void expand_inline_constructs(const char *in, size_t len, sbuf_t *out)
{
    const char *sigil  = get_sigil();
    size_t      siglen = strlen(sigil);
    int         in_sq  = 0;
    int         in_dq  = 0;

    for (size_t i = 0; i < len;)
    {
        char c = in[i];
        if (c == '\\' && !in_sq && i + 1 < len)
        {
            sbuf_append_mem(out, in + i, 2);
            i += 2;
            continue;
        }
        if (c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
            sbuf_append_ch(out, c);
            i++;
            continue;
        }
        if (c == '"' && !in_sq)
        {
            in_dq = !in_dq;
            sbuf_append_ch(out, c);
            i++;
            continue;
        }
        if (in_sq)
        {
            sbuf_append_ch(out, c);
            i++;
            continue;
        }

        /* Comment start outside quotes */
        if (!in_dq && c == '#' && (i == 0 || isspace((unsigned char) in[i - 1])))
        {
            sbuf_append_mem(out, in + i, len - i);
            break;
        }

        /* @{<expr>} -> $(@calc "<expr>") */
        if ((strncmp(in + i, sigil, siglen) == 0 && i + siglen < len && in[i + siglen] == '{') ||
            (in[i] == '@' && i + 1 < len && in[i + 1] == '{'))
        {
            size_t adv =
                (strncmp(in + i, sigil, siglen) == 0 && in[i + siglen] == '{') ? (siglen + 1) : 2;
            size_t start = i + adv;
            size_t j     = start;
            int    depth = 1;
            while (j < len && depth > 0)
            {
                if (in[j] == '{')
                {
                    depth++;
                }
                else if (in[j] == '}')
                {
                    depth--;
                    if (depth == 0)
                    {
                        break;
                    }
                }
                j++;
            }
            if (j < len && depth == 0)
            {
                sbuf_append_str(out, "$(");
                sbuf_append_str(out, sigil);
                sbuf_append_str(out, "calc \"");
                sbuf_append_dq_escaped(out, in + start, j - start);
                sbuf_append_str(out, "\")");
                i = j + 1;
                continue;
            }
        }

        /* Inside double quotes: @s.im.prop / @f.fps.param -> $(@get s.im.prop) */
        if (in_dq && (strncmp(in + i, sigil, siglen) == 0 || in[i] == '@'))
        {
            size_t adv    = (strncmp(in + i, sigil, siglen) == 0) ? siglen : 1;
            int    has_eq = 0;
            size_t elen   = scan_entity_path(in + i + adv, &has_eq);
            if (elen > 0 && !has_eq)
            {
                /* Require at least 2 dots (e.g. s.im.prop) inside quotes */
                int ndots = 0;
                for (size_t d = 0; d < elen; d++)
                {
                    if (in[i + adv + d] == '.')
                    {
                        ndots++;
                    }
                }
                if (ndots >= 2)
                {
                    sbuf_append_str(out, "$(");
                    sbuf_append_str(out, sigil);
                    sbuf_append_str(out, "get ");
                    sbuf_append_mem(out, in + i + adv, elen);
                    sbuf_append_ch(out, ')');
                    i += adv + elen;
                    continue;
                }
            }
        }

        /* [ -S name ] / [ ! -S name ] / [ -F name ] / [ -P name ] outside quotes */
        if (!in_dq && c == '[' &&
            (i == 0 || isspace((unsigned char) in[i - 1]) || in[i - 1] == ';') && i + 1 < len &&
            isspace((unsigned char) in[i + 1]))
        {
            size_t j = i + 1;
            while (j < len && isspace((unsigned char) in[j]))
            {
                j++;
            }
            int neg = 0;
            if (j + 1 < len && in[j] == '!' && isspace((unsigned char) in[j + 1]))
            {
                neg = 1;
                j++;
                while (j < len && isspace((unsigned char) in[j]))
                {
                    j++;
                }
            }
            if (j + 2 < len && in[j] == '-' &&
                (in[j + 1] == 'S' || in[j + 1] == 'F' || in[j + 1] == 'P') &&
                isspace((unsigned char) in[j + 2]))
            {
                char   flag = in[j + 1];
                size_t k    = j + 2;
                while (k < len && isspace((unsigned char) in[k]))
                {
                    k++;
                }
                size_t nstart = k;
                while (k < len && !isspace((unsigned char) in[k]) && in[k] != ']')
                {
                    k++;
                }
                size_t nend = k;
                while (k < len && isspace((unsigned char) in[k]))
                {
                    k++;
                }
                if (nend > nstart && k < len && in[k] == ']')
                {
                    if (neg)
                    {
                        sbuf_append_str(out, "! ");
                    }
                    sbuf_append_str(out, sigil);
                    sbuf_append_str(out, "exists ");
                    sbuf_append_ch(out, (flag == 'S') ? 's' : ((flag == 'F') ? 'f' : 'p'));
                    sbuf_append_ch(out, '.');
                    sbuf_append_mem(out, in + nstart, nend - nstart);
                    i = k + 1;
                    continue;
                }
            }
            /* Rewrite single-bracket [ ... =~ ... ] to [[ ... =~ ... ]] */
            {
                size_t k        = i + 1;
                int    sq_inner = 0;
                int    dq_inner = 0;
                int    has_re   = 0;
                while (k < len)
                {
                    if (in[k] == '\'' && !dq_inner)
                    {
                        sq_inner = !sq_inner;
                    }
                    else if (in[k] == '"' && !sq_inner)
                    {
                        dq_inner = !dq_inner;
                    }
                    else if (!sq_inner && !dq_inner)
                    {
                        if (in[k] == ';')
                        {
                            break;
                        }
                        if (k + 3 < len && isspace((unsigned char) in[k]) && in[k + 1] == '=' &&
                            in[k + 2] == '~' && isspace((unsigned char) in[k + 3]))
                        {
                            has_re = 1;
                        }
                        if (in[k] == ']' && isspace((unsigned char) in[k - 1]))
                        {
                            if (has_re)
                            {
                                sbuf_append_str(out, "[[");
                                sbuf_append_mem(out, in + i + 1, k - (i + 1));
                                sbuf_append_str(out, "]]");
                                i = k + 1;
                                goto next_inline_char;
                            }
                            break;
                        }
                    }
                    k++;
                }
            }
        }

        sbuf_append_ch(out, c);
        i++;
    next_inline_char:;
    }
}

/* ============================================================
 *  Statement-Level Translator
 * ============================================================ */

static int is_entity_taking_cmd(const char *cmd, const char *sigil)
{
    size_t siglen = strlen(sigil);
    if (strncmp(cmd, sigil, siglen) == 0)
    {
        cmd += siglen;
    }
    else if (cmd[0] == '@')
    {
        cmd++;
    }
    return (strcmp(cmd, "get") == 0 || strcmp(cmd, "set") == 0 || strcmp(cmd, "exists") == 0 ||
            strcmp(cmd, "running") == 0 || strcmp(cmd, "waitfor") == 0 ||
            strcmp(cmd, "waitany") == 0 || strcmp(cmd, "on") == 0 ||
            strcmp(cmd, "streamcat") == 0 || strcmp(cmd, "streamwrite") == 0);
}

static void append_args_with_quoting(const char *args,
                                     size_t      len,
                                     int         is_milk_cmd,
                                     int         preserve_entity_arg,
                                     sbuf_t     *out)
{
    const char *sigil  = get_sigil();
    size_t      siglen = strlen(sigil);
    size_t      i      = 0;

    while (i < len)
    {
        if (isspace((unsigned char) args[i]))
        {
            sbuf_append_ch(out, args[i]);
            i++;
            continue;
        }
        if (args[i] == '#')
        {
            sbuf_append_mem(out, args + i, len - i);
            break;
        }

        /* Scan one argument token */
        size_t tstart     = i;
        int    in_sq      = 0;
        int    in_dq      = 0;
        int    has_quotes = 0;
        int    br_depth   = 0;
        int    has_slice  = 0;
        int    has_glob   = 0;

        while (i < len)
        {
            char c = args[i];
            if (!in_sq && !in_dq && br_depth == 0 && isspace((unsigned char) c))
            {
                break;
            }
            if (c == '\\' && !in_sq && i + 1 < len)
            {
                i += 2;
                continue;
            }
            if (c == '\'' && !in_dq)
            {
                in_sq      = !in_sq;
                has_quotes = 1;
            }
            else if (c == '"' && !in_sq)
            {
                in_dq      = !in_dq;
                has_quotes = 1;
            }
            else if (!in_sq && !in_dq)
            {
                if (c == '[')
                {
                    br_depth++;
                    has_slice = 1;
                }
                else if (c == ']' && br_depth > 0)
                {
                    br_depth--;
                }
                else if (c == '*' || c == '?')
                {
                    has_glob = 1;
                }
            }
            i++;
        }

        size_t tlen = i - tstart;
        if (!has_quotes && !preserve_entity_arg)
        {
            /* Check for inline @s.im.prop / @f.fps.param */
            const char *tok  = args + tstart;
            size_t      toff = 0;
            if (tlen > siglen && strncmp(tok, sigil, siglen) == 0)
            {
                toff = siglen;
            }
            else if (tlen > 1 && tok[0] == '@')
            {
                toff = 1;
            }
            if (toff > 0)
            {
                int    has_eq = 0;
                size_t elen   = scan_entity_path(tok + toff, &has_eq);
                if (elen == tlen - toff && !has_eq)
                {
                    sbuf_append_str(out, "$(");
                    sbuf_append_str(out, sigil);
                    sbuf_append_str(out, "get ");
                    sbuf_append_mem(out, tok + toff, elen);
                    sbuf_append_ch(out, ')');
                    continue;
                }
            }
        }

        if (is_milk_cmd && !has_quotes && (has_slice || has_glob))
        {
            sbuf_append_ch(out, '"');
            sbuf_append_dq_escaped(out, args + tstart, tlen);
            sbuf_append_ch(out, '"');
        }
        else
        {
            sbuf_append_mem(out, args + tstart, tlen);
        }
    }
}

static int is_calc_assignment_stmt(const char *s, size_t len, int *out_simple_arith)
{
    *out_simple_arith = 0;
    size_t i          = 0;
    if (i >= len || (!isalpha((unsigned char) s[i]) && s[i] != '_'))
    {
        return 0;
    }
    while (i < len && (isalnum((unsigned char) s[i]) || s[i] == '_'))
    {
        i++;
    }

    int has_slice            = 0;
    int slice_is_2d_or_range = 0;
    if (i < len && s[i] == '[')
    {
        has_slice = 1;
        i++;
        while (i < len && s[i] != ']')
        {
            if (s[i] == ',' || s[i] == ':')
            {
                slice_is_2d_or_range = 1;
            }
            i++;
        }
        if (i >= len || s[i] != ']')
        {
            /* Unclosed bracket before '=' -> only treat as calc if space before '=' earlier */
            return 0;
        }
        i++;
    }

    int ws_before = 0;
    while (i < len && (s[i] == ' ' || s[i] == '\t'))
    {
        ws_before = 1;
        i++;
    }

    int op_len = 0;
    if (i + 1 < len && (s[i] == '+' || s[i] == '-' || s[i] == '*' || s[i] == '/') &&
        s[i + 1] == '=')
    {
        op_len = 2;
    }
    else if (i < len && s[i] == '=' && (i + 1 >= len || s[i + 1] != '='))
    {
        op_len = 1;
    }
    else
    {
        return 0;
    }

    size_t rhs      = i + (size_t) op_len;
    int    ws_after = 0;
    while (rhs < len && (s[rhs] == ' ' || s[rhs] == '\t'))
    {
        ws_after = 1;
        rhs++;
    }

    /* Check if RHS is pure $(( ... )) so we can emit native bash var=$(( ... )) */
    if (!has_slice && op_len == 1 && rhs + 3 <= len && strncmp(s + rhs, "$((", 3) == 0)
    {
        *out_simple_arith = 1;
        return 1;
    }

    if (ws_before || op_len == 2 || slice_is_2d_or_range)
    {
        return 1;
    }

    /* Check for unspaced _bv=(437.06+58.47)/5 where '(' is followed after ')' by math op */
    if (!has_slice && op_len == 1 && !ws_after && rhs < len && s[rhs] == '(')
    {
        size_t k     = rhs + 1;
        int    depth = 1;
        while (k < len && depth > 0)
        {
            if (s[k] == '(')
            {
                depth++;
            }
            else if (s[k] == ')')
            {
                depth--;
                if (depth == 0)
                {
                    break;
                }
            }
            k++;
        }
        if (k + 1 < len && depth == 0 &&
            (s[k + 1] == '/' || s[k + 1] == '*' || s[k + 1] == '+' || s[k + 1] == '-'))
        {
            return 1;
        }
    }

    return 0;
}

static void translate_single_stmt(const char *stmt, size_t len, int allow_legacy, sbuf_t *out)
{
    const char *sigil  = get_sigil();
    size_t      siglen = strlen(sigil);

    /* Preserve leading whitespace */
    size_t i = 0;
    while (i < len && (stmt[i] == ' ' || stmt[i] == '\t'))
    {
        sbuf_append_ch(out, stmt[i]);
        i++;
    }
    if (i >= len || stmt[i] == '#')
    {
        sbuf_append_mem(out, stmt + i, len - i);
        return;
    }

    /* 1. Legacy '! cmd' shell bypass at statement start (before flow-control keywords) */
    if (allow_legacy && stmt[i] == '!' && (len - i) > 1)
    {
        size_t k = i + 1;
        while (k < len && (stmt[k] == ' ' || stmt[k] == '\t'))
        {
            k++;
        }
        if (k < len && strncmp(stmt + k, sigil, siglen) != 0 && stmt[k] != '@' && stmt[k] != '[')
        {
            sbuf_append_mem(out, stmt + k, len - k);
            return;
        }
    }

    /* Peel off leading shell flow-control prefixes (then, do, else, elif, if, while, until, !) */
    for (;;)
    {
        static const char *const prefixes[] = { "then",  "do", "else", "elif", "while",
                                                "until", "if", "!",    NULL };
        int                      matched    = 0;
        for (int p = 0; prefixes[p] != NULL; p++)
        {
            size_t plen = strlen(prefixes[p]);
            if (i + plen < len && strncmp(stmt + i, prefixes[p], plen) == 0 &&
                (stmt[i + plen] == ' ' || stmt[i + plen] == '\t'))
            {
                sbuf_append_mem(out, stmt + i, plen);
                i += plen;
                while (i < len && (stmt[i] == ' ' || stmt[i] == '\t'))
                {
                    sbuf_append_ch(out, stmt[i]);
                    i++;
                }
                matched = 1;
                break;
            }
        }
        if (!matched)
        {
            break;
        }
    }

    if (i >= len || stmt[i] == '#')
    {
        sbuf_append_mem(out, stmt + i, len - i);
        return;
    }

    /* Trim trailing whitespace Length for statement analysis */
    size_t end = len;
    while (end > i && (stmt[end - 1] == ' ' || stmt[end - 1] == '\t'))
    {
        end--;
    }
    size_t      slen = end - i;
    const char *s    = stmt + i;

    /* 2. Legacy '<stream> |> <cmd>' pipeline */
    if (allow_legacy)
    {
        for (size_t k = 0; k + 1 < slen; k++)
        {
            if (s[k] == '|' && s[k + 1] == '>')
            {
                size_t s_end = k;
                while (s_end > 0 && (s[s_end - 1] == ' ' || s[s_end - 1] == '\t'))
                {
                    s_end--;
                }
                size_t c_start = k + 2;
                while (c_start < slen && (s[c_start] == ' ' || s[c_start] == '\t'))
                {
                    c_start++;
                }
                if (s_end > 0 && c_start < slen)
                {
                    sbuf_append_str(out, "if ");
                    sbuf_append_str(out, sigil);
                    sbuf_append_str(out, "exists s.");
                    sbuf_append_mem(out, s, s_end);
                    sbuf_append_str(out, "; then ");
                    if (strncmp(s + c_start, sigil, siglen) != 0 && s[c_start] != '@')
                    {
                        sbuf_append_str(out, sigil);
                    }
                    sbuf_append_mem(out, s + c_start, slen - c_start);
                    sbuf_append_ch(out, ' ');
                    sbuf_append_mem(out, s, s_end);
                    sbuf_append_str(out, "; else echo \"Error: stream ");
                    sbuf_append_mem(out, s, s_end);
                    sbuf_append_str(out, " does not exist\" >&2; false; fi");
                    sbuf_append_mem(out, stmt + end, len - end);
                    return;
                }
            }
        }
    }

    /* 3. Tier-2 '@ <expr>' calculator statement */
    if ((slen > siglen && strncmp(s, sigil, siglen) == 0 &&
         (s[siglen] == ' ' || s[siglen] == '\t')) ||
        (slen > 1 && s[0] == '@' && (s[1] == ' ' || s[1] == '\t')))
    {
        size_t adv = (strncmp(s, sigil, siglen) == 0 && (s[siglen] == ' ' || s[siglen] == '\t'))
                         ? siglen
                         : 1;
        while (adv < slen && (s[adv] == ' ' || s[adv] == '\t'))
        {
            adv++;
        }
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, "calc \"");
        sbuf_append_dq_escaped(out, s + adv, slen - adv);
        sbuf_append_ch(out, '"');
        sbuf_append_mem(out, stmt + end, len - end);
        return;
    }

    /* 4. Tier-2 '@<entity>.<name>.<field>=<val>' or bare '@<entity>.<name>.<field>' */
    if ((slen > siglen && strncmp(s, sigil, siglen) == 0) || (slen > 1 && s[0] == '@'))
    {
        size_t adv    = (strncmp(s, sigil, siglen) == 0) ? siglen : 1;
        int    has_eq = 0;
        size_t elen   = scan_entity_path(s + adv, &has_eq);
        if (elen > 0)
        {
            if (has_eq)
            {
                size_t vstart = adv + elen + 1;
                sbuf_append_str(out, sigil);
                sbuf_append_str(out, "set ");
                sbuf_append_mem(out, s + adv, elen);
                sbuf_append_ch(out, ' ');
                sbuf_append_mem(out, s + vstart, slen - vstart);
                sbuf_append_mem(out, stmt + end, len - end);
                return;
            }
            if (adv + elen == slen)
            {
                sbuf_append_str(out, sigil);
                sbuf_append_str(out, "get ");
                sbuf_append_mem(out, s + adv, elen);
                sbuf_append_mem(out, stmt + end, len - end);
                return;
            }
        }
    }

    /* 5. Legacy spaced calculator assignment: 'a = 2 + 3', 'a += 5', 'im[1,2] = 42' */
    if (allow_legacy)
    {
        int simple_arith = 0;
        if (is_calc_assignment_stmt(s, slen, &simple_arith))
        {
            if (simple_arith)
            {
                /* Rewrite '_uc = $(( ... ))' -> '_uc=$(( ... ))' */
                size_t k = 0;
                while (k < slen && (isalnum((unsigned char) s[k]) || s[k] == '_'))
                {
                    sbuf_append_ch(out, s[k]);
                    k++;
                }
                while (k < slen && (s[k] == ' ' || s[k] == '\t'))
                {
                    k++;
                }
                sbuf_append_ch(out, '=');
                if (k < slen && s[k] == '=')
                {
                    k++;
                }
                while (k < slen && (s[k] == ' ' || s[k] == '\t'))
                {
                    k++;
                }
                sbuf_append_mem(out, s + k, slen - k);
                sbuf_append_mem(out, stmt + end, len - end);
                return;
            }
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, "calc \"");
            sbuf_append_dq_escaped(out, s, slen);
            sbuf_append_ch(out, '"');
            sbuf_append_mem(out, stmt + end, len - end);
            return;
        }
    }

    /* Extract first word of statement */
    size_t wlen = 0;
    while (wlen < slen && !isspace((unsigned char) s[wlen]))
    {
        wlen++;
    }
    char firstword[256];
    if (wlen >= sizeof(firstword))
    {
        sbuf_append_mem(out, s, len - i);
        return;
    }
    memcpy(firstword, s, wlen);
    firstword[wlen] = '\0';

    /* 6. Already sigiled '@<cmd> ...' */
    if (strncmp(firstword, sigil, siglen) == 0 || firstword[0] == '@')
    {
        int preserve_ent = is_entity_taking_cmd(firstword, sigil);
        sbuf_append_mem(out, s, wlen);
        append_args_with_quoting(s + wlen, len - (i + wlen), 1, preserve_ent, out);
        return;
    }

    /* 7. Legacy milk-cli shorthands when allow_legacy != 0 */
    if (allow_legacy)
    {
        if (strcmp(firstword, "?") == 0)
        {
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, "h");
            append_args_with_quoting(s + wlen, len - (i + wlen), 1, 0, out);
            return;
        }
        if (strcmp(firstword, "m?") == 0 || strcmp(firstword, "lm?") == 0)
        {
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, "hm");
            append_args_with_quoting(s + wlen, len - (i + wlen), 1, 0, out);
            return;
        }
        if (strcmp(firstword, "cmd?") == 0)
        {
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, "hc");
            append_args_with_quoting(s + wlen, len - (i + wlen), 1, 0, out);
            return;
        }
        if (strcmp(firstword, "f?") == 0)
        {
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, "hf");
            append_args_with_quoting(s + wlen, len - (i + wlen), 1, 0, out);
            return;
        }
        if (strcmp(firstword, "assert") == 0 || strcmp(firstword, "on_update") == 0 ||
            strcmp(firstword, "on_fpschange") == 0)
        {
            sbuf_append_str(out, "milkcall ");
            sbuf_append_str(out, firstword);
            size_t astart = wlen;
            while (astart < slen && (s[astart] == ' ' || s[astart] == '\t'))
            {
                astart++;
            }
            if (astart < slen)
            {
                sbuf_append_str(out, " \"");
                sbuf_append_dq_escaped(out, s + astart, slen - astart);
                sbuf_append_ch(out, '"');
            }
            sbuf_append_mem(out, stmt + end, len - end);
            return;
        }
        if (strcmp(firstword, "assigncheck") == 0 || strcmp(firstword, "dpdigits") == 0 ||
            strcmp(firstword, "procstat") == 0 || strcmp(firstword, "procctl") == 0 ||
            strcmp(firstword, "procwait") == 0)
        {
            sbuf_append_str(out, "milkcall ");
            sbuf_append_str(out, firstword);
            append_args_with_quoting(s + wlen, len - (i + wlen), 1, 0, out);
            return;
        }
        if (strcmp(firstword, "time") == 0 && wlen == slen)
        {
            sbuf_append_str(out, "milkcall time");
            sbuf_append_mem(out, stmt + end, len - end);
            return;
        }
        if (strcmp(firstword, "defer") == 0)
        {
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, "defer ");
            size_t astart = wlen;
            while (astart < slen && (s[astart] == ' ' || s[astart] == '\t'))
            {
                astart++;
            }
            translate_single_stmt(s + astart, len - (i + astart), allow_legacy, out);
            return;
        }
        if (strcmp(firstword, "alias") == 0 && wlen < slen)
        {
            /* Rewrite 2-arg 'alias name "cmd"' -> 'alias name="cmd"' */
            size_t k = wlen;
            while (k < slen && (s[k] == ' ' || s[k] == '\t'))
            {
                k++;
            }
            size_t nstart = k;
            while (k < slen && (isalnum((unsigned char) s[k]) || s[k] == '_'))
            {
                k++;
            }
            size_t nend = k;
            if (nend > nstart && k < slen && (s[k] == ' ' || s[k] == '\t'))
            {
                while (k < slen && (s[k] == ' ' || s[k] == '\t'))
                {
                    k++;
                }
                if (k < slen)
                {
                    sbuf_append_str(out, "alias ");
                    sbuf_append_mem(out, s + nstart, nend - nstart);
                    sbuf_append_ch(out, '=');
                    sbuf_append_mem(out, s + k, len - (i + k));
                    return;
                }
            }
        }
        if (is_registered_milk_command(firstword))
        {
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, firstword);
            append_args_with_quoting(s + wlen, len - (i + wlen), 1, 0, out);
            return;
        }
    }

    /* Default: standard bash command, still expand inline @s.im.prop in args */
    sbuf_append_mem(out, s, wlen);
    append_args_with_quoting(s + wlen, len - (i + wlen), 0, 0, out);
}

static void translate_line(const char *line, size_t len, int allow_legacy, sbuf_t *out)
{
    sbuf_t inl;
    sbuf_init(&inl);
    expand_inline_constructs(line, len, &inl);
    if (inl.data == NULL)
    {
        return;
    }

    const char *s      = inl.data;
    size_t      n      = inl.len;
    size_t      seg_st = 0;
    int         in_sq  = 0;
    int         in_dq  = 0;
    int         pdepth = 0;

    for (size_t i = 0; i < n; i++)
    {
        char c = s[i];
        if (c == '\\' && !in_sq && i + 1 < n)
        {
            i++;
            continue;
        }
        if (c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
            continue;
        }
        if (c == '"' && !in_sq)
        {
            in_dq = !in_dq;
            continue;
        }
        if (in_sq || in_dq)
        {
            continue;
        }
        if (c == '#' && (i == 0 || isspace((unsigned char) s[i - 1])))
        {
            break;
        }
        if (c == '(')
        {
            if (i > 0 && s[i - 1] == '$')
            {
                pdepth++;
            }
            continue;
        }
        if (c == ')' && pdepth > 0)
        {
            pdepth--;
            continue;
        }
        if (pdepth > 0)
        {
            continue;
        }

        /* Statement separators: ';', '&&', '||', '|', '{' */
        int sep_len = 0;
        if (c == ';')
        {
            /* Ignore ';;' in case statements */
            if (i + 1 < n && s[i + 1] == ';')
            {
                sep_len = 2;
            }
            else
            {
                sep_len = 1;
            }
        }
        else if (c == '&' && i + 1 < n && s[i + 1] == '&')
        {
            sep_len = 2;
        }
        else if (c == '|' && i + 1 < n && s[i + 1] == '|')
        {
            sep_len = 2;
        }
        else if (c == '|' && (i + 1 >= n || s[i + 1] != '>'))
        {
            sep_len = 1;
        }
        else if (c == '{' && (i == 0 || isspace((unsigned char) s[i - 1])) &&
                 (i + 1 >= n || isspace((unsigned char) s[i + 1])))
        {
            size_t b = seg_st;
            while (b < i && isspace((unsigned char) s[b]))
            {
                b++;
            }
            if (b == i || (i - b > 9 && strncmp(s + b, "function ", 9) == 0) ||
                (i >= 2 &&
                 (s[i - 1] == ')' || (isspace((unsigned char) s[i - 1]) && s[i - 2] == ')'))))
            {
                sep_len = 1;
            }
        }

        if (sep_len > 0)
        {
            translate_single_stmt(s + seg_st, i - seg_st, allow_legacy, out);
            sbuf_append_mem(out, s + i, (size_t) sep_len);
            i += (size_t) (sep_len - 1);
            seg_st = i + 1;
        }
    }

    if (seg_st <= n)
    {
        translate_single_stmt(s + seg_st, n - seg_st, allow_legacy, out);
    }

    free(inl.data);
}

static int detect_heredoc_delim(const char *line, size_t len, char *delim, size_t dmax)
{
    int in_sq = 0;
    int in_dq = 0;
    for (size_t i = 0; i + 1 < len; i++)
    {
        char c = line[i];
        if (c == '\\' && !in_sq && i + 1 < len)
        {
            i++;
            continue;
        }
        if (c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
        }
        else if (c == '"' && !in_sq)
        {
            in_dq = !in_dq;
        }
        else if (!in_sq && !in_dq)
        {
            if (c == '#' && (i == 0 || isspace((unsigned char) line[i - 1])))
            {
                break;
            }
            if (c == '<' && line[i + 1] == '<' && (i + 2 >= len || line[i + 2] != '<'))
            {
                size_t k = i + 2;
                if (k < len && line[k] == '-')
                {
                    k++;
                }
                while (k < len && (line[k] == ' ' || line[k] == '\t'))
                {
                    k++;
                }
                if (k < len && (line[k] == '\'' || line[k] == '"' || line[k] == '\\'))
                {
                    k++;
                }
                size_t di = 0;
                while (k < len && !isspace((unsigned char) line[k]) && line[k] != '\'' &&
                       line[k] != '"' && line[k] != ';' && di + 1 < dmax)
                {
                    delim[di++] = line[k++];
                }
                delim[di] = '\0';
                return (di > 0) ? 1 : 0;
            }
        }
    }
    return 0;
}

char *milk_translate_tier2_to_tier1(const char *src, int allow_legacy)
{
    if (src == NULL)
    {
        return NULL;
    }

    sbuf_t out;
    sbuf_init(&out);

    char        heredoc_delim[128] = { '\0' };
    int         in_heredoc         = 0;
    const char *p                  = src;

    while (*p != '\0')
    {
        const char *nl   = strchr(p, '\n');
        size_t      llen = (nl != NULL) ? (size_t) (nl - p) : strlen(p);

        if (in_heredoc)
        {
            sbuf_append_mem(&out, p, llen);
            const char *t  = p;
            size_t      tl = llen;
            while (tl > 0 && *t == '\t')
            {
                t++;
                tl--;
            }
            if (tl == strlen(heredoc_delim) && strncmp(t, heredoc_delim, tl) == 0)
            {
                in_heredoc = 0;
            }
        }
        else
        {
            translate_line(p, llen, allow_legacy, &out);
            if (detect_heredoc_delim(p, llen, heredoc_delim, sizeof(heredoc_delim)))
            {
                in_heredoc = 1;
            }
        }

        if (nl != NULL)
        {
            sbuf_append_ch(&out, '\n');
            p = nl + 1;
        }
        else
        {
            break;
        }
    }

    return out.data;
}

/* ============================================================
 *  Supervised Bash Co-Process & In-Process Socket Server
 * ============================================================ */

static int   g_listen_fd     = -1;
static int   g_cmd_pipe_w    = -1;
static int   g_status_pipe_r = -1;
static pid_t g_bash_pid      = -1;
static char  g_sockpath[128] = { '\0' };

static int has_unclosed_syntax(const char *s)
{
    int sq  = 0;
    int dq  = 0;
    int sub = 0;
    for (size_t i = 0; s[i] != '\0'; i++)
    {
        char c = s[i];
        if (c == '\\' && !sq && s[i + 1] != '\0')
        {
            i++;
            continue;
        }
        if (!sq && !dq && c == '#')
        {
            while (s[i] != '\0' && s[i] != '\n')
            {
                i++;
            }
            if (s[i] == '\0')
            {
                break;
            }
            continue;
        }
        if (c == '\'' && !dq)
        {
            sq = !sq;
        }
        else if (c == '"' && !sq)
        {
            dq = !dq;
        }
        else if (!sq && c == '$' && s[i + 1] == '(')
        {
            sub++;
            i++;
        }
        else if (!sq && c == ')' && sub > 0)
        {
            sub--;
        }
    }
    return (sq || dq || sub > 0) ? 1 : 0;
}

int milk_bash_engine_init(void)
{
    if (g_listen_fd >= 0 && g_bash_pid > 0)
    {
        return 0;
    }

    snprintf(g_sockpath, sizeof(g_sockpath), "@milk-cli-%d-%d", (int) getuid(), (int) getpid());
    g_listen_fd = milk_server_create_listener(g_sockpath);
    if (g_listen_fd < 0)
    {
        return -1;
    }
    setenv("MILKCLI_SOCK", g_sockpath, 1);

    int cmd_pipe[2]    = { -1, -1 };
    int status_pipe[2] = { -1, -1 };
    if (pipe2(cmd_pipe, O_CLOEXEC) != 0 || pipe2(status_pipe, O_CLOEXEC) != 0)
    {
        if (cmd_pipe[0] >= 0)
        {
            close(cmd_pipe[0]);
            close(cmd_pipe[1]);
        }
        close(g_listen_fd);
        g_listen_fd = -1;
        return -1;
    }

    cli_export_vars_to_env();
    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    if (pid < 0)
    {
        close(cmd_pipe[0]);
        close(cmd_pipe[1]);
        close(status_pipe[0]);
        close(status_pipe[1]);
        close(g_listen_fd);
        g_listen_fd = -1;
        return -1;
    }

    if (pid == 0)
    {
        /* Child: supervised bash co-process */
        setpgid(0, 0);
        close(g_listen_fd);
        close(cmd_pipe[1]);
        close(status_pipe[0]);

        dup2(cmd_pipe[0], 198);
        dup2(status_pipe[1], 199);
        if (cmd_pipe[0] != 198)
        {
            close(cmd_pipe[0]);
        }
        if (status_pipe[1] != 199)
        {
            close(status_pipe[1]);
        }
        fcntl(198, F_SETFD, 0);
        fcntl(199, F_SETFD, 0);

        const char *loop_script =
            "shopt -s expand_aliases\n"
            "if [[ -n \"${MILK_INSTALLDIR:-}\" && -f \"${MILK_INSTALLDIR}/bin/milk.bash\" ]]; "
            "then\n"
            "    builtin source \"${MILK_INSTALLDIR}/bin/milk.bash\"\n"
            "fi\n"
            "unalias() { builtin unalias \"$@\" 2>/dev/null || true; }\n"
            "source() {\n"
            "    if [[ $# -eq 0 ]]; then echo \"Usage: source <filename>\" >&2; return 1; fi\n"
            "    if [[ ! -f \"$1\" ]]; then echo \"Error: source: file not found: $1\" >&2; return "
            "1; fi\n"
            "    local __milk_f=\"$1\"; shift\n"
            "    builtin source <(\"${MILK_INSTALLDIR}/bin/milk-call\" translate --legacy "
            "\"${__milk_f}\") \"$@\"\n"
            "}\n"
            "declare -gA __MILK_INC_ONCE=()\n"
            "include_once() {\n"
            "    if [[ $# -eq 0 ]]; then echo \"Usage: include_once <filename>\" >&2; return 1; "
            "fi\n"
            "    if [[ ! -f \"$1\" ]]; then echo \"Error: include_once: file not found: $1\" >&2; "
            "return 1; fi\n"
            "    local __rp; __rp=\"$(realpath \"$1\" 2>/dev/null || echo \"$1\")\"\n"
            "    [[ -n \"${__MILK_INC_ONCE[$__rp]:-}\" ]] && return 0\n"
            "    __MILK_INC_ONCE[$__rp]=1\n"
            "    source \"$@\"\n"
            "}\n"
            "__MILK_EXIT_REQ=0\n"
            "__MILK_EXIT_CODE=0\n"
            "exit() { __MILK_EXIT_REQ=1; __MILK_EXIT_CODE=\"${1:-0}\"; return "
            "\"${__MILK_EXIT_CODE}\"; }\n"
            "quit() { exit \"$@\"; }\n"
            "exitCLI() { exit \"$@\"; }\n"
            "while IFS= read -r -d '' __milk_cmd <&198; do\n"
            "    eval \"${__milk_cmd}\" 198<&- 199>&-\n"
            "    __milk_rc=$?\n"
            "    printf '%d\\t%d\\t%s\\0' \"${__milk_rc}\" \"${__MILK_EXIT_REQ}\" \"${PWD}\" "
            ">&199\n"
            "    if [[ ${__MILK_EXIT_REQ} -eq 1 ]]; then\n"
            "        builtin exit \"${__MILK_EXIT_CODE}\"\n"
            "    fi\n"
            "done\n";

        execl("/bin/bash", "bash", "--norc", "--noprofile", "-c", loop_script, (char *) NULL);
        _exit(127);
    }

    close(cmd_pipe[0]);
    close(status_pipe[1]);
    g_cmd_pipe_w    = cmd_pipe[1];
    g_status_pipe_r = status_pipe[0];
    g_bash_pid      = pid;

    /* Run an empty warmup command so milk.bash initialization completes now */
    int dummy_exit = 0;
    milk_bash_engine_exec(":", &dummy_exit);
    return 0;
}

int milk_bash_engine_exec(const char *cmdline, int *out_exit_req)
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
    if (g_listen_fd < 0 || g_bash_pid <= 0)
    {
        if (milk_bash_engine_init() != 0)
        {
            return 1;
        }
    }

    /* Check if bash co-process died unexpectedly; respawn if needed */
    int wstatus = 0;
    if (waitpid(g_bash_pid, &wstatus, WNOHANG) == g_bash_pid)
    {
        close(g_cmd_pipe_w);
        close(g_status_pipe_r);
        close(g_listen_fd);
        g_cmd_pipe_w    = -1;
        g_status_pipe_r = -1;
        g_listen_fd     = -1;
        g_bash_pid      = -1;
        if (milk_bash_engine_init() != 0)
        {
            return 1;
        }
    }

    if (has_unclosed_syntax(cmdline))
    {
        fprintf(stderr, "Syntax error: unexpected end of input\n");
        fflush(stderr);
    }

    char *translated = milk_translate_tier2_to_tier1(cmdline, 1);
    if (translated == NULL)
    {
        return 1;
    }

    size_t tlen = strlen(translated) + 1;
    size_t sent = 0;
    while (sent < tlen)
    {
        ssize_t n = write(g_cmd_pipe_w, translated + sent, tlen - sent);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            free(translated);
            return 1;
        }
        sent += (size_t) n;
    }
    free(translated);

    /* Poll listen_fd (servicing milk commands in-process) and status_pipe_r */
    char   sbuf[4096];
    size_t spos          = 0;
    int    shutdown_flag = 0;
    dcsigINT             = 0;

    while (spos < sizeof(sbuf) - 1)
    {
        if (dcsigINT)
        {
            dcsigINT = 0;
            if (g_bash_pid > 0)
            {
                kill(-g_bash_pid, SIGINT);
            }
        }

        struct pollfd pfds[2];
        pfds[0].fd      = g_listen_fd;
        pfds[0].events  = POLLIN;
        pfds[0].revents = 0;
        pfds[1].fd      = g_status_pipe_r;
        pfds[1].events  = POLLIN | POLLHUP | POLLERR;
        pfds[1].revents = 0;

        int pr = poll(pfds, 2, 50);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        if (pfds[0].revents & POLLIN)
        {
            milk_server_accept_and_handle(g_listen_fd, &shutdown_flag);
        }

        if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
        {
            ssize_t r = read(g_status_pipe_r, sbuf + spos, sizeof(sbuf) - 1 - spos);
            if (r > 0)
            {
                int found_nul = 0;
                for (ssize_t k = 0; k < r; k++)
                {
                    if (sbuf[spos + (size_t) k] == '\0')
                    {
                        found_nul = 1;
                        break;
                    }
                }
                spos += (size_t) r;
                if (found_nul)
                {
                    break;
                }
            }
            else if (r == 0 || (r < 0 && errno != EINTR && errno != EAGAIN))
            {
                /* Co-process exited unexpectedly (e.g. fatal bash parameter expansion ${:?}) */
                int st = 0;
                if (waitpid(g_bash_pid, &st, 0) == g_bash_pid)
                {
                    g_bash_pid = -1;
                }
                close(g_cmd_pipe_w);
                close(g_status_pipe_r);
                close(g_listen_fd);
                g_cmd_pipe_w    = -1;
                g_status_pipe_r = -1;
                g_listen_fd     = -1;
                int rc_err      = WIFEXITED(st) ? WEXITSTATUS(st) : 1;
                cli_last_retval = (rc_err != 0) ? rc_err : 1;
                return cli_last_retval;
            }
        }
    }

    sbuf[spos]  = '\0';
    int   rc    = 0;
    int   exreq = 0;
    char *tab1  = strchr(sbuf, '\t');
    if (tab1 != NULL)
    {
        *tab1      = '\0';
        rc         = (int) strtol(sbuf, NULL, 10);
        char *tab2 = strchr(tab1 + 1, '\t');
        if (tab2 != NULL)
        {
            *tab2            = '\0';
            exreq            = (int) strtol(tab1 + 1, NULL, 10);
            const char *npwd = tab2 + 1;
            if (npwd[0] != '\0')
            {
                if (chdir(npwd) == 0)
                {
                    setenv("PWD", npwd, 1);
                }
            }
        }
    }

    if (out_exit_req != NULL)
    {
        *out_exit_req = exreq;
    }
    cli_last_retval = rc;
    return rc;
}

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

    int rc = milk_bash_engine_exec(buf, out_exit_req);
    free(buf);
    return rc;
}

void milk_bash_engine_shutdown(void)
{
    if (g_cmd_pipe_w >= 0)
    {
        close(g_cmd_pipe_w);
        g_cmd_pipe_w = -1;
    }

    /* Service any final socket calls triggered by EXIT / @defer traps in bash */
    if (g_bash_pid > 0 && g_listen_fd >= 0 && g_status_pipe_r >= 0)
    {
        int shutdown_flag = 0;
        for (int iter = 0; iter < 200; iter++)
        {
            struct pollfd pfds[2];
            pfds[0].fd      = g_listen_fd;
            pfds[0].events  = POLLIN;
            pfds[0].revents = 0;
            pfds[1].fd      = g_status_pipe_r;
            pfds[1].events  = POLLIN | POLLHUP | POLLERR;
            pfds[1].revents = 0;

            int pr = poll(pfds, 2, 25);
            if (pr <= 0)
            {
                if (waitpid(g_bash_pid, NULL, WNOHANG) == g_bash_pid)
                {
                    g_bash_pid = -1;
                    break;
                }
                continue;
            }
            if (pfds[0].revents & POLLIN)
            {
                milk_server_accept_and_handle(g_listen_fd, &shutdown_flag);
            }
            if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
            {
                char drain[256];
                if (read(g_status_pipe_r, drain, sizeof(drain)) <= 0)
                {
                    /* Drain any final pending client connection */
                    struct pollfd dpfd = { g_listen_fd, POLLIN, 0 };
                    while (poll(&dpfd, 1, 10) > 0 && (dpfd.revents & POLLIN))
                    {
                        milk_server_accept_and_handle(g_listen_fd, &shutdown_flag);
                    }
                    break;
                }
            }
        }
    }

    if (g_status_pipe_r >= 0)
    {
        close(g_status_pipe_r);
        g_status_pipe_r = -1;
    }
    if (g_bash_pid > 0)
    {
        waitpid(g_bash_pid, NULL, 0);
        g_bash_pid = -1;
    }
    if (g_listen_fd >= 0)
    {
        close(g_listen_fd);
        g_listen_fd = -1;
    }
}
