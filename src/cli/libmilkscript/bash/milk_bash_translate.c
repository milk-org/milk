// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_translate.c
 * @brief Line-preserving translator from milk-cli shorthand syntax to milk.bash.
 *
 * Role & Goal:
 *   Translates interactive milk-cli input and `.milk` scripts into valid
 *   `milk.bash` code while preserving 1:1 line numbering (`$LINENO`):
 *   - Inline `@{<expr>}` -> `$(@calc "<expr>")`
 *   - Inline `@s.<stream>.<prop>` / `@f.<fps>.<param>` -> `$(@get ...)`
 *   - `@ <expr>` -> `@calc "<expr>"`
 *   - `@<entity>.<name>.<field>=<val>` -> `@set ...`
 *   - Interactive REPL shorthands (`?`, `m?`, `cmd?`, `f?`, and un-sigiled
 *     registered milk commands when `allow_legacy` is enabled).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "milk_bash_engine_internal.h"

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
    while (i > plen && s[i - 1] == '.')
    {
        i--;
        dots--;
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
        "if",         "then",    "elif",     "else",         "fi",          "for",
        "in",         "while",   "until",    "do",           "done",        "case",
        "esac",       "select",  "function", "time",         "coproc",      "{",
        "}",          "!",       "[[",       "]]",           "[",           "]",
        "echo",       "printf",  "cd",       "pwd",          "read",        "export",
        "unset",      "shift",   "alias",    "unalias",      "source",      ".",
        "exit",       "quit",    "exitCLI",  "return",       "break",       "continue",
        "local",      "declare", "typeset",  "readonly",     "let",         "eval",
        "exec",       "set",     "shopt",    "trap",         "type",        "command",
        "builtin",    "enable",  "wait",     "kill",         "jobs",        "bg",
        "fg",         "true",    "false",    "test",         "sleep",       "rm",
        "ls",         "cat",     "grep",     "awk",          "sed",         "find",
        "xargs",      "sort",    "head",     "tail",         "wc",          "cut",
        "tr",         "mkdir",   "rmdir",    "cp",           "mv",          "ln",
        "chmod",      "touch",   "date",     "uname",        "hostname",    "whoami",
        "id",         "env",     "bookmark", "include_once", "milkcall",    "milkcalc",
        "milkget",    "milkset", "milktest", "milkgrab",     "milkwaitfor", "milkexists",
        "milkserver", "milklog", NULL
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

    static const char *const extra_cmds[] = {
        "help",     "imgstat",   "info.imgstat", "imstats",
        "mk2Dim",   "mk3Dim",    "mk2Drnd",      "imgen.mk2Drnd",
        "mkrndim",  "readshmim", "listim",       "deleteim",
        "loadfits", "savefits",  "fpslist",      "streamlist",
        "proclist", "procstat",  "procctl",      "procwait",
        "dpdigits", "lm?",       "m?",           "cmd?",
        "f?",       "?",         "vars",         "savehistory",
        NULL
    };
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
 *  Inline Expansion Helpers (@{...} and @s.<name>.<prop>)
 * ============================================================ */

static int is_calc_or_test_stmt_prefix(const char *s, size_t len, const char *sigil, size_t siglen)
{
    size_t i = 0;
    while (i < len && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    for (;;)
    {
        static const char *const prefixes[] = { "then",  "do", "else", "elif", "while",
                                                "until", "if", "!",    NULL };
        int                      matched    = 0;
        for (int p = 0; prefixes[p] != NULL; p++)
        {
            size_t plen = strlen(prefixes[p]);
            if (i + plen < len && strncmp(s + i, prefixes[p], plen) == 0 &&
                (s[i + plen] == ' ' || s[i + plen] == '\t'))
            {
                i += plen;
                while (i < len && (s[i] == ' ' || s[i] == '\t'))
                {
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
    if (i >= len)
    {
        return 0;
    }
    size_t adv = 0;
    if (i + siglen <= len && strncmp(s + i, sigil, siglen) == 0)
    {
        adv = siglen;
    }
    else if (s[i] == '@')
    {
        adv = 1;
    }
    if (adv == 0)
    {
        return 0;
    }
    const char *rest = s + i + adv;
    size_t      rlen = len - (i + adv);
    if (rlen > 0 && (rest[0] == ' ' || rest[0] == '\t'))
    {
        return 1;
    }
    if ((rlen > 5 && strncmp(rest, "calc", 4) == 0 && (rest[4] == ' ' || rest[4] == '\t')) ||
        (rlen > 5 && strncmp(rest, "test", 4) == 0 && (rest[4] == ' ' || rest[4] == '\t')) ||
        (rlen > 8 && strncmp(rest, "waitfor", 7) == 0 && (rest[7] == ' ' || rest[7] == '\t')))
    {
        return 1;
    }
    return 0;
}

static void expand_inline_constructs(const char *in, size_t len, sbuf_t *out)
{
    const char *sigil      = get_sigil();
    size_t      siglen     = strlen(sigil);
    int         in_sq      = 0;
    int         in_dq      = 0;
    size_t      stmt_start = 0;

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

        if (!in_dq && (c == ';' || c == '|' || c == '&'))
        {
            stmt_start = i + 1;
        }

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

        /* Inside double quotes or RHS of VAR=@s.im.prop: @s.im.prop -> $(@get s.im.prop) */
        if ((in_dq || (i > 0 && in[i - 1] == '=')) &&
            (i == 0 || (!isalnum((unsigned char) in[i - 1]) && in[i - 1] != '_')) &&
            (strncmp(in + i, sigil, siglen) == 0 || in[i] == '@'))
        {
            int skip_for_calc = in_dq && is_calc_or_test_stmt_prefix(in + stmt_start,
                                                                     i - stmt_start, sigil, siglen);
            if (!skip_for_calc)
            {
                size_t adv    = (strncmp(in + i, sigil, siglen) == 0) ? siglen : 1;
                int    has_eq = 0;
                size_t elen   = scan_entity_path(in + i + adv, &has_eq);
                if (elen > 0 && !has_eq)
                {
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
        }

        sbuf_append_ch(out, c);
        i++;
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

/**
 * @brief Translate interactive REPL shorthands (`?`, `m?`, `cmd?`, `f?`, and un-sigiled milk commands).
 *
 * @return 1 if handled, 0 otherwise
 */
static int try_translate_interactive_word(const char *firstword,
                                          const char *s,
                                          size_t      slen,
                                          size_t      wlen,
                                          size_t      tail_len,
                                          const char *sigil,
                                          sbuf_t     *out)
{
    size_t full_rem = slen + tail_len - wlen;

    if (strcmp(firstword, "?") == 0)
    {
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, "h");
        append_args_with_quoting(s + wlen, full_rem, 1, 0, out);
        return 1;
    }
    if (strcmp(firstword, "m?") == 0 || strcmp(firstword, "lm?") == 0)
    {
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, "hm");
        append_args_with_quoting(s + wlen, full_rem, 1, 0, out);
        return 1;
    }
    if (strcmp(firstword, "cmd?") == 0)
    {
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, "hc");
        append_args_with_quoting(s + wlen, full_rem, 1, 0, out);
        return 1;
    }
    if (strcmp(firstword, "f?") == 0)
    {
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, "hf");
        append_args_with_quoting(s + wlen, full_rem, 1, 0, out);
        return 1;
    }
    if (strcmp(firstword, "dpdigits") == 0 || strcmp(firstword, "procstat") == 0 ||
        strcmp(firstword, "procctl") == 0 || strcmp(firstword, "procwait") == 0 ||
        strcmp(firstword, "vars") == 0 || strcmp(firstword, "history") == 0 ||
        strcmp(firstword, "searchhist") == 0 || strcmp(firstword, "fhist") == 0 ||
        strcmp(firstword, "ghistory") == 0 || strcmp(firstword, "lhistory") == 0 ||
        strcmp(firstword, "savehistory") == 0 || strcmp(firstword, "setprompt") == 0 ||
        strcmp(firstword, "sessionlog") == 0 || strcmp(firstword, "synhl") == 0 ||
        strcmp(firstword, "syndiag") == 0 || strcmp(firstword, "cliindent") == 0 ||
        strcmp(firstword, "showmatch") == 0 || strcmp(firstword, "helprl") == 0 ||
        strcmp(firstword, "fhelp") == 0 || strcmp(firstword, "fparam") == 0 ||
        strcmp(firstword, "help") == 0)
    {
        sbuf_append_str(out, "milkcall ");
        sbuf_append_str(out, firstword);
        append_args_with_quoting(s + wlen, full_rem, 1, 0, out);
        return 1;
    }
    if (is_registered_milk_command(firstword))
    {
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, firstword);
        append_args_with_quoting(s + wlen, full_rem, 1, 0, out);
        return 1;
    }
    return 0;
}

static void translate_single_stmt(const char *stmt, size_t len, int allow_legacy, sbuf_t *out)
{
    const char *sigil  = get_sigil();
    size_t      siglen = strlen(sigil);

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

    /* Peel off leading shell flow-control prefixes */
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

    size_t end = len;
    while (end > i && (stmt[end - 1] == ' ' || stmt[end - 1] == '\t'))
    {
        end--;
    }
    size_t      slen = end - i;
    const char *s    = stmt + i;

    /* 1. Shorthand '@ <expr>' calculator statement */
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

    /* 2. Shorthand '@<entity>.<name>.<field>=<val>' or bare '@<entity>.<name>.<field>' */
    if ((slen > siglen && strncmp(s, sigil, siglen) == 0) || (slen > 1 && s[0] == '@'))
    {
        size_t adv    = (strncmp(s, sigil, siglen) == 0) ? siglen : 1;
        int    has_eq = 0;
        size_t elen   = scan_entity_path(s + adv, &has_eq);
        if (elen > 0)
        {
            size_t eq_pos = adv + elen;
            while (eq_pos < slen && (s[eq_pos] == ' ' || s[eq_pos] == '\t'))
            {
                eq_pos++;
            }
            if (has_eq ||
                (eq_pos < slen && s[eq_pos] == '=' && (eq_pos + 1 >= slen || s[eq_pos + 1] != '=')))
            {
                size_t vstart = eq_pos + 1;
                while (vstart < slen && (s[vstart] == ' ' || s[vstart] == '\t'))
                {
                    vstart++;
                }
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

    /* 3. Already sigiled '@<cmd> ...' */
    if (strncmp(firstword, sigil, siglen) == 0 || firstword[0] == '@')
    {
        int preserve_ent = is_entity_taking_cmd(firstword, sigil);
        sbuf_append_mem(out, s, wlen);
        append_args_with_quoting(s + wlen, len - (i + wlen), 1, preserve_ent, out);
        return;
    }

    /* 4. Interactive milk-cli shorthands when allow_legacy != 0 */
    if (allow_legacy &&
        try_translate_interactive_word(firstword, s, slen, wlen, len - end, sigil, out))
    {
        return;
    }

    /* Default: standard bash command */
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

        int sep_len = 0;
        if (c == ';')
        {
            sep_len = (i + 1 < n && s[i + 1] == ';') ? 2 : 1;
        }
        else if (c == '&' && i + 1 < n && s[i + 1] == '&')
        {
            sep_len = 2;
        }
        else if (c == '|' && i + 1 < n && s[i + 1] == '|')
        {
            sep_len = 2;
        }
        else if (c == '|')
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

/**
 * @brief Check whether @s contains unclosed quotes or `$()` command substitutions.
 *
 * @param s Input command or script string
 * @return 1 if unclosed syntax is detected, 0 otherwise
 */
int milk_bash_has_unclosed_syntax(const char *s)
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

/**
 * @brief Translate milk-cli shorthands into standard milk.bash syntax.
 *
 * @param src          Input script or command line
 * @param allow_legacy Non-zero to also translate un-sigiled interactive milk-cli commands
 * @return Newly heap-allocated translated string (caller must `free()`), or NULL
 */
char *milk_translate_shorthand_to_bash(const char *src, int allow_legacy)
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
