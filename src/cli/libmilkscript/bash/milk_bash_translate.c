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
#include "milk_shell_lex.h"

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

/**
 * milk_bash_funsub_disabled() - 1 if `${| ...; }` must not be emitted (MILK_BASH_NO_FUNSUB, or bash < 5.3).
 *
 * The probe result is exported so child translators (milk-call translate) inherit it.
 */
int milk_bash_funsub_disabled(void)
{
    const char *nf = getenv("MILK_BASH_NO_FUNSUB");
    if (nf != NULL && nf[0] != '\0')
    {
        return strcmp(nf, "0") != 0;
    }
    int   major = 0;
    int   minor = 0;
    FILE *pp =
        popen("/bin/bash -c 'echo ${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}' 2>/dev/null", "r");
    if (pp != NULL)
    {
        if (fscanf(pp, "%d %d", &major, &minor) != 2)
        {
            major = 0;
        }
        pclose(pp);
    }
    int disabled = (major < 5 || (major == 5 && minor < 3));
    if (disabled)
    {
        fprintf(stderr,
                "milk: /bin/bash %d.%d lacks ${| ...; } (needs >= 5.3); inline @ expansions will "
                "fork\n",
                major, minor);
    }
    setenv("MILK_BASH_NO_FUNSUB", disabled ? "1" : "0", 1);
    return disabled;
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
    static const char *const pfx[] = { "s.",   "l.",      "f.",   "p.",     "q.",    "fps.",
                                       "seq.", "stream.", "loc.", "local.", "proc.", NULL };
    for (int i = 0; pfx[i] != NULL; i++)
    {
        size_t plen = strlen(pfx[i]);
        if (strncmp(s, pfx[i], plen) == 0 && (isalnum((unsigned char) s[plen]) || s[plen] == '_' ||
                                              s[plen] == '*' || s[plen] == '?' || s[plen] == '$'))
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
    while (s[i] != '\0')
    {
        if (s[i] == '$' && s[i + 1] == '{') /* dynamic part: s.${name}.xsize */
        {
            size_t j = i + 2;
            while (s[j] != '\0' && s[j] != '}' && s[j] != '\n')
            {
                j++;
            }
            if (s[j] != '}')
            {
                break;
            }
            i = j + 1;
            continue;
        }
        if (s[i] == '$' && (isalpha((unsigned char) s[i + 1]) || s[i + 1] == '_')) /* s.$name.x */
        {
            i++;
            while (isalnum((unsigned char) s[i]) || s[i] == '_')
            {
                i++;
            }
            continue;
        }
        if (!(isalnum((unsigned char) s[i]) || s[i] == '_' || s[i] == '-' || s[i] == '.' ||
              s[i] == '*' || s[i] == '?'))
        {
            break;
        }
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
    if (s[i] == '=' && s[i + 1] != '=')
    {
        *has_eq = 1;
    }
    return i;
}

static int is_bash_reserved_or_builtin(const char *w)
{
    static const char *const kw[] = {
        "if",         "then",    "elif",       "else",         "fi",          "for",
        "in",         "while",   "until",      "do",           "done",        "case",
        "esac",       "select",  "function",   "time",         "coproc",      "{",
        "}",          "!",       "[[",         "]]",           "[",           "]",
        "echo",       "printf",  "cd",         "pwd",          "read",        "export",
        "unset",      "shift",   "alias",      "unalias",      "source",      ".",
        "exit",       "quit",    "exitCLI",    "return",       "break",       "continue",
        "local",      "declare", "typeset",    "readonly",     "let",         "eval",
        "exec",       "set",     "shopt",      "trap",         "type",        "command",
        "builtin",    "enable",  "wait",       "kill",         "jobs",        "bg",
        "fg",         "true",    "false",      "test",         "sleep",       "rm",
        "ls",         "cat",     "grep",       "awk",          "sed",         "find",
        "xargs",      "sort",    "head",       "tail",         "wc",          "cut",
        "tr",         "mkdir",   "rmdir",      "cp",           "mv",          "ln",
        "chmod",      "touch",   "date",       "uname",        "hostname",    "whoami",
        "id",         "env",     "bookmark",   "include_once", "milkcall",    "milkcalc",
        "milkget",    "milkset", "milktest",   "milkgrab",     "milkwaitfor", "milkwaitany",
        "milkexists", "milkls",  "milkserver", "milklog",      NULL
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
        "help",          "imgstat", "info.imgstat", "imstats",  "mk2Dim",   "mk3Dim",   "mk2Drnd",
        "imgen.mk2Drnd", "mkrndim", "readshmim",    "listim",   "deleteim", "loadfits", "savefits",
        "fpslist",       "fpsset",  "streamlist",   "proclist", "procstat", "procctl",  "procwait",
        "dpdigits",      "lm?",     "m?",           "cmd?",     "f?",       "?",        "vars",
        "savehistory",   NULL
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
 *  Translation state: the source and its context map (milk_shell_lex)
 * ============================================================ */

static const char *g_src = NULL; /* text being translated */
static msl_map_t   g_map;        /* context of every byte of g_src */
static int         g_allow_legacy = 0;
static int g_skip_dq_ent = 0; /* current statement is @calc/@test/...: calc reads "@s.x.y" */

static inline int cx(const char *p)
{
    return g_map.ctx[p - g_src];
}

static inline int dp(const char *p)
{
    return g_map.depth[p - g_src];
}

/** code_at() - 1 if @p is unquoted shell code at command-substitution depth @d. */
static inline int code_at(const char *p, int d)
{
    return cx(p) == MSL_CODE && dp(p) == d;
}

/** sig_len() - length of the sigil if [@p, @p + @n) starts with it, else 0. */
static size_t sig_len(const char *p, size_t n)
{
    const char *sigil = get_sigil();
    size_t      sl    = strlen(sigil);
    return (n >= sl && strncmp(p, sigil, sl) == 0) ? sl : 0;
}

static void translate_code(const char *p, size_t n, int d, sbuf_t *out);

/**
 * emit_inline() - emit an @get/@calc substitution (funsub, or $() on old bash); @dq wraps it in "..."
 * so the result is neither word-split nor globbed.
 */
static void emit_inline(sbuf_t *out, int is_calc, const char *body, size_t n, int dq)
{
    int funsub = !milk_bash_funsub_disabled();
    if (dq)
    {
        sbuf_append_ch(out, '"');
    }
    sbuf_append_str(out, funsub ? "${| " : "$(");
    sbuf_append_str(out, get_sigil());
    if (is_calc)
    {
        sbuf_append_str(out, funsub ? "calc -v REPLY \"" : "calc \"");
        sbuf_append_dq_escaped(out, body, n);
        sbuf_append_ch(out, '"');
    }
    else
    {
        sbuf_append_str(out, funsub ? "get -v REPLY " : "get ");
        sbuf_append_mem(out, body, n);
    }
    sbuf_append_str(out, funsub ? "; }" : ")");
    if (dq)
    {
        sbuf_append_ch(out, '"');
    }
}

/** entity_ref_len() - length of a bare `s.a.b`-style reference (>= 2 dots, no `=`) at @p, or 0. */
static size_t entity_ref_len(const char *p)
{
    int    has_eq = 0;
    size_t elen   = scan_entity_path(p, &has_eq);
    if (elen == 0 || has_eq)
    {
        return 0;
    }
    int ndots = 0;
    for (size_t k = 0; k < elen; k++)
    {
        ndots += (p[k] == '.');
    }
    return (ndots >= 2) ? elen : 0;
}

/**
 * try_inline() - expand the construct starting at @p[i], if any: `\@` escape, `@{expr}`, `@s.x.y`,
 * `${@s.x.y}`. Returns the number of source bytes consumed (0 if nothing to expand).
 */
static size_t try_inline(const char *p, size_t n, size_t i, int d, int ent, sbuf_t *out)
{
    const char *q  = p + i;
    int         c  = cx(q);
    size_t      sl = 0;

    if ((c == MSL_CODE || c == MSL_DQ) && q[0] == '\\' && i + 1 < n)
    {
        sl = sig_len(q + 1, n - i - 1);
        if (c == MSL_DQ && sl > 0) /* "\@{..}" is a literal @{..} */
        {
            sbuf_append_mem(out, q + 1, sl);
            return 1 + sl;
        }
        sbuf_append_mem(out, q, 2); /* escaped character: never the start of a construct */
        return 2;
    }

    if (c == MSL_PARAM && q[0] == '$' && i + 2 < n && q[1] == '{' &&
        (sl = sig_len(q + 2, n - i - 2)) > 0)
    {
        size_t elen = entity_ref_len(q + 2 + sl);
        if (elen > 0 && i + 2 + sl + elen < n && q[2 + sl + elen] == '}') /* ${@s.x.y} */
        {
            const char *k = q - 1;
            while (k >= g_src && cx(k) == MSL_PARAM)
            {
                k--;
            }
            emit_inline(out, 0, q + 2 + sl, elen, !(k >= g_src && cx(k) == MSL_DQ));
            return 2 + sl + elen + 1;
        }
        return 0;
    }

    if ((c != MSL_CODE && c != MSL_DQ && c != MSL_ARITH) || (sl = sig_len(q, n - i)) == 0)
    {
        return 0;
    }

    char prev = (q > g_src) ? q[-1] : ' ';
    if (i + sl < n && q[sl] == '{') /* @{expr}: braces counted in the opener's context only */
    {
        if (c == MSL_CODE && (isalnum((unsigned char) prev) || prev == '_'))
        {
            return 0; /* foo@{a,b} is bash brace expansion */
        }
        int depth = 1;
        for (size_t j = i + sl + 1; j < n; j++)
        {
            const char *r = p + j;
            if (cx(r) != c || dp(r) != d)
            {
                continue;
            }
            depth += (*r == '{') - (*r == '}');
            if (depth == 0)
            {
                emit_inline(out, 1, q + sl + 1, j - (i + sl + 1), c != MSL_DQ);
                return j + 1 - i;
            }
        }
        return 0;
    }

    if (!ent || c == MSL_ARITH || (c == MSL_DQ && g_skip_dq_ent) || isalnum((unsigned char) prev) ||
        prev == '_' || prev == '.' || prev == '$')
    {
        return 0;
    }
    size_t elen = entity_ref_len(q + sl);
    if (elen == 0 || i + sl + elen > n)
    {
        return 0;
    }
    emit_inline(out, 0, q + sl, elen, c == MSL_CODE);
    return sl + elen;
}

/**
 * emit_src() - copy source bytes [@p, @p + @n) at depth @d to @out, expanding inline constructs
 * (@ent: also bare entity references) and translating nested $(...), `...` and ${| ...; } bodies.
 */
static void emit_src(const char *p, size_t n, int d, int ent, sbuf_t *out)
{
    size_t i = 0;
    while (i < n)
    {
        const char *q = p + i;
        if (dp(q) > d)
        {
            if (cx(q) == MSL_OPEN && dp(q) == d + 1)
            {
                size_t j = i;
                while (j < n && cx(p + j) == MSL_OPEN && dp(p + j) == d + 1)
                {
                    j++;
                }
                sbuf_append_mem(out, q, j - i);
                size_t k = j;
                while (k < n && !(cx(p + k) == MSL_CLOSE && dp(p + k) == d + 1))
                {
                    k++;
                }
                translate_code(p + j, k - j, d + 1, out);
                i = k;
                continue;
            }
            sbuf_append_ch(out, *q);
            i++;
            continue;
        }
        size_t used = try_inline(p, n, i, d, ent, out);
        if (used > 0)
        {
            i += used;
            continue;
        }
        sbuf_append_ch(out, *q);
        i++;
    }
}

/**
 * inline_span() - length of a balanced `@{...}` starting at @p[i] in code at depth @d, or 0.
 * Such a span is one opaque unit: its spaces, `&&`, `*` belong to the calc expression.
 */
static size_t inline_span(const char *p, size_t n, size_t i, int d)
{
    const char *q    = p + i;
    size_t      sl   = code_at(q, d) ? sig_len(q, n - i) : 0;
    char        prev = (q > g_src) ? q[-1] : ' ';
    if (sl == 0 || i + sl >= n || q[sl] != '{' || isalnum((unsigned char) prev) || prev == '_')
    {
        return 0;
    }
    int depth = 1;
    for (size_t j = i + sl + 1; j < n; j++)
    {
        if (code_at(p + j, d))
        {
            depth += (p[j] == '{') - (p[j] == '}');
            if (depth == 0)
            {
                return j + 1 - i;
            }
        }
    }
    return 0;
}

/** word_len() - length of the shell word at @p (ends at unquoted whitespace at depth @d). */
static size_t word_len(const char *p, size_t n, int d)
{
    size_t i = 0;
    while (i < n && !(code_at(p + i, d) && isspace((unsigned char) p[i])))
    {
        size_t sp = inline_span(p, n, i, d);
        i += (sp > 0) ? sp : ((code_at(p + i, d) && p[i] == '\\' && i + 1 < n) ? 2 : 1);
    }
    return i;
}

/** is_one_quoted_word() - 1 if @p[0..n) is a single word made only of quoted parts. */
static int is_one_quoted_word(const char *p, size_t n, int d)
{
    if (n < 2 || (p[0] != '\'' && p[0] != '"'))
    {
        return 0;
    }
    for (size_t i = 0; i < n; i++)
    {
        if (code_at(p + i, d))
        {
            return 0;
        }
    }
    return 1;
}

/** has_unquoted_space() - 1 if @p[0..n) holds whitespace that is unquoted shell code at depth @d. */
static int has_unquoted_space(const char *p, size_t n, int d)
{
    for (size_t i = 0; i < n; i++)
    {
        size_t sp = inline_span(p, n, i, d);
        if (sp > 0)
        {
            i += sp - 1;
            continue;
        }
        if (code_at(p + i, d) && isspace((unsigned char) p[i]))
        {
            return 1;
        }
    }
    return 0;
}

/** has_unquoted_pipe_op() - 1 if @s[0..len) contains a `|>` pipeline operator in code at depth @d. */
static int has_unquoted_pipe_op(const char *s, size_t len, int d)
{
    for (size_t i = 0; i + 1 < len; i++)
    {
        size_t sp = inline_span(s, len, i, d);
        if (sp > 0)
        {
            i += sp - 1;
            continue;
        }
        if (code_at(s + i, d) && s[i] == '|' && s[i + 1] == '>')
        {
            return 1;
        }
    }
    return 0;
}

/**
 * append_calc_expr() - emit @s[0..n) as one double-quoted calc argument; a backslash-newline is
 * kept outside the quotes so line numbers do not shift.
 */
static void append_calc_expr(sbuf_t *out, const char *s, size_t n, int d)
{
    sbuf_append_ch(out, '"');
    for (size_t i = 0; i < n; i++)
    {
        if (s[i] == '\\' && i + 1 < n && s[i + 1] == '\n' && code_at(s + i, d))
        {
            sbuf_append_str(out, "\"\\\n\"");
            i++;
            continue;
        }
        if (s[i] == '"' || s[i] == '\\' || s[i] == '`')
        {
            sbuf_append_ch(out, '\\');
        }
        sbuf_append_ch(out, s[i]);
    }
    sbuf_append_ch(out, '"');
}

/* ============================================================
 *  Statement-Level Translator
 * ============================================================ */

static const char *const flow_prefixes[] = { "then",  "do", "else", "elif", "while",
                                             "until", "if", "!",    "time", NULL };

/** skip_flow_prefix() - length of one leading flow keyword (+ blanks) at @s, or 0. */
static size_t skip_flow_prefix(const char *s, size_t n)
{
    for (int k = 0; flow_prefixes[k] != NULL; k++)
    {
        size_t pl = strlen(flow_prefixes[k]);
        if (pl < n && strncmp(s, flow_prefixes[k], pl) == 0 && (s[pl] == ' ' || s[pl] == '\t'))
        {
            while (pl < n && (s[pl] == ' ' || s[pl] == '\t'))
            {
                pl++;
            }
            return pl;
        }
    }
    return 0;
}

/** is_calc_shorthand() - 1 if the statement at @s is an `@ <expr>` calculator statement. */
static int is_calc_shorthand(const char *s, size_t n)
{
    size_t i = 0;
    while (i < n && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    size_t k;
    while (i < n && (k = skip_flow_prefix(s + i, n - i)) > 0)
    {
        i += k;
    }
    size_t sl = sig_len(s + i, n - i);
    return sl > 0 && i + sl < n && (s[i + sl] == ' ' || s[i + sl] == '\t');
}

static int is_calc_or_test_stmt_prefix(const char *s, size_t len)
{
    size_t i = 0;
    while (i < len && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    size_t k;
    while (i < len && (k = skip_flow_prefix(s + i, len - i)) > 0)
    {
        i += k;
    }
    size_t adv = (i < len) ? sig_len(s + i, len - i) : 0;
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
    static const char *const calc_like[] = { "calc", "test", "assert", "waitfor", NULL };
    for (int c = 0; calc_like[c] != NULL; c++)
    {
        size_t cl = strlen(calc_like[c]);
        if (rlen > cl && strncmp(rest, calc_like[c], cl) == 0 &&
            (rest[cl] == ' ' || rest[cl] == '\t'))
        {
            return 1;
        }
    }
    return 0;
}

static int is_entity_taking_cmd(const char *cmd)
{
    cmd += sig_len(cmd, strlen(cmd));
    return (strcmp(cmd, "ls") == 0 || strcmp(cmd, "get") == 0 || strcmp(cmd, "set") == 0 ||
            strcmp(cmd, "exists") == 0 || strcmp(cmd, "running") == 0 ||
            strcmp(cmd, "waitfor") == 0 || strcmp(cmd, "waitany") == 0 || strcmp(cmd, "on") == 0 ||
            strcmp(cmd, "snapshot") == 0 || strcmp(cmd, "diff") == 0 ||
            strcmp(cmd, "restore") == 0 || strcmp(cmd, "assert") == 0 ||
            strcmp(cmd, "prompt") == 0 || strcmp(cmd, "streamcat") == 0 ||
            strcmp(cmd, "streamwrite") == 0);
}

/**
 * append_args() - emit argument words of a command: milk commands get unquoted globs/slices
 * quoted; entity-taking commands (@preserve_entity_arg) keep bare entity specs literal.
 */
static void append_args(const char *args,
                        size_t      len,
                        int         is_milk_cmd,
                        int         preserve_entity_arg,
                        int         d,
                        sbuf_t     *out)
{
    size_t i = 0;
    while (i < len)
    {
        if (code_at(args + i, d) && isspace((unsigned char) args[i]))
        {
            sbuf_append_ch(out, args[i]);
            i++;
            continue;
        }
        size_t tstart     = i;
        int    has_quotes = 0;
        int    br_depth   = 0;
        int    has_slice  = 0;
        int    has_glob   = 0;
        while (i < len)
        {
            const char *q = args + i;
            if (!code_at(q, d))
            {
                has_quotes = 1;
                i++;
                continue;
            }
            if (br_depth == 0 && isspace((unsigned char) *q))
            {
                break;
            }
            size_t sp = inline_span(args, len, i, d);
            if (sp > 0)
            {
                i += sp;
                continue;
            }
            if (*q == '\\' && i + 1 < len)
            {
                i += 2;
                continue;
            }
            if (*q == '[')
            {
                br_depth++;
                has_slice = 1;
            }
            else if (*q == ']' && br_depth > 0)
            {
                br_depth--;
            }
            else if (*q == '*' || *q == '?')
            {
                has_glob = 1;
            }
            i++;
        }
        size_t tlen = i - tstart;
        if (is_milk_cmd && !has_quotes && (has_slice || has_glob))
        {
            sbuf_append_ch(out, '"');
            sbuf_append_dq_escaped(out, args + tstart, tlen);
            sbuf_append_ch(out, '"');
        }
        else
        {
            emit_src(args + tstart, tlen, d, !preserve_entity_arg, out);
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
                                          size_t      rem_len,
                                          size_t      wlen,
                                          int         d,
                                          sbuf_t     *out)
{
    const char *sigil = get_sigil();
    static const struct
    {
        const char *word;
        const char *cmd;
    } help_alias[] = { { "?", "h" },     { "m?", "hm" }, { "lm?", "hm" },
                       { "cmd?", "hc" }, { "f?", "hf" }, { NULL, NULL } };
    for (int k = 0; help_alias[k].word != NULL; k++)
    {
        if (strcmp(firstword, help_alias[k].word) == 0)
        {
            sbuf_append_str(out, sigil);
            sbuf_append_str(out, help_alias[k].cmd);
            append_args(s + wlen, rem_len - wlen, 1, 0, d, out);
            return 1;
        }
    }
    static const char *const callwords[] = { "dpdigits",   "procstat", "procctl",     "procwait",
                                             "vars",       "history",  "searchhist",  "fhist",
                                             "ghistory",   "lhistory", "savehistory", "setprompt",
                                             "sessionlog", "synhl",    "syndiag",     "cliindent",
                                             "showmatch",  "helprl",   "fhelp",       "fparam",
                                             "help",       NULL };
    for (int k = 0; callwords[k] != NULL; k++)
    {
        if (strcmp(firstword, callwords[k]) == 0)
        {
            sbuf_append_str(out, "milkcall ");
            sbuf_append_str(out, firstword);
            append_args(s + wlen, rem_len - wlen, 1, 0, d, out);
            return 1;
        }
    }
    if (is_registered_milk_command(firstword))
    {
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, firstword);
        append_args(s + wlen, rem_len - wlen, 1, 0, d, out);
        return 1;
    }
    return 0;
}

/** is_assignment_word() - 1 if the word at @s is NAME=... */
static int is_assignment_word(const char *s, size_t n)
{
    if (n == 0 || !(isalpha((unsigned char) s[0]) || s[0] == '_'))
    {
        return 0;
    }
    for (size_t i = 1; i < n; i++)
    {
        if (s[i] == '=')
        {
            return 1;
        }
        if (!(isalnum((unsigned char) s[i]) || s[i] == '_'))
        {
            return 0;
        }
    }
    return 0;
}

static void translate_stmt_body(const char *stmt, size_t len, int d, sbuf_t *out)
{
    const char *sigil  = get_sigil();
    size_t      siglen = strlen(sigil);

    size_t i = 0;
    while (i < len && (stmt[i] == ' ' || stmt[i] == '\t'))
    {
        sbuf_append_ch(out, stmt[i]);
        i++;
    }
    /* leading flow keywords, `time`, and NAME=value prefixes do not change the command word */
    for (;;)
    {
        size_t k = skip_flow_prefix(stmt + i, len - i);
        if (k == 0 && g_allow_legacy)
        {
            size_t w = word_len(stmt + i, len - i, d);
            if (w < len - i && is_assignment_word(stmt + i, w))
            {
                emit_src(stmt + i, w, d, 1, out);
                i += w;
                while (i < len && (stmt[i] == ' ' || stmt[i] == '\t'))
                {
                    sbuf_append_ch(out, stmt[i]);
                    i++;
                }
                continue;
            }
        }
        if (k == 0)
        {
            break;
        }
        sbuf_append_mem(out, stmt + i, k);
        i += k;
    }
    if (i >= len)
    {
        return;
    }

    size_t end = len;
    while (end > i && (stmt[end - 1] == ' ' || stmt[end - 1] == '\t'))
    {
        end--;
    }
    size_t      slen = end - i;
    const char *s    = stmt + i;

    /* 0. Unquoted '|>' pipeline operator -> single-round-trip '@pipe "..."' */
    if (has_unquoted_pipe_op(s, slen, d))
    {
        size_t poff = 0;
        if (slen > siglen + 5 && strncmp(s, sigil, siglen) == 0 &&
            strncmp(s + siglen, "pipe", 4) == 0 && isspace((unsigned char) s[siglen + 4]))
        {
            poff = siglen + 4;
        }
        while (poff < slen && (s[poff] == ' ' || s[poff] == '\t'))
        {
            poff++;
        }
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, "pipe ");
        append_calc_expr(out, s + poff, slen - poff, d);
        sbuf_append_mem(out, stmt + end, len - end);
        return;
    }

    /* 1. Shorthand '@ [-v <var>] <expr>' calculator statement */
    if (slen > siglen && sig_len(s, slen) && (s[siglen] == ' ' || s[siglen] == '\t'))
    {
        size_t adv = siglen;
        while (adv < slen && (s[adv] == ' ' || s[adv] == '\t'))
        {
            adv++;
        }
        sbuf_append_str(out, sigil);
        sbuf_append_str(out, "calc ");
        if (adv + 3 < slen && strncmp(s + adv, "-v", 2) == 0 &&
            (s[adv + 2] == ' ' || s[adv + 2] == '\t'))
        {
            size_t vs = adv + 2;
            while (vs < slen && (s[vs] == ' ' || s[vs] == '\t'))
            {
                vs++;
            }
            size_t ve = vs + word_len(s + vs, slen - vs, d);
            if (ve > vs && ve < slen)
            {
                sbuf_append_str(out, "-v ");
                sbuf_append_mem(out, s + vs, ve - vs);
                sbuf_append_ch(out, ' ');
                adv = ve;
                while (adv < slen && (s[adv] == ' ' || s[adv] == '\t'))
                {
                    adv++;
                }
            }
        }
        if (is_one_quoted_word(s + adv, slen - adv, d))
        {
            sbuf_append_mem(out, s + adv, slen - adv);
        }
        else
        {
            append_calc_expr(out, s + adv, slen - adv, d);
        }
        sbuf_append_mem(out, stmt + end, len - end);
        return;
    }

    /* 2. Shorthand '@<entity>.<name>.<field>=<val>' or bare '@<entity>.<name>.<field>' */
    int legacy_ent =
        g_allow_legacy &&
        (strncmp(s, "s.", 2) == 0 || strncmp(s, "f.", 2) == 0 || strncmp(s, "p.", 2) == 0 ||
         strncmp(s, "q.", 2) == 0 || strncmp(s, "stream.", 7) == 0 || strncmp(s, "fps.", 4) == 0 ||
         strncmp(s, "proc.", 5) == 0 || strncmp(s, "seq.", 4) == 0);
    if ((slen > siglen && sig_len(s, slen)) || legacy_ent)
    {
        size_t adv    = legacy_ent ? 0 : siglen;
        int    has_eq = 0;
        size_t elen   = scan_entity_path(s + adv, &has_eq);
        if (elen > 0 && adv + elen <= slen)
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
                if (has_unquoted_space(s + vstart, slen - vstart, d))
                {
                    sbuf_append_str(out, "{ echo 'milk: ");
                    sbuf_append_mem(out, s + adv, elen);
                    sbuf_append_str(out, "=: quote multi-word values' >&2; false; }");
                    sbuf_append_mem(out, stmt + end, len - end);
                    return;
                }
                sbuf_append_str(out, sigil);
                sbuf_append_str(out, "set ");
                append_args(s + adv, elen, 1, 1, d, out);
                sbuf_append_ch(out, ' ');
                emit_src(s + vstart, slen - vstart, d, 1, out);
                sbuf_append_mem(out, stmt + end, len - end);
                return;
            }
            if (adv + elen == slen)
            {
                char ent_tmp[256];
                int  is_set_act = 0;
                if (elen < sizeof(ent_tmp))
                {
                    memcpy(ent_tmp, s + adv, elen);
                    ent_tmp[elen]                       = '\0';
                    static const char *const fps_act[]  = { "confstart", "confstop", "confstep",
                                                            "runstart",  "runstop",  "reset",
                                                            NULL };
                    static const char *const proc_act[] = { "run",  "resume", "pause",
                                                            "step", "incr",   "stop",
                                                            "exit", "kill",   NULL };
                    const char              *ldot       = strrchr(ent_tmp, '.');
                    int isf = (strncmp(ent_tmp, "f.", 2) == 0 || strncmp(ent_tmp, "fps.", 4) == 0);
                    int isp =
                        (strncmp(ent_tmp, "p.", 2) == 0 || strncmp(ent_tmp, "proc.", 5) == 0 ||
                         strncmp(ent_tmp, "procinfo.", 9) == 0);
                    const char *const *acts = isf ? fps_act : (isp ? proc_act : NULL);
                    for (int a = 0; ldot != NULL && acts != NULL && acts[a] != NULL; a++)
                    {
                        if (strcmp(ldot + 1, acts[a]) == 0)
                        {
                            is_set_act = 1;
                        }
                    }
                }
                sbuf_append_str(out, sigil);
                sbuf_append_str(out, is_set_act ? "set " : "get ");
                append_args(s + adv, elen, 1, 1, d, out);
                sbuf_append_mem(out, stmt + end, len - end);
                return;
            }
        }
    }

    size_t wlen = word_len(s, slen, d);
    char   firstword[256];
    if (wlen >= sizeof(firstword))
    {
        emit_src(s, len - i, d, 1, out);
        return;
    }
    memcpy(firstword, s, wlen);
    firstword[wlen] = '\0';

    /* 3. Already sigiled '@<cmd> ...': the command word itself is never expanded */
    if (sig_len(firstword, wlen) > 0)
    {
        sbuf_append_mem(out, s, wlen);
        append_args(s + wlen, len - (i + wlen), 1, is_entity_taking_cmd(firstword), d, out);
        return;
    }

    /* 4. Interactive milk-cli shorthands when legacy translation is on */
    if (g_allow_legacy && try_translate_interactive_word(firstword, s, len - i, wlen, d, out))
    {
        return;
    }

    /* Default: standard bash command */
    emit_src(s, wlen, d, 1, out);
    append_args(s + wlen, len - (i + wlen), 0, 0, d, out);
}

static void translate_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
{
    if (len == 0)
    {
        return;
    }
    int saved     = g_skip_dq_ent;
    g_skip_dq_ent = is_calc_or_test_stmt_prefix(stmt, len);
    translate_stmt_body(stmt, len, d, out);
    g_skip_dq_ent = saved;
}

/** blank_or_fn_head() - 1 if @s[0..n) is blanks, or ends with `)` / starts with `function `. */
static int blank_or_fn_head(const char *s, size_t n)
{
    size_t a = 0;
    while (a < n && (s[a] == ' ' || s[a] == '\t'))
    {
        a++;
    }
    size_t b = n;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t'))
    {
        b--;
    }
    return a == b || s[b - 1] == ')' || (b - a > 9 && strncmp(s + a, "function ", 9) == 0);
}

/**
 * translate_code() - translate shell code [@p, @p + @n) at command-substitution depth @d:
 * split it into statements at unquoted separators of that depth and translate each one.
 * Comments and here-document bodies are copied verbatim.
 */
static void translate_code(const char *p, size_t n, int d, sbuf_t *out)
{
    size_t st   = 0;
    size_t i    = 0;
    int    calc = -1; /* current statement is `@ <expr>` (only ; and newline end it) */
    while (i < n)
    {
        const char *q = p + i;
        int         c = cx(q);
        if (dp(q) > d)
        {
            i++;
            continue;
        }
        if (c == MSL_COMMENT || c == MSL_HEREDOC)
        {
            translate_stmt(p + st, i - st, d, out);
            size_t j = i;
            while (j < n && cx(p + j) == c && dp(p + j) == d)
            {
                j++;
            }
            sbuf_append_mem(out, q, j - i);
            st = i = j;
            calc   = -1;
            continue;
        }
        if (c != MSL_CODE)
        {
            i++;
            continue;
        }
        if (*q == '\\')
        {
            i += 2;
            continue;
        }
        size_t sp = inline_span(p, n, i, d);
        if (sp > 0)
        {
            if (calc < 0)
            {
                calc = is_calc_shorthand(p + st, n - st);
            }
            i += sp;
            continue;
        }
        if (calc < 0 && !isspace((unsigned char) *q))
        {
            calc = is_calc_shorthand(p + st, n - st);
        }
        char   ch  = *q;
        char   n1  = (i + 1 < n) ? q[1] : '\0';
        size_t sep = 0;
        if (ch == '\n')
        {
            sep = 1;
        }
        else if (ch == ';')
        {
            sep = (n1 == ';' || n1 == '&') ? ((n1 == ';' && i + 2 < n && q[2] == '&') ? 3 : 2) : 1;
        }
        else if (calc == 1)
        {
            sep = 0;
        }
        else if ((ch == '&' && n1 == '&') || (ch == '|' && n1 == '|') || (ch == '|' && n1 == '&'))
        {
            sep = 2;
        }
        else if (ch == '|' && n1 == '>')
        {
            i += 2;
            continue;
        }
        else if (ch == '|')
        {
            sep = 1;
        }
        else if (ch == '&' && n1 != '>' && !(i > 0 && (q[-1] == '>' || q[-1] == '<')))
        {
            sep = 1;
        }
        else if (ch == '(' && strspn(p + st, " \t") >= i - st) /* subshell */
        {
            sep = 1;
        }
        else if (ch == '{' && (i + 1 >= n || isspace((unsigned char) n1)) &&
                 blank_or_fn_head(p + st, i - st)) /* group or function body */
        {
            sep = 1;
        }
        if (sep > 0)
        {
            translate_stmt(p + st, i - st, d, out);
            sbuf_append_mem(out, q, sep);
            i += sep;
            st   = i;
            calc = -1;
            continue;
        }
        i++;
    }
    translate_stmt(p + st, n - st, d, out);
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
    size_t n = strlen(src);
    if (msl_scan(src, n, &g_map) != 0)
    {
        return NULL;
    }
    g_src          = src;
    g_allow_legacy = allow_legacy;
    g_skip_dq_ent  = 0;

    sbuf_t out;
    sbuf_init(&out);
    translate_code(src, n, 0, &out);

    msl_free(&g_map);
    g_src = NULL;
    return out.data;
}
