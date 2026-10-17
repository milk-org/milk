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

#include "milk_bash_engine_internal.h"
#include "milk_catalog.h"
#include "milk_entity_ns.h"
#include "milk_lexicon.h"
#include "milk_shell_lex.h"
#include "milk_renames.h"

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
    return "@";
}

/* ============================================================
 *  Entity & Command Recognition Helpers
 * ============================================================ */

static int is_entity_prefix(const char *s, size_t *out_len)
{
    const char *rest = s;
    if (milk_entity_ns_kind(s, &rest) == MILK_NS_NONE ||
        !(isalnum((unsigned char) *rest) || *rest == '_' || *rest == '*' || *rest == '?' ||
          *rest == '$'))
    {
        return 0;
    }
    if (out_len != NULL)
    {
        *out_len = (size_t) (rest - s);
    }
    return 1;
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

/** is_qualified_milk_command() - 1 if @w is a catalogued `module.command` (the installed modules). */
static int is_qualified_milk_command(const char *w)
{
    const char *dot = (w != NULL) ? strchr(w, '.') : NULL;
    if (dot == NULL || dot == w || dot[1] == '\0')
    {
        return 0;
    }
    return milk_catalog_has(w);
}

/* ============================================================
 *  Translation state: the source and its context map (milk_shell_lex)
 * ============================================================ */

static const char *g_src = NULL; /* text being translated */
static msl_map_t   g_map;        /* context of every byte of g_src */
static int         g_allow_legacy = 0;
static int   g_skip_dq_ent    = 0; /* current statement is @calc/@test/...: calc reads "@s.x.y" */
static int   g_modernize      = 0; /* translate_code() rewrites removed spellings instead */
static FILE *g_report         = NULL; /* where the modernize pass reports changes */
static const char *g_fname    = "-";
static int         g_nchanges = 0;

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
static void report_note(const char *at, const char *word, const char *why);

/** append_inline_label() - the label a failed @{@body} reports: ` '@{body}'`, single-quoted for the shell. */
static void append_inline_label(sbuf_t *out, const char *body, size_t n)
{
    sbuf_append_str(out, " '@{");
    for (size_t k = 0; k < n; k++)
    {
        if (body[k] == '\'')
        {
            sbuf_append_str(out, "'\\''");
        }
        else
        {
            sbuf_append_ch(out, body[k]);
        }
    }
    sbuf_append_str(out, "}'");
}

/**
 * emit_inline() - emit a milkget/milkcalc substitution (funsub, or $() on old bash); @dq wraps it in
 * "..." so the result is neither word-split nor globbed. A failure calls __milk_inline_err (milk.bash).
 * The builtins are called directly: the @get/@calc functions add a few µs to every expansion.
 */
static void emit_inline(sbuf_t *out, int is_calc, const char *body, size_t n, int dq)
{
    if (g_modernize) /* the removed inline forms (@s.x.y, ${@s.x.y}) become @{...} */
    {
        sbuf_append_str(out, "@{");
        sbuf_append_mem(out, body, n);
        sbuf_append_ch(out, '}');
        return;
    }
    int funsub = !milk_bash_funsub_disabled();
    if (dq)
    {
        sbuf_append_ch(out, '"');
    }
    sbuf_append_str(out, funsub ? "${| " : "$(");
    if (is_calc)
    {
        sbuf_append_str(out, funsub ? "milkcalc -v REPLY \"" : "milkcalc \"");
        sbuf_append_dq_escaped(out, body, n);
        sbuf_append_ch(out, '"');
    }
    else
    {
        sbuf_append_str(out, funsub ? "milkget -v REPLY " : "milkget ");
        sbuf_append_mem(out, body, n);
    }
    sbuf_append_str(out, " || __milk_inline_err $?");
    if (!is_calc) /* calc reports its own errors */
    {
        append_inline_label(out, body, n);
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

/** inline_close() - index in @p of the `}` closing the `{` at @p[@i] (counted in that brace's context and
 * depth @d), or 0. */
static size_t inline_close(const char *p, size_t n, size_t i, int d)
{
    int c     = cx(p + i);
    int depth = 1;
    for (size_t j = i + 1; j < n; j++)
    {
        if (cx(p + j) == c && dp(p + j) == d && (depth += (p[j] == '{') - (p[j] == '}')) == 0)
        {
            return j;
        }
    }
    return 0;
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
        sl        = sig_len(q + 1, n - i - 1);
        int brace = (sl > 0 && i + 1 + sl < n && q[1 + sl] == '{');
        if (c == MSL_DQ && sl > 0 && (brace ? !g_modernize : g_modernize))
        {
            /* "\@{..}" is a literal @{..}; modernizing, the old "\@s.x" escape is no longer needed */
            sbuf_append_mem(out, q + 1, sl);
            return 1 + sl;
        }
        sbuf_append_mem(out, q, 2); /* escaped character: never the start of a construct */
        return 2;
    }

    if (g_modernize && c == MSL_PARAM && q[0] == '$' && i + 2 < n && q[1] == '{' &&
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
        if (isalnum((unsigned char) prev) || prev == '_')
        {
            return 0; /* foo@{a,b} is bash brace expansion, "HEAD@{1}" is text */
        }
        size_t j = inline_close(p, n, i + sl, d);
        if (j == 0)
        {
            return 0;
        }
        const char *b  = q + sl + 1;
        size_t      bn = j - (i + sl + 1);
        if (g_modernize) /* already the modern form */
        {
            sbuf_append_mem(out, q, j + 1 - i);
            return j + 1 - i;
        }
        emit_inline(out, entity_ref_len(b) != bn, b, bn, c != MSL_DQ); /* @{s.x.y}: milkget */
        return j + 1 - i;
    }

    if (!ent || !g_modernize || c == MSL_ARITH || (c == MSL_DQ && g_skip_dq_ent) ||
        isalnum((unsigned char) prev) || prev == '_' || prev == '.' || prev == '$')
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
            strcmp(cmd, "config") == 0 || strcmp(cmd, "grab") == 0 || strcmp(cmd, "replay") == 0);
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
 * try_translate_module_cmd() - `module.command args`: a function of the catalog; its arguments get the
 * milk-command treatment (unquoted globs and slices quoted).
 *
 * @return 1 if handled, 0 otherwise
 */
static int try_translate_module_cmd(const char *firstword,
                                    const char *s,
                                    size_t      rem_len,
                                    size_t      wlen,
                                    int         d,
                                    sbuf_t     *out)
{
    if (!is_qualified_milk_command(firstword))
    {
        return 0;
    }
    sbuf_append_str(out, firstword);
    append_args(s + wlen, rem_len - wlen, 1, 0, d, out);
    return 1;
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

/**
 * modernize_legacy_stmt() - second modernize pass: rewrite the removed statement forms (`@ expr`,
 * `@s.x.y`, `@f.x.p=v`, `a |> b`) and inline entities (`@s.x.y`, `${@s.x.y}`) into their keywords and
 * `@{...}`, and quote the globs and slices that milk commands used to receive unexpanded.
 */
static void modernize_legacy_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
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
    if (slen > siglen && sig_len(s, slen))
    {
        size_t adv    = siglen;
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
                    report_note(stmt + i, "@<entity>=<words>",
                                "has an unquoted multi-word value: quote it");
                    sbuf_append_mem(out, stmt + i, len - i);
                    return;
                }
                sbuf_append_str(out, sigil);
                sbuf_append_str(out, "set ");
                append_args(s + adv, elen, 1, 1, d, out);
                sbuf_append_ch(out, '=');
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
                    milk_ns_kind_t           ek         = milk_entity_ns_kind(ent_tmp, NULL);
                    int                      isf        = (ek == MILK_NS_FPS);
                    int                      isp        = (ek == MILK_NS_PROC);
                    const char *const       *acts       = isf ? fps_act : (isp ? proc_act : NULL);
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

    /* `source f.milk` / `. f.milk`: plain bash now; milk scripts are sourced with @source */
    if ((strcmp(firstword, "source") == 0 || strcmp(firstword, ".") == 0) && wlen < slen)
    {
        size_t a = wlen;
        while (a < slen && (s[a] == ' ' || s[a] == '\t'))
        {
            a++;
        }
        size_t fl = word_len(s + a, slen - a, d);
        if (fl > 5 && strncmp(s + a + fl - 5, ".milk", 5) == 0)
        {
            sbuf_append_str(out, "@source");
            emit_src(s + wlen, len - (i + wlen), d, 1, out);
            return;
        }
    }

    /* 3. Already sigiled '@<cmd> ...': the command word itself is never expanded */
    if (sig_len(firstword, wlen) > 0)
    {
        sbuf_append_mem(out, s, wlen);
        append_args(s + wlen, len - (i + wlen), 1, is_entity_taking_cmd(firstword), d, out);
        return;
    }

    /* 4. Module command (`mem.listim`) */
    if (g_allow_legacy && try_translate_module_cmd(firstword, s, len - i, wlen, d, out))
    {
        return;
    }

    /* Default: standard bash command */
    emit_src(s, wlen, d, 1, out);
    append_args(s + wlen, len - (i + wlen), 0, 0, d, out);
}

/** report_change() - report the rewrite of @old (found at @at in the source) into @new_. */
static void report_change(const char *at, const char *old, size_t oldn, const char *new_)
{
    int line = 1;
    for (const char *q = g_src; q < at; q++)
    {
        line += (*q == '\n');
    }
    if (g_report != NULL)
    {
        fprintf(g_report, "%s:%d: '%.*s' -> '%s'\n", g_fname, line, (int) oldn, old, new_);
    }
    g_nchanges++;
}

/** is_module_cmd_word() - 1 if @w is `[@]module.cmd` (one dot). */
static int is_module_cmd_word(const char *w)
{
    w += (*w == '@');
    if (!islower((unsigned char) *w))
    {
        return 0;
    }
    const char *dot = strchr(w, '.');
    if (dot == NULL || dot[1] == '\0' || strchr(dot + 1, '.') != NULL)
    {
        return 0;
    }
    for (const char *q = w; *q != '\0'; q++)
    {
        if (!isalnum((unsigned char) *q) && *q != '_' && *q != '.')
        {
            return 0;
        }
    }
    return 1;
}

/** is_tag_word() - 1 if @p[0..n) is a parameter tag `.name` (no value). */
static int is_tag_word(const char *p, size_t n)
{
    if (n < 2 || p[0] != '.' || !(isalpha((unsigned char) p[1]) || p[1] == '_'))
    {
        return 0;
    }
    for (size_t k = 2; k < n; k++)
    {
        if (!isalnum((unsigned char) p[k]) && p[k] != '_' && p[k] != '.')
        {
            return 0;
        }
    }
    return 1;
}

/** report_note() - report @word (at @at in the source) that the modernize pass cannot rewrite. */
static void report_note(const char *at, const char *word, const char *why)
{
    int line = 1;
    for (const char *q = g_src; q < at; q++)
    {
        line += (*q == '\n');
    }
    if (g_report != NULL)
    {
        fprintf(g_report, "%s:%d: '%s' %s\n", g_fname, line, word, why);
    }
    g_nchanges++;
}

/* plain_word() - 1 if @p[0..n) is a literal shell word: no quote, expansion or glob */
static int plain_word(const char *p, size_t n)
{
    for (size_t k = 0; k < n; k++)
    {
        if (strchr("'\"$`\\*?[]{}()<>|&;", p[k]) != NULL)
        {
            return 0;
        }
    }
    return n > 0;
}

/**
 * modernize_args() - rewrite statement `@old args` whose arguments move (milk_renames.c MILK_REN_ARGS)
 * into @out, from the arguments at @a; 0 if they are not literal words (the user rewrites it).
 */
static int modernize_args(const char *old,
                          const char *stmt,
                          size_t      a,
                          size_t      len,
                          int         d,
                          sbuf_t     *out)
{
    char   arg[4][256];
    int    n = 0;
    char   rep[1100];
    size_t end = len;

    while (end > a && (stmt[end - 1] == ' ' || stmt[end - 1] == '\t' || stmt[end - 1] == '\n'))
    {
        end--;
    }
    while (a < end)
    {
        while (a < end && (stmt[a] == ' ' || stmt[a] == '\t'))
        {
            a++;
        }
        size_t wl = word_len(stmt + a, end - a, d);
        if (wl == 0 || n == 4 || wl >= sizeof(arg[0]) || !plain_word(stmt + a, wl))
        {
            return 0;
        }
        memcpy(arg[n], stmt + a, wl);
        arg[n++][wl] = '\0';
        a += wl;
    }

    const char *x0 = (n > 0) ? arg[0] : "";
    if (strcmp(old, "@usleep") == 0 && n == 1)
    {
        char *e;
        long  us = strtol(x0, &e, 10);
        if (*e != '\0' || us < 0)
        {
            return 0;
        }
        snprintf(rep, sizeof(rep), "@sleep %g", (double) us * 1e-6);
    }
    else if (strcmp(old, "@mloadas") == 0 && n == 2)
    {
        snprintf(rep, sizeof(rep), "@module load %s --as %s", x0, arg[1]);
    }
    else if (strcmp(old, "@fpsset") == 0 && (n == 3 || n == 4))
    {
        const char *f = (strncmp(x0, "f.", 2) == 0) ? x0 + 2 : x0;
        snprintf(rep, sizeof(rep), "@set f.%s.%s=%s%s%s", f, arg[1], arg[2], (n == 4) ? " " : "",
                 (n == 4) ? arg[3] : "");
    }
    else if (strcmp(old, "@fpsset") == 0 && n == 2)
    {
        snprintf(rep, sizeof(rep), "@set %s%s=%s", (strncmp(x0, "f.", 2) == 0) ? "" : "f.", x0,
                 arg[1]);
    }
    else if (strcmp(old, "@procctl") == 0 && n == 2)
    {
        snprintf(rep, sizeof(rep), "@set p.%s.ctrl=%s", (strncmp(x0, "p.", 2) == 0) ? x0 + 2 : x0,
                 arg[1]);
    }
    else if (strcmp(old, "@procwait") == 0 && n >= 1 && n <= 3)
    {
        snprintf(rep, sizeof(rep), "@waitfor p.%s.status=%s -t %s",
                 (strncmp(x0, "p.", 2) == 0) ? x0 + 2 : x0, (n >= 2) ? arg[1] : "STOPPED",
                 (n == 3) ? arg[2] : "30");
    }
    else if (strcmp(old, "@procstat") == 0 && n <= 1)
    {
        if (n == 0)
        {
            snprintf(rep, sizeof(rep), "@ls -l 'p.*'");
        }
        else
        {
            snprintf(rep, sizeof(rep), "@get p.%s.list", (strncmp(x0, "p.", 2) == 0) ? x0 + 2 : x0);
        }
    }
    else
    {
        return 0;
    }
    sbuf_append_str(out, rep);
    sbuf_append_mem(out, stmt + end, len - end);
    return 1;
}

/**
 * tag_value_args() - 1 if the arguments [@i, @len) of @stmt are the removed sticky setting `.tag value`;
 * @tag and @val get the offsets of its two words, @vend the end of the value.
 */
static int tag_value_args(const char *stmt,
                          size_t      i,
                          size_t      len,
                          int         d,
                          size_t     *tag,
                          size_t     *val,
                          size_t     *vend)
{
    size_t a = i;
    while (a < len && (stmt[a] == ' ' || stmt[a] == '\t'))
    {
        a++;
    }
    size_t w1 = word_len(stmt + a, len - a, d);
    size_t b  = a + w1;
    while (b < len && (stmt[b] == ' ' || stmt[b] == '\t'))
    {
        b++;
    }
    size_t w2 = (b > a + w1) ? word_len(stmt + b, len - b, d) : 0;
    size_t e  = b + w2;
    while (e < len && (stmt[e] == ' ' || stmt[e] == '\t'))
    {
        e++;
    }
    *tag  = a;
    *val  = b;
    *vend = b + w2;
    return a > i && w2 > 0 && e == len && is_tag_word(stmt + a, w1);
}

/**
 * modernize_stmt() - copy statement @stmt, rewriting the removed spellings of milk_renames.h: the
 * command word, `@<old ns>.` paths anywhere, and bare `<old ns>.` paths in entity-taking arguments.
 */
static void modernize_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
{
    size_t i = 0;
    while (i < len && (stmt[i] == ' ' || stmt[i] == '\t'))
    {
        sbuf_append_ch(out, stmt[i++]);
    }
    for (size_t k; i < len && (k = skip_flow_prefix(stmt + i, len - i)) > 0; i += k)
    {
        sbuf_append_mem(out, stmt + i, k);
    }
    char   word[256] = "";
    size_t wstart    = i;
    size_t w         = (i < len) ? word_len(stmt + i, len - i, d) : 0;
    if (w > 0 && w < sizeof(word))
    {
        memcpy(word, stmt + i, w);
        word[w]            = '\0';
        milk_ren_how_t how = MILK_REN_WORD;
        const char *nw = (strcmp(word, "@") != 0) ? milk_rename_cmd(word, &how) : NULL; /* pass 2 */
        if (nw != NULL && how == MILK_REN_ARGS &&
            modernize_args(word, stmt, wstart + w, len, d, out))
        {
            report_change(stmt + i, stmt + i, len - i, "(rewritten)");
            return;
        }
        if (nw != NULL && how != MILK_REN_WORD)
        {
            char why[256];
            snprintf(why, sizeof(why), "was removed: rewrite it as %s", nw);
            report_note(stmt + i, word, why);
            nw = NULL;
        }
        else if (nw == NULL && word[0] == '@' && word[1] != '\0' &&
                 milk_lexicon_keyword(word + 1) == NULL)
        {
            nw = (strchr(word + 1, '.') == NULL)       ? milk_catalog_short(word + 1) /* @short */
                 : is_qualified_milk_command(word + 1) ? word + 1 /* @module.cmd */
                                                       : NULL;
            if (nw != NULL && strcmp(nw, "?") == 0)
            {
                report_note(stmt + i, word, "names several module commands: qualify it");
                nw = NULL;
            }
        }
        if (nw != NULL)
        {
            report_change(stmt + i, stmt + i, w, nw);
            sbuf_append_str(out, nw);
            i += w;
            memmove(word, nw, strlen(nw) + 1); /* nw may point into word (`@module.cmd`) */
        }
    }
    size_t tg, vl, ve;
    if (w > 0 && is_module_cmd_word(word) &&
        tag_value_args(stmt, wstart + w, len, d, &tg, &vl, &ve))
    {
        char   rep[1024];
        size_t tl = word_len(stmt + tg, len - tg, d);
        snprintf(rep, sizeof(rep), "--default %.*s=%.*s", (int) tl, stmt + tg, (int) (ve - vl),
                 stmt + vl);
        report_change(stmt + tg, stmt + tg, ve - tg, rep);
        sbuf_append_mem(out, stmt + i, tg - i); /* the command word unless renamed, the blanks */
        sbuf_append_str(out, rep);
        sbuf_append_mem(out, stmt + ve, len - ve);
        return;
    }
    int ent_args = is_entity_taking_cmd(word) || is_calc_or_test_stmt_prefix(stmt, len);
    while (i < len)
    {
        if (dp(stmt + i) == d + 1 &&
            cx(stmt + i) == MSL_OPEN) /* $(...), `...`: modernize the body */
        {
            size_t j = i;
            while (j < len && cx(stmt + j) == MSL_OPEN && dp(stmt + j) == d + 1)
            {
                j++;
            }
            size_t k = j;
            while (k < len && !(cx(stmt + k) == MSL_CLOSE && dp(stmt + k) == d + 1))
            {
                k++;
            }
            sbuf_append_mem(out, stmt + i, j - i);
            translate_code(stmt + j, k - j, d + 1, out);
            i = k;
            continue;
        }
        const char *p      = stmt + i;
        int         c      = cx(p);
        size_t      at     = (*p == '@') ? 1 : 0;
        size_t      oldlen = 0;
        const char *nn     = NULL;
        int         bound  = (p == g_src) || !(isalnum((unsigned char) p[-1]) || p[-1] == '_' ||
                                               p[-1] == '.' || p[-1] == '$' || p[-1] == '@');
        if (bound && (c == MSL_CODE || c == MSL_DQ || c == MSL_SQ) && (at || ent_args))
        {
            nn = milk_rename_ns(p + at, &oldlen);
        }
        if (nn != NULL && (isalnum((unsigned char) p[at + oldlen]) || p[at + oldlen] == '_' ||
                           p[at + oldlen] == '*' || p[at + oldlen] == '$'))
        {
            char rep[64];
            snprintf(rep, sizeof(rep), "%s%s", at ? "@" : "", nn);
            report_change(p, p, at + oldlen, rep);
            sbuf_append_str(out, rep);
            i += at + oldlen;
            continue;
        }
        sbuf_append_ch(out, *p);
        i++;
    }
}

/** reads_var() - 1 if the script expands shell variable @name[0..@n) ($name, ${name...}). */
static int reads_var(const char *name, size_t n)
{
    for (const char *q = strchr(g_src, '$'); q != NULL; q = strchr(q + 1, '$'))
    {
        const char *v = q + 1 + (q[1] == '{');
        if (strncmp(v, name, n) == 0 && !isalnum((unsigned char) v[n]) && v[n] != '_')
        {
            return 1;
        }
    }
    return 0;
}

/** calc_target() - length of the variable that calc statement @s[0..@n) assigns (`T = e`, `T op= e`), or
 * 0; *@op gets the operator ('=' or one of + - * /) and *@rhs the offset of e. */
static size_t calc_target(const char *s, size_t n, size_t *t0, char *op, size_t *rhs)
{
    size_t t = 0;
    while (t < n && (s[t] == ' ' || s[t] == '\t'))
    {
        t++;
    }
    *t0 = t;
    if (t >= n || !(isalpha((unsigned char) s[t]) || s[t] == '_'))
    {
        return 0;
    }
    while (t < n && (isalnum((unsigned char) s[t]) || s[t] == '_'))
    {
        t++;
    }
    size_t tn = t - *t0;
    while (t < n && (s[t] == ' ' || s[t] == '\t'))
    {
        t++;
    }
    if (t + 1 < n && strchr("+-*/", s[t]) != NULL && s[t + 1] == '=')
    {
        *op = s[t];
        t += 2;
    }
    else if (t < n && s[t] == '=' && (t + 1 >= n || s[t + 1] != '='))
    {
        *op = '=';
        t++;
    }
    else
    {
        return 0;
    }
    while (t < n && (s[t] == ' ' || s[t] == '\t'))
    {
        t++;
    }
    *rhs = t;
    return (t < n) ? tn : 0;
}

/**
 * modernize_calc_bind() - calc variables no longer come back to bash by themselves: the statement in @out
 * from @at, `@calc "T = e"` (or `T op= e`) whose T the script reads as $T, becomes `@calc -v T "e"`.
 * Reported at @stmt when it cannot be rewritten.
 */
static void modernize_calc_bind(const char *stmt, sbuf_t *out, size_t at)
{
    char  *s = strndup(out->data + at, out->len - at);
    size_t n = (s != NULL) ? strlen(s) : 0;
    size_t i = 0, k, t0, rhs;
    char   op;

    while (i < n && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    while (i < n && (k = skip_flow_prefix(s + i, n - i)) > 0)
    {
        i += k;
    }
    size_t a = i + 5;
    size_t e = n;
    while (e > a && isspace((unsigned char) s[e - 1]))
    {
        e--;
    }
    if (s == NULL || n < i + 8 || strncmp(s + i, "@calc", 5) != 0 || (s[a] != ' ' && s[a] != '\t'))
    {
        free(s);
        return;
    }
    while (a < e && (s[a] == ' ' || s[a] == '\t'))
    {
        a++;
    }
    char        q    = s[a];
    const char *body = s + a + 1;
    size_t      bl   = (e - a >= 2) ? e - a - 2 : 0;
    if ((q != '"' && q != '\'') || e - a < 2 || s[e - 1] != q || memchr(body, q, bl) != NULL)
    {
        free(s);
        return;
    }
    size_t tn = calc_target(body, bl, &t0, &op, &rhs);
    if (tn > 0 && memchr(body, ';', bl) == NULL && memchr(body, '\\', bl) == NULL &&
        reads_var(body + t0, tn))
    {
        sbuf_t r;
        sbuf_init(&r);
        sbuf_append_mem(&r, s, i);
        sbuf_append_str(&r, "@calc -v ");
        sbuf_append_mem(&r, body + t0, tn);
        sbuf_append_ch(&r, ' ');
        sbuf_append_ch(&r, q);
        if (op != '=') /* T op= e: T op (e) */
        {
            sbuf_append_mem(&r, body + t0, tn);
            sbuf_append_ch(&r, ' ');
            sbuf_append_ch(&r, op);
            sbuf_append_str(&r, " (");
        }
        sbuf_append_mem(&r, body + rhs, bl - rhs);
        sbuf_append_str(&r, (op != '=') ? ")" : "");
        sbuf_append_ch(&r, q);
        sbuf_append_mem(&r, s + e, n - e);
        out->len = at;
        sbuf_append_mem(out, r.data, r.len);
        free(r.data);
        free(s);
        return;
    }
    for (size_t b = 0; b < bl;) /* the statements of a list, or one with a quote or backslash */
    {
        size_t sl = b;
        while (sl < bl && body[sl] != ';')
        {
            sl++;
        }
        if ((tn = calc_target(body + b, sl - b, &t0, &op, &rhs)) > 0 &&
            reads_var(body + b + t0, tn))
        {
            char word[128];
            snprintf(word, sizeof(word), "%.*s", (int) tn, body + b + t0);
            report_note(stmt, word, "is set in calc only now: bind it with @calc -v");
        }
        b = sl + 1;
    }
    free(s);
}

/**
 * emit_assign_inline() - a statement that is only `NAME=@{...}` or `NAME="@{...}"` binds NAME with
 * `milkget -v` / `milkcalc -v`, without a substitution: a few µs instead of ~11 (or a fork on old bash).
 * Grouped, so `a && x=@{...} || b` keeps its meaning; on failure NAME is emptied, as the substitution
 * would leave it. 1 if @out got the statement.
 */
static int emit_assign_inline(const char *s, size_t n, int d, sbuf_t *out)
{
    size_t i = 0;
    size_t k;
    while (i < n && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    while (i < n && (k = skip_flow_prefix(s + i, n - i)) > 0)
    {
        i += k;
    }
    size_t e = n;
    while (e > i && (s[e - 1] == ' ' || s[e - 1] == '\t'))
    {
        e--;
    }
    size_t a = i;
    if (a >= e || !code_at(s + a, d) || !(isalpha((unsigned char) s[a]) || s[a] == '_'))
    {
        return 0;
    }
    while (a < e && (isalnum((unsigned char) s[a]) || s[a] == '_'))
    {
        a++;
    }
    size_t v  = a + 1; /* the value: @{...} or "@{...}" */
    int    dq = (v < e && s[v] == '"');
    size_t o  = v + dq; /* the sigil */
    if (v >= e || s[a] != '=' || !code_at(s + a, d) ||
        (dq ? cx(s + o) != MSL_DQ : !code_at(s + o, d)))
    {
        return 0;
    }
    size_t sl = sig_len(s + o, e - o);
    size_t j  = (sl > 0 && o + sl < e && s[o + sl] == '{') ? inline_close(s, e, o + sl, d) : 0;
    if (j == 0 || j + 1 + dq != e || (dq && s[j + 1] != '"'))
    {
        return 0;
    }
    const char *b       = s + o + sl + 1;
    size_t      bn      = j - (o + sl + 1);
    int         is_calc = (entity_ref_len(b) != bn);

    sbuf_append_mem(out, s, i); /* blanks and flow keywords */
    sbuf_append_str(out, is_calc ? "{ milkcalc -v " : "{ milkget -v ");
    sbuf_append_mem(out, s + i, a - i);
    if (is_calc)
    {
        sbuf_append_str(out, " \"");
        sbuf_append_dq_escaped(out, b, bn);
        sbuf_append_str(out, "\" || __milk_inline_err $? ''");
    }
    else
    {
        sbuf_append_ch(out, ' ');
        sbuf_append_mem(out, b, bn);
        sbuf_append_str(out, " || __milk_inline_err $?");
        append_inline_label(out, b, bn);
    }
    sbuf_append_ch(out, ' ');
    sbuf_append_mem(out, s + i, a - i);
    sbuf_append_str(out, "; }");
    sbuf_append_mem(out, s + e, n - e);
    return 1;
}

static void translate_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
{
    if (len == 0)
    {
        return;
    }
    if (g_modernize == 1)
    {
        modernize_stmt(stmt, len, d, out);
        return;
    }
    if (g_modernize == 2)
    {
        size_t at     = out->len;
        int    saved  = g_skip_dq_ent;
        g_skip_dq_ent = is_calc_or_test_stmt_prefix(stmt, len);
        modernize_legacy_stmt(stmt, len, d, out);
        if (g_skip_dq_ent) /* calc text names entities without @: "@s.x.y * 2" -> "s.x.y * 2" */
        {
            size_t w = at;
            for (size_t r = at; r < out->len; r++)
            {
                char pc = (r > at) ? out->data[r - 1] : ' ';
                if (!(out->data[r] == '@' && entity_ref_len(out->data + r + 1) > 0 &&
                      !isalnum((unsigned char) pc) && pc != '_' && pc != '.' && pc != '$'))
                {
                    out->data[w++] = out->data[r];
                }
            }
            out->len     = w;
            out->data[w] = '\0';
        }
        g_skip_dq_ent = saved;
        modernize_calc_bind(stmt, out, at);
        if (out->len - at != len || memcmp(out->data + at, stmt, len) != 0)
        {
            char *nw = strndup(out->data + at, out->len - at);
            report_change(stmt, stmt, len, (nw != NULL) ? nw : "");
            free(nw);
        }
        return;
    }
    if (!emit_assign_inline(stmt, len, d, out))
    {
        emit_src(stmt, len, d, 0, out); /* only @{...} (and \@) is milk syntax */
    }
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
 * @param allow_legacy Non-zero to also translate un-sigiled `module.command` names
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
    (void) allow_legacy; /* the removed forms are no longer translated (milk-script --modernize) */
    g_src          = src;
    g_allow_legacy = 0;
    g_skip_dq_ent  = 0;

    sbuf_t out;
    sbuf_init(&out);
    translate_code(src, n, 0, &out);

    msl_free(&g_map);
    g_src = NULL;
    return out.data;
}

/* modernize_pass() - run modernize pass @pass (1: words, 2: statement forms) over @src */
static char *modernize_pass(const char *src,
                            int         pass,
                            const char *fname,
                            FILE       *report,
                            int        *nchanges)
{
    size_t n = strlen(src);
    if (msl_scan(src, n, &g_map) != 0)
    {
        return NULL;
    }
    g_src          = src;
    g_allow_legacy = 1;
    g_skip_dq_ent  = 0;
    g_modernize    = pass;
    g_report       = report;
    g_fname        = (fname != NULL) ? fname : "-";
    g_nchanges     = 0;

    sbuf_t out;
    sbuf_init(&out);
    translate_code(src, n, 0, &out);

    g_modernize    = 0;
    g_allow_legacy = 0;
    msl_free(&g_map);
    g_src = NULL;
    *nchanges += g_nchanges;
    return out.data;
}

char *milk_bash_modernize(const char *src, const char *fname, FILE *report, int *nchanges)
{
    int   n  = 0;
    char *s1 = (src != NULL) ? modernize_pass(src, 1, fname, report, &n) : NULL;
    char *s2 = (s1 != NULL) ? modernize_pass(s1, 2, fname, report, &n) : NULL;

    free(s1);
    if (nchanges != NULL)
    {
        *nchanges = n;
    }
    return s2;
}
