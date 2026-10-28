// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_translate.c
 * @brief Line-preserving translator for `@{...}` inline milk.bash expansions.
 *
 * Role & Goal:
 *   Translates `.milk` scripts and interactive bash lines into executable
 *   `milk.bash` code while preserving 1:1 line numbering (`$LINENO`):
 *   - Standalone `NAME=@{<expr>}` / `NAME="@{<expr>}"` -> `{ milkcalc -v NAME "<expr>" ...; }`
 *   - Standalone `NAME=@{<entity>}` -> `{ milkget -v NAME <entity> ...; }`
 *   - Inline `@{<expr>}` -> `${| milkcalc -v REPLY "<expr>" ...; }` (or `$(milkcalc ...)` < 5.3)
 *   - Inline `@{<entity>}` -> `${| milkget -v REPLY <entity> ...; }` (or `$(milkget ...)` < 5.3)
 *   - Escaped `\@{...}` inside double quotes -> literal `@{...}`
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_bash_engine_internal.h"
#include "milk_entity_ns.h"
#include "milk_shell_lex.h"

static const char *g_mbt_src = NULL;
static msl_map_t   g_mbt_map;

static void mbt_translate_code(const char *p, size_t n, int d, sbuf_t *out);

/** Return the `msl_ctx_t` lexical context of byte @p within `g_mbt_src`. */
static inline int mbt_cx(const char *p)
{
    return g_mbt_map.ctx[p - g_mbt_src];
}

/** Return the command-substitution nesting depth of byte @p within `g_mbt_src`. */
static inline int mbt_dp(const char *p)
{
    return g_mbt_map.depth[p - g_mbt_src];
}

/** Check whether @p is unquoted shell code at substitution depth @d. */
static inline int mbt_code_at(const char *p, int d)
{
    return mbt_cx(p) == MSL_CODE && mbt_dp(p) == d;
}

/** Return 1 if @p[0..@n) starts with `@`, else 0. */
static inline size_t mbt_sig_len(const char *p, size_t n)
{
    return (n >= 1 && p[0] == '@') ? 1 : 0;
}

/** Check whether bash 5.3+ `${| ...; }` funsub is unavailable (cached in `MILK_BASH_NO_FUNSUB`). */
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

/** Check whether @s begins with a canonical entity namespace prefix (`s.`, `p.`, `f.`, ...). */
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

/** Scan a dotted entity property path starting at @s (without leading `@`), setting @has_eq on `=`. */
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

/** Append single-quoted `'@{body}'` error label to @out. */
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

/** Emit a `milkget` or `milkcalc` substitution for an inline `@{...}` construct. */
static void emit_inline(sbuf_t *out, int is_calc, const char *body, size_t n, int dq)
{
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
    if (!is_calc)
    {
        append_inline_label(out, body, n);
    }
    sbuf_append_str(out, funsub ? "; }" : ")");
    if (dq)
    {
        sbuf_append_ch(out, '"');
    }
}

/** Return length of a bare `s.a.b` entity reference at @p (without leading `@`), or 0. */
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

/** 1 if the @bn bytes at @b, inside `@{…}`, are a signed integer: git's `@{1}` and `@{-1}`, left as written. */
static int git_brace(const char *b, size_t bn)
{
    size_t k = (bn > 0 && b[0] == '-') ? 1 : 0;
    if (k == bn)
    {
        return 0;
    }
    for (; k < bn; k++)
    {
        if (!isdigit((unsigned char) b[k]))
        {
            return 0;
        }
    }
    return 1;
}

/** Find the matching `}` for the `{` at `@p[@i]` at lexical depth @d, or 0 if unclosed. */
static size_t inline_close(const char *p, size_t n, size_t i, int d)
{
    int c     = mbt_cx(p + i);
    int depth = 1;
    for (size_t j = i + 1; j < n; j++)
    {
        if (mbt_cx(p + j) == c && mbt_dp(p + j) == d &&
            (depth += (p[j] == '{') - (p[j] == '}')) == 0)
        {
            return j;
        }
    }
    return 0;
}

/** Expand an inline construct (`\@` or `@{expr}`) at `@p[@i]`, returning bytes consumed or 0. */
static size_t try_inline(const char *p, size_t n, size_t i, int d, sbuf_t *out)
{
    const char *q  = p + i;
    int         c  = mbt_cx(q);
    size_t      sl = 0;

    if ((c == MSL_CODE || c == MSL_DQ) && q[0] == '\\' && i + 1 < n)
    {
        sl        = mbt_sig_len(q + 1, n - i - 1);
        int brace = (sl > 0 && i + 1 + sl < n && q[1 + sl] == '{');
        if (c == MSL_DQ && brace)
        {
            sbuf_append_mem(out, q + 1, sl);
            return 1 + sl;
        }
        sbuf_append_mem(out, q, 2);
        return 2;
    }

    if ((c != MSL_CODE && c != MSL_DQ && c != MSL_ARITH) || (sl = mbt_sig_len(q, n - i)) == 0)
    {
        return 0;
    }

    char prev = (q > g_mbt_src) ? q[-1] : ' ';
    if (i + sl < n && q[sl] == '{')
    {
        if (isalnum((unsigned char) prev) || prev == '_')
        {
            return 0;
        }
        size_t j = inline_close(p, n, i + sl, d);
        if (j == 0)
        {
            return 0;
        }
        const char *b  = q + sl + 1;
        size_t      bn = j - (i + sl + 1);
        if (git_brace(b, bn))
        {
            return 0;
        }
        emit_inline(out, entity_ref_len(b) != bn, b, bn, c != MSL_DQ);
        return j + 1 - i;
    }

    return 0;
}

/** Copy `[@p, @p + @n)` at depth @d, expanding inline `@{...}` and `$(...)`. */
static void emit_src(const char *p, size_t n, int d, sbuf_t *out)
{
    size_t i = 0;
    while (i < n)
    {
        const char *q = p + i;
        if (mbt_dp(q) > d)
        {
            if (mbt_cx(q) == MSL_OPEN && mbt_dp(q) == d + 1)
            {
                size_t j = i;
                while (j < n && mbt_cx(p + j) == MSL_OPEN && mbt_dp(p + j) == d + 1)
                {
                    j++;
                }
                sbuf_append_mem(out, q, j - i);
                size_t k = j;
                while (k < n && !(mbt_cx(p + k) == MSL_CLOSE && mbt_dp(p + k) == d + 1))
                {
                    k++;
                }
                mbt_translate_code(p + j, k - j, d + 1, out);
                i = k;
                continue;
            }
            sbuf_append_ch(out, *q);
            i++;
            continue;
        }
        size_t used = try_inline(p, n, i, d, out);
        if (used > 0)
        {
            i += used;
            continue;
        }
        sbuf_append_ch(out, *q);
        i++;
    }
}

/** Return byte length of a balanced `@{...}` at `@p[@i]` in code at depth @d, or 0. */
static size_t inline_span(const char *p, size_t n, size_t i, int d)
{
    const char *q    = p + i;
    size_t      sl   = mbt_code_at(q, d) ? mbt_sig_len(q, n - i) : 0;
    char        prev = (q > g_mbt_src) ? q[-1] : ' ';
    if (sl == 0 || i + sl >= n || q[sl] != '{' || isalnum((unsigned char) prev) || prev == '_')
    {
        return 0;
    }
    size_t j = inline_close(p, n, i + sl, d);
    return (j > 0 && !git_brace(q + sl + 1, j - (i + sl + 1))) ? (j + 1 - i) : 0;
}

static const char *const flow_prefixes[] = { "then",  "do", "else", "elif", "while",
                                             "until", "if", "!",    "time", NULL };

/**
 * skip_flow_prefix - Skip one leading shell flow keyword (`then`, `do`, `if`, `!`, `time`, ...)
 * @s: Statement slice
 * @n: Length of @s in bytes
 *
 * Return: Number of bytes consumed (keyword plus trailing blanks), or 0 if none.
 */
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

/**
 * emit_assign_inline - Rewrite standalone `NAME=@{...}` into direct `milkget -v` / `milkcalc -v`
 * @s:   Statement slice
 * @n:   Length of @s in bytes
 * @d:   Command-substitution depth
 * @out: Output buffer
 *
 * Return: 1 if @s was emitted as a direct `-v` binding, 0 otherwise.
 */
static int emit_assign_inline(const char *s, size_t n, int d, sbuf_t *out)
{
    size_t i = 0;
    size_t k;
    while (i < n && mbt_code_at(s + i, d) && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    while (i < n && (k = skip_flow_prefix(s + i, n - i)) > 0)
    {
        i += k;
    }
    size_t e = n;
    while (e > i && mbt_code_at(s + e - 1, d) && (s[e - 1] == ' ' || s[e - 1] == '\t'))
    {
        e--;
    }
    size_t a = i;
    if (a >= e || !mbt_code_at(s + a, d) || !(isalpha((unsigned char) s[a]) || s[a] == '_'))
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
    if (v >= e || o >= e || s[a] != '=' || !mbt_code_at(s + a, d) ||
        (dq ? mbt_cx(s + o) != MSL_DQ : !mbt_code_at(s + o, d)))
    {
        return 0;
    }
    size_t sl = mbt_sig_len(s + o, e - o);
    size_t j  = (sl > 0 && o + sl < e && s[o + sl] == '{') ? inline_close(s, e, o + sl, d) : 0;
    if (j == 0 || j + 1 + dq != e || (dq && s[j + 1] != '"') ||
        git_brace(s + o + sl + 1, j - (o + sl + 1)))
    {
        return 0;
    }
    const char *b       = s + o + sl + 1;
    size_t      bn      = j - (o + sl + 1);
    int         is_calc = (entity_ref_len(b) != bn);

    sbuf_append_mem(out, s, i);
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

/**
 * translate_stmt - Translate a single top-level or nested shell statement
 * @stmt: Statement slice
 * @len:  Length of @stmt in bytes
 * @d:    Command-substitution depth
 * @out:  Output string buffer
 */
static void translate_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
{
    if (len == 0)
    {
        return;
    }
    if (!emit_assign_inline(stmt, len, d, out))
    {
        emit_src(stmt, len, d, out);
    }
}

/**
 * blank_or_flow_head - Check whether `@s[0..@n)` contains only blanks and leading flow keywords
 * @s: Statement prefix
 * @n: Length of @s in bytes
 *
 * Return: 1 if @s has only whitespace and flow prefixes (`then`, `if`, `do`, ...), 0 otherwise.
 */
static int blank_or_flow_head(const char *s, size_t n)
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
    return i == n;
}

/**
 * blank_or_fn_head - Check whether `@s[0..@n)` is blank, flow keywords, or a function header
 * @s: Statement prefix preceding `{`
 * @n: Length of @s in bytes
 *
 * Return: 1 if `{` starts a group or function body, 0 otherwise.
 */
static int blank_or_fn_head(const char *s, size_t n)
{
    if (blank_or_flow_head(s, n))
    {
        return 1;
    }
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
 * stmt_sep_len - Return byte length of a statement separator at `@p[@i]`, or 0
 * @p:  Shell code slice
 * @n:  Length of @p in bytes
 * @st: Start offset of the current statement within @p
 * @i:  Candidate separator offset within @p
 *
 * Return: Number of separator bytes (`1`..`3`), or 0 if `@p[@i]` is not a separator.
 */
static size_t stmt_sep_len(const char *p, size_t n, size_t st, size_t i)
{
    const char *q  = p + i;
    char        ch = *q;
    char        n1 = (i + 1 < n) ? q[1] : '\0';

    if (ch == '\n')
    {
        return 1;
    }
    if (ch == ';')
    {
        return (n1 == ';' || n1 == '&') ? ((n1 == ';' && i + 2 < n && q[2] == '&') ? 3 : 2) : 1;
    }
    if ((ch == '&' && n1 == '&') || (ch == '|' && n1 == '|') || (ch == '|' && n1 == '&'))
    {
        return 2;
    }
    if (ch == '|')
    {
        return 1;
    }
    if (ch == '&' && n1 != '>' && !(i > 0 && (q[-1] == '>' || q[-1] == '<')))
    {
        return 1;
    }
    if (ch == '(' && blank_or_flow_head(p + st, i - st))
    {
        return 1;
    }
    if (ch == '{' && (i + 1 >= n || isspace((unsigned char) n1)) &&
        blank_or_fn_head(p + st, i - st))
    {
        return 1;
    }
    return 0;
}

/**
 * mbt_translate_code - Split shell code `[@p, @p + @n)` at depth @d into statements and translate
 * @p:   Shell code slice
 * @n:   Length of @p in bytes
 * @d:   Command-substitution depth
 * @out: Output string buffer
 */
static void mbt_translate_code(const char *p, size_t n, int d, sbuf_t *out)
{
    size_t st = 0;
    size_t i  = 0;
    while (i < n)
    {
        const char *q = p + i;
        int         c = mbt_cx(q);
        if (mbt_dp(q) > d)
        {
            i++;
            continue;
        }
        if (c == MSL_COMMENT || c == MSL_HEREDOC)
        {
            translate_stmt(p + st, i - st, d, out);
            size_t j = i;
            while (j < n && mbt_cx(p + j) == c && mbt_dp(p + j) == d)
            {
                j++;
            }
            sbuf_append_mem(out, q, j - i);
            st = i = j;
            continue;
        }
        if (c != MSL_CODE)
        {
            i++;
            continue;
        }
        if (*q == '\\')
        {
            i += (i + 1 < n) ? 2 : 1;
            continue;
        }
        size_t sp = inline_span(p, n, i, d);
        if (sp > 0)
        {
            i += sp;
            continue;
        }
        size_t sep = stmt_sep_len(p, n, st, i);
        if (sep > 0)
        {
            translate_stmt(p + st, i - st, d, out);
            sbuf_append_mem(out, q, sep);
            i += sep;
            st = i;
            continue;
        }
        i++;
    }
    translate_stmt(p + st, n - st, d, out);
}

/**
 * milk_translate_shorthand_to_bash - Translate `@{...}` inline expansions into standard `milk.bash`
 * @src:          Input script or command line
 * @allow_legacy: Unused legacy compatibility flag (retained for ABI compatibility)
 *
 * Scans @src with `msl_scan()` and rewrites `@{...}` and `\@` constructs while
 * preserving 1:1 line numbering so bash `$LINENO` matches the source file.
 *
 * Return: Newly heap-allocated translated string (caller must `free()`), or NULL.
 */
char *milk_translate_shorthand_to_bash(const char *src, int allow_legacy)
{
    if (src == NULL)
    {
        return NULL;
    }
    size_t n = strlen(src);
    if (msl_scan(src, n, &g_mbt_map) != 0)
    {
        return NULL;
    }
    (void) allow_legacy;
    g_mbt_src = src;

    sbuf_t out;
    sbuf_init(&out);
    mbt_translate_code(src, n, 0, &out);

    msl_free(&g_mbt_map);
    g_mbt_src = NULL;
    return out.data;
}
