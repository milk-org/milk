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
 *   - Inline `@{<expr>}` -> `${| milkcalc -v REPLY "<expr>" ...; }` (or `$(milkcalc ...)` on bash < 5.3)
 *   - Inline `@{<entity>}` -> `${| milkget -v REPLY <entity> ...; }` (or `$(milkget ...)` on bash < 5.3)
 *   - Escaped `\@{...}` inside double quotes -> literal `@{...}`
 *   Also provides the shared lexical statement walker used by `milk_bash_modernize()`.
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
#include "milk_shell_lex.h"

/**
 * milk_bash_funsub_disabled - Check whether bash 5.3+ `${| ...; }` function substitution is unavailable
 *
 * Probes `/bin/bash` version once and caches the result in `MILK_BASH_NO_FUNSUB`
 * so child `milk-call translate` invocations inherit it without re-spawning bash.
 *
 * Return: 1 if `${| ...; }` must not be emitted, 0 if supported.
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

/* ============================================================
 *  Entity & Command Recognition Helpers
 * ============================================================ */

/**
 * is_entity_prefix - Check whether @s begins with a canonical entity namespace prefix
 * @s:       Input string to inspect
 * @out_len: Optional output receiving the prefix length (including trailing `.`)
 *
 * Verifies that @s starts with `s.`, `f.`, `p.`, `q.`, or `l.` followed by a
 * valid entity name start character or shell expansion.
 *
 * Return: 1 if @s starts with an entity namespace prefix, 0 otherwise.
 */
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

/**
 * mbt_scan_entity_path - Scan a dotted entity property path starting at @s
 * @s:      Start of candidate entity path (without leading `@`)
 * @has_eq: Set to 1 if the scanned path is immediately followed by `=` (not `==`)
 *
 * Recognizes `<ns>.<name>.<field>` paths (including embedded `$var` or `${var}`
 * segments) and requires at least two dots so module commands (`mem.listim`) are
 * never mistaken for entity property references.
 *
 * Return: Byte length of the entity path, or 0 if @s is not a valid entity path.
 */
size_t mbt_scan_entity_path(const char *s, int *has_eq)
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

/**
 * mbt_is_qualified_milk_command - Check whether @w is a catalogued `module.command` name
 * @w: Candidate command word (without leading `@`)
 *
 * Return: 1 if @w is present in the installed module command catalog, 0 otherwise.
 */
int mbt_is_qualified_milk_command(const char *w)
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

const char *g_mbt_src = NULL; /* text being translated */
msl_map_t   g_mbt_map;        /* context of every byte of g_mbt_src */
int         g_mbt_allow_legacy = 0;
int         g_mbt_skip_dq_ent  = 0; /* current statement is @calc/@test/...: calc reads "@s.x.y" */
int         g_mbt_modernize    = 0; /* mbt_translate_code() rewrites removed spellings instead */
FILE       *g_mbt_report       = NULL; /* where the modernize pass reports changes */
const char *g_mbt_fname        = "-";
int         g_mbt_nchanges     = 0;

/**
 * append_inline_label - Append single-quoted `'@{body}'` error label to @out
 * @out:  Output buffer
 * @body: Raw text inside `@{...}`
 * @n:    Length of @body in bytes
 *
 * Formats the diagnostic label passed to `__milk_inline_err` when an inline
 * `milkget` expansion fails at runtime.
 */
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
 * emit_inline - Emit a `milkget` or `milkcalc` substitution for an inline `@{...}` construct
 * @out:     Output string buffer
 * @is_calc: Non-zero for calculator expressions, 0 for single entity property reads
 * @body:    Content inside `@{...}`
 * @n:       Length of @body in bytes
 * @dq:      Non-zero to wrap the expansion in `"..."` to prevent word-splitting/globbing
 *
 * Emits zero-fork `${| milkcalc -v REPLY ...; }` on bash >= 5.3 (or `$(milkcalc ...)`
 * on older bash), or rewrites legacy inline forms into `@{...}` during `--modernize`.
 */
static void emit_inline(sbuf_t *out, int is_calc, const char *body, size_t n, int dq)
{
    if (g_mbt_modernize) /* the removed inline forms (@s.x.y, ${@s.x.y}) become @{...} */
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

/**
 * mbt_entity_ref_len - Return length of a bare `s.a.b` entity reference at @p, or 0
 * @p: Candidate string (without leading `@`)
 *
 * Requires at least two dots and no trailing `=` so `@{s.im.xsize}` dispatches
 * to `milkget` while `@{s.im.xsize * 2}` or `@{f.fps.p=1}` dispatches to `milkcalc`.
 *
 * Return: Byte length of the read-only entity reference, or 0.
 */
size_t mbt_entity_ref_len(const char *p)
{
    int    has_eq = 0;
    size_t elen   = mbt_scan_entity_path(p, &has_eq);
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
 * inline_close - Find the matching `}` for the `{` at `@p[@i]` at lexical depth @d
 * @p: Source slice
 * @n: Length of @p in bytes
 * @i: Offset of the opening `{`
 * @d: Command-substitution depth of the opening `{`
 *
 * Counts nested braces only when they appear in the same lexical context and
 * substitution depth as the opening `{`.
 *
 * Return: Offset of the closing `}` within @p, or 0 if unclosed.
 */
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

/**
 * try_inline - Expand an inline construct (`\@`, `@{expr}`, or legacy `@s.x.y`) at `@p[@i]`
 * @p:   Source slice
 * @n:   Length of @p in bytes
 * @i:   Current byte offset in @p
 * @d:   Current command-substitution depth
 * @ent: Non-zero when modernizing legacy bare `@s.x.y` entity references
 * @out: Output buffer receiving expanded text
 *
 * Return: Number of source bytes consumed from `@p + @i`, or 0 if no inline construct starts at @i.
 */
static size_t try_inline(const char *p, size_t n, size_t i, int d, int ent, sbuf_t *out)
{
    const char *q  = p + i;
    int         c  = mbt_cx(q);
    size_t      sl = 0;

    if ((c == MSL_CODE || c == MSL_DQ) && q[0] == '\\' && i + 1 < n)
    {
        sl        = mbt_sig_len(q + 1, n - i - 1);
        int brace = (sl > 0 && i + 1 + sl < n && q[1 + sl] == '{');
        if (c == MSL_DQ && sl > 0 && (brace ? !g_mbt_modernize : g_mbt_modernize))
        {
            /* "\@{..}" is a literal @{..}; modernizing, the old "\@s.x" escape is no longer needed */
            sbuf_append_mem(out, q + 1, sl);
            return 1 + sl;
        }
        sbuf_append_mem(out, q, 2); /* escaped character: never the start of a construct */
        return 2;
    }

    if (g_mbt_modernize && c == MSL_PARAM && q[0] == '$' && i + 2 < n && q[1] == '{' &&
        (sl = mbt_sig_len(q + 2, n - i - 2)) > 0)
    {
        size_t elen = mbt_entity_ref_len(q + 2 + sl);
        if (elen > 0 && i + 2 + sl + elen < n && q[2 + sl + elen] == '}') /* ${@s.x.y} */
        {
            const char *k = q - 1;
            while (k >= g_mbt_src && mbt_cx(k) == MSL_PARAM)
            {
                k--;
            }
            emit_inline(out, 0, q + 2 + sl, elen, !(k >= g_mbt_src && mbt_cx(k) == MSL_DQ));
            return 2 + sl + elen + 1;
        }
        return 0;
    }

    if ((c != MSL_CODE && c != MSL_DQ && c != MSL_ARITH) || (sl = mbt_sig_len(q, n - i)) == 0)
    {
        return 0;
    }

    char prev = (q > g_mbt_src) ? q[-1] : ' ';
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
        if (g_mbt_modernize) /* already the modern form */
        {
            sbuf_append_mem(out, q, j + 1 - i);
            return j + 1 - i;
        }
        emit_inline(out, mbt_entity_ref_len(b) != bn, b, bn, c != MSL_DQ); /* @{s.x.y}: milkget */
        return j + 1 - i;
    }

    if (!ent || !g_mbt_modernize || c == MSL_ARITH || (c == MSL_DQ && g_mbt_skip_dq_ent) ||
        isalnum((unsigned char) prev) || prev == '_' || prev == '.' || prev == '$')
    {
        return 0;
    }
    size_t elen = mbt_entity_ref_len(q + sl);
    if (elen == 0 || i + sl + elen > n)
    {
        return 0;
    }
    emit_inline(out, 0, q + sl, elen, c == MSL_CODE);
    return sl + elen;
}

/**
 * mbt_emit_src - Copy source bytes `[@p, @p + @n)` at depth @d, expanding inline `@{...}` and `$(...)`
 * @p:   Source slice
 * @n:   Length of @p in bytes
 * @d:   Current command-substitution depth
 * @ent: Non-zero to also rewrite legacy bare `@s.x.y` references during `--modernize`
 * @out: Output buffer
 *
 * Recursively translates nested `$(...)`, `` `...` ``, and `${| ...; }` command
 * substitutions via `mbt_translate_code()`.
 */
void mbt_emit_src(const char *p, size_t n, int d, int ent, sbuf_t *out)
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
 * mbt_inline_span - Return byte length of a balanced `@{...}` construct at `@p[@i]` in code at depth @d
 * @p: Source slice
 * @n: Length of @p in bytes
 * @i: Offset to test
 * @d: Expected command-substitution depth
 *
 * Treats `@{...}` as an atomic span so statement splitters and word tokenizers
 * ignore spaces, `&&`, `|`, and `;` inside calculator expressions.
 *
 * Return: Span length in bytes, or 0 if `@p[@i]` is not a balanced `@{...}`.
 */
size_t mbt_inline_span(const char *p, size_t n, size_t i, int d)
{
    const char *q    = p + i;
    size_t      sl   = mbt_code_at(q, d) ? mbt_sig_len(q, n - i) : 0;
    char        prev = (q > g_mbt_src) ? q[-1] : ' ';
    if (sl == 0 || i + sl >= n || q[sl] != '{' || isalnum((unsigned char) prev) || prev == '_')
    {
        return 0;
    }
    int depth = 1;
    for (size_t j = i + sl + 1; j < n; j++)
    {
        if (mbt_code_at(p + j, d))
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

/**
 * mbt_word_len - Return length of the shell word starting at @p at depth @d
 * @p: Start of word
 * @n: Remaining byte count
 * @d: Command-substitution depth
 *
 * Advances until unquoted whitespace at depth @d, keeping `@{...}` spans and
 * backslash-escaped characters intact.
 *
 * Return: Word length in bytes.
 */
size_t mbt_word_len(const char *p, size_t n, int d)
{
    size_t i = 0;
    while (i < n && !(mbt_code_at(p + i, d) && isspace((unsigned char) p[i])))
    {
        size_t sp = mbt_inline_span(p, n, i, d);
        i += (sp > 0) ? sp : ((mbt_code_at(p + i, d) && p[i] == '\\' && i + 1 < n) ? 2 : 1);
    }
    return i;
}

static const char *const flow_prefixes[] = { "then",  "do", "else", "elif", "while",
                                             "until", "if", "!",    "time", NULL };

/**
 * mbt_skip_flow_prefix - Skip one leading shell flow keyword (`then`, `do`, `if`, `!`, `time`, ...)
 * @s: Statement slice
 * @n: Length of @s in bytes
 *
 * Return: Number of bytes consumed (keyword plus trailing blanks), or 0 if none.
 */
size_t mbt_skip_flow_prefix(const char *s, size_t n)
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
 * mbt_is_calc_shorthand - Check whether statement @s starts with legacy `@ <expr>` shorthand
 * @s: Statement slice
 * @n: Length of @s in bytes
 *
 * Used by `mbt_translate_code()` during `--modernize` so `&&` and `||` inside a
 * legacy `@ <expr>` statement are kept inside the calculator expression.
 *
 * Return: 1 if @s begins with `@ ` after optional flow keywords, 0 otherwise.
 */
int mbt_is_calc_shorthand(const char *s, size_t n)
{
    size_t i = 0;
    while (i < n && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    size_t k;
    while (i < n && (k = mbt_skip_flow_prefix(s + i, n - i)) > 0)
    {
        i += k;
    }
    size_t sl = mbt_sig_len(s + i, n - i);
    return sl > 0 && i + sl < n && (s[i + sl] == ' ' || s[i + sl] == '\t');
}

/**
 * emit_assign_inline - Rewrite standalone `NAME=@{...}` into direct `milkget -v` / `milkcalc -v`
 * @s:   Statement slice
 * @n:   Length of @s in bytes
 * @d:   Command-substitution depth
 * @out: Output buffer
 *
 * Avoids command-substitution overhead on standalone `NAME=@{...}` assignments by
 * binding `NAME` directly via `milkcalc -v NAME` or `milkget -v NAME` inside `{ ...; }`.
 *
 * Return: 1 if @s was emitted as a direct `-v` binding, 0 otherwise.
 */
static int emit_assign_inline(const char *s, size_t n, int d, sbuf_t *out)
{
    size_t i = 0;
    size_t k;
    while (i < n && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    while (i < n && (k = mbt_skip_flow_prefix(s + i, n - i)) > 0)
    {
        i += k;
    }
    size_t e = n;
    while (e > i && (s[e - 1] == ' ' || s[e - 1] == '\t'))
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
    if (v >= e || s[a] != '=' || !mbt_code_at(s + a, d) ||
        (dq ? mbt_cx(s + o) != MSL_DQ : !mbt_code_at(s + o, d)))
    {
        return 0;
    }
    size_t sl = mbt_sig_len(s + o, e - o);
    size_t j  = (sl > 0 && o + sl < e && s[o + sl] == '{') ? inline_close(s, e, o + sl, d) : 0;
    if (j == 0 || j + 1 + dq != e || (dq && s[j + 1] != '"'))
    {
        return 0;
    }
    const char *b       = s + o + sl + 1;
    size_t      bn      = j - (o + sl + 1);
    int         is_calc = (mbt_entity_ref_len(b) != bn);

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

/**
 * translate_stmt - Translate or modernize a single top-level or nested shell statement
 * @stmt: Statement slice
 * @len:  Length of @stmt in bytes
 * @d:    Command-substitution depth
 * @out:  Output string buffer
 *
 * Dispatches to `mbt_modernize_stmt_dispatch()` during `milk_bash_modernize()`
 * passes, or translates `@{...}` constructs in normal runtime mode.
 */
static void translate_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
{
    if (len == 0)
    {
        return;
    }
    if (g_mbt_modernize != 0)
    {
        mbt_modernize_stmt_dispatch(stmt, len, d, out);
        return;
    }
    if (!emit_assign_inline(stmt, len, d, out))
    {
        mbt_emit_src(stmt, len, d, 0, out); /* only @{...} (and \@) is milk syntax */
    }
}

/**
 * blank_or_fn_head - Check whether `@s[0..@n)` is blank or a shell function header (`foo()`, `function f`)
 * @s: Statement prefix preceding `{`
 * @n: Length of @s in bytes
 *
 * Used by `mbt_translate_code()` to recognize `{` as a compound-command separator
 * when it opens a brace group or shell function body.
 *
 * Return: 1 if `{` starts a group or function body, 0 otherwise.
 */
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
 * mbt_translate_code - Split shell code `[@p, @p + @n)` at depth @d into statements and translate each
 * @p:   Shell code slice
 * @n:   Length of @p in bytes
 * @d:   Command-substitution depth
 * @out: Output string buffer
 *
 * Preserves comments and here-document bodies verbatim and splits at unquoted
 * shell statement separators (`\n`, `;`, `&&`, `||`, `|`, `&`, `(`, `{`).
 */
void mbt_translate_code(const char *p, size_t n, int d, sbuf_t *out)
{
    size_t st   = 0;
    size_t i    = 0;
    int    calc = -1; /* current statement is `@ <expr>` (only ; and newline end it) */
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
        size_t sp = mbt_inline_span(p, n, i, d);
        if (sp > 0)
        {
            if (calc < 0)
            {
                calc = mbt_is_calc_shorthand(p + st, n - st);
            }
            i += sp;
            continue;
        }
        if (calc < 0 && !isspace((unsigned char) *q))
        {
            calc = mbt_is_calc_shorthand(p + st, n - st);
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
    (void) allow_legacy; /* the removed forms are no longer translated (milk-script --modernize) */
    g_mbt_src          = src;
    g_mbt_allow_legacy = 0;
    g_mbt_skip_dq_ent  = 0;

    sbuf_t out;
    sbuf_init(&out);
    mbt_translate_code(src, n, 0, &out);

    msl_free(&g_mbt_map);
    g_mbt_src = NULL;
    return out.data;
}
