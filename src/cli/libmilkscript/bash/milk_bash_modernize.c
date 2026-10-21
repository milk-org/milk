// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_modernize.c
 * @brief Two-pass `--modernize` migration engine for legacy milk scripts.
 *
 * Role & Goal:
 *   Implements `milk_bash_modernize()` (invoked by `milk-script --modernize`),
 *   which rewrites legacy spellings in two passes:
 *   - Pass 1 (`modernize_stmt`): Rewrites removed command words (`@fpsset`,
 *     `@procctl`, `@usleep`, `@mloadas`, etc.), sticky `.tag value` defaults,
 *     and legacy entity namespace prefixes (`@stream.` -> `@s.`, `@fps.` -> `@f.`).
 *   - Pass 2 (`modernize_legacy_stmt`): Rewrites removed statement shorthands
 *     (`@ <expr>` -> `@calc "<expr>"`, `@s.x.y` -> `@get s.x.y`,
 *     `@f.x.p=v` -> `@set f.x.p=v`, `a |> b` -> `@pipe "a |> b"`, bare
 *     inline `@s.x.y` -> `@{s.x.y}`) and binds calc-assigned variables read
 *     later in bash (`@calc "T = e"` -> `@calc -v T "e"`).
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_bash_engine_internal.h"
#include "milk_catalog.h"
#include "milk_entity_ns.h"
#include "milk_lexicon.h"
#include "milk_renames.h"

/**
 * mbt_report_note - Report a diagnostic note for a legacy construct at @at that cannot be auto-rewritten
 * @at:   Pointer into `g_mbt_src` where the construct starts
 * @word: Offending token or construct label
 * @why:  Human-readable explanation of how to rewrite it manually
 *
 * Computes the 1-based line number of @at in `g_mbt_src`, prints a diagnostic
 * to `g_mbt_report` when non-NULL, and increments `g_mbt_nchanges`.
 */
void mbt_report_note(const char *at, const char *word, const char *why)
{
    int line = 1;
    for (const char *q = g_mbt_src; q < at; q++)
    {
        line += (*q == '\n');
    }
    if (g_mbt_report != NULL)
    {
        fprintf(g_mbt_report, "%s:%d: '%s' %s\n", g_mbt_fname, line, word, why);
    }
    g_mbt_nchanges++;
}

/**
 * report_change - Record an automatic modernization rewrite from @old to @new_
 * @at:   Pointer into `g_mbt_src` where @old began
 * @old:  Original source slice
 * @oldn: Byte length of @old
 * @new_: Replacement text
 *
 * Logs `fname:line: 'old' -> 'new'` to `g_mbt_report` when enabled and
 * increments the change counter `g_mbt_nchanges`.
 */
static void report_change(const char *at, const char *old, size_t oldn, const char *new_)
{
    int line = 1;
    for (const char *q = g_mbt_src; q < at; q++)
    {
        line += (*q == '\n');
    }
    if (g_mbt_report != NULL)
    {
        fprintf(g_mbt_report, "%s:%d: '%.*s' -> '%s'\n", g_mbt_fname, line, (int) oldn, old, new_);
    }
    g_mbt_nchanges++;
}

/**
 * is_one_quoted_word - Check whether `@p[0..@n)` is a single word composed solely of quoted spans
 * @p: Source slice
 * @n: Length of @p in bytes
 * @d: Command-substitution depth
 *
 * Prevents double-wrapping an expression that is already quoted when rewriting
 * `@ "expr"` into `@calc "expr"`.
 *
 * Return: 1 if @p is entirely quoted at depth @d, 0 otherwise.
 */
static int is_one_quoted_word(const char *p, size_t n, int d)
{
    if (n < 2 || (p[0] != '\'' && p[0] != '"'))
    {
        return 0;
    }
    for (size_t i = 0; i < n; i++)
    {
        if (mbt_code_at(p + i, d))
        {
            return 0;
        }
    }
    return 1;
}

/**
 * has_unquoted_space - Check whether `@p[0..@n)` contains unquoted whitespace at depth @d
 * @p: Source slice
 * @n: Length of @p in bytes
 * @d: Command-substitution depth
 *
 * Ignores whitespace inside `@{...}` spans so multi-token inline calculator
 * expressions are not mistaken for multi-word shell arguments.
 *
 * Return: 1 if unquoted whitespace is present outside `@{...}`, 0 otherwise.
 */
static int has_unquoted_space(const char *p, size_t n, int d)
{
    for (size_t i = 0; i < n; i++)
    {
        size_t sp = mbt_inline_span(p, n, i, d);
        if (sp > 0)
        {
            i += sp - 1;
            continue;
        }
        if (mbt_code_at(p + i, d) && isspace((unsigned char) p[i]))
        {
            return 1;
        }
    }
    return 0;
}

/**
 * has_unquoted_pipe_op - Check whether `@s[0..@len)` contains an unquoted `|>` operator at depth @d
 * @s:   Statement slice
 * @len: Length of @s in bytes
 * @d:   Command-substitution depth
 *
 * Return: 1 if `|>` appears in unquoted code outside `@{...}`, 0 otherwise.
 */
static int has_unquoted_pipe_op(const char *s, size_t len, int d)
{
    for (size_t i = 0; i + 1 < len; i++)
    {
        size_t sp = mbt_inline_span(s, len, i, d);
        if (sp > 0)
        {
            i += sp - 1;
            continue;
        }
        if (mbt_code_at(s + i, d) && s[i] == '|' && s[i + 1] == '>')
        {
            return 1;
        }
    }
    return 0;
}

/**
 * append_calc_expr - Emit `@s[0..@n)` as a double-quoted argument while keeping `\<newline>` outside quotes
 * @out: Output string buffer
 * @s:   Expression slice
 * @n:   Length of @s in bytes
 * @d:   Command-substitution depth
 *
 * Keeps line-continuation `\<newline>` sequences outside double quotes (`"\"` +
 * newline + `"`) so 1:1 line numbering is preserved across multi-line expressions.
 */
static void append_calc_expr(sbuf_t *out, const char *s, size_t n, int d)
{
    sbuf_append_ch(out, '"');
    for (size_t i = 0; i < n; i++)
    {
        if (s[i] == '\\' && i + 1 < n && s[i + 1] == '\n' && mbt_code_at(s + i, d))
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

/**
 * is_calc_or_test_stmt_prefix - Check whether statement @s starts with `@ `, `@calc`, `@test`, `@assert`, or `@waitfor`
 * @s:   Statement slice
 * @len: Length of @s in bytes
 *
 * Used during `--modernize` pass 2 so entity references inside calculator or
 * predicate expressions have their leading `@` stripped (`@s.x.y` -> `s.x.y`)
 * rather than being wrapped in `@{...}`.
 *
 * Return: 1 if @s is a calculator/predicate statement, 0 otherwise.
 */
static int is_calc_or_test_stmt_prefix(const char *s, size_t len)
{
    size_t i = 0;
    while (i < len && (s[i] == ' ' || s[i] == '\t'))
    {
        i++;
    }
    size_t k;
    while (i < len && (k = mbt_skip_flow_prefix(s + i, len - i)) > 0)
    {
        i += k;
    }
    size_t adv = (i < len) ? mbt_sig_len(s + i, len - i) : 0;
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

/**
 * is_entity_taking_cmd - Check whether command word @cmd accepts bare entity specifications
 * @cmd: Command word (with or without leading `@`)
 *
 * Commands that accept entity paths directly (`@get`, `@set`, `@ls`, `@waitfor`,
 * etc.) preserve bare `s.im.cnt0` arguments literally instead of wrapping them
 * in `@{...}` during modernization.
 *
 * Return: 1 if @cmd takes entity path arguments, 0 otherwise.
 */
static int is_entity_taking_cmd(const char *cmd)
{
    cmd += mbt_sig_len(cmd, strlen(cmd));
    return (strcmp(cmd, "ls") == 0 || strcmp(cmd, "get") == 0 || strcmp(cmd, "set") == 0 ||
            strcmp(cmd, "exists") == 0 || strcmp(cmd, "running") == 0 ||
            strcmp(cmd, "waitfor") == 0 || strcmp(cmd, "waitany") == 0 || strcmp(cmd, "on") == 0 ||
            strcmp(cmd, "snapshot") == 0 || strcmp(cmd, "diff") == 0 ||
            strcmp(cmd, "restore") == 0 || strcmp(cmd, "assert") == 0 ||
            strcmp(cmd, "config") == 0 || strcmp(cmd, "grab") == 0 || strcmp(cmd, "replay") == 0);
}

/**
 * append_args - Emit command argument words while quoting unquoted globs/slices for milk commands
 * @args:                Argument slice following the command word
 * @len:                 Length of @args in bytes
 * @is_milk_cmd:         Non-zero if the command is a milk keyword or module command
 * @preserve_entity_arg: Non-zero if bare entity paths should not be wrapped in `@{...}`
 * @d:                   Command-substitution depth
 * @out:                 Output string buffer
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
        if (mbt_code_at(args + i, d) && isspace((unsigned char) args[i]))
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
            if (!mbt_code_at(q, d))
            {
                has_quotes = 1;
                i++;
                continue;
            }
            if (br_depth == 0 && isspace((unsigned char) *q))
            {
                break;
            }
            size_t sp = mbt_inline_span(args, len, i, d);
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
            mbt_emit_src(args + tstart, tlen, d, !preserve_entity_arg, out);
        }
    }
}

/**
 * try_translate_module_cmd - Modernize arguments of a qualified `module.command` invocation
 * @firstword: NUL-terminated first word of the statement
 * @s:         Statement slice starting at @firstword
 * @rem_len:   Remaining byte length of @s
 * @wlen:      Byte length of @firstword
 * @d:         Command-substitution depth
 * @out:       Output string buffer
 *
 * Quotes unquoted globs and image slices passed to catalog module commands.
 *
 * Return: 1 if @firstword is a qualified milk command and was emitted, 0 otherwise.
 */
static int try_translate_module_cmd(const char *firstword,
                                    const char *s,
                                    size_t      rem_len,
                                    size_t      wlen,
                                    int         d,
                                    sbuf_t     *out)
{
    if (!mbt_is_qualified_milk_command(firstword))
    {
        return 0;
    }
    sbuf_append_str(out, firstword);
    append_args(s + wlen, rem_len - wlen, 1, 0, d, out);
    return 1;
}

/**
 * is_assignment_word - Check whether `@s[0..@n)` is a valid shell `NAME=value` prefix word
 * @s: Word slice
 * @n: Length of @s in bytes
 *
 * Return: 1 if @s begins with a valid shell identifier followed by `=`, 0 otherwise.
 */
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
 * modernize_legacy_stmt - Pass 2 statement modernizer for removed shorthands and inline entities
 * @stmt: Statement slice
 * @len:  Length of @stmt in bytes
 * @d:    Command-substitution depth
 * @out:  Output string buffer
 *
 * Rewrites removed statement forms (`@ expr` -> `@calc "expr"`, `@s.x.y` ->
 * `@get s.x.y`, `@f.x.p=v` -> `@set f.x.p=v`, `a |> b` -> `@pipe "a |> b"`,
 * `source f.milk` -> `@source f.milk`) and wraps bare inline `@s.x.y` entity
 * references into `@{s.x.y}`.
 */
static void modernize_legacy_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
{
    const char *sigil  = "@";
    size_t      siglen = 1;

    size_t i = 0;
    while (i < len && (stmt[i] == ' ' || stmt[i] == '\t'))
    {
        sbuf_append_ch(out, stmt[i]);
        i++;
    }
    /* leading flow keywords, `time`, and NAME=value prefixes do not change the command word */
    for (;;)
    {
        size_t k = mbt_skip_flow_prefix(stmt + i, len - i);
        if (k == 0 && g_mbt_allow_legacy)
        {
            size_t w = mbt_word_len(stmt + i, len - i, d);
            if (w < len - i && is_assignment_word(stmt + i, w))
            {
                mbt_emit_src(stmt + i, w, d, 1, out);
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
    if (slen > siglen && mbt_sig_len(s, slen) && (s[siglen] == ' ' || s[siglen] == '\t'))
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
            size_t ve = vs + mbt_word_len(s + vs, slen - vs, d);
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
    if (slen > siglen && mbt_sig_len(s, slen))
    {
        size_t adv    = siglen;
        int    has_eq = 0;
        size_t elen   = mbt_scan_entity_path(s + adv, &has_eq);
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
                    mbt_report_note(stmt + i, "@<entity>=<words>",
                                    "has an unquoted multi-word value: quote it");
                    sbuf_append_mem(out, stmt + i, len - i);
                    return;
                }
                sbuf_append_str(out, sigil);
                sbuf_append_str(out, "set ");
                append_args(s + adv, elen, 1, 1, d, out);
                sbuf_append_ch(out, '=');
                mbt_emit_src(s + vstart, slen - vstart, d, 1, out);
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

    size_t wlen = mbt_word_len(s, slen, d);
    char   firstword[256];
    if (wlen >= sizeof(firstword))
    {
        mbt_emit_src(s, len - i, d, 1, out);
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
        size_t fl = mbt_word_len(s + a, slen - a, d);
        if (fl > 5 && strncmp(s + a + fl - 5, ".milk", 5) == 0)
        {
            sbuf_append_str(out, "@source");
            mbt_emit_src(s + wlen, len - (i + wlen), d, 1, out);
            return;
        }
    }

    /* 3. Already sigiled '@<cmd> ...': the command word itself is never expanded */
    if (mbt_sig_len(firstword, wlen) > 0)
    {
        sbuf_append_mem(out, s, wlen);
        append_args(s + wlen, len - (i + wlen), 1, is_entity_taking_cmd(firstword), d, out);
        return;
    }

    /* 4. Module command (`mem.listim`) */
    if (g_mbt_allow_legacy && try_translate_module_cmd(firstword, s, len - i, wlen, d, out))
    {
        return;
    }

    /* Default: standard bash command */
    mbt_emit_src(s, wlen, d, 1, out);
    append_args(s + wlen, len - (i + wlen), 0, 0, d, out);
}

/**
 * is_module_cmd_word - Check whether @w has the shape `[@]module.cmd` (single dot, lowercase prefix)
 * @w: NUL-terminated word to inspect
 *
 * Return: 1 if @w matches `[@]module.cmd`, 0 otherwise.
 */
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

/**
 * is_tag_word - Check whether `@p[0..@n)` is a legacy parameter tag `.name` without `=`
 * @p: Word slice
 * @n: Length of @p in bytes
 *
 * Return: 1 if @p is a `.tag` word, 0 otherwise.
 */
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

/**
 * plain_word - Check whether `@p[0..@n)` is a literal shell word free of quotes, expansions, or globs
 * @p: Word slice
 * @n: Length of @p in bytes
 *
 * Return: 1 if @p contains no shell metacharacters, 0 otherwise.
 */
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
 * modernize_args - Rewrite legacy commands whose arguments rearrange (`MILK_REN_ARGS`)
 * @old:  Removed command name (`@usleep`, `@mloadas`, `@fpsset`, `@procctl`, `@procwait`, `@procstat`)
 * @stmt: Statement slice
 * @a:    Offset in @stmt where arguments start
 * @len:  Total length of @stmt
 * @d:    Command-substitution depth
 * @out:  Output string buffer
 *
 * Return: 1 if the statement was rewritten into @out, 0 if arguments are non-literal.
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
        size_t wl = mbt_word_len(stmt + a, end - a, d);
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
 * tag_value_args - Check whether `@stmt[@i..@len)` has the legacy sticky-default form `.tag value`
 * @stmt: Statement slice
 * @i:    Offset after the command word
 * @len:  Total length of @stmt
 * @d:    Command-substitution depth
 * @tag:  Output start offset of `.tag`
 * @val:  Output start offset of `value`
 * @vend: Output end offset of `value`
 *
 * Return: 1 if the argument list is a `.tag value` pair, 0 otherwise.
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
    size_t w1 = mbt_word_len(stmt + a, len - a, d);
    size_t b  = a + w1;
    while (b < len && (stmt[b] == ' ' || stmt[b] == '\t'))
    {
        b++;
    }
    size_t w2 = (b > a + w1) ? mbt_word_len(stmt + b, len - b, d) : 0;
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
 * modernize_stmt - Pass 1 statement modernizer for removed command words and legacy namespace prefixes
 * @stmt: Statement slice
 * @len:  Length of @stmt in bytes
 * @d:    Command-substitution depth
 * @out:  Output string buffer
 *
 * Rewrites removed command names from `milk_renames.h`, converts `module.cmd .tag val`
 * into `module.cmd --default .tag=val`, and renames long entity namespace prefixes
 * (`@stream.` -> `@s.`, `@fps.` -> `@f.`, `@proc.` -> `@p.`).
 */
static void modernize_stmt(const char *stmt, size_t len, int d, sbuf_t *out)
{
    size_t i = 0;
    while (i < len && (stmt[i] == ' ' || stmt[i] == '\t'))
    {
        sbuf_append_ch(out, stmt[i++]);
    }
    for (size_t k; i < len && (k = mbt_skip_flow_prefix(stmt + i, len - i)) > 0; i += k)
    {
        sbuf_append_mem(out, stmt + i, k);
    }
    char   word[256] = "";
    size_t wstart    = i;
    size_t w         = (i < len) ? mbt_word_len(stmt + i, len - i, d) : 0;
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
            mbt_report_note(stmt + i, word, why);
            nw = NULL;
        }
        else if (nw == NULL && word[0] == '@' && word[1] != '\0' &&
                 milk_lexicon_keyword(word + 1) == NULL)
        {
            nw = (strchr(word + 1, '.') == NULL)           ? milk_catalog_short(word + 1)
                 : mbt_is_qualified_milk_command(word + 1) ? word + 1
                                                           : NULL;
            if (nw != NULL && strcmp(nw, "?") == 0)
            {
                mbt_report_note(stmt + i, word, "names several module commands: qualify it");
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
        size_t tl = mbt_word_len(stmt + tg, len - tg, d);
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
        if (mbt_dp(stmt + i) == d + 1 &&
            mbt_cx(stmt + i) == MSL_OPEN) /* $(...), `...`: modernize the body */
        {
            size_t j = i;
            while (j < len && mbt_cx(stmt + j) == MSL_OPEN && mbt_dp(stmt + j) == d + 1)
            {
                j++;
            }
            size_t k = j;
            while (k < len && !(mbt_cx(stmt + k) == MSL_CLOSE && mbt_dp(stmt + k) == d + 1))
            {
                k++;
            }
            sbuf_append_mem(out, stmt + i, j - i);
            mbt_translate_code(stmt + j, k - j, d + 1, out);
            i = k;
            continue;
        }
        const char *p      = stmt + i;
        int         c      = mbt_cx(p);
        size_t      at     = (*p == '@') ? 1 : 0;
        size_t      oldlen = 0;
        const char *nn     = NULL;
        int         bound  = (p == g_mbt_src) || !(isalnum((unsigned char) p[-1]) || p[-1] == '_' ||
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

/**
 * reads_var - Check whether `g_mbt_src` expands shell variable `@name[0..@n)` via `$name` or `${name...}`
 * @name: Variable identifier slice
 * @n:    Length of @name in bytes
 *
 * Used by `modernize_calc_bind()` to detect whether a variable assigned inside
 * `@calc "T = expr"` is subsequently read by bash and therefore requires `-v T`.
 *
 * Return: 1 if `$name` or `${name...}` appears in `g_mbt_src`, 0 otherwise.
 */
static int reads_var(const char *name, size_t n)
{
    for (const char *q = strchr(g_mbt_src, '$'); q != NULL; q = strchr(q + 1, '$'))
    {
        const char *v = q + 1 + (q[1] == '{');
        if (strncmp(v, name, n) == 0 && !isalnum((unsigned char) v[n]) && v[n] != '_')
        {
            return 1;
        }
    }
    return 0;
}

/**
 * calc_target - Parse a top-level variable assignment `T = e` or `T op= e` inside a calc expression
 * @s:   Calculator expression slice
 * @n:   Length of @s in bytes
 * @t0:  Output start offset of variable name `T`
 * @op:  Output assignment operator (`'='`, `'+'`, `'-'`, `'*'`, or `'/'`)
 * @rhs: Output start offset of the right-hand side expression `e`
 *
 * Return: Byte length of target identifier `T`, or 0 if @s is not an assignment.
 */
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
 * modernize_calc_bind - Convert `@calc "T = e"` into `@calc -v T "e"` when bash reads `$T`
 * @stmt: Original statement pointer in `g_mbt_src` (for line-number reporting)
 * @out:  Output buffer containing the modernized statement starting at @at
 * @at:   Offset in @out where the current statement starts
 *
 * Since `@calc` no longer exports all calculator variables back into the shell
 * environment automatically, this helper rewrites single assignments whose LHS
 * variable is read as `$T` in the script to use explicit `-v T` binding.
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
    while (i < n && (k = mbt_skip_flow_prefix(s + i, n - i)) > 0)
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
            mbt_report_note(stmt, word, "is set in calc only now: bind it with @calc -v");
        }
        b = sl + 1;
    }
    free(s);
}

/**
 * mbt_modernize_stmt_dispatch - Dispatch statement @stmt to pass 1 or pass 2 of `--modernize`
 * @stmt: Statement slice
 * @len:  Length of @stmt in bytes
 * @d:    Command-substitution depth
 * @out:  Output string buffer
 */
void mbt_modernize_stmt_dispatch(const char *stmt, size_t len, int d, sbuf_t *out)
{
    if (g_mbt_modernize == 1)
    {
        modernize_stmt(stmt, len, d, out);
        return;
    }
    if (g_mbt_modernize == 2)
    {
        size_t at         = out->len;
        int    saved      = g_mbt_skip_dq_ent;
        g_mbt_skip_dq_ent = is_calc_or_test_stmt_prefix(stmt, len);
        modernize_legacy_stmt(stmt, len, d, out);
        if (g_mbt_skip_dq_ent) /* calc text names entities without @: "@s.x.y * 2" -> "s.x.y * 2" */
        {
            size_t w = at;
            for (size_t r = at; r < out->len; r++)
            {
                char pc = (r > at) ? out->data[r - 1] : ' ';
                if (!(out->data[r] == '@' && mbt_entity_ref_len(out->data + r + 1) > 0 &&
                      !isalnum((unsigned char) pc) && pc != '_' && pc != '.' && pc != '$'))
                {
                    out->data[w++] = out->data[r];
                }
            }
            out->len     = w;
            out->data[w] = '\0';
        }
        g_mbt_skip_dq_ent = saved;
        modernize_calc_bind(stmt, out, at);
        if (out->len - at != len || memcmp(out->data + at, stmt, len) != 0)
        {
            char *nw = strndup(out->data + at, out->len - at);
            report_change(stmt, stmt, len, (nw != NULL) ? nw : "");
            free(nw);
        }
    }
}

/**
 * modernize_pass - Execute one modernization pass (@pass = 1 or 2) over script text @src
 * @src:      Input script text
 * @pass:     Pass number (`1` for command/namespace renames, `2` for legacy statement forms)
 * @fname:    Source filename for diagnostic reporting
 * @report:   Optional diagnostic stream
 * @nchanges: In/out accumulator of applied rewrites and notes
 *
 * Return: Newly heap-allocated modernized script string, or NULL on error.
 */
static char *modernize_pass(const char *src,
                            int         pass,
                            const char *fname,
                            FILE       *report,
                            int        *nchanges)
{
    size_t n = strlen(src);
    if (msl_scan(src, n, &g_mbt_map) != 0)
    {
        return NULL;
    }
    g_mbt_src          = src;
    g_mbt_allow_legacy = 1;
    g_mbt_skip_dq_ent  = 0;
    g_mbt_modernize    = pass;
    g_mbt_report       = report;
    g_mbt_fname        = (fname != NULL) ? fname : "-";
    g_mbt_nchanges     = 0;

    sbuf_t out;
    sbuf_init(&out);
    mbt_translate_code(src, n, 0, &out);

    g_mbt_modernize    = 0;
    g_mbt_allow_legacy = 0;
    msl_free(&g_mbt_map);
    g_mbt_src = NULL;
    *nchanges += g_mbt_nchanges;
    return out.data;
}

/**
 * milk_bash_modernize - Run both `--modernize` passes over script text @src
 * @src:      Input script text
 * @fname:    Filename used in diagnostic messages (or NULL for `"-"`)
 * @report:   Optional `FILE *` receiving `fname:line: 'old' -> 'new'` lines
 * @nchanges: Optional output count of applied changes and notes
 *
 * Return: Newly heap-allocated modernized script text (caller must `free()`), or NULL.
 */
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
