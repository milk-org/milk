// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_eval.c
 * @brief Calculator (`@calc`/`@test`), assertion (`@assert`), and batch pipeline (`@pipe`) evaluation.
 *
 * Role & Goal:
 *   Evaluates `@calc` / `@test` expressions (`milk_server_exec_calc_or_test`),
 *   diagnostic entity/boolean assertions (`milk_server_assert`), and multi-step
 *   `@pipe` chains with `_` output-image forwarding (`milk_server_batch_or_pipe`).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_memory.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_parser.h"
#include "cli_calc_prog.h"
#include "milk_cmd_index.h"
#include "milk_entity_ns.h"
#include "milk_proto.h"
#include "milk_server_internal.h"
#include "milk_shell_lex.h"
#include "milk_shm_prop.h"

extern int cli_float_digits;

/**
 * eval_scalar_subexpr - Evaluate a scalar sub-expression for `@assert` failure diagnostics
 * @subexpr: Sub-expression text (LHS or RHS of a failed relational comparison)
 * @out:     Output buffer receiving the formatted scalar value
 * @outsz:   Capacity of @out in bytes
 *
 * Expands any entity references in @subexpr, evaluates it quietly into a temporary
 * variable (`_milk_asrt_tmp`), and formats the resulting integer, string, or float
 * so `@assert` can report `(actual: LHS vs RHS)` on failure.
 *
 * Return: 0 on success, -1 if @subexpr could not be evaluated to a scalar.
 */
static int eval_scalar_subexpr(const char *subexpr, char *out, size_t outsz)
{
    char expanded[4096];
    milk_server_expand_entities_in_expr(subexpr, expanded, sizeof(expanded));
    char assign[4200];
    snprintf(assign, sizeof(assign), "_milk_asrt_tmp = (%s)", expanded);
    int eval_rc = 2;
    int saved_q = dcquiet;
    dcquiet     = 1;
    cli_calc_eval_bash(assign, 0, &eval_rc);
    dcquiet   = saved_q;
    long vidx = variable_ID("_milk_asrt_tmp");
    if (eval_rc == 0 && vidx != -1)
    {
        if (data.core.variable[vidx].type == 1)
        {
            snprintf(out, outsz, "%ld", data.core.variable[vidx].value.l);
        }
        else if (data.core.variable[vidx].type == 2)
        {
            snprintf(out, outsz, "%s", data.core.variable[vidx].value.s);
        }
        else
        {
            snprintf(out, outsz, "%.*g", cli_float_digits, data.core.variable[vidx].value.f);
        }
        data.core.variable[vidx].used = 0;
        return 0;
    }
    return -1;
}

/* Per-byte classes from top_level_mask(), for the splitters and tokenizers below */
enum
{
    TL_LIT = 0, /* quoted, escaped, `${...}` or here-doc text */
    TL_NEST,    /* unquoted, inside ( ) or [ ] (the brackets themselves too) */
    TL_QUOTE,   /* quote character or escaping backslash, removed when words are built */
    TL_TOP,     /* unquoted, outside brackets */
    TL_COMMENT  /* # comment */
};

/**
 * top_level_mask - Classify each byte of @s by quoting and bracket nesting depth
 * @s:      Input expression or pipeline script
 * @n:      Length of @s in bytes
 * @parens: Non-zero to track `( )` nesting in addition to `[ ]`
 *
 * Combines `msl_scan()` shell-quoting classification with bracket depth so
 * statement splitters (`;`, `|>`) and relational-operator finders ignore
 * delimiters inside quotes, comments, slices, or parenthesized sub-expressions.
 *
 * Return: Heap-allocated `TL_*` byte mask of length @n + 1, or NULL on failure.
 */
static unsigned char *top_level_mask(const char *s, size_t n, int parens)
{
    msl_map_t m;
    if (msl_scan(s, n, &m) != 0)
    {
        return NULL;
    }
    unsigned char *tl = (unsigned char *) calloc(n + 1, 1);
    int            bd = 0;
    int            pd = 0;
    int            cd = 0;
    for (size_t i = 0; tl != NULL && i < n; i++)
    {
        char c   = s[i];
        int  ctx = m.ctx[i];
        if (ctx == MSL_SQ || ctx == MSL_ANSI)
        {
            tl[i] = (c == '\'') ? TL_QUOTE : TL_LIT;
        }
        else if (ctx == MSL_COMMENT)
        {
            tl[i] = TL_COMMENT;
        }
        else if (ctx == MSL_PARAM || ctx == MSL_HEREDOC)
        {
            tl[i] = TL_LIT;
        }
        else if (c == '\\' && i + 1 < n)
        {
            tl[i]   = TL_QUOTE;
            tl[++i] = TL_LIT;
        }
        else if (ctx == MSL_DQ)
        {
            tl[i] = (c == '"') ? TL_QUOTE : TL_LIT;
        }
        else
        {
            if (c == '[' || (parens && (c == '(' || c == '{')))
            {
                *(c == '[' ? &bd : (c == '(' ? &pd : &cd)) += 1;
            }
            int top = (bd == 0 && pd == 0 && cd == 0);
            if ((c == ']' && bd > 0) || (parens && c == ')' && pd > 0) ||
                (parens && c == '}' && cd > 0))
            {
                *(c == ']' ? &bd : (c == ')' ? &pd : &cd)) -= 1;
                top = 0;
            }
            tl[i] = top ? TL_TOP : TL_NEST;
        }
    }
    msl_free(&m);
    return tl;
}

/**
 * split_top_level - Invoke @fn on each top-level statement in @s
 * @s:      Input string containing one or more statements
 * @parens: Non-zero to treat `(` `)` as nested when scanning for separators
 * @pipe:   Non-zero to treat `|>` as a statement separator in addition to `;` and `\n`
 * @fn:     Callback invoked for each NUL-terminated statement segment
 * @ctx:    Opaque context pointer forwarded to @fn
 *
 * Splits compound `@calc` expressions or `@pipe` scripts at unquoted top-level
 * separators and stops at the first non-zero status returned by @fn.
 *
 * Return: 0 if all segments succeeded, or the first non-zero return value from @fn.
 */
static int split_top_level(const char *s,
                           int         parens,
                           int         pipe,
                           int (*fn)(char *seg, void *ctx),
                           void *ctx)
{
    size_t         n  = strlen(s);
    unsigned char *tl = top_level_mask(s, n, parens);
    if (tl == NULL)
    {
        return 1;
    }
    int    rc = 0;
    size_t st = 0;
    for (size_t i = 0; i <= n && rc == 0; i++)
    {
        int top = (i < n && tl[i] == TL_TOP);
        int sep = (i == n || (top && (s[i] == ';' || s[i] == '\n'))) ? 1
                  : (pipe && top && !strncmp(s + i, "|>", 2))        ? 2
                                                                     : 0;
        if (sep == 1 && i < n && parens)
        {
            size_t k = i + 1;
            while (k < n && (s[k] == ' ' || s[k] == '\t' || s[k] == ';' || s[k] == '\n' ||
                             tl[k] == TL_COMMENT))
            {
                k++;
            }
            if ((k + 4 <= n && strncmp(s + k, "elif", 4) == 0 &&
                 (isspace((unsigned char) s[k + 4]) || s[k + 4] == '(')) ||
                (k + 4 <= n && strncmp(s + k, "else", 4) == 0 &&
                 (isspace((unsigned char) s[k + 4]) || s[k + 4] == '{')))
            {
                sep = 0;
            }
        }
        if (sep == 0)
        {
            continue;
        }
        char *seg = strndup(s + st, i - st);
        rc        = (seg != NULL) ? fn(seg, ctx) : 1;
        free(seg);
        i += (size_t) sep - 1;
        st = i + 1;
    }
    free(tl);
    return rc;
}

/**
 * split_top_level_relop - Split a single-comparison expression into LHS and RHS
 * @expr:  Input boolean expression
 * @lhs:   Output buffer for the left-hand side
 * @lhssz: Capacity of @lhs in bytes
 * @rhs:   Output buffer for the right-hand side
 * @rhssz: Capacity of @rhs in bytes
 *
 * Locates a unique top-level relational operator (`==`, `!=`, `<=`, `>=`, `<`, `>`)
 * outside parentheses and quotes when no logical `&&`/`||` is present, enabling
 * `@assert` to print both sides' evaluated values on failure.
 *
 * Return: 1 if a single top-level relational operator was split, 0 otherwise.
 */
static int split_top_level_relop(const char *expr, char *lhs, size_t lhssz, char *rhs, size_t rhssz)
{
    const char    *op_pos   = NULL;
    size_t         op_len   = 0;
    int            op_count = 0;
    size_t         n        = strlen(expr);
    unsigned char *tl       = top_level_mask(expr, n, 1);
    if (tl == NULL)
    {
        return 0;
    }

    for (size_t i = 0; i < n; i++)
    {
        char c = expr[i];
        if (tl[i] != TL_TOP)
        {
            continue;
        }
        if ((c == '&' && expr[i + 1] == '&') || (c == '|' && expr[i + 1] == '|'))
        {
            free(tl);
            return 0;
        }
        if (strncmp(expr + i, "<=", 2) == 0 || strncmp(expr + i, ">=", 2) == 0 ||
            strncmp(expr + i, "==", 2) == 0 || strncmp(expr + i, "!=", 2) == 0)
        {
            op_pos = expr + i;
            op_len = 2;
            op_count++;
            i++;
        }
        else if (c == '<' || c == '>')
        {
            op_pos = expr + i;
            op_len = 1;
            op_count++;
        }
    }
    free(tl);

    if (op_count != 1 || op_pos == NULL)
    {
        return 0;
    }

    size_t l_len = (size_t) (op_pos - expr);
    if (l_len == 0 || l_len >= lhssz)
    {
        return 0;
    }
    memcpy(lhs, expr, l_len);
    lhs[l_len] = '\0';
    snprintf(rhs, rhssz, "%s", op_pos + op_len);
    return 1;
}

/**
 * is_bare_entity_arg - Check whether @s is a two-segment entity handle (`s.im`, `f.fps`, `p.proc`)
 * @s: Token to inspect
 *
 * Used by `@assert` so a bare entity argument without flags or operators
 * (`@assert @s.im`) defaults to `--exists` instead of expression evaluation.
 *
 * Return: 1 if @s is a bare `<ns>.<name>` entity reference, 0 otherwise.
 */
static int is_bare_entity_arg(const char *s)
{
    if (s == NULL || s[0] == '\0')
    {
        return 0;
    }
    while (*s == '@')
    {
        s++;
    }
    const char *dot1  = strchr(s, '.');
    size_t      nslen = (dot1 != NULL) ? (size_t) (dot1 - s) : 0;
    if (dot1 == NULL || !milk_is_entity_ns_prefix(s, nslen) || strchr(dot1 + 1, '.') != NULL)
    {
        return 0;
    }
    for (const char *p = dot1 + 1; *p != '\0'; p++)
    {
        if (!isalnum((unsigned char) *p) && *p != '_' && *p != '-')
        {
            return 0;
        }
    }
    return (dot1[1] != '\0') ? 1 : 0;
}

/**
 * has_expr_operators - Check whether @s contains arithmetic, comparison, or call syntax
 * @s: String to inspect
 *
 * Disambiguates a trailing failure message from the second token of an unquoted
 * multi-word boolean expression in `@assert`.
 *
 * Return: 1 if @s contains expression operator characters, 0 otherwise.
 */
static int has_expr_operators(const char *s)
{
    for (const char *p = s; *p != '\0'; p++)
    {
        if (*p == '<' || *p == '>' || *p == '=' || *p == '!' || *p == '(' || *p == '+' ||
            *p == '-' || *p == '*' || *p == '/')
        {
            return 1;
        }
    }
    return 0;
}

/**
 * milk_server_assert - Execute the `@assert` built-in command
 * @argc: Argument count (including `assert`)
 * @argv: Argument vector
 *
 * Supports both entity health flags (`--exists`, `--running`, `--dim`, `--cnt-min`,
 * `--fps-min`, `--max-age`) and arbitrary calculator boolean expressions, printing
 * actual vs. expected diagnostics to `stderr` when an assertion fails.
 *
 * Return: 0 if the assertion holds, 1 if it fails, 2 on usage/syntax error.
 */
int milk_server_assert(int argc, char **argv)
{
    int         chk_exists  = 0;
    int         chk_running = 0;
    int         has_fps_min = 0;
    double      fps_min     = 0.0;
    int         has_max_age = 0;
    double      max_age     = 0.0;
    int         has_cnt_min = 0;
    uint64_t    cnt_min     = 0;
    const char *dim_spec    = NULL;
    const char *msg         = NULL;
    const char *pos[64];
    int         npos = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--exists") == 0)
        {
            chk_exists = 1;
        }
        else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--running") == 0)
        {
            chk_running = 1;
        }
        else if (strcmp(argv[i], "--fps-min") == 0 && i + 1 < argc)
        {
            has_fps_min = 1;
            fps_min     = strtod(argv[++i], NULL);
        }
        else if (strncmp(argv[i], "--fps-min=", 10) == 0)
        {
            has_fps_min = 1;
            fps_min     = strtod(argv[i] + 10, NULL);
        }
        else if (strcmp(argv[i], "--max-age") == 0 && i + 1 < argc)
        {
            has_max_age = 1;
            max_age     = strtod(argv[++i], NULL);
        }
        else if (strncmp(argv[i], "--max-age=", 10) == 0)
        {
            has_max_age = 1;
            max_age     = strtod(argv[i] + 10, NULL);
        }
        else if (strcmp(argv[i], "--cnt-min") == 0 && i + 1 < argc)
        {
            has_cnt_min = 1;
            cnt_min     = (uint64_t) strtoull(argv[++i], NULL, 10);
        }
        else if (strncmp(argv[i], "--cnt-min=", 10) == 0)
        {
            has_cnt_min = 1;
            cnt_min     = (uint64_t) strtoull(argv[i] + 10, NULL, 10);
        }
        else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
        {
            dim_spec = argv[++i];
        }
        else if (strncmp(argv[i], "--dim=", 6) == 0)
        {
            dim_spec = argv[i] + 6;
        }
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--msg") == 0) && i + 1 < argc)
        {
            msg = argv[++i];
        }
        else if (strncmp(argv[i], "--msg=", 6) == 0)
        {
            msg = argv[i] + 6;
        }
        else if (npos < 64)
        {
            pos[npos++] = argv[i];
        }
    }

    int any_flag = (chk_exists || chk_running || has_fps_min || has_max_age || has_cnt_min ||
                    dim_spec != NULL);
    if (npos == 0)
    {
        fprintf(stderr, "Usage: @assert [-m <msg>] \"<boolean-expr>\" [<msg>]\n"
                        "   or: @assert [--exists|-e] [--running|-r] [--fps-min <hz>] "
                        "[--max-age <sec>] [--cnt-min <N>] [--dim <WxH[xD]>] <entity> [<msg>]\n");
        return 2;
    }

    if (!any_flag && is_bare_entity_arg(pos[0]) && (npos == 1 || !has_expr_operators(pos[1])))
    {
        chk_exists = 1;
        any_flag   = 1;
    }

    const char *m_sep = (msg != NULL && msg[0] != '\0') ? " -- " : "";
    const char *m_txt = (msg != NULL && msg[0] != '\0') ? msg : "";

    if (any_flag)
    {
        const char *target = pos[0];
        if (msg == NULL && npos >= 2)
        {
            msg   = pos[1];
            m_sep = (msg[0] != '\0') ? " -- " : "";
            m_txt = msg;
        }

        const char *clean = target;
        while (*clean == '@')
        {
            clean++;
        }
        const char    *sname = clean;
        milk_ns_kind_t ck    = milk_entity_ns_kind(clean, &sname);
        if (ck != MILK_NS_STREAM)
        {
            sname = clean;
        }

        int exists = milk_shm_prop_exists(clean, 0);
        if (!exists && (ck == MILK_NS_STREAM || strchr(clean, '.') == NULL))
        {
            if (image_ID(sname, data.core.image, data.core.NB_MAX_IMAGE) != -1)
            {
                exists = 1;
            }
        }
        if (!exists)
        {
            fprintf(stderr, "ASSERT FAILED: %s --exists (actual: missing)%s%s\n", target, m_sep,
                    m_txt);
            return 1;
        }
        if (chk_running && !milk_shm_prop_exists(clean, 1))
        {
            fprintf(stderr, "ASSERT FAILED: %s --running (actual: not running)%s%s\n", target,
                    m_sep, m_txt);
            return 1;
        }
        if (dim_spec != NULL)
        {
            char xbuf[64] = "", ybuf[64] = "", zbuf[64] = "", nbuf[64] = "";
            char qspec[300];
            snprintf(qspec, sizeof(qspec), "s.%s.xsize", sname);
            milk_shm_prop_get(qspec, xbuf, sizeof(xbuf));
            snprintf(qspec, sizeof(qspec), "s.%s.ysize", sname);
            milk_shm_prop_get(qspec, ybuf, sizeof(ybuf));
            snprintf(qspec, sizeof(qspec), "s.%s.zsize", sname);
            milk_shm_prop_get(qspec, zbuf, sizeof(zbuf));
            snprintf(qspec, sizeof(qspec), "s.%s.naxis", sname);
            milk_shm_prop_get(qspec, nbuf, sizeof(nbuf));
            int nax = atoi(nbuf);
            int want_3d =
                (strchr(dim_spec, 'x') != NULL && strchr(strchr(dim_spec, 'x') + 1, 'x') != NULL);
            char act_dim[160];
            if (nax >= 3 || want_3d)
            {
                snprintf(act_dim, sizeof(act_dim), "%sx%sx%s", xbuf, ybuf, zbuf);
            }
            else
            {
                snprintf(act_dim, sizeof(act_dim), "%sx%s", xbuf, ybuf);
            }
            if (strcmp(act_dim, dim_spec) != 0)
            {
                fprintf(stderr, "ASSERT FAILED: %s --dim %s (actual: %s)%s%s\n", target, dim_spec,
                        act_dim, m_sep, m_txt);
                return 1;
            }
        }
        if (has_cnt_min)
        {
            char cbuf[64] = "0";
            char qspec[280];
            if (milk_entity_ns_kind(clean, NULL) == MILK_NS_PROC)
            {
                snprintf(qspec, sizeof(qspec), "%s.loopcnt", clean);
            }
            else
            {
                snprintf(qspec, sizeof(qspec), "s.%s.cnt0", sname);
            }
            milk_shm_prop_get(qspec, cbuf, sizeof(cbuf));
            uint64_t act_cnt = (uint64_t) strtoull(cbuf, NULL, 10);
            if (act_cnt < cnt_min)
            {
                fprintf(stderr, "ASSERT FAILED: %s --cnt-min %llu (actual: %llu)%s%s\n", target,
                        (unsigned long long) cnt_min, (unsigned long long) act_cnt, m_sep, m_txt);
                return 1;
            }
        }
        if (has_fps_min)
        {
            char fbuf[64] = "0";
            char qspec[280];
            if (milk_entity_ns_kind(clean, NULL) == MILK_NS_PROC)
            {
                snprintf(qspec, sizeof(qspec), "%s.fps_hz", clean);
            }
            else
            {
                snprintf(qspec, sizeof(qspec), "s.%s.fps", sname);
            }
            milk_shm_prop_get(qspec, fbuf, sizeof(fbuf));
            double act_fps = strtod(fbuf, NULL);
            if (act_fps < fps_min)
            {
                fprintf(stderr, "ASSERT FAILED: %s --fps-min %g (actual: %s Hz)%s%s\n", target,
                        fps_min, fbuf, m_sep, m_txt);
                return 1;
            }
        }
        if (has_max_age)
        {
            char abuf[64] = "0";
            char qspec[280];
            snprintf(qspec, sizeof(qspec), "s.%s.age", sname);
            milk_shm_prop_get(qspec, abuf, sizeof(abuf));
            double act_age = strtod(abuf, NULL);
            if (act_age > max_age)
            {
                fprintf(stderr, "ASSERT FAILED: %s --max-age %g (actual: %s s)%s%s\n", target,
                        max_age, abuf, m_sep, m_txt);
                return 1;
            }
        }
        return 0;
    }

    /* Expression mode */
    char expr[4096];
    expr[0] = '\0';
    if (npos == 2 && msg == NULL && has_expr_operators(pos[0]))
    {
        snprintf(expr, sizeof(expr), "%s", pos[0]);
        msg   = pos[1];
        m_sep = (msg[0] != '\0') ? " -- " : "";
        m_txt = msg;
    }
    else
    {
        size_t epos = 0;
        for (int i = 0; i < npos && epos + 2 < sizeof(expr); i++)
        {
            epos += (size_t) snprintf(expr + epos, sizeof(expr) - epos, "%s%s", (i > 0) ? " " : "",
                                      pos[i]);
        }
    }

    char expanded[8192];
    milk_server_expand_entities_in_expr(expr, expanded, sizeof(expanded));
    int eval_rc = 2;
    cli_calc_eval_bash(expanded, 1, &eval_rc);
    if (eval_rc == 0)
    {
        return 0;
    }

    char lhs[2048], rhs[2048], lval[256], rval[256];
    if (split_top_level_relop(expr, lhs, sizeof(lhs), rhs, sizeof(rhs)) &&
        eval_scalar_subexpr(lhs, lval, sizeof(lval)) == 0 &&
        eval_scalar_subexpr(rhs, rval, sizeof(rval)) == 0)
    {
        fprintf(stderr, "ASSERT FAILED: %s (actual: %s vs %s)%s%s\n", expr, lval, rval, m_sep,
                m_txt);
    }
    else
    {
        fprintf(stderr, "ASSERT FAILED: %s%s%s\n", expr, m_sep, m_txt);
    }
    return (eval_rc == 2) ? 2 : 1;
}

/**
 * try_exec_entity_lhs_assignment - Handle `@f.fps.param = rhs_expr` inside `@calc`
 * @expr:   Single calculator statement
 * @out_rc: Receives 0 on success or 1 on property write error
 *
 * Allows `@calc` statements to assign computed scalar or string expressions
 * directly to shared-memory entity properties (`s.<im>.kw.<k>`, `f.<fps>.<param>`).
 *
 * Return: 1 if @expr was an entity property assignment, 0 otherwise.
 */
static int try_exec_entity_lhs_assignment(const char *expr, int *out_rc)
{
    const char *p = expr;
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }
    const char *lstart = (*p == '@') ? (p + 1) : p;
    const char *le     = lstart;
    int         ndots  = 0;
    while (*le != '\0' && (isalnum((unsigned char) *le) || *le == '_' || *le == '.' || *le == '-'))
    {
        if (*le == '.')
        {
            ndots++;
        }
        le++;
    }
    const char *dot1  = strchr(lstart, '.');
    size_t      nslen = (dot1 != NULL && dot1 < le) ? (size_t) (dot1 - lstart) : 0;
    if (ndots < 2 || (size_t) (le - lstart) >= 256 || !milk_is_entity_ns_prefix(lstart, nslen))
    {
        return 0;
    }

    const char *eq = le;
    while (*eq == ' ' || *eq == '\t')
    {
        eq++;
    }
    if (*eq != '=' || eq[1] == '=')
    {
        return 0;
    }

    char lhs[256];
    memcpy(lhs, lstart, (size_t) (le - lstart));
    lhs[le - lstart] = '\0';
    const char *rhs  = eq + 1;
    while (*rhs == ' ' || *rhs == '\t')
    {
        rhs++;
    }
    char exp_rhs[4096];
    milk_server_expand_entities_in_expr(rhs, exp_rhs, sizeof(exp_rhs));
    size_t rlen = strlen(exp_rhs);
    while (rlen > 0 && (exp_rhs[rlen - 1] == ' ' || exp_rhs[rlen - 1] == '\t'))
    {
        exp_rhs[--rlen] = '\0';
    }
    if (rlen >= 2 && ((exp_rhs[0] == '"' && exp_rhs[rlen - 1] == '"') ||
                      (exp_rhs[0] == '\'' && exp_rhs[rlen - 1] == '\'')))
    {
        exp_rhs[rlen - 1] = '\0';
        *out_rc           = (milk_shm_prop_set(lhs, exp_rhs + 1, NULL) == 0) ? 0 : 1;
        if (*out_rc == 0 && cli_calc_out_mode == CLI_CALC_OUT_REPL)
        {
            printf("%s %s\n", lhs, exp_rhs + 1);
            fflush(stdout);
        }
        return 1;
    }

    char assign_expr[4200];
    snprintf(assign_expr, sizeof(assign_expr), "_milk_ent_rhs = %s", exp_rhs);
    int eval_rc       = 2;
    int saved_q       = dcquiet;
    int saved_om      = cli_calc_out_mode;
    dcquiet           = 1;
    cli_calc_out_mode = CLI_CALC_OUT_VALUES;
    cli_calc_eval_bash(assign_expr, 0, &eval_rc);
    cli_calc_out_mode = saved_om;
    dcquiet           = saved_q;
    long vidx         = variable_ID("_milk_ent_rhs");
    if (eval_rc == 0 && vidx != -1)
    {
        char vbuf[128];
        if (data.core.variable[vidx].type == 1)
        {
            snprintf(vbuf, sizeof(vbuf), "%ld", data.core.variable[vidx].value.l);
        }
        else if (data.core.variable[vidx].type == 2)
        {
            snprintf(vbuf, sizeof(vbuf), "%s", data.core.variable[vidx].value.s);
        }
        else
        {
            snprintf(vbuf, sizeof(vbuf), "%.*g", cli_float_digits,
                     data.core.variable[vidx].value.f);
        }
        data.core.variable[vidx].used = 0;
        *out_rc                       = (milk_shm_prop_set(lhs, vbuf, NULL) == 0) ? 0 : 1;
        if (*out_rc == 0 && cli_calc_out_mode == CLI_CALC_OUT_REPL)
        {
            printf("%s %s\n", lhs, vbuf);
            fflush(stdout);
        }
    }
    else
    {
        *out_rc = (milk_shm_prop_set(lhs, exp_rhs, NULL) == 0) ? 0 : 1;
        if (*out_rc == 0 && cli_calc_out_mode == CLI_CALC_OUT_REPL)
        {
            printf("%s %s\n", lhs, exp_rhs);
            fflush(stdout);
        }
    }
    return 1;
}

/**
 * substitute_underscore_token - Replace standalone `_` tokens with @last_img in a `@pipe` step
 * @in:       Raw pipeline step text
 * @last_img: Name of the image produced by the preceding pipeline stage
 * @out:      Output buffer receiving the substituted step
 * @outsz:    Capacity of @out in bytes
 *
 * Enables concise `@pipe` stage chaining (`a = zeros(8,8) |> b = _ + 1 |> stats _`)
 * without touching `_` inside identifiers, strings, or property paths.
 */
static void substitute_underscore_token(const char *in,
                                        const char *last_img,
                                        char       *out,
                                        size_t      outsz)
{
    size_t         ilen = strlen(in);
    unsigned char *tl =
        (last_img != NULL && last_img[0] != '\0') ? top_level_mask(in, ilen, 0) : NULL;
    if (tl == NULL)
    {
        snprintf(out, outsz, "%s", in);
        return;
    }
    size_t pos  = 0;
    size_t llen = strlen(last_img);

    for (size_t i = 0; i < ilen && pos + 1 < outsz;)
    {
        if ((tl[i] == TL_TOP || tl[i] == TL_NEST) && in[i] == '_')
        {
            int prev_ok = (i == 0) || (!isalnum((unsigned char) in[i - 1]) && in[i - 1] != '_' &&
                                       in[i - 1] != '.');
            int next_ok = (i + 1 >= ilen) || (!isalnum((unsigned char) in[i + 1]) &&
                                              in[i + 1] != '_' && in[i + 1] != '.');
            if (prev_ok && next_ok && pos + llen < outsz)
            {
                memcpy(out + pos, last_img, llen);
                pos += llen;
                i++;
                continue;
            }
        }
        out[pos++] = in[i++];
    }
    out[pos] = '\0';
    free(tl);
}

/**
 * exec_batch_calc_expr - Evaluate one calculator stage inside `@pipe` and track its LHS image
 * @expr:        Calculator expression or assignment
 * @is_test:     Non-zero if the stage is a `@test` boolean check
 * @last_img:    Updated with the LHS image name if the assignment produced an image
 * @last_img_sz: Capacity of @last_img in bytes
 *
 * Runs @expr through the calculator and records any newly assigned LHS image in
 * @last_img so subsequent `@pipe` stages can reference it as `_`.
 *
 * Return: 0 on success, non-zero on evaluation error or false `@test`.
 */
static int exec_batch_calc_expr(const char *expr, int is_test, char *last_img, size_t last_img_sz)
{
    int rc = 0;
    if (is_test || !try_exec_entity_lhs_assignment(expr, &rc))
    {
        char expanded[8192];
        milk_server_expand_entities_in_expr(expr, expanded, sizeof(expanded));
        int eval_rc = 2;
        cli_calc_eval_bash(expanded, is_test, &eval_rc);
        rc = eval_rc;
        if (rc == 0 && !is_test)
        {
            const char *p = expanded;
            while (*p == ' ' || *p == '\t')
            {
                p++;
            }
            char   lhs[128];
            size_t l = 0;
            while (p[l] != '\0' && (isalnum((unsigned char) p[l]) || p[l] == '_') &&
                   l + 1 < sizeof(lhs))
            {
                lhs[l] = p[l];
                l++;
            }
            lhs[l] = '\0';
            if (l > 0 && image_ID(lhs, data.core.image, data.core.NB_MAX_IMAGE) != -1)
            {
                snprintf(last_img, last_img_sz, "%s", lhs);
            }
        }
    }
    return rc;
}

/**
 * tokenize_step_line - Split a `@pipe` command step into unquoted words
 * @step:     Step line to tokenize
 * @argv_out: Output pointer array receiving word pointers into @arena
 * @max_args: Maximum number of entries in @argv_out
 * @arena:    Scratch character buffer storing NUL-terminated words
 * @arena_sz: Capacity of @arena in bytes
 *
 * Splits @step on top-level whitespace while stripping outer shell quotes and
 * stopping at `#` comments, producing an `argv` vector for `milk_server_dispatch_argv()`.
 *
 * Return: Number of words placed in @argv_out.
 */
static int tokenize_step_line(const char *step,
                              char      **argv_out,
                              int         max_args,
                              char       *arena,
                              size_t      arena_sz)
{
    size_t         n  = strlen(step);
    unsigned char *tl = top_level_mask(step, n, 0);
    if (tl == NULL)
    {
        return 0;
    }
    int    argc    = 0;
    size_t apos    = 0;
    int    in_word = 0;
    for (size_t i = 0; i < n && tl[i] != TL_COMMENT && apos + 1 < arena_sz; i++)
    {
        if (tl[i] == TL_TOP && (step[i] == ' ' || step[i] == '\t'))
        {
            if (in_word)
            {
                arena[apos++] = '\0';
                in_word       = 0;
            }
            continue;
        }
        if (!in_word)
        {
            if (argc >= max_args)
            {
                break;
            }
            argv_out[argc++] = arena + apos;
            in_word          = 1;
        }
        if (tl[i] != TL_QUOTE)
        {
            arena[apos++] = step[i];
        }
    }
    if (in_word)
    {
        arena[apos] = '\0';
    }
    free(tl);
    return argc;
}

/**
 * exec_single_batch_step - Execute one stage of a `@pipe` script
 * @raw_step:    Raw text of the pipeline step
 * @last_img:    In/out buffer holding the `_` image name across stages
 * @last_img_sz: Capacity of @last_img in bytes
 *
 * Substitutes `_` with @last_img, dispatches either a calculator expression or a
 * CLI/entity command, and updates @last_img when the step produces an image.
 *
 * Return: 0 on success, non-zero on command or expression failure.
 */
static int exec_single_batch_step(const char *raw_step, char *last_img, size_t last_img_sz)
{
    while (*raw_step == ' ' || *raw_step == '\t' || *raw_step == '\r' || *raw_step == '\n')
    {
        raw_step++;
    }
    if (*raw_step == '\0' || *raw_step == '#')
    {
        return 0;
    }

    char step_buf[4096];
    substitute_underscore_token(raw_step, last_img, step_buf, sizeof(step_buf));
    size_t slen = strlen(step_buf);
    if (slen >= sizeof(step_buf) - 1)
    {
        fprintf(stderr, "ERROR: @pipe step too long (max %zu chars)\n", sizeof(step_buf) - 2);
        return 1;
    }
    while (slen > 0 && (step_buf[slen - 1] == ' ' || step_buf[slen - 1] == '\t' ||
                        step_buf[slen - 1] == '\r' || step_buf[slen - 1] == '\n'))
    {
        step_buf[--slen] = '\0';
    }
    if (slen == 0)
    {
        return 0;
    }

    const char *s = step_buf;
    if (s[0] == '@')
    {
        s++;
    }
    if (s[0] == ' ' || s[0] == '\t')
    {
        while (*s == ' ' || *s == '\t')
        {
            s++;
        }
        return exec_batch_calc_expr(s, 0, last_img, last_img_sz);
    }
    if (strncmp(s, "calc", 4) == 0 && (s[4] == ' ' || s[4] == '\t'))
    {
        s += 4;
        while (*s == ' ' || *s == '\t')
        {
            s++;
        }
        return exec_batch_calc_expr(s, 0, last_img, last_img_sz);
    }
    if (strncmp(s, "test", 4) == 0 && (s[4] == ' ' || s[4] == '\t'))
    {
        s += 4;
        while (*s == ' ' || *s == '\t')
        {
            s++;
        }
        return exec_batch_calc_expr(s, 1, last_img, last_img_sz);
    }

    char *s_argv[64];
    char  arena[4096];
    int   s_argc = tokenize_step_line(step_buf, s_argv, 64, arena, sizeof(arena));
    if (s_argc <= 0)
    {
        return 0;
    }

    const char *w0 = s_argv[0];
    if (w0[0] == '@')
    {
        w0++;
    }

    if (strcmp(w0, "set") == 0 && s_argc >= 2)
    {
        return milk_server_set_argv(s_argc, s_argv);
    }
    if (strcmp(w0, "get") == 0 && s_argc >= 2)
    {
        char val[16384];
        if (milk_shm_prop_get(s_argv[1], val, sizeof(val)) == 0)
        {
            printf("%s\n", val);
            return 0;
        }
        return 1;
    }

    int ambig  = 0;
    int cmdidx = milk_server_resolve_cmd_index(w0, &ambig);
    if (cmdidx < 0 && !ambig && strcmp(w0, "streamcat") != 0 && strcmp(w0, "streamwrite") != 0 &&
        (strchr(step_buf, '=') != NULL || strchr(step_buf, '(') != NULL ||
         strchr(step_buf, '+') != NULL || strchr(step_buf, '-') != NULL ||
         strchr(step_buf, '*') != NULL || strchr(step_buf, '/') != NULL))
    {
        return exec_batch_calc_expr(s, 0, last_img, last_img_sz);
    }

    int rc = milk_server_dispatch_argv(s_argc, s_argv);
    if (rc == 0)
    {
        for (int k = 1; k < s_argc; k++)
        {
            IMGID tmpid = imgid_make_from_name(s_argv[k]);
            if (!tmpid.stream_parse_error && tmpid.name[0] != '\0' &&
                image_ID(tmpid.name, data.core.image, data.core.NB_MAX_IMAGE) != -1)
            {
                snprintf(last_img, last_img_sz, "%s", tmpid.name);
            }
            imgid_free(&tmpid);
        }
    }
    return rc;
}

/**
 * batch_step_cb - `split_top_level()` callback that runs one `@pipe` step
 * @seg: Pipeline segment string
 * @ctx: Pointer to the 256-byte `last_img` buffer
 *
 * Return: 0 on success, non-zero to abort the pipeline.
 */
static int batch_step_cb(char *seg, void *ctx)
{
    return exec_single_batch_step(seg, (char *) ctx, 256);
}

/**
 * milk_server_batch_or_pipe - Execute a `@pipe` multi-step command/calculator chain
 * @argc: Argument count (including `pipe`)
 * @argv: Argument vector (or `argc == 1` when reading steps from `stdin`)
 *
 * Runs multiple commands and calculator expressions in a single server round-trip,
 * propagating the most recent output image as `_` (persisted in `_last_img`).
 *
 * Return: 0 if all pipeline stages succeeded, non-zero on the first failing stage.
 */
int milk_server_batch_or_pipe(int argc, char **argv)
{
    char  *script = NULL;
    size_t slen   = 0;

    if (argc <= 1)
    {
        if (isatty(STDIN_FILENO))
        {
            return 0;
        }
        size_t cap = 4096;
        script     = (char *) malloc(cap);
        if (script == NULL)
        {
            return 1;
        }
        ssize_t nr;
        while ((nr = read(STDIN_FILENO, script + slen, cap - slen - 1)) > 0)
        {
            slen += (size_t) nr;
            if (slen + 1024 >= cap)
            {
                cap *= 2;
                char *ns = (char *) realloc(script, cap);
                if (ns == NULL)
                {
                    free(script);
                    return 1;
                }
                script = ns;
            }
        }
        script[slen] = '\0';
    }
    else if (argc == 2)
    {
        script = strdup(argv[1]);
        slen   = (script != NULL) ? strlen(script) : 0;
    }
    else
    {
        int    has_sep_tok = 0;
        size_t cap         = 64;
        for (int i = 1; i < argc; i++)
        {
            cap += strlen(argv[i]) + 4;
            if (strcmp(argv[i], "|>") == 0 || strcmp(argv[i], ";") == 0)
            {
                has_sep_tok = 1;
            }
        }
        script = (char *) malloc(cap);
        if (script == NULL)
        {
            return 1;
        }
        int multi_stage_args =
            (!has_sep_tok && (strchr(argv[1], ' ') != NULL || strchr(argv[1], '=') != NULL));
        for (int i = 1; i < argc; i++)
        {
            if (has_sep_tok && (strcmp(argv[i], "|>") == 0 || strcmp(argv[i], ";") == 0))
            {
                memcpy(script + slen, "; ", 2);
                slen += 2;
            }
            else
            {
                if (i > 1)
                {
                    if (multi_stage_args)
                    {
                        memcpy(script + slen, "; ", 2);
                        slen += 2;
                    }
                    else
                    {
                        script[slen++] = ' ';
                    }
                }
                size_t l = strlen(argv[i]);
                memcpy(script + slen, argv[i], l);
                slen += l;
            }
        }
        script[slen] = '\0';
    }

    if (script == NULL)
    {
        return 1;
    }

    char last_img[256] = "";
    long vidx_last     = variable_ID("_last_img");
    if (vidx_last != -1 && data.core.variable[vidx_last].type == 2)
    {
        snprintf(last_img, sizeof(last_img), "%s", data.core.variable[vidx_last].value.s);
    }

    int rc = split_top_level(script, 0, 1, batch_step_cb, last_img);

    if (last_img[0] != '\0')
    {
        create_variable_string_ID("_last_img", last_img);
    }

    free(script);
    return rc;
}

/**
 * calc_stmt_cb - `split_top_level()` callback that evaluates one `@calc`/`@test` statement
 * @seg: Single semicolon-separated statement
 * @ctx: Pointer to `int is_test`
 *
 * Return: 0 on success, non-zero on error or false `@test`.
 */
static int calc_stmt_cb(char *seg, void *ctx)
{
    int    is_test = *(const int *) ctx;
    int    rc      = 0;
    size_t len     = strlen(seg);
    while (len > 0 && (seg[len - 1] == ' ' || seg[len - 1] == '\t'))
    {
        seg[--len] = '\0';
    }
    seg += strspn(seg, " \t");
    if (seg[0] == '\0')
    {
        return 0;
    }
    if (strlen(seg) >= 4096)
    {
        seg[4095] = '\0';
    }
    if (is_test || !try_exec_entity_lhs_assignment(seg, &rc))
    {
        char expanded[8192];
        milk_server_expand_entities_in_expr(seg, expanded, sizeof(expanded));
        int eval_rc = 2;
        cli_calc_eval_bash(expanded, is_test, &eval_rc);
        rc = eval_rc;
    }
    return rc;
}

/**
 * milk_server_exec_calc_or_test - Execute a `MILK_MSG_CALC` (`@calc`) or `MILK_MSG_TEST` (`@test`) payload
 * @client_fd:  Client socket descriptor (unused; scalar bindings return on stdout)
 * @is_test:    Non-zero for boolean `@test` evaluation
 * @flags:      Output mode flags (`MILK_FLAG_PIXELS`, `MILK_FLAG_BIND`, `MILK_FLAG_REPL`)
 * @payload:    NUL-terminated expression followed by packed bash variable bindings
 * @len:        Total payload length in bytes
 * @log_cmd:    Output buffer receiving the human-readable command for the server log
 * @log_cmd_sz: Capacity of @log_cmd in bytes
 *
 * Synchronizes referenced bash scalars into `data.core.variable[]`, splits the
 * expression on top-level semicolons/newlines, and evaluates each statement.
 *
 * Return: 0 on success (or true `@test`), 1 on false `@test` or runtime error, 2 on syntax error.
 */
int milk_server_exec_calc_or_test(int            client_fd,
                                  int            is_test,
                                  uint8_t        flags,
                                  const uint8_t *payload,
                                  uint32_t       len,
                                  char          *log_cmd,
                                  size_t         log_cmd_sz)
{
    if (payload == NULL || len == 0)
    {
        return is_test ? 1 : 0;
    }

    const char *expr = (const char *) payload;
    snprintf(log_cmd, log_cmd_sz, "%s %s", is_test ? "@test" : "@=", expr);
    uint32_t off = (uint32_t) (strlen(expr) + 1);
    if (off + sizeof(uint32_t) <= len)
    {
        uint32_t nvars = 0;
        memcpy(&nvars, payload + off, sizeof(uint32_t));
        off += (uint32_t) sizeof(uint32_t);
        milk_server_sync_bash_vars(payload, len, nvars, &off, (flags & MILK_FLAG_REPL) != 0);
    }

    (void) client_fd; /* calc variables reach bash only through -v (its output) */
    if (!is_test && strncmp(expr, "__explain ", 10) == 0)
    {
        /* `@calc --explain`: the whole text is one program, not split at `;` */
        int xrc = cli_calc_prog_explain(expr + 10, stdout);
        fflush(stdout);
        return xrc;
    }
    cli_calc_out_mode = (flags & MILK_FLAG_PIXELS) ? CLI_CALC_OUT_PIXELS
                        : (flags & MILK_FLAG_BIND) ? CLI_CALC_OUT_BIND
                        : (flags & MILK_FLAG_REPL) ? CLI_CALC_OUT_REPL
                                                   : CLI_CALC_OUT_VALUES;
    int rc            = split_top_level(expr, 1, 0, calc_stmt_cb, &is_test);
    cli_calc_out_mode = CLI_CALC_OUT_VALUES;
    if (!is_test)
    {
        milk_server_mark_expr_vars_synced(expr);
    }
    return rc;
}

/**
 * milk_server_exec_calc_repl_inproc - Evaluate a Calc mode REPL line directly in `milk-cli`
 * @expr: Calculator expression string
 *
 * Executes @expr in-process with `MILK_FLAG_REPL` output formatting and records
 * the operation in the shared-memory server log when active, bypassing the bash
 * co-process and UNIX domain socket round-trip.
 *
 * Return: 0 on success, non-zero on evaluation or syntax error.
 */
int milk_server_exec_calc_repl_inproc(const char *expr)
{
    if (expr == NULL || expr[0] == '\0')
    {
        return 0;
    }

    milk_server_cmd_lock();
    struct timespec ts_start;
    if (g_srvlog_shm != NULL)
    {
        clock_gettime(CLOCK_REALTIME, &ts_start);
    }

    char     log_cmd[MILK_SRVLOG_CMD_MAX] = "";
    uint32_t len                          = (uint32_t) strlen(expr) + 1;
    int      rc = milk_server_exec_calc_or_test(-1, 0, MILK_FLAG_REPL, (const uint8_t *) expr, len,
                                                log_cmd, sizeof(log_cmd));
    fflush(stdout);
    fflush(stderr);

    if (g_srvlog_shm != NULL && log_cmd[0] != '\0')
    {
        struct timespec ts_end;
        clock_gettime(CLOCK_REALTIME, &ts_end);
        int64_t dur_ns = ((int64_t) ts_end.tv_sec - (int64_t) ts_start.tv_sec) * 1000000000LL +
                         ((int64_t) ts_end.tv_nsec - (int64_t) ts_start.tv_nsec);
        milk_srvlog_record(g_srvlog_shm, &ts_start, (dur_ns > 0) ? (uint64_t) dur_ns : 0, getpid(),
                           MILK_MSG_CALC, rc, log_cmd, "");
    }
    milk_server_cmd_unlock();
    return rc;
}
