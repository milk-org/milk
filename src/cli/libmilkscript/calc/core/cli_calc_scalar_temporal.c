// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_scalar_temporal.c
 * @brief In-shell local scalar expression evaluator and live SHM property reader for the bash client.
 *
 * libc, libm and the shm property readers only: this file is part of the thin bash client.
 */

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "cli_calc_scalar.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

#define LOCAL_MAX_TOK 128

typedef struct
{
    cli_token_type              type;
    long                        l;
    double                      d;
    const cli_calc_scalar_fn_t *fn;
} ltok_t;

typedef struct
{
    ltok_t t[LOCAL_MAX_TOK];
    int    n;
    int    pos;
    int    bail; /* the engine must evaluate it */
} lexp_t;

/**
 * num_l - Construct a `VAL_LONG` scalar value without pulling in CLIcore symbols
 * @v: Signed integer value
 *
 * Return: `val_t` initialized with `type = VAL_LONG` and `lval = @v`
 */
static val_t num_l(long v)
{
    val_t r;
    r.type    = VAL_LONG;
    r.lval    = v;
    r.dval    = 0.0;
    r.sval[0] = '\0';
    return r;
}

/**
 * num_d - Construct a `VAL_DOUBLE` scalar value without pulling in CLIcore symbols
 * @v: Double-precision floating-point value
 *
 * Return: `val_t` initialized with `type = VAL_DOUBLE` and `dval = @v`
 */
static val_t num_d(double v)
{
    val_t r;
    r.type    = VAL_DOUBLE;
    r.lval    = 0;
    r.dval    = v;
    r.sval[0] = '\0';
    return r;
}

/**
 * num - Extract a scalar `val_t` as a `double`
 * @v: Scalar calculator value (`VAL_LONG` or `VAL_DOUBLE`)
 *
 * Return: Numeric value of @v as a `double`
 */
static double num(const val_t *v)
{
    return (v->type == VAL_LONG) ? (double) v->lval : v->dval;
}

/**
 * push - Append a token to the in-shell scalar evaluator's token buffer
 * @x:    Local lexer/evaluator state
 * @type: Token type
 * @l:    Integer literal value (for `TOK_LONG`)
 * @d:    Floating-point literal value (for `TOK_DOUBLE`)
 * @fn:   Scalar function descriptor (for `TOK_FUNC_*`), or NULL
 *
 * Appends a lightweight token to `x->t`, failing if `LOCAL_MAX_TOK - 1` is
 * reached so the expression falls back to the server engine.
 *
 * Return: 0 on success, -1 if the local token buffer is full
 */
static int push(lexp_t *x, cli_token_type type, long l, double d, const cli_calc_scalar_fn_t *fn)
{
    if (x->n >= LOCAL_MAX_TOK - 1)
    {
        return -1;
    }
    x->t[x->n++] = (ltok_t) { type, l, d, fn };
    return 0;
}

/**
 * lex_entity - Read a live stream/FPS/process property at @p as numeric tokens
 * @p: Source pointer starting at an identifier
 * @x: Local lexer/evaluator state
 *
 * Return: Byte length of the property reference consumed, or 0 if not a numeric SHM property
 */
static size_t lex_entity(const char *p, lexp_t *x)
{
    size_t e     = 0;
    int    ndots = 0;
    while (isalnum((unsigned char) p[e]) || p[e] == '_' || p[e] == '.' || p[e] == '-')
    {
        ndots += (p[e++] == '.');
    }
    while (e > 0 && p[e - 1] == '.')
    {
        e--;
        ndots--;
    }
    char tok[256];
    char val[512];
    if (ndots < 2 || e >= sizeof(tok))
    {
        return 0;
    }
    memcpy(tok, p, e);
    tok[e]           = '\0';
    milk_ns_kind_t k = milk_entity_ns_kind(tok, NULL);
    if (k == MILK_NS_NONE || k == MILK_NS_LOCAL || milk_shm_prop_get(tok, val, sizeof(val)) != 0)
    {
        return 0;
    }
    const char    *v = val + (val[0] == '-');
    cli_token_type type;
    long           l = 0;
    double         d = 0.0;
    size_t         n = cli_calc_scan_number(v, &type, &l, &d);
    if (n == 0 || v[n] != '\0' || (val[0] == '-' && push(x, TOK_OP_MINUS, 0, 0.0, NULL) != 0) ||
        push(x, type, l, d, NULL) != 0)
    {
        return 0;
    }
    return e;
}

/**
 * lex - Tokenize a pure scalar expression string into @x
 * @s: Expression string to tokenize
 * @x: Local lexer/evaluator state
 *
 * Return: 0 on success, -1 if @s contains constructs requiring the server engine
 */
static int lex(const char *s, lexp_t *x)
{
    x->n = x->pos = x->bail = 0;
    for (const char *p = s; *p != '\0';)
    {
        const cli_calc_scalar_fn_t *fn;
        cli_token_type              type;
        long                        l = 0;
        double                      d = 0.0;
        int                         len;
        size_t                      n;

        if (*p == ' ' || *p == '\t')
        {
            p++;
        }
        else if ((type = cli_calc_scan_op(p, &len)) != TOK_EOF)
        {
            if (cli_calc_op_prec(type) == 0 && type != TOK_OP_QUESTION) /* an assignment */
            {
                return -1;
            }
            if (push(x, type, 0, 0.0, NULL) != 0)
            {
                return -1;
            }
            p += len;
        }
        else if ((n = cli_calc_scan_number(p, &type, &l, &d)) > 0)
        {
            if (isalnum((unsigned char) p[n]) || p[n] == '_' || p[n] == '.' ||
                push(x, type, l, d, NULL) != 0)
            {
                return -1; /* 2x, 0xfg: the engine reports it */
            }
            p += n;
        }
        else if ((fn = cli_calc_scalar_fn(p)) != NULL)
        {
            type = (fn->nargs == 1)   ? TOK_FUNC_D_D
                   : (fn->nargs == 2) ? TOK_FUNC_DD_D
                                      : TOK_FUNC_DDD_D;
            if (push(x, type, 0, 0.0, fn) != 0)
            {
                return -1;
            }
            p += strlen(fn->name);
        }
        else if ((isalpha((unsigned char) *p) || *p == '_') && (n = lex_entity(p, x)) > 0)
        {
            p += n;
        }
        else
        {
            return -1; /* a name, string, slice, ';' ...: the engine's business */
        }
    }
    x->t[x->n].type = TOK_EOF;
    return (x->n > 0) ? 0 : -1;
}

static val_t lexpr(lexp_t *x, int min_prec, int live);

/**
 * expect - Consume expected token @t or mark @x for fallback to the server engine
 * @x: Local lexer/evaluator state
 * @t: Expected token type
 */
static void expect(lexp_t *x, cli_token_type t)
{
    if (!x->bail && x->t[x->pos].type != t)
    {
        x->bail = 1;
    }
    x->pos += !x->bail;
}

/**
 * lprimary - Parse and optionally evaluate one primary atom in the local evaluator
 * @x:    Local lexer/evaluator state
 * @live: 1 to evaluate, 0 to parse only (short-circuit inactive branch)
 *
 * Return: Evaluated scalar value (or 0 when `@live == 0` or `x->bail`)
 */
static val_t lprimary(lexp_t *x, int live)
{
    const ltok_t *k = &x->t[x->pos];
    val_t         v;

    if (x->bail || k->type == TOK_EOF)
    {
        x->bail = 1;
        return num_l(0);
    }
    x->pos++;
    switch (k->type)
    {
    case TOK_LONG:
        return num_l(k->l);
    case TOK_DOUBLE:
        return num_d(k->d);
    case TOK_OP_MINUS: /* binds to the atom: -2^2 is 4 */
        v = lprimary(x, live);
        return (v.type == VAL_LONG) ? num_l((long) (0UL - (unsigned long) v.lval)) : num_d(-v.dval);
    case TOK_OP_BNOT:
        v = lprimary(x, live);
        return num_l(!live ? 0 : ~((v.type == VAL_LONG) ? v.lval : (long) v.dval));
    case TOK_OP_NOT:
        v = lprimary(x, live);
        return num_l(num(&v) == 0.0);
    case TOK_LPAREN:
        v = lexpr(x, 0, live);
        expect(x, TOK_RPAREN);
        return v;
    case TOK_FUNC_D_D:
    case TOK_FUNC_DD_D:
    case TOK_FUNC_DDD_D:
    {
        double a[3] = { 0.0, 0.0, 0.0 };
        for (int i = 0; i < k->fn->nargs && !x->bail; i++)
        {
            if (i > 0)
            {
                expect(x, TOK_COMMA);
            }
            v    = lexpr(x, 0, live);
            a[i] = num(&v);
        }
        expect(x, TOK_RPAREN);
        if (!live || x->bail)
        {
            return num_d(0.0);
        }
        return num_d((k->fn->nargs == 1) ? ((double (*)(double)) k->fn->fn)(a[0])
                     : (k->fn->nargs == 2)
                         ? ((double (*)(double, double)) k->fn->fn)(a[0], a[1])
                         : ((double (*)(double, double, double)) k->fn->fn)(a[0], a[1], a[2]));
    }
    default:
        x->bail = 1;
        return num_l(0);
    }
}

/**
 * lexpr - Pratt precedence-climbing evaluator for the in-shell local calculator
 * @x:        Local lexer/evaluator state
 * @min_prec: Minimum operator binding power to consume
 * @live:     1 to evaluate, 0 to parse only
 *
 * Return: Evaluated scalar value
 */
static val_t lexpr(lexp_t *x, int min_prec, int live)
{
    val_t left = lprimary(x, live);

    while (!x->bail)
    {
        cli_token_type op   = x->t[x->pos].type;
        int            prec = cli_calc_op_prec(op);
        if (prec < min_prec)
        {
            break;
        }
        x->pos++;
        if (op == TOK_OP_QUESTION) /* only the chosen branch is evaluated */
        {
            int   c  = live && num(&left) != 0.0;
            val_t tv = lexpr(x, 0, c);
            expect(x, TOK_OP_COLON);
            val_t fv = lexpr(x, 0, live && !c);
            left     = c ? tv : fv;
            continue;
        }
        int next = cli_calc_op_right_assoc(op) ? prec : prec + 1;
        if (live && (op == TOK_OP_AND || op == TOK_OP_OR) &&
            (num(&left) != 0.0) == (op == TOK_OP_OR))
        {
            lexpr(x, next, 0); /* the result is known: the right side is not evaluated */
            left = num_l(op == TOK_OP_OR);
            continue;
        }
        val_t right = lexpr(x, next, live);
        if (live && !x->bail && cli_calc_scalar_binop(op, &left, &right, &left) != NULL)
        {
            x->bail = 1; /* the engine reports the error */
        }
    }
    return left;
}

/**
 * cli_calc_local - Evaluate a pure scalar expression in-process without contacting the server
 * @expr:   Null-terminated expression string to evaluate
 * @digits: Significant digits to use when formatting floating-point output
 * @out:    Destination buffer for the formatted numeric result string
 * @outsz:  Size of @out in bytes
 * @truth:  Output boolean truth value (1 if result is non-zero and not NaN, 0 otherwise)
 *
 * Tokenizes and evaluates @expr locally when it consists solely of numbers,
 * scalar operators, built-in math/cast functions, and numeric SHM entity
 * properties, avoiding a socket round-trip to the `milk-cli` server.
 *
 * Return: 1 if @expr was evaluated locally, 0 if the server engine must handle it
 */
int cli_calc_local(const char *expr, int digits, char *out, size_t outsz, int *truth)
{
    lexp_t x;

    if (expr == NULL || lex(expr, &x) != 0)
    {
        return 0;
    }
    val_t v = lexpr(&x, 0, 1);
    if (x.bail || x.t[x.pos].type != TOK_EOF)
    {
        return 0;
    }
    if (v.type == VAL_LONG)
    {
        *truth = (v.lval != 0);
        snprintf(out, outsz, "%ld", v.lval);
    }
    else
    {
        *truth = (v.dval != 0.0 && cli_calc_isnan(v.dval) == 0.0);
        snprintf(out, outsz, "%.*g", digits, v.dval);
    }
    return 1;
}
