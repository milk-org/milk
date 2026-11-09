// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_scalar_ops.c
 * @brief Scalar arithmetic/string kernels and zero-heap `mcalc_local()` fast path.
 *
 * This module serves two closely related roles in `libmilkcalc`:
 *
 * 1. **Shared Scalar Arithmetic & String Operations**:
 *    Provides overflow-checked 64-bit integer exponentiation (`mc_scalar_pow`),
 *    hybrid integer/double binary operator evaluation with automatic float64
 *    promotion on signed overflow (`mc_scalar_binop`), value-to-string
 *    formatting (`mc_val_to_str`), and scalar string functions (`mc_scalar_str_fn`:
 *    `strlen`, `toupper`, `tolower`, `substr`, `replace`, `hex`, `oct`, `bin`).
 *
 * 2. **Zero-Allocation `mcalc_local()` Fast Path**:
 *    Evaluates pure scalar arithmetic expressions (including constants `pi`,
 *    `nan`, `inf`, `in_local` math functions, and shared-memory entity
 *    properties `s.<stream>.<prop>`, `f.<fps>.<prop>`, `p.<proc>.<prop>`)
 *    entirely on the stack (`mc_lexp_t`) without allocating an AST or touching
 *    `milk_data`. When an expression references calculator variables, images,
 *    or assignments, `lex_local()` returns -1 so the caller seamlessly falls
 *    back to the full AST pipeline.
 */

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "front/mc_ast.h"
#include "mcalc.h"
#include "mcalc_host.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

/* ========================================================================= */
/* Scalar Value Constructors and Integer Exponentiation                      */
/* ========================================================================= */

/** mk_l() - Construct an `MC_VAL_LONG` scalar value holding @v. */
static mc_val_t mk_l(long v)
{
    mc_val_t r;
    r.type    = MC_VAL_LONG;
    r.lval    = v;
    r.dval    = 0.0;
    r.sval[0] = '\0';
    return r;
}

/** mk_d() - Construct an `MC_VAL_DOUBLE` scalar value holding @v. */
static mc_val_t mk_d(double v)
{
    mc_val_t r;
    r.type    = MC_VAL_DOUBLE;
    r.lval    = 0;
    r.dval    = v;
    r.sval[0] = '\0';
    return r;
}

/** val_num() - Extract numeric value of @v as `double`. */
static inline double val_num(const mc_val_t *v)
{
    return (v->type == MC_VAL_LONG) ? (double) v->lval : v->dval;
}

/**
 * ipow_checked - Compute integer power `b^e` with signed 64-bit overflow detection
 * @b:   Base integer
 * @e:   Non-negative exponent (`@e >= 0`)
 * @out: Output pointer receiving exact `b^e` when no overflow occurs
 *
 * Uses binary exponentiation by squaring coupled with `__builtin_mul_overflow`
 * so that any 64-bit signed overflow is caught before undefined behavior occurs.
 *
 * Return: 0 on exact fit in `long`, -1 on signed 64-bit overflow
 */
static int ipow_checked(long b, long e, long *out)
{
    long r = 1;
    while (e > 0)
    {
        if ((e & 1) && __builtin_mul_overflow(r, b, &r))
        {
            return -1;
        }
        e >>= 1;
        if (e > 0 && __builtin_mul_overflow(b, b, &b))
        {
            return -1;
        }
    }
    *out = r;
    return 0;
}

/**
 * mc_scalar_pow - Evaluate `pow(@a, @b)` preserving exact 64-bit integers when possible
 * @a:   Base scalar value
 * @b:   Exponent scalar value
 * @out: Output scalar result (`MC_VAL_LONG` if exact integer fit, else `MC_VAL_DOUBLE`)
 *
 * When both @a and @b are `MC_VAL_LONG` with `@b->lval >= 0` and `@a^@b` fits in
 * a signed 64-bit `long`, returns an exact `MC_VAL_LONG`; otherwise promotes
 * both operands to `double` and calls `pow()`.
 */
void mc_scalar_pow(const mc_val_t *a, const mc_val_t *b, mc_val_t *out)
{
    long r = 0;
    if (a->type == MC_VAL_LONG && b->type == MC_VAL_LONG && b->lval >= 0 &&
        ipow_checked(a->lval, b->lval, &r) == 0)
    {
        *out = mk_l(r);
        return;
    }
    *out = mk_d(pow(val_num(a), val_num(b)));
}

/* ========================================================================= */
/* Binary Scalar Operators and String Functions                              */
/* ========================================================================= */

/**
 * long_binop - Evaluate a binary operator on two `long` operands with overflow fallback
 * @op:  Binary operator token (`MC_TOK_PLUS`, `MC_TOK_SHL`, etc.)
 * @a:   Left integer operand
 * @b:   Right integer operand
 * @out: Output scalar value (`MC_VAL_LONG` or promoted `MC_VAL_DOUBLE` on overflow)
 * @err: Output static error message string (set on modulo by zero, else NULL)
 *
 * Arithmetic (`+`, `-`, `*`) uses compiler overflow builtins and transparently
 * promotes to `MC_VAL_DOUBLE` on 64-bit signed overflow. Shift counts outside
 * `[0, 63]` and `LONG_MIN % -1` are guarded against undefined behavior.
 *
 * Return: 1 if @op was handled as an integer operation, 0 if caller should use float64
 */
static int long_binop(mc_tok_type_t op, long a, long b, mc_val_t *out, const char **err)
{
    long r;
    switch (op)
    {
    case MC_TOK_PLUS:
        *out = __builtin_add_overflow(a, b, &r) ? mk_d((double) a + (double) b) : mk_l(r);
        return 1;
    case MC_TOK_MINUS:
        *out = __builtin_sub_overflow(a, b, &r) ? mk_d((double) a - (double) b) : mk_l(r);
        return 1;
    case MC_TOK_MUL:
        *out = __builtin_mul_overflow(a, b, &r) ? mk_d((double) a * (double) b) : mk_l(r);
        return 1;
    case MC_TOK_MOD:
        *out = mk_l((b == 0 || (a == LONG_MIN && b == -1)) ? 0 : a % b);
        *err = (b == 0) ? "Modulo by zero" : NULL;
        return 1;
    case MC_TOK_LT:
        *out = mk_l(a < b);
        return 1;
    case MC_TOK_LE:
        *out = mk_l(a <= b);
        return 1;
    case MC_TOK_GT:
        *out = mk_l(a > b);
        return 1;
    case MC_TOK_GE:
        *out = mk_l(a >= b);
        return 1;
    case MC_TOK_EQ:
        *out = mk_l(a == b);
        return 1;
    case MC_TOK_NE:
        *out = mk_l(a != b);
        return 1;
    case MC_TOK_AND:
        *out = mk_l(a && b);
        return 1;
    case MC_TOK_OR:
        *out = mk_l(a || b);
        return 1;
    case MC_TOK_BAND:
        *out = mk_l(a & b);
        return 1;
    case MC_TOK_BOR:
        *out = mk_l(a | b);
        return 1;
    case MC_TOK_BXOR:
        *out = mk_l(a ^ b);
        return 1;
    case MC_TOK_SHL:
        *out = mk_l((b < 0 || b >= 64) ? 0 : a << b);
        return 1;
    case MC_TOK_SHR:
        *out = mk_l((b < 0 || b >= 64) ? 0 : a >> b);
        return 1;
    default:
        return 0;
    }
}

/**
 * mc_scalar_binop - Apply binary operator @op to numeric scalar values @l and @r
 * @op:  Binary operator token
 * @l:   Left numeric operand
 * @r:   Right numeric operand
 * @out: Output scalar value
 *
 * Uses exact 64-bit integer arithmetic via `long_binop()` when both operands
 * are `MC_VAL_LONG` (except `/` which always performs floating-point division),
 * and falls back to `double` arithmetic otherwise.
 *
 * Return: NULL on success, or a static error string (e.g. `"Division by zero"`)
 */
const char *mc_scalar_binop(mc_tok_type_t op, const mc_val_t *l, const mc_val_t *r, mc_val_t *out)
{
    const char *err = NULL;
    mc_val_t    res;

    if (l->type == MC_VAL_LONG && r->type == MC_VAL_LONG && op != MC_TOK_DIV &&
        long_binop(op, l->lval, r->lval, &res, &err))
    {
        *out = res;
        return err;
    }

    double lv = val_num(l);
    double rv = val_num(r);
    switch (op)
    {
    case MC_TOK_PLUS:
        res = mk_d(lv + rv);
        break;
    case MC_TOK_MINUS:
        res = mk_d(lv - rv);
        break;
    case MC_TOK_MUL:
        res = mk_d(lv * rv);
        break;
    case MC_TOK_DIV:
        res = mk_d((rv == 0.0) ? 0.0 : lv / rv);
        err = (rv == 0.0) ? "Division by zero" : NULL;
        break;
    case MC_TOK_MOD:
        res = mk_d((rv == 0.0) ? 0.0 : mc_fmodd(lv, rv));
        err = (rv == 0.0) ? "Modulo by zero" : NULL;
        break;
    case MC_TOK_LT:
        res = mk_l(lv < rv);
        break;
    case MC_TOK_LE:
        res = mk_l(lv <= rv);
        break;
    case MC_TOK_GT:
        res = mk_l(lv > rv);
        break;
    case MC_TOK_GE:
        res = mk_l(lv >= rv);
        break;
    case MC_TOK_EQ:
        res = mk_l(lv == rv);
        break;
    case MC_TOK_NE:
        res = mk_l(lv != rv);
        break;
    case MC_TOK_AND:
        res = mk_l(lv != 0.0 && rv != 0.0);
        break;
    case MC_TOK_OR:
        res = mk_l(lv != 0.0 || rv != 0.0);
        break;
    case MC_TOK_BAND:
        res = mk_l((long) lv & (long) rv);
        break;
    case MC_TOK_BOR:
        res = mk_l((long) lv | (long) rv);
        break;
    case MC_TOK_BXOR:
        res = mk_l((long) lv ^ (long) rv);
        break;
    case MC_TOK_SHL:
    case MC_TOK_SHR:
    {
        long s = (long) rv;
        res = mk_l((s < 0 || s >= 64) ? 0 : (op == MC_TOK_SHL) ? (long) lv << s : (long) lv >> s);
        break;
    }
    default:
        res = mk_l(0);
        err = "Unknown operator";
        break;
    }
    *out = res;
    return err;
}

/**
 * mc_val_to_str - Format a runtime `mc_val_t` into a null-terminated string buffer
 * @v:                Scalar or string value to format
 * @buf:              Destination string buffer
 * @bufsz:            Capacity of @buf in bytes
 * @use_float_digits: Non-zero to format doubles with `mcalc_float_digits` precision
 */
void mc_val_to_str(const mc_val_t *v, char *buf, size_t bufsz, int use_float_digits)
{
    if (v->type == MC_VAL_STRING)
    {
        snprintf(buf, bufsz, "%s", v->sval);
    }
    else if (v->type == MC_VAL_LONG)
    {
        snprintf(buf, bufsz, "%ld", v->lval);
    }
    else if (use_float_digits)
    {
        snprintf(buf, bufsz, "%.*g", mcalc_float_digits, v->dval);
    }
    else
    {
        snprintf(buf, bufsz, "%g", v->dval);
    }
}

/**
 * mc_scalar_str_fn - Execute a built-in string or base-formatting function
 * @fn:   Built-in function descriptor (`MC_FN_STR`)
 * @args: Array of evaluated argument values
 * @out:  Output value (`MC_VAL_LONG` for `strlen`, `MC_VAL_STRING` for all others)
 *
 * Implements `strlen(s)`, `toupper(s)`, `tolower(s)`, `substr(s, off, len)`,
 * `replace(s, old, new)`, `hex(n)`, `oct(n)`, and `bin(n)`.
 *
 * Return: NULL on success, or a static error description on invalid argument types
 */
const char *mc_scalar_str_fn(const mc_func_t *fn, const mc_val_t *args, mc_val_t *out)
{
    if (strcmp(fn->name, "strlen") == 0)
    {
        char buf[MC_TOK_STR_MAX];
        mc_val_to_str(&args[0], buf, sizeof(buf), 0);
        *out = mk_l((long) strlen(buf));
        return NULL;
    }
    if (strcmp(fn->name, "toupper") == 0 || strcmp(fn->name, "tolower") == 0)
    {
        mc_val_to_str(&args[0], out->sval, sizeof(out->sval), 0);
        int is_up = (fn->name[2] == 'u');
        for (int i = 0; out->sval[i] != '\0'; i++)
        {
            out->sval[i] = is_up ? (char) toupper((unsigned char) out->sval[i])
                                 : (char) tolower((unsigned char) out->sval[i]);
        }
        out->type = MC_VAL_STRING;
        return NULL;
    }
    if (strcmp(fn->name, "substr") == 0)
    {
        if (args[0].type != MC_VAL_STRING)
        {
            return "substr: first arg must be string";
        }
        char sv[MC_TOK_STR_MAX];
        mc_val_to_str(&args[0], sv, sizeof(sv), 0);
        int slen = (int) strlen(sv);
        int off  = (int) val_num(&args[1]);
        int len  = (int) val_num(&args[2]);
        off      = (off < 0) ? 0 : (off > slen) ? slen : off;
        len      = (len < 0) ? 0 : (off + len > slen) ? (slen - off) : len;
        memcpy(out->sval, sv + off, (size_t) len);
        out->sval[len] = '\0';
        out->type      = MC_VAL_STRING;
        return NULL;
    }
    if (strcmp(fn->name, "replace") == 0)
    {
        if (args[0].type != MC_VAL_STRING || args[1].type != MC_VAL_STRING ||
            args[2].type != MC_VAL_STRING)
        {
            return "replace: all args must be strings";
        }
        char s0[MC_TOK_STR_MAX];
        char s1[MC_TOK_STR_MAX];
        char s2[MC_TOK_STR_MAX];
        mc_val_to_str(&args[0], s0, sizeof(s0), 0);
        mc_val_to_str(&args[1], s1, sizeof(s1), 0);
        mc_val_to_str(&args[2], s2, sizeof(s2), 0);
        out->type = MC_VAL_STRING;
        int s1len = (int) strlen(s1);
        if (s1len == 0)
        {
            snprintf(out->sval, sizeof(out->sval), "%s", s0);
            return NULL;
        }
        const char *p    = s0;
        char       *w    = out->sval;
        char       *wend = out->sval + sizeof(out->sval) - 1;
        int         rlen = (int) strlen(s2);
        while (*p != '\0' && w < wend)
        {
            if (strncmp(p, s1, (size_t) s1len) == 0)
            {
                if (w + rlen < wend)
                {
                    memcpy(w, s2, (size_t) rlen);
                    w += rlen;
                }
                p += s1len;
            }
            else
            {
                *w++ = *p++;
            }
        }
        *w = '\0';
        return NULL;
    }

    /* Base-formatting functions: hex(n), oct(n), bin(n) */
    long iv   = (args[0].type == MC_VAL_LONG) ? args[0].lval : (long) val_num(&args[0]);
    out->type = MC_VAL_STRING;
    if (strcmp(fn->name, "hex") == 0 || strcmp(fn->name, "oct") == 0)
    {
        snprintf(out->sval, sizeof(out->sval), (fn->name[0] == 'h') ? "0x%lx" : "0o%lo", iv);
        return NULL;
    }
    char *w = out->sval;
    *w++    = '0';
    *w++    = 'b';
    if (iv == 0)
    {
        *w++ = '0';
    }
    else
    {
        unsigned long uv = (unsigned long) iv;
        char          bits[65];
        int           bi = 0;
        while (uv > 0 && bi < 64)
        {
            bits[bi++] = (uv & 1UL) ? '1' : '0';
            uv >>= 1;
        }
        for (int i = bi - 1; i >= 0; i--)
        {
            *w++ = bits[i];
        }
    }
    *w = '\0';
    return NULL;
}

/* ========================================================================= */
/* Zero-Allocation Stack Evaluator (mcalc_local)                             */
/* ========================================================================= */

#define MC_LOCAL_MAX_TOK 128

/**
 * struct mc_ltok_t - Compact stack token for `mcalc_local()`
 * @type: Token type
 * @l:    Integer literal value when `@type == MC_TOK_LONG`
 * @d:    Floating-point value when `@type == MC_TOK_DOUBLE`
 * @fn:   Resolved `in_local` function descriptor when `@type == MC_TOK_IDENT`
 */
typedef struct
{
    mc_tok_type_t    type;
    long             l;
    double           d;
    const mc_func_t *fn;
} mc_ltok_t;

/**
 * struct mc_lexp_t - Stack-allocated token stream and parser state for `mcalc_local()`
 * @t:    Fixed-capacity array of compact tokens
 * @n:    Number of tokens pushed into @t
 * @pos:  Current consumption cursor during recursive evaluation
 * @bail: Non-zero if syntax error or unsupported construct encountered
 */
typedef struct
{
    mc_ltok_t t[MC_LOCAL_MAX_TOK];
    int       n;
    int       pos;
    int       bail;
} mc_lexp_t;

/**
 * lpush - Append one compact token to a stack `mc_lexp_t` stream
 * @x:    Target local token stream
 * @type: Token type
 * @l:    Integer payload
 * @d:    Double payload
 * @fn:   Function pointer payload (for `MC_TOK_IDENT`)
 *
 * Return: 0 on success, -1 if `MC_LOCAL_MAX_TOK` capacity is reached
 */
static int lpush(mc_lexp_t *x, mc_tok_type_t type, long l, double d, const mc_func_t *fn)
{
    if (x->n >= MC_LOCAL_MAX_TOK - 1)
    {
        return -1;
    }
    x->t[x->n++] = (mc_ltok_t) { type, l, d, fn };
    return 0;
}

/**
 * scan_local_op - Match an operator or punctuation token in `mcalc_local()`
 * @p:   Input pointer positioned at candidate operator
 * @len: Output byte length of matched token (1 or 2)
 *
 * Return: Matched `mc_tok_type_t`, or `MC_TOK_EOF` if not a local operator
 */
static mc_tok_type_t scan_local_op(const char *p, int *len)
{
    static const struct
    {
        char          s[3];
        mc_tok_type_t t;
    } two[]                            = { { "<=", MC_TOK_LE },         { ">=", MC_TOK_GE },
                                           { "==", MC_TOK_EQ },         { "!=", MC_TOK_NE },
                                           { "&&", MC_TOK_AND },        { "||", MC_TOK_OR },
                                           { "+=", MC_TOK_ADD_ASSIGN }, { "-=", MC_TOK_SUB_ASSIGN },
                                           { "*=", MC_TOK_MUL_ASSIGN }, { "/=", MC_TOK_DIV_ASSIGN },
                                           { "<<", MC_TOK_SHL },        { ">>", MC_TOK_SHR },
                                           { "^^", MC_TOK_BXOR } };
    static const char          one_s[] = "+-*/%<>!()|&~,=?:";
    static const mc_tok_type_t one_t[] = {
        MC_TOK_PLUS, MC_TOK_MINUS, MC_TOK_MUL,    MC_TOK_DIV,    MC_TOK_MOD,  MC_TOK_LT,
        MC_TOK_GT,   MC_TOK_NOT,   MC_TOK_LPAREN, MC_TOK_RPAREN, MC_TOK_BOR,  MC_TOK_BAND,
        MC_TOK_BNOT, MC_TOK_COMMA, MC_TOK_ASSIGN, MC_TOK_QUEST,  MC_TOK_COLON
    };

    for (size_t i = 0; i < sizeof(two) / sizeof(two[0]); i++)
    {
        if (p[0] == two[i].s[0] && p[1] == two[i].s[1])
        {
            *len = 2;
            return two[i].t;
        }
    }
    const char *c = (*p != '\0') ? strchr(one_s, *p) : NULL;
    if (c != NULL)
    {
        *len = 1;
        return one_t[c - one_s];
    }
    return MC_TOK_EOF;
}

/**
 * match_local_fn - Match a built-in function call `name(` supported by `mcalc_local()`
 * @p:      Input pointer positioned at candidate identifier
 * @out_fn: Output pointer receiving the matched `mc_func_t` descriptor
 *
 * Only functions flagged with `fn->in_local == 1` are accepted; any other
 * function returns 0 so `mcalc_local()` declines the expression and lets the
 * full evaluator handle it.
 *
 * Return: Bytes consumed including the opening `(`, or 0 if not a local function
 */
static size_t match_local_fn(const char *p, const mc_func_t **out_fn)
{
    size_t k = 0;
    while (isalnum((unsigned char) p[k]) || p[k] == '_')
    {
        k++;
    }
    if (k == 0 || p[k] != '(')
    {
        return 0;
    }
    const mc_func_t *fn = mc_func_find_len(p, k);
    if (fn == NULL || !fn->in_local)
    {
        return 0;
    }
    *out_fn = fn;
    return k + 1; /* include '(' */
}

/**
 * lex_local_entity - Resolve a numeric shared-memory property `s/f/p.<name>.<prop>` inline
 * @p: Input pointer positioned at candidate dotted entity reference
 * @x: Target local token stream receiving the resolved numeric literal token(s)
 *
 * Reads a numeric shared-memory stream/FPS/process property (`s.`, `f.`, `p.`) as the session's
 * resolver does: the same path rule (mc_scan_ident()) and typed reader (milk_shm_prop_get_num()).
 *
 * Return: Number of bytes consumed from @p, or 0 if not a numeric SHM property
 */
static size_t lex_local_entity(const char *p, mc_lexp_t *x)
{
    size_t e = mc_scan_ident(p);
    char   tok[256];
    if (e == 0 || e >= sizeof(tok))
    {
        return 0;
    }
    memcpy(tok, p, e);
    tok[e]             = '\0';
    const char    *d1  = strchr(tok, '.');
    milk_ns_kind_t k   = milk_entity_ns_kind(tok, NULL);
    double         d   = 0.0;
    long           l   = 0;
    int            isl = 0;
    if (d1 == NULL || strchr(d1 + 1, '.') == NULL || k == MILK_NS_NONE || k == MILK_NS_LOCAL ||
        milk_shm_prop_get_num(tok, &d, &l, &isl) != 0 ||
        lpush(x, isl ? MC_TOK_LONG : MC_TOK_DOUBLE, l, d, NULL) != 0)
    {
        return 0;
    }
    return e;
}

/**
 * lex_local - Tokenize a pure scalar expression into a stack `mc_lexp_t` buffer
 * @s: Null-terminated expression string
 * @x: Output stack token stream
 *
 * Rejects assignment operators, variables, image references, strings, and
 * non-`in_local` functions by returning -1 so the caller falls back to the
 * full calculator pipeline.
 *
 * Return: 0 if the entire expression was tokenized for local evaluation, -1 otherwise
 */
static int lex_local(const char *s, mc_lexp_t *x)
{
    x->n    = 0;
    x->pos  = 0;
    x->bail = 0;
    for (const char *p = s; *p != '\0';)
    {
        const mc_func_t *fn = NULL;
        mc_tok_type_t    type;
        long             l   = 0;
        double           d   = 0.0;
        int              len = 0;
        size_t           n   = 0;

        if (*p == ' ' || *p == '\t')
        {
            p++;
        }
        else if ((type = scan_local_op(p, &len)) != MC_TOK_EOF)
        {
            if (mc_op_prec(type) == 0 && type != MC_TOK_QUEST)
            {
                return -1;
            }
            if (lpush(x, type, 0, 0.0, NULL) != 0)
            {
                return -1;
            }
            p += len;
        }
        else if ((n = mc_scan_number(p, &type, &l, &d)) > 0)
        {
            if (isalnum((unsigned char) p[n]) || p[n] == '_' || p[n] == '.' ||
                lpush(x, type, l, d, NULL) != 0)
            {
                return -1;
            }
            p += n;
        }
        else if ((n = match_local_fn(p, &fn)) > 0)
        {
            if (lpush(x, MC_TOK_IDENT, 0, 0.0, fn) != 0)
            {
                return -1;
            }
            p += n;
        }
        else if ((isalpha((unsigned char) *p) || *p == '_') && (n = lex_local_entity(p, x)) > 0)
        {
            p += n;
        }
        else if (strncmp(p, "pi", 2) == 0 && !isalnum((unsigned char) p[2]) && p[2] != '_' &&
                 p[2] != '.')
        {
            if (lpush(x, MC_TOK_DOUBLE, 0, M_PI, NULL) != 0)
            {
                return -1;
            }
            p += 2;
        }
        else if ((strncmp(p, "nan", 3) == 0 || strncmp(p, "inf", 3) == 0) &&
                 !isalnum((unsigned char) p[3]) && p[3] != '_' && p[3] != '.')
        {
            uint64_t cbits = (p[0] == 'n') ? 0x7ff8000000000000ULL : 0x7ff0000000000000ULL;
            memcpy(&d, &cbits, sizeof(double));
            if (lpush(x, MC_TOK_DOUBLE, 0, d, NULL) != 0)
            {
                return -1;
            }
            p += 3;
        }
        else
        {
            return -1;
        }
    }
    x->t[x->n].type = MC_TOK_EOF;
    return (x->n > 0) ? 0 : -1;
}

static mc_val_t eval_local_expr(mc_lexp_t *x, int min_prec, int live);

/**
 * lexpect - Consume expected token @t in `mcalc_local()` or set `x->bail`
 * @x: Active local expression state
 * @t: Expected token type
 */
static void lexpect(mc_lexp_t *x, mc_tok_type_t t)
{
    if (!x->bail && x->t[x->pos].type != t)
    {
        x->bail = 1;
    }
    if (!x->bail)
    {
        x->pos++;
    }
}

/**
 * eval_local_primary - Evaluate a primary or prefix-unary operand in `mcalc_local()`
 * @x:    Active local expression state
 * @live: 1 if side effects and division-by-zero checks are active, 0 when parsing
 *        past a short-circuited branch
 *
 * Return: Evaluated `mc_val_t` (or dummy 0 when `@live == 0` or `x->bail` is set)
 */
static mc_val_t eval_local_primary(mc_lexp_t *x, int live)
{
    const mc_ltok_t *k = &x->t[x->pos];
    mc_val_t         v;

    if (x->bail || k->type == MC_TOK_EOF)
    {
        x->bail = 1;
        return mk_l(0);
    }
    x->pos++;
    switch (k->type)
    {
    case MC_TOK_LONG:
        return mk_l(k->l);
    case MC_TOK_DOUBLE:
        return mk_d(k->d);
    case MC_TOK_MINUS:
        v = eval_local_primary(x, live);
        return (v.type == MC_VAL_LONG) ? mk_l((long) (0UL - (unsigned long) v.lval))
                                       : mk_d(-v.dval);
    case MC_TOK_BNOT:
        v = eval_local_primary(x, live);
        return mk_l(!live ? 0 : ~((v.type == MC_VAL_LONG) ? v.lval : (long) v.dval));
    case MC_TOK_NOT:
        v = eval_local_primary(x, live);
        return mk_l(val_num(&v) == 0.0);
    case MC_TOK_LPAREN:
        v = eval_local_expr(x, 0, live);
        lexpect(x, MC_TOK_RPAREN);
        return v;
    case MC_TOK_IDENT:
    {
        mc_val_t va[3] = { 0 };
        double   a[3]  = { 0.0, 0.0, 0.0 };
        int      nargs = k->fn->min_args;
        for (int i = 0; i < nargs && !x->bail; i++)
        {
            if (i > 0)
            {
                lexpect(x, MC_TOK_COMMA);
            }
            va[i] = eval_local_expr(x, 0, live);
            a[i]  = val_num(&va[i]);
        }
        lexpect(x, MC_TOK_RPAREN);
        if (!live || x->bail)
        {
            return mk_d(0.0);
        }
        if (nargs == 2 && strcmp(k->fn->name, "pow") == 0)
        {
            mc_val_t r;
            mc_scalar_pow(&va[0], &va[1], &r);
            return r;
        }
        if (nargs == 1 && k->fn->fn1 != NULL)
        {
            return mk_d(k->fn->fn1(a[0]));
        }
        if (nargs == 2 && k->fn->fn2 != NULL)
        {
            return mk_d(k->fn->fn2(a[0], a[1]));
        }
        if (nargs == 3 && k->fn->fn3 != NULL)
        {
            return mk_d(k->fn->fn3(a[0], a[1], a[2]));
        }
        x->bail = 1;
        return mk_d(0.0);
    }
    default:
        x->bail = 1;
        return mk_l(0);
    }
}

/**
 * eval_local_expr - Precedence-climbing evaluator for `mcalc_local()` token streams
 * @x:        Active local expression state
 * @min_prec: Minimum operator precedence to bind at this recursion level
 * @live:     1 if evaluation is active, 0 if consuming tokens in an untaken branch
 *
 * Evaluates binary operators, chained relational comparisons (`a <= x <= b`),
 * short-circuiting `&&`/`||`, and ternary `c ? t : f` directly on the token
 * stream without building an AST.
 *
 * Return: Evaluated `mc_val_t`
 */
static mc_val_t eval_local_expr(mc_lexp_t *x, int min_prec, int live)
{
    mc_val_t left = eval_local_primary(x, live);

    while (!x->bail)
    {
        mc_tok_type_t op   = x->t[x->pos].type;
        int           prec = mc_op_prec(op);
        if (prec < min_prec)
        {
            break;
        }
        x->pos++;
        if (op == MC_TOK_QUEST)
        {
            int      c  = live && val_num(&left) != 0.0;
            mc_val_t tv = eval_local_expr(x, 0, c);
            lexpect(x, MC_TOK_COLON);
            mc_val_t fv = eval_local_expr(x, 0, live && !c);
            left        = c ? tv : fv;
            continue;
        }
        int next = (op == MC_TOK_QUEST || mc_is_assign_tok(op)) ? prec : prec + 1;
        if (live && (op == MC_TOK_AND || op == MC_TOK_OR) &&
            (val_num(&left) != 0.0) == (op == MC_TOK_OR))
        {
            eval_local_expr(x, next, 0);
            left = mk_l(op == MC_TOK_OR);
            continue;
        }
        mc_val_t right = eval_local_expr(x, next, live);
        if (mc_is_rel_tok(op) && mc_is_rel_tok(x->t[x->pos].type))
        {
            mc_val_t cmp = { 0 };
            if (live && !x->bail && mc_scalar_binop(op, &left, &right, &cmp) != NULL)
            {
                x->bail = 1;
            }
            int      all_ok = live && !x->bail && val_num(&cmp) != 0.0;
            mc_val_t prev   = right;
            while (!x->bail && mc_is_rel_tok(x->t[x->pos].type))
            {
                mc_tok_type_t cop = x->t[x->pos++].type;
                mc_val_t      cur = eval_local_expr(x, prec + 1, all_ok);
                if (all_ok && !x->bail)
                {
                    if (mc_scalar_binop(cop, &prev, &cur, &cmp) != NULL)
                    {
                        x->bail = 1;
                    }
                    all_ok = (val_num(&cmp) != 0.0);
                    prev   = cur;
                }
            }
            left = mk_l(all_ok);
            continue;
        }
        if (live && !x->bail && mc_scalar_binop(op, &left, &right, &left) != NULL)
        {
            x->bail = 1;
        }
    }
    return left;
}

/**
 * mcalc_local - Fast zero-heap evaluator for pure scalar and SHM-property expressions
 * @expr:   Null-terminated expression string to evaluate
 * @digits: Significant digits for `%.*g` floating-point formatting
 * @out:    Output string buffer receiving the formatted scalar result
 * @outsz:  Capacity of @out buffer in bytes
 * @truth:  Optional output receiving boolean truth value (1 if non-zero and non-NaN, else 0)
 *
 * Evaluates @expr on the stack without touching `milk_data` or heap memory. If
 * @expr uses variables, images, strings, assignments, or non-local functions,
 * returns 0 so the caller can fall back to the full calculator.
 *
 * Return: 1 if evaluated locally into @out, 0 if the caller must use the full calculator
 */
int mcalc_local(const char *expr, int digits, char *out, size_t outsz, int *truth)
{
    mc_lexp_t x;
    if (expr == NULL || lex_local(expr, &x) != 0)
    {
        return 0;
    }
    mc_val_t v = eval_local_expr(&x, 0, 1);
    if (x.bail || x.t[x.pos].type != MC_TOK_EOF)
    {
        return 0;
    }
    if (v.type == MC_VAL_LONG)
    {
        if (truth != NULL)
        {
            *truth = mc_truth(&v);
        }
        snprintf(out, outsz, "%ld", v.lval);
    }
    else
    {
        if (truth != NULL)
        {
            *truth = mc_truth(&v);
        }
        snprintf(out, outsz, "%.*g", digits, v.dval);
    }
    return 1;
}
