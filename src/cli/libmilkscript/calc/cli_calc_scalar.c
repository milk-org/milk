// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_scalar.c
 * @brief The calculator's scalar rules and the in-shell evaluator built on them (see cli_calc_scalar.h).
 *
 * libc, libm and the shm property readers only: this file is part of the thin bash client.
 */

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStruct.h"
#include "cli_calc_scalar.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"

/**
 * cli_calc_isnan - Test whether an IEEE-754 double is NaN using bit inspection
 * @a: Floating-point value to inspect
 *
 * Inspects the raw IEEE-754 exponent and mantissa bits directly so NaN detection
 * remains reliable even when compiled with `-ffast-math` / `-ffinite-math-only`.
 *
 * Return: 1.0 if @a is NaN, 0.0 otherwise
 */
double cli_calc_isnan(double a)
{
    uint64_t u;
    memcpy(&u, &a, sizeof(u));
    return ((u & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL &&
            (u & 0x000fffffffffffffULL) != 0)
               ? 1.0
               : 0.0;
}

/**
 * cli_calc_isfinite - Test whether an IEEE-754 double is finite using bit inspection
 * @a: Floating-point value to inspect
 *
 * Checks that the IEEE-754 exponent bits of @a are not all ones, avoiding
 * `-ffast-math` folding of standard `isfinite()`.
 *
 * Return: 1.0 if @a is finite, 0.0 if infinite or NaN
 */
double cli_calc_isfinite(double a)
{
    uint64_t u;
    memcpy(&u, &a, sizeof(u));
    return ((u & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL) ? 1.0 : 0.0;
}

/** posi() - Heaviside step: 1 if @a > 0, else 0. */
static double posi(double a)
{
    return (a > 0.0) ? 1.0 : 0.0;
}

/** clamp() - @a limited to [@lo, @hi] (the calculator's `trunc` / `clamp`). */
static double clamp(double a, double lo, double hi)
{
    return (a > hi) ? hi : (a < lo) ? lo : a;
}

/**
 * cast_f32 - Round a double-precision scalar to single-precision `float`
 * @a: Input value
 *
 * Implements the calculator `f32(x)` scalar conversion and serves as the
 * function pointer tag for `_DATATYPE_FLOAT` image casts.
 *
 * Return: @a converted to `float` and widened back to `double`
 */
static double cast_f32(double a)
{
    return (double) (float) a;
}

/**
 * cast_f64 - Identity conversion for double-precision `f64(x)` casts
 * @a: Input value
 *
 * Implements the calculator `f64(x)` conversion and serves as the function
 * pointer tag for `_DATATYPE_DOUBLE` image casts.
 *
 * Return: @a unchanged
 */
static double cast_f64(double a)
{
    return a;
}

/**
 * cast_u8 - Truncate a scalar to an unsigned 8-bit integer (`u8(x)`)
 * @a: Input value
 *
 * Converts @a via `int64_t` to `uint8_t` and serves as the function pointer tag
 * for `_DATATYPE_UINT8` image casts.
 *
 * Return: @a cast to `uint8_t`, widened to `double`
 */
static double cast_u8(double a)
{
    return (double) (uint8_t) (int64_t) a;
}

/**
 * cast_i8 - Truncate a scalar to a signed 8-bit integer (`i8(x)`)
 * @a: Input value
 *
 * Converts @a via `int64_t` to `int8_t` and serves as the function pointer tag
 * for `_DATATYPE_INT8` image casts.
 *
 * Return: @a cast to `int8_t`, widened to `double`
 */
static double cast_i8(double a)
{
    return (double) (int8_t) (int64_t) a;
}

/**
 * cast_u16 - Truncate a scalar to an unsigned 16-bit integer (`u16(x)`)
 * @a: Input value
 *
 * Converts @a via `int64_t` to `uint16_t` and serves as the function pointer tag
 * for `_DATATYPE_UINT16` image casts.
 *
 * Return: @a cast to `uint16_t`, widened to `double`
 */
static double cast_u16(double a)
{
    return (double) (uint16_t) (int64_t) a;
}

/**
 * cast_i16 - Truncate a scalar to a signed 16-bit integer (`i16(x)`)
 * @a: Input value
 *
 * Converts @a via `int64_t` to `int16_t` and serves as the function pointer tag
 * for `_DATATYPE_INT16` image casts.
 *
 * Return: @a cast to `int16_t`, widened to `double`
 */
static double cast_i16(double a)
{
    return (double) (int16_t) (int64_t) a;
}

/**
 * cast_u32 - Truncate a scalar to an unsigned 32-bit integer (`u32(x)`)
 * @a: Input value
 *
 * Converts @a via `int64_t` to `uint32_t` and serves as the function pointer tag
 * for `_DATATYPE_UINT32` image casts.
 *
 * Return: @a cast to `uint32_t`, widened to `double`
 */
static double cast_u32(double a)
{
    return (double) (uint32_t) (int64_t) a;
}

/**
 * cast_i32 - Truncate a scalar to a signed 32-bit integer (`i32(x)`)
 * @a: Input value
 *
 * Converts @a via `int64_t` to `int32_t` and serves as the function pointer tag
 * for `_DATATYPE_INT32` image casts.
 *
 * Return: @a cast to `int32_t`, widened to `double`
 */
static double cast_i32(double a)
{
    return (double) (int32_t) (int64_t) a;
}

/**
 * cast_u64 - Truncate a scalar to an unsigned 64-bit integer (`u64(x)`)
 * @a: Input value
 *
 * Converts @a to `uint64_t` (routing negative values through `int64_t`) and
 * serves as the function pointer tag for `_DATATYPE_UINT64` image casts.
 *
 * Return: @a cast to `uint64_t`, widened to `double`
 */
static double cast_u64(double a)
{
    return (double) (uint64_t) (a >= 0.0 ? a : (int64_t) a);
}

/**
 * cast_i64 - Truncate a scalar to a signed 64-bit integer (`i64(x)`)
 * @a: Input value
 *
 * Converts @a to `int64_t` and serves as the function pointer tag for
 * `_DATATYPE_INT64` image casts.
 *
 * Return: @a cast to `int64_t`, widened to `double`
 */
static double cast_i64(double a)
{
    return (double) (int64_t) a;
}

/**
 * cli_calc_cast_datatype - Map a scalar cast function pointer to its ImageStreamIO datatype
 * @fn: Unary scalar function pointer to inspect
 *
 * Checks whether @fn is one of `cast_f32`..`cast_i64` so image evaluators can
 * allocate an output image of the requested explicit datatype when a cast wraps
 * an image expression.
 *
 * Return: Matching `_DATATYPE_*` constant, or 0 if @fn is not a type-cast function
 */
uint8_t cli_calc_cast_datatype(double (*fn)())
{
    if (fn == (double (*)()) cast_f32)
    {
        return _DATATYPE_FLOAT;
    }
    if (fn == (double (*)()) cast_f64)
    {
        return _DATATYPE_DOUBLE;
    }
    if (fn == (double (*)()) cast_u8)
    {
        return _DATATYPE_UINT8;
    }
    if (fn == (double (*)()) cast_i8)
    {
        return _DATATYPE_INT8;
    }
    if (fn == (double (*)()) cast_u16)
    {
        return _DATATYPE_UINT16;
    }
    if (fn == (double (*)()) cast_i16)
    {
        return _DATATYPE_INT16;
    }
    if (fn == (double (*)()) cast_u32)
    {
        return _DATATYPE_UINT32;
    }
    if (fn == (double (*)()) cast_i32)
    {
        return _DATATYPE_INT32;
    }
    if (fn == (double (*)()) cast_u64)
    {
        return _DATATYPE_UINT64;
    }
    if (fn == (double (*)()) cast_i64)
    {
        return _DATATYPE_INT64;
    }
    return 0;
}

static const cli_calc_scalar_fn_t scalar_fns[] = { { "sin(", 1, (double (*)()) sin },
                                                   { "cos(", 1, (double (*)()) cos },
                                                   { "exp(", 1, (double (*)()) exp },
                                                   { "ln(", 1, (double (*)()) log },
                                                   { "log(", 1, (double (*)()) log10 },
                                                   { "tan(", 1, (double (*)()) tan },
                                                   { "atan(", 1, (double (*)()) atan },
                                                   { "sqrt(", 1, (double (*)()) sqrt },
                                                   { "cbrt(", 1, (double (*)()) cbrt },
                                                   { "ceil(", 1, (double (*)()) ceil },
                                                   { "floor(", 1, (double (*)()) floor },
                                                   { "asin(", 1, (double (*)()) asin },
                                                   { "acos(", 1, (double (*)()) acos },
                                                   { "posi(", 1, (double (*)()) posi },
                                                   { "abs(", 1, (double (*)()) fabs },
                                                   { "fabs(", 1, (double (*)()) fabs },
                                                   { "round(", 1, (double (*)()) round },
                                                   { "isnan(", 1, (double (*)()) cli_calc_isnan },
                                                   { "isfinite(", 1,
                                                     (double (*)()) cli_calc_isfinite },
                                                   { "f32(", 1, (double (*)()) cast_f32 },
                                                   { "f64(", 1, (double (*)()) cast_f64 },
                                                   { "u8(", 1, (double (*)()) cast_u8 },
                                                   { "i8(", 1, (double (*)()) cast_i8 },
                                                   { "u16(", 1, (double (*)()) cast_u16 },
                                                   { "i16(", 1, (double (*)()) cast_i16 },
                                                   { "u32(", 1, (double (*)()) cast_u32 },
                                                   { "i32(", 1, (double (*)()) cast_i32 },
                                                   { "u64(", 1, (double (*)()) cast_u64 },
                                                   { "i64(", 1, (double (*)()) cast_i64 },
                                                   { "atan2(", 2, (double (*)()) atan2 },
                                                   { "fmod(", 2, (double (*)()) fmod },
                                                   { "min(", 2, (double (*)()) fmin },
                                                   { "max(", 2, (double (*)()) fmax },
                                                   { "trunc(", 3, (double (*)()) clamp },
                                                   { "clamp(", 3, (double (*)()) clamp },
                                                   { NULL, 0, NULL } };

/**
 * cli_calc_scalar_fn - Match a built-in scalar math or cast function call prefix
 * @p: Source pointer to test against `<name>(` prefixes
 *
 * Looks up @p in the shared `scalar_fns` table used by both the main tokenizer
 * and the lightweight in-shell evaluator.
 *
 * Return: Pointer to the matching `cli_calc_scalar_fn_t` descriptor, or NULL
 */
const cli_calc_scalar_fn_t *cli_calc_scalar_fn(const char *p)
{
    for (int i = 0; scalar_fns[i].name != NULL; i++)
    {
        if (strncmp(p, scalar_fns[i].name, strlen(scalar_fns[i].name)) == 0)
        {
            return &scalar_fns[i];
        }
    }
    return NULL;
}

/**
 * cli_calc_op_prec - Return the Pratt parser precedence level for an operator token
 * @t: Operator token type
 *
 * Defines the unified operator precedence hierarchy shared across the main
 * calculator, fused image compiler, and in-shell scalar evaluator.
 *
 * Return: Non-negative precedence level, or -1 if @t is not an infix/ternary operator
 */
int cli_calc_op_prec(cli_token_type t)
{
    switch (t)
    {
    case TOK_OP_QUESTION:
        return 0; // below ||, reachable from a top-level parse_expr(0)
    case TOK_OP_OR:
        return 1;
    case TOK_OP_AND:
        return 2;
    case TOK_OP_BOR:
        return 3;
    case TOK_OP_BXOR:
        return 4;
    case TOK_OP_BAND:
        return 5;
    case TOK_OP_EQ:
    case TOK_OP_NEQ:
        return 6;
    case TOK_OP_LT:
    case TOK_OP_LE:
    case TOK_OP_GT:
    case TOK_OP_GE:
        return 7;
    case TOK_OP_LSHIFT:
    case TOK_OP_RSHIFT:
        return 8;
    case TOK_OP_PLUS:
    case TOK_OP_MINUS:
        return 9;
    case TOK_OP_STAR:
    case TOK_OP_SLASH:
    case TOK_OP_MOD:
        return 10;
    case TOK_OP_CARET:
        return 11;
    case TOK_EQUAL:
    case TOK_OP_PLUS_EQ:
    case TOK_OP_MINUS_EQ:
    case TOK_OP_STAR_EQ:
    case TOK_OP_SLASH_EQ:
        return 0;
    default:
        return -1;
    }
}

/**
 * cli_calc_op_right_assoc - Check whether an operator token is right-associative
 * @t: Operator token type
 *
 * Identifies exponentiation (`^`), assignments (`=`, `op=`), and ternary (`?`)
 * so precedence climbing uses `prec` instead of `prec + 1` for the right-hand
 * recursive call.
 *
 * Return: 1 if @t is right-associative, 0 otherwise
 */
int cli_calc_op_right_assoc(cli_token_type t)
{
    return (t == TOK_OP_CARET) || (t == TOK_EQUAL) || (t == TOK_OP_PLUS_EQ) ||
           (t == TOK_OP_MINUS_EQ) || (t == TOK_OP_STAR_EQ) || (t == TOK_OP_SLASH_EQ) ||
           (t == TOK_OP_QUESTION);
}

/**
 * num_l - Construct a `VAL_LONG` scalar value without pulling in CLIcore symbols
 * @v: Signed integer value
 *
 * Local `val_t` constructor used in `cli_calc_scalar.c` so the thin bash client
 * does not depend on `cli_calc_parser.c`.
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
 * Local `val_t` constructor used in `cli_calc_scalar.c` so the thin bash client
 * does not depend on `cli_calc_parser.c`.
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
 * Promotes `VAL_LONG` to `double` or returns `v->dval` directly.
 *
 * Return: Numeric value of @v as a `double`
 */
static double num(const val_t *v)
{
    return (v->type == VAL_LONG) ? (double) v->lval : v->dval;
}

/** ipow_checked() - @b^@e for @e >= 0; returns 0 and sets *@out, or -1 on overflow. */
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

/** long_binop() - @a @op @b on longs, as a long (or a double when it overflows); 0 if @op is not one. */
static int long_binop(cli_token_type op, long a, long b, val_t *out, const char **err)
{
    long r;
    switch (op) // overflowing integer results fall back to double
    {
    case TOK_OP_PLUS:
        *out = __builtin_add_overflow(a, b, &r) ? num_d((double) a + (double) b) : num_l(r);
        return 1;
    case TOK_OP_MINUS:
        *out = __builtin_sub_overflow(a, b, &r) ? num_d((double) a - (double) b) : num_l(r);
        return 1;
    case TOK_OP_STAR:
        *out = __builtin_mul_overflow(a, b, &r) ? num_d((double) a * (double) b) : num_l(r);
        return 1;
    case TOK_OP_MOD:
        *out = num_l((b == 0 || (a == LONG_MIN && b == -1)) ? 0 : a % b);
        *err = (b == 0) ? "Modulo by zero" : NULL;
        return 1;
    case TOK_OP_CARET:
        *out =
            (b < 0 || ipow_checked(a, b, &r) != 0) ? num_d(pow((double) a, (double) b)) : num_l(r);
        return 1;
    case TOK_OP_LT:
        *out = num_l(a < b);
        return 1;
    case TOK_OP_LE:
        *out = num_l(a <= b);
        return 1;
    case TOK_OP_GT:
        *out = num_l(a > b);
        return 1;
    case TOK_OP_GE:
        *out = num_l(a >= b);
        return 1;
    case TOK_OP_EQ:
        *out = num_l(a == b);
        return 1;
    case TOK_OP_NEQ:
        *out = num_l(a != b);
        return 1;
    case TOK_OP_AND:
        *out = num_l(a && b);
        return 1;
    case TOK_OP_OR:
        *out = num_l(a || b);
        return 1;
    case TOK_OP_BAND:
        *out = num_l(a & b);
        return 1;
    case TOK_OP_BOR:
        *out = num_l(a | b);
        return 1;
    case TOK_OP_BXOR:
        *out = num_l(a ^ b);
        return 1;
    case TOK_OP_LSHIFT:
        *out = num_l((b < 0 || b >= 64) ? 0 : a << b);
        return 1;
    case TOK_OP_RSHIFT:
        *out = num_l((b < 0 || b >= 64) ? 0 : a >> b);
        return 1;
    default:
        return 0;
    }
}

/**
 * cli_calc_scalar_binop - Evaluate a binary operator on two numeric scalar values
 * @op:  Operator token type
 * @l:   Left numeric operand (`VAL_LONG` or `VAL_DOUBLE`)
 * @r:   Right numeric operand (`VAL_LONG` or `VAL_DOUBLE`)
 * @out: Output scalar result
 *
 * Preserves exact 64-bit integer arithmetic when both operands are `VAL_LONG`
 * (falling back to `double` on overflow or `/` division) and otherwise computes
 * in double precision.
 *
 * Return: NULL on success, or a static error string (e.g. division/modulo by zero)
 */
const char *cli_calc_scalar_binop(cli_token_type op, const val_t *l, const val_t *r, val_t *out)
{
    const char *err = NULL;
    val_t       res;

    /* long OP long -> long (except / -> double) */
    if (l->type == VAL_LONG && r->type == VAL_LONG && op != TOK_OP_SLASH &&
        long_binop(op, l->lval, r->lval, &res, &err))
    {
        *out = res;
        return err;
    }

    /* any numeric mix -> double */
    double lv = num(l);
    double rv = num(r);
    switch (op)
    {
    case TOK_OP_PLUS:
        res = num_d(lv + rv);
        break;
    case TOK_OP_MINUS:
        res = num_d(lv - rv);
        break;
    case TOK_OP_STAR:
        res = num_d(lv * rv);
        break;
    case TOK_OP_SLASH:
        res = num_d((rv == 0.0) ? 0.0 : lv / rv);
        err = (rv == 0.0) ? "Division by zero" : NULL;
        break;
    case TOK_OP_MOD:
        res = num_d((rv == 0.0) ? 0.0 : fmod(lv, rv));
        err = (rv == 0.0) ? "Modulo by zero" : NULL;
        break;
    case TOK_OP_CARET:
        res = num_d(pow(lv, rv));
        break;
    case TOK_OP_LT:
        res = num_l(lv < rv);
        break;
    case TOK_OP_LE:
        res = num_l(lv <= rv);
        break;
    case TOK_OP_GT:
        res = num_l(lv > rv);
        break;
    case TOK_OP_GE:
        res = num_l(lv >= rv);
        break;
    case TOK_OP_EQ:
        res = num_l(lv == rv);
        break;
    case TOK_OP_NEQ:
        res = num_l(lv != rv);
        break;
    case TOK_OP_AND:
        res = num_l(lv != 0 && rv != 0);
        break;
    case TOK_OP_OR:
        res = num_l(lv != 0 || rv != 0);
        break;
    case TOK_OP_BAND:
        res = num_l((long) lv & (long) rv);
        break;
    case TOK_OP_BOR:
        res = num_l((long) lv | (long) rv);
        break;
    case TOK_OP_BXOR:
        res = num_l((long) lv ^ (long) rv);
        break;
    case TOK_OP_LSHIFT:
    case TOK_OP_RSHIFT:
    {
        long s = (long) rv;
        res    = num_l((s < 0 || s >= 64)      ? 0
                       : (op == TOK_OP_LSHIFT) ? (long) lv << s
                                               : (long) lv >> s);
        break;
    }
    default:
        res = num_l(0);
        err = "Unknown operator";
        break;
    }
    *out = res;
    return err;
}

/**
 * cli_calc_scan_op - Scan an operator or punctuation token at the start of @p
 * @p:   Input string pointer
 * @len: Output number of characters consumed (1 or 2) when matched
 *
 * Checks two-character operators (`<=`, `>=`, `==`, `!=`, `&&`, `||`, `+=`,
 * etc.) before single-character operators and delimiters.
 *
 * Return: Matched `cli_token_type`, or `TOK_EOF` if @p does not start with an operator
 */
cli_token_type cli_calc_scan_op(const char *p, int *len)
{
    static const struct
    {
        char           s[3];
        cli_token_type t;
    } two[] = { { "<=", TOK_OP_LE },       { ">=", TOK_OP_GE },       { "==", TOK_OP_EQ },
                { "!=", TOK_OP_NEQ },      { "&&", TOK_OP_AND },      { "||", TOK_OP_OR },
                { "+=", TOK_OP_PLUS_EQ },  { "-=", TOK_OP_MINUS_EQ }, { "*=", TOK_OP_STAR_EQ },
                { "/=", TOK_OP_SLASH_EQ }, { "<<", TOK_OP_LSHIFT },   { ">>", TOK_OP_RSHIFT },
                { "^^", TOK_OP_BXOR } };
    static const char           one_s[] = "+-*/^%<>!()|&~,=?:";
    static const cli_token_type one_t[] = {
        TOK_OP_PLUS, TOK_OP_MINUS, TOK_OP_STAR, TOK_OP_SLASH, TOK_OP_CARET,    TOK_OP_MOD,
        TOK_OP_LT,   TOK_OP_GT,    TOK_OP_NOT,  TOK_LPAREN,   TOK_RPAREN,      TOK_OP_BOR,
        TOK_OP_BAND, TOK_OP_BNOT,  TOK_COMMA,   TOK_EQUAL,    TOK_OP_QUESTION, TOK_OP_COLON
    };
    const char *c;

    for (size_t i = 0; i < sizeof(two) / sizeof(two[0]); i++)
    {
        if (p[0] == two[i].s[0] && p[1] == two[i].s[1])
        {
            *len = 2;
            return two[i].t;
        }
    }
    if (*p != '\0' && (c = strchr(one_s, *p)) != NULL)
    {
        *len = 1;
        return one_t[c - one_s];
    }
    return TOK_EOF;
}

/**
 * cli_calc_scan_number - Scan an integer or floating-point literal at @p
 * @p:    Input string pointer
 * @type: Output token type (`TOK_LONG` or `TOK_DOUBLE`)
 * @l:    Output integer value when `@type == TOK_LONG`
 * @d:    Output floating-point value when `@type == TOK_DOUBLE`
 *
 * Parses decimal, hexadecimal (`0x`), octal (`0o`), and binary (`0b`) integers
 * as well as decimal/scientific floating-point literals.
 *
 * Return: Number of bytes consumed from @p, or 0 if no numeric literal starts at @p
 */
size_t cli_calc_scan_number(const char *p, cli_token_type *type, long *l, double *d)
{
    const char *start    = p;
    char       *end      = NULL;
    int         is_float = 0;

    if (!(isdigit((unsigned char) *p) || (*p == '.' && isdigit((unsigned char) p[1]))))
    {
        return 0;
    }
    if (*p == '0' && p[1] != '\0' && !isdigit((unsigned char) p[1]) && p[1] != '.')
    {
        int base = (p[1] == 'x' || p[1] == 'X')   ? 16
                   : (p[1] == 'o' || p[1] == 'O') ? 8
                   : (p[1] == 'b' || p[1] == 'B') ? 2
                                                  : 0;
        if (base != 0) /* 0x... is read whole by strtol; 0o and 0b are skipped first */
        {
            *type = TOK_LONG;
            *l    = strtol((base == 16) ? p : p + 2, &end, base);
            return (size_t) (end - start);
        }
    }
    while (isdigit((unsigned char) *p))
    {
        p++;
    }
    if (*p == '.')
    {
        is_float = 1;
        p++;
        while (isdigit((unsigned char) *p))
        {
            p++;
        }
    }
    if (*p == 'e' || *p == 'E')
    {
        is_float = 1;
        p++;
        p += (*p == '+' || *p == '-');
        while (isdigit((unsigned char) *p))
        {
            p++;
        }
    }
    *type = is_float ? TOK_DOUBLE : TOK_LONG;
    if (is_float)
    {
        *d = strtod(start, NULL);
    }
    else
    {
        *l = strtol(start, NULL, 10);
    }
    return (size_t) (p - start);
}

/* ------------------------------------------------------------------ in-shell evaluator */

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

/** lex_entity() - the stream/FPS/process/sequencer property at @p as number tokens (what the server's
 * expansion substitutes); its length, or 0 if it is no such property or not a number. */
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

/** lex() - tokenize @s into @x; -1 if it holds anything the engine must handle. */
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

/** expect() - step over token @t, or mark @x for the engine. */
static void expect(lexp_t *x, cli_token_type t)
{
    if (!x->bail && x->t[x->pos].type != t)
    {
        x->bail = 1;
    }
    x->pos += !x->bail;
}

/** lprimary() - one atom, as the engine's parse_primary(); @live: evaluate (else only parse). */
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

/** lexpr() - an expression of binding power >= @min_prec, as the engine's parse_expr(). */
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
