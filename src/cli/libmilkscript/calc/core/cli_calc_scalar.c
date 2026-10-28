// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_scalar.c
 * @brief The calculator's scalar rules, operators, number scanner, and math function table.
 *
 * libc and libm only: this file is part of the thin bash client.
 */

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStruct.h"
#include "cli_calc_scalar.h"

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

/**
 * cli_calc_posi - Evaluate the Heaviside step function
 * @a: Input value
 *
 * Return: 1.0 if @a > 0, else 0.0
 */
double cli_calc_posi(double a)
{
    return (a > 0.0) ? 1.0 : 0.0;
}

/**
 * cli_calc_sign - Evaluate the signum function
 * @a: Input value
 *
 * Return: 1.0 if @a > 0, -1.0 if @a < 0, else 0.0
 */
double cli_calc_sign(double a)
{
    return (double) ((a > 0.0) - (a < 0.0));
}

/**
 * clamp - Restrict a scalar value to the closed interval [@lo, @hi]
 * @a:  Value to clamp
 * @lo: Lower bound
 * @hi: Upper bound
 *
 * Return: @a clamped to [@lo, @hi]
 */
static double clamp(double a, double lo, double hi)
{
    return (a > hi) ? hi : (a < lo) ? lo : a;
}

/* Explicit scalar and image datatype casts (`f32(x)` .. `i64(x)`) */
#define CALC_CAST_TYPES(X)                                                     \
    X(f32, _DATATYPE_FLOAT, (double) (float) a)                                \
    X(f64, _DATATYPE_DOUBLE, a)                                                \
    X(u8, _DATATYPE_UINT8, (double) (uint8_t) (int64_t) a)                     \
    X(i8, _DATATYPE_INT8, (double) (int8_t) (int64_t) a)                       \
    X(u16, _DATATYPE_UINT16, (double) (uint16_t) (int64_t) a)                  \
    X(i16, _DATATYPE_INT16, (double) (int16_t) (int64_t) a)                    \
    X(u32, _DATATYPE_UINT32, (double) (uint32_t) (int64_t) a)                  \
    X(i32, _DATATYPE_INT32, (double) (int32_t) (int64_t) a)                    \
    X(u64, _DATATYPE_UINT64, (double) (uint64_t) (a >= 0.0 ? a : (int64_t) a)) \
    X(i64, _DATATYPE_INT64, (double) (int64_t) a)

#define DEF_CAST_FN(NAME, DT, EXPR)     \
    static double cast_##NAME(double a) \
    {                                   \
        return (EXPR);                  \
    }
CALC_CAST_TYPES(DEF_CAST_FN)
#undef DEF_CAST_FN

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
#define CHECK_CAST_DT(NAME, DT, EXPR)     \
    if (fn == (double (*)()) cast_##NAME) \
    {                                     \
        return (DT);                      \
    }
    CALC_CAST_TYPES(CHECK_CAST_DT)
#undef CHECK_CAST_DT
    return 0;
}

static const cli_calc_scalar_fn_t scalar_fns[] = { { "sin(", 1, (double (*)()) sin },
                                                   { "cos(", 1, (double (*)()) cos },
                                                   { "exp(", 1, (double (*)()) exp },
                                                   { "exp2(", 1, (double (*)()) exp2 },
                                                   { "ln(", 1, (double (*)()) log },
                                                   { "log(", 1, (double (*)()) log10 },
                                                   { "log2(", 1, (double (*)()) log2 },
                                                   { "tan(", 1, (double (*)()) tan },
                                                   { "sinh(", 1, (double (*)()) sinh },
                                                   { "cosh(", 1, (double (*)()) cosh },
                                                   { "tanh(", 1, (double (*)()) tanh },
                                                   { "erf(", 1, (double (*)()) erf },
                                                   { "erfc(", 1, (double (*)()) erfc },
                                                   { "atan(", 1, (double (*)()) atan },
                                                   { "sqrt(", 1, (double (*)()) sqrt },
                                                   { "cbrt(", 1, (double (*)()) cbrt },
                                                   { "ceil(", 1, (double (*)()) ceil },
                                                   { "floor(", 1, (double (*)()) floor },
                                                   { "asin(", 1, (double (*)()) asin },
                                                   { "acos(", 1, (double (*)()) acos },
                                                   { "posi(", 1, (double (*)()) cli_calc_posi },
                                                   { "sign(", 1, (double (*)()) cli_calc_sign },
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
                                                   { "hypot(", 2, (double (*)()) hypot },
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
        return 0; /* below ||, reachable from a top-level parse_expr(0) */
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
 * ipow_checked - Compute integer power @b^@e with overflow detection
 * @b:   Base integer
 * @e:   Non-negative exponent
 * @out: Output result
 *
 * Return: 0 on success, -1 on 64-bit signed integer overflow
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
 * long_binop - Evaluate @a @op @b on 64-bit integers with overflow fallback to double
 * @op:  Operator token type
 * @a:   Left integer operand
 * @b:   Right integer operand
 * @out: Output scalar result
 * @err: Output static error message on modulo by zero
 *
 * Return: 1 if @op was handled as an integer binary operator, 0 otherwise
 */
int cli_calc_long_binop(cli_token_type op, long a, long b, val_t *out, const char **err)
{
    long r;
    switch (op) /* overflowing integer results fall back to double */
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
        cli_calc_long_binop(op, l->lval, r->lval, &res, &err))
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
