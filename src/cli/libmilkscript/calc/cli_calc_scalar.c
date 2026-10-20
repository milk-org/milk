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

double cli_calc_isnan(double a)
{
    uint64_t u;
    memcpy(&u, &a, sizeof(u));
    return ((u & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL &&
            (u & 0x000fffffffffffffULL) != 0)
               ? 1.0
               : 0.0;
}

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

static double cast_f32(double a)
{
    return (double) (float) a;
}

static double cast_f64(double a)
{
    return a;
}

static double cast_u8(double a)
{
    return (double) (uint8_t) (int64_t) a;
}

static double cast_i8(double a)
{
    return (double) (int8_t) (int64_t) a;
}

static double cast_u16(double a)
{
    return (double) (uint16_t) (int64_t) a;
}

static double cast_i16(double a)
{
    return (double) (int16_t) (int64_t) a;
}

static double cast_u32(double a)
{
    return (double) (uint32_t) (int64_t) a;
}

static double cast_i32(double a)
{
    return (double) (int32_t) (int64_t) a;
}

static double cast_u64(double a)
{
    return (double) (uint64_t) (a >= 0.0 ? a : (int64_t) a);
}

static double cast_i64(double a)
{
    return (double) (int64_t) a;
}

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

int cli_calc_op_right_assoc(cli_token_type t)
{
    return (t == TOK_OP_CARET) || (t == TOK_EQUAL) || (t == TOK_OP_PLUS_EQ) ||
           (t == TOK_OP_MINUS_EQ) || (t == TOK_OP_STAR_EQ) || (t == TOK_OP_SLASH_EQ) ||
           (t == TOK_OP_QUESTION);
}

static val_t num_l(long v)
{
    val_t r;
    r.type    = VAL_LONG;
    r.lval    = v;
    r.dval    = 0.0;
    r.sval[0] = '\0';
    return r;
}

static val_t num_d(double v)
{
    val_t r;
    r.type    = VAL_DOUBLE;
    r.lval    = 0;
    r.dval    = v;
    r.sval[0] = '\0';
    return r;
}

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
