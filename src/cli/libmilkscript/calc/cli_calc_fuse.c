// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse.c
 * @brief One-pass evaluation of element-wise image expressions, and typed in-place updates.
 *
 * Role & Goal:
 *   `x = (a*2 + b) / c` used to make one `_tmpcalc` image per operator. When an expression is made
 *   only of same-shape real images, numbers, arithmetic, comparisons and per-pixel math functions,
 *   it is compiled here to a small stack program and run block by block (OpenMP over blocks), with
 *   one output image, or none when the result is assigned to an existing image. `where(c, a, b)` and
 *   `c ? a : b` with an image condition select per pixel; `ix iy ir itheta (im[, cx, cy])` compute
 *   coordinates. Anything else (slices, broadcasting, reductions, strings, `&&`, a scalar `?:`,
 *   complex images) is left to the normal evaluator, which also produces the error messages. Results
 *   follow the same type rule as the normal path: DOUBLE if any image is DOUBLE, else FLOAT.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "libmilkdata/milk_type_dispatch.h"

#define FUSE_MAXINS 128
#define FUSE_MAXIMG 16
#define FUSE_MAXSTK 8
#define FUSE_BLK 1024
#define FUSE_OMP_MIN 131072

typedef enum
{
    FI_IMG, /* push image img */
    FI_BIN, /* binary op; cpos 1: c is the left operand, 2: c is the right one, 0: two stack slots */
    FI_NEG,
    FI_NOT,
    FI_FN1,
    FI_FN2, /* as FI_BIN, with fn */
    FI_FN3, /* fn(top, c, c2) */
    FI_SEL, /* cond ? t : f; cpos bit 0: t on the stack (else c), bit 1: f on the stack (else c2) */
    FI_COORD, /* push coordinate cpos (CALC_COORD_*) of image img's pixels about (c, c2) */
    FI_SVAR,  /* scalar tape: read calculator variable vid */
    FI_CONST  /* push scalar constant c onto the vector stack */
} fi_kind;

typedef struct
{
    fi_kind        k;
    cli_token_type op;
    int            vid;
    int            s[3]; /* > 0: 1-based index in sval[]; 0: use c[] */
    double         c[3];
    double (*fn)();
} fsins;

typedef struct
{
    fi_kind        k;
    cli_token_type op;
    int            img;
    int            cpos;
    int            s1; /* > 0: 1-based index in sval[] for c */
    int            s2; /* > 0: 1-based index in sval[] for c2 */
    double         c;
    double         c2;
    double (*fn)();
} fins;

typedef struct
{
    fins         ins[FUSE_MAXINS];
    int          n;
    fsins        sins[FUSE_MAXINS];
    int          ns;
    const IMAGE *img[FUSE_MAXIMG];
    int          nimg;
    int          depth;
    int          maxdepth;
    int          nops; /* vector operations emitted */
    int          out_double;
    uint8_t      out_dt; /* > 0 when outermost op is an explicit cast f32..i64 */
} fprog;

static int keep_vars; /* compiling a kept program: variables are read at each run, not folded */

/** Compile-time value: a scalar (constant c when s1 == 0, else sins[s1 - 1]), or a vector on stk[]. */
typedef struct
{
    int    vec;
    int    s1;
    double c;
} fval;

/**
 * emit - Append a vector stack instruction to a fused image program
 * @p:      Fused program being compiled
 * @in:     Vector instruction to append
 * @ddepth: Change in vector evaluation stack depth caused by @in
 *
 * Appends @in to `p->ins`, updates current and peak stack depth (`p->maxdepth`),
 * and counts non-load vector operations so the compiler can reject programs
 * that exceed `FUSE_MAXINS` or `FUSE_MAXSTK`.
 *
 * Return: 0 on success, -1 if instruction or stack capacity is exceeded
 */
static int emit(fprog *p, fins in, int ddepth)
{
    if (p->n >= FUSE_MAXINS)
    {
        return -1;
    }
    p->ins[p->n++] = in;
    p->depth += ddepth;
    if (p->depth > p->maxdepth)
    {
        p->maxdepth = p->depth;
    }
    if (in.k != FI_IMG)
    {
        p->nops++;
    }
    return (p->maxdepth > FUSE_MAXSTK) ? -1 : 0;
}

/**
 * emit_scalar - Append a scalar tape instruction for runtime variable evaluation
 * @p:   Fused program being compiled
 * @in:  Scalar instruction (`FI_SVAR`, `FI_NEG`, `FI_NOT`, `FI_BIN`, `FI_FN*`)
 * @out: Output compile-time value referencing the 1-based slot in `p->sins`
 *
 * Records scalar sub-expressions that depend on calculator variables in kept
 * programs so their values are recomputed once per frame before the pixel loop.
 *
 * Return: 0 on success, -1 if `FUSE_MAXINS` scalar instructions are exceeded
 */
static int emit_scalar(fprog *p, fsins in, fval *out)
{
    if (p->ns >= FUSE_MAXINS)
    {
        return -1;
    }
    p->sins[p->ns++] = in;
    *out             = (fval) { .vec = 0, .s1 = p->ns, .c = 0.0 };
    return 0;
}

/**
 * is_real_datatype - Check whether an ImageStreamIO datatype is supported by the fused engine
 * @dt: ImageStreamIO `_DATATYPE_*` code
 *
 * Accepts standard integer and 32/64-bit floating-point real datatypes while
 * excluding complex and half-precision (`_DATATYPE_HALF`) images.
 *
 * Return: 1 if @dt is a supported real datatype, 0 otherwise
 */
static int is_real_datatype(uint8_t dt)
{
    return dt != _DATATYPE_COMPLEX_FLOAT && dt != _DATATYPE_COMPLEX_DOUBLE &&
           dt != _DATATYPE_HALF && ImageStreamIO_typesize(dt) > 0;
}

/**
 * same_shape - Check whether two images have identical axis count and dimensions
 * @a: First image to compare
 * @b: Second image to compare
 *
 * Verifies that @a and @b share the exact same `naxis` and `size[i]` along all
 * axes, which is required for element-wise fused evaluation without broadcasting.
 *
 * Return: 1 if @a and @b have the same shape, 0 otherwise
 */
static int same_shape(const IMAGE *a, const IMAGE *b)
{
    if (a->md[0].naxis != b->md[0].naxis)
    {
        return 0;
    }
    for (int i = 0; i < a->md[0].naxis; i++)
    {
        if (a->md[0].size[i] != b->md[0].size[i])
        {
            return 0;
        }
    }
    return 1;
}

/** add_image() - register @im as a leaf (same shape as the others, real pixels); index or -1. */
static int add_image(fprog *p, const IMAGE *im)
{
    for (int i = 0; i < p->nimg; i++)
    {
        if (p->img[i] == im)
        {
            return i;
        }
    }
    if (p->nimg >= FUSE_MAXIMG || !is_real_datatype(im->md[0].datatype) ||
        (p->nimg > 0 && !same_shape(p->img[0], im)))
    {
        return -1;
    }
    if (im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p->out_double = 1;
    }
    p->img[p->nimg] = im;
    return p->nimg++;
}

/**
 * scalar_bin - Evaluate a fusible binary operator on two double-precision scalars
 * @op: Arithmetic or comparison operator token
 * @a:  Left scalar operand
 * @b:  Right scalar operand
 *
 * Folds constant scalar sub-expressions at compile time and evaluates scalar
 * tape instructions (`FI_BIN`) at the start of each kept-program frame.
 *
 * Return: Result of `@a @op @b` as a `double`
 */
static double scalar_bin(cli_token_type op, double a, double b)
{
    switch (op)
    {
    case TOK_OP_PLUS:
        return a + b;
    case TOK_OP_MINUS:
        return a - b;
    case TOK_OP_STAR:
        return a * b;
    case TOK_OP_SLASH:
        return a / b;
    case TOK_OP_MOD:
        return fmod(a, b);
    case TOK_OP_CARET:
        return pow(a, b);
    case TOK_OP_LT:
        return a < b;
    case TOK_OP_LE:
        return a <= b;
    case TOK_OP_GT:
        return a > b;
    case TOK_OP_GE:
        return a >= b;
    case TOK_OP_EQ:
        return a == b;
    case TOK_OP_NEQ:
        return a != b;
    case TOK_OP_AND:
        return (a != 0.0) && (b != 0.0);
    default:
        return (a != 0.0) || (b != 0.0);
    }
}

/**
 * fusible_op - Test whether a binary operator token can be compiled into a fused program
 * @op: Operator token type
 *
 * Accepts arithmetic (`+`, `-`, `*`, `/`, `%`, `^`), relational/equality
 * (`<`, `<=`, `>`, `>=`, `==`, `!=`), and logical (`&&`, `||`) operators.
 *
 * Return: 1 if @op is supported by the fused engine, 0 otherwise
 */
static int fusible_op(cli_token_type op)
{
    return (op >= TOK_OP_PLUS && op <= TOK_OP_NEQ) || op == TOK_OP_AND || op == TOK_OP_OR;
}

static int c_expr(fprog *p, int min_prec, fval *out);

/** c_args() - compile @n comma-separated arguments and the closing ')'. */
static int c_args(fprog *p, int n, fval *a)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    for (int i = 0; i < n; i++)
    {
        if (i > 0)
        {
            if (cur()->type != TOK_COMMA)
            {
                return -1;
            }
            adv();
        }
        if (c_expr(p, 0, &a[i]) != 0)
        {
            return -1;
        }
    }
    if (cur()->type != TOK_RPAREN)
    {
        return -1;
    }
    adv();
    return 0;
}

/**
 * c_func - Compile a 1-, 2-, or 3-argument math or cast function call into @p
 * @p:   Fused program being compiled
 * @ft:  Function token (`TOK_FUNC_D_D`, `TOK_FUNC_DD_D`, or `TOK_FUNC_DDD_D`)
 * @out: Output compile-time value descriptor (scalar or vector)
 *
 * Folds pure scalar arguments into a constant or scalar tape instruction, and
 * emits `FI_FN1`, `FI_FN2`, or `FI_FN3` when any argument is an image vector
 * (recording explicit output cast datatypes via `cli_calc_cast_datatype()`).
 *
 * Return: 0 on success, -1 if argument types or arity are not fusible
 */
static int c_func(fprog *p, const cli_token *ft, fval *out)
{
    fval a[3]  = { { 0, 0, 0.0 }, { 0, 0, 0.0 }, { 0, 0, 0.0 } };
    int  nargs = (ft->type == TOK_FUNC_D_D) ? 1 : (ft->type == TOK_FUNC_DD_D) ? 2 : 3;
    if (c_args(p, nargs, a) != 0)
    {
        return -1;
    }
    out->vec = a[0].vec || (nargs > 1 && a[1].vec) || (nargs > 2 && a[2].vec);
    if (!out->vec)
    {
        if (a[0].s1 == 0 && (nargs < 2 || a[1].s1 == 0) && (nargs < 3 || a[2].s1 == 0))
        {
            out->s1 = 0;
            out->c =
                (nargs == 1) ? ((double (*)(double)) ft->fnctptr)(a[0].c)
                : (nargs == 2)
                    ? ((double (*)(double, double)) ft->fnctptr)(a[0].c, a[1].c)
                    : ((double (*)(double, double, double)) ft->fnctptr)(a[0].c, a[1].c, a[2].c);
            return 0;
        }
        return emit_scalar(p,
                           (fsins) { .k  = (nargs == 1)   ? FI_FN1
                                           : (nargs == 2) ? FI_FN2
                                                          : FI_FN3,
                                     .fn = ft->fnctptr,
                                     .s  = { a[0].s1, a[1].s1, a[2].s1 },
                                     .c  = { a[0].c, a[1].c, a[2].c } },
                           out);
    }
    if (ft->type == TOK_FUNC_D_D)
    {
        p->out_dt = cli_calc_cast_datatype(ft->fnctptr);
        if (p->out_dt == _DATATYPE_DOUBLE)
        {
            p->out_double = 1;
        }
        return emit(p, (fins) { .k = FI_FN1, .fn = ft->fnctptr }, 0);
    }
    p->out_dt = 0;
    if (ft->type == TOK_FUNC_DDD_D) /* trunc(im, lo, hi): only the first argument may be an image */
    {
        if (a[1].vec || a[2].vec)
        {
            return -1;
        }
        return emit(p,
                    (fins) { .k  = FI_FN3,
                             .fn = ft->fnctptr,
                             .s1 = a[1].s1,
                             .c  = a[1].c,
                             .s2 = a[2].s1,
                             .c2 = a[2].c },
                    0);
    }
    /* image forms: (im, x) any; (x, im) and (im, im) min/max/fmod/hypot/pow/atan2 */
    double (*fn)() = ft->fnctptr;
    int both       = a[0].vec && a[1].vec;
    int cmut       = (fn == (double (*)()) fmin || fn == (double (*)()) fmax ||
                      fn == (double (*)()) fmod || fn == (double (*)()) hypot);
    if ((both && !cmut && fn != (double (*)()) pow && fn != (double (*)()) atan2) ||
        (!a[0].vec && !cmut && fn != (double (*)()) atan2))
    {
        return -1;
    }
    fins in = { .k    = FI_FN2,
                .fn   = fn,
                .cpos = both       ? 0
                        : a[0].vec ? 2
                                   : 1,
                .s1   = a[0].vec ? a[1].s1 : a[0].s1,
                .c    = a[0].vec ? a[1].c : a[0].c };
    return emit(p, in, both ? -1 : 0);
}

/** c_select() - per pixel `cond ? t : f`; the condition is on the stack under t and f. */
static int c_select(fprog *p, fval t, fval f, fval *out)
{
    p->out_dt = 0;
    out->vec  = 1;
    out->s1   = 0;
    return emit(p,
                (fins) { .k    = FI_SEL,
                         .cpos = t.vec | (f.vec << 1),
                         .s1   = t.s1,
                         .c    = t.c,
                         .s2   = f.s1,
                         .c2   = f.c },
                -(t.vec + f.vec));
}

/** c_coord() - ix/iy/ir/itheta(im[, cx, cy]) or rand/randn(im[, seed]): im gives the size, not pixels. */
static int c_coord(fprog *p, const cli_token *ft, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    cli_token *t            = adv();
    fval       c[2];
    int        kind = cli_calc_coord_kind(ft);

    if (t->type != TOK_IMAGE || strchr(t->sval, '[') != NULL)
    {
        return -1;
    }
    imageID id = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
    int     k  = (id == -1) ? -1 : add_image(p, &data.core.image[id]);
    if (k < 0)
    {
        return -1;
    }
    if (kind == CALC_COORD_RAND || kind == CALC_COORD_RANDN)
    {
        c[0] = (fval) { 0, 0, 0.0 };
        c[1] = (fval) { 0, 0, 0.0 };
        if (cur()->type == TOK_COMMA)
        {
            adv();
            if (c_expr(p, 0, &c[0]) != 0 || c[0].vec || cur()->type != TOK_RPAREN)
            {
                return -1;
            }
            adv();
            c[1].c = 1.0;
        }
        else if (cur()->type == TOK_RPAREN)
        {
            adv();
        }
        else
        {
            return -1;
        }
        p->out_dt = 0;
        out->vec  = 1;
        out->s1   = 0;
        return emit(p,
                    (fins) { .k    = FI_COORD,
                             .img  = k,
                             .cpos = kind,
                             .s1   = c[0].s1,
                             .c    = c[0].c,
                             .s2   = 0,
                             .c2   = c[1].c },
                    1);
    }
    const IMAGE *im = p->img[k];
    c[0]            = (fval) { 0, 0, (double) (im->md[0].size[0] / 2) };
    c[1]            = (fval) { 0, 0, (double) ((im->md[0].naxis > 1) ? im->md[0].size[1] / 2 : 0) };
    if (cur()->type == TOK_COMMA)
    {
        adv();
        if (c_args(p, 2, c) != 0 || c[0].vec || c[1].vec)
        {
            return -1;
        }
    }
    else if (cur()->type == TOK_RPAREN)
    {
        adv();
    }
    else
    {
        return -1;
    }
    p->out_dt = 0;
    out->vec  = 1;
    out->s1   = 0;
    return emit(p,
                (fins) { .k    = FI_COORD,
                         .img  = k,
                         .cpos = kind,
                         .s1   = c[0].s1,
                         .c    = c[0].c,
                         .s2   = c[1].s1,
                         .c2   = c[1].c },
                1);
}

/**
 * c_primary - Compile a primary expression atom into a fused program
 * @p:   Fused program being compiled
 * @out: Output compile-time value descriptor
 *
 * Handles numeric literals, unary `-` and `!`, parenthesized sub-expressions,
 * scalar variables, whole-image references, math functions, `where()`, and
 * coordinate generators (`ix`, `iy`, `ir`, `itheta`), bailing out with -1 on
 * slices, assignments, or unsupported tokens.
 *
 * Return: 0 on success, -1 if the atom cannot be fused
 */
static int c_primary(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    cli_token *t            = adv();

    switch (t->type)
    {
    case TOK_LONG:
        *out = (fval) { 0, 0, (double) t->val_l };
        return 0;
    case TOK_DOUBLE:
        *out = (fval) { 0, 0, t->val_d };
        return 0;
    case TOK_OP_MINUS:
    case TOK_OP_NOT:
        if (c_primary(p, out) != 0)
        {
            return -1;
        }
        if (!out->vec)
        {
            if (out->s1 == 0)
            {
                out->c = (t->type == TOK_OP_MINUS) ? -out->c : (out->c == 0.0);
                return 0;
            }
            return emit_scalar(p,
                               (fsins) { .k = (t->type == TOK_OP_MINUS) ? FI_NEG : FI_NOT,
                                         .s = { out->s1, 0, 0 } },
                               out);
        }
        p->out_dt = 0;
        return emit(p, (fins) { .k = (t->type == TOK_OP_MINUS) ? FI_NEG : FI_NOT }, 0);
    case TOK_LPAREN:
        if (c_expr(p, 0, out) != 0 || cur()->type != TOK_RPAREN)
        {
            return -1;
        }
        adv();
        return 0;
    case TOK_VAR:
    {
        if (cur()->type >= TOK_EQUAL && cur()->type <= TOK_OP_SLASH_EQ)
        {
            return -1;
        }
        long vid = keep_vars ? variable_ID(t->sval) : -1;
        if (vid >= 0 && data.core.variable[vid].type != 2)
        {
            return emit_scalar(p, (fsins) { .k = FI_SVAR, .vid = (int) vid }, out);
        }
        val_t v = eval_existing_var_ref(t->sval);
        if (v.type != VAL_LONG && v.type != VAL_DOUBLE)
        {
            return -1;
        }
        *out = (fval) { 0, 0, to_double(v) };
        return 0;
    }
    case TOK_IMAGE:
    {
        if (strchr(t->sval, '[') != NULL ||
            (cur()->type >= TOK_EQUAL && cur()->type <= TOK_OP_SLASH_EQ))
        {
            return -1;
        }
        imageID id = image_ID(t->sval, data.core.image, data.core.NB_MAX_IMAGE);
        int     k  = (id == -1) ? -1 : add_image(p, &data.core.image[id]);
        out->vec   = 1;
        out->s1    = 0;
        return (k < 0) ? -1 : emit(p, (fins) { .k = FI_IMG, .img = k }, 1);
    }
    case TOK_FUNC_D_D:
    case TOK_FUNC_DD_D:
    case TOK_FUNC_DDD_D:
        return c_func(p, t, out);
    case TOK_FUNC_WHERE:
    {
        fval a[3];
        if (c_args(p, 3, a) != 0 || !a[0].vec) /* a number picks its branch: the normal path */
        {
            return -1;
        }
        return c_select(p, a[1], a[2], out);
    }
    case TOK_FUNC_COORD:
        return c_coord(p, t, out);
    default:
        return -1;
    }
}

/**
 * c_expr - Compile an expression with minimum precedence @min_prec into a fused program
 * @p:        Fused program being compiled
 * @min_prec: Minimum operator binding precedence to consume
 * @out:      Output compile-time value descriptor
 *
 * Uses Pratt precedence climbing to compile binary arithmetic, comparisons, and
 * image-conditioned ternary `? :` selections into `p->ins` (or `p->sins` for
 * scalar sub-expressions), returning -1 if any non-fusible construct is met.
 *
 * Return: 0 on success, -1 if the expression cannot be fused
 */
static int c_expr(fprog *p, int min_prec, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();

    if (c_primary(p, out) != 0)
    {
        return -1;
    }
    for (;;)
    {
        cli_token_type op   = cur()->type;
        int            prec = cli_calc_op_prec(op);
        if (prec < min_prec)
        {
            return 0;
        }
        if (op == TOK_OP_QUESTION && out->vec) /* image condition: select per pixel */
        {
            fval t;
            fval f;
            adv();
            if (c_expr(p, 0, &t) != 0 || cur()->type != TOK_OP_COLON)
            {
                return -1;
            }
            adv();
            if (c_expr(p, 0, &f) != 0 || c_select(p, t, f, out) != 0)
            {
                return -1;
            }
            continue;
        }
        if (!fusible_op(op) || (!out->vec && (op == TOK_OP_AND || op == TOK_OP_OR) &&
                                (out->s1 != 0 || (out->c != 0.0) == (op == TOK_OP_OR))))
        {
            return -1;
        }
        adv();
        fval r;
        if (c_expr(p, (op == TOK_OP_CARET) ? prec : prec + 1, &r) != 0)
        {
            return -1;
        }
        p->out_dt = 0;
        if (!out->vec && !r.vec)
        {
            if ((op == TOK_OP_SLASH || op == TOK_OP_MOD) && r.s1 == 0 && r.c == 0.0)
            {
                return -1; /* the normal path reports the division by zero */
            }
            if (out->s1 == 0 && r.s1 == 0)
            {
                out->c = scalar_bin(op, out->c, r.c);
            }
            else if (emit_scalar(p,
                                 (fsins) { .k  = FI_BIN,
                                           .op = op,
                                           .s  = { out->s1, r.s1, 0 },
                                           .c  = { out->c, r.c, 0.0 } },
                                 out) != 0)
            {
                return -1;
            }
            continue;
        }
        if (!r.vec && r.s1 == 0 && (op == TOK_OP_SLASH || op == TOK_OP_MOD) && r.c == 0.0)
        {
            return -1;
        }
        if (!out->vec && (op == TOK_OP_MOD || op == TOK_OP_CARET))
        {
            return -1; /* scalar % image, scalar ^ image: not supported by the normal path either */
        }
        int both = out->vec && r.vec;
        if (emit(p,
                 (fins) { .k    = FI_BIN,
                          .op   = op,
                          .cpos = both       ? 0
                                  : out->vec ? 2
                                             : 1,
                          .s1   = out->vec ? r.s1 : out->s1,
                          .c    = out->vec ? r.c : out->c },
                 both ? -1 : 0) != 0)
        {
            return -1;
        }
        out->vec = 1;
        out->s1  = 0;
    }
}

/* ---------------------------------------------------------------- execution */

typedef struct
{
    uint64_t n;
    double   sum;
    double   mean;
    double   m2;
    double   min;
    double   max;
} fred_acc;

/**
 * fred_merge - Combine a block or thread reduction accumulator @b into @a
 * @a: Destination accumulator updated in place
 * @b: Source accumulator to merge
 *
 * Uses Chan's parallel variance recurrence for `mean` and `m2` so block-wise
 * and OpenMP-thread reductions preserve two-pass numerical stability.
 *
 * Return: void
 */
static void fred_merge(fred_acc *a, const fred_acc *b)
{
    if (b->n == 0)
    {
        return;
    }
    if (a->n == 0)
    {
        *a = *b;
        return;
    }
    uint64_t n = a->n + b->n;
    double   d = b->mean - a->mean;
    a->sum += b->sum;
    a->mean += d * ((double) b->n / (double) n);
    a->m2 += b->m2 + d * d * ((double) a->n * (double) b->n / (double) n);
    if (b->min < a->min || cli_calc_isnan(a->min) != 0.0)
    {
        a->min = b->min;
    }
    if (b->max > a->max || cli_calc_isnan(a->max) != 0.0)
    {
        a->max = b->max;
    }
    a->n = n;
}

/**
 * load_block - Convert a contiguous block of image pixels to `double`
 * @im:  Source real image
 * @off: Starting pixel index in @im
 * @len: Number of pixels to load (`<= FUSE_BLK`)
 * @dst: Destination `double` buffer of length @len
 *
 * Dispatches on `im->md[0].datatype` to widen @len pixels starting at @off into
 * a cache-resident stack slot for fused vector execution.
 *
 * Return: void
 */
static void load_block(const IMAGE *im, uint64_t off, int len, double *restrict dst)
{
    uint8_t dt = im->md[0].datatype;
#define LOAD_BODY(MBR)                                                 \
    {                                                                  \
        const __typeof__(im->array.MBR[0]) *src = im->array.MBR + off; \
        for (int j = 0; j < len; j++)                                  \
        {                                                              \
            dst[j] = (double) src[j];                                  \
        }                                                              \
    }
    MILK_FOR_EACH_DATATYPE(dt, LOAD_BODY, LOAD_BODY(D))
#undef LOAD_BODY
}

/**
 * store_block - Write a block of evaluated `double` pixels or a constant into an image
 * @im:  Destination real image
 * @off: Starting pixel index in @im
 * @len: Number of pixels to write (`<= FUSE_BLK`)
 * @v:   Source `double` buffer of length @len (ignored when @isc is non-zero)
 * @isc: Non-zero to fill the block with scalar constant @c instead of @v
 * @c:   Scalar constant value used when @isc is non-zero
 *
 * Converts each double-precision result (or scalar constant @c) to the native
 * datatype of @im and stores it at `[@off, @off + @len)`.
 *
 * Return: void
 */
static void store_block(IMAGE   *im,
                        uint64_t off,
                        int      len,
                        const double *restrict v,
                        int    isc,
                        double c)
{
    uint8_t dt = im->md[0].datatype;
#define STORE_BODY(MBR)                                               \
    {                                                                 \
        __typeof__(im->array.MBR[0]) *out = im->array.MBR + off;      \
        for (int j = 0; j < len; j++)                                 \
        {                                                             \
            out[j] = (__typeof__(im->array.MBR[0])) (isc ? c : v[j]); \
        }                                                             \
    }
    MILK_FOR_EACH_DATATYPE(dt, STORE_BODY, STORE_BODY(D))
#undef STORE_BODY
}

/** eval_scalars() - evaluate per-frame scalar variable sub-expressions into instruction constants. */
static void eval_scalars(fprog *p)
{
    for (int i = 0; i < p->n; i++)
    {
        if (p->ins[i].k == FI_COORD &&
            (p->ins[i].cpos == CALC_COORD_RAND || p->ins[i].cpos == CALC_COORD_RANDN) &&
            p->ins[i].s1 == 0 && p->ins[i].c2 == 0.0)
        {
            p->ins[i].c = (double) cli_calc_next_rng_seed();
        }
    }
    if (p->ns == 0)
    {
        return;
    }
    double sval[FUSE_MAXINS];
    for (int i = 0; i < p->ns; i++)
    {
        const fsins *s = &p->sins[i];
        double       a = (s->s[0] > 0) ? sval[s->s[0] - 1] : s->c[0];
        double       b = (s->s[1] > 0) ? sval[s->s[1] - 1] : s->c[1];
        double       d = (s->s[2] > 0) ? sval[s->s[2] - 1] : s->c[2];
        switch (s->k)
        {
        case FI_SVAR:
        {
            const VARIABLE *v = &data.core.variable[s->vid];
            sval[i]           = (v->type == 1) ? (double) v->value.l : v->value.f;
            break;
        }
        case FI_NEG:
            sval[i] = -a;
            break;
        case FI_NOT:
            sval[i] = (double) (a == 0.0);
            break;
        case FI_BIN:
            sval[i] = scalar_bin(s->op, a, b);
            break;
        case FI_FN1:
            sval[i] = ((double (*)(double)) s->fn)(a);
            break;
        case FI_FN2:
            sval[i] = ((double (*)(double, double)) s->fn)(a, b);
            break;
        default:
            sval[i] = ((double (*)(double, double, double)) s->fn)(a, b, d);
            break;
        }
    }
    for (int i = 0; i < p->n; i++)
    {
        if (p->ins[i].s1 > 0)
        {
            p->ins[i].c = sval[p->ins[i].s1 - 1];
        }
        if (p->ins[i].s2 > 0)
        {
            p->ins[i].c2 = sval[p->ins[i].s2 - 1];
        }
    }
}

/** store_block_d() - store a block of double results into @im (any real datatype). */
static void store_block_d(IMAGE *im, uint64_t off, int len, const double *restrict v)
{
    store_block(im, off, len, v, 0, 0.0);
}

/** load_block_f() - copy a block of a FLOAT image (the float32 path only runs on FLOAT images). */
static void load_block_f(const IMAGE *im, uint64_t off, int len, float *restrict dst)
{
    __builtin_memcpy(dst, im->array.F + off, (size_t) len * sizeof(float));
}

/** store_block_f() - copy a block of float results into a FLOAT image. */
static void store_block_f(IMAGE *im, uint64_t off, int len, const float *restrict v)
{
    __builtin_memcpy(im->array.F + off, v, (size_t) len * sizeof(float));
}

/* block kernels in double (any datatype; results converted on store) */
#define FT double
#define FK(name) name##_d
#define FM(fn) fn
#define FK_LOAD load_block
#define FK_STORE store_block_d
#include "cli_calc_fuse_kernels.h"
#undef FT
#undef FK
#undef FM
#undef FK_LOAD
#undef FK_STORE

/* block kernels in float32 (every image FLOAT: twice the SIMD width, float libm) */
#define FT float
#define FK(name) name##_f
#define FM(fn) fn##f
#define FK_LOAD load_block_f
#define FK_STORE store_block_f
#include "cli_calc_fuse_kernels.h"
#undef FT
#undef FK
#undef FM
#undef FK_LOAD
#undef FK_STORE

/** all_float_prog() - 1 if every image @p reads is FLOAT and @p has no non-FLOAT explicit cast. */
static int all_float_prog(const fprog *p)
{
    if (p->out_double || (p->out_dt != 0 && p->out_dt != _DATATYPE_FLOAT))
    {
        return 0;
    }
    for (int i = 0; i < p->nimg; i++)
    {
        if (p->img[i]->md[0].datatype != _DATATYPE_FLOAT)
        {
            return 0;
        }
    }
    return 1;
}

/** all_float() - 1 if @out and every image @p reads are FLOAT: the program runs in float32. */
static int all_float(const fprog *p, const IMAGE *out)
{
    return (out->md[0].datatype == _DATATYPE_FLOAT) && all_float_prog(p);
}

/**
 * run_program - Execute a compiled fused program across all blocks of an output image
 * @p:   Compiled fused program to execute
 * @out: Destination image receiving the evaluated pixels
 *
 * Evaluates per-frame scalar tape instructions (`eval_scalars()`) and then
 * processes @out in `FUSE_BLK`-pixel chunks, parallelizing across blocks with
 * OpenMP when `nelement >= FUSE_OMP_MIN`. The program runs in float32 when @out
 * and every image it reads are FLOAT, else in double.
 *
 * Return: void
 */
static void run_program(fprog *p, IMAGE *out)
{
    eval_scalars(p);
    uint64_t nel  = out->md[0].nelement;
    uint64_t nblk = (nel + FUSE_BLK - 1) / FUSE_BLK;
    if (all_float(p, out))
    {
#pragma omp parallel if (nel >= FUSE_OMP_MIN)
        {
            MILK_ALIGNED(64) float stk[FUSE_MAXSTK][FUSE_BLK];
#pragma omp for schedule(static)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t off = b * FUSE_BLK;
                int      len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
                run_block_f(p, out, off, len, stk);
            }
        }
        return;
    }
#pragma omp parallel if (nel >= FUSE_OMP_MIN)
    {
        MILK_ALIGNED(64) double stk[FUSE_MAXSTK][FUSE_BLK];
#pragma omp for schedule(static)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * FUSE_BLK;
            int      len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
            run_block_d(p, out, off, len, stk);
        }
    }
}

/**
 * run_reduce - Evaluate one or two fused programs block-by-block into a scalar reduction
 * @p1:      Primary fused program (value expression)
 * @p2:      Optional secondary fused program (mask or second `dot()` operand), or NULL
 * @op:      Reduction operation (`RED_SUM`..`RED_DOT`)
 * @out_acc: Output combined accumulator
 *
 * Evaluates @p1 (and @p2 if non-NULL) in L1-resident `FUSE_BLK` tiles without
 * allocating a temporary image, reducing each tile in `double` and combining
 * thread-local accumulators with `fred_merge()`.
 *
 * Return: void
 */
static void run_reduce(fprog *p1, fprog *p2, reduce_op op, fred_acc *out_acc)
{
    eval_scalars(p1);
    if (p2 != NULL)
    {
        eval_scalars(p2);
    }
    uint64_t nel  = p1->img[0]->md[0].nelement;
    uint64_t nblk = (nel + FUSE_BLK - 1) / FUSE_BLK;
    *out_acc      = (fred_acc) { 0, 0.0, 0.0, 0.0, NAN, NAN };

    if (all_float_prog(p1) && (p2 == NULL || all_float_prog(p2)))
    {
#pragma omp parallel if (nel >= FUSE_OMP_MIN)
        {
            MILK_ALIGNED(64) float stk1[FUSE_MAXSTK][FUSE_BLK];
            MILK_ALIGNED(64) float stk2[FUSE_MAXSTK][FUSE_BLK];
            fred_acc               loc = { 0, 0.0, 0.0, 0.0, NAN, NAN };
#pragma omp for schedule(static) nowait
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t off = b * FUSE_BLK;
                int      len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
                eval_block_f(p1, off, len, stk1);
                if (p2 != NULL)
                {
                    eval_block_f(p2, off, len, stk2);
                }
                reduce_block_f(op, stk1[0], (p2 != NULL) ? stk2[0] : NULL, len, &loc);
            }
#pragma omp critical
            {
                fred_merge(out_acc, &loc);
            }
        }
        return;
    }
#pragma omp parallel if (nel >= FUSE_OMP_MIN)
    {
        MILK_ALIGNED(64) double stk1[FUSE_MAXSTK][FUSE_BLK];
        MILK_ALIGNED(64) double stk2[FUSE_MAXSTK][FUSE_BLK];
        fred_acc                loc = { 0, 0.0, 0.0, 0.0, NAN, NAN };
#pragma omp for schedule(static) nowait
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * FUSE_BLK;
            int      len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
            eval_block_d(p1, off, len, stk1);
            if (p2 != NULL)
            {
                eval_block_d(p2, off, len, stk2);
            }
            reduce_block_d(op, stk1[0], (p2 != NULL) ? stk2[0] : NULL, len, &loc);
        }
#pragma omp critical
        {
            fred_merge(out_acc, &loc);
        }
    }
}

/** compile() - compile the expression at the current token; tokens are rewound unless it returns 0. */
static int compile(fprog *p, int min_prec, fval *v)
{
    int pos = cli_calc_tok_pos();
    memset(p, 0, sizeof(*p));
    if (c_expr(p, min_prec, v) != 0)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------- public */

/**
 * cli_calc_fuse_reduce - Attempt fused map-reduce evaluation of an image reduction call
 * @op:      Reduction operation (`RED_SUM`..`RED_DOT`)
 * @is_long: Non-zero if the reduction returns `VAL_LONG` (`icount`), 0 for `VAL_DOUBLE`
 * @out:     Output `val_t` receiving the reduced scalar when fused
 *
 * Compiles the argument expression (and optional mask or second `dot()` operand)
 * into fused block programs and reduces them in L1 cache without allocating any
 * temporary `_tmpcalc` images, rewinding the token cursor if not applicable.
 *
 * Return: 1 if fused and evaluated into @out, 0 to fall back to normal evaluation
 */
int cli_calc_fuse_reduce(reduce_op op, int is_long, val_t *out)
{
    if (parse_mode == 0 || !cli_calc_tokens_have_image())
    {
        return 0;
    }
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    static fprog p1;
    static fprog p2;
    fval         v1;
    fval         v2;
    int          pos = cli_calc_tok_pos();

    if (compile(&p1, 0, &v1) != 0 || !v1.vec)
    {
        cli_calc_tok_seek(pos);
        return 0;
    }

    int has_p2 = 0;
    if (op == RED_DOT)
    {
        if (cur()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return 0;
        }
        adv();
        if (compile(&p2, 0, &v2) != 0 || !v2.vec || !same_shape(p1.img[0], p2.img[0]) ||
            (p1.nops == 0 && p2.nops == 0))
        {
            cli_calc_tok_seek(pos);
            return 0;
        }
        has_p2 = 1;
    }
    else if (cur()->type == TOK_COMMA)
    {
        if (op != RED_SUM && op != RED_MEAN && op != RED_MIN && op != RED_MAX && op != RED_STD &&
            op != RED_VAR && op != RED_NORM && op != RED_ASUM)
        {
            cli_calc_tok_seek(pos);
            return 0;
        }
        adv();
        if (compile(&p2, 0, &v2) != 0 || !v2.vec || !same_shape(p1.img[0], p2.img[0]))
        {
            cli_calc_tok_seek(pos);
            return 0;
        }
        has_p2 = 1;
    }

    if (cur()->type != TOK_RPAREN)
    {
        cli_calc_tok_seek(pos);
        return 0;
    }
    adv();

    fred_acc acc;
    run_reduce(&p1, has_p2 ? &p2 : NULL, op, &acc);
    if (acc.n == 0)
    {
        *out = is_long
                   ? mk_long(0)
                   : mk_double((op == RED_SUM || op == RED_ASUM || op == RED_NORM || op == RED_DOT)
                                   ? 0.0
                                   : NAN);
        return 1;
    }
    double r = 0.0;
    switch (op)
    {
    case RED_SUM:
    case RED_ASUM:
    case RED_DOT:
        r = acc.sum;
        break;
    case RED_MEAN:
        r = acc.sum / (double) acc.n;
        break;
    case RED_MIN:
        r = acc.min;
        break;
    case RED_MAX:
        r = acc.max;
        break;
    case RED_VAR:
        r = acc.m2 / (double) acc.n;
        break;
    case RED_STD:
        r = sqrt(acc.m2 / (double) acc.n);
        break;
    case RED_NORM:
        r = sqrt(acc.sum);
        break;
    case RED_COUNT:
    default:
        r = (double) acc.n;
        break;
    }
    *out = is_long ? mk_long((long) r) : mk_double(r);
    return 1;
}

/**
 * cli_calc_fuse_expr - Attempt one-pass fused evaluation of an element-wise image expression
 * @min_prec: Minimum operator precedence for the expression at the current token
 * @out:      Output `val_t` set to the temporary result image name when fused
 *
 * Compiles the upcoming sub-expression into a single-pass block program when it
 * involves at least one vector operation on same-shape real images, allocating
 * a single `_tmpcalc` output image and rewinding the token stream if fusion is
 * not applicable.
 *
 * Return: 1 if fused and evaluated into @out, 0 to fall back to normal evaluation, -1 on error
 */
int cli_calc_fuse_expr(int min_prec, val_t *out)
{
    if (parse_mode == 0 || !cli_calc_tokens_have_image())
    {
        return 0;
    }
    static fprog
         p; /* the evaluator is single-threaded; recursion never reaches here twice at once */
    fval v;
    int  pos = cli_calc_tok_pos();
    if (compile(&p, min_prec, &v) != 0)
    {
        return 0;
    }
    if (p.nops == 0 || !v.vec) /* a bare image or a constant: nothing to fuse */
    {
        cli_calc_tok_seek(pos);
        return 0;
    }

    const IMAGE *ref    = p.img[0];
    const char  *tmpn   = alloc_tmpname();
    imageID      tid    = -1;
    uint32_t     sz[3]  = { ref->md[0].size[0], ref->md[0].size[1], ref->md[0].size[2] };
    uint8_t      out_dt = p.out_dt ? p.out_dt : (p.out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    create_image_ID(tmpn, ref->md[0].naxis, sz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return -1;
    }
    run_program(&p, &data.core.image[tid]);
    *out = mk_string(tmpn);
    return 1;
}

/** new_output() - a temporary image of @ref's size, DOUBLE if @dbl else FLOAT; its name, or NULL. */
static const char *new_output(const IMAGE *ref, int dbl, IMAGE **out)
{
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[3] = { ref->md[0].size[0], ref->md[0].size[1], ref->md[0].size[2] };

    create_image_ID(tmpn, ref->md[0].naxis, sz, dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, 0, 10, 0,
                    &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return NULL;
    }
    *out = &data.core.image[tid];
    return tmpn;
}

/**
 * cli_calc_select - Evaluate a per-pixel conditional selection `c ? t : f` into a temporary image
 * @c:  Condition image (non-zero pixel selects the true branch)
 * @t:  True-branch image, or NULL when the true branch is scalar @tc
 * @tc: Scalar true-branch value used when @t is NULL
 * @f:  False-branch image, or NULL when the false branch is scalar @fc
 * @fc: Scalar false-branch value used when @f is NULL
 *
 * Builds and executes a short fused `FI_SEL` program when @c (and any non-NULL
 * branch images @t, @f) are same-shape real images.
 *
 * Return: Name of the temporary output image, or NULL if shapes or types are incompatible
 */
const char *cli_calc_select(const IMAGE *c, const IMAGE *t, double tc, const IMAGE *f, double fc)
{
    static fprog p;
    IMAGE       *out;

    memset(&p, 0, sizeof(p));
    if (add_image(&p, c) != 0 || (t != NULL && add_image(&p, t) < 0) ||
        (f != NULL && add_image(&p, f) < 0))
    {
        return NULL;
    }
    p.ins[p.n++] = (fins) { .k = FI_IMG, .img = 0 };
    if (t != NULL)
    {
        p.ins[p.n++] = (fins) { .k = FI_IMG, .img = add_image(&p, t) };
    }
    if (f != NULL)
    {
        p.ins[p.n++] = (fins) { .k = FI_IMG, .img = add_image(&p, f) };
    }
    p.ins[p.n++] =
        (fins) { .k = FI_SEL, .cpos = (t != NULL) | ((f != NULL) << 1), .c = tc, .c2 = fc };
    const char *tmpn = new_output(c, p.out_double, &out);
    if (tmpn != NULL)
    {
        run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_coord - Generate a coordinate image (`ix`, `iy`, `ir`, `itheta`) shaped like @ref
 * @ref:  Reference image providing output dimensions and precision
 * @kind: Coordinate selector (`CALC_COORD_X`, `CALC_COORD_Y`, `CALC_COORD_R`, `CALC_COORD_THETA`)
 * @cx:   X-axis center coordinate subtracted from pixel column index
 * @cy:   Y-axis center coordinate subtracted from pixel row index
 *
 * Allocates a temporary `FLOAT` or `DOUBLE` image matching @ref and populates
 * it with pixel coordinates relative to `(@cx, @cy)` using the fused `FI_COORD`
 * block runner.
 *
 * Return: Name of the temporary coordinate image, or NULL on allocation failure
 */
const char *cli_calc_coord(const IMAGE *ref, int kind, double cx, double cy)
{
    static fprog p;
    IMAGE       *out;

    memset(&p, 0, sizeof(p));
    p.img[p.nimg++]  = ref;
    p.ins[p.n++]     = (fins) { .k = FI_COORD, .img = 0, .cpos = kind, .c = cx, .c2 = cy };
    const char *tmpn = new_output(ref, ref->md[0].datatype == _DATATYPE_DOUBLE, &out);
    if (tmpn != NULL)
    {
        run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_next_rng_seed - Return the next 32-bit seed for unseeded `rand()` / `randn()` calls
 *
 * Mixes the realtime clock and PID on first use and advances a Weyl sequence
 * on each call so consecutive `rand(im)` / `randn(im)` calls and `milk.calc`
 * frames produce independent random realizations.
 *
 * Return: Non-zero 32-bit seed value
 */
uint32_t cli_calc_next_rng_seed(void)
{
    static uint32_t rng_state = 0;
    if (rng_state == 0)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        uint32_t s =
            (uint32_t) ts.tv_nsec ^ (uint32_t) ts.tv_sec ^ ((uint32_t) getpid() * 0x9e3779b9u);
        rng_state = (s != 0) ? s : 0x6d2b79f5u;
    }
    rng_state += 0x9e3779b9u;
    return cli_calc_hash32(rng_state);
}

/**
 * cli_calc_coord_kind - Map a `TOK_FUNC_COORD` token to its `CALC_COORD_*` enum value
 * @t: Coordinate or RNG function token (`ix(`, `iy(`, `ir(`, `itheta(`, `rand(`, `randn(`)
 *
 * Inspects `t->sval` to identify which pixel coordinate or random generator was called.
 *
 * Return: `CALC_COORD_X`, `CALC_COORD_Y`, `CALC_COORD_R`, `CALC_COORD_THETA`, `CALC_COORD_RAND`, or `CALC_COORD_RANDN`
 */
int cli_calc_coord_kind(const cli_token *t)
{
    if (strncmp(t->sval, "rand(", 5) == 0 || strncmp(t->sval, "urand(", 6) == 0)
    {
        return CALC_COORD_RAND;
    }
    if (strncmp(t->sval, "randn(", 6) == 0 || strncmp(t->sval, "grand(", 6) == 0)
    {
        return CALC_COORD_RANDN;
    }
    return (strncmp(t->sval, "ix(", 3) == 0)   ? CALC_COORD_X
           : (strncmp(t->sval, "iy(", 3) == 0) ? CALC_COORD_Y
           : (strncmp(t->sval, "ir(", 3) == 0) ? CALC_COORD_R
                                               : CALC_COORD_THETA;
}

/**
 * cli_calc_eval_const_scalar - Parse and fold a constant scalar expression from `eval_tokens`
 * @out: Output double value
 *
 * Return: 0 on success (when the sub-expression is a compile-time scalar constant), -1 otherwise
 */
int cli_calc_eval_const_scalar(double *out)
{
    fprog tp;
    fval  dv;
    int   save_keep = keep_vars;
    memset(&tp, 0, sizeof(tp));
    keep_vars = 0;
    int rc    = c_expr(&tp, 0, &dv);
    keep_vars = save_keep;
    if (rc != 0 || dv.vec || dv.s1 != 0)
    {
        return -1;
    }
    *out = dv.c;
    return 0;
}

typedef struct
{
    fprog                p;
    IMAGE               *dst;
    uint64_t             last_cnt0[FUSE_MAXIMG];
    struct timespec      last_check;
    cli_calc_kept_matrix m;
} fkept;

/**
 * cli_calc_fuse_keep - Compile an image assignment RHS into a persistent `fkept` program
 * @dst: Destination image being assigned
 * @aop: Assignment operator (`=`, `+=`, `-=`, or `*=`)
 *
 * Compiles either a direct matrix/reshape operation (`cli_calc_compile_kept_matrix()`)
 * or an element-wise fused expression with live scalar variable references
 * (`keep_vars = 1`) that writes directly into @dst on each `cli_calc_fused_run()`.
 *
 * Return: Newly allocated `fkept` program pointer, or NULL if not fusible
 */
void *cli_calc_fuse_keep(IMAGE *dst, cli_token_type aop)
{
    fkept *k = calloc(1, sizeof(*k));
    fval   v;
    int    pos = cli_calc_tok_pos();

    if (k == NULL)
    {
        return NULL;
    }
    if (cli_calc_compile_kept_matrix(&k->m, dst, aop, 1, 0) == 0)
    {
        k->dst    = dst;
        k->p.nimg = k->m.nimg;
        for (int i = 0; i < k->p.nimg; i++)
        {
            k->p.img[i]     = k->m.img[i];
            k->last_cnt0[i] = k->p.img[i]->md[0].cnt0;
        }
        clock_gettime(CLOCK_MONOTONIC_COARSE, &k->last_check);
        return k;
    }
    memset(k, 0, sizeof(*k));
    if (!is_real_datatype(dst->md[0].datatype))
    {
        free(k);
        return NULL;
    }
    keep_vars = 1;
    int rc    = compile(&k->p, 0, &v);
    keep_vars = 0;
    if (rc == 0 && !v.vec && aop != TOK_OP_SLASH_EQ)
    {
        if (add_image(&k->p, dst) < 0 ||
            emit(&k->p, (fins) { .k = FI_CONST, .s1 = v.s1, .c = v.c }, 1) != 0)
        {
            cli_calc_tok_seek(pos);
            free(k);
            return NULL;
        }
        v.vec = 1;
    }
    if (rc != 0 || !v.vec || k->p.nimg == 0 || !same_shape(dst, k->p.img[0]) ||
        aop == TOK_OP_SLASH_EQ)
    {
        cli_calc_tok_seek(pos);
        free(k);
        return NULL;
    }
    if (aop != TOK_EQUAL) /* dst op= expr  ->  dst = dst op (expr) */
    {
        int i = (k->p.n + 2 <= FUSE_MAXINS && k->p.maxdepth + 1 <= FUSE_MAXSTK)
                    ? add_image(&k->p, dst)
                    : -1;
        if (i < 0)
        {
            cli_calc_tok_seek(pos);
            free(k);
            return NULL;
        }
        memmove(&k->p.ins[1], &k->p.ins[0], sizeof(fins) * (size_t) k->p.n);
        k->p.ins[0]          = (fins) { .k = FI_IMG, .img = i };
        k->p.ins[k->p.n + 1] = (fins) { .k    = FI_BIN,
                                        .op   = (aop == TOK_OP_PLUS_EQ)    ? TOK_OP_PLUS
                                                : (aop == TOK_OP_MINUS_EQ) ? TOK_OP_MINUS
                                                                           : TOK_OP_STAR,
                                        .cpos = 0 };
        k->p.n += 2;
    }
    k->dst = dst;
    for (int i = 0; i < k->p.nimg; i++)
    {
        k->last_cnt0[i] = k->p.img[i]->md[0].cnt0;
    }
    clock_gettime(CLOCK_MONOTONIC_COARSE, &k->last_check);
    return k;
}

/**
 * cli_calc_fused_run - Execute a pre-compiled `fkept` statement and notify output stream readers
 * @kept: Opaque `fkept` pointer returned by `cli_calc_fuse_keep()`
 *
 * Periodically checks whether shared-memory input or output streams were
 * recreated, validates image shapes and types, executes the kept element-wise
 * or matrix/reshape operation into `k->dst`, and posts stream semaphores via
 * `processinfo_update_output_stream()`.
 *
 * Return: 0 on success, -1 if an input or output image changed shape or type
 */
int cli_calc_fused_run(void *kept)
{
    fkept          *k = kept;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    int check_shm = (ts.tv_sec != k->last_check.tv_sec || ts.tv_nsec != k->last_check.tv_nsec);
    if (check_shm)
    {
        k->last_check = ts;
    }

    for (int i = -1; i < k->p.nimg; i++) /* a stream re-created by its owner may have a new size */
    {
        IMAGE *im = (IMAGE *) ((i < 0) ? k->dst : k->p.img[i]);
        if (check_shm && im->used && im->md != NULL)
        {
            if (i < 0 || im->md[0].cnt0 == k->last_cnt0[i])
            {
                cli_calc_refresh_shared_image(im);
            }
            if (i >= 0)
            {
                k->last_cnt0[i] = im->md[0].cnt0;
            }
        }
        if (k->m.kind == 0 &&
            (!im->used || !is_real_datatype(im->md[0].datatype) || !same_shape(im, k->dst)))
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                     im->name);
            return -1;
        }
    }
    if (k->m.kind != 0)
    {
        return cli_calc_exec_kept_matrix(&k->m, k->dst);
    }
    k->dst->md[0].write = 1;
    run_program(&k->p, k->dst);
    processinfo_update_output_stream(NULL, k->dst, NULL);
    return 0;
}

/**
 * cli_calc_fuse_assign - Attempt to fuse and execute a whole-image assignment in place
 * @dst: Existing destination image on the left-hand side
 * @aop: Assignment operator (`=`, `+=`, `-=`, `*=`, or `/=`)
 *
 * Compiles the right-hand side expression and evaluates it directly into @dst
 * without allocating a temporary image when shapes and datatypes match the
 * normal evaluator's semantics, rewinding the token stream otherwise.
 *
 * Return: 1 if fused and written to @dst, 0 to fall back to normal evaluation, -1 on error
 */
int cli_calc_fuse_assign(IMAGE *dst, cli_token_type aop)
{
    if (parse_mode == 0 || !is_real_datatype(dst->md[0].datatype))
    {
        return 0;
    }
    int                  pos = cli_calc_tok_pos();
    cli_calc_kept_matrix km;
    if (cli_calc_compile_kept_matrix(&km, dst, aop, 0,
                                     aop == TOK_EQUAL && dst->md[0].shared != 1) == 0)
    {
        if (cli_calc_exec_kept_matrix(&km, dst) == 0)
        {
            return 1;
        }
        cli_calc_tok_seek(pos);
        return 0;
    }
    static fprog p;
    fval         v;
    if (compile(&p, 0, &v) != 0)
    {
        return 0;
    }
    if (!v.vec) /* im op= number */
    {
        dst->md[0].write = 1;
        if (cli_calc_inplace(dst, aop, NULL, v.c) != 0)
        {
            dst->md[0].write = 0;
            return -1;
        }
        processinfo_update_output_stream(NULL, dst, NULL);
        return 1;
    }
    /* write straight into dst only where the normal path would keep dst's type and shape */
    uint8_t out_dt = p.out_dt ? p.out_dt : (p.out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    if ((p.nops == 0 && aop == TOK_EQUAL) || !same_shape(dst, p.img[0]) || aop == TOK_OP_SLASH_EQ ||
        (aop == TOK_EQUAL && dst->md[0].shared != 1 && dst->md[0].datatype != out_dt))
    {
        cli_calc_tok_seek(pos);
        return 0;
    }
    if (aop != TOK_EQUAL) /* dst op= expr  ->  dst = dst op (expr) */
    {
        if (p.n + 2 > FUSE_MAXINS || p.maxdepth + 1 > FUSE_MAXSTK)
        {
            cli_calc_tok_seek(pos);
            return 0;
        }
        int k = add_image(&p, dst);
        if (k < 0)
        {
            cli_calc_tok_seek(pos);
            return 0;
        }
        memmove(&p.ins[1], &p.ins[0], sizeof(fins) * (size_t) p.n);
        p.ins[0]          = (fins) { .k = FI_IMG, .img = k };
        cli_token_type op = (aop == TOK_OP_PLUS_EQ)    ? TOK_OP_PLUS
                            : (aop == TOK_OP_MINUS_EQ) ? TOK_OP_MINUS
                                                       : TOK_OP_STAR;
        p.ins[p.n + 1]    = (fins) { .k = FI_BIN, .op = op, .cpos = 0 };
        p.n += 2;
    }
    dst->md[0].write = 1;
    run_program(&p, dst);
    processinfo_update_output_stream(NULL, dst, NULL);
    return 1;
}

/**
 * cli_calc_inplace - Update an image in place with a scalar or same-size source image
 * @dst: Destination image to modify
 * @aop: Assignment or compound operator (`=`, `+=`, `-=`, `*=`, or `/=`)
 * @src: Optional source image of the same element count, or NULL when @val is used
 * @val: Scalar operand used when @src is NULL
 *
 * Delegates complex images to `cli_calc_inplace_complex()`, checks for zero
 * divisors on `/=`, and executes typed block-wise updates across @dst.
 *
 * Return: 0 on success, -1 on division by zero or unsupported datatype
 */
int cli_calc_inplace(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val)
{
    if (aop == TOK_OP_SLASH_EQ && src == NULL && val == 0.0)
    {
        parse_errmsg("Division by zero");
        return -1;
    }
    if (!is_real_datatype(dst->md[0].datatype) ||
        (src != NULL && !is_real_datatype(src->md[0].datatype)))
    {
        return cli_calc_inplace_complex(dst, aop, src, val);
    }
    uint64_t nel = dst->md[0].nelement;
    if (src != NULL &&
        aop == TOK_OP_SLASH_EQ) /* like the scalar case: a zero divisor is an error */
    {
        MILK_ALIGNED(64) double blk[FUSE_BLK];
        int                     bad = 0;
        for (uint64_t off = 0; off < nel && !bad; off += FUSE_BLK)
        {
            int len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
            load_block(src, off, len, blk);
            for (int j = 0; j < len && !bad; j++)
            {
                bad = (blk[j] == 0.0);
            }
        }
        if (bad)
        {
            parse_errmsg("Division by zero");
            return -1;
        }
    }

    static fprog p;
    memset(&p, 0, sizeof(p));
    if (aop == TOK_EQUAL && src == NULL)
    {
#pragma omp parallel for schedule(static) if (nel >= FUSE_OMP_MIN)
        for (uint64_t off = 0; off < nel; off += FUSE_BLK)
        {
            store_block(dst, off, (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK), NULL, 1,
                        val);
        }
        return 0;
    }
    cli_token_type op = (aop == TOK_OP_PLUS_EQ)    ? TOK_OP_PLUS
                        : (aop == TOK_OP_MINUS_EQ) ? TOK_OP_MINUS
                        : (aop == TOK_OP_STAR_EQ)  ? TOK_OP_STAR
                                                   : TOK_OP_SLASH;
    p.img[p.nimg++]   = dst;
    if (src != NULL)
    {
        p.img[p.nimg++] = src;
    }
    if (aop == TOK_EQUAL) /* src -> dst, converting the type */
    {
        p.ins[p.n++] = (fins) { .k = FI_IMG, .img = 1 };
    }
    else
    {
        p.ins[p.n++] = (fins) { .k = FI_IMG, .img = 0 };
        if (src != NULL)
        {
            p.ins[p.n++] = (fins) { .k = FI_IMG, .img = 1 };
        }
        p.ins[p.n++] = (fins) { .k = FI_BIN, .op = op, .cpos = (src != NULL) ? 0 : 2, .c = val };
    }
    run_program(&p, dst);
    return 0;
}
