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
    default:
        return a != b;
    }
}

/**
 * fusible_op - Test whether a binary operator token can be compiled into a fused program
 * @op: Operator token type
 *
 * Accepts arithmetic (`+`, `-`, `*`, `/`, `%`, `^`) and relational/equality
 * (`<`, `<=`, `>`, `>=`, `==`, `!=`) operators, rejecting logical short-circuit
 * and bitwise operators.
 *
 * Return: 1 if @op is supported by the fused engine, 0 otherwise
 */
static int fusible_op(cli_token_type op)
{
    return op >= TOK_OP_PLUS && op <= TOK_OP_NEQ; /* arithmetic and comparisons */
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
    /* image forms the normal path supports: (im, x) any; (x, im) and (im, im) min/max/fmod(/pow) */
    double (*fn)() = ft->fnctptr;
    int both       = a[0].vec && a[1].vec;
    int cmut =
        (fn == (double (*)()) fmin || fn == (double (*)()) fmax || fn == (double (*)()) fmod);
    if ((both && !cmut && fn != (double (*)()) pow) || (!a[0].vec && !cmut))
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

/** c_coord() - ix/iy/ir/itheta(im[, cx, cy]): im (a plain image) gives the size, not pixels. */
static int c_coord(fprog *p, const cli_token *ft, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    cli_token *t            = adv();
    fval       c[2];

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
                         .cpos = cli_calc_coord_kind(ft),
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
        if (!fusible_op(op))
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

/* apply EXPR(a, b) with a/b each either a vector slot or the constant c */
#define BIN_LOOPS(EXPR)                \
    if (in->cpos == 0)                 \
    {                                  \
        for (int j = 0; j < len; j++)  \
        {                              \
            double a = l[j], b = r[j]; \
            l[j] = (EXPR);             \
        }                              \
    }                                  \
    else if (in->cpos == 2)            \
    {                                  \
        const double b = in->c;        \
        for (int j = 0; j < len; j++)  \
        {                              \
            double a = l[j];           \
            l[j]     = (EXPR);         \
        }                              \
    }                                  \
    else                               \
    {                                  \
        const double a = in->c;        \
        for (int j = 0; j < len; j++)  \
        {                              \
            double b = l[j];           \
            l[j]     = (EXPR);         \
        }                              \
    }

/**
 * run_bin - Execute a binary vector instruction (`FI_BIN`) over a pixel block
 * @in:  Instruction specifying the operator (`in->op`), constant mode (`in->cpos`), and `in->c`
 * @l:   Left/in-place vector stack slot of length @len
 * @r:   Right vector stack slot of length @len (used when `in->cpos == 0`)
 * @len: Number of elements in the current block
 *
 * Evaluates arithmetic and comparison operators across @len elements, using
 * algebraic fast paths for common exponents (`^2`, `^0.5`, `^3`, `^-1`) to
 * avoid calling `pow()` in the inner loop.
 *
 * Return: void
 */
static void run_bin(const fins *in, double *restrict l, const double *restrict r, int len)
{
    switch (in->op)
    {
    case TOK_OP_PLUS:
        BIN_LOOPS(a + b);
        break;
    case TOK_OP_MINUS:
        BIN_LOOPS(a - b);
        break;
    case TOK_OP_STAR:
        BIN_LOOPS(a * b);
        break;
    case TOK_OP_SLASH:
        BIN_LOOPS(a / b);
        break;
    case TOK_OP_MOD:
        BIN_LOOPS(fmod(a, b));
        break;
    case TOK_OP_CARET:
        if (in->cpos == 2 && in->c == 2.0)
        {
            for (int j = 0; j < len; j++)
            {
                double a = l[j];
                l[j]     = a * a;
            }
        }
        else if (in->cpos == 2 && in->c == 0.5)
        {
            for (int j = 0; j < len; j++)
            {
                l[j] = sqrt(l[j]);
            }
        }
        else if (in->cpos == 2 && in->c == 3.0)
        {
            for (int j = 0; j < len; j++)
            {
                double a = l[j];
                l[j]     = a * a * a;
            }
        }
        else if (in->cpos == 2 && in->c == -1.0)
        {
            for (int j = 0; j < len; j++)
            {
                l[j] = 1.0 / l[j];
            }
        }
        else
        {
            BIN_LOOPS(pow(a, b));
        }
        break;
    case TOK_OP_LT:
        BIN_LOOPS((double) (a < b));
        break;
    case TOK_OP_LE:
        BIN_LOOPS((double) (a <= b));
        break;
    case TOK_OP_GT:
        BIN_LOOPS((double) (a > b));
        break;
    case TOK_OP_GE:
        BIN_LOOPS((double) (a >= b));
        break;
    case TOK_OP_EQ:
        BIN_LOOPS((double) (a == b));
        break;
    default:
        BIN_LOOPS((double) (a != b));
        break;
    }
}

/**
 * run_fn2 - Execute a two-argument math function instruction (`FI_FN2`) over a pixel block
 * @in:  Instruction specifying the function pointer (`in->fn`), `in->cpos`, and `in->c`
 * @l:   Left/in-place vector stack slot of length @len
 * @r:   Right vector stack slot of length @len (used when `in->cpos == 0`)
 * @len: Number of elements in the current block
 *
 * Inlines `fmin` and `fmax` comparisons for vectorization and dispatches other
 * binary math functions (`atan2`, `fmod`, `pow`) through `in->fn`.
 *
 * Return: void
 */
static void run_fn2(const fins *in, double *restrict l, const double *restrict r, int len)
{
    double (*f)(double, double) = (double (*)(double, double)) in->fn;
    if (f == fmin)
    {
        BIN_LOOPS((a < b) ? a : b);
    }
    else if (f == fmax)
    {
        BIN_LOOPS((a > b) ? a : b);
    }
    else
    {
        BIN_LOOPS(f(a, b));
    }
}
#undef BIN_LOOPS

/** coord_block() - coordinate @kind of pixels [@off, @off + @len) of an @nx x @ny (x n) image about (cx, cy). */
static void coord_block(int      kind,
                        uint32_t nx,
                        uint32_t ny,
                        double   cx,
                        double   cy,
                        uint64_t off,
                        int      len,
                        double *restrict t)
{
    uint32_t x = (uint32_t) (off % nx);
    uint32_t y = (uint32_t) ((off / nx) % ny);

    for (int j = 0; j < len; j++)
    {
        double dx = (double) x - cx;
        double dy = (double) y - cy;
        t[j]      = (kind == CALC_COORD_X)   ? dx
                    : (kind == CALC_COORD_Y) ? dy
                    : (kind == CALC_COORD_R) ? sqrt(dx * dx + dy * dy)
                                             : atan2(dy, dx);
        if (++x == nx)
        {
            x = 0;
            y = (y + 1 == ny) ? 0 : y + 1;
        }
    }
}

#define SEL_LOOP(T, F)                    \
    for (int j = 0; j < len; j++)         \
    {                                     \
        c[j] = (c[j] != 0.0) ? (T) : (F); \
    }

/** run_sel() - c := c ? t : f, with t or f the constant in->c / in->c2 when not on the stack. */
static void run_sel(const fins *in,
                    double *restrict c,
                    const double *restrict tv,
                    const double *restrict fv,
                    int len)
{
    switch (in->cpos)
    {
    case 0:
        SEL_LOOP(in->c, in->c2);
        break;
    case 1:
        SEL_LOOP(tv[j], in->c2);
        break;
    case 2:
        SEL_LOOP(in->c, fv[j]);
        break;
    default:
        SEL_LOOP(tv[j], fv[j]);
        break;
    }
}
#undef SEL_LOOP

/** eval_scalars() - evaluate per-frame scalar variable sub-expressions into instruction constants. */
static void eval_scalars(fprog *p)
{
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

/** run_block() - evaluate @p on pixels [@off, @off + @len) and store into @out. */
static void run_block(const fprog *p, IMAGE *out, uint64_t off, int len, double (*stk)[FUSE_BLK])
{
    int sp = 0;
    for (int i = 0; i < p->n; i++)
    {
        const fins *in = &p->ins[i];
        double     *t  = stk[(sp > 0) ? sp - 1 : 0];
        switch (in->k)
        {
        case FI_IMG:
            load_block(p->img[in->img], off, len, stk[sp++]);
            break;
        case FI_BIN:
            if (in->cpos == 0)
            {
                run_bin(in, stk[sp - 2], t, len);
                sp--;
            }
            else
            {
                run_bin(in, t, NULL, len);
            }
            break;
        case FI_FN2:
            if (in->cpos == 0)
            {
                run_fn2(in, stk[sp - 2], t, len);
                sp--;
            }
            else
            {
                run_fn2(in, t, NULL, len);
            }
            break;
        case FI_NEG:
            for (int j = 0; j < len; j++)
            {
                t[j] = 0.0 - t[j];
            }
            break;
        case FI_NOT:
            for (int j = 0; j < len; j++)
            {
                t[j] = (double) (t[j] == 0.0);
            }
            break;
        case FI_FN1:
        {
            double (*f)(double) = (double (*)(double)) in->fn;
            if (f == fabs)
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = fabs(t[j]);
                }
            }
            else if (f == sqrt)
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = sqrt(t[j]);
                }
            }
            else if (f == floor)
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = floor(t[j]);
                }
            }
            else if (f == ceil)
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = ceil(t[j]);
                }
            }
            else if (f == round)
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = round(t[j]);
                }
            }
            else if (f == cli_calc_isnan)
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = cli_calc_isnan(t[j]);
                }
            }
            else if (f == cli_calc_isfinite)
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = cli_calc_isfinite(t[j]);
                }
            }
            else
            {
                for (int j = 0; j < len; j++)
                {
                    t[j] = f(t[j]);
                }
            }
            break;
        }
        case FI_FN3:
        {
            double lo = in->c;
            double hi = in->c2;
            for (int j = 0; j < len; j++)
            {
                double x = t[j];
                t[j]     = (x < lo) ? lo : (x > hi) ? hi : x;
            }
            break;
        }
        case FI_SEL:
        {
            int nv = (in->cpos & 1) + (in->cpos >> 1);
            run_sel(in, stk[sp - 1 - nv], (in->cpos & 1) ? stk[sp - nv] : NULL,
                    (in->cpos & 2) ? stk[sp - 1] : NULL, len);
            sp -= nv;
            break;
        }
        case FI_CONST:
        {
            double  x = in->c;
            double *d = stk[sp++];
            for (int j = 0; j < len; j++)
            {
                d[j] = x;
            }
            break;
        }
        default:
        {
            const IMAGE *im = p->img[in->img];
            coord_block(in->cpos, im->md[0].size[0], (im->md[0].naxis > 1) ? im->md[0].size[1] : 1,
                        in->c, in->c2, off, len, stk[sp++]);
            break;
        }
        }
    }
    store_block(out, off, len, stk[0], 0, 0.0);
}

/**
 * run_program - Execute a compiled fused program across all blocks of an output image
 * @p:   Compiled fused program to execute
 * @out: Destination image receiving the evaluated pixels
 *
 * Evaluates per-frame scalar tape instructions (`eval_scalars()`) and then
 * processes @out in `FUSE_BLK`-pixel chunks, parallelizing across blocks with
 * OpenMP when `nelement >= FUSE_OMP_MIN`.
 *
 * Return: void
 */
static void run_program(fprog *p, IMAGE *out)
{
    eval_scalars(p);
    uint64_t nel  = out->md[0].nelement;
    uint64_t nblk = (nel + FUSE_BLK - 1) / FUSE_BLK;
#pragma omp parallel if (nel >= FUSE_OMP_MIN)
    {
        MILK_ALIGNED(64) double stk[FUSE_MAXSTK][FUSE_BLK];
#pragma omp for schedule(static)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * FUSE_BLK;
            int      len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
            run_block(p, out, off, len, stk);
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
 * cli_calc_coord_kind - Map a `TOK_FUNC_COORD` token to its `CALC_COORD_*` enum value
 * @t: Coordinate function token (`ix(`, `iy(`, `ir(`, or `itheta(`)
 *
 * Inspects `t->sval` to identify which pixel coordinate generator was called.
 *
 * Return: `CALC_COORD_X`, `CALC_COORD_Y`, `CALC_COORD_R`, or `CALC_COORD_THETA`
 */
int cli_calc_coord_kind(const cli_token *t)
{
    return (strncmp(t->sval, "ix(", 3) == 0)   ? CALC_COORD_X
           : (strncmp(t->sval, "iy(", 3) == 0) ? CALC_COORD_Y
           : (strncmp(t->sval, "ir(", 3) == 0) ? CALC_COORD_R
                                               : CALC_COORD_THETA;
}

enum
{
    FKEPT_ELEMWISE  = 0,
    FKEPT_MATMUL    = 1,
    FKEPT_TRANSPOSE = 2,
    FKEPT_RESHAPE   = 3
};

typedef struct
{
    fprog           p;
    IMAGE          *dst;
    uint64_t        last_cnt0[FUSE_MAXIMG];
    struct timespec last_check;
    int             kind;
    int             transA;
    int             transB;
    int             allow_reshape;
} fkept;

/**
 * parse_kept_img_or_trans - Parse `im` or `transpose(im)` as an operand of a kept `matmul()`
 * @out_im:    Output pointer to the resolved `IMAGE`
 * @out_trans: Output flag set to 1 if wrapped in `transpose(...)`, 0 otherwise
 *
 * Consumes a plain image token or `transpose(im)` from `eval_tokens` so
 * `matmul(transpose(A), B)` can be executed directly via BLAS transpose flags
 * without materializing an intermediate transposed image.
 *
 * Return: 0 on success, -1 if the tokens do not match `im` or `transpose(im)`
 */
static int parse_kept_img_or_trans(IMAGE **out_im, int *out_trans)
{
    int trans = 0;
    if (cur_eval()->type == TOK_FUNC_IMIM_IM && strcmp(cur_eval()->sval, "transpose(") == 0)
    {
        advance_eval();
        trans = 1;
    }
    if (cur_eval()->type != TOK_IMAGE || strchr(cur_eval()->sval, '[') != NULL)
    {
        return -1;
    }
    imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
    if (id == -1)
    {
        return -1;
    }
    advance_eval();
    if (trans)
    {
        if (cur_eval()->type != TOK_RPAREN)
        {
            return -1;
        }
        advance_eval();
    }
    *out_im    = &data.core.image[id];
    *out_trans = trans;
    return 0;
}

/**
 * parse_kept_reshape_dims - Parse and validate target dimensions of a kept `reshape(...)` call
 * @nel: Total number of elements expected in the reshaped image
 * @dst: Pre-allocated destination image whose shape must match the parsed dimensions
 *
 * Parses up to 3 comma-separated constant dimension arguments (allowing at most
 * one `-1` inferred axis) and verifies that the resulting shape matches @dst.
 *
 * Return: 0 if the parsed dimensions are valid and match @dst, -1 otherwise
 */
static int parse_kept_reshape_dims(uint64_t nel, const IMAGE *dst)
{
    int      ndims    = 0;
    long     dims[3]  = { 1, 1, 1 };
    int      neg_idx  = -1;
    uint64_t prod_pos = 1;

    while (cur_eval()->type == TOK_COMMA && ndims < 3)
    {
        advance_eval();
        fprog tp;
        fval  dv;
        memset(&tp, 0, sizeof(tp));
        if (c_expr(&tp, 0, &dv) != 0 || dv.vec || dv.s1 != 0)
        {
            return -1;
        }
        long d = llround(dv.c);
        if (d == -1)
        {
            if (neg_idx != -1)
            {
                return -1;
            }
            neg_idx = ndims;
        }
        else if (d <= 0)
        {
            return -1;
        }
        else
        {
            dims[ndims] = d;
            prod_pos *= (uint64_t) d;
        }
        ndims++;
    }
    if (ndims == 0 || cur_eval()->type != TOK_RPAREN)
    {
        return -1;
    }
    advance_eval();
    if (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        return -1;
    }
    if (neg_idx != -1)
    {
        if (prod_pos == 0 || nel % prod_pos != 0)
        {
            return -1;
        }
        dims[neg_idx] = (long) (nel / prod_pos);
        prod_pos      = nel;
    }
    if (prod_pos != nel || dst->md[0].naxis != ndims)
    {
        return -1;
    }
    for (int i = 0; i < ndims; i++)
    {
        if (dst->md[0].size[i] != (uint32_t) dims[i])
        {
            return -1;
        }
    }
    return 0;
}

/**
 * compile_kept_matrix_op - Try to compile a top-level `transpose`, `matmul`, or `reshape` into @k
 * @k:   Kept-statement structure to populate
 * @dst: Destination image assigned by the statement
 *
 * Recognizes `dst = transpose(A)`, `dst = matmul(A, B)` (with optional
 * `transpose()` on either operand), `dst = reshape(matmul(...), ...)`, and
 * `dst = reshape(A, ...)` so `milk.calc` can run them directly into @dst
 * without per-frame temporary allocations.
 *
 * Return: 0 on successful compilation, -1 if the RHS is not a supported matrix/reshape op
 */
static int compile_kept_matrix_op(fkept *k, IMAGE *dst)
{
    int pos = cli_calc_tok_pos();
    if (cur_eval()->type != TOK_FUNC_IMIM_IM)
    {
        return -1;
    }
    if (strcmp(cur_eval()->sval, "transpose(") == 0)
    {
        advance_eval();
        if (cur_eval()->type != TOK_IMAGE || strchr(cur_eval()->sval, '[') != NULL)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
        advance_eval();
        if (id == -1 || cur_eval()->type != TOK_RPAREN)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if ((cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE) ||
            cli_calc_transpose_check(&data.core.image[id], dst) != 0)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        k->kind     = FKEPT_TRANSPOSE;
        k->p.img[0] = &data.core.image[id];
        k->p.nimg   = 1;
        return 0;
    }
    if (strcmp(cur_eval()->sval, "matmul(") == 0)
    {
        advance_eval();
        IMAGE *imA    = NULL;
        IMAGE *imB    = NULL;
        int    transA = 0;
        int    transB = 0;
        if (parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_RPAREN)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if ((cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE) ||
            cli_calc_matmul_check(imA, transA, imB, transB, dst, 0) != 0)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        k->kind          = FKEPT_MATMUL;
        k->transA        = transA;
        k->transB        = transB;
        k->allow_reshape = 0;
        k->p.img[0]      = imA;
        k->p.img[1]      = imB;
        k->p.nimg        = 2;
        return 0;
    }
    if (strcmp(cur_eval()->sval, "reshape(") == 0)
    {
        advance_eval();
        if (cur_eval()->type == TOK_FUNC_IMIM_IM && strcmp(cur_eval()->sval, "matmul(") == 0)
        {
            advance_eval();
            IMAGE *imA    = NULL;
            IMAGE *imB    = NULL;
            int    transA = 0;
            int    transB = 0;
            if (parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            advance_eval();
            if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_RPAREN)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            advance_eval();
            if (cli_calc_matmul_check(imA, transA, imB, transB, dst, 1) != 0 ||
                parse_kept_reshape_dims(dst->md[0].nelement, dst) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            k->kind          = FKEPT_MATMUL;
            k->transA        = transA;
            k->transB        = transB;
            k->allow_reshape = 1;
            k->p.img[0]      = imA;
            k->p.img[1]      = imB;
            k->p.nimg        = 2;
            return 0;
        }
        if (cur_eval()->type == TOK_IMAGE && strchr(cur_eval()->sval, '[') == NULL)
        {
            imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
            advance_eval();
            if (id == -1 || cli_calc_reshape_check(&data.core.image[id], dst) != 0 ||
                parse_kept_reshape_dims(dst->md[0].nelement, dst) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            k->kind     = FKEPT_RESHAPE;
            k->p.img[0] = &data.core.image[id];
            k->p.nimg   = 1;
            return 0;
        }
        cli_calc_tok_seek(pos);
        return -1;
    }
    return -1;
}

/**
 * cli_calc_fuse_keep - Compile an image assignment RHS into a persistent `fkept` program
 * @dst: Destination image being assigned
 * @aop: Assignment operator (`=`, `+=`, `-=`, or `*=`)
 *
 * Compiles either a direct matrix/reshape operation (`compile_kept_matrix_op()`)
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
    if (aop == TOK_EQUAL && compile_kept_matrix_op(k, dst) == 0)
    {
        k->dst = dst;
        for (int i = 0; i < k->p.nimg; i++)
        {
            k->last_cnt0[i] = k->p.img[i]->md[0].cnt0;
        }
        clock_gettime(CLOCK_MONOTONIC_COARSE, &k->last_check);
        return k;
    }
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
        if (k->kind == FKEPT_ELEMWISE &&
            (!im->used || !is_real_datatype(im->md[0].datatype) || !same_shape(im, k->dst)))
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                     im->name);
            return -1;
        }
    }
    if (k->kind == FKEPT_MATMUL)
    {
        if (cli_calc_matmul_check(k->p.img[0], k->transA, k->p.img[1], k->transB, k->dst,
                                  k->allow_reshape) != 0)
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr),
                     "matmul: %s or input changed size or type", k->dst->name);
            return -1;
        }
        k->dst->md[0].write = 1;
        if (cli_calc_matmul_into(k->p.img[0], k->transA, k->p.img[1], k->transB, k->dst) != 0)
        {
            k->dst->md[0].write = 0;
            return -1;
        }
        processinfo_update_output_stream(NULL, k->dst, NULL);
        return 0;
    }
    if (k->kind == FKEPT_TRANSPOSE)
    {
        if (cli_calc_transpose_check(k->p.img[0], k->dst) != 0)
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr),
                     "transpose: %s or input changed size or type", k->dst->name);
            return -1;
        }
        k->dst->md[0].write = 1;
        if (cli_calc_transpose_into(k->p.img[0], k->dst) != 0)
        {
            k->dst->md[0].write = 0;
            return -1;
        }
        processinfo_update_output_stream(NULL, k->dst, NULL);
        return 0;
    }
    if (k->kind == FKEPT_RESHAPE)
    {
        if (cli_calc_reshape_check(k->p.img[0], k->dst) != 0)
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr),
                     "reshape: %s or input changed size or type", k->dst->name);
            return -1;
        }
        k->dst->md[0].write = 1;
        if (cli_calc_reshape_into(k->p.img[0], k->dst) != 0)
        {
            k->dst->md[0].write = 0;
            return -1;
        }
        processinfo_update_output_stream(NULL, k->dst, NULL);
        return 0;
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
    static fprog p;
    fval         v;
    int          pos = cli_calc_tok_pos();
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
