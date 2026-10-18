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
    FI_SVAR   /* push the value calculator variable img has when the program runs (kept programs) */
} fi_kind;

typedef struct
{
    fi_kind        k;
    cli_token_type op;
    int            img;
    int            cpos;
    double         c;
    double         c2;
    double (*fn)();
} fins;

typedef struct
{
    fins         ins[FUSE_MAXINS];
    int          n;
    const IMAGE *img[FUSE_MAXIMG];
    int          nimg;
    int          depth;
    int          maxdepth;
    int          nops; /* vector operations emitted */
    int          out_double;
} fprog;

static int keep_vars; /* compiling a kept program: variables are read at each run, not folded */

/** Compile-time value: a folded constant, or code that leaves a vector on the stack. */
typedef struct
{
    int    vec;
    double c;
} fval;

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

static int is_real_datatype(uint8_t dt)
{
    return dt != _DATATYPE_COMPLEX_FLOAT && dt != _DATATYPE_COMPLEX_DOUBLE &&
           dt != _DATATYPE_HALF && ImageStreamIO_typesize(dt) > 0;
}

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

static int c_func(fprog *p, const cli_token *ft, fval *out)
{
    fval a[3];
    int  nargs = (ft->type == TOK_FUNC_D_D) ? 1 : (ft->type == TOK_FUNC_DD_D) ? 2 : 3;
    if (c_args(p, nargs, a) != 0)
    {
        return -1;
    }
    out->vec = a[0].vec || (nargs > 1 && a[1].vec) || (nargs > 2 && a[2].vec);
    if (ft->type == TOK_FUNC_D_D)
    {
        out->c = out->vec ? 0 : ((double (*)(double)) ft->fnctptr)(a[0].c);
        return out->vec ? emit(p, (fins) { .k = FI_FN1, .fn = ft->fnctptr }, 0) : 0;
    }
    if (ft->type == TOK_FUNC_DDD_D) /* trunc(im, lo, hi): only the first argument may be an image */
    {
        if (a[1].vec || a[2].vec)
        {
            return -1;
        }
        out->c = out->vec
                     ? 0
                     : ((double (*)(double, double, double)) ft->fnctptr)(a[0].c, a[1].c, a[2].c);
        return out->vec
                   ? emit(p, (fins) { .k = FI_FN3, .fn = ft->fnctptr, .c = a[1].c, .c2 = a[2].c },
                          0)
                   : 0;
    }
    if (!out->vec)
    {
        out->c = ((double (*)(double, double)) ft->fnctptr)(a[0].c, a[1].c);
        return 0;
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
                .c    = a[0].vec ? a[1].c : a[0].c };
    return emit(p, in, both ? -1 : 0);
}

/** c_select() - per pixel `cond ? t : f`; the condition is on the stack under t and f. */
static int c_select(fprog *p, fval t, fval f, fval *out)
{
    out->vec = 1;
    return emit(p, (fins) { .k = FI_SEL, .cpos = t.vec | (f.vec << 1), .c = t.c, .c2 = f.c },
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
    c[0].c          = (double) (im->md[0].size[0] / 2);
    c[1].c          = (double) ((im->md[0].naxis > 1) ? im->md[0].size[1] / 2 : 0);
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
    out->vec = 1;
    return emit(
        p,
        (fins) {
            .k = FI_COORD, .img = k, .cpos = cli_calc_coord_kind(ft), .c = c[0].c, .c2 = c[1].c },
        1);
}

static int c_primary(fprog *p, fval *out)
{
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    cli_token *t            = adv();

    switch (t->type)
    {
    case TOK_LONG:
        *out = (fval) { 0, (double) t->val_l };
        return 0;
    case TOK_DOUBLE:
        *out = (fval) { 0, t->val_d };
        return 0;
    case TOK_OP_MINUS:
    case TOK_OP_NOT:
        if (c_primary(p, out) != 0)
        {
            return -1;
        }
        if (!out->vec)
        {
            out->c = (t->type == TOK_OP_MINUS) ? -out->c : (out->c == 0.0);
            return 0;
        }
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
            out->vec = 1;
            return emit(p, (fins) { .k = FI_SVAR, .img = (int) vid }, 1);
        }
        val_t v = eval_existing_var_ref(t->sval);
        if (v.type != VAL_LONG && v.type != VAL_DOUBLE)
        {
            return -1;
        }
        *out = (fval) { 0, to_double(v) };
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
        if (!out->vec && !r.vec)
        {
            if ((op == TOK_OP_SLASH || op == TOK_OP_MOD) && r.c == 0.0)
            {
                return -1; /* the normal path reports the division by zero */
            }
            out->c = scalar_bin(op, out->c, r.c);
            continue;
        }
        if (!r.vec && (op == TOK_OP_SLASH || op == TOK_OP_MOD) && r.c == 0.0)
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
                          .c    = out->vec ? r.c : out->c },
                 both ? -1 : 0) != 0)
        {
            return -1;
        }
        out->vec = 1;
    }
}

/* ---------------------------------------------------------------- execution */

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
        BIN_LOOPS(pow(a, b));
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

static void run_fn2(const fins *in, double *restrict l, const double *restrict r, int len)
{
    double (*f)(double, double) = (double (*)(double, double)) in->fn;
    BIN_LOOPS(f(a, b));
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
            for (int j = 0; j < len; j++)
            {
                t[j] = f(t[j]);
            }
            break;
        }
        case FI_FN3:
        {
            double (*f)(double, double, double) = (double (*)(double, double, double)) in->fn;
            for (int j = 0; j < len; j++)
            {
                t[j] = f(t[j], in->c, in->c2);
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
        case FI_SVAR:
        {
            const VARIABLE *v = &data.core.variable[in->img];
            double          x = (v->type == 1) ? (double) v->value.l : v->value.f;
            double         *d = stk[sp++];
            for (int j = 0; j < len; j++)
            {
                d[j] = x;
            }
            break;
        }
        case FI_COORD:
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

static void run_program(const fprog *p, IMAGE *out)
{
    uint64_t nel  = out->md[0].nelement;
    uint64_t nblk = (nel + FUSE_BLK - 1) / FUSE_BLK;
#pragma omp parallel if (nel >= FUSE_OMP_MIN)
    {
        double (*stk)[FUSE_BLK] = malloc(sizeof(double) * FUSE_MAXSTK * FUSE_BLK);
        if (stk != NULL)
        {
#pragma omp for schedule(static)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t off = b * FUSE_BLK;
                int      len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
                run_block(p, out, off, len, stk);
            }
            free(stk);
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

    const IMAGE *ref   = p.img[0];
    const char  *tmpn  = alloc_tmpname();
    imageID      tid   = -1;
    uint32_t     sz[3] = { ref->md[0].size[0], ref->md[0].size[1], ref->md[0].size[2] };
    create_image_ID(tmpn, ref->md[0].naxis, sz, p.out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT,
                    0, 10, 0, &tid);
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

int cli_calc_coord_kind(const cli_token *t)
{
    return (strncmp(t->sval, "ix(", 3) == 0)   ? CALC_COORD_X
           : (strncmp(t->sval, "iy(", 3) == 0) ? CALC_COORD_Y
           : (strncmp(t->sval, "ir(", 3) == 0) ? CALC_COORD_R
                                               : CALC_COORD_THETA;
}

typedef struct
{
    fprog  p;
    IMAGE *dst;
} fkept;

void *cli_calc_fuse_keep(IMAGE *dst, cli_token_type aop)
{
    fkept *k = calloc(1, sizeof(*k));
    fval   v;
    int    pos = cli_calc_tok_pos();

    if (k == NULL || !is_real_datatype(dst->md[0].datatype))
    {
        free(k);
        return NULL;
    }
    keep_vars = 1;
    int rc    = compile(&k->p, 0, &v);
    keep_vars = 0;
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
    return k;
}

int cli_calc_fused_run(void *kept)
{
    fkept *k = kept;

    for (int i = -1; i < k->p.nimg; i++) /* a stream re-created by its owner may have a new size */
    {
        const IMAGE *im = (i < 0) ? k->dst : k->p.img[i];
        if (!im->used || !is_real_datatype(im->md[0].datatype) || !same_shape(im, k->dst))
        {
            snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s changed size or type",
                     im->name);
            return -1;
        }
    }
    k->dst->md[0].write = 1;
    run_program(&k->p, k->dst);
    processinfo_update_output_stream(NULL, k->dst, NULL);
    return 0;
}

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
    uint8_t out_dt = p.out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
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
        double *blk = malloc(sizeof(double) * FUSE_BLK);
        int     bad = (blk == NULL);
        for (uint64_t off = 0; off < nel && !bad; off += FUSE_BLK)
        {
            int len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
            load_block(src, off, len, blk);
            for (int j = 0; j < len && !bad; j++)
            {
                bad = (blk[j] == 0.0);
            }
        }
        free(blk);
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
