// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_kernels.h
 * @brief Block kernels of the fused evaluator, written once and instantiated per element type.
 *
 * Internal to `cli_calc_fuse_exec.c`, which includes this file once per element type after defining:
 *   FT        element type of the stack (double or float)
 *   FK(name)  name of an instantiated function (e.g. name##_d)
 *   FM(fn)    the libm function of that type (fn for double, fn##f for float)
 *   FK_LOAD   load_block-like function: (const IMAGE *, uint64_t off, int len, FT *dst)
 *   FK_STORE  store_block-like function: (IMAGE *, uint64_t off, int len, const FT *src)
 *   FK_LOADV  load through operand k's view: (const fprog *, int k, uint64_t off, int len, FT *dst)
 *   FK_STOREV store through the destination view: (const fprog *, IMAGE *, uint64_t, int, const FT *)
 *
 * Math functions are called directly (never through the instruction's function pointer, except for
 * functions without a fast form), so that GCC can vectorize the loops with libmvec under -Ofast.
 * No include guard: included once per element type.
 */

/* apply EXPR(a, b) with a/b each either a vector slot or the constant c */
#define BIN_LOOPS(EXPR)                                   \
    if (in->cpos == 0)                                    \
    {                                                     \
        _Pragma("omp simd") for (int j = 0; j < len; j++) \
        {                                                 \
            const FT a = l[j], b = r[j];                  \
            l[j] = (EXPR);                                \
        }                                                 \
    }                                                     \
    else if (in->cpos == 2)                               \
    {                                                     \
        const FT b = (FT) in->c;                          \
        _Pragma("omp simd") for (int j = 0; j < len; j++) \
        {                                                 \
            const FT a = l[j];                            \
            l[j]       = (EXPR);                          \
        }                                                 \
    }                                                     \
    else                                                  \
    {                                                     \
        const FT a = (FT) in->c;                          \
        _Pragma("omp simd") for (int j = 0; j < len; j++) \
        {                                                 \
            const FT b = l[j];                            \
            l[j]       = (EXPR);                          \
        }                                                 \
    }

/* dst[j] = EXPR(x) with x = src[j] */
#define MAP_LOOP(EXPR)                                \
    _Pragma("omp simd") for (int j = 0; j < len; j++) \
    {                                                 \
        const FT x = src[j];                          \
        dst[j]     = (EXPR);                          \
    }

/**
 * FK(run_bin) - Execute a binary operator instruction (`FI_BIN`) over a pixel block
 * @in:  Instruction: operator, operand layout (`cpos`) and constant operand (`c`)
 * @l:   Left / in-place vector stack slot of length @len
 * @r:   Right vector stack slot (used when `in->cpos == 0`)
 * @len: Number of elements in the block
 *
 * Return: void
 */
static void FK(run_bin)(const fins *in, FT *restrict l, const FT *restrict r, int len)
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
        BIN_LOOPS(FM(fmod)(a, b));
        break;
    case TOK_OP_CARET:
        if (in->cpos == 2)
        {
            FK(elem_pow_const)(l, l, in->c, len);
        }
        else
        {
            BIN_LOOPS(FM(pow)(a, b));
        }
        break;
    case TOK_OP_LT:
        BIN_LOOPS((FT) (a < b));
        break;
    case TOK_OP_LE:
        BIN_LOOPS((FT) (a <= b));
        break;
    case TOK_OP_GT:
        BIN_LOOPS((FT) (a > b));
        break;
    case TOK_OP_GE:
        BIN_LOOPS((FT) (a >= b));
        break;
    case TOK_OP_EQ:
        BIN_LOOPS((FT) (a == b));
        break;
    case TOK_OP_NEQ:
        BIN_LOOPS((FT) (a != b));
        break;
    case TOK_OP_AND:
        BIN_LOOPS((FT) ((a != (FT) 0) & (b != (FT) 0)));
        break;
    case TOK_OP_BAND:
        BIN_LOOPS((FT) ((int64_t) a & (int64_t) b));
        break;
    case TOK_OP_BOR:
        BIN_LOOPS((FT) ((int64_t) a | (int64_t) b));
        break;
    case TOK_OP_BXOR:
        BIN_LOOPS((FT) ((int64_t) a ^ (int64_t) b));
        break;
    case TOK_OP_LSHIFT:
        BIN_LOOPS((FT) (((int64_t) b >= 0 && (int64_t) b < 64)
                            ? (int64_t) ((uint64_t) (int64_t) a << (int64_t) b)
                            : 0));
        break;
    case TOK_OP_RSHIFT:
        BIN_LOOPS((FT) (((int64_t) b >= 0 && (int64_t) b < 64) ? ((int64_t) a >> (int64_t) b)
                                                               : (((int64_t) a < 0) ? -1 : 0)));
        break;
    default:
        BIN_LOOPS((FT) ((a != (FT) 0) | (b != (FT) 0)));
        break;
    }
}

/**
 * FK(run_fn2) - Execute a two-argument function instruction (`FI_FN2`) over a pixel block
 * @in:  Instruction: function (`in->fn`), operand layout (`cpos`) and constant operand (`c`)
 * @l:   Left / in-place vector stack slot of length @len
 * @r:   Right vector stack slot (used when `in->cpos == 0`)
 * @len: Number of elements in the block
 *
 * Return: void
 */
static void FK(run_fn2)(const fins *in, FT *restrict l, const FT *restrict r, int len)
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
    else if (f == fmod)
    {
        BIN_LOOPS(FM(fmod)(a, b));
    }
    else if (f == atan2)
    {
        BIN_LOOPS(FM(atan2)(a, b));
    }
    else if (f == hypot)
    {
        BIN_LOOPS(FM(sqrt)(a * a + b * b));
    }
    else if (f == pow)
    {
        if (in->cpos == 2)
        {
            FK(elem_pow_const)(l, l, in->c, len);
        }
        else
        {
            BIN_LOOPS(FM(pow)(a, b));
        }
    }
    else
    {
        BIN_LOOPS((FT) f((double) a, (double) b));
    }
}

/**
 * FK(run_fn1) - Execute a one-argument function instruction (`FI_FN1`) over a pixel block
 * @in:  The instruction: its `fn1` kind selects the vector form, `fn` the fallback
 * @src: Source vector buffer of length @len
 * @dst: Destination vector buffer of length @len (may equal @src)
 * @len: Number of elements in the block
 *
 * Return: void
 */
static void FK(run_fn1)(const fins *in, const FT *src, FT *dst, int len)
{
#define FN1_CASE(KIND, NAME)   \
    case KIND:                 \
        MAP_LOOP(FM(NAME)(x)); \
        return;
    switch (in->fn1)
    {
        FN1_CASE(FN1_FABS, fabs)
        FN1_CASE(FN1_SQRT, sqrt)
        FN1_CASE(FN1_FLOOR, floor)
        FN1_CASE(FN1_CEIL, ceil)
        FN1_CASE(FN1_ROUND, round)
        FN1_CASE(FN1_SIN, sin)
        FN1_CASE(FN1_COS, cos)
        FN1_CASE(FN1_TAN, tan)
        FN1_CASE(FN1_SINH, sinh)
        FN1_CASE(FN1_COSH, cosh)
        FN1_CASE(FN1_TANH, tanh)
        FN1_CASE(FN1_EXP, exp)
        FN1_CASE(FN1_EXP2, exp2)
        FN1_CASE(FN1_LOG, log)
        FN1_CASE(FN1_LOG10, log10)
        FN1_CASE(FN1_LOG2, log2)
        FN1_CASE(FN1_ERF, erf)
        FN1_CASE(FN1_ERFC, erfc)
        FN1_CASE(FN1_ATAN, atan)
        FN1_CASE(FN1_ASIN, asin)
        FN1_CASE(FN1_ACOS, acos)
        FN1_CASE(FN1_CBRT, cbrt)
    case FN1_POSI:
        MAP_LOOP((FT) (x > (FT) 0));
        return;
    case FN1_SIGN:
        MAP_LOOP((FT) ((x > (FT) 0) - (x < (FT) 0)));
        return;
    default:
        break;
    }
#undef FN1_CASE
    /* casts and the bit-test NaN checks (-Ofast would fold isnan() away): through the pointer, in double */
    double (*f)(double) = (double (*)(double)) in->fn;
    MAP_LOOP((FT) f((double) x));
}

/* IMG_LOOP(EXPR): d[j] = EXPR of x = (FT) src[j] over image im's element type */
#define IMG_BODY_(MBR, EXPR)                                           \
    {                                                                  \
        const __typeof__(im->array.MBR[0]) *src = im->array.MBR + off; \
        _Pragma("omp simd") for (int j = 0; j < len; j++)              \
        {                                                              \
            const FT x = (FT) src[j];                                  \
            d[j]       = (EXPR);                                       \
        }                                                              \
    }
#define IMG_ADD(MBR) IMG_BODY_(MBR, x + c)
#define IMG_SUB(MBR) IMG_BODY_(MBR, x - c)
#define IMG_RSUB(MBR) IMG_BODY_(MBR, c - x)
#define IMG_MUL(MBR) IMG_BODY_(MBR, x *c)
#define IMG_DIV(MBR) IMG_BODY_(MBR, x / c)
#define IMG_RDIV(MBR) IMG_BODY_(MBR, c / x)
#define IMG_LOOP(B) MILK_FOR_EACH_DATATYPE(dt, B, B(D))

/**
 * FK(run_imgc) - `FI_IMGC`: push image @in->img op c (or c op image) in one pass
 */
static void FK(run_imgc)(const fprog *p, const fins *in, uint64_t off, int len, FT *restrict d)
{
    const IMAGE *im = p->img[in->img];
    if (p->view[in->img].is_view)
    {
        FK_LOADV(p, in->img, off, len, d);
        FK(run_bin)(in, d, NULL, len);
        return;
    }
    const FT c  = (FT) in->c;
    uint8_t  dt = im->md[0].datatype;
    switch (in->op)
    {
    case TOK_OP_PLUS:
        IMG_LOOP(IMG_ADD);
        break;
    case TOK_OP_MINUS:
        if (in->cpos == 2)
        {
            IMG_LOOP(IMG_SUB);
        }
        else
        {
            IMG_LOOP(IMG_RSUB);
        }
        break;
    case TOK_OP_STAR:
        IMG_LOOP(IMG_MUL);
        break;
    default:
        if (in->cpos == 2)
        {
            IMG_LOOP(IMG_DIV);
        }
        else
        {
            IMG_LOOP(IMG_RDIV);
        }
        break;
    }
}

/* VIMG_LOOP: l[j] = l[j] OP ((FT) src[j] * c) over image im's element type */
#define VIMG_BODY_(MBR, OP)                                            \
    {                                                                  \
        const __typeof__(im->array.MBR[0]) *src = im->array.MBR + off; \
        _Pragma("omp simd") for (int j = 0; j < len; j++)              \
        {                                                              \
            l[j] = l[j] OP((FT) src[j] * c);                           \
        }                                                              \
    }
#define VIMG_ADD(MBR) VIMG_BODY_(MBR, +)
#define VIMG_SUB(MBR) VIMG_BODY_(MBR, -)
#define VIMG_MUL(MBR) VIMG_BODY_(MBR, *)
#define VIMG_DIV(MBR) VIMG_BODY_(MBR, /)

/**
 * FK(run_vimgc) - `FI_VIMGC`: top @l = @l op (image @in->img * c), @scratch for a view operand
 */
static void FK(run_vimgc)(const fprog *p,
                          const fins  *in,
                          uint64_t     off,
                          int          len,
                          FT *restrict l,
                          FT *restrict scratch)
{
    const IMAGE *im = p->img[in->img];
    const FT     c  = (FT) in->c;
    if (p->view[in->img].is_view)
    {
        FK_LOADV(p, in->img, off, len, scratch);
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            scratch[j] *= c;
        }
        fins b = { .k = FI_BIN, .op = in->op, .cpos = 0 };
        FK(run_bin)(&b, l, scratch, len);
        return;
    }
    uint8_t dt = im->md[0].datatype;
    switch (in->op)
    {
    case TOK_OP_PLUS:
        IMG_LOOP(VIMG_ADD);
        break;
    case TOK_OP_MINUS:
        IMG_LOOP(VIMG_SUB);
        break;
    case TOK_OP_STAR:
        IMG_LOOP(VIMG_MUL);
        break;
    default:
        IMG_LOOP(VIMG_DIV);
        break;
    }
}

/**
 * FK(run_imgimg) - `FI_IMGIMG`: push image @in->img op image @in->cpos in one pass (@scratch for b)
 */
static void FK(run_imgimg)(const fprog *p,
                           const fins  *in,
                           uint64_t     off,
                           int          len,
                           FT *restrict d,
                           FT *restrict scratch)
{
    const IMAGE *im = p->img[in->img];
    const IMAGE *bm = p->img[in->cpos];
    if (!p->view[in->img].is_view && !p->view[in->cpos].is_view && im->md[0].datatype == FK_DT &&
        bm->md[0].datatype == FK_DT)
    {
        const FT *a = (const FT *) ((const char *) im->array.raw) + off;
        const FT *b = (const FT *) ((const char *) bm->array.raw) + off;
        switch (in->op)
        {
        case TOK_OP_PLUS:
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                d[j] = a[j] + b[j];
            }
            break;
        case TOK_OP_MINUS:
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                d[j] = a[j] - b[j];
            }
            break;
        case TOK_OP_STAR:
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                d[j] = a[j] * b[j];
            }
            break;
        default:
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                d[j] = a[j] / b[j];
            }
            break;
        }
        return;
    }
    if (p->view[in->img].is_view)
    {
        FK_LOADV(p, in->img, off, len, d);
    }
    else
    {
        FK_LOAD(im, off, len, d);
    }
    fins b = { .k = FI_VIMGC, .op = in->op, .img = in->cpos, .c = 1.0 };
    FK(run_vimgc)(p, &b, off, len, d, scratch);
}

#undef IMG_BODY_
#undef IMG_ADD
#undef IMG_SUB
#undef IMG_RSUB
#undef IMG_MUL
#undef IMG_DIV
#undef IMG_RDIV
#undef VIMG_BODY_
#undef VIMG_ADD
#undef VIMG_SUB
#undef VIMG_MUL
#undef VIMG_DIV
#undef IMG_LOOP

#include "COREMOD_arith/image_medfilt_block.h"

void FK(cli_calc_fuse_medfilt_cols)(int K, int N, FT col[7][FUSE_BLK + 6], FT *restrict d)
{
    FK(arith_medfilt_cols)(K, N, col, d);
}

/**
 * FK(eval_spatial_ins) - Execute spatial, slot, extra-dst, or user instruction @in on block
 */
static inline int FK(eval_spatial_ins)(const fprog *p,
                                       const fins  *in,
                                       uint64_t     off,
                                       int          len,
                                       FT (*stk)[FUSE_BLK],
                                       int sp)
{
    FT *t = stk[(sp > 0) ? sp - 1 : 0];
    switch (in->k)
    {
    case FI_CPART:
        FK(cplx_part_block)(p->img[in->img], in->cpos, off, len, stk[sp++]);
        break;
    case FI_SHIFT:
        FK_SHIFT(p, in->img, (int64_t) llround(in->c), (int64_t) llround(in->c2), off, len,
                 stk[sp++]);
        break;
    case FI_LUT:
        FK_LUT(p->img[in->img], in->c, in->c2, len, t);
        break;
    case FI_MEDFILT:
        FK_MEDFILT(p, in->img, in->cpos, off, len, stk[sp++]);
        break;
    case FI_GATHER:
        FK_GATHER(p->img[in->img], in->c, len, t);
        break;
    case FI_INTERP2:
        FK_INTERP2(p->img[in->img], len, stk[sp - 2], t);
        sp--;
        break;
    case FI_STORE_SLOT:
        __builtin_memcpy(stk[FUSE_MAXSTK - 1 - in->cpos], t, (size_t) len * sizeof(FT));
        sp--;
        break;
    case FI_LOAD_SLOT:
        __builtin_memcpy(stk[sp++], stk[FUSE_MAXSTK - 1 - in->cpos], (size_t) len * sizeof(FT));
        break;
    case FI_STORE_DST:
        FK_STORE(p->extra_dst[in->img], off, len, t);
        if (in->cpos == 0)
        {
            sp--;
        }
        break;
    case FI_USER_BLK:
        if (in->cpos == 2)
        {
            FK_USER_BLK(p, in->img, len, stk[sp - 2], t);
            sp--;
        }
        else
        {
            FK_USER_BLK(p, in->img, len, t, NULL);
        }
        break;
    case FI_USER_STENCIL:
        FK_USER_STENCIL(p, in->img, in->cpos, off, len, stk[sp++]);
        break;
    case FI_MATMUL:
        FK_MATMUL(p, in->cpos, off, len, stk[sp++]);
        break;
    case FI_EYE:
        FK_EYE(p, (int64_t) llround(in->c), off, len, stk[sp++]);
        break;
    default: /* FI_COORD: coordinates of the iteration space (the image's own when it is whole) */
        FK(coord_block)(in->cpos, p->ishape[0], p->ishape[1], in->c, in->c2, off, len, stk[sp++]);
        break;
    } /* switch (in->k) */
    return sp;
}

/**
 * FK(eval_block) - Evaluate program @p on pixels [@off, @off + @len) leaving result in `stk[0]`
 * @p:   Compiled program (scalar constants already evaluated)
 * @off: First pixel of the block
 * @len: Number of pixels (`<= FUSE_BLK`)
 * @stk: Per-thread vector stack, FUSE_MAXSTK slots of FUSE_BLK elements
 *
 * Return: void
 */
static void FK(eval_block)(const fprog *p, uint64_t off, int len, FT (*stk)[FUSE_BLK])
{
    int sp = 0;
    for (int i = 0; i < p->n; i++)
    {
        const fins *in = &p->ins[i];
        FT         *t  = stk[(sp > 0) ? sp - 1 : 0];
        switch (in->k)
        {
        case FI_IMG:
            if (p->view[in->img].is_view)
            {
                FK_LOADV(p, in->img, off, len, stk[sp++]);
            }
            else if (p->img[in->img]->md[0].datatype == FK_DT && i + 1 < p->n &&
                     (p->ins[i + 1].k == FI_FN1 || p->ins[i + 1].k == FI_FN3 ||
                      p->ins[i + 1].k == FI_NEG))
            {
                const FT   *src = (const FT *) p->img[in->img]->array.raw + off;
                FT         *d   = stk[sp++];
                const fins *nx  = &p->ins[++i];
                if (nx->k == FI_FN1)
                {
                    FK(run_fn1)(nx, src, d, len);
                }
                else if (nx->k == FI_FN3)
                {
                    FK(elem_clamp)(d, src, nx->c, nx->c2, len);
                }
                else
                {
                    _Pragma("omp simd") for (int j = 0; j < len; j++)
                    {
                        d[j] = (FT) 0 - src[j];
                    }
                }
            }
            else
            {
                FK_LOAD(p->img[in->img], off, len, stk[sp++]);
            }
            break;
        case FI_BIN:
            if (in->cpos == 0)
            {
                FK(run_bin)(in, stk[sp - 2], t, len);
                sp--;
            }
            else
            {
                FK(run_bin)(in, t, NULL, len);
            }
            break;
        case FI_FN2:
            if (in->cpos == 0)
            {
                FK(run_fn2)(in, stk[sp - 2], t, len);
                sp--;
            }
            else
            {
                FK(run_fn2)(in, t, NULL, len);
            }
            break;
        case FI_NEG:
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                t[j] = (FT) 0 - t[j];
            }
            break;
        case FI_NOT:
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                t[j] = (FT) (t[j] == (FT) 0);
            }
            break;
        case FI_BNOT:
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                t[j] = (FT) (~(int64_t) t[j]);
            }
            break;
        case FI_FN1:
            FK(run_fn1)(in, t, t, len);
            break;
        case FI_IMGC:
            FK(run_imgc)(p, in, off, len, stk[sp++]);
            break;
        case FI_IMGIMG:
            FK(run_imgimg)(p, in, off, len, stk[sp], stk[sp + 1]);
            sp++;
            break;
        case FI_VIMGC:
            FK(run_vimgc)(p, in, off, len, t, stk[sp]);
            break;
        case FI_FN3:
            FK(elem_clamp)(t, t, in->c, in->c2, len);
            break;
        case FI_SEL:
        {
            int nv = (in->cpos & 1) + (in->cpos >> 1);
            FK(elem_sel)(in->cpos, in->c, in->c2, stk[sp - 1 - nv],
                         (in->cpos & 1) ? stk[sp - nv] : NULL, (in->cpos & 2) ? stk[sp - 1] : NULL,
                         len);
            sp -= nv;
            break;
        }
        case FI_CONST:
        {
            const FT x = (FT) in->c;
            FT      *d = stk[sp++];
            _Pragma("omp simd") for (int j = 0; j < len; j++)
            {
                d[j] = x;
            }
            break;
        }
        case FI_POP:
            sp--;
            break;
        default:
            sp = FK(eval_spatial_ins)(p, in, off, len, stk, sp);
            break;
        } /* switch (in->k) */
    } /* for (int i = 0; i < p->n; i++) */
}

/**
 * FK(run_block) - Evaluate program @p on pixels [@off, @off + @len) and store the result into @out
 * @p:   Compiled program (scalar constants already evaluated)
 * @out: Destination image
 * @off: First pixel of the block
 * @len: Number of pixels (`<= FUSE_BLK`)
 * @stk: Per-thread vector stack, FUSE_MAXSTK slots of FUSE_BLK elements
 *
 * Return: void
 */
static void FK(run_block)(const fprog *p, IMAGE *out, uint64_t off, int len, FT (*stk)[FUSE_BLK])
{
    if (!p->dstv.is_view && out->md[0].datatype == FK_DT && p->nextra_dst == 0)
    {
        FT *dst = (FT *) out->array.raw + off;
        if (p->n == 1 && p->ins[0].k == FI_IMGIMG)
        {
            FK(run_imgimg)(p, &p->ins[0], off, len, dst, stk[0]);
            return;
        }
        if (p->n == 1 && p->ins[0].k == FI_IMGC)
        {
            FK(run_imgc)(p, &p->ins[0], off, len, dst);
            return;
        }
        if (p->n == 2 && p->ins[0].k == FI_IMG && !p->view[p->ins[0].img].is_view &&
            p->img[p->ins[0].img]->md[0].datatype == FK_DT)
        {
            const IMAGE *im0 = p->img[p->ins[0].img];
            const FT    *src = (const FT *) im0->array.raw + off;
            if (p->ins[1].k == FI_FN1)
            {
                FK(run_fn1)(&p->ins[1], src, dst, len);
                return;
            }
            if (p->ins[1].k == FI_FN3)
            {
                FK(elem_clamp)(dst, src, p->ins[1].c, p->ins[1].c2, len);
                return;
            }
            if (p->ins[1].k == FI_VIMGC && im0 == out)
            {
                FK(run_vimgc)(p, &p->ins[1], off, len, dst, stk[0]);
                return;
            }
        }
    }
    FK(eval_block)(p, off, len, stk);
    if (p->dstv.is_view)
    {
        FK_STOREV(p, out, off, len, stk[0]);
    }
    else
    {
        FK_STORE(out, off, len, stk[0]);
    }
}

#undef MAP_LOOP
#undef BIN_LOOPS
