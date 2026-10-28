// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    im_complex.c
 * @brief   Complex number per-pixel image operations
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "libmilkcommon/milk_compiler.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "imgid_arith_helpers.h"

#include "image_arith__Cim_Cim__Cim.h"
#include "imfunctions.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define CPLX_OMP_FOR _Pragma("omp parallel for schedule(static) num_threads(nth) if (nth > 1)")
#else
#    define CPLX_OMP_FOR
#endif

#define CPLX_RUN_BLOCKS(nel, nth, CALL_EXPR)                                    \
    do                                                                          \
    {                                                                           \
        uint64_t nblk_ = ((nel) + 1023) / 1024;                                 \
        CPLX_OMP_FOR                                                            \
        for (uint64_t blk_ = 0; blk_ < nblk_; blk_++)                           \
        {                                                                       \
            uint64_t off = blk_ * 1024;                                         \
            int      len = (int) (((nel) - off < 1024) ? ((nel) - off) : 1024); \
            (CALL_EXPR);                                                        \
        }                                                                       \
        return RETURN_SUCCESS;                                                  \
    } while (0)

/**
 * arith_complex_read_pix - Read pixel @i of any real or complex image @im as `(re, imv)`
 * @im:  Source image
 * @i:   Linear pixel index
 * @re:  Output real component
 * @imv: Output imaginary component (`0.0` for real images)
 *
 * Return: void
 */
void arith_complex_read_pix(const IMAGE *im, uint64_t i, double *re, double *imv)
{
    uint8_t dt = im->md[0].datatype;
    *imv       = 0.0;
    if (dt == _DATATYPE_COMPLEX_FLOAT)
    {
        *re  = im->array.CF[i].re;
        *imv = im->array.CF[i].im;
        return;
    }
    if (dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        *re  = im->array.CD[i].re;
        *imv = im->array.CD[i].im;
        return;
    }
#define PIX_BODY(MBR)                    \
    {                                    \
        *re = (double) im->array.MBR[i]; \
        return;                          \
    }
    MILK_FOR_EACH_DATATYPE(dt, PIX_BODY, PIX_BODY(D))
#undef PIX_BODY
    *re = 0.0;
}

/** Store `(re, imv)` into pixel @i of complex image @im (`COMPLEX_FLOAT` or `COMPLEX_DOUBLE`). */
static inline void set_pix(IMAGE *im, uint64_t i, double re, double imv)
{
    if (im->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        im->array.CF[i].re = (float) re;
        im->array.CF[i].im = (float) imv;
    }
    else
    {
        im->array.CD[i].re = re;
        im->array.CD[i].im = imv;
    }
}

/** Evaluate `(ar + i*ai) op (br + i*bi)` for `ARITH_CPLX_ADD`..`ARITH_CPLX_DIV`. */
static inline void cop(int op, double ar, double ai, double br, double bi, double *r, double *i)
{
    switch (op)
    {
    case ARITH_CPLX_ADD:
        *r = ar + br;
        *i = ai + bi;
        break;
    case ARITH_CPLX_SUB:
        *r = ar - br;
        *i = ai - bi;
        break;
    case ARITH_CPLX_MUL:
        *r = ar * br - ai * bi;
        *i = ar * bi + ai * br;
        break;
    default:
    {
        double d = br * br + bi * bi;
        *r       = (ar * br + ai * bi) / d;
        *i       = (ai * br - ar * bi) / d;
        break;
    }
    }
}

/**
 * arith_complex_binop_im - Evaluate a binary complex operation into @out
 * @op:  `ARITH_CPLX_ADD`, `ARITH_CPLX_SUB`, `ARITH_CPLX_MUL`, or `ARITH_CPLX_DIV`
 * @li:  Left image operand, or NULL when scalar @lc is used
 * @lc:  Left scalar real value used when @li is NULL
 * @ri:  Right image operand, or NULL when scalar @rc is used
 * @rc:  Right scalar real value used when @ri is NULL
 * @out: Destination complex image (`COMPLEX_FLOAT` or `COMPLEX_DOUBLE`, may alias @li or @ri)
 *
 * Return: `RETURN_SUCCESS` or `RETURN_FAILURE`
 */
errno_t arith_complex_binop_im(int          op,
                               const IMAGE *li,
                               double       lc,
                               const IMAGE *ri,
                               double       rc,
                               IMAGE       *out)
{
    uint64_t nel = out->md[0].nelement;
    int      dbl = (out->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE);
    int      nth = milk_omp_team_size(nel, 262144);

    if (!dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && ri != NULL &&
        ri->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        CPLX_RUN_BLOCKS(nel, nth,
                        arith_cplx_binop_block_f_inline(op, li->array.CF + off, ri->array.CF + off,
                                                        out->array.CF + off, len));
    }

    if (dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE && ri != NULL &&
        ri->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        CPLX_RUN_BLOCKS(nel, nth,
                        arith_cplx_binop_block_d_inline(op, li->array.CD + off, ri->array.CD + off,
                                                        out->array.CD + off, len));
    }

    if (!dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && ri == NULL &&
        (op == ARITH_CPLX_MUL || op == ARITH_CPLX_DIV))
    {
        float s = (op == ARITH_CPLX_MUL) ? (float) rc : (float) (1.0 / rc);
        CPLX_RUN_BLOCKS(
            nel, nth,
            arith_cplx_scale_block_f_inline(li->array.CF + off, s, out->array.CF + off, len));
    }

    if (dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE && ri == NULL &&
        (op == ARITH_CPLX_MUL || op == ARITH_CPLX_DIV))
    {
        double s = (op == ARITH_CPLX_MUL) ? rc : (1.0 / rc);
        CPLX_RUN_BLOCKS(
            nel, nth,
            arith_cplx_scale_block_d_inline(li->array.CD + off, s, out->array.CD + off, len));
    }

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
    for (uint64_t i = 0; i < nel; i++)
    {
        double ar = lc, ai = 0.0, br = rc, bi = 0.0, r, im;
        if (li != NULL)
        {
            arith_complex_read_pix(li, i, &ar, &ai);
        }
        if (ri != NULL)
        {
            arith_complex_read_pix(ri, i, &br, &bi);
        }
        cop(op, ar, ai, br, bi, &r, &im);
        set_pix(out, i, r, im);
    }
    return RETURN_SUCCESS;
}

/**
 * arith_complex_inplace_im - Perform in-place assignment or compound update on complex image @dst
 * @op:  `ARITH_CPLX_ADD`, `ARITH_CPLX_SUB`, `ARITH_CPLX_MUL`, `ARITH_CPLX_DIV`, or `ARITH_CPLX_ASSIGN`
 * @dst: Destination complex image
 * @src: Optional source image, or NULL when scalar @val is used
 * @val: Scalar real value used when @src is NULL
 *
 * Return: `RETURN_SUCCESS` or `RETURN_FAILURE` (on division by zero)
 */
errno_t arith_complex_inplace_im(int op, IMAGE *dst, const IMAGE *src, double val)
{
    uint64_t nel = dst->md[0].nelement;
    int      nth = milk_omp_team_size(nel, 262144);
    if (src != NULL && op == ARITH_CPLX_DIV)
    {
        for (uint64_t i = 0; i < nel; i++)
        {
            double re, im;
            arith_complex_read_pix(src, i, &re, &im);
            if (re == 0.0 && im == 0.0)
            {
                return RETURN_FAILURE;
            }
        }
    }
    if (dst->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && src != NULL &&
        src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        if (op == ARITH_CPLX_ASSIGN)
        {
            if (dst->array.CF != src->array.CF)
            {
                __builtin_memcpy(dst->array.CF, src->array.CF, nel * sizeof(complex_float));
            }
            return RETURN_SUCCESS;
        }
        return arith_complex_binop_im(op, dst, 0.0, src, 0.0, dst);
    }
    if (dst->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE && src != NULL &&
        src->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        if (op == ARITH_CPLX_ASSIGN)
        {
            if (dst->array.CD != src->array.CD)
            {
                __builtin_memcpy(dst->array.CD, src->array.CD, nel * sizeof(complex_double));
            }
            return RETURN_SUCCESS;
        }
        return arith_complex_binop_im(op, dst, 0.0, src, 0.0, dst);
    }

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
    for (uint64_t i = 0; i < nel; i++)
    {
        double br = val, bi = 0.0, ar, ai, r, im;
        if (src != NULL)
        {
            arith_complex_read_pix(src, i, &br, &bi);
        }
        if (op == ARITH_CPLX_ASSIGN)
        {
            set_pix(dst, i, br, bi);
            continue;
        }
        arith_complex_read_pix(dst, i, &ar, &ai);
        cop(op, ar, ai, br, bi, &r, &im);
        set_pix(dst, i, r, im);
    }
    return RETURN_SUCCESS;
}

/**
 * arith_complex_neg_im - Negate every pixel of complex image @src into @dst
 * @src: Source complex image
 * @dst: Destination complex image (same datatype and element count, may alias @src)
 *
 * Return: `RETURN_SUCCESS`
 */
errno_t arith_complex_neg_im(const IMAGE *src, IMAGE *dst)
{
    uint64_t nel = src->md[0].nelement;
    uint64_t n2  = nel * 2;
    int      nth = milk_omp_team_size(nel, 262144);
    if (src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const float *s = (const float *) src->array.CF;
        float       *d = (float *) dst->array.CF;
#pragma omp parallel for simd schedule(static) num_threads(nth) if (parallel : nth > 1)
        for (uint64_t k = 0; k < n2; k++)
        {
            d[k] = 0.0f - s[k];
        }
        return RETURN_SUCCESS;
    }
    const double *s = (const double *) src->array.CD;
    double       *d = (double *) dst->array.CD;
#pragma omp parallel for simd schedule(static) num_threads(nth) if (parallel : nth > 1)
    for (uint64_t k = 0; k < n2; k++)
    {
        d[k] = 0.0 - s[k];
    }
    return RETURN_SUCCESS;
}

/**
 * arith_complex_conj_im - Conjugate every pixel of complex image @src into @dst
 * @src: Source complex image
 * @dst: Destination complex image (same datatype and element count, may alias @src)
 *
 * Return: `RETURN_SUCCESS`
 */
errno_t arith_complex_conj_im(const IMAGE *src, IMAGE *dst)
{
    uint64_t nel = src->md[0].nelement;
    int      nth = milk_omp_team_size(nel, 262144);
    if (src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        CPLX_RUN_BLOCKS(
            nel, nth,
            arith_cplx_conj_block_f_inline(src->array.CF + off, dst->array.CF + off, len));
    }
    CPLX_RUN_BLOCKS(nel, nth,
                    arith_cplx_conj_block_d_inline(src->array.CD + off, dst->array.CD + off, len));
}

/**
 * arith_complex_part_im - Extract `real`, `imag`, `abs`, or `arg` of @src into @dst
 * @src:   Source image
 * @which: `ARITH_CPLX_REAL` (0), `ARITH_CPLX_IMAG` (1), `ARITH_CPLX_ABS` (2), or `ARITH_CPLX_ARG` (3)
 * @dst:   Destination `FLOAT` or `DOUBLE` image
 *
 * Return: `RETURN_SUCCESS`
 */
errno_t arith_complex_part_im(const IMAGE *src, int which, IMAGE *dst)
{
    uint64_t nel = src->md[0].nelement;
    int      dbl = (dst->md[0].datatype == _DATATYPE_DOUBLE);
    int      nth = milk_omp_team_size(nel, 262144);

    if (!dbl && src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        float *restrict d = MILK_ASSUME_ALIGNED(dst->array.F);
        CPLX_RUN_BLOCKS(nel, nth, arith_cplx_part_im_block_f_inline(src, which, off, len, d + off));
    }

    if (dbl && arith_is_complex_dt(src->md[0].datatype))
    {
        double *restrict d = MILK_ASSUME_ALIGNED(dst->array.D);
        CPLX_RUN_BLOCKS(nel, nth, arith_cplx_part_im_block_d_inline(src, which, off, len, d + off));
    }

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
    for (uint64_t i = 0; i < nel; i++)
    {
        double re, im;
        arith_complex_read_pix(src, i, &re, &im);
        double v = (which == ARITH_CPLX_REAL)   ? re
                   : (which == ARITH_CPLX_IMAG) ? im
                   : (which == ARITH_CPLX_ABS)  ? sqrt(re * re + im * im)
                                                : atan2(im, re);
        if (dbl)
        {
            dst->array.D[i] = v;
        }
        else
        {
            dst->array.F[i] = (float) v;
        }
    }
    return RETURN_SUCCESS;
}

/**
 * arith_complex_cplx_im - Pack real and imaginary operands into complex image @out
 * @a:   Real-part image, or NULL when scalar @ac is used
 * @ac:  Scalar real part used when @a is NULL
 * @b:   Imaginary-part image, or NULL when scalar @bc is used
 * @bc:  Scalar imaginary part used when @b is NULL
 * @out: Destination `COMPLEX_FLOAT` or `COMPLEX_DOUBLE` image
 *
 * Return: `RETURN_SUCCESS`
 */
errno_t arith_complex_cplx_im(const IMAGE *a, double ac, const IMAGE *b, double bc, IMAGE *out)
{
    uint64_t nel = out->md[0].nelement;
    int      dbl = (out->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE);
    int      nth = milk_omp_team_size(nel, 262144);

    if (!dbl && (a == NULL || a->md[0].datatype == _DATATYPE_FLOAT) &&
        (b == NULL || b->md[0].datatype == _DATATYPE_FLOAT))
    {
        const float *restrict af = (a != NULL) ? a->array.F : NULL;
        const float *restrict bf = (b != NULL) ? b->array.F : NULL;
        const float fac          = (float) ac;
        const float fbc          = (float) bc;
        CPLX_RUN_BLOCKS(nel, nth,
                        arith_cplx_cplx_block_f_inline((af != NULL) ? (af + off) : NULL, fac,
                                                       (bf != NULL) ? (bf + off) : NULL, fbc,
                                                       out->array.CF + off, len));
    }

    if (dbl && (a == NULL || a->md[0].datatype == _DATATYPE_DOUBLE) &&
        (b == NULL || b->md[0].datatype == _DATATYPE_DOUBLE))
    {
        const double *restrict af = (a != NULL) ? a->array.D : NULL;
        const double *restrict bf = (b != NULL) ? b->array.D : NULL;
        CPLX_RUN_BLOCKS(nel, nth,
                        arith_cplx_cplx_block_d_inline((af != NULL) ? (af + off) : NULL, ac,
                                                       (bf != NULL) ? (bf + off) : NULL, bc,
                                                       out->array.CD + off, len));
    }

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
    for (uint64_t i = 0; i < nel; i++)
    {
        double re = ac, im = bc, unused;
        if (a != NULL)
        {
            arith_complex_read_pix(a, i, &re, &unused);
        }
        if (b != NULL)
        {
            arith_complex_read_pix(b, i, &im, &unused);
        }
        set_pix(out, i, re, im);
    }
    return RETURN_SUCCESS;
}

/**
 * arith_complex_expi_im - Evaluate `cos(phi) + i*sin(phi)` of real image @phi into @out
 * @phi: Source real phase image
 * @out: Destination `COMPLEX_FLOAT` or `COMPLEX_DOUBLE` image
 *
 * Return: `RETURN_SUCCESS`
 */
errno_t arith_complex_expi_im(const IMAGE *phi, IMAGE *out)
{
    uint64_t nel = out->md[0].nelement;
    int      nth = milk_omp_team_size(nel, 262144);
    if (phi->md[0].datatype == _DATATYPE_FLOAT && out->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        CPLX_RUN_BLOCKS(
            nel, nth, arith_cplx_expi_block_f_inline(phi->array.F + off, out->array.CF + off, len));
    }

    if (phi->md[0].datatype == _DATATYPE_DOUBLE && out->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        CPLX_RUN_BLOCKS(
            nel, nth, arith_cplx_expi_block_d_inline(phi->array.D + off, out->array.CD + off, len));
    }

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
    for (uint64_t i = 0; i < nel; i++)
    {
        double p = 0.0, unused = 0.0;
        arith_complex_read_pix(phi, i, &p, &unused);
        set_pix(out, i, cos(p), sin(p));
    }
    return RETURN_SUCCESS;
}
