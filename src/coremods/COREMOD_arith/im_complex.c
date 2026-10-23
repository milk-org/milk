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
#    define OMP_NELEMENT_LIMIT 100000
#endif

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

    if (!dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && ri != NULL &&
        ri->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const complex_float *a   = li->array.CF;
        const complex_float *b   = ri->array.CF;
        complex_float       *dst = out->array.CF;
        if (op == ARITH_CPLX_ADD)
        {
            const float *af = (const float *) a;
            const float *bf = (const float *) b;
            float       *df = (float *) dst;
            uint64_t     n2 = nel * 2;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t k = 0; k < n2; k++)
            {
                df[k] = af[k] + bf[k];
            }
            return RETURN_SUCCESS;
        }
        if (op == ARITH_CPLX_SUB)
        {
            const float *af = (const float *) a;
            const float *bf = (const float *) b;
            float       *df = (float *) dst;
            uint64_t     n2 = nel * 2;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t k = 0; k < n2; k++)
            {
                df[k] = af[k] - bf[k];
            }
            return RETURN_SUCCESS;
        }
        if (op == ARITH_CPLX_MUL)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                float ar  = a[i].re;
                float ai  = a[i].im;
                float br  = b[i].re;
                float bi  = b[i].im;
                dst[i].re = ar * br - ai * bi;
                dst[i].im = ar * bi + ai * br;
            }
            return RETURN_SUCCESS;
        }
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t i = 0; i < nel; i++)
        {
            float ar  = a[i].re;
            float ai  = a[i].im;
            float br  = b[i].re;
            float bi  = b[i].im;
            float d   = br * br + bi * bi;
            dst[i].re = (ar * br + ai * bi) / d;
            dst[i].im = (ai * br - ar * bi) / d;
        }
        return RETURN_SUCCESS;
    }

    if (!dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && ri == NULL &&
        (op == ARITH_CPLX_MUL || op == ARITH_CPLX_DIV))
    {
        const float *af = (const float *) li->array.CF;
        float       *df = (float *) out->array.CF;
        uint64_t     n2 = nel * 2;
        float        s  = (op == ARITH_CPLX_MUL) ? (float) rc : (float) (1.0 / rc);
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t k = 0; k < n2; k++)
        {
            df[k] = af[k] * s;
        }
        return RETURN_SUCCESS;
    }

#pragma omp parallel for schedule(static) if (nel >= 131072)
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
        src->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE && op == ARITH_CPLX_ASSIGN)
    {
        if (dst->array.CD != src->array.CD)
        {
            __builtin_memcpy(dst->array.CD, src->array.CD, nel * sizeof(complex_double));
        }
        return RETURN_SUCCESS;
    }

#pragma omp parallel for schedule(static) if (nel >= 131072)
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
    if (src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const float *s = (const float *) src->array.CF;
        float       *d = (float *) dst->array.CF;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t k = 0; k < n2; k++)
        {
            d[k] = 0.0f - s[k];
        }
        return RETURN_SUCCESS;
    }
    const double *s = (const double *) src->array.CD;
    double       *d = (double *) dst->array.CD;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
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
    if (src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const complex_float *s = src->array.CF;
        complex_float       *d = dst->array.CF;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t i = 0; i < nel; i++)
        {
            d[i].re = s[i].re;
            d[i].im = 0.0f - s[i].im;
        }
        return RETURN_SUCCESS;
    }
    const complex_double *s = src->array.CD;
    complex_double       *d = dst->array.CD;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
    for (uint64_t i = 0; i < nel; i++)
    {
        d[i].re = s[i].re;
        d[i].im = 0.0 - s[i].im;
    }
    return RETURN_SUCCESS;
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

    if (!dbl && src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const complex_float *restrict s = src->array.CF;
        float *restrict d               = dst->array.F;
        if (which == ARITH_CPLX_REAL)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                d[i] = s[i].re;
            }
        }
        else if (which == ARITH_CPLX_IMAG)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                d[i] = s[i].im;
            }
        }
        else if (which == ARITH_CPLX_ABS)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                float re = s[i].re;
                float im = s[i].im;
                d[i]     = sqrtf(re * re + im * im);
            }
        }
        else
        {
            uint64_t nblk = (nel + 255) / 256;
#pragma omp                   parallel for schedule(static) if (nel >= 131072)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t               off = b * 256;
                int                    len = (int) ((nel - off < 256) ? (nel - off) : 256);
                MILK_ALIGNED(64) float rb[256];
                MILK_ALIGNED(64) float ib[256];
                for (int j = 0; j < len; j++)
                {
                    rb[j] = s[off + (uint64_t) j].re;
                    ib[j] = s[off + (uint64_t) j].im;
                }
                for (int j = 0; j < len; j++)
                {
                    d[off + (uint64_t) j] = atan2f(ib[j], rb[j]);
                }
            }
        }
        return RETURN_SUCCESS;
    }

#pragma omp parallel for schedule(static) if (nel >= 131072)
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

    if (!dbl && (a == NULL || a->md[0].datatype == _DATATYPE_FLOAT) &&
        (b == NULL || b->md[0].datatype == _DATATYPE_FLOAT))
    {
        const float *restrict af  = (a != NULL) ? a->array.F : NULL;
        const float *restrict bf  = (b != NULL) ? b->array.F : NULL;
        complex_float *restrict d = out->array.CF;
        const float fac           = (float) ac;
        const float fbc           = (float) bc;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t i = 0; i < nel; i++)
        {
            d[i].re = (af != NULL) ? af[i] : fac;
            d[i].im = (bf != NULL) ? bf[i] : fbc;
        }
        return RETURN_SUCCESS;
    }

#pragma omp parallel for schedule(static) if (nel >= 131072)
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
    if (phi->md[0].datatype == _DATATYPE_FLOAT && out->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const float *restrict src   = phi->array.F;
        complex_float *restrict dst = out->array.CF;
        uint64_t nblk               = (nel + 255) / 256;
#pragma omp parallel for schedule(static) if (nel >= 131072)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off              = b * 256;
            int      len              = (int) ((nel - off < 256) ? (nel - off) : 256);
            const float *restrict s   = src + off;
            complex_float *restrict d = dst + off;
            MILK_ALIGNED(64) float cr[256];
            MILK_ALIGNED(64) float ci[256];
            for (int j = 0; j < len; j++)
            {
                cr[j] = cosf(s[j]);
            }
            for (int j = 0; j < len; j++)
            {
                ci[j] = sinf(s[j]);
            }
            for (int j = 0; j < len; j++)
            {
                d[j].re = cr[j];
                d[j].im = ci[j];
            }
        }
        return RETURN_SUCCESS;
    }

    if (phi->md[0].datatype == _DATATYPE_DOUBLE && out->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        const double *restrict src   = phi->array.D;
        complex_double *restrict dst = out->array.CD;
        uint64_t nblk                = (nel + 255) / 256;
#pragma omp parallel for schedule(static) if (nel >= 131072)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off               = b * 256;
            int      len               = (int) ((nel - off < 256) ? (nel - off) : 256);
            const double *restrict s   = src + off;
            complex_double *restrict d = dst + off;
            MILK_ALIGNED(64) double cr[256];
            MILK_ALIGNED(64) double ci[256];
            for (int j = 0; j < len; j++)
            {
                cr[j] = cos(s[j]);
            }
            for (int j = 0; j < len; j++)
            {
                ci[j] = sin(s[j]);
            }
            for (int j = 0; j < len; j++)
            {
                d[j].re = cr[j];
                d[j].im = ci[j];
            }
        }
        return RETURN_SUCCESS;
    }

#pragma omp parallel for schedule(static) if (nel >= 131072)
    for (uint64_t i = 0; i < nel; i++)
    {
        double p = 0.0, unused = 0.0;
        arith_complex_read_pix(phi, i, &p, &unused);
        set_pix(out, i, cos(p), sin(p));
    }
    return RETURN_SUCCESS;
}

// complex float (CF), complex float (CF) -> complex float (CF)
errno_t arith_image_function_CF_CF__CF(const char *ID_name1,
                                       const char *ID_name2,
                                       const char *ID_out,
                                       complex_float (*pt2function)(complex_float, complex_float))
{
    IMGID img1 = imgid_make_from_name(ID_name1);
    resolveIMGID(&img1, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID img2 = imgid_make_from_name(ID_name2);
    resolveIMGID(&img2, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID imgout      = imgid_make_from_name(ID_out);
    imgout.mdt->naxis = img1.md->naxis;
    for (uint8_t i = 0; i < img1.md->naxis; i++)
    {
        imgout.mdt->size[i] = img1.md->size[i];
    }
    imgout.mdt->datatype = img1.md->datatype;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    uint64_t nelement = img1.md->nelement;

#ifdef _OPENMP
#    pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#    pragma omp for simd
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout.im->array.CF[ii] = pt2function(img1.im->array.CF[ii], img2.im->array.CF[ii]);
        }
#ifdef _OPENMP
    }
#endif
    RegisterIMGID(&imgout, dcimg, dcnimg);
    imgid_free(&img1);
    imgid_free(&img2);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}

// complex double (CD), complex double (CD) -> complex double (CD)
errno_t arith_image_function_CD_CD__CD(const char *ID_name1,
                                       const char *ID_name2,
                                       const char *ID_out,
                                       complex_double (*pt2function)(complex_double,
                                                                     complex_double))
{
    IMGID img1 = imgid_make_from_name(ID_name1);
    resolveIMGID(&img1, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID img2 = imgid_make_from_name(ID_name2);
    resolveIMGID(&img2, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID imgout      = imgid_make_from_name(ID_out);
    imgout.mdt->naxis = img1.md->naxis;
    for (uint8_t i = 0; i < img1.md->naxis; i++)
    {
        imgout.mdt->size[i] = img1.md->size[i];
    }
    imgout.mdt->datatype = img1.md->datatype;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    uint64_t nelement = img1.md->nelement;

#ifdef _OPENMP
#    pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#    pragma omp for simd
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            imgout.im->array.CD[ii] = pt2function(img1.im->array.CD[ii], img2.im->array.CD[ii]);
        }
#ifdef _OPENMP
    }
#endif
    RegisterIMGID(&imgout, dcimg, dcnimg);
    imgid_free(&img1);
    imgid_free(&img2);
    imgid_free(&imgout);
    return RETURN_SUCCESS;
}
