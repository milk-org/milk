// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__Cim_Cim__Cim.h
 * @brief   Complex image arithmetic and component extraction routines
 */

#ifndef COREMOD_ARITH_IMAGE_ARITH_CIM_CIM_CIM_H
#define COREMOD_ARITH_IMAGE_ARITH_CIM_CIM_CIM_H

#include <math.h>
#include <stdint.h>
#include "ImageStreamIO/ImageStruct.h"
#include "libmilkcommon/milk_compiler.h"

/** Complex binary and in-place operation selectors for `arith_complex_*_im`. */
enum
{
    ARITH_CPLX_ADD,
    ARITH_CPLX_SUB,
    ARITH_CPLX_MUL,
    ARITH_CPLX_DIV,
    ARITH_CPLX_ASSIGN
};

/** Complex component selectors for `arith_complex_part_im`. */
enum
{
    ARITH_CPLX_REAL = 0,
    ARITH_CPLX_IMAG = 1,
    ARITH_CPLX_ABS  = 2,
    ARITH_CPLX_ARG  = 3
};

/** Return 1 if @dt is `_DATATYPE_COMPLEX_FLOAT` or `_DATATYPE_COMPLEX_DOUBLE`, 0 otherwise. */
static inline int arith_is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

#define ARITH_DEFINE_CPLX_BLOCK_INLINE(SFX, FT, CT, SQRT_FN, ATAN2_FN, COS_FN, SIN_FN)            \
    static inline void arith_cplx_part_block_##SFX##_inline(const CT *restrict src, int which,    \
                                                            int len, FT *restrict dst)            \
    {                                                                                             \
        if (which == ARITH_CPLX_REAL)                                                             \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                dst[j] = src[j].re;                                                               \
            }                                                                                     \
        }                                                                                         \
        else if (which == ARITH_CPLX_IMAG)                                                        \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                dst[j] = src[j].im;                                                               \
            }                                                                                     \
        }                                                                                         \
        else if (which == ARITH_CPLX_ABS)                                                         \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                FT re  = src[j].re;                                                               \
                FT im  = src[j].im;                                                               \
                dst[j] = SQRT_FN(re * re + im * im);                                              \
            }                                                                                     \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            for (int off = 0; off < len; off += 256)                                              \
            {                                                                                     \
                int clen             = (len - off < 256) ? (len - off) : 256;                     \
                const CT *restrict s = src + off;                                                 \
                FT *restrict d       = dst + off;                                                 \
                MILK_ALIGNED(64) FT rb[256];                                                      \
                MILK_ALIGNED(64) FT ib[256];                                                      \
                _Pragma("omp simd") for (int j = 0; j < clen; j++)                                \
                {                                                                                 \
                    rb[j] = s[j].re;                                                              \
                    ib[j] = s[j].im;                                                              \
                }                                                                                 \
                _Pragma("omp simd") for (int j = 0; j < clen; j++)                                \
                {                                                                                 \
                    d[j] = ATAN2_FN(ib[j], rb[j]);                                                \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    static inline void arith_cplx_binop_block_##SFX##_inline(int op, const CT *a, const CT *b,    \
                                                             CT *dst, int len)                    \
    {                                                                                             \
        if (op == ARITH_CPLX_ADD)                                                                 \
        {                                                                                         \
            const FT *af = (const FT *) a;                                                        \
            const FT *bf = (const FT *) b;                                                        \
            FT       *df = (FT *) dst;                                                            \
            int       n2 = len * 2;                                                               \
            _Pragma("omp simd") for (int k = 0; k < n2; k++)                                      \
            {                                                                                     \
                df[k] = af[k] + bf[k];                                                            \
            }                                                                                     \
        }                                                                                         \
        else if (op == ARITH_CPLX_SUB)                                                            \
        {                                                                                         \
            const FT *af = (const FT *) a;                                                        \
            const FT *bf = (const FT *) b;                                                        \
            FT       *df = (FT *) dst;                                                            \
            int       n2 = len * 2;                                                               \
            _Pragma("omp simd") for (int k = 0; k < n2; k++)                                      \
            {                                                                                     \
                df[k] = af[k] - bf[k];                                                            \
            }                                                                                     \
        }                                                                                         \
        else if (op == ARITH_CPLX_MUL)                                                            \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                FT ar     = a[j].re;                                                              \
                FT ai     = a[j].im;                                                              \
                FT br     = b[j].re;                                                              \
                FT bi     = b[j].im;                                                              \
                dst[j].re = ar * br - ai * bi;                                                    \
                dst[j].im = ar * bi + ai * br;                                                    \
            }                                                                                     \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                FT ar     = a[j].re;                                                              \
                FT ai     = a[j].im;                                                              \
                FT br     = b[j].re;                                                              \
                FT bi     = b[j].im;                                                              \
                FT d      = br * br + bi * bi;                                                    \
                dst[j].re = (ar * br + ai * bi) / d;                                              \
                dst[j].im = (ai * br - ar * bi) / d;                                              \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    static inline void arith_cplx_scale_block_##SFX##_inline(const CT *a, FT s, CT *dst, int len) \
    {                                                                                             \
        const FT *af = (const FT *) a;                                                            \
        FT       *df = (FT *) dst;                                                                \
        int       n2 = len * 2;                                                                   \
        _Pragma("omp simd") for (int k = 0; k < n2; k++)                                          \
        {                                                                                         \
            df[k] = af[k] * s;                                                                    \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    static inline void arith_cplx_conj_block_##SFX##_inline(const CT *src, CT *dst, int len)      \
    {                                                                                             \
        _Pragma("omp simd") for (int j = 0; j < len; j++)                                         \
        {                                                                                         \
            dst[j].re = src[j].re;                                                                \
            dst[j].im = (FT) 0 - src[j].im;                                                       \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    static inline void arith_cplx_cplx_block_##SFX##_inline(                                      \
        const FT *restrict af, FT fac, const FT *restrict bf, FT fbc, CT *restrict dst, int len)  \
    {                                                                                             \
        if (af != NULL && bf != NULL)                                                             \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                dst[j].re = af[j];                                                                \
                dst[j].im = bf[j];                                                                \
            }                                                                                     \
        }                                                                                         \
        else if (af != NULL)                                                                      \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                dst[j].re = af[j];                                                                \
                dst[j].im = fbc;                                                                  \
            }                                                                                     \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                     \
            {                                                                                     \
                dst[j].re = fac;                                                                  \
                dst[j].im = (bf != NULL) ? bf[j] : fbc;                                           \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    static inline void arith_cplx_expi_block_##SFX##_inline(const FT *restrict src,               \
                                                            CT *restrict dst, int len)            \
    {                                                                                             \
        for (int off = 0; off < len; off += 256)                                                  \
        {                                                                                         \
            int clen             = (len - off < 256) ? (len - off) : 256;                         \
            const FT *restrict s = src + off;                                                     \
            CT *restrict d       = dst + off;                                                     \
            MILK_ALIGNED(64) FT cr[256];                                                          \
            MILK_ALIGNED(64) FT ci[256];                                                          \
            _Pragma("omp simd") for (int j = 0; j < clen; j++)                                    \
            {                                                                                     \
                cr[j] = COS_FN(s[j]);                                                             \
            }                                                                                     \
            _Pragma("omp simd") for (int j = 0; j < clen; j++)                                    \
            {                                                                                     \
                ci[j] = SIN_FN(s[j]);                                                             \
            }                                                                                     \
            _Pragma("omp simd") for (int j = 0; j < clen; j++)                                    \
            {                                                                                     \
                d[j].re = cr[j];                                                                  \
                d[j].im = ci[j];                                                                  \
            }                                                                                     \
        }                                                                                         \
    }

ARITH_DEFINE_CPLX_BLOCK_INLINE(f, float, complex_float, sqrtf, atan2f, cosf, sinf)
ARITH_DEFINE_CPLX_BLOCK_INLINE(d, double, complex_double, sqrt, atan2, cos, sin)

#undef ARITH_DEFINE_CPLX_BLOCK_INLINE

/** Extract component @which of complex image @im on `[@off, @off + @len)` into `float` @dst. */
static inline void arith_cplx_part_im_block_f_inline(const IMAGE *im,
                                                     int          which,
                                                     uint64_t     off,
                                                     int          len,
                                                     float *restrict dst)
{
    arith_cplx_part_block_f_inline(im->array.CF + off, which, len, dst);
}

/** Extract component @which of complex image @im on `[@off, @off + @len)` into `double` @dst. */
static inline void arith_cplx_part_im_block_d_inline(const IMAGE *im,
                                                     int          which,
                                                     uint64_t     off,
                                                     int          len,
                                                     double *restrict dst)
{
    if (im->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        arith_cplx_part_block_d_inline(im->array.CD + off, which, len, dst);
        return;
    }
    const complex_float *restrict s = im->array.CF + off;
    if (which == ARITH_CPLX_REAL)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = (double) s[j].re;
        }
    }
    else if (which == ARITH_CPLX_IMAG)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = (double) s[j].im;
        }
    }
    else if (which == ARITH_CPLX_ABS)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            double re = (double) s[j].re;
            double im = (double) s[j].im;
            dst[j]    = sqrt(re * re + im * im);
        }
    }
    else
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = (double) atan2f(s[j].im, s[j].re);
        }
    }
}

/** Read pixel @i of any real or complex image @im as `(re, imv)`. */
void arith_complex_read_pix(const IMAGE *im, uint64_t i, double *re, double *imv);

/** Evaluate `li/lc op ri/rc` into complex image @out (`ARITH_CPLX_ADD`..`ARITH_CPLX_DIV`). */
errno_t arith_complex_binop_im(int          op,
                               const IMAGE *li,
                               double       lc,
                               const IMAGE *ri,
                               double       rc,
                               IMAGE       *out);

/** Evaluate `dst op= src/val` in place on complex image @dst (`ARITH_CPLX_ADD`..`ARITH_CPLX_ASSIGN`). */
errno_t arith_complex_inplace_im(int op, IMAGE *dst, const IMAGE *src, double val);

/** Negate every pixel of complex image @src into @dst (may alias @src). */
errno_t arith_complex_neg_im(const IMAGE *src, IMAGE *dst);

/** Conjugate every pixel of complex image @src into @dst (may alias @src). */
errno_t arith_complex_conj_im(const IMAGE *src, IMAGE *dst);

/** Extract `real` (0), `imag` (1), `abs` (2), or `arg` (3) of @src into `FLOAT`/`DOUBLE` image @dst. */
errno_t arith_complex_part_im(const IMAGE *src, int which, IMAGE *dst);

/** Pack real part (`@a` or `@ac`) and imaginary part (`@b` or `@bc`) into complex image @out. */
errno_t arith_complex_cplx_im(const IMAGE *a, double ac, const IMAGE *b, double bc, IMAGE *out);

/** Evaluate `cos(phi) + i*sin(phi)` of real image @phi into complex image @out. */
errno_t arith_complex_expi_im(const IMAGE *phi, IMAGE *out);

/** Add two complex images by name. */
int arith_image_Cadd(const char *ID1_name, const char *ID2_name, const char *ID_out);

/** Subtract two complex images by name. */
int arith_image_Csub(const char *ID1_name, const char *ID2_name, const char *ID_out);

/** Multiply two complex images by name. */
int arith_image_Cmult(const char *ID1_name, const char *ID2_name, const char *ID_out);

/** Divide two complex images by name. */
int arith_image_Cdiv(const char *ID1_name, const char *ID2_name, const char *ID_out);

#endif // COREMOD_ARITH_IMAGE_ARITH_CIM_CIM_CIM_H
