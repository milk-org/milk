// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_elem_block.h
 * @brief   Inline SIMD block kernels and 10-datatype block converters for element-wise arithmetic
 */

#ifndef COREMOD_ARITH_IMAGE_ELEM_BLOCK_H
#define COREMOD_ARITH_IMAGE_ELEM_BLOCK_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <libfps/IMGID.h>
#include "libmilkcommon/milk_compiler.h"
#include "libmilkdata/milk_type_dispatch.h"

#define ARITH_ELEM_BLK 1024

/**
 * arith_load_block_d_inline - Convert a contiguous block of image pixels to `double`
 * @im:  Source image
 * @off: Starting linear pixel index
 * @len: Number of pixels to load
 * @dst: Output buffer of length @len
 */
static inline void arith_load_block_d_inline(const IMAGE *im,
                                             uint64_t     off,
                                             int          len,
                                             double *restrict dst)
{
    uint8_t dt = im->md[0].datatype;
    if (dt == _DATATYPE_DOUBLE)
    {
        __builtin_memcpy(dst, im->array.D + off, (size_t) len * sizeof(double));
        return;
    }
#define ARITH_LOAD_D_BODY(MBR)                                         \
    {                                                                  \
        const __typeof__(im->array.MBR[0]) *src = im->array.MBR + off; \
        _Pragma("omp simd") for (int j = 0; j < len; j++)              \
        {                                                              \
            dst[j] = (double) src[j];                                  \
        }                                                              \
    }
    MILK_FOR_EACH_DATATYPE(dt, ARITH_LOAD_D_BODY, ARITH_LOAD_D_BODY(D))
#undef ARITH_LOAD_D_BODY
}

/**
 * arith_load_block_f_inline - Convert a contiguous block of image pixels to `float`
 * @im:  Source image
 * @off: Starting linear pixel index
 * @len: Number of pixels to load
 * @dst: Output buffer of length @len
 */
static inline void arith_load_block_f_inline(const IMAGE *im,
                                             uint64_t     off,
                                             int          len,
                                             float *restrict dst)
{
    uint8_t dt = im->md[0].datatype;
    if (dt == _DATATYPE_FLOAT)
    {
        __builtin_memcpy(dst, im->array.F + off, (size_t) len * sizeof(float));
        return;
    }
#define ARITH_LOAD_F_BODY(MBR)                                         \
    {                                                                  \
        const __typeof__(im->array.MBR[0]) *src = im->array.MBR + off; \
        _Pragma("omp simd") for (int j = 0; j < len; j++)              \
        {                                                              \
            dst[j] = (float) src[j];                                   \
        }                                                              \
    }
    MILK_FOR_EACH_DATATYPE(dt, ARITH_LOAD_F_BODY, ARITH_LOAD_F_BODY(D))
#undef ARITH_LOAD_F_BODY
}

/**
 * arith_store_block_d_inline - Write a block of `double` values or a constant into @im
 * @im:  Destination image
 * @off: Starting linear pixel index
 * @len: Number of pixels to write
 * @v:   Input vector buffer (ignored when @isc is non-zero)
 * @isc: Non-zero to fill with constant @c instead of reading @v
 * @c:   Constant fill value when @isc is non-zero
 */
static inline void arith_store_block_d_inline(IMAGE   *im,
                                              uint64_t off,
                                              int      len,
                                              const double *restrict v,
                                              int    isc,
                                              double c)
{
    uint8_t dt = im->md[0].datatype;
    if (!isc && dt == _DATATYPE_DOUBLE)
    {
        __builtin_memcpy(im->array.D + off, v, (size_t) len * sizeof(double));
        return;
    }
#define ARITH_STORE_D_BODY(MBR)                                                       \
    {                                                                                 \
        __typeof__(im->array.MBR[0]) *out = im->array.MBR + off;                      \
        if (isc)                                                                      \
        {                                                                             \
            const __typeof__(im->array.MBR[0]) cv = (__typeof__(im->array.MBR[0])) c; \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                         \
            {                                                                         \
                out[j] = cv;                                                          \
            }                                                                         \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                         \
            {                                                                         \
                out[j] = (__typeof__(im->array.MBR[0])) v[j];                         \
            }                                                                         \
        }                                                                             \
    }
    MILK_FOR_EACH_DATATYPE(dt, ARITH_STORE_D_BODY, ARITH_STORE_D_BODY(D))
#undef ARITH_STORE_D_BODY
}

/**
 * arith_store_block_f_inline - Write a block of `float` values into @im
 * @im:  Destination image
 * @off: Starting linear pixel index
 * @len: Number of pixels to write
 * @v:   Input vector buffer of length @len
 */
static inline void arith_store_block_f_inline(IMAGE   *im,
                                              uint64_t off,
                                              int      len,
                                              const float *restrict v)
{
    uint8_t dt = im->md[0].datatype;
    if (dt == _DATATYPE_FLOAT)
    {
        __builtin_memcpy(im->array.F + off, v, (size_t) len * sizeof(float));
        return;
    }
#define ARITH_STORE_F_BODY(MBR)                                  \
    {                                                            \
        __typeof__(im->array.MBR[0]) *out = im->array.MBR + off; \
        _Pragma("omp simd") for (int j = 0; j < len; j++)        \
        {                                                        \
            out[j] = (__typeof__(im->array.MBR[0])) v[j];        \
        }                                                        \
    }
    MILK_FOR_EACH_DATATYPE(dt, ARITH_STORE_F_BODY, ARITH_STORE_F_BODY(D))
#undef ARITH_STORE_F_BODY
}

#define ARITH_DEFINE_ELEM_BLOCK_INLINE(SFX, FT, FM)                                             \
    static inline void arith_elem_pow_const_##SFX##_inline(FT * dst, const FT *src, double e,   \
                                                           int len)                             \
    {                                                                                           \
        double e2d = e * 2.0;                                                                   \
        int    e2  = (e2d >= -4.0 && e2d <= 8.0 && e2d == (double) (int) e2d) ? (int) e2d : 99; \
        switch (e2)                                                                             \
        {                                                                                       \
        case 0:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                dst[j] = (FT) 1;                                                                \
            }                                                                                   \
            break;                                                                              \
        case 2:                                                                                 \
            if (dst != src)                                                                     \
            {                                                                                   \
                __builtin_memcpy(dst, src, (size_t) len * sizeof(FT));                          \
            }                                                                                   \
            break;                                                                              \
        case 4:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                const FT x = src[j];                                                            \
                dst[j]     = x * x;                                                             \
            }                                                                                   \
            break;                                                                              \
        case 6:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                const FT x = src[j];                                                            \
                dst[j]     = x * x * x;                                                         \
            }                                                                                   \
            break;                                                                              \
        case 8:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                const FT x  = src[j];                                                           \
                const FT x2 = x * x;                                                            \
                dst[j]      = x2 * x2;                                                          \
            }                                                                                   \
            break;                                                                              \
        case -2:                                                                                \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                dst[j] = (FT) 1 / src[j];                                                       \
            }                                                                                   \
            break;                                                                              \
        case -4:                                                                                \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                const FT x = src[j];                                                            \
                dst[j]     = (FT) 1 / (x * x);                                                  \
            }                                                                                   \
            break;                                                                              \
        case 1:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                dst[j] = FM(sqrt)(src[j]);                                                      \
            }                                                                                   \
            break;                                                                              \
        case -1:                                                                                \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                dst[j] = (FT) 1 / FM(sqrt)(src[j]);                                             \
            }                                                                                   \
            break;                                                                              \
        case 3:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                const FT x = src[j];                                                            \
                dst[j]     = x * FM(sqrt)(x);                                                   \
            }                                                                                   \
            break;                                                                              \
        case 5:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                const FT x = src[j];                                                            \
                dst[j]     = x * x * FM(sqrt)(x);                                               \
            }                                                                                   \
            break;                                                                              \
        default:                                                                                \
        {                                                                                       \
            const FT b = (FT) e;                                                                \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                dst[j] = FM(pow)(src[j], b);                                                    \
            }                                                                                   \
            break;                                                                              \
        }                                                                                       \
        }                                                                                       \
    }                                                                                           \
                                                                                                \
    static inline void arith_elem_clamp_##SFX##_inline(FT * dst, const FT *src, double lo_d,    \
                                                       double hi_d, int len)                    \
    {                                                                                           \
        const FT lo = (FT) lo_d;                                                                \
        const FT hi = (FT) hi_d;                                                                \
        _Pragma("omp simd") for (int j = 0; j < len; j++)                                       \
        {                                                                                       \
            const FT x = src[j];                                                                \
            dst[j]     = (x < lo) ? lo : (x > hi) ? hi : x;                                     \
        }                                                                                       \
    }                                                                                           \
                                                                                                \
    static inline void arith_elem_sel_##SFX##_inline(int cpos, double tc_d, double fc_d,        \
                                                     FT *restrict c, const FT *restrict tv,     \
                                                     const FT *restrict fv, int len)            \
    {                                                                                           \
        const FT tc = (FT) tc_d;                                                                \
        const FT fc = (FT) fc_d;                                                                \
        switch (cpos)                                                                           \
        {                                                                                       \
        case 0:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                c[j] = (c[j] != (FT) 0) ? tc : fc;                                              \
            }                                                                                   \
            break;                                                                              \
        case 1:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                c[j] = (c[j] != (FT) 0) ? tv[j] : fc;                                           \
            }                                                                                   \
            break;                                                                              \
        case 2:                                                                                 \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                c[j] = (c[j] != (FT) 0) ? tc : fv[j];                                           \
            }                                                                                   \
            break;                                                                              \
        default:                                                                                \
            _Pragma("omp simd") for (int j = 0; j < len; j++)                                   \
            {                                                                                   \
                c[j] = (c[j] != (FT) 0) ? tv[j] : fv[j];                                        \
            }                                                                                   \
            break;                                                                              \
        }                                                                                       \
    }

#define ARITH_FM_D(fn) fn
#define ARITH_FM_F(fn) fn##f
ARITH_DEFINE_ELEM_BLOCK_INLINE(d, double, ARITH_FM_D)
ARITH_DEFINE_ELEM_BLOCK_INLINE(f, float, ARITH_FM_F)
#undef ARITH_FM_D
#undef ARITH_FM_F
#undef ARITH_DEFINE_ELEM_BLOCK_INLINE

/**
 * arith_elem_fn1_ptr_d_inline - Evaluate unary function @fn over a `double` block with SIMD dispatch
 * @dst: Output buffer of length @len (may alias @src)
 * @src: Input buffer of length @len
 * @fn:  Unary function pointer
 * @len: Number of elements in the block
 */
static inline void arith_elem_fn1_ptr_d_inline(double       *dst,
                                               const double *src,
                                               double (*fn)(double),
                                               int len)
{
#define ARITH_FN1_DEVIRT(NAME)                            \
    if (fn == &NAME)                                      \
    {                                                     \
        _Pragma("omp simd") for (int j = 0; j < len; j++) \
        {                                                 \
            dst[j] = NAME(src[j]);                        \
        }                                                 \
        return;                                           \
    }
    ARITH_FN1_DEVIRT(fabs)
    ARITH_FN1_DEVIRT(sqrt)
    ARITH_FN1_DEVIRT(floor)
    ARITH_FN1_DEVIRT(ceil)
    ARITH_FN1_DEVIRT(round)
    ARITH_FN1_DEVIRT(sin)
    ARITH_FN1_DEVIRT(cos)
    ARITH_FN1_DEVIRT(tan)
    ARITH_FN1_DEVIRT(sinh)
    ARITH_FN1_DEVIRT(cosh)
    ARITH_FN1_DEVIRT(tanh)
    ARITH_FN1_DEVIRT(exp)
    ARITH_FN1_DEVIRT(exp2)
    ARITH_FN1_DEVIRT(log)
    ARITH_FN1_DEVIRT(log10)
    ARITH_FN1_DEVIRT(log2)
    ARITH_FN1_DEVIRT(erf)
    ARITH_FN1_DEVIRT(erfc)
    ARITH_FN1_DEVIRT(atan)
    ARITH_FN1_DEVIRT(asin)
    ARITH_FN1_DEVIRT(acos)
    ARITH_FN1_DEVIRT(cbrt)
#undef ARITH_FN1_DEVIRT
    for (int j = 0; j < len; j++)
    {
        dst[j] = fn(src[j]);
    }
}

/**
 * arith_elem_fn2_ptr_d_inline - Evaluate binary function @fn over two `double` blocks
 * @dst: Output buffer of length @len (may alias @a)
 * @a:   Left input buffer of length @len
 * @b:   Right input buffer of length @len
 * @fn:  Binary function pointer
 * @len: Number of elements in the block
 */
static inline void arith_elem_fn2_ptr_d_inline(double       *dst,
                                               const double *a,
                                               const double *b,
                                               double (*fn)(double, double),
                                               int len)
{
    if (fn == &fmin)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = (a[j] < b[j]) ? a[j] : b[j];
        }
    }
    else if (fn == &fmax)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = (a[j] > b[j]) ? a[j] : b[j];
        }
    }
    else if (fn == &fmod)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = fmod(a[j], b[j]);
        }
    }
    else if (fn == &atan2)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = atan2(a[j], b[j]);
        }
    }
    else if (fn == &hypot)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = sqrt(a[j] * a[j] + b[j] * b[j]);
        }
    }
    else if (fn == &pow)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = pow(a[j], b[j]);
        }
    }
    else
    {
        for (int j = 0; j < len; j++)
        {
            dst[j] = fn(a[j], b[j]);
        }
    }
}

/**
 * arith_elem_fn1f_ptr_d_inline - Evaluate binary function @fn over a `double` block and scalar @b
 * @dst: Output buffer of length @len (may alias @a)
 * @a:   Left input buffer of length @len
 * @b:   Right scalar constant
 * @fn:  Binary function pointer
 * @len: Number of elements in the block
 */
static inline void arith_elem_fn1f_ptr_d_inline(double       *dst,
                                                const double *a,
                                                double        b,
                                                double (*fn)(double, double),
                                                int len)
{
    if (fn == &pow)
    {
        arith_elem_pow_const_d_inline(dst, a, b, len);
    }
    else if (fn == &fmin)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = (a[j] < b) ? a[j] : b;
        }
    }
    else if (fn == &fmax)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = (a[j] > b) ? a[j] : b;
        }
    }
    else if (fn == &fmod)
    {
        _Pragma("omp simd") for (int j = 0; j < len; j++)
        {
            dst[j] = fmod(a[j], b);
        }
    }
    else
    {
        for (int j = 0; j < len; j++)
        {
            dst[j] = fn(a[j], b);
        }
    }
}

#endif // COREMOD_ARITH_IMAGE_ELEM_BLOCK_H
