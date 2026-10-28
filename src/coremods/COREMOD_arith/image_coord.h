// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_coord.h
 * @brief   Pixel coordinate and hash-PRNG image generators
 */

#ifndef COREMOD_ARITH_IMAGE_COORD_H
#define COREMOD_ARITH_IMAGE_COORD_H

#include <math.h>
#include <stdint.h>
#include "ImageStreamIO/ImageStruct.h"
#include "libmilkcommon/milk_compiler.h"

/** Coordinate and PRNG generator kinds for `arith_coord_block_*` and `arith_image_fill_coord`. */
enum
{
    ARITH_COORD_X,
    ARITH_COORD_Y,
    ARITH_COORD_R,
    ARITH_COORD_THETA,
    ARITH_COORD_RAND,
    ARITH_COORD_RANDN
};

/**
 * arith_hash32 - Avalanching 32-bit integer hash finalizer
 * @h: Input 32-bit word
 *
 * Return: Mixed 32-bit hash value.
 */
static inline uint32_t arith_hash32(uint32_t h)
{
    h = ((h >> 16) ^ h) * 0x45d9f3bu;
    h = ((h >> 16) ^ h) * 0x45d9f3bu;
    return (h >> 16) ^ h;
}

/**
 * arith_rng_draw - Evaluate a single uniform or Gaussian pseudo-random draw
 * @kind: `ARITH_COORD_RAND` or `ARITH_COORD_RANDN`
 * @seed: 32-bit seed value
 *
 * Return: Pseudo-random `double` in `[0, 1)` for `ARITH_COORD_RAND` or `N(0, 1)` for `RANDN`.
 */
static inline double arith_rng_draw(int kind, uint32_t seed)
{
    uint32_t h0 = arith_hash32(seed ^ 0x9e3779b9u);
    if (kind == ARITH_COORD_RAND)
    {
        return (double) (int32_t) (h0 >> 8) * 5.9604644775390625e-8;
    }
    uint32_t h1  = arith_hash32((seed ^ 0x9e3779b9u) ^ 0x85ebca6bu);
    int32_t  sum = ((int32_t) h0 >> 22) + ((int32_t) (h0 << 10) >> 22) +
                   ((int32_t) (h0 << 20) >> 22) + ((int32_t) h1 >> 22) +
                   ((int32_t) (h1 << 10) >> 22) + ((int32_t) (h1 << 20) >> 22) + 3;
    return (double) sum * 0.0013810685772936816;
}

/** Return the next 32-bit Weyl-sequence seed for unseeded random image generation. */
uint32_t arith_next_rng_seed(void);

#define ARITH_DEFINE_COORD_BLOCK_INLINE(FN_NAME, FT, SQRT_FN, ATAN2_FN)                             \
    static inline void FN_NAME(int kind, uint32_t nx, uint32_t ny, double cx, double cy,            \
                               uint64_t off, int len, FT *restrict t)                               \
    {                                                                                               \
        if (kind == ARITH_COORD_RAND)                                                               \
        {                                                                                           \
            uint32_t s = ((uint32_t) (int64_t) cx ^ 0x9e3779b9u) + (uint32_t) off * 747796405u;     \
            _Pragma("omp simd") for (int k = 0; k < len; k++)                                       \
            {                                                                                       \
                uint32_t h = arith_hash32(s + (uint32_t) k * 747796405u);                           \
                t[k]       = (FT) (int32_t) (h >> 8) * (FT) 5.9604644775390625e-8;                  \
            }                                                                                       \
            return;                                                                                 \
        }                                                                                           \
        if (kind == ARITH_COORD_RANDN)                                                              \
        {                                                                                           \
            uint32_t s0 = ((uint32_t) (int64_t) cx ^ 0x9e3779b9u) + (uint32_t) off * 747796405u;    \
            _Pragma("omp simd") for (int k = 0; k < len; k++)                                       \
            {                                                                                       \
                uint32_t s   = s0 + (uint32_t) k * 747796405u; /* a function of off + k only: the \
                                                                  value does not depend on how the \
                                                                  image is cut into blocks */ \
                uint32_t h0  = arith_hash32(s);                                                     \
                uint32_t h1  = arith_hash32(s ^ 0x85ebca6bu);                                       \
                int32_t  sum = ((int32_t) h0 >> 22) + ((int32_t) (h0 << 10) >> 22) +                \
                               ((int32_t) (h0 << 20) >> 22) + ((int32_t) h1 >> 22) +                \
                               ((int32_t) (h1 << 10) >> 22) + ((int32_t) (h1 << 20) >> 22) + 3;     \
                t[k]         = (FT) sum * (FT) 0.0013810685772936816;                               \
            }                                                                                       \
            return;                                                                                 \
        }                                                                                           \
                                                                                                    \
        uint32_t x;                                                                                 \
        uint32_t y;                                                                                 \
        if (LIKELY((off >> 32) == 0))                                                               \
        {                                                                                           \
            uint32_t o32 = (uint32_t) off;                                                          \
            x            = o32 % nx;                                                                \
            y            = (o32 / nx) % ny;                                                         \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            x = (uint32_t) (off % nx);                                                              \
            y = (uint32_t) ((off / nx) % ny);                                                       \
        }                                                                                           \
        int j = 0;                                                                                  \
                                                                                                    \
        while (j < len)                                                                             \
        {                                                                                           \
            uint32_t rem     = nx - x; /* up to 2^32 - 1: not an int */                             \
            int      cnt     = ((uint32_t) (len - j) < rem) ? (len - j) : (int) rem;                \
            const FT bx      = (FT) ((double) x - cx);                                              \
            const FT dy      = (FT) ((double) y - cy);                                              \
            FT *restrict row = t + j;                                                               \
                                                                                                    \
            if (kind == ARITH_COORD_X)                                                              \
            {                                                                                       \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                                   \
                {                                                                                   \
                    row[k] = bx + (FT) k;                                                           \
                }                                                                                   \
            }                                                                                       \
            else if (kind == ARITH_COORD_Y)                                                         \
            {                                                                                       \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                                   \
                {                                                                                   \
                    row[k] = dy;                                                                    \
                }                                                                                   \
            }                                                                                       \
            else if (kind == ARITH_COORD_R)                                                         \
            {                                                                                       \
                const FT dy2 = dy * dy;                                                             \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                                   \
                {                                                                                   \
                    const FT dx = bx + (FT) k;                                                      \
                    row[k]      = SQRT_FN(dx * dx + dy2);                                           \
                }                                                                                   \
            }                                                                                       \
            else                                                                                    \
            {                                                                                       \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                                   \
                {                                                                                   \
                    const FT dx = bx + (FT) k;                                                      \
                    row[k]      = ATAN2_FN(dy, dx);                                                 \
                }                                                                                   \
            }                                                                                       \
            j += cnt;                                                                               \
            x = 0;                                                                                  \
            y = (y + 1 == ny) ? 0 : y + 1;                                                          \
        }                                                                                           \
    }

ARITH_DEFINE_COORD_BLOCK_INLINE(arith_coord_block_f_inline, float, sqrtf, atan2f)
ARITH_DEFINE_COORD_BLOCK_INLINE(arith_coord_block_d_inline, double, sqrt, atan2)
#undef ARITH_DEFINE_COORD_BLOCK_INLINE

/** Fill a block of `float` pixels with coordinate or PRNG values of kind @kind. */
void arith_coord_block_f(int      kind,
                         uint32_t nx,
                         uint32_t ny,
                         double   cx,
                         double   cy,
                         uint64_t off,
                         int      len,
                         float *restrict t);

/** Fill a block of `double` pixels with coordinate or PRNG values of kind @kind. */
void arith_coord_block_d(int      kind,
                         uint32_t nx,
                         uint32_t ny,
                         double   cx,
                         double   cy,
                         uint64_t off,
                         int      len,
                         double *restrict t);

/** Fill a `FLOAT` or `DOUBLE` image with coordinate or PRNG values of kind @kind. */
errno_t arith_image_fill_coord(IMAGE *dst, int kind, double cx, double cy);

/** Create a 2D `FLOAT` planar slope image with gradients `(@sx, @sy)`. */
imageID arith_make_slopexy(const char *ID_name, uint32_t l1, uint32_t l2, double sx, double sy);

#endif // COREMOD_ARITH_IMAGE_COORD_H
