// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_generate.h
 * @brief   Pixel coordinate and hash-PRNG image generators
 */

#ifndef COREMOD_ARITH_IMAGE_GENERATE_H
#define COREMOD_ARITH_IMAGE_GENERATE_H

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

/*
 * The random generators are counter-based: a value is a function of the seed and of the pixel
 * index only, so an image does not depend on how it is cut into blocks or threads. The seed
 * gives a 64-bit key (arith_rng_key(), another one for `randn` than for `rand`). A pixel's
 * 32-bit words are hashes of its index with the two halves of the key (arith_rng_w32()):
 * `rand` is such a word, `randn` is Box-Muller on two of them. The calculator's kernels hold
 * the same code as C text (src/calc/jit/mc_prelude.h): keep the two identical.
 */

/** arith_mix64() - SplitMix64's finalizer: every input bit changes about half the output bits. */
static inline uint64_t arith_mix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/** arith_rng_key() - start of the sequence for @seed (any double; -0.0 is 0.0), for rand or,
 * with @normal, for randn. */
static inline uint64_t arith_rng_key(double seed, int normal)
{
    union
    {
        double   d;
        uint64_t u;
    } c = { .d = seed + 0.0 };
    return arith_mix64(c.u ^ (normal ? 0x85ebca6b2c1b3c6dULL : 0ULL));
}

/** arith_rng_uf() - uniform float in [0, 1) from the top 24 bits of the word @w. */
static inline float arith_rng_uf(uint32_t w)
{
    return (float) (int32_t) (w >> 8) * 0x1p-24f;
}

/** arith_rng_ud() - uniform double in [0, 1) from the top 52 bits of @b (no integer to double
 * conversion, which AVX2 cannot vectorize). */
static inline double arith_rng_ud(uint64_t b)
{
    union
    {
        uint64_t u;
        double   d;
    } c = { .u = (b >> 12) | 0x3ff0000000000000ULL };
    return c.d - 1.0;
}

/** arith_mix32() - 32-bit avalanche hash (the "lowbias32" constants). */
static inline uint32_t arith_mix32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

/**
 * arith_rng_w32 - 32-bit word of a pixel for @key, the generator of `rand` and `randn`
 * @key:  Key of the seed
 * @lo:   Low 32 bits of the pixel index
 * @hi:   High 32 bits of the pixel index, times 0x9e3779b9
 * @salt: 0, or another value for one more word of the same pixel
 *
 * Two rounds of arith_mix32(), each with one half of the key: 32-bit multiplies only, which
 * AVX2 does eight at a time, where SplitMix64's 64-bit ones go four at a time in three steps
 * each. For one key the pixels below 2^32 get distinct words: a permutation, which the 24 bits
 * a float keeps do not show.
 */
static inline uint32_t arith_rng_w32(uint64_t key, uint32_t lo, uint32_t hi, uint32_t salt)
{
    uint32_t h = arith_mix32(lo + (uint32_t) key);
    return arith_mix32(h ^ (uint32_t) (key >> 32) ^ salt ^ hi);
}

/**
 * arith_rng_hi - The @hi of arith_rng_w32() for pixel @off of a run of pixels
 * @off: Pixel index
 * @rlo: Low 32 bits of the index of the run's first pixel
 * @rhi: High 32 bits of that index, times 0x9e3779b9
 *
 * A loop over a run (a row, a block) then works on the low word of the index alone, eight
 * pixels to a vector: the high word is the run's, plus one after the low word wrapped.
 */
static inline uint32_t arith_rng_hi(uint64_t off, uint32_t rlo, uint32_t rhi)
{
    return rhi + (((uint32_t) off < rlo) ? 0x9e3779b9U : 0U);
}

/** arith_rng_d() - uniform double in [0, 1) of a pixel: its word @salt on top, then 20 bits
 * of one more word. */
static inline double arith_rng_d(uint64_t key, uint32_t lo, uint32_t hi, uint32_t salt)
{
    uint64_t w1 = arith_rng_w32(key, lo, hi, salt);
    uint64_t w2 = arith_rng_w32(key, lo, hi, salt ^ 0x68e31da4U);
    return arith_rng_ud((w1 << 32) | w2);
}

/** arith_rand_f() - `rand` of pixel @off as a float in [0, 1): the top 24 bits of its word. */
static inline float arith_rand_f(uint64_t key, uint64_t off, uint32_t rlo, uint32_t rhi)
{
    return arith_rng_uf(arith_rng_w32(key, (uint32_t) off, arith_rng_hi(off, rlo, rhi), 0));
}

/** arith_rand_d() - `rand` of pixel @off as a double in [0, 1): 52 bits, the float's on top. */
static inline double arith_rand_d(uint64_t key, uint64_t off, uint32_t rlo, uint32_t rhi)
{
    return arith_rng_d(key, (uint32_t) off, arith_rng_hi(off, rlo, rhi), 0);
}

/** arith_randn_f() - `randn` of pixel @off as a float: Box-Muller on two words, the radius
 * from 31 bits (reaches 6.6 sigma), the angle from 24. */
static inline float arith_randn_f(uint64_t key, uint64_t off, uint32_t rlo, uint32_t rhi)
{
    uint32_t hi = arith_rng_hi(off, rlo, rhi);
    uint32_t w1 = arith_rng_w32(key, (uint32_t) off, hi, 0);
    uint32_t w2 = arith_rng_w32(key, (uint32_t) off, hi, 0xb5297a4dU);
    float    u1 = ((float) (int32_t) (w1 >> 1) + 0.5f) * 0x1p-31f; /* in (0, 1] */
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * arith_rng_uf(w2));
}

/** arith_randn_d() - `randn` of pixel @off as a double: the same with 52 bits for the radius
 * (reaches 8.5 sigma) and for the angle, the float's words on top. */
static inline double arith_randn_d(uint64_t key, uint64_t off, uint32_t rlo, uint32_t rhi)
{
    uint32_t hi = arith_rng_hi(off, rlo, rhi);
    double   u1 = arith_rng_d(key, (uint32_t) off, hi, 0) + 0x1p-53; /* in (0, 1) */
    double   u2 = arith_rng_d(key, (uint32_t) off, hi, 0xb5297a4dU);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/**
 * arith_rng_draw - Evaluate a single uniform or Gaussian pseudo-random draw
 * @kind: `ARITH_COORD_RAND` or `ARITH_COORD_RANDN`
 * @seed: Seed value
 *
 * Return: Pixel 0 of the image that @seed gives: a `double` in `[0, 1)` for `ARITH_COORD_RAND`,
 *         or `N(0, 1)` for `RANDN`.
 */
static inline double arith_rng_draw(int kind, double seed)
{
    uint64_t key = arith_rng_key(seed, kind == ARITH_COORD_RANDN);
    return (kind == ARITH_COORD_RAND) ? arith_rand_d(key, 0, 0, 0) : arith_randn_d(key, 0, 0, 0);
}

/** Return the next seed for unseeded random image generation: 53 bits, exact in a double. */
uint64_t arith_next_rng_seed(void);

#define ARITH_DEFINE_COORD_BLOCK_INLINE(FN_NAME, FT, SQRT_FN, ATAN2_FN, U_FN, N_FN)      \
    static inline void FN_NAME(int kind, uint32_t nx, uint32_t ny, double cx, double cy, \
                               uint64_t off, int len, FT *restrict t)                    \
    {                                                                                    \
        if (kind == ARITH_COORD_RAND || kind == ARITH_COORD_RANDN)                       \
        {                                                                                \
            const int      normal = (kind == ARITH_COORD_RANDN);                         \
            const uint64_t key    = arith_rng_key(cx, normal);                           \
            const uint32_t rlo    = (uint32_t) off;                                      \
            const uint32_t rhi    = (uint32_t) (off >> 32) * 0x9e3779b9U;                \
            if (normal)                                                                  \
            {                                                                            \
                _Pragma("omp simd") for (int k = 0; k < len; k++)                        \
                {                                                                        \
                    t[k] = N_FN(key, off + (uint64_t) k, rlo, rhi);                      \
                }                                                                        \
                return;                                                                  \
            }                                                                            \
            _Pragma("omp simd") for (int k = 0; k < len; k++)                            \
            {                                                                            \
                t[k] = U_FN(key, off + (uint64_t) k, rlo, rhi);                          \
            }                                                                            \
            return;                                                                      \
        }                                                                                \
                                                                                         \
        uint32_t x;                                                                      \
        uint32_t y;                                                                      \
        if (LIKELY((off >> 32) == 0))                                                    \
        {                                                                                \
            uint32_t o32 = (uint32_t) off;                                               \
            x            = o32 % nx;                                                     \
            y            = (o32 / nx) % ny;                                              \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            x = (uint32_t) (off % nx);                                                   \
            y = (uint32_t) ((off / nx) % ny);                                            \
        }                                                                                \
        int j = 0;                                                                       \
                                                                                         \
        while (j < len)                                                                  \
        {                                                                                \
            uint32_t rem     = nx - x; /* up to 2^32 - 1: not an int */                  \
            int      cnt     = ((uint32_t) (len - j) < rem) ? (len - j) : (int) rem;     \
            const FT bx      = (FT) ((double) x - cx);                                   \
            const FT dy      = (FT) ((double) y - cy);                                   \
            FT *restrict row = t + j;                                                    \
                                                                                         \
            if (kind == ARITH_COORD_X)                                                   \
            {                                                                            \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                        \
                {                                                                        \
                    row[k] = bx + (FT) k;                                                \
                }                                                                        \
            }                                                                            \
            else if (kind == ARITH_COORD_Y)                                              \
            {                                                                            \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                        \
                {                                                                        \
                    row[k] = dy;                                                         \
                }                                                                        \
            }                                                                            \
            else if (kind == ARITH_COORD_R)                                              \
            {                                                                            \
                const FT dy2 = dy * dy;                                                  \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                        \
                {                                                                        \
                    const FT dx = bx + (FT) k;                                           \
                    row[k]      = SQRT_FN(dx * dx + dy2);                                \
                }                                                                        \
            }                                                                            \
            else                                                                         \
            {                                                                            \
                _Pragma("omp simd") for (int k = 0; k < cnt; k++)                        \
                {                                                                        \
                    const FT dx = bx + (FT) k;                                           \
                    row[k]      = ATAN2_FN(dy, dx);                                      \
                }                                                                        \
            }                                                                            \
            j += cnt;                                                                    \
            x = 0;                                                                       \
            y = (y + 1 == ny) ? 0 : y + 1;                                               \
        }                                                                                \
    }

ARITH_DEFINE_COORD_BLOCK_INLINE(arith_coord_block_f_inline,
                                float,
                                sqrtf,
                                atan2f,
                                arith_rand_f,
                                arith_randn_f)
ARITH_DEFINE_COORD_BLOCK_INLINE(arith_coord_block_d_inline,
                                double,
                                sqrt,
                                atan2,
                                arith_rand_d,
                                arith_randn_d)
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

#endif // COREMOD_ARITH_IMAGE_GENERATE_H
