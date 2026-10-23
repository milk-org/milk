// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_coord.c
 * @brief   Pixel coordinate and hash-PRNG image generators
 */

#include <math.h>
#include <time.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "image_coord.h"

#define COORD_BLK 1024
#define COORD_OMP_MIN 16384

/**
 * arith_next_rng_seed - Return the next 32-bit seed for unseeded random image generation
 *
 * Mixes the realtime clock and PID on first use and advances a Weyl sequence
 * on each call so consecutive calls produce independent random realizations.
 *
 * Return: Non-zero 32-bit seed value
 */
uint32_t arith_next_rng_seed(void)
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
    return arith_hash32(rng_state);
}

#define DEFINE_COORD_BLOCK(FN_NAME, FT, SQRT_FN, ATAN2_FN)                                        \
    void FN_NAME(int kind, uint32_t nx, uint32_t ny, double cx, double cy, uint64_t off, int len, \
                 FT *restrict t)                                                                  \
    {                                                                                             \
        if (kind == ARITH_COORD_RAND)                                                             \
        {                                                                                         \
            uint32_t s = ((uint32_t) (int64_t) cx ^ 0x9e3779b9u) + (uint32_t) off * 747796405u;   \
            for (int k = 0; k < len; k++)                                                         \
            {                                                                                     \
                uint32_t h = arith_hash32(s + (uint32_t) k * 747796405u);                         \
                t[k]       = (FT) (int32_t) (h >> 8) * (FT) 5.9604644775390625e-8;                \
            }                                                                                     \
            return;                                                                               \
        }                                                                                         \
        if (kind == ARITH_COORD_RANDN)                                                            \
        {                                                                                         \
            uint32_t s0 = ((uint32_t) (int64_t) cx ^ 0x9e3779b9u) + (uint32_t) off * 747796405u;  \
            uint32_t s1 = s0 ^ 0x85ebca6bu;                                                       \
            for (int k = 0; k < len; k++)                                                         \
            {                                                                                     \
                uint32_t h0  = arith_hash32(s0 + (uint32_t) k * 747796405u);                      \
                uint32_t h1  = arith_hash32(s1 + (uint32_t) k * 2891336453u);                     \
                int32_t  sum = ((int32_t) h0 >> 22) + ((int32_t) (h0 << 10) >> 22) +              \
                               ((int32_t) (h0 << 20) >> 22) + ((int32_t) h1 >> 22) +              \
                               ((int32_t) (h1 << 10) >> 22) + ((int32_t) (h1 << 20) >> 22) + 3;   \
                t[k]         = (FT) sum * (FT) 0.0013810685772936816;                             \
            }                                                                                     \
            return;                                                                               \
        }                                                                                         \
                                                                                                  \
        uint32_t x = (uint32_t) (off % nx);                                                       \
        uint32_t y = (uint32_t) ((off / nx) % ny);                                                \
        int      j = 0;                                                                           \
                                                                                                  \
        while (j < len)                                                                           \
        {                                                                                         \
            int      rem     = (int) (nx - x);                                                    \
            int      cnt     = (len - j < rem) ? (len - j) : rem;                                 \
            const FT bx      = (FT) ((double) x - cx);                                            \
            const FT dy      = (FT) ((double) y - cy);                                            \
            FT *restrict row = t + j;                                                             \
                                                                                                  \
            if (kind == ARITH_COORD_X)                                                            \
            {                                                                                     \
                for (int k = 0; k < cnt; k++)                                                     \
                {                                                                                 \
                    row[k] = bx + (FT) k;                                                         \
                }                                                                                 \
            }                                                                                     \
            else if (kind == ARITH_COORD_Y)                                                       \
            {                                                                                     \
                for (int k = 0; k < cnt; k++)                                                     \
                {                                                                                 \
                    row[k] = dy;                                                                  \
                }                                                                                 \
            }                                                                                     \
            else if (kind == ARITH_COORD_R)                                                       \
            {                                                                                     \
                const FT dy2 = dy * dy;                                                           \
                for (int k = 0; k < cnt; k++)                                                     \
                {                                                                                 \
                    const FT dx = bx + (FT) k;                                                    \
                    row[k]      = SQRT_FN(dx * dx + dy2);                                         \
                }                                                                                 \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                for (int k = 0; k < cnt; k++)                                                     \
                {                                                                                 \
                    const FT dx = bx + (FT) k;                                                    \
                    row[k]      = ATAN2_FN(dy, dx);                                               \
                }                                                                                 \
            }                                                                                     \
            j += cnt;                                                                             \
            x = 0;                                                                                \
            y = (y + 1 == ny) ? 0 : y + 1;                                                        \
        }                                                                                         \
    }

/**
 * arith_coord_block_f - Fill a `float` block with coordinate or PRNG values
 * @kind: `ARITH_COORD_*` selector
 * @nx:   Image width (axis 0)
 * @ny:   Image height (axis 1)
 * @cx:   Center X coordinate (or RNG seed when @kind is `ARITH_COORD_RAND`/`RANDN`)
 * @cy:   Center Y coordinate
 * @off:  Linear starting pixel offset
 * @len:  Number of pixels in block
 * @t:    Output `float` buffer of length @len
 *
 * Return: void
 */
DEFINE_COORD_BLOCK(arith_coord_block_f, float, sqrtf, atan2f)

/**
 * arith_coord_block_d - Fill a `double` block with coordinate or PRNG values
 * @kind: `ARITH_COORD_*` selector
 * @nx:   Image width (axis 0)
 * @ny:   Image height (axis 1)
 * @cx:   Center X coordinate (or RNG seed when @kind is `ARITH_COORD_RAND`/`RANDN`)
 * @cy:   Center Y coordinate
 * @off:  Linear starting pixel offset
 * @len:  Number of pixels in block
 * @t:    Output `double` buffer of length @len
 *
 * Return: void
 */
DEFINE_COORD_BLOCK(arith_coord_block_d, double, sqrt, atan2)

/**
 * arith_image_fill_coord - Populate a `FLOAT` or `DOUBLE` image with coordinates or random values
 * @dst:  Destination image (`_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`)
 * @kind: `ARITH_COORD_*` selector
 * @cx:   X center (or seed for `ARITH_COORD_RAND`/`ARITH_COORD_RANDN`; 0 with @cy == 0 auto-seeds)
 * @cy:   Y center (set to 1.0 when passing an explicit seed of 0)
 *
 * Return: `RETURN_SUCCESS` or `RETURN_FAILURE`
 */
errno_t arith_image_fill_coord(IMAGE *dst, int kind, double cx, double cy)
{
    if (dst == NULL || dst->md == NULL)
    {
        return RETURN_FAILURE;
    }

    if ((kind == ARITH_COORD_RAND || kind == ARITH_COORD_RANDN) && cx == 0.0 && cy == 0.0)
    {
        cx = (double) arith_next_rng_seed();
    }

    uint32_t nx   = dst->md[0].size[0];
    uint32_t ny   = (dst->md[0].naxis > 1) ? dst->md[0].size[1] : 1;
    uint64_t nel  = dst->md[0].nelement;
    uint64_t nblk = (nel + COORD_BLK - 1) / COORD_BLK;

    if (dst->md[0].datatype == _DATATYPE_FLOAT)
    {
#pragma omp parallel for schedule(static) if (nel >= COORD_OMP_MIN)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * COORD_BLK;
            int      len = (int) ((nel - off < COORD_BLK) ? (nel - off) : COORD_BLK);
            arith_coord_block_f(kind, nx, ny, cx, cy, off, len, dst->array.F + off);
        }
        return RETURN_SUCCESS;
    }

    if (dst->md[0].datatype == _DATATYPE_DOUBLE)
    {
#pragma omp parallel for schedule(static) if (nel >= COORD_OMP_MIN)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * COORD_BLK;
            int      len = (int) ((nel - off < COORD_BLK) ? (nel - off) : COORD_BLK);
            arith_coord_block_d(kind, nx, ny, cx, cy, off, len, dst->array.D + off);
        }
        return RETURN_SUCCESS;
    }

    return RETURN_FAILURE;
}

/**
 * arith_make_slopexy - Create a 2D `FLOAT` planar slope image
 * @ID_name: Output image name
 * @l1:      X dimension
 * @l2:      Y dimension
 * @sx:      Slope along X
 * @sy:      Slope along Y
 *
 * Computes `sx * ii + sy * jj - (sx * (l1 / 2) + sy * (l2 / 2))` for each pixel `(ii, jj)`.
 *
 * Return: Image ID of the created image
 */
imageID arith_make_slopexy(const char *ID_name, uint32_t l1, uint32_t l2, double sx, double sy)
{
    DEBUG_TRACE_FSTART();

    imageID ID;
    create_2Dimage_ID(ID_name, l1, l2, &ID);
    uint32_t nx         = dcimg[ID].md[0].size[0];
    uint32_t ny         = dcimg[ID].md[0].size[1];
    double   coeff      = sx * (nx / 2) + sy * (ny / 2);
    float *restrict out = dcimg[ID].array.F;

#pragma omp parallel for schedule(static) if ((uint64_t) nx * ny >= COORD_OMP_MIN)
    for (uint32_t jj = 0; jj < ny; jj++)
    {
        double row_base     = sy * jj - coeff;
        float *restrict row = out + (uint64_t) jj * nx;
        for (uint32_t ii = 0; ii < nx; ii++)
        {
            row[ii] = (float) (sx * ii + row_base);
        }
    }

    DEBUG_TRACE_FEXIT();
    return ID;
}
