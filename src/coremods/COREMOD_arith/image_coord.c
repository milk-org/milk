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

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(IMAGE_METADATA) % 64 == 0,
               "IMAGE_METADATA size must be a multiple of 64 bytes for SIMD alignment");
#endif

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
void arith_coord_block_f(int      kind,
                         uint32_t nx,
                         uint32_t ny,
                         double   cx,
                         double   cy,
                         uint64_t off,
                         int      len,
                         float *restrict t)
{
    arith_coord_block_f_inline(kind, nx, ny, cx, cy, off, len, t);
}

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
void arith_coord_block_d(int      kind,
                         uint32_t nx,
                         uint32_t ny,
                         double   cx,
                         double   cy,
                         uint64_t off,
                         int      len,
                         double *restrict t)
{
    arith_coord_block_d_inline(kind, nx, ny, cx, cy, off, len, t);
}

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
    int      nth  = milk_omp_team_size(nel, 262144);

    if (dst->md[0].datatype == _DATATYPE_FLOAT)
    {
        float *MILK_RESTRICT pf = MILK_ASSUME_ALIGNED(dst->array.F);
#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * COORD_BLK;
            int      len = (int) ((nel - off < COORD_BLK) ? (nel - off) : COORD_BLK);
            arith_coord_block_f_inline(kind, nx, ny, cx, cy, off, len, pf + off);
        }
        return RETURN_SUCCESS;
    }

    if (dst->md[0].datatype == _DATATYPE_DOUBLE)
    {
        double *MILK_RESTRICT pd = MILK_ASSUME_ALIGNED(dst->array.D);
#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * COORD_BLK;
            int      len = (int) ((nel - off < COORD_BLK) ? (nel - off) : COORD_BLK);
            arith_coord_block_d_inline(kind, nx, ny, cx, cy, off, len, pd + off);
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
    int nth             = milk_omp_team_size((uint64_t) nx * ny, 262144);

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
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
