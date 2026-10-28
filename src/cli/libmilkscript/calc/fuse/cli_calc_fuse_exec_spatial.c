// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_exec_spatial.c
 * @brief Block evaluators for `shift`, `lut`, `medfilt`, `pixremap` (`gather`), and `interp2`.
 *
 * Role & Goal:
 *   Implements cache-resident double and float32 block evaluators for spatial stencils,
 *   1-D linear lookup tables, 2-D Young-tableau median filters, 1-D index gather, and
 *   2-D bilinear interpolation, using `-ffast-math`-safe bit-level NaN checks.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "cli_calc_fuse_internal.h"
#include "libmilkdata/milk_type_dispatch.h"

/**
 * cli_calc_fuse_shift_block_d - Load 2D edge-clamped shifted block in `double`
 * @p:   Compiled fused program
 * @k:   Operand slot index
 * @dx:  X pixel shift
 * @dy:  Y pixel shift
 * @off: Starting pixel index
 * @len: Number of pixels in block
 * @dst: Output double buffer
 *
 * Return: void
 */
void cli_calc_fuse_shift_block_d(const fprog *p,
                                 int          k,
                                 int64_t      dx,
                                 int64_t      dy,
                                 uint64_t     off,
                                 int          len,
                                 double *restrict dst)
{
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    uint8_t dt = im->md[0].datatype;
#define SHIFT_BODY(MBR)                                                                          \
    {                                                                                            \
        int64_t sy                              = (int64_t) y_ + dy;                             \
        sy                                      = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy);     \
        const __typeof__(im->array.MBR[0]) *row = im->array.MBR + ((int64_t) z_ * ny + sy) * nx; \
        double                             *d   = dst + J;                                       \
        int64_t                             s0  = (int64_t) x_ + dx;                             \
        int     i0  = (s0 < 0) ? (((int64_t) N < -s0) ? N : (int) (-s0)) : 0;                    \
        int64_t lim = nx - s0;                                                                   \
        int     i1  = (lim <= (int64_t) i0) ? i0 : ((lim < (int64_t) N) ? (int) lim : N);        \
        double  v0  = (double) row[0];                                                           \
        for (int i = 0; i < i0; i++)                                                             \
        {                                                                                        \
            d[i] = v0;                                                                           \
        }                                                                                        \
        const __typeof__(im->array.MBR[0]) *srow = row + s0;                                     \
        _Pragma("omp simd") for (int i = i0; i < i1; i++)                                        \
        {                                                                                        \
            d[i] = (double) srow[i];                                                             \
        }                                                                                        \
        double vn = (double) row[nx - 1];                                                        \
        for (int i = i1; i < N; i++)                                                             \
        {                                                                                        \
            d[i] = vn;                                                                           \
        }                                                                                        \
    }
    VIEW_WALK(p, v, off, len, { MILK_FOR_EACH_DATATYPE(dt, SHIFT_BODY, SHIFT_BODY(D)) });
#undef SHIFT_BODY
}

/**
 * cli_calc_fuse_shift_block_f - Load 2D edge-clamped shifted block in `float`
 * @p:   Compiled fused program
 * @k:   Operand slot index
 * @dx:  X pixel shift
 * @dy:  Y pixel shift
 * @off: Starting pixel index
 * @len: Number of pixels in block
 * @dst: Output float buffer
 *
 * Return: void
 */
void cli_calc_fuse_shift_block_f(const fprog *p,
                                 int          k,
                                 int64_t      dx,
                                 int64_t      dy,
                                 uint64_t     off,
                                 int          len,
                                 float *restrict dst)
{
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const float *a = im->array.F;
    VIEW_WALK(p, v, off, len, {
        int64_t sy       = (int64_t) y_ + dy;
        sy               = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy);
        const float *row = a + ((int64_t) z_ * ny + sy) * nx;
        float       *d   = dst + J;
        int64_t      s0  = (int64_t) x_ + dx;
        int          i0  = (s0 < 0) ? (((int64_t) N < -s0) ? N : (int) (-s0)) : 0;
        int64_t      lim = nx - s0;
        int          i1  = (lim <= (int64_t) i0) ? i0 : ((lim < (int64_t) N) ? (int) lim : N);
        float        v0  = row[0];
        for (int i = 0; i < i0; i++)
        {
            d[i] = v0;
        }
        if (i1 > i0)
        {
            __builtin_memcpy(d + i0, row + s0 + i0, (size_t) (i1 - i0) * sizeof(float));
        }
        float vn = row[nx - 1];
        for (int i = i1; i < N; i++)
        {
            d[i] = vn;
        }
    });
}

/**
 * cli_calc_fuse_medfilt_block_d - Evaluate KxK 2D median filter on a `double` block
 * @p:   Compiled fused program
 * @k:   Operand slot index
 * @K:   Window width (3, 5, or 7)
 * @off: Starting pixel index
 * @len: Number of pixels in block
 * @dst: Output double buffer
 *
 * Return: void
 */
void cli_calc_fuse_medfilt_block_d(const fprog *p,
                                   int          k,
                                   int          K,
                                   uint64_t     off,
                                   int          len,
                                   double *restrict dst)
{
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const int               rad = K >> 1;
    uint8_t                 dt  = im->md[0].datatype;
    MILK_ALIGNED(64) double col[7][FUSE_BLK + 6];
#define MEDFILT_LOAD_BODY(MBR)                                          \
    {                                                                   \
        const int W = N + K - 1;                                        \
        for (int r = 0; r < K; r++)                                     \
        {                                                               \
            int64_t sy = (int64_t) y_ + (r - rad);                      \
            sy         = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy);     \
            const __typeof__(im->array.MBR[0]) *row =                   \
                im->array.MBR + ((int64_t) z_ * ny + sy) * nx;          \
            double *cr = col[r];                                        \
            for (int u = 0; u < W; u++)                                 \
            {                                                           \
                int64_t sx = (int64_t) x_ + u - rad;                    \
                sx         = (sx < 0) ? 0 : ((sx >= nx) ? nx - 1 : sx); \
                cr[u]      = (double) row[sx];                          \
            }                                                           \
        }                                                               \
        cli_calc_fuse_medfilt_cols_d(K, N, col, dst + J);               \
    }
    VIEW_WALK(p, v, off, len,
              { MILK_FOR_EACH_DATATYPE(dt, MEDFILT_LOAD_BODY, MEDFILT_LOAD_BODY(D)) });
#undef MEDFILT_LOAD_BODY
}

/**
 * cli_calc_fuse_medfilt_block_f - Evaluate KxK 2D median filter on a `float` block
 * @p:   Compiled fused program
 * @k:   Operand slot index
 * @K:   Window width (3, 5, or 7)
 * @off: Starting pixel index
 * @len: Number of pixels in block
 * @dst: Output float buffer
 *
 * Return: void
 */
void cli_calc_fuse_medfilt_block_f(const fprog *p,
                                   int          k,
                                   int          K,
                                   uint64_t     off,
                                   int          len,
                                   float *restrict dst)
{
    const IMAGE  *im = p->img[k];
    const fview  *v  = &p->view[k];
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const int              rad = K >> 1;
    const float           *a   = im->array.F;
    MILK_ALIGNED(64) float col[7][FUSE_BLK + 6];
    VIEW_WALK(p, v, off, len, {
        const int W = N + K - 1;
        for (int r = 0; r < K; r++)
        {
            int64_t sy       = (int64_t) y_ + (r - rad);
            sy               = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy);
            const float *row = a + ((int64_t) z_ * ny + sy) * nx;
            float       *cr  = col[r];
            for (int u = 0; u < W; u++)
            {
                int64_t sx = (int64_t) x_ + u - rad;
                sx         = (sx < 0) ? 0 : ((sx >= nx) ? nx - 1 : sx);
                cr[u]      = row[sx];
            }
        }
        cli_calc_fuse_medfilt_cols_f(K, N, col, dst + J);
    });
}

/**
 * cli_calc_fuse_prep_matmuls - Prepare per-matmul single-thread contiguous column caches
 * @p:   Compiled fused program
 * @nth: Active worker team size (`1` enables contiguous full-column caching when `M > FUSE_BLK`)
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int cli_calc_fuse_prep_matmuls(fprog *p, int nth)
{
    for (int m = 0; m < p->nmatmul; m++)
    {
        fuse_matmul_t *mm = &p->matmul[m];
        if (nth == 1 && !mm->tA && mm->M > FUSE_BLK)
        {
            size_t need = (size_t) mm->M * sizeof(double);
            if (need > mm->col_cap)
            {
                void *nb = realloc(mm->col_buf, need);
                if (nb == NULL)
                {
                    return -1;
                }
                mm->col_buf = nb;
                mm->col_cap = need;
            }
            mm->cached_col = -1;
        }
        else
        {
            mm->cached_col = -2;
        }
    }
    return 0;
}

/**
 * cli_calc_fuse_matmul_block_d - Evaluate `FI_MATMUL` block `[off, off + len)` in `double`
 * @p:   Compiled fused program
 * @m:   Matmul descriptor slot (`p->matmul[m]`)
 * @off: Starting flat output pixel index
 * @len: Number of pixels in block
 * @dst: Output double buffer
 */
void cli_calc_fuse_matmul_block_d(const fprog *p,
                                  int          m,
                                  uint64_t     off,
                                  int          len,
                                  double *restrict dst)
{
    fuse_matmul_t *mm  = (fuse_matmul_t *) &p->matmul[m];
    const IMAGE   *imA = p->img[mm->imgA];
    const IMAGE   *imB = p->img[mm->imgB];
    if (mm->cached_col >= -1 && mm->col_buf != NULL && mm->M > (uint64_t) len)
    {
        double  *cbuf = (double *) mm->col_buf;
        uint64_t done = 0;
        while (done < (uint64_t) len)
        {
            uint64_t idx = off + done;
            uint64_t col = idx / mm->M;
            uint64_t row = idx % mm->M;
            uint64_t rem = (uint64_t) len - done;
            if (col >= mm->N)
            {
                break;
            }
            uint64_t seg = (mm->M - row < rem) ? (mm->M - row) : rem;
            if (mm->cached_col != (int64_t) col)
            {
                arith_image_matmul_block_d(imA, 0, imB, mm->tB, cbuf, mm->M, mm->K, mm->N,
                                           col * mm->M, mm->M);
                mm->cached_col = (int64_t) col;
            }
            __builtin_memcpy(dst + done, cbuf + row, sizeof(double) * (size_t) seg);
            done += seg;
        }
        return;
    }
    arith_image_matmul_block_d(imA, mm->tA, imB, mm->tB, dst, mm->M, mm->K, mm->N, off,
                               (uint64_t) len);
}

/**
 * cli_calc_fuse_matmul_block_f - Evaluate `FI_MATMUL` block `[off, off + len)` in `float`
 * @p:   Compiled fused program
 * @m:   Matmul descriptor slot (`p->matmul[m]`)
 * @off: Starting flat output pixel index
 * @len: Number of pixels in block
 * @dst: Output float buffer
 */
void cli_calc_fuse_matmul_block_f(const fprog *p, int m, uint64_t off, int len, float *restrict dst)
{
    fuse_matmul_t *mm  = (fuse_matmul_t *) &p->matmul[m];
    const IMAGE   *imA = p->img[mm->imgA];
    const IMAGE   *imB = p->img[mm->imgB];
    if (mm->cached_col >= -1 && mm->col_buf != NULL && mm->M > (uint64_t) len)
    {
        float   *cbuf = (float *) mm->col_buf;
        uint64_t done = 0;
        while (done < (uint64_t) len)
        {
            uint64_t idx = off + done;
            uint64_t col = idx / mm->M;
            uint64_t row = idx % mm->M;
            uint64_t rem = (uint64_t) len - done;
            if (col >= mm->N)
            {
                break;
            }
            uint64_t seg = (mm->M - row < rem) ? (mm->M - row) : rem;
            if (mm->cached_col != (int64_t) col)
            {
                arith_matmul_block_f(imA->array.F, 0, imB->array.F, mm->tB, cbuf, mm->M, mm->K,
                                     mm->N, col * mm->M, mm->M);
                mm->cached_col = (int64_t) col;
            }
            __builtin_memcpy(dst + done, cbuf + row, sizeof(float) * (size_t) seg);
            done += seg;
        }
        return;
    }
    arith_matmul_block_f(imA->array.F, mm->tA, imB->array.F, mm->tB, dst, mm->M, mm->K, mm->N, off,
                         (uint64_t) len);
}

/**
 * cli_calc_fuse_eye_block_d - Fill `FI_EYE` block `[off, off + len)` in `double`
 * @p:     Compiled fused program
 * @kdiag: Diagonal offset (`0` = main diagonal)
 * @off:   Starting flat output pixel index
 * @len:   Number of pixels in block
 * @dst:   Output double buffer
 */
void cli_calc_fuse_eye_block_d(const fprog *p,
                               int64_t      kdiag,
                               uint64_t     off,
                               int          len,
                               double *restrict dst)
{
    memset(dst, 0, sizeof(double) * (size_t) len);
    if (len <= 0)
    {
        return;
    }
    uint64_t nx = p->ishape[0];
    uint64_t c0 = off / nx;
    uint64_t c1 = (off + (uint64_t) len - 1) / nx;
    for (uint64_t c = c0; c <= c1; c++)
    {
        int64_t r = (int64_t) c - kdiag;
        if (r >= 0 && (uint64_t) r < nx)
        {
            uint64_t idx = c * nx + (uint64_t) r;
            if (idx >= off && idx < off + (uint64_t) len)
            {
                dst[idx - off] = 1.0;
            }
        }
    }
}

/**
 * cli_calc_fuse_eye_block_f - Fill `FI_EYE` block `[off, off + len)` in `float`
 * @p:     Compiled fused program
 * @kdiag: Diagonal offset (`0` = main diagonal)
 * @off:   Starting flat output pixel index
 * @len:   Number of pixels in block
 * @dst:   Output float buffer
 */
void cli_calc_fuse_eye_block_f(const fprog *p,
                               int64_t      kdiag,
                               uint64_t     off,
                               int          len,
                               float *restrict dst)
{
    memset(dst, 0, sizeof(float) * (size_t) len);
    if (len <= 0)
    {
        return;
    }
    uint64_t nx = p->ishape[0];
    uint64_t c0 = off / nx;
    uint64_t c1 = (off + (uint64_t) len - 1) / nx;
    for (uint64_t c = c0; c <= c1; c++)
    {
        int64_t r = (int64_t) c - kdiag;
        if (r >= 0 && (uint64_t) r < nx)
        {
            uint64_t idx = c * nx + (uint64_t) r;
            if (idx >= off && idx < off + (uint64_t) len)
            {
                dst[idx - off] = 1.0f;
            }
        }
    }
}
