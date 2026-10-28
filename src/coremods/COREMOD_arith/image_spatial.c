// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_spatial.c
 * @brief Shared spatial, CSR segment-reduction, separable Gaussian, cumsum, and binning kernels.
 *
 * Role & Goal:
 *   Provides pure `IMAGE`-level spatial and pre-stage kernels shared by `libmilkscript/calc`
 *   and Core/Plugin modules (`image_filter`, `image_basic`, `COREMOD_arith`).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "arith_team.h"
#include "image_elem_block.h"
#include "image_spatial.h"
#include "image_stats.h"
#include "libmilkcommon/milk_compiler.h"
#include "libmilkdata/milk_type_dispatch.h"

/* =========================================================================
 * 1. CSR Map (`pixunmap`) and Radial Profile (`rprof`) Engine
 * ========================================================================= */

/**
 * arith_csr_cache_free - Release all heap buffers owned by @csr and zero the struct
 * @csr: CSR cache structure to free
 *
 * Return: void
 */
void arith_csr_cache_free(arith_csr_cache_t *csr)
{
    if (csr == NULL)
    {
        return;
    }
    free(csr->row_ptr);
    free(csr->pix_idx);
    free(csr->inv_cnt);
    free(csr->pix_bin);
    memset(csr, 0, sizeof(*csr));
}

/** ensure_csr_caps - Grow CSR cache buffers of @csr to accommodate @nbins and @nel. */
static int ensure_csr_caps(arith_csr_cache_t *csr, int nbins, uint64_t nel)
{
    size_t need_ptr = (size_t) (nbins + 1);
    size_t need_idx = (size_t) nel;
    if (need_ptr > csr->ptr_cap)
    {
        int32_t *rp = (int32_t *) realloc(csr->row_ptr, need_ptr * sizeof(int32_t));
        if (rp != NULL)
        {
            csr->row_ptr = rp;
        }
        double *ic = (double *) realloc(csr->inv_cnt, need_ptr * sizeof(double));
        if (ic != NULL)
        {
            csr->inv_cnt = ic;
        }
        if (rp == NULL || ic == NULL)
        {
            return -1;
        }
        csr->ptr_cap = need_ptr;
    }
    if (need_idx > csr->idx_cap)
    {
        int32_t *pi = (int32_t *) realloc(csr->pix_idx, need_idx * sizeof(int32_t));
        if (pi == NULL)
        {
            return -1;
        }
        csr->pix_idx = pi;
        csr->idx_cap = need_idx;
    }
    return 0;
}

/**
 * arith_image_map_nbins - Scan @map_im to infer `max(map) + 1` bin count for `pixunmap`
 * @map_im: Index map image
 *
 * Return: Inferred positive bin count (`<= 16777216`), or -1 if invalid.
 */
int arith_image_map_nbins(const IMAGE *map_im)
{
    if (map_im == NULL || map_im->md == NULL || !arith_is_real_dt(map_im->md[0].datatype))
    {
        return -1;
    }
    uint64_t nel   = map_im->md[0].nelement;
    uint8_t  dt    = map_im->md[0].datatype;
    int64_t  max_i = -1;
#define MAP_MAX_BODY(MBR)                                                \
    {                                                                    \
        const __typeof__(map_im->array.MBR[0]) *src = map_im->array.MBR; \
        for (uint64_t ii = 0; ii < nel; ii++)                            \
        {                                                                \
            double m = (double) src[ii];                                 \
            if (m >= 0.0 && m < 16777216.0)                              \
            {                                                            \
                int64_t v = (int64_t) m;                                 \
                if (v > max_i)                                           \
                {                                                        \
                    max_i = v;                                           \
                }                                                        \
            }                                                            \
        }                                                                \
    }
    MILK_FOR_EACH_DATATYPE(dt, MAP_MAX_BODY, MAP_MAX_BODY(D))
#undef MAP_MAX_BODY
    return (max_i >= 0 && max_i < 16777216) ? (int) (max_i + 1) : -1;
}

/**
 * arith_image_rprof_nbins - Compute radial bin count from @ishape corners around `(cx, cy)`
 * @ishape: 3D iteration shape
 * @cx:     Center X coordinate
 * @cy:     Center Y coordinate
 *
 * Return: Number of radial bins (`floor(r_max + 0.5) + 1`), or -1 if out of bounds.
 */
int arith_image_rprof_nbins(const uint32_t ishape[3], double cx, double cy)
{
    double dx0 = fabs(cx);
    double dx1 = fabs((double) (ishape[0] > 0 ? ishape[0] - 1 : 0) - cx);
    double dy0 = fabs(cy);
    double dy1 = fabs((double) (ishape[1] > 0 ? ishape[1] - 1 : 0) - cy);
    double mdx = (dx0 > dx1) ? dx0 : dx1;
    double mdy = (dy0 > dy1) ? dy0 : dy1;
    int    nb  = (int) llround(sqrt(mdx * mdx + mdy * mdy)) + 1;
    return (nb > 0 && nb <= 16777216) ? nb : -1;
}

/**
 * arith_csr_build_map - Build CSR bin index tables from integer/float bin map image @map_im
 * @csr:    CSR cache to populate
 * @map_im: Bin index map image
 * @nbins:  Number of output bins
 * @nel:    Number of map elements
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int arith_csr_build_map(arith_csr_cache_t *csr, const IMAGE *map_im, int nbins, uint64_t nel)
{
    if (ensure_csr_caps(csr, nbins, nel) != 0)
    {
        return -1;
    }
    int32_t *rp  = csr->row_ptr;
    int32_t *pi  = csr->pix_idx;
    double  *inv = csr->inv_cnt;
    memset(rp, 0, (size_t) (nbins + 1) * sizeof(int32_t));
    uint8_t dt    = map_im->md[0].datatype;
    double  max_n = (double) nbins;
#define CSR_CNT_BODY(MBR)                                                \
    {                                                                    \
        const __typeof__(map_im->array.MBR[0]) *src = map_im->array.MBR; \
        for (uint64_t ii = 0; ii < nel; ii++)                            \
        {                                                                \
            double m = (double) src[ii];                                 \
            if (!arith_isnan_d(m) && m >= 0.0 && m < max_n)              \
            {                                                            \
                int b = (int) (int64_t) m;                               \
                if (b >= 0 && b < nbins)                                 \
                {                                                        \
                    rp[b + 1]++;                                         \
                }                                                        \
            }                                                            \
        }                                                                \
    }
    MILK_FOR_EACH_DATATYPE(dt, CSR_CNT_BODY, CSR_CNT_BODY(D))
#undef CSR_CNT_BODY
    for (int b = 0; b < nbins; b++)
    {
        int32_t cnt = rp[b + 1];
        inv[b]      = (cnt > 0) ? (1.0 / (double) cnt) : 0.0;
        rp[b + 1] += rp[b];
    }
    int32_t *cur = (int32_t *) malloc((size_t) nbins * sizeof(int32_t));
    if (cur == NULL)
    {
        return -1;
    }
    memcpy(cur, rp, (size_t) nbins * sizeof(int32_t));
#define CSR_FILL_BODY(MBR)                                               \
    {                                                                    \
        const __typeof__(map_im->array.MBR[0]) *src = map_im->array.MBR; \
        for (uint64_t ii = 0; ii < nel; ii++)                            \
        {                                                                \
            double m = (double) src[ii];                                 \
            if (!arith_isnan_d(m) && m >= 0.0 && m < max_n)              \
            {                                                            \
                int b = (int) (int64_t) m;                               \
                if (b >= 0 && b < nbins)                                 \
                {                                                        \
                    pi[cur[b]++] = (int32_t) ii;                         \
                }                                                        \
            }                                                            \
        }                                                                \
    }
    MILK_FOR_EACH_DATATYPE(dt, CSR_FILL_BODY, CSR_FILL_BODY(D))
#undef CSR_FILL_BODY
    free(cur);
    csr->valid    = 1;
    csr->map_im   = map_im;
    csr->map_cnt0 = map_im->md[0].cnt0;
    csr->nbins    = nbins;
    return 0;
}

/**
 * arith_csr_build_rprof - Build CSR radial-bin index tables for 2D shape @ishape around `(cx, cy)`
 * @csr:    CSR cache to populate
 * @ishape: 3D iteration shape
 * @cx:     Center X coordinate
 * @cy:     Center Y coordinate
 * @nbins:  Number of radial bins
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int arith_csr_build_rprof(arith_csr_cache_t *csr,
                          const uint32_t     ishape[3],
                          double             cx,
                          double             cy,
                          int                nbins)
{
    uint64_t nx  = ishape[0];
    uint64_t ny  = ishape[1];
    uint64_t nel = nx * ny;
    if (ensure_csr_caps(csr, nbins, nel) != 0)
    {
        return -1;
    }
    int32_t *rp   = csr->row_ptr;
    int32_t *pi   = csr->pix_idx;
    double  *inv  = csr->inv_cnt;
    int32_t *bins = (int32_t *) malloc((size_t) nel * sizeof(int32_t));
    if (bins == NULL)
    {
        return -1;
    }
    memset(rp, 0, (size_t) (nbins + 1) * sizeof(int32_t));
    for (uint64_t y = 0; y < ny; y++)
    {
        double   dy  = (double) y - cy;
        double   dy2 = dy * dy;
        uint64_t row = y * nx;
        for (uint64_t x = 0; x < nx; x++)
        {
            double dx = (double) x - cx;
            int    b  = (int) llround(sqrt(dx * dx + dy2));
            if (b >= 0 && b < nbins)
            {
                bins[row + x] = b;
                rp[b + 1]++;
            }
            else
            {
                bins[row + x] = -1;
            }
        }
    }
    for (int b = 0; b < nbins; b++)
    {
        int32_t cnt = rp[b + 1];
        inv[b]      = (cnt > 0) ? (1.0 / (double) cnt) : 0.0;
        rp[b + 1] += rp[b];
    }
    int32_t *cur = (int32_t *) malloc((size_t) nbins * sizeof(int32_t));
    if (cur == NULL)
    {
        free(bins);
        return -1;
    }
    memcpy(cur, rp, (size_t) nbins * sizeof(int32_t));
    for (uint64_t ii = 0; ii < nel; ii++)
    {
        int b = bins[ii];
        if (b >= 0)
        {
            pi[cur[b]++] = (int32_t) ii;
        }
    }
    free(cur);
    free(bins);
    csr->valid = 1;
    memcpy(csr->ishape, ishape, sizeof(csr->ishape));
    csr->nbins = nbins;
    csr->cx    = cx;
    csr->cy    = cy;
    return 0;
}

/**
 * arith_csr_ensure_rprof_bins - Build per-pixel radial bin table of @csr for @ishape
 * @csr:    CSR cache structure
 * @ishape: 3D iteration shape
 * @cx:     Center X coordinate
 * @cy:     Center Y coordinate
 * @nbins:  Number of radial bins
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int arith_csr_ensure_rprof_bins(arith_csr_cache_t *csr,
                                const uint32_t     ishape[3],
                                double             cx,
                                double             cy,
                                int                nbins)
{
    uint64_t nx  = ishape[0];
    uint64_t ny  = ishape[1];
    uint64_t nel = nx * ny;
    if (csr->bin_valid && csr->nbins == nbins && csr->cx == cx && csr->cy == cy &&
        memcmp(csr->ishape, ishape, sizeof(csr->ishape)) == 0)
    {
        return 0;
    }
    if (nel > csr->bin_cap)
    {
        int32_t *nb = (int32_t *) realloc(csr->pix_bin, nel * sizeof(int32_t));
        if (nb == NULL)
        {
            return -1;
        }
        csr->pix_bin = nb;
        csr->bin_cap = nel;
    }
    int32_t *bins = csr->pix_bin;
    for (uint64_t y = 0; y < ny; y++)
    {
        double   dy  = (double) y - cy;
        double   dy2 = dy * dy;
        uint64_t row = y * nx;
        for (uint64_t x = 0; x < nx; x++)
        {
            double dx     = (double) x - cx;
            int    b      = (int) floor(sqrt(dx * dx + dy2) + 0.5);
            bins[row + x] = (b >= 0 && b < nbins) ? b : -1;
        }
    }
    csr->bin_valid = 1;
    csr->nbins     = nbins;
    csr->cx        = cx;
    csr->cy        = cy;
    memcpy(csr->ishape, ishape, sizeof(csr->ishape));
    return 0;
}

typedef struct
{
    const IMAGE   *src_im;
    const int32_t *rp;
    const int32_t *pi;
    const double  *inv;
    IMAGE         *res;
} csr_gather_t;

/** csr_gather_job() - team job: the means of bins [lo, hi) from the CSR tables. */
static void csr_gather_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const csr_gather_t *j   = (const csr_gather_t *) ctx;
    uint8_t             sdt = j->src_im->md[0].datatype;
    uint8_t             odt = j->res->md[0].datatype;
    (void) tid;
    (void) nth;
#define CSR_GATHER_BODY(MBR)                                                         \
    {                                                                                \
        const __typeof__(j->src_im->array.MBR[0]) *src = j->src_im->array.MBR;       \
        for (uint64_t b = lo; b < hi; b++)                                           \
        {                                                                            \
            int32_t s0  = j->rp[b];                                                  \
            int32_t s1  = j->rp[b + 1];                                              \
            double  sum = 0.0;                                                       \
            _Pragma("omp simd reduction(+ : sum)") for (int32_t q = s0; q < s1; q++) \
            {                                                                        \
                sum += (double) src[j->pi[q]];                                       \
            }                                                                        \
            double val = sum * j->inv[b];                                            \
            if (odt == _DATATYPE_FLOAT)                                              \
            {                                                                        \
                j->res->array.F[b] = (float) val;                                    \
            }                                                                        \
            else                                                                     \
            {                                                                        \
                j->res->array.D[b] = val;                                            \
            }                                                                        \
        }                                                                            \
    }
    MILK_FOR_EACH_DATATYPE(sdt, CSR_GATHER_BODY, CSR_GATHER_BODY(D))
#undef CSR_GATHER_BODY
}

/**
 * arith_csr_gather - Gather bin means from @src_im into @res using cached CSR tables @csr
 * @src_im: Source image
 * @csr:    Populated CSR table cache
 * @nbins:  Number of output bins
 * @nel:    Number of source elements (for team sizing)
 * @res:    Destination 1D float32/float64 image
 *
 * Return: void
 */
void arith_csr_gather(const IMAGE             *src_im,
                      const arith_csr_cache_t *csr,
                      int                      nbins,
                      uint64_t                 nel,
                      IMAGE                   *res)
{
    int nth = milk_omp_team_size(nel, ARITH_SPATIAL_OMP_MIN);
    if (nth > nbins / 16)
    {
        nth = (nbins >= 16) ? nbins / 16 : 1;
    }
    csr_gather_t job = { src_im, csr->row_ptr, csr->pix_idx, csr->inv_cnt, res };
    arith_team_run_chunks(nth, (uint64_t) nbins, 16, csr_gather_job, &job);
}

/* =========================================================================
 * 2. Separable 2-Pass SIMD Gaussian Filter
 * ========================================================================= */

/**
 * arith_build_gauss_weights - Compute normalized 1D symmetric Gaussian tap weights
 * @sigma: Standard deviation in pixels
 * @max_R: Maximum tap radius (`<= 0` defaults to `min(ceil(3 * sigma), 32)`)
 * @out_R: Output tap radius `R`
 * @w:     Output half-kernel weights `w[0 .. R]`
 *
 * Return: void
 */
void arith_build_gauss_weights(double sigma, int max_R, int *out_R, double w[ARITH_GAUSS_MAX_R + 1])
{
    if (sigma <= 0.0)
    {
        *out_R = 0;
        w[0]   = 1.0;
        return;
    }
    int R   = (max_R > 0) ? max_R : (int) ceil(3.0 * sigma);
    int cap = (max_R > 0) ? ARITH_GAUSS_MAX_R : 32;
    if (R > cap)
    {
        R = cap;
    }
    *out_R         = R;
    double inv_2s2 = 0.5 / (sigma * sigma);
    double sum     = 1.0;
    w[0]           = 1.0;
    for (int d = 1; d <= R; d++)
    {
        w[d] = exp(-(double) (d * d) * inv_2s2);
        sum += 2.0 * w[d];
    }
    double inv_sum = 1.0 / sum;
    for (int d = 0; d <= R; d++)
    {
        w[d] *= inv_sum;
    }
}

/** GAUSS_JOBS - x-pass and y-pass team jobs for one element type (rows [lo, hi) of all planes). */
#define GAUSS_JOBS(SFX, FT)                                                              \
    typedef struct                                                                       \
    {                                                                                    \
        FT       *io;                                                                    \
        FT       *tmp;                                                                   \
        FT       *pads;                                                                  \
        int64_t   nx;                                                                    \
        int64_t   ny;                                                                    \
        int64_t   pad_stride;                                                            \
        int       Rx;                                                                    \
        int       Ry;                                                                    \
        const FT *wx;                                                                    \
        const FT *wy;                                                                    \
    } gauss_ctx_##SFX;                                                                   \
                                                                                         \
    static void gauss_x_job_##SFX(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi) \
    {                                                                                    \
        const gauss_ctx_##SFX *g   = (const gauss_ctx_##SFX *) ctx;                      \
        const int64_t          nx  = g->nx;                                              \
        FT                    *pad = g->pads + (int64_t) tid * g->pad_stride;            \
        FT                    *p0  = pad + g->Rx;                                        \
        (void) nth;                                                                      \
        for (uint64_t r = lo; r < hi; r++)                                               \
        {                                                                                \
            const FT *srow = g->io + (int64_t) r * nx;                                   \
            FT       *trow = g->tmp + (int64_t) r * nx;                                  \
            for (int d = 0; d < g->Rx; d++)                                              \
            {                                                                            \
                pad[d]     = srow[0];                                                    \
                p0[nx + d] = srow[nx - 1];                                               \
            }                                                                            \
            memcpy(p0, srow, (size_t) nx * sizeof(FT));                                  \
            const FT w0 = g->wx[0];                                                      \
            _Pragma("omp simd") for (int64_t x = 0; x < nx; x++)                         \
            {                                                                            \
                trow[x] = w0 * p0[x];                                                    \
            }                                                                            \
            for (int d = 1; d <= g->Rx; d++)                                             \
            {                                                                            \
                const FT  wd = g->wx[d];                                                 \
                const FT *pl = p0 - d;                                                   \
                const FT *pr = p0 + d;                                                   \
                _Pragma("omp simd") for (int64_t x = 0; x < nx; x++)                     \
                {                                                                        \
                    trow[x] += wd * (pl[x] + pr[x]);                                     \
                }                                                                        \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    static void gauss_y_job_##SFX(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi) \
    {                                                                                    \
        const gauss_ctx_##SFX *g  = (const gauss_ctx_##SFX *) ctx;                       \
        const int64_t          nx = g->nx;                                               \
        const int64_t          ny = g->ny;                                               \
        (void) tid;                                                                      \
        (void) nth;                                                                      \
        for (uint64_t r = lo; r < hi; r++)                                               \
        {                                                                                \
            const int64_t z      = (int64_t) r / ny;                                     \
            const int64_t y      = (int64_t) r % ny;                                     \
            const FT     *tplane = g->tmp + z * ny * nx;                                 \
            FT           *orow   = g->io + (int64_t) r * nx;                             \
            const FT     *tc     = tplane + y * nx;                                      \
            const FT      w0     = g->wy[0];                                             \
            _Pragma("omp simd") for (int64_t x = 0; x < nx; x++)                         \
            {                                                                            \
                orow[x] = w0 * tc[x];                                                    \
            }                                                                            \
            for (int d = 1; d <= g->Ry; d++)                                             \
            {                                                                            \
                const FT  wd = g->wy[d];                                                 \
                int64_t   ym = (y - d < 0) ? 0 : (y - d);                                \
                int64_t   yp = (y + d >= ny) ? (ny - 1) : (y + d);                       \
                const FT *tm = tplane + ym * nx;                                         \
                const FT *tp = tplane + yp * nx;                                         \
                _Pragma("omp simd") for (int64_t x = 0; x < nx; x++)                     \
                {                                                                        \
                    orow[x] += wd * (tm[x] + tp[x]);                                     \
                }                                                                        \
            }                                                                            \
        }                                                                                \
    }

GAUSS_JOBS(f, float)
GAUSS_JOBS(d, double)
#undef GAUSS_JOBS

/**
 * arith_gauss_pass_f - Separable 2D Gaussian convolution in float32 across worker team
 * @io:  In-place float32 image array (`nx * ny * nz`)
 * @tmp: Scratch buffer of size `(nx * ny * nz + nth * (nx + 2 * Rx + 16))` floats
 * @nx:  X dimension
 * @ny:  Y dimension
 * @nz:  Z dimension
 * @Rx:  X tap radius
 * @Ry:  Y tap radius
 * @wx:  X tap weights `wx[0 .. Rx]`
 * @wy:  Y tap weights `wy[0 .. Ry]`
 * @nth: Number of worker threads
 *
 * Return: void
 */
void arith_gauss_pass_f(float        *io,
                        float        *tmp,
                        int64_t       nx,
                        int64_t       ny,
                        int64_t       nz,
                        int           Rx,
                        int           Ry,
                        const double *wx,
                        const double *wy,
                        int           nth)
{
    float wxf[ARITH_GAUSS_MAX_R + 1];
    float wyf[ARITH_GAUSS_MAX_R + 1];
    for (int d = 0; d <= Rx; d++)
    {
        wxf[d] = (float) wx[d];
    }
    for (int d = 0; d <= Ry; d++)
    {
        wyf[d] = (float) wy[d];
    }
    int64_t     pad_stride = nx + 2 * (int64_t) Rx + 16;
    gauss_ctx_f g          = { io, tmp, tmp + nx * ny * nz, nx, ny, pad_stride, Rx, Ry, wxf, wyf };
    arith_team_run(nth, (uint64_t) (ny * nz), gauss_x_job_f, &g);
    arith_team_run(nth, (uint64_t) (ny * nz), gauss_y_job_f, &g);
}

/**
 * arith_gauss_pass_d - Separable 2D Gaussian convolution in float64 across worker team
 * @io:  In-place float64 image array (`nx * ny * nz`)
 * @tmp: Scratch buffer of size `(nx * ny * nz + nth * (nx + 2 * Rx + 16))` doubles
 * @nx:  X dimension
 * @ny:  Y dimension
 * @nz:  Z dimension
 * @Rx:  X tap radius
 * @Ry:  Y tap radius
 * @wx:  X tap weights `wx[0 .. Rx]`
 * @wy:  Y tap weights `wy[0 .. Ry]`
 * @nth: Number of worker threads
 *
 * Return: void
 */
void arith_gauss_pass_d(double       *io,
                        double       *tmp,
                        int64_t       nx,
                        int64_t       ny,
                        int64_t       nz,
                        int           Rx,
                        int           Ry,
                        const double *wx,
                        const double *wy,
                        int           nth)
{
    int64_t     pad_stride = nx + 2 * (int64_t) Rx + 16;
    gauss_ctx_d g          = { io, tmp, tmp + nx * ny * nz, nx, ny, pad_stride, Rx, Ry, wx, wy };
    arith_team_run(nth, (uint64_t) (ny * nz), gauss_x_job_d, &g);
    arith_team_run(nth, (uint64_t) (ny * nz), gauss_y_job_d, &g);
}

/**
 * arith_image_gauss2d_im - Separable 2D Gaussian convolution from @src into @dst
 * @src:         Source image (may equal @dst for in-place filtering on float32/float64)
 * @dst:         Destination image (float32 or float64, same geometry as @src)
 * @sx:          X standard deviation in pixels
 * @sy:          Y standard deviation in pixels
 * @max_R:       Optional explicit radius cap (`<= 0` uses `min(ceil(3 * sigma), 32)`)
 * @scratch_buf: Optional reusable scratch pointer (or NULL to allocate/free internally)
 * @scratch_cap: Capacity in bytes of `*scratch_buf`
 *
 * Return: 0 on success, -1 on error.
 */
int arith_image_gauss2d_im(const IMAGE *src,
                           IMAGE       *dst,
                           double       sx,
                           double       sy,
                           int          max_R,
                           void       **scratch_buf,
                           size_t      *scratch_cap)
{
    if (src == NULL || dst == NULL || src->md == NULL || dst->md == NULL)
    {
        return -1;
    }
    const int64_t nx = (int64_t) dst->md[0].size[0];
    const int64_t ny =
        (dst->md[0].naxis > 1 && dst->md[0].size[1] > 0) ? (int64_t) dst->md[0].size[1] : 1;
    const int64_t nz =
        (dst->md[0].naxis > 2 && dst->md[0].size[2] > 0) ? (int64_t) dst->md[0].size[2] : 1;
    const uint64_t nel    = (uint64_t) (nx * ny * nz);
    const uint8_t  sdt    = src->md[0].datatype;
    const uint8_t  ddt    = dst->md[0].datatype;
    const int      is_flt = (ddt == _DATATYPE_FLOAT);
    if (!is_flt && ddt != _DATATYPE_DOUBLE)
    {
        return -1;
    }

    if (src != dst)
    {
        if (sdt == ddt)
        {
            memcpy(dst->array.raw, src->array.raw, nel * (is_flt ? sizeof(float) : sizeof(double)));
        }
        else if (is_flt)
        {
            for (uint64_t off = 0; off < nel; off += ARITH_ELEM_BLK)
            {
                uint64_t len = (nel - off < ARITH_ELEM_BLK) ? nel - off : ARITH_ELEM_BLK;
                arith_load_block_f_inline(src, off, len, dst->array.F + off);
            }
        }
        else
        {
            for (uint64_t off = 0; off < nel; off += ARITH_ELEM_BLK)
            {
                uint64_t len = (nel - off < ARITH_ELEM_BLK) ? nel - off : ARITH_ELEM_BLK;
                arith_load_block_d_inline(src, off, len, dst->array.D + off);
            }
        }
    }

    int    Rx = 0, Ry = 0;
    double wx[ARITH_GAUSS_MAX_R + 1];
    double wy[ARITH_GAUSS_MAX_R + 1];
    arith_build_gauss_weights(sx, max_R, &Rx, wx);
    arith_build_gauss_weights(sy, max_R, &Ry, wy);
    if (Rx == 0 && Ry == 0)
    {
        return 0;
    }

    size_t el_bytes = is_flt ? sizeof(float) : sizeof(double);
    int    nth = milk_omp_team_size(nel * (uint64_t) (1 + (Rx + Ry) / 4), ARITH_SPATIAL_OMP_MIN);
    if ((int64_t) nth > ny * nz)
    {
        nth = (int) (ny * nz);
    }
    int64_t pad_stride = nx + 2 * (int64_t) Rx + 16;
    size_t  need       = ((size_t) nel + (size_t) nth * (size_t) pad_stride) * el_bytes;

    void *local_buf = NULL;
    void *work_buf  = NULL;
    if (scratch_buf != NULL && scratch_cap != NULL)
    {
        if (*scratch_cap < need)
        {
            void *na = realloc(*scratch_buf, need);
            if (na == NULL)
            {
                return -1;
            }
            *scratch_buf = na;
            *scratch_cap = need;
        }
        work_buf = *scratch_buf;
    }
    else
    {
        local_buf = malloc(need);
        if (local_buf == NULL)
        {
            return -1;
        }
        work_buf = local_buf;
    }

    if (is_flt)
    {
        arith_gauss_pass_f(dst->array.F, (float *) work_buf, nx, ny, nz, Rx, Ry, wx, wy, nth);
    }
    else
    {
        arith_gauss_pass_d(dst->array.D, (double *) work_buf, nx, ny, nz, Rx, Ry, wx, wy, nth);
    }

    free(local_buf);
    return 0;
}

/* =========================================================================
 * 3. Cumulative Sum (`cumsum`) Along Axis 0, 1, or 2
 * ========================================================================= */

typedef struct
{
    void    *data;
    int      is_flt;
    int      axis;
    int64_t  nx;
    int64_t  ny;
    int64_t  nz;
    uint64_t nxc;
} cumsum_job_t;

#define CUMSUM_XC 256

/** CUMSUM_LINE - inclusive running sum along @j->axis of items [lo, hi), accumulated in double. */
#define CUMSUM_LINE(FT)                                                                      \
    do                                                                                       \
    {                                                                                        \
        FT *v = (FT *) j->data;                                                              \
        if (j->axis == 0)                                                                    \
        {                                                                                    \
            for (uint64_t r = lo; r < hi; r++)                                               \
            {                                                                                \
                FT    *row = v + (int64_t) r * j->nx;                                        \
                double acc = 0.0;                                                            \
                for (int64_t x = 0; x < j->nx; x++)                                          \
                {                                                                            \
                    acc += (double) row[x];                                                  \
                    row[x] = (FT) acc;                                                       \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            double acc[CUMSUM_XC];                                                           \
            for (uint64_t c = lo; c < hi; c++)                                               \
            {                                                                                \
                int64_t line = (int64_t) (c / j->nxc);                                       \
                int64_t x0   = (int64_t) (c % j->nxc) * CUMSUM_XC;                           \
                int64_t x1   = (x0 + CUMSUM_XC < j->nx) ? x0 + CUMSUM_XC : j->nx;            \
                int64_t n    = (j->axis == 1) ? j->ny : j->nz;                               \
                FT     *base = (j->axis == 1) ? v + line * j->nx * j->ny : v + line * j->nx; \
                int64_t step = (j->axis == 1) ? j->nx : j->nx * j->ny;                       \
                for (int64_t x = x0; x < x1; x++)                                            \
                {                                                                            \
                    acc[x - x0] = 0.0;                                                       \
                }                                                                            \
                for (int64_t t = 0; t < n; t++)                                              \
                {                                                                            \
                    FT *row = base + t * step;                                               \
                    _Pragma("omp simd") for (int64_t x = x0; x < x1; x++)                    \
                    {                                                                        \
                        acc[x - x0] += (double) row[x];                                      \
                        row[x] = (FT) acc[x - x0];                                           \
                    }                                                                        \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
    } while (0)

/** cumsum_job() - team job: lines [lo, hi) of the cumulative sum. */
static void cumsum_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const cumsum_job_t *j = (const cumsum_job_t *) ctx;
    (void) tid;
    (void) nth;
    if (j->is_flt)
    {
        CUMSUM_LINE(float);
    }
    else
    {
        CUMSUM_LINE(double);
    }
}
#undef CUMSUM_LINE

/**
 * arith_image_cumsum_inplace_im - Compute inclusive cumulative sum of @im along @axis in-place
 * @im:   Target float32 or float64 image
 * @axis: Axis index (`0 .. naxis - 1`)
 *
 * Return: 0 on success, -1 if @axis is out of range or @im is invalid.
 */
int arith_image_cumsum_inplace_im(IMAGE *im, int axis)
{
    if (im == NULL || im->md == NULL || axis < 0 || axis >= im->md[0].naxis)
    {
        return -1;
    }
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const int64_t nz =
        (im->md[0].naxis > 2 && im->md[0].size[2] > 0) ? (int64_t) im->md[0].size[2] : 1;
    cumsum_job_t job    = { im->array.raw,
                            (im->md[0].datatype == _DATATYPE_FLOAT),
                            axis,
                            nx,
                            ny,
                            nz,
                            (uint64_t) ((nx + CUMSUM_XC - 1) / CUMSUM_XC) };
    uint64_t     nitems = (axis == 0)   ? (uint64_t) (ny * nz)
                          : (axis == 1) ? (uint64_t) nz * job.nxc
                                        : (uint64_t) ny * job.nxc;
    int          nth    = milk_omp_team_size((uint64_t) (nx * ny * nz), ARITH_SPATIAL_OMP_MIN);
    arith_team_run(nth, nitems, cumsum_job, &job);
    return 0;
}

/* =========================================================================
 * 4. 2D/3D Box Binning (`bin2d` / `imcontract`)
 * ========================================================================= */

/**
 * arith_image_bin3d_im - Sum `bx x by x bz` pixel blocks from @src into @dst
 * @src: Source 2D/3D image (real or complex float/double)
 * @dst: Destination image with dimensions `(nx / bx, ny / by, nz / bz)`
 * @bx:  X binning factor (`>= 1`)
 * @by:  Y binning factor (`>= 1`)
 * @bz:  Z binning factor (`>= 1`)
 *
 * Return: 0 on success, -1 on error.
 */
int arith_image_bin3d_im(const IMAGE *src, IMAGE *dst, int bx, int by, int bz)
{
    if (src == NULL || dst == NULL || src->md == NULL || dst->md == NULL || bx <= 0 || by <= 0 ||
        bz <= 0)
    {
        return -1;
    }
    const int64_t snx = (int64_t) src->md[0].size[0];
    const int64_t sny =
        (src->md[0].naxis > 1 && src->md[0].size[1] > 0) ? (int64_t) src->md[0].size[1] : 1;
    const int64_t dnx = (int64_t) dst->md[0].size[0];
    const int64_t dny =
        (dst->md[0].naxis > 1 && dst->md[0].size[1] > 0) ? (int64_t) dst->md[0].size[1] : 1;
    const int64_t dnz =
        (dst->md[0].naxis > 2 && dst->md[0].size[2] > 0) ? (int64_t) dst->md[0].size[2] : 1;
    const uint8_t  dt  = src->md[0].datatype;
    const uint64_t nel = (uint64_t) snx * (uint64_t) sny * (uint64_t) (dnz * bz);
    const int      nth = milk_omp_team_size(nel, ARITH_SPATIAL_OMP_MIN);

    if (dt == _DATATYPE_FLOAT && dst->md[0].datatype == _DATATYPE_FLOAT)
    {
        const float *MILK_RESTRICT pin  = MILK_ASSUME_ALIGNED(src->array.F);
        float *MILK_RESTRICT       pout = MILK_ASSUME_ALIGNED(dst->array.F);
#ifdef _OPENMP
#    pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
#endif
        for (int64_t kk = 0; kk < dnz; kk++)
        {
            for (int64_t jj = 0; jj < dny; jj++)
            {
                float *orow = pout + (kk * dny + jj) * dnx;
                memset(orow, 0, (size_t) dnx * sizeof(float));
                for (int k = 0; k < bz; k++)
                {
                    const float *splane = pin + (kk * bz + k) * snx * sny;
                    for (int j = 0; j < by; j++)
                    {
                        const float *srow = splane + (jj * by + j) * snx;
                        for (int64_t ii = 0; ii < dnx; ii++)
                        {
                            const float *sp  = srow + ii * bx;
                            float        sum = 0.0f;
#pragma omp simd reduction(+ : sum)
                            for (int i = 0; i < bx; i++)
                            {
                                sum += sp[i];
                            }
                            orow[ii] += sum;
                        }
                    }
                }
            }
        }
        return 0;
    }

    if (dt == _DATATYPE_DOUBLE && dst->md[0].datatype == _DATATYPE_DOUBLE)
    {
        const double *MILK_RESTRICT pin  = MILK_ASSUME_ALIGNED(src->array.D);
        double *MILK_RESTRICT       pout = MILK_ASSUME_ALIGNED(dst->array.D);
#ifdef _OPENMP
#    pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
#endif
        for (int64_t kk = 0; kk < dnz; kk++)
        {
            for (int64_t jj = 0; jj < dny; jj++)
            {
                double *orow = pout + (kk * dny + jj) * dnx;
                memset(orow, 0, (size_t) dnx * sizeof(double));
                for (int k = 0; k < bz; k++)
                {
                    const double *splane = pin + (kk * bz + k) * snx * sny;
                    for (int j = 0; j < by; j++)
                    {
                        const double *srow = splane + (jj * by + j) * snx;
                        for (int64_t ii = 0; ii < dnx; ii++)
                        {
                            const double *sp  = srow + ii * bx;
                            double        sum = 0.0;
#pragma omp simd reduction(+ : sum)
                            for (int i = 0; i < bx; i++)
                            {
                                sum += sp[i];
                            }
                            orow[ii] += sum;
                        }
                    }
                }
            }
        }
        return 0;
    }

    if (dt == _DATATYPE_COMPLEX_FLOAT && dst->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const complex_float *MILK_RESTRICT pin  = MILK_ASSUME_ALIGNED(src->array.CF);
        complex_float *MILK_RESTRICT       pout = MILK_ASSUME_ALIGNED(dst->array.CF);
#ifdef _OPENMP
#    pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
#endif
        for (int64_t kk = 0; kk < dnz; kk++)
        {
            for (int64_t jj = 0; jj < dny; jj++)
            {
                complex_float *orow = pout + (kk * dny + jj) * dnx;
                memset(orow, 0, (size_t) dnx * sizeof(complex_float));
                for (int k = 0; k < bz; k++)
                {
                    const complex_float *splane = pin + (kk * bz + k) * snx * sny;
                    for (int j = 0; j < by; j++)
                    {
                        const complex_float *srow = splane + (jj * by + j) * snx;
                        for (int64_t ii = 0; ii < dnx; ii++)
                        {
                            const complex_float *sp  = srow + ii * bx;
                            float                sre = 0.0f;
                            float                sim = 0.0f;
#pragma omp simd reduction(+ : sre, sim)
                            for (int i = 0; i < bx; i++)
                            {
                                sre += sp[i].re;
                                sim += sp[i].im;
                            }
                            orow[ii].re += sre;
                            orow[ii].im += sim;
                        }
                    }
                }
            }
        }
        return 0;
    }

    if (dt == _DATATYPE_COMPLEX_DOUBLE && dst->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        const complex_double *MILK_RESTRICT pin  = MILK_ASSUME_ALIGNED(src->array.CD);
        complex_double *MILK_RESTRICT       pout = MILK_ASSUME_ALIGNED(dst->array.CD);
#ifdef _OPENMP
#    pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
#endif
        for (int64_t kk = 0; kk < dnz; kk++)
        {
            for (int64_t jj = 0; jj < dny; jj++)
            {
                complex_double *orow = pout + (kk * dny + jj) * dnx;
                memset(orow, 0, (size_t) dnx * sizeof(complex_double));
                for (int k = 0; k < bz; k++)
                {
                    const complex_double *splane = pin + (kk * bz + k) * snx * sny;
                    for (int j = 0; j < by; j++)
                    {
                        const complex_double *srow = splane + (jj * by + j) * snx;
                        for (int64_t ii = 0; ii < dnx; ii++)
                        {
                            const complex_double *sp  = srow + ii * bx;
                            double                sre = 0.0;
                            double                sim = 0.0;
#pragma omp simd reduction(+ : sre, sim)
                            for (int i = 0; i < bx; i++)
                            {
                                sre += sp[i].re;
                                sim += sp[i].im;
                            }
                            orow[ii].re += sre;
                            orow[ii].im += sim;
                        }
                    }
                }
            }
        }
        return 0;
    }

    return -1;
}

/* =========================================================================
 * 5. Spatial Block Kernels (`lut`, `gather`/`pixremap`, `interp2`)
 * ========================================================================= */

/**
 * arith_lut_block_d - Evaluate 1D linear lookup table interpolation on a `double` block
 * @tbl: 1D lookup table image
 * @x0:  Table origin coordinate
 * @dx:  Table step size
 * @len: Number of elements in block
 * @io:  In/out double buffer
 *
 * Return: void
 */
void arith_lut_block_d(const IMAGE *tbl, double x0, double dx, int len, double *restrict io)
{
    const int64_t N      = (int64_t) tbl->md[0].nelement;
    const double  inv_dx = (dx != 0.0) ? (1.0 / dx) : 0.0;
    const double  max_u  = (double) (N - 1);
    uint8_t       dt     = tbl->md[0].datatype;
#define LUT_BODY(MBR)                                              \
    {                                                              \
        const __typeof__(tbl->array.MBR[0]) *src = tbl->array.MBR; \
        for (int j = 0; j < len; j++)                              \
        {                                                          \
            double u = (io[j] - x0) * inv_dx;                      \
            if (arith_isnan_d(u))                                  \
            {                                                      \
                io[j] = NAN;                                       \
            }                                                      \
            else if (u <= 0.0)                                     \
            {                                                      \
                io[j] = (double) src[0];                           \
            }                                                      \
            else if (u >= max_u)                                   \
            {                                                      \
                io[j] = (double) src[N - 1];                       \
            }                                                      \
            else                                                   \
            {                                                      \
                int64_t idx = (int64_t) u;                         \
                if (idx < 0)                                       \
                {                                                  \
                    idx = 0;                                       \
                }                                                  \
                else if (idx > N - 2)                              \
                {                                                  \
                    idx = N - 2;                                   \
                }                                                  \
                double frac = u - (double) idx;                    \
                double v0   = (double) src[idx];                   \
                double v1   = (double) src[idx + 1];               \
                io[j]       = v0 + frac * (v1 - v0);               \
            }                                                      \
        }                                                          \
    }
    MILK_FOR_EACH_DATATYPE(dt, LUT_BODY, LUT_BODY(D))
#undef LUT_BODY
}

/**
 * arith_lut_block_f - Evaluate 1D linear lookup table interpolation on a `float` block
 * @tbl: 1D lookup table image
 * @x0:  Table origin coordinate
 * @dx:  Table step size
 * @len: Number of elements in block
 * @io:  In/out float buffer
 *
 * Return: void
 */
void arith_lut_block_f(const IMAGE *tbl, double x0, double dx, int len, float *restrict io)
{
    const int64_t N      = (int64_t) tbl->md[0].nelement;
    const float   x0_f   = (float) x0;
    const float   inv_dx = (dx != 0.0) ? (float) (1.0 / dx) : 0.0f;
    const float   max_u  = (float) (N - 1);
    const float  *src    = tbl->array.F;
    for (int j = 0; j < len; j++)
    {
        float u = (io[j] - x0_f) * inv_dx;
        if (arith_isnan_f(u))
        {
            io[j] = NAN;
        }
        else if (u <= 0.0f)
        {
            io[j] = src[0];
        }
        else if (u >= max_u)
        {
            io[j] = src[N - 1];
        }
        else
        {
            int64_t idx = (int64_t) u;
            if (idx < 0)
            {
                idx = 0;
            }
            else if (idx > N - 2)
            {
                idx = N - 2;
            }
            float frac = u - (float) idx;
            float v0   = src[idx];
            float v1   = src[idx + 1];
            io[j]      = v0 + frac * (v1 - v0);
        }
    }
}

/**
 * arith_gather_block_d - Evaluate 1D index gather `pixremap` on a `double` block
 * @im:   Source lookup image
 * @fill: Out-of-range fill value
 * @len:  Number of elements in block
 * @io:   In/out double buffer (input: indices, output: gathered samples)
 *
 * Return: void
 */
void arith_gather_block_d(const IMAGE *im, double fill, int len, double *restrict io)
{
    const int64_t N     = (int64_t) im->md[0].nelement;
    const double  max_n = (double) N;
    uint8_t       dt    = im->md[0].datatype;
#define GATHER_BODY(MBR)                                                        \
    {                                                                           \
        const __typeof__(im->array.MBR[0]) *src = im->array.MBR;                \
        for (int j = 0; j < len; j++)                                           \
        {                                                                       \
            double m = io[j];                                                   \
            if (!arith_isnan_d(m) && m >= 0.0 && m < max_n)                     \
            {                                                                   \
                int64_t idx = (int64_t) m;                                      \
                io[j]       = (idx >= 0 && idx < N) ? (double) src[idx] : fill; \
            }                                                                   \
            else                                                                \
            {                                                                   \
                io[j] = fill;                                                   \
            }                                                                   \
        }                                                                       \
    }
    MILK_FOR_EACH_DATATYPE(dt, GATHER_BODY, GATHER_BODY(D))
#undef GATHER_BODY
}

/**
 * arith_gather_block_f - Evaluate 1D index gather `pixremap` on a `float` block
 * @im:   Source lookup image
 * @fill: Out-of-range fill value
 * @len:  Number of elements in block
 * @io:   In/out float buffer (input: indices, output: gathered samples)
 *
 * Return: void
 */
void arith_gather_block_f(const IMAGE *im, double fill, int len, float *restrict io)
{
    const int64_t N      = (int64_t) im->md[0].nelement;
    const float   max_n  = (float) N;
    const float   fill_f = (float) fill;
    const float  *src    = im->array.F;
    for (int j = 0; j < len; j++)
    {
        float m = io[j];
        if (!arith_isnan_f(m) && m >= 0.0f && m < max_n)
        {
            int64_t idx = (int64_t) m;
            io[j]       = (idx >= 0 && idx < N) ? src[idx] : fill_f;
        }
        else
        {
            io[j] = fill_f;
        }
    }
}

/**
 * arith_interp2_block_d - Evaluate 2D bilinear sampling on a `double` block
 * @im:  Source 2D image
 * @len: Number of elements in block
 * @xio: In/out double buffer (input: X coordinates, output: interpolated samples)
 * @ym:  Input double buffer of Y coordinates
 *
 * Return: void
 */
void arith_interp2_block_d(const IMAGE *im,
                           int          len,
                           double *restrict xio,
                           const double *restrict ym)
{
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const double  max_x  = (double) (nx - 1);
    const double  max_y  = (double) (ny - 1);
    const int64_t max_ix = (nx > 1) ? (nx - 2) : 0;
    const int64_t max_iy = (ny > 1) ? (ny - 2) : 0;
    const int64_t dx1    = (nx > 1) ? 1 : 0;
    const int64_t dy1    = (ny > 1) ? nx : 0;
    uint8_t       dt     = im->md[0].datatype;
#define INTERP2_BODY(MBR)                                                \
    {                                                                    \
        const __typeof__(im->array.MBR[0]) *src = im->array.MBR;         \
        for (int j = 0; j < len; j++)                                    \
        {                                                                \
            double x = xio[j], y = ym[j];                                \
            if (arith_isnan_d(x) || arith_isnan_d(y))                    \
            {                                                            \
                xio[j] = NAN;                                            \
                continue;                                                \
            }                                                            \
            x            = (x < 0.0) ? 0.0 : ((x > max_x) ? max_x : x);  \
            y            = (y < 0.0) ? 0.0 : ((y > max_y) ? max_y : y);  \
            int64_t ix   = (int64_t) x;                                  \
            int64_t iy   = (int64_t) y;                                  \
            ix           = (ix < 0) ? 0 : ((ix > max_ix) ? max_ix : ix); \
            iy           = (iy < 0) ? 0 : ((iy > max_iy) ? max_iy : iy); \
            double  fx   = x - (double) ix;                              \
            double  fy   = y - (double) iy;                              \
            int64_t base = iy * nx + ix;                                 \
            double  v00  = (double) src[base];                           \
            double  v10  = (double) src[base + dx1];                     \
            double  v01  = (double) src[base + dy1];                     \
            double  v11  = (double) src[base + dy1 + dx1];               \
            double  r0   = v00 + fx * (v10 - v00);                       \
            double  r1   = v01 + fx * (v11 - v01);                       \
            xio[j]       = r0 + fy * (r1 - r0);                          \
        }                                                                \
    }
    MILK_FOR_EACH_DATATYPE(dt, INTERP2_BODY, INTERP2_BODY(D))
#undef INTERP2_BODY
}

/**
 * arith_interp2_block_f - Evaluate 2D bilinear sampling on a `float` block
 * @im:  Source 2D image
 * @len: Number of elements in block
 * @xio: In/out float buffer (input: X coordinates, output: interpolated samples)
 * @ym:  Input float buffer of Y coordinates
 *
 * Return: void
 */
void arith_interp2_block_f(const IMAGE *im, int len, float *restrict xio, const float *restrict ym)
{
    const int64_t nx = (int64_t) im->md[0].size[0];
    const int64_t ny =
        (im->md[0].naxis > 1 && im->md[0].size[1] > 0) ? (int64_t) im->md[0].size[1] : 1;
    const float   max_x  = (float) (nx - 1);
    const float   max_y  = (float) (ny - 1);
    const int64_t max_ix = (nx > 1) ? (nx - 2) : 0;
    const int64_t max_iy = (ny > 1) ? (ny - 2) : 0;
    const int64_t dx1    = (nx > 1) ? 1 : 0;
    const int64_t dy1    = (ny > 1) ? nx : 0;
    const float  *src    = im->array.F;
    for (int j = 0; j < len; j++)
    {
        float x = xio[j], y = ym[j];
        if (arith_isnan_f(x) || arith_isnan_f(y))
        {
            xio[j] = NAN;
            continue;
        }
        x            = (x < 0.0f) ? 0.0f : ((x > max_x) ? max_x : x);
        y            = (y < 0.0f) ? 0.0f : ((y > max_y) ? max_y : y);
        int64_t ix   = (int64_t) x;
        int64_t iy   = (int64_t) y;
        ix           = (ix < 0) ? 0 : ((ix > max_ix) ? max_ix : ix);
        iy           = (iy < 0) ? 0 : ((iy > max_iy) ? max_iy : iy);
        float   fx   = x - (float) ix;
        float   fy   = y - (float) iy;
        int64_t base = iy * nx + ix;
        float   v00  = src[base];
        float   v10  = src[base + dx1];
        float   v01  = src[base + dy1];
        float   v11  = src[base + dy1 + dx1];
        float   r0   = v00 + fx * (v10 - v00);
        float   r1   = v01 + fx * (v11 - v01);
        xio[j]       = r0 + fy * (r1 - r0);
    }
}

/* =========================================================================
 * 6. 2D/3D Expansion, Resize, Rotation, Circular Symmetry, and 1-Pixel Blur
 * ========================================================================= */

/**
 * arith_image_expand3d_im - Expand @src by integer factors (@bx, @by, @bz) into @dst
 * @src: Input 2D/3D image (`FLOAT`, `DOUBLE`, `COMPLEX_FLOAT`, or `COMPLEX_DOUBLE`)
 * @dst: Output 2D/3D image (`size[0] >= src->size[0] * bx`, etc., matching datatype)
 * @bx:  X expansion factor (>= 1)
 * @by:  Y expansion factor (>= 1)
 * @bz:  Z expansion factor (>= 1)
 *
 * Return: 0 on success, -1 on invalid arguments.
 */
int arith_image_expand3d_im(const IMAGE *src, IMAGE *dst, int bx, int by, int bz)
{
    if (src == NULL || dst == NULL || bx < 1 || by < 1 || bz < 1)
    {
        return -1;
    }
    uint8_t dt  = src->md[0].datatype;
    int64_t s_x = (int64_t) src->md[0].size[0];
    int64_t s_y = (src->md[0].naxis > 1) ? (int64_t) src->md[0].size[1] : 1;
    int64_t s_z = (src->md[0].naxis > 2) ? (int64_t) src->md[0].size[2] : 1;
    int64_t o_x = (int64_t) dst->md[0].size[0];
    int64_t o_y = (dst->md[0].naxis > 1) ? (int64_t) dst->md[0].size[1] : 1;
    int64_t o_z = (dst->md[0].naxis > 2) ? (int64_t) dst->md[0].size[2] : 1;

    if (dst->md[0].datatype != dt || o_x < s_x * bx || o_y < s_y * by || o_z < s_z * bz)
    {
        return -1;
    }

    size_t el_bytes = 0;
    int    ncomp    = 1;
    if (dt == _DATATYPE_FLOAT)
    {
        el_bytes = sizeof(float);
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        el_bytes = sizeof(double);
    }
    else if (dt == _DATATYPE_COMPLEX_FLOAT)
    {
        el_bytes = 2 * sizeof(float);
        ncomp    = 2;
    }
    else if (dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        el_bytes = 2 * sizeof(double);
        ncomp    = 2;
    }
    else
    {
        return -1;
    }

    uint64_t total_out = (uint64_t) o_x * (uint64_t) o_y * (uint64_t) o_z;
    int      nth       = milk_omp_team_size(total_out, ARITH_SPATIAL_OMP_MIN);
    size_t   row_bytes = (size_t) (s_x * bx) * el_bytes;

#pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
    for (int64_t iz = 0; iz < s_z; iz++)
    {
        for (int64_t iy = 0; iy < s_y; iy++)
        {
            int64_t oz0 = iz * bz;
            int64_t oy0 = iy * by;
            if (dt == _DATATYPE_FLOAT || dt == _DATATYPE_COMPLEX_FLOAT)
            {
                const float *srow  = src->array.F + ((iz * s_y + iy) * s_x) * ncomp;
                float       *drow0 = dst->array.F + ((oz0 * o_y + oy0) * o_x) * ncomp;
                if (ncomp == 1)
                {
                    for (int64_t ix = 0; ix < s_x; ix++)
                    {
                        float  v = srow[ix];
                        float *d = drow0 + ix * bx;
#pragma omp simd
                        for (int dx = 0; dx < bx; dx++)
                        {
                            d[dx] = v;
                        }
                    }
                }
                else
                {
                    for (int64_t ix = 0; ix < s_x; ix++)
                    {
                        float  vr = srow[2 * ix];
                        float  vi = srow[2 * ix + 1];
                        float *d  = drow0 + 2 * ix * bx;
#pragma omp simd
                        for (int dx = 0; dx < bx; dx++)
                        {
                            d[2 * dx]     = vr;
                            d[2 * dx + 1] = vi;
                        }
                    }
                }
                for (int dy = 1; dy < by; dy++)
                {
                    float *drow = dst->array.F + ((oz0 * o_y + oy0 + dy) * o_x) * ncomp;
                    memcpy(drow, drow0, row_bytes);
                }
            }
            else
            {
                const double *srow  = src->array.D + ((iz * s_y + iy) * s_x) * ncomp;
                double       *drow0 = dst->array.D + ((oz0 * o_y + oy0) * o_x) * ncomp;
                if (ncomp == 1)
                {
                    for (int64_t ix = 0; ix < s_x; ix++)
                    {
                        double  v = srow[ix];
                        double *d = drow0 + ix * bx;
#pragma omp simd
                        for (int dx = 0; dx < bx; dx++)
                        {
                            d[dx] = v;
                        }
                    }
                }
                else
                {
                    for (int64_t ix = 0; ix < s_x; ix++)
                    {
                        double  vr = srow[2 * ix];
                        double  vi = srow[2 * ix + 1];
                        double *d  = drow0 + 2 * ix * bx;
#pragma omp simd
                        for (int dx = 0; dx < bx; dx++)
                        {
                            d[2 * dx]     = vr;
                            d[2 * dx + 1] = vi;
                        }
                    }
                }
                for (int dy = 1; dy < by; dy++)
                {
                    double *drow = dst->array.D + ((oz0 * o_y + oy0 + dy) * o_x) * ncomp;
                    memcpy(drow, drow0, row_bytes);
                }
            }
        }
    }

    if (bz > 1)
    {
        size_t slice_bytes = (size_t) o_x * (size_t) o_y * el_bytes;
        char  *base        = (char *) dst->array.raw;
        for (int64_t iz = 0; iz < s_z; iz++)
        {
            const char *s0 = base + (size_t) (iz * bz) * slice_bytes;
            for (int dz = 1; dz < bz; dz++)
            {
                memcpy(base + (size_t) (iz * bz + dz) * slice_bytes, s0, slice_bytes);
            }
        }
    }
    return 0;
}

/**
 * arith_image_resize2d_im - Bilinear 2D resize of @src into @dst
 * @src: Input 2D `FLOAT` or `DOUBLE` image
 * @dst: Output 2D `FLOAT` or `DOUBLE` image
 *
 * Return: 0 on success, -1 on unsupported datatype.
 */
int arith_image_resize2d_im(const IMAGE *src, IMAGE *dst)
{
    if (src == NULL || dst == NULL || src->md[0].datatype != dst->md[0].datatype)
    {
        return -1;
    }
    uint8_t  dt     = src->md[0].datatype;
    uint32_t nx_in  = src->md[0].size[0];
    uint32_t ny_in  = (src->md[0].naxis > 1) ? src->md[0].size[1] : 1;
    uint32_t nx_out = dst->md[0].size[0];
    uint32_t ny_out = (dst->md[0].naxis > 1) ? dst->md[0].size[1] : 1;
    if (nx_in == 0 || ny_in == 0 || nx_out == 0 || ny_out == 0)
    {
        return -1;
    }

    int nth = milk_omp_team_size((uint64_t) nx_out * (uint64_t) ny_out, ARITH_SPATIAL_OMP_MIN);

    if (dt == _DATATYPE_FLOAT)
    {
        const float *restrict in_p = src->array.F;
        float *restrict out_p      = dst->array.F;
        const float sx             = (float) nx_in / (float) nx_out;
        const float sy             = (float) ny_in / (float) ny_out;

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint32_t jj = 0; jj < ny_out; jj++)
        {
            float  yf1  = (float) jj * sy;
            long   jj1  = (long) yf1;
            float  tf   = yf1 - (float) jj1;
            float *drow = out_p + (size_t) jj * nx_out;
            if (jj1 < 0 || jj1 + 1 >= (long) ny_in)
            {
                memset(drow, 0, (size_t) nx_out * sizeof(float));
                continue;
            }
            const float *r0   = in_p + (size_t) jj1 * nx_in;
            const float *r1   = r0 + nx_in;
            const float  wy0  = 1.0f - tf;
            const float  wy1  = tf;
            const long   limx = (long) nx_in - 1;

#pragma omp simd
            for (uint32_t ii = 0; ii < nx_out; ii++)
            {
                float xf1 = (float) ii * sx;
                long  ii1 = (long) xf1;
                float uf  = xf1 - (float) ii1;
                if (ii1 >= 0 && ii1 < limx)
                {
                    float v0 = r0[ii1] * (1.0f - uf) + r0[ii1 + 1] * uf;
                    float v1 = r1[ii1] * (1.0f - uf) + r1[ii1 + 1] * uf;
                    drow[ii] = v0 * wy0 + v1 * wy1;
                }
                else
                {
                    drow[ii] = 0.0f;
                }
            }
        }
        return 0;
    }
    if (dt == _DATATYPE_DOUBLE)
    {
        const double *restrict in_p = src->array.D;
        double *restrict out_p      = dst->array.D;
        const double sx             = (double) nx_in / (double) nx_out;
        const double sy             = (double) ny_in / (double) ny_out;

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint32_t jj = 0; jj < ny_out; jj++)
        {
            double  yf1  = (double) jj * sy;
            long    jj1  = (long) yf1;
            double  tf   = yf1 - (double) jj1;
            double *drow = out_p + (size_t) jj * nx_out;
            if (jj1 < 0 || jj1 + 1 >= (long) ny_in)
            {
                memset(drow, 0, (size_t) nx_out * sizeof(double));
                continue;
            }
            const double *r0   = in_p + (size_t) jj1 * nx_in;
            const double *r1   = r0 + nx_in;
            const double  wy0  = 1.0 - tf;
            const double  wy1  = tf;
            const long    limx = (long) nx_in - 1;

#pragma omp simd
            for (uint32_t ii = 0; ii < nx_out; ii++)
            {
                double xf1 = (double) ii * sx;
                long   ii1 = (long) xf1;
                double uf  = xf1 - (double) ii1;
                if (ii1 >= 0 && ii1 < limx)
                {
                    double v0 = r0[ii1] * (1.0 - uf) + r0[ii1 + 1] * uf;
                    double v1 = r1[ii1] * (1.0 - uf) + r1[ii1 + 1] * uf;
                    drow[ii]  = v0 * wy0 + v1 * wy1;
                }
                else
                {
                    drow[ii] = 0.0;
                }
            }
        }
        return 0;
    }
    return -1;
}

/**
 * arith_image_rotate2d_im - Rotate 2D image @src by @angle radians around `(nx/2, ny/2)` into @dst
 * @src:   Input 2D `FLOAT` or `DOUBLE` image
 * @dst:   Output 2D image of matching dimensions and datatype
 * @angle: Rotation angle in radians
 *
 * Return: 0 on success, -1 on unsupported datatype or mismatched dimensions.
 */
int arith_image_rotate2d_im(const IMAGE *src, IMAGE *dst, double angle)
{
    if (src == NULL || dst == NULL || src->md[0].datatype != dst->md[0].datatype)
    {
        return -1;
    }
    uint8_t  dt = src->md[0].datatype;
    uint32_t nx = src->md[0].size[0];
    uint32_t ny = (src->md[0].naxis > 1) ? src->md[0].size[1] : 1;
    if (nx == 0 || ny == 0 || dst->md[0].size[0] != nx ||
        ((dst->md[0].naxis > 1 ? dst->md[0].size[1] : 1) != ny))
    {
        return -1;
    }

    int nth = milk_omp_team_size((uint64_t) nx * (uint64_t) ny, ARITH_SPATIAL_OMP_MIN);

    if (dt == _DATATYPE_FLOAT)
    {
        const float *restrict in_p = src->array.F;
        float *restrict out_p      = dst->array.F;
        const float c              = cosf((float) angle);
        const float s              = sinf((float) angle);
        const float cx             = (float) (nx / 2);
        const float cy             = (float) (ny / 2);
        const float limx           = (float) nx;
        const float limy           = (float) ny;
        const long  max_x          = (long) nx - 1;
        const long  max_y          = (long) ny - 1;

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint32_t jj = 0; jj < ny; jj++)
        {
            const float dy   = (float) jj - cy;
            const float bx0  = cx + dy * s;
            const float by0  = cy + dy * c;
            float      *drow = out_p + (size_t) jj * nx;

#pragma omp simd
            for (uint32_t ii = 0; ii < nx; ii++)
            {
                float dx = (float) ii - cx;
                float xs = bx0 + dx * c;
                float ys = by0 - dx * s;
                if (xs >= 0.0f && ys >= 0.0f && xs < limx && ys < limy)
                {
                    long  ix0 = (long) xs;
                    long  iy0 = (long) ys;
                    long  ix1 = (ix0 < max_x) ? (ix0 + 1) : max_x;
                    long  iy1 = (iy0 < max_y) ? (iy0 + 1) : max_y;
                    float fx  = xs - (float) ix0;
                    float fy  = ys - (float) iy0;
                    float v00 = in_p[(size_t) iy0 * nx + (size_t) ix0];
                    float v10 = in_p[(size_t) iy0 * nx + (size_t) ix1];
                    float v01 = in_p[(size_t) iy1 * nx + (size_t) ix0];
                    float v11 = in_p[(size_t) iy1 * nx + (size_t) ix1];
                    float r0  = v00 + fx * (v10 - v00);
                    float r1  = v01 + fx * (v11 - v01);
                    drow[ii]  = r0 + fy * (r1 - r0);
                }
                else
                {
                    drow[ii] = 0.0f;
                }
            }
        }
        return 0;
    }
    if (dt == _DATATYPE_DOUBLE)
    {
        const double *restrict in_p = src->array.D;
        double *restrict out_p      = dst->array.D;
        const double c              = cos(angle);
        const double s              = sin(angle);
        const double cx             = (double) (nx / 2);
        const double cy             = (double) (ny / 2);
        const double limx           = (double) nx;
        const double limy           = (double) ny;
        const long   max_x          = (long) nx - 1;
        const long   max_y          = (long) ny - 1;

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint32_t jj = 0; jj < ny; jj++)
        {
            const double dy   = (double) jj - cy;
            const double bx0  = cx + dy * s;
            const double by0  = cy + dy * c;
            double      *drow = out_p + (size_t) jj * nx;

#pragma omp simd
            for (uint32_t ii = 0; ii < nx; ii++)
            {
                double dx = (double) ii - cx;
                double xs = bx0 + dx * c;
                double ys = by0 - dx * s;
                if (xs >= 0.0 && ys >= 0.0 && xs < limx && ys < limy)
                {
                    long   ix0 = (long) xs;
                    long   iy0 = (long) ys;
                    long   ix1 = (ix0 < max_x) ? (ix0 + 1) : max_x;
                    long   iy1 = (iy0 < max_y) ? (iy0 + 1) : max_y;
                    double fx  = xs - (double) ix0;
                    double fy  = ys - (double) iy0;
                    double v00 = in_p[(size_t) iy0 * nx + (size_t) ix0];
                    double v10 = in_p[(size_t) iy0 * nx + (size_t) ix1];
                    double v01 = in_p[(size_t) iy1 * nx + (size_t) ix0];
                    double v11 = in_p[(size_t) iy1 * nx + (size_t) ix1];
                    double r0  = v00 + fx * (v10 - v00);
                    double r1  = v01 + fx * (v11 - v01);
                    drow[ii]   = r0 + fy * (r1 - r0);
                }
                else
                {
                    drow[ii] = 0.0;
                }
            }
        }
        return 0;
    }
    return -1;
}

/**
 * arith_image_circsym_im - Extract circularly symmetric (`subtract == 0`) or asymmetric
 *                          (`subtract != 0`) component of 2D image @src around (@cx, @cy)
 * @src:      Input 2D image
 * @dst:      Output 2D `FLOAT` image of matching dimensions
 * @cx:       X center coordinate
 * @cy:       Y center coordinate
 * @subtract: 0 to write the azimuthally averaged profile, non-zero to write `src - profile`
 *
 * Return: 0 on success, -1 on error.
 */
int arith_image_circsym_im(const IMAGE *src, IMAGE *dst, double cx, double cy, int subtract)
{
    if (src == NULL || dst == NULL || dst->md[0].datatype != _DATATYPE_FLOAT)
    {
        return -1;
    }
    uint32_t nx      = src->md[0].size[0];
    uint32_t ny      = (src->md[0].naxis > 1) ? src->md[0].size[1] : 1;
    long     nb_step = (long) (nx / 2);
    if (nx == 0 || ny == 0 || nb_step <= 0 || dst->md[0].size[0] != nx ||
        ((dst->md[0].naxis > 1 ? dst->md[0].size[1] : 1) != ny))
    {
        return -1;
    }

    arith_csr_cache_t csr       = { 0 };
    uint32_t          ishape[3] = { nx, ny, 1 };
    if (arith_csr_build_rprof(&csr, ishape, cx, cy, 1.0) != 0)
    {
        return -1;
    }
    long   alloc_n = (csr.nbins > nb_step) ? (long) csr.nbins : nb_step;
    float *mean    = (float *) calloc((size_t) alloc_n, sizeof(float));
    if (mean == NULL)
    {
        arith_csr_cache_free(&csr);
        return -1;
    }
    IMAGE_METADATA rmd = { 0 };
    IMAGE          rim = { 0 };
    rmd.naxis          = 1;
    rmd.size[0]        = (uint32_t) csr.nbins;
    rmd.nelement       = (uint64_t) csr.nbins;
    rmd.datatype       = _DATATYPE_FLOAT;
    rim.md             = &rmd;
    rim.array.F        = mean;
    arith_csr_gather(src, &csr, ARITH_RED_MEAN, 0, &rim);
    arith_csr_cache_free(&csr);

    float *restrict out_p      = dst->array.F;
    const float *restrict in_p = (src->md[0].datatype == _DATATYPE_FLOAT) ? src->array.F : NULL;
    const float f_cx           = (float) cx;
    const float f_cy           = (float) cy;
    const long  lim            = nb_step - 1;
    int         nth = milk_omp_team_size((uint64_t) nx * (uint64_t) ny, ARITH_SPATIAL_OMP_MIN);

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
    for (uint32_t jj = 0; jj < ny; jj++)
    {
        float        dy2  = ((float) jj - f_cy) * ((float) jj - f_cy);
        float       *drow = out_p + (size_t) jj * nx;
        const float *srow = (in_p != NULL) ? (in_p + (size_t) jj * nx) : NULL;

        if (!subtract)
        {
#pragma omp simd
            for (uint32_t ii = 0; ii < nx; ii++)
            {
                float dx   = (float) ii - f_cx;
                float dist = sqrtf(dx * dx + dy2);
                long  idx  = (long) dist;
                float frac = dist - (float) idx;
                if (idx >= 0 && idx < lim)
                {
                    drow[ii] = (1.0f - frac) * mean[idx] + frac * mean[idx + 1];
                }
                else
                {
                    drow[ii] = 0.0f;
                }
            }
        }
        else if (srow != NULL)
        {
#pragma omp simd
            for (uint32_t ii = 0; ii < nx; ii++)
            {
                float dx   = (float) ii - f_cx;
                float dist = sqrtf(dx * dx + dy2);
                long  idx  = (long) dist;
                float frac = dist - (float) idx;
                if (idx >= 0 && idx < lim)
                {
                    float sym = (1.0f - frac) * mean[idx] + frac * mean[idx + 1];
                    drow[ii]  = srow[ii] - sym;
                }
                else
                {
                    drow[ii] = 0.0f;
                }
            }
        }
        else
        {
            for (uint32_t ii = 0; ii < nx; ii++)
            {
                float dx   = (float) ii - f_cx;
                float dist = sqrtf(dx * dx + dy2);
                long  idx  = (long) dist;
                float frac = dist - (float) idx;
                if (idx >= 0 && idx < lim)
                {
                    float sym = (1.0f - frac) * mean[idx] + frac * mean[idx + 1];
                    float sv  = 0.0f;
                    arith_load_block_f_inline(src, (uint64_t) jj * nx + ii, 1, &sv);
                    drow[ii] = sv - sym;
                }
                else
                {
                    drow[ii] = 0.0f;
                }
            }
        }
    }

    free(mean);
    return 0;
}

/**
 * arith_image_bblur1pix_im - Apply @niter iterations of 1-pixel 3x3 box-corner blur to @src
 * @src:   Input 2D image (any real numeric datatype)
 * @dst:   Output 2D `FLOAT` image of matching dimensions
 * @amp:   Side-pixel weight (`coeff1 = amp`, `coeff2 = amp^2`, `coeff0 = 1 - 4*(coeff1 + coeff2)`)
 * @niter: Number of blur iterations
 *
 * Return: 0 on success, -1 on error.
 */
int arith_image_bblur1pix_im(const IMAGE *src, IMAGE *dst, double amp, int niter)
{
    if (src == NULL || dst == NULL || dst->md[0].datatype != _DATATYPE_FLOAT)
    {
        return -1;
    }
    uint32_t xsize = src->md[0].size[0];
    uint32_t ysize = (src->md[0].naxis > 1) ? src->md[0].size[1] : 1;
    uint64_t nel   = (uint64_t) xsize * (uint64_t) ysize;
    if (xsize == 0 || ysize == 0 || dst->md[0].size[0] != xsize ||
        ((dst->md[0].naxis > 1 ? dst->md[0].size[1] : 1) != ysize))
    {
        return -1;
    }

    float coeff1 = (float) amp;
    float coeff2 = coeff1 * coeff1;
    float coeff0 = 1.0f - 4.0f * (coeff1 + coeff2);

    float *buf0 = (float *) malloc(sizeof(float) * nel);
    float *buf1 = (float *) calloc(nel, sizeof(float));
    if (buf0 == NULL || buf1 == NULL)
    {
        free(buf0);
        free(buf1);
        return -1;
    }

    int nth = milk_omp_team_size(nel, ARITH_SPATIAL_OMP_MIN);
#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
    for (uint64_t off = 0; off < nel; off += ARITH_ELEM_BLK)
    {
        uint64_t len = (nel - off < ARITH_ELEM_BLK) ? (nel - off) : ARITH_ELEM_BLK;
        arith_load_block_f_inline(src, off, len, buf0 + off);
    }

    float *cur = buf0;
    float *nxt = buf1;
    for (int iter = 0; iter < niter && xsize >= 3 && ysize >= 3; iter++)
    {
        /* Zero border rows and columns of nxt on first pass (or when iter == 0 / iter == 1) */
        if (iter < 2)
        {
            memset(nxt, 0, (size_t) xsize * sizeof(float));
            memset(nxt + (size_t) (ysize - 1) * xsize, 0, (size_t) xsize * sizeof(float));
            for (uint32_t jj = 1; jj < ysize - 1; jj++)
            {
                nxt[(size_t) jj * xsize]             = 0.0f;
                nxt[(size_t) jj * xsize + xsize - 1] = 0.0f;
            }
        }

#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint32_t jj = 1; jj < ysize - 1; jj++)
        {
            const float *restrict r_prev = cur + (size_t) (jj - 1) * xsize;
            const float *restrict r_curr = cur + (size_t) jj * xsize;
            const float *restrict r_next = cur + (size_t) (jj + 1) * xsize;
            float *restrict r_out        = nxt + (size_t) jj * xsize;

#pragma omp simd
            for (uint32_t ii = 1; ii < xsize - 1; ii++)
            {
                float sides   = r_curr[ii + 1] + r_curr[ii - 1] + r_next[ii] + r_prev[ii];
                float corners = r_next[ii + 1] + r_next[ii - 1] + r_prev[ii + 1] + r_prev[ii - 1];
                r_out[ii]     = coeff0 * r_curr[ii] + coeff1 * sides + coeff2 * corners;
            }
        }
        float *tmp = cur;
        cur        = nxt;
        nxt        = tmp;
    }

    memcpy(dst->array.F, cur, sizeof(float) * nel);
    free(buf0);
    free(buf1);
    return 0;
}
