// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_prestage.c
 * @brief Non-elementwise pre-stage and axis-reduction constructors for `libmilkcalc`.
 *
 * Implements the `MC_STEP_CTOR` operations dispatched via `mc_exec_prestage()` when an expression
 * either stands alone as a constructor statement or is hoisted into a temporary slot (`_tmpcalcN`)
 * ahead of an elementwise pass:
 *   - **Temporal exponential moving average (`ema(im, alpha)`)**: Maintains a per-step double
 *     accumulator buffer in `cs->scratch_buf` across frames (`milk.calc` unit iterations).
 *   - **Prefix sum (`cumsum(im[, axis])`)**: Copies and promotes the input image to float/double
 *     and computes an in-place prefix sum along the requested axis.
 *   - **2D separable Gaussian blur (`gauss(im, sx[, sy])`)**: Delegates to
 *     `arith_image_gauss2d_im` with persistent scratch storage in `cs->scratch_buf`.
 *   - **2D convolution and box mean (`conv2`, `boxmean`)**: Dispatches to
 *     `mc_exec_conv2_or_boxmean`.
 *   - **Radial profile (`rprof(im[, cx, cy])`) and bin-map average (`pixunmap(im, map[, n])`)**:
 *     Uses a cached Compressed Sparse Row (CSR) gather schedule (`arith_csr_cache_t` in `cs->csr`)
 *     when the input and map are materialized 2D images, or falls back to a multi-threaded L1
 *     tile-streamed scatter-accumulate over `mc_red_src_t`.
 *   - **Per-axis reductions (`itot`, `imean`, `imin`, `imax`, `istd`, `irms`, `ivar`, `norm`,
 *     `asum`, `icount`, `imedian` with an `axis` argument)**: Collapses one dimension (or keeps a
 *     size-1 broadcast axis when hoisted inside an elementwise expression, `cs->is_anon == 2`)
 *     using SIMD horizontal lane vectorization for `axis > 0`.
 */

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/arith_team.h"
#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/variable_ID.h"
#include "linalg/mc_linalg.h"
#include "mc_plan.h"

/* Block size (in elements) for axis reductions and L1-streamed segmented scatter. */
#define MC_RED_BLK 1024

/* ============================================================================================== */
/* Diagnostics and Step Resource Cleanup                                                          */
/* ============================================================================================== */

/**
 * pre_err - Format a diagnostic message into @err if not already populated
 * @err:   Destination error buffer (may be NULL)
 * @errsz: Capacity of @err in bytes
 * @fmt:   printf-style format string
 *
 * Return: Always -1 for convenient tail-return error propagation
 */
__attribute__((format(printf, 3, 4))) static int pre_err(char       *err,
                                                         size_t      errsz,
                                                         const char *fmt,
                                                         ...)
{
    if (err != NULL && errsz > 0 && err[0] == '\0')
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errsz, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/**
 * mc_ctor_step_free - Release cached CSR tables, extension states, and scratch buffers on @cs
 * @cs: Constructor step descriptor to clean up (may be NULL)
 *
 * Called when tearing down or re-planning an `mcalc_prog_t`.
 *
 * Return: void
 */
void mc_ctor_step_free(mc_ctor_step_t *cs)
{
    if (cs == NULL)
    {
        return;
    }
    if (cs->csr != NULL)
    {
        arith_csr_cache_free((arith_csr_cache_t *) cs->csr);
        free(cs->csr);
        cs->csr = NULL;
    }
    if (cs->ext_state != NULL && cs->ext_state_free != NULL)
    {
        cs->ext_state_free(cs->ext_state);
        cs->ext_state = NULL;
    }
    free(cs->scratch_buf);
    cs->scratch_buf = NULL;
    cs->scratch_cap = 0;
}

/* ============================================================================================== */
/* Parallel Per-Axis Reduction Engine                                                             */
/* ============================================================================================== */

/**
 * struct ax_team_job_t - Shared context for parallel per-axis reduction across `arith_team`
 * @im:       Source image being reduced along one axis
 * @oim:      Destination image receiving reduced values
 * @inner:    Product of dimensions below the reduced axis (`stride` between adjacent axis samples)
 * @n:        Extent of the reduced axis
 * @op:       Reduction operation (`ARITH_RED_*`)
 * @fast_vec: 1 to use contiguous horizontal SIMD chunking across `inner` (`axis > 0`), 0 for
 *            per-lane reduction
 */
typedef struct
{
    const IMAGE    *im;
    IMAGE          *oim;
    uint64_t        inner;
    uint64_t        n;
    arith_reduce_op op;
    int             fast_vec;
} ax_team_job_t;

/**
 * ax_team_cb - Worker callback executing per-axis reduction work items `[lo, hi)`
 * @ctx: Pointer to `ax_team_job_t`
 * @tid: Worker thread index (unused)
 * @nth: Total number of worker threads (unused)
 * @lo:  Starting work item index (inclusive)
 * @hi:  Ending work item index (exclusive)
 *
 * Two execution modes are supported:
 *   - `fast_vec == 0`: each work item `idx` is a single output pixel `(o, i)`. When `inner == 1`
 *     (`axis == 0`), the reduction lane is contiguous in memory and is accumulated via
 *     `arith_stat_block_{f,d}_inline`; otherwise `arith_image_reduce_lane` walks with stride
 *     `inner`.
 *   - `fast_vec == 1`: when `axis > 0` and the operation is `SUM`, `MEAN`, `NORM`, or `ASUM` on
 *     float/double data, each work item `c` is a 1024-wide horizontal strip along `inner`,
 *     enabling `#pragma omp simd` vectorization across adjacent columns while sweeping `k = 0..n`.
 *
 * Return: void
 */
static void ax_team_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) tid;
    (void) nth;
    const ax_team_job_t *j     = (const ax_team_job_t *) ctx;
    uint8_t              dt    = j->im->md[0].datatype;
    int                  out_d = (j->oim->md[0].datatype == _DATATYPE_DOUBLE);
    uint64_t             inner = j->inner;
    uint64_t             n     = j->n;

    if (!j->fast_vec)
    {
        for (uint64_t idx = lo; idx < hi; idx++)
        {
            uint64_t o    = idx / inner;
            uint64_t i    = idx % inner;
            uint64_t base = o * n * inner + i;
            double   r;
            if (inner == 1 && (dt == _DATATYPE_FLOAT || dt == _DATATYPE_DOUBLE))
            {
                arith_stat_acc_t acc = { 0, 0.0, 0.0, 0.0, NAN, NAN };
                for (uint64_t off = 0; off < n; off += MC_RED_BLK)
                {
                    int len = (int) ((n - off < MC_RED_BLK) ? (n - off) : MC_RED_BLK);
                    if (dt == _DATATYPE_FLOAT)
                    {
                        arith_stat_block_f_inline(j->op, j->im->array.F + base + off, NULL, len,
                                                  &acc);
                    }
                    else
                    {
                        arith_stat_block_d_inline(j->op, j->im->array.D + base + off, NULL, len,
                                                  &acc);
                    }
                }
                r = arith_stat_finish_inline(&acc, j->op, 0);
            }
            else
            {
                r = arith_image_reduce_lane(j->im, dt, base, n, inner, j->op);
            }
            if (out_d)
            {
                j->oim->array.D[idx] = r;
            }
            else
            {
                j->oim->array.F[idx] = (float) r;
            }
        }
        return;
    }

    /* Fast horizontal SIMD path for axis > 0 sum/mean/norm/asum reductions. */
    uint64_t nchk = (inner + MC_RED_BLK - 1) / MC_RED_BLK;
    for (uint64_t c = lo; c < hi; c++)
    {
        uint64_t o    = c / nchk;
        uint64_t idx0 = (c % nchk) * MC_RED_BLK;
        int      len  = (int) ((inner - idx0 < MC_RED_BLK) ? (inner - idx0) : MC_RED_BLK);
        MILK_ALIGNED(64) double s[MC_RED_BLK] = { 0 };
        for (uint64_t k = 0; k < n; k++)
        {
            uint64_t base = (o * n + k) * inner + idx0;
            if (dt == _DATATYPE_FLOAT)
            {
                const float *p = j->im->array.F + base;
#pragma omp simd
                for (int q = 0; q < len; q++)
                {
                    double x = (double) p[q];
                    s[q] += (j->op == ARITH_RED_ASUM)   ? fabs(x)
                            : (j->op == ARITH_RED_NORM) ? (x * x)
                                                        : x;
                }
            }
            else
            {
                const double *p = j->im->array.D + base;
#pragma omp simd
                for (int q = 0; q < len; q++)
                {
                    double x = p[q];
                    s[q] += (j->op == ARITH_RED_ASUM)   ? fabs(x)
                            : (j->op == ARITH_RED_NORM) ? (x * x)
                                                        : x;
                }
            }
        }
        double inv_n = 1.0 / (double) n;
        for (int q = 0; q < len; q++)
        {
            double r = (j->op == ARITH_RED_MEAN)   ? (s[q] * inv_n)
                       : (j->op == ARITH_RED_NORM) ? sqrt(s[q])
                                                   : s[q];
            if (out_d)
            {
                j->oim->array.D[o * inner + idx0 + (uint64_t) q] = r;
            }
            else
            {
                j->oim->array.F[o * inner + idx0 + (uint64_t) q] = (float) r;
            }
        }
    }
}

/* ============================================================================================== */
/* Segmented Scatter-Accumulate Fallback (`rprof` / `pixunmap` on Compound Sources)               */
/* ============================================================================================== */

/**
 * struct seg_scatter_job_t - Shared context for thread-privatized bin scatter-accumulation
 * @s1:      Value reduction source (`mc_red_src_t`)
 * @s2:      Bin-index reduction source for `pixunmap` (NULL when `pix_bin != NULL`)
 * @pix_bin: Precomputed per-pixel integer bin table for `rprof` (NULL for dynamic map)
 * @nbins:   Number of output bins
 * @nel:     Total number of pixels in @s1
 * @per:     Byte stride per worker thread in @buf (`nbins * (sizeof(double) + sizeof(uint64_t))`)
 * @buf:     Scratch buffer holding per-thread `sum[nbins]` and `cnt[nbins]` arrays
 * @max_n:   Upper bound `(double) nbins` for dynamic map index validation
 */
typedef struct
{
    const mc_red_src_t *s1;
    const mc_red_src_t *s2;
    const int32_t      *pix_bin;
    int                 nbins;
    uint64_t            nel;
    size_t              per;
    void               *buf;
    double              max_n;
} seg_scatter_job_t;

/**
 * seg_scatter_cb - Worker callback accumulating bin sums and counts over tiles `[lo, hi)`
 * @ctx: Pointer to `seg_scatter_job_t`
 * @tid: Worker thread index selecting the private slice in `j->buf`
 * @nth: Total number of worker threads (unused)
 * @lo:  Starting tile index (inclusive)
 * @hi:  Ending tile index (exclusive)
 *
 * Evaluates `s1` (and `s2` when no precomputed `pix_bin` table is available) in 1024-element L1
 * tiles and scatters into thread-private `(sum, cnt)` histograms to avoid atomic contention.
 *
 * Return: void
 */
static void seg_scatter_cb(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    (void) nth;
    const seg_scatter_job_t *j   = (const seg_scatter_job_t *) ctx;
    double                  *sum = (double *) ((char *) j->buf + j->per * (size_t) tid);
    uint64_t                *cnt = (uint64_t *) (sum + j->nbins);
    memset(sum, 0, (size_t) j->nbins * sizeof(double));
    memset(cnt, 0, (size_t) j->nbins * sizeof(uint64_t));

    MILK_ALIGNED(64) double b1[MC_RED_BLK];
    MILK_ALIGNED(64) double b2[MC_RED_BLK];
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t off = b * MC_RED_BLK;
        int      len = (int) ((j->nel - off < MC_RED_BLK) ? (j->nel - off) : MC_RED_BLK);
        mc_eval_src_block_d(j->s1, off, len, b1);
        if (j->pix_bin != NULL)
        {
            const int32_t *pb = j->pix_bin + off;
            for (int q = 0; q < len; q++)
            {
                int bb = pb[q];
                if (bb >= 0)
                {
                    sum[bb] += b1[q];
                    cnt[bb]++;
                }
            }
            continue;
        }
        mc_eval_src_block_d(j->s2, off, len, b2);
        for (int q = 0; q < len; q++)
        {
            double m = b2[q];
            if (!arith_isnan_d(m) && m >= 0.0 && m < j->max_n)
            {
                int bb = (int) (int64_t) m;
                if (bb >= 0 && bb < j->nbins)
                {
                    sum[bb] += b1[q];
                    cnt[bb]++;
                }
            }
        }
    }
}

/* ============================================================================================== */
/* Pre-Stage Constructor Dispatcher                                                               */
/* ============================================================================================== */

/**
 * select_out_dt - Choose output datatype for a pre-stage destination slot
 * @slot:    Destination program slot
 * @pref_dt: Preferred datatype (`_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`) derived from input
 *
 * Preserves the existing datatype when writing to a pre-existing shared-memory stream so readers
 * attached to that stream are not disrupted.
 *
 * Return: Selected ImageStreamIO `_DATATYPE_*` code
 */
static uint8_t select_out_dt(const mc_slot_t *slot, uint8_t pref_dt)
{
    return (slot->existed_at_plan && slot->im_id >= 0 && milk_data.image[slot->im_id].used &&
            milk_data.image[slot->im_id].md[0].shared == 1)
               ? milk_data.image[slot->im_id].md[0].datatype
               : (pref_dt ? pref_dt : _DATATYPE_FLOAT);
}

/**
 * mc_exec_prestage - Execute a pre-stage spatial, temporal, or axis-reduction constructor step
 * @prog:      Enclosing planned program
 * @cs:        Constructor step descriptor (`MC_STEP_CTOR`)
 * @avoid_id:  Image ID to avoid aliasing with when reading inputs, or -1
 * @call_site: Enclosing `def` call expression when invoked from an inlined user function, or NULL
 * @in_unit:   1 when running inside a `milk.calc` unit (default outputs in shared memory)
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Dispatches `ema`, `cumsum`, `gauss`, `conv2`, `boxmean`, `rprof`, `pixunmap`, and per-axis
 * reductions (`itot`, `imean`, `imin`, `imax`, `istd`, `irms`, `ivar`, `norm`, `asum`, `icount`,
 * `imedian`).
 *
 * Return: 0 on success, -1 on error
 */
int mc_exec_prestage(mcalc_prog_t    *prog,
                     mc_ctor_step_t  *cs,
                     imageID          avoid_id,
                     const mc_expr_t *call_site,
                     int              in_unit,
                     char            *err,
                     size_t           errsz)
{
    mc_slot_t       *slot  = &prog->slots[cs->slot_idx];
    const mc_expr_t *rhs   = call_site ? call_site->u.call.udef_body : cs->rhs_expr;
    const char      *fn    = rhs->u.call.fn->name;
    int              nargs = rhs->u.call.nargs;
    int              is_pu = (strcmp(fn, "pixunmap") == 0);
    int              is_rp = (strcmp(fn, "rprof") == 0);
    int              rc    = 0;
    mc_red_src_t     s1    = { .tmp_id = -1 };
    mc_red_src_t     s2    = { .tmp_id = -1 };

    if (mc_prepare_red_src(rhs->u.call.args[0], call_site, avoid_id, is_pu || is_rp, &s1, err,
                           errsz) != 0)
    {
        return -1;
    }

    int is_ax = !is_pu && !is_rp && strcmp(fn, "ema") != 0 && strcmp(fn, "cumsum") != 0 &&
                strcmp(fn, "gauss") != 0 && strcmp(fn, "conv2") != 0 && strcmp(fn, "boxmean") != 0;
    uint8_t pref_dt =
        (s1.dt == _DATATYPE_DOUBLE || s1.dt == _DATATYPE_INT64 || s1.dt == _DATATYPE_UINT64 ||
         (is_ax && (s1.dt == _DATATYPE_INT32 || s1.dt == _DATATYPE_UINT32)))
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    uint8_t out_dt = select_out_dt(slot, pref_dt);

    /* Case 1: Exponential moving average `ema(im, alpha)`. */
    if (strcmp(fn, "ema") == 0)
    {
        double alpha = 0.0;
        if (mc_eval_scalar_d(rhs->u.call.args[1], call_site, &alpha, err, errsz) != 0)
        {
            mc_release_red_src(&s1);
            return -1;
        }
        imageID oid = mc_out_prepare(slot, s1.naxis, s1.ishape, out_dt, in_unit, err, errsz);
        if (oid < 0)
        {
            mc_release_red_src(&s1);
            return -1;
        }
        size_t need = (size_t) s1.nel * sizeof(double);
        if (cs->scratch_cap < need)
        {
            double *nb = realloc(cs->scratch_buf, need);
            if (nb == NULL)
            {
                mc_release_red_src(&s1);
                return -1;
            }
            cs->scratch_buf = nb;
            cs->scratch_cap = need;
            cs->ema_init    = 0;
        }
        long   nvid  = variable_ID("_n");
        long   nval  = (nvid >= 0 && milk_data.variable[nvid].type == 1)
                           ? milk_data.variable[nvid].value.l
                           : (nvid >= 0 ? (long) milk_data.variable[nvid].value.f : 0);
        int    first = (!in_unit || !cs->ema_init || (nvid >= 0 && nval == 0));
        IMAGE *dst   = &milk_data.image[oid];
        for (uint64_t i = 0; i < s1.nel; i++)
        {
            double x           = arith_read_im_pix(s1.im, s1.dt, i);
            double v           = first ? x : ((1.0 - alpha) * cs->scratch_buf[i] + alpha * x);
            cs->scratch_buf[i] = v;
            mc_write_im_pix(dst, out_dt, i, v);
        }
        cs->ema_init = 1;
        mc_release_red_src(&s1);
        mc_out_post(oid);
        return 0;
    }

    /* Case 2: Cumulative prefix sum `cumsum(im[, axis])`. */
    if (strcmp(fn, "cumsum") == 0)
    {
        double axd = 0.0;
        if ((nargs == 2 &&
             mc_eval_scalar_d(rhs->u.call.args[1], call_site, &axd, err, errsz) != 0) ||
            llround(axd) < 0 || llround(axd) >= (long) s1.naxis)
        {
            mc_release_red_src(&s1);
            return (err[0] == '\0') ? pre_err(err, errsz, "cumsum: axis out of range") : -1;
        }
        imageID oid = mc_out_prepare(slot, s1.naxis, s1.ishape, out_dt, in_unit, err, errsz);
        if (oid < 0)
        {
            mc_release_red_src(&s1);
            return -1;
        }
        IMAGE *dst = &milk_data.image[oid];
        for (uint64_t i = 0; i < s1.nel; i++)
        {
            double v = arith_read_im_pix(s1.im, s1.dt, i);
            if (out_dt == _DATATYPE_DOUBLE)
            {
                dst->array.D[i] = v;
            }
            else
            {
                dst->array.F[i] = (float) v;
            }
        }
        rc = arith_image_cumsum_inplace_im(dst, (int) llround(axd));
        mc_release_red_src(&s1);
        if (rc == 0)
        {
            mc_out_post(oid);
            return 0;
        }
        return -1;
    }

    /* Case 3: Separable 2D Gaussian blur `gauss(im, sx[, sy])`. */
    if (strcmp(fn, "gauss") == 0)
    {
        double sx = 0.0;
        double sy = 0.0;
        if (mc_eval_scalar_d(rhs->u.call.args[1], call_site, &sx, err, errsz) != 0 ||
            (nargs == 3 &&
             mc_eval_scalar_d(rhs->u.call.args[2], call_site, &sy, err, errsz) != 0) ||
            sx < 0.0 || (nargs == 3 && sy < 0.0))
        {
            mc_release_red_src(&s1);
            return (err[0] == '\0') ? pre_err(err, errsz, "gauss: sigma must be non-negative") : -1;
        }
        sy          = (nargs == 3) ? sy : sx;
        imageID oid = mc_out_prepare(slot, s1.naxis, s1.ishape, out_dt, in_unit, err, errsz);
        rc = (oid < 0) ? -1
                       : arith_image_gauss2d_im(s1.im, &milk_data.image[oid], sx, sy, 0,
                                                (void **) &cs->scratch_buf, &cs->scratch_cap);
        mc_release_red_src(&s1);
        if (rc == 0)
        {
            mc_out_post(oid);
            return 0;
        }
        return -1;
    }

    /* Case 4: 2D direct convolution (`conv2`) and sliding box average (`boxmean`). */
    if (strcmp(fn, "conv2") == 0 || strcmp(fn, "boxmean") == 0)
    {
        rc = mc_exec_conv2_or_boxmean(slot, rhs, call_site, &s1, out_dt, avoid_id, in_unit, err,
                                      errsz);
        mc_release_red_src(&s1);
        return rc;
    }

    /* Case 5: Azimuthal radial profile (`rprof`) and bin-map average (`pixunmap`). */
    if (is_pu || is_rp)
    {
        double cx    = (double) (s1.ishape[0] / 2);
        double cy    = (double) ((s1.naxis > 1) ? (s1.ishape[1] / 2) : 0);
        double nd    = -1.0;
        int    nbins = 0;

        if (is_rp)
        {
            if (nargs == 3 &&
                (mc_eval_scalar_d(rhs->u.call.args[1], call_site, &cx, err, errsz) != 0 ||
                 mc_eval_scalar_d(rhs->u.call.args[2], call_site, &cy, err, errsz) != 0))
            {
                mc_release_red_src(&s1);
                return -1;
            }
            nbins = arith_image_rprof_nbins(s1.ishape, cx, cy);
        }
        else
        {
            if (mc_prepare_red_src(rhs->u.call.args[1], call_site, avoid_id, 1, &s2, err, errsz) !=
                    0 ||
                (nargs == 3 &&
                 mc_eval_scalar_d(rhs->u.call.args[2], call_site, &nd, err, errsz) != 0) ||
                s1.nel != s2.nel)
            {
                mc_release_red_src(&s1);
                mc_release_red_src(&s2);
                return (err[0] == '\0') ? pre_err(err, errsz, "pixunmap: map size must match image")
                                        : -1;
            }
            if (nargs == 3 && llround(nd) > 0)
            {
                nbins = (int) llround(nd);
            }
            else if (s2.im != NULL)
            {
                nbins = arith_image_map_nbins(s2.im);
            }
            else
            {
                double mx = 0.0;
                mc_run_stat_max(&s2, &mx, err, errsz);
                nbins = (!arith_isnan_d(mx) && mx >= 0.0) ? ((int) (int64_t) mx + 1) : 0;
            }
        }

        if (nbins <= 0 || nbins > 16777216)
        {
            mc_release_red_src(&s1);
            mc_release_red_src(&s2);
            return pre_err(err, errsz, "%s: invalid bin count", fn);
        }

        uint32_t osz[3] = { (uint32_t) nbins, 1, 1 };
        imageID  oid    = mc_out_prepare(slot, 2, osz, out_dt, in_unit, err, errsz);
        if (oid < 0)
        {
            mc_release_red_src(&s1);
            mc_release_red_src(&s2);
            return -1;
        }
        IMAGE *res = &milk_data.image[oid];
        if (cs->csr == NULL)
        {
            cs->csr = calloc(1, sizeof(arith_csr_cache_t));
        }
        arith_csr_cache_t *csr = (arith_csr_cache_t *) cs->csr;

        /* Fast CSR gather path when input (and map) are materialized 2D images. */
        if (s1.im != NULL && s1.tmp_id < 0 && s1.ishape[2] == 1 && s1.nel < (uint64_t) INT32_MAX &&
            (is_rp || (s2.im != NULL && s2.tmp_id < 0)))
        {
            int ok = is_rp ? ((csr->valid && csr->nbins == nbins && csr->cx == cx &&
                               csr->cy == cy && memcmp(csr->ishape, s1.ishape, 12) == 0) ||
                              arith_csr_build_rprof(csr, s1.ishape, cx, cy, nbins) == 0)
                           : ((csr->valid && csr->map_im == s2.im &&
                               csr->map_cnt0 == s2.im->md[0].cnt0 && csr->nbins == nbins) ||
                              arith_csr_build_map(csr, s2.im, nbins, s1.nel) == 0);
            if (ok)
            {
                arith_csr_gather(s1.im, csr, nbins, s1.nel, res);
                mc_release_red_src(&s1);
                mc_release_red_src(&s2);
                mc_out_post(oid);
                return 0;
            }
        }

        /* Fallback L1-streamed multi-threaded scatter-accumulate. */
        const int32_t *pix_bin = NULL;
        if (is_rp && arith_csr_ensure_rprof_bins(csr, s1.ishape, cx, cy, nbins) == 0)
        {
            pix_bin = csr->pix_bin;
        }
        uint64_t nblk = (s1.nel + MC_RED_BLK - 1) / MC_RED_BLK;
        int      nth  = milk_omp_team_size(s1.nel * 2, 65536);
        if ((uint64_t) nth > nblk)
        {
            nth = (int) nblk;
        }
        while (nth > 1 && (size_t) nth * (size_t) nbins * 16 > ((size_t) 64 << 20))
        {
            nth /= 2;
        }
        size_t per  = (size_t) nbins * (sizeof(double) + sizeof(uint64_t));
        size_t need = per * (size_t) nth;
        if (cs->scratch_cap < need)
        {
            double *nb = realloc(cs->scratch_buf, need);
            if (nb == NULL)
            {
                mc_release_red_src(&s1);
                mc_release_red_src(&s2);
                return -1;
            }
            cs->scratch_buf = nb;
            cs->scratch_cap = need;
        }
        seg_scatter_job_t job = { &s1, is_rp ? NULL : &s2, pix_bin,       nbins, s1.nel,
                                  per, cs->scratch_buf,    (double) nbins };
        nth                   = arith_team_run(nth, nblk, seg_scatter_cb, &job);

        const double   *sum0 = cs->scratch_buf;
        const uint64_t *cnt0 = (const uint64_t *) (sum0 + nbins);
        for (int b = 0; b < nbins; b++)
        {
            double   s = sum0[b];
            uint64_t c = cnt0[b];
            for (int t = 1; t < nth; t++)
            {
                const double   *st = (const double *) ((const char *) sum0 + per * (size_t) t);
                const uint64_t *ct = (const uint64_t *) (st + nbins);
                s += st[b];
                c += ct[b];
            }
            double r = (c > 0) ? (s / (double) c) : 0.0;
            if (out_dt == _DATATYPE_DOUBLE)
            {
                res->array.D[b] = r;
            }
            else
            {
                res->array.F[b] = (float) r;
            }
        }
        mc_release_red_src(&s1);
        mc_release_red_src(&s2);
        mc_out_post(oid);
        return 0;
    }

    /* Case 6: Per-axis reduction (`imean(im, axis)`, `istd(im, axis)`, etc.). */
    double axd = 0.0;
    if (mc_eval_scalar_d(rhs->u.call.args[1], call_site, &axd, err, errsz) != 0 ||
        llround(axd) < 0 || llround(axd) >= (long) s1.naxis)
    {
        mc_release_red_src(&s1);
        return (err[0] == '\0') ? pre_err(err, errsz, "Axis out of range") : -1;
    }
    long     ax    = llround(axd);
    long     nax   = s1.naxis;
    uint32_t sz[3] = { 1, 1, 1 };
    int      k     = 0;

    /*
     * When `cs->is_anon == 2`, the reduction is hoisted inside a larger elementwise pass on `s1`,
     * so the reduced axis is kept at extent 1 for automatic broadcast alignment. Otherwise the
     * reduced axis is squeezed out.
     */
    for (int a = 0; a < (int) nax; a++)
    {
        if (cs->is_anon == 2)
        {
            sz[a] = (a == ax) ? 1 : s1.ishape[a];
        }
        else if (a != ax)
        {
            sz[k++] = s1.ishape[a];
        }
    }
    imageID oid = mc_out_prepare(slot, (uint8_t) (cs->is_anon == 2 ? nax : (k > 0 ? k : 1)), sz,
                                 out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        mc_release_red_src(&s1);
        return -1;
    }

    uint64_t inner = 1;
    uint64_t outer = 1;
    uint64_t n     = s1.ishape[ax];
    for (int a = 0; a < (int) nax; a++)
    {
        if (a < ax)
        {
            inner *= s1.ishape[a];
        }
        else if (a > ax)
        {
            outer *= s1.ishape[a];
        }
    }

    arith_reduce_op op = (arith_reduce_op) mc_map_reduce_op(fn);
    int      fast_vec  = (ax > 0 && (s1.dt == _DATATYPE_FLOAT || s1.dt == _DATATYPE_DOUBLE) &&
                          (op == ARITH_RED_SUM || op == ARITH_RED_MEAN || op == ARITH_RED_NORM ||
                           op == ARITH_RED_ASUM));
    uint64_t nitems =
        fast_vec ? (outer * ((inner + MC_RED_BLK - 1) / MC_RED_BLK)) : (outer * inner);
    int nth = milk_omp_team_size(s1.nel, 65536);
    if ((uint64_t) nth > nitems)
    {
        nth = (int) nitems;
    }
    ax_team_job_t job = { s1.im, &milk_data.image[oid], inner, n, op, fast_vec };
    arith_team_run(nth, nitems, ax_team_cb, &job);
    mc_release_red_src(&s1);
    mc_out_post(oid);
    return 0;
}
