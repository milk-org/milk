// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_prestage_seg.c
 * @brief CSR gather and 4-way scatter evaluators for `pixunmap` and `rprof` pre-stages.
 *
 * Role & Goal:
 *   Builds and caches CSR row-pointer tables (`ax_csr_cache`) for static leaf `pixunmap`
 *   and `rprof` calls, and provides a 4-way interleaved block scatter accumulator when
 *   the image or bin map is a dynamic expression.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_calc_fuse_internal.h"
#include "libmilkdata/milk_type_dispatch.h"

/** run_seg_reduce_csr - Execute cached CSR gather for leaf `pixunmap` or `rprof`. */
static int run_seg_reduce_csr(fprog *p, int slot, int nbins, uint64_t nel)
{
    fprog        *sub    = p->ax_p[slot];
    fprog        *map_s  = p->ax_p2[slot];
    IMAGE        *res    = &p->ax_im[slot];
    ax_csr_cache *csr    = &p->ax_csr[slot];
    const IMAGE  *src_im = sub->img[0];

    if (p->ax_kind[slot] == AX_KIND_RPROF)
    {
        double cx = p->ax_c[slot][0];
        double cy = p->ax_c[slot][1];
        if (!csr->valid || csr->nbins != nbins || csr->cx != cx || csr->cy != cy ||
            memcmp(csr->ishape, sub->ishape, sizeof(csr->ishape)) != 0)
        {
            if (arith_csr_build_rprof(csr, sub->ishape, cx, cy, nbins) != 0)
            {
                return -1;
            }
        }
    }
    else
    {
        const IMAGE *map_im = map_s->img[0];
        if (!csr->valid || csr->map_im != map_im || csr->map_cnt0 != map_im->md[0].cnt0 ||
            csr->nbins != nbins)
        {
            if (arith_csr_build_map(csr, map_im, nbins, nel) != 0)
            {
                return -1;
            }
        }
    }

    arith_csr_gather(src_im, csr, nbins, nel, res);
    return 0;
}

typedef struct
{
    const fprog   *sub;
    const fprog   *map_sub;
    const int32_t *pix_bin;
    int            nbins;
    uint64_t       nel;
    size_t         per;
    void          *buf;
    double         max_n;
} scatter_job_t;

/** scatter_job() - team job: blocks [lo, hi) of the expression scattered into thread @tid's partials. */
static void scatter_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const scatter_job_t *j   = (const scatter_job_t *) ctx;
    double              *sum = (double *) ((char *) j->buf + j->per * (size_t) tid);
    uint64_t            *cnt = (uint64_t *) (sum + j->nbins);
    (void) nth;
    memset(sum, 0, (size_t) j->nbins * sizeof(double));
    memset(cnt, 0, (size_t) j->nbins * sizeof(uint64_t));
    MILK_ALIGNED(64) double stkd[FUSE_MAXSTK][FUSE_BLK];
    MILK_ALIGNED(64) double mstkd[FUSE_MAXSTK][FUSE_BLK];
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t off = b * FUSE_BLK;
        int      len = (int) ((j->nel - off < FUSE_BLK) ? j->nel - off : FUSE_BLK);
        cli_calc_fuse_eval_block_d(j->sub, off, len, stkd);
        if (j->pix_bin != NULL)
        {
            const int32_t *pb = j->pix_bin + off;
            for (int q = 0; q < len; q++)
            {
                int bb = pb[q];
                if (bb >= 0)
                {
                    sum[bb] += stkd[0][q];
                    cnt[bb]++;
                }
            }
            continue;
        }
        cli_calc_fuse_eval_block_d(j->map_sub, off, len, mstkd);
        for (int q = 0; q < len; q++)
        {
            double m = mstkd[0][q];
            if (!cli_calc_isnan(m) && m >= 0.0 && m < j->max_n)
            {
                int bb = (int) (int64_t) m;
                if (bb >= 0 && bb < j->nbins)
                {
                    sum[bb] += stkd[0][q];
                    cnt[bb]++;
                }
            }
        }
    }
}

/** run_seg_reduce_scatter - Block scatter of an expression into bins, threads on private partials. */
static int run_seg_reduce_scatter(fprog *p, int slot, int nbins, uint64_t nel)
{
    fprog         *sub     = p->ax_p[slot];
    fprog         *map_sub = p->ax_p2[slot];
    IMAGE         *res     = &p->ax_im[slot];
    ax_csr_cache  *csr     = &p->ax_csr[slot];
    int            kind    = p->ax_kind[slot];
    const int32_t *pix_bin = NULL;

    if (kind == AX_KIND_RPROF)
    {
        if (arith_csr_ensure_rprof_bins(csr, sub->ishape, p->ax_c[slot][0], p->ax_c[slot][1],
                                        nbins) != 0)
        {
            return -1;
        }
        pix_bin = csr->pix_bin;
    }
    uint64_t nblk = (nel + FUSE_BLK - 1) / FUSE_BLK;
    int      nth  = milk_omp_team_size(fprog_effective_nel(sub) * 2, FUSE_OMP_MIN);
    if ((uint64_t) nth > nblk)
    {
        nth = (int) nblk;
    }
    while (nth > 1 && (size_t) nth * (size_t) nbins * 16 > ((size_t) 64 << 20))
    {
        nth /= 2; /* partials of nth * nbins must stay small */
    }
    size_t per  = (size_t) nbins * (sizeof(double) + sizeof(uint64_t));
    size_t need = per * (size_t) nth;
    if (sub->alias_cap < need)
    {
        void *na = realloc(sub->alias_buf, need);
        if (na == NULL)
        {
            return -1;
        }
        sub->alias_buf = na;
        sub->alias_cap = need;
    }
    const double max_n = (double) nbins;

    scatter_job_t job = { sub, map_sub, pix_bin, nbins, nel, per, sub->alias_buf, max_n };
    nth               = cli_calc_team_run(nth, nblk, scatter_job, &job);

    const double   *sum0 = (const double *) sub->alias_buf;
    const uint64_t *cnt0 = (const uint64_t *) (sum0 + nbins);
    for (int b = 0; b < nbins; b++)
    {
        double   s = sum0[b];
        uint64_t c = cnt0[b];
        for (int t = 1; t < nth; t++)
        {
            const double *st = (const double *) ((const char *) sub->alias_buf + per * (size_t) t);
            const uint64_t *ct = (const uint64_t *) (st + nbins);
            s += st[b];
            c += ct[b];
        }
        double r = (c > 0) ? (s / (double) c) : 0.0;
        if (res->md[0].datatype == _DATATYPE_FLOAT)
        {
            res->array.F[b] = (float) r;
        }
        else
        {
            res->array.D[b] = r;
        }
    }
    return 0;
}

/**
 * cli_calc_fuse_run_seg_reduce - Evaluate segment-mean (`pixunmap` or `rprof`) pre-stage @slot
 * @p:    Parent fused program
 * @slot: Pre-stage slot index
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_run_seg_reduce(fprog *p, int slot)
{
    fprog *sub     = p->ax_p[slot];
    fprog *map_sub = p->ax_p2[slot];
    IMAGE *res     = &p->ax_im[slot];
    int    kind    = p->ax_kind[slot];

    double sval[FUSE_MAXINS];
    if (cli_calc_fuse_eval_scalars(sub, sval) != 0 ||
        cli_calc_fuse_refresh_views(sub, sval, NULL) != 0 ||
        cli_calc_fuse_finalize_views(sub) != 0 || cli_calc_fuse_run_axes(sub) != 0)
    {
        return -1;
    }
    if (map_sub != NULL &&
        (cli_calc_fuse_eval_scalars(map_sub, sval) != 0 ||
         cli_calc_fuse_refresh_views(map_sub, sval, NULL) != 0 ||
         cli_calc_fuse_finalize_views(map_sub) != 0 || cli_calc_fuse_run_axes(map_sub) != 0))
    {
        return -1;
    }

    const uint64_t nel   = sub->inel;
    const int      nbins = (int) res->md[0].nelement;
    if (nbins <= 0 || (map_sub != NULL && map_sub->inel != nel))
    {
        return -1;
    }

    if (nel < (uint64_t) INT32_MAX && sub->nops == 0 && sub->nimg == 1 && !sub->view[0].is_view &&
        sub->ishape[2] == 1 &&
        (kind == AX_KIND_RPROF || (map_sub != NULL && map_sub->nops == 0 && map_sub->nimg == 1 &&
                                   !map_sub->view[0].is_view)))
    {
        return run_seg_reduce_csr(p, slot, nbins, nel);
    }
    return run_seg_reduce_scatter(p, slot, nbins, nel);
}

static int push_seg_image(fprog *p, const IMAGE *im)
{
    int k = cli_calc_fuse_add_image(p, im);
    if (k < 0)
    {
        return -1;
    }
    p->ins[p->n++] = (fins) { .k = FI_IMG, .img = k };
    return 0;
}

/**
 * cli_calc_pixunmap - Evaluate a standalone segment-mean `pixunmap(im, map[, n])`
 * @im:  Source image
 * @map: Bin index map image
 * @n:   Explicit bin count (`<= 0` to auto-detect from `max(map) + 1`)
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_pixunmap(const IMAGE *im, const IMAGE *map, int64_t n)
{
    static fprog p;
    static fprog sub_im;
    static fprog sub_map;

    if (im == NULL || map == NULL || im->md[0].nelement != map->md[0].nelement)
    {
        return NULL;
    }
    int nbins = (n > 0) ? (int) n : cli_calc_fuse_map_nbins(map);
    if (nbins <= 0 || nbins > 16777216)
    {
        return NULL;
    }
    cli_calc_fuse_reset_fprog(&p);
    cli_calc_fuse_reset_fprog(&sub_im);
    cli_calc_fuse_reset_fprog(&sub_map);
    if (push_seg_image(&sub_im, im) < 0 || push_seg_image(&sub_map, map) < 0)
    {
        return NULL;
    }
    uint8_t dt = (im->md[0].datatype == _DATATYPE_DOUBLE || im->md[0].datatype == _DATATYPE_INT64 ||
                  im->md[0].datatype == _DATATYPE_UINT64)
                     ? _DATATYPE_DOUBLE
                     : _DATATYPE_FLOAT;
    uint32_t    osz[3] = { (uint32_t) nbins, 1, 1 };
    const char *tmpn   = alloc_tmpname();
    imageID     tid    = -1;
    cli_calc_tmp_image(tmpn, 2, osz, dt, 1, &tid);
    if (tid == -1)
    {
        return NULL;
    }
    p.ax_p[0]    = &sub_im;
    p.ax_p2[0]   = &sub_map;
    p.ax_kind[0] = AX_KIND_SEG;
    p.ax_im[0]   = data.core.image[tid];
    int rc       = cli_calc_fuse_run_seg_reduce(&p, 0);
    p.ax_p[0]    = NULL;
    p.ax_p2[0]   = NULL;
    return (rc == 0) ? tmpn : NULL;
}

/**
 * cli_calc_rprof - Evaluate a standalone radial profile `rprof(im, cx, cy)`
 * @im: Source 2D image
 * @cx: Center X coordinate
 * @cy: Center Y coordinate
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_rprof(const IMAGE *im, double cx, double cy)
{
    static fprog p;
    static fprog sub_im;

    if (im == NULL)
    {
        return NULL;
    }
    cli_calc_fuse_reset_fprog(&p);
    cli_calc_fuse_reset_fprog(&sub_im);
    if (push_seg_image(&sub_im, im) < 0 || cli_calc_fuse_finalize_views(&sub_im) != 0)
    {
        return NULL;
    }
    int nbins = cli_calc_fuse_rprof_nbins(sub_im.ishape, cx, cy);
    if (nbins <= 0)
    {
        return NULL;
    }
    uint8_t dt = (im->md[0].datatype == _DATATYPE_DOUBLE || im->md[0].datatype == _DATATYPE_INT64 ||
                  im->md[0].datatype == _DATATYPE_UINT64)
                     ? _DATATYPE_DOUBLE
                     : _DATATYPE_FLOAT;
    uint32_t    osz[3] = { (uint32_t) nbins, 1, 1 };
    const char *tmpn   = alloc_tmpname();
    imageID     tid    = -1;
    cli_calc_tmp_image(tmpn, 2, osz, dt, 1, &tid);
    if (tid == -1)
    {
        return NULL;
    }
    p.ax_p[0]    = &sub_im;
    p.ax_p2[0]   = NULL;
    p.ax_kind[0] = AX_KIND_RPROF;
    p.ax_c[0][0] = cx;
    p.ax_c[0][1] = cy;
    p.ax_im[0]   = data.core.image[tid];
    int rc       = cli_calc_fuse_run_seg_reduce(&p, 0);
    p.ax_p[0]    = NULL;
    return (rc == 0) ? tmpn : NULL;
}

/**
 * cli_calc_gauss - Evaluate a standalone 2D Gaussian blur `gauss(im, sx, sy)`
 * @im: Source image
 * @sx: X standard deviation
 * @sy: Y standard deviation
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_gauss(const IMAGE *im, double sx, double sy)
{
    static fprog p;
    static fprog sub_im;

    if (im == NULL || sx < 0.0 || sy < 0.0)
    {
        return NULL;
    }
    cli_calc_fuse_reset_fprog(&p);
    cli_calc_fuse_reset_fprog(&sub_im);
    if (push_seg_image(&sub_im, im) < 0 || cli_calc_fuse_finalize_views(&sub_im) != 0)
    {
        return NULL;
    }
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, sub_im.inaxis, sub_im.ishape,
                       sub_im.out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return NULL;
    }
    p.ax_p[0]    = &sub_im;
    p.ax_kind[0] = AX_KIND_GAUSS;
    p.ax_c[0][0] = sx;
    p.ax_c[0][1] = sy;
    p.ax_im[0]   = data.core.image[tid];
    int rc       = cli_calc_fuse_run_gauss_prestage(&p, 0);
    p.ax_p[0]    = NULL;
    return (rc == 0) ? tmpn : NULL;
}

typedef struct
{
    IMAGE   *dst;
    uint64_t nel;
    double   val;
} fill_job_t;

/** fill_job() - team job: blocks [lo, hi) of @dst set to a constant. */
static void fill_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const fill_job_t *j = (const fill_job_t *) ctx;
    (void) tid;
    (void) nth;
    for (uint64_t b = lo; b < hi; b++)
    {
        uint64_t off = b * FUSE_BLK;
        cli_calc_fuse_store_block(j->dst, off,
                                  (int) ((j->nel - off < FUSE_BLK) ? j->nel - off : FUSE_BLK), NULL,
                                  1, j->val);
    }
}

/**
 * cli_calc_cumsum - Evaluate a standalone cumulative sum `cumsum(im[, axis])`
 * @im:   Source image
 * @axis: Axis to sum along (`0 .. naxis - 1`)
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_cumsum(const IMAGE *im, int axis)
{
    static fprog p;
    static fprog sub_im;

    if (im == NULL || axis < 0 || axis >= im->md[0].naxis)
    {
        parse_errmsg("cumsum: axis out of range");
        return NULL;
    }
    cli_calc_fuse_reset_fprog(&p);
    cli_calc_fuse_reset_fprog(&sub_im);
    if (push_seg_image(&sub_im, im) < 0 || cli_calc_fuse_finalize_views(&sub_im) != 0)
    {
        return NULL;
    }
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, sub_im.inaxis, sub_im.ishape,
                       sub_im.out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return NULL;
    }
    p.ax_p[0]    = &sub_im;
    p.ax_kind[0] = AX_KIND_CUMSUM;
    p.ax_c[0][0] = (double) axis;
    p.ax_im[0]   = data.core.image[tid];
    int rc       = cli_calc_fuse_run_cumsum_prestage(&p, 0);
    p.ax_p[0]    = NULL;
    return (rc == 0) ? tmpn : NULL;
}

/**
 * cli_calc_inplace - Apply an in-place assignment or compound assignment (`=`, `+=`, `-=`, `*=`, `/=`)
 * @dst: Destination image
 * @aop: Assignment operator
 * @src: Optional source image (or NULL for scalar @val)
 * @val: Scalar operand when @src is NULL
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_inplace(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val)
{
    if (aop == TOK_OP_SLASH_EQ && src == NULL && val == 0.0)
    {
        parse_errmsg("Division by zero");
        return -1;
    }
    if (!is_real_datatype(dst->md[0].datatype) ||
        (src != NULL && !is_real_datatype(src->md[0].datatype)))
    {
        return cli_calc_inplace_complex(dst, aop, src, val);
    }
    uint64_t nel = dst->md[0].nelement;
    if (src != NULL && aop == TOK_OP_SLASH_EQ)
    {
        MILK_ALIGNED(64) double blk[FUSE_BLK];
        int                     bad = 0;
        for (uint64_t off = 0; off < nel && !bad; off += FUSE_BLK)
        {
            int len = (int) ((nel - off < FUSE_BLK) ? nel - off : FUSE_BLK);
            cli_calc_fuse_load_block(src, off, len, blk);
            for (int j = 0; j < len && !bad; j++)
            {
                bad = (blk[j] == 0.0);
            }
        }
        if (bad)
        {
            parse_errmsg("Division by zero");
            return -1;
        }
    }

    if (aop == TOK_EQUAL && src == NULL)
    {
        int        nth = milk_omp_team_size(nel, FUSE_OMP_MIN);
        fill_job_t job = { dst, nel, val };
        cli_calc_team_run(nth, (nel + FUSE_BLK - 1) / FUSE_BLK, fill_job, &job);
        return 0;
    }
    fprog *p = (fprog *) calloc(1, sizeof(*p));
    if (p == NULL)
    {
        return -1;
    }
    cli_calc_fuse_reset_fprog(p);
    cli_token_type op = (aop == TOK_OP_PLUS_EQ)    ? TOK_OP_PLUS
                        : (aop == TOK_OP_MINUS_EQ) ? TOK_OP_MINUS
                        : (aop == TOK_OP_STAR_EQ)  ? TOK_OP_STAR
                                                   : TOK_OP_SLASH;
    cli_calc_fuse_add_flat_image(p, dst);
    if (src != NULL)
    {
        cli_calc_fuse_add_flat_image(p, src);
    }
    if (aop == TOK_EQUAL)
    {
        p->ins[p->n++] = (fins) { .k = FI_IMG, .img = 1 };
    }
    else
    {
        p->ins[p->n++] = (fins) { .k = FI_IMG, .img = 0 };
        if (src != NULL)
        {
            p->ins[p->n++] = (fins) { .k = FI_IMG, .img = 1 };
        }
        p->ins[p->n++] = (fins) { .k = FI_BIN, .op = op, .cpos = (src != NULL) ? 0 : 2, .c = val };
    }
    int rc = cli_calc_fuse_run_program(p, dst);
    cli_calc_fuse_free_fprog_children(p);
    free(p);
    return rc;
}
