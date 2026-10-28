// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_prestage.c
 * @brief Pre-stage evaluators for axis reductions, separable Gaussian convolution, and bin sizing.
 *
 * Role & Goal:
 *   Evaluates axis reductions (`AX_KIND_AXIS`), separable 2D Gaussian blurs (`AX_KIND_GAUSS`),
 *   and dispatches all registered pre-stage slots (`cli_calc_fuse_run_axes`) before main
 *   fused program execution.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "cli_calc_fuse_internal.h"
#include "libmilkdata/milk_type_dispatch.h"


/*
 * Axis reductions: the sub-program's iteration space is walked in row segments (nx contiguous
 * samples of row (y, z)); a segment reduces into one output (axis 0) or adds elementwise into a
 * row of outputs (axes 1 and 2). Threads own disjoint outputs, so there is nothing to merge: rows
 * for axis 0, row groups across every plane for axis 2, planes for axis 1 of a cube. A 2-D axis-1
 * reduction (every row hits the same outputs) uses per-thread partials merged at the end.
 */

/** AXIS_ACCUM_ROW - add the N samples s[] of one segment to one output (OST 0) or a row (OST 1). */
#define AXIS_ACCUM_ROW(OST, s, N, S, op, a0, a1, aref, acnt)                                  \
    do                                                                                        \
    {                                                                                         \
        if (OST == 0)                                                                         \
        {                                                                                     \
            double t0_ = 0.0, t1_ = 0.0;                                                      \
            switch (op)                                                                       \
            {                                                                                 \
            case RED_STD:                                                                     \
            case RED_VAR:                                                                     \
            {                                                                                 \
                if ((acnt)[S] == 0.0 && (N) > 0)                                              \
                {                                                                             \
                    (aref)[S] = (double) (s)[0];                                              \
                }                                                                             \
                const double r_ = (aref)[S];                                                  \
                _Pragma("omp simd reduction(+ : t0_, t1_)") for (int i = 0; i < (N); i++)     \
                {                                                                             \
                    double d_ = (double) (s)[i] - r_;                                         \
                    t0_ += d_;                                                                \
                    t1_ += d_ * d_;                                                           \
                }                                                                             \
                (a0)[S] += t0_;                                                               \
                (a1)[S] += t1_;                                                               \
                (acnt)[S] += (double) (N);                                                    \
                break;                                                                        \
            }                                                                                 \
            case RED_MIN:                                                                     \
            case RED_MAX:                                                                     \
            {                                                                                 \
                double m_ = (a0)[S];                                                          \
                double c_ = (acnt)[S];                                                        \
                for (int i = 0; i < (N); i++)                                                 \
                {                                                                             \
                    double v_ = (double) (s)[i];                                              \
                    if (!arith_isnan_d(v_) &&                                                 \
                        (c_ == 0.0 || ((op) == RED_MIN ? (v_ < m_) : (v_ > m_))))             \
                    {                                                                         \
                        m_ = v_;                                                              \
                        c_ = 1.0;                                                             \
                    }                                                                         \
                }                                                                             \
                (a0)[S]   = m_;                                                               \
                (acnt)[S] = c_;                                                               \
                break;                                                                        \
            }                                                                                 \
            case RED_ASUM:                                                                    \
                _Pragma("omp simd reduction(+ : t0_)") for (int i = 0; i < (N); i++)          \
                {                                                                             \
                    t0_ += fabs((double) (s)[i]);                                             \
                }                                                                             \
                (a0)[S] += t0_;                                                               \
                break;                                                                        \
            case RED_NORM:                                                                    \
                _Pragma("omp simd reduction(+ : t0_)") for (int i = 0; i < (N); i++)          \
                {                                                                             \
                    double v_ = (double) (s)[i];                                              \
                    t0_ += v_ * v_;                                                           \
                }                                                                             \
                (a0)[S] += t0_;                                                               \
                break;                                                                        \
            default:                                                                          \
                _Pragma("omp simd reduction(+ : t0_)") for (int i = 0; i < (N); i++)          \
                {                                                                             \
                    t0_ += (double) (s)[i];                                                   \
                }                                                                             \
                (a0)[S] += t0_;                                                               \
                break;                                                                        \
            }                                                                                 \
        }                                                                                     \
        else                                                                                  \
        {                                                                                     \
            double *restrict o0_ = (a0) + (S);                                                \
            switch (op)                                                                       \
            {                                                                                 \
            case RED_STD:                                                                     \
            case RED_VAR:                                                                     \
            {                                                                                 \
                double *restrict o1_ = (a1) + (S);                                            \
                double *restrict or_ = (aref) + (S);                                          \
                double *restrict oc_ = (acnt) + (S);                                          \
                for (int i = 0; i < (N); i++)                                                 \
                {                                                                             \
                    double v_ = (double) (s)[i];                                              \
                    if (oc_[i] == 0.0)                                                        \
                    {                                                                         \
                        or_[i] = v_;                                                          \
                    }                                                                         \
                    double d_ = v_ - or_[i];                                                  \
                    o0_[i] += d_;                                                             \
                    o1_[i] += d_ * d_;                                                        \
                    oc_[i] += 1.0;                                                            \
                }                                                                             \
                break;                                                                        \
            }                                                                                 \
            case RED_MIN:                                                                     \
            case RED_MAX:                                                                     \
            {                                                                                 \
                double *restrict oc_ = (acnt) + (S);                                          \
                for (int i = 0; i < (N); i++)                                                 \
                {                                                                             \
                    double v_ = (double) (s)[i];                                              \
                    if (!arith_isnan_d(v_) &&                                                 \
                        (oc_[i] == 0.0 || ((op) == RED_MIN ? (v_ < o0_[i]) : (v_ > o0_[i])))) \
                    {                                                                         \
                        o0_[i] = v_;                                                          \
                        oc_[i] = 1.0;                                                         \
                    }                                                                         \
                }                                                                             \
                break;                                                                        \
            }                                                                                 \
            case RED_ASUM:                                                                    \
                _Pragma("omp simd") for (int i = 0; i < (N); i++)                             \
                {                                                                             \
                    o0_[i] += fabs((double) (s)[i]);                                          \
                }                                                                             \
                break;                                                                        \
            case RED_NORM:                                                                    \
                _Pragma("omp simd") for (int i = 0; i < (N); i++)                             \
                {                                                                             \
                    double v_ = (double) (s)[i];                                              \
                    o0_[i] += v_ * v_;                                                        \
                }                                                                             \
                break;                                                                        \
            default:                                                                          \
                _Pragma("omp simd") for (int i = 0; i < (N); i++)                             \
                {                                                                             \
                    o0_[i] += (double) (s)[i];                                                \
                }                                                                             \
                break;                                                                        \
            }                                                                                 \
        }                                                                                     \
    } while (0)

typedef struct
{
    const fprog *sub;
    const fview *wv; /* iteration element -> output index */
    int          ost;
    reduce_op    op;
    int          use_flt;
    double      *a0;
    double      *a1;
    double      *aref;
    double      *acnt;
} axis_acc_t;

/** accum_range() - evaluate @sub on iteration elements [lo, hi) and accumulate their row segments. */
static void accum_range(const axis_acc_t *ac, uint64_t lo, uint64_t hi, void *stk)
{
    const fprog *sub = ac->sub;
    for (uint64_t off = lo; off < hi; off += FUSE_BLK)
    {
        int len = (int) ((hi - off < FUSE_BLK) ? hi - off : FUSE_BLK);
        if (ac->use_flt)
        {
            float (*stkf)[FUSE_BLK] = (float (*)[FUSE_BLK]) stk;
            cli_calc_fuse_eval_block_f(sub, off, len, stkf);
            VIEW_WALK(sub, ac->wv, off, len, {
                const float *s = stkf[0] + J;
                AXIS_ACCUM_ROW(ac->ost, s, N, S, ac->op, ac->a0, ac->a1, ac->aref, ac->acnt);
            });
        }
        else
        {
            double (*stkd)[FUSE_BLK] = (double (*)[FUSE_BLK]) stk;
            cli_calc_fuse_eval_block_d(sub, off, len, stkd);
            VIEW_WALK(sub, ac->wv, off, len, {
                const double *s = stkd[0] + J;
                AXIS_ACCUM_ROW(ac->ost, s, N, S, ac->op, ac->a0, ac->a1, ac->aref, ac->acnt);
            });
        }
    }
}

/** merge_axis_thread_acc - Merge thread-local axis accumulators into shared output accumulators. */
static void merge_axis_thread_acc(uint64_t      nout,
                                  reduce_op     op,
                                  const double *l0,
                                  const double *l1,
                                  const double *lref,
                                  const double *lcnt,
                                  double       *a0,
                                  double       *a1,
                                  double       *aref,
                                  double       *acnt)
{
    for (uint64_t o = 0; o < nout; o++)
    {
        if (op == RED_STD || op == RED_VAR)
        {
            if (lcnt[o] > 0.0)
            {
                if (acnt[o] == 0.0)
                {
                    a0[o]   = l0[o];
                    a1[o]   = l1[o];
                    aref[o] = lref[o];
                    acnt[o] = lcnt[o];
                }
                else
                {
                    double d = lref[o] - aref[o];
                    a0[o] += l0[o] + lcnt[o] * d;
                    a1[o] += l1[o] + 2.0 * d * l0[o] + lcnt[o] * d * d;
                    acnt[o] += lcnt[o];
                }
            }
        }
        else if (op == RED_MIN)
        {
            if (lcnt[o] > 0.0 && (acnt[o] == 0.0 || l0[o] < a0[o]))
            {
                a0[o]   = l0[o];
                acnt[o] = 1.0;
            }
        }
        else if (op == RED_MAX)
        {
            if (lcnt[o] > 0.0 && (acnt[o] == 0.0 || l0[o] > a0[o]))
            {
                a0[o]   = l0[o];
                acnt[o] = 1.0;
            }
        }
        else
        {
            a0[o] += l0[o];
        }
    }
}

/** axis_needs_all() - 1 if @op keeps four accumulators per output, 0 if the sum alone. */
static inline int axis_needs_all(reduce_op op)
{
    return op == RED_STD || op == RED_VAR || op == RED_MIN || op == RED_MAX;
}

typedef struct
{
    axis_acc_t ac;
    int        axis;
    int        twod_ax1;
    uint64_t   nchunk;
    uint64_t   rpc;
    uint64_t   nx;
    uint64_t   ny;
    uint64_t   nz;
    int        npart;
    double    *acc;
    uint64_t   nout;
    int        all;
} axis_job_t;

/** axis_job() - team job: chunks [lo, hi) of rows or planes, own outputs or private partials. */
static void axis_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const axis_job_t       *j = (const axis_job_t *) ctx;
    MILK_ALIGNED(64) double stk[FUSE_MAXSTK][FUSE_BLK];
    axis_acc_t              my = j->ac;
    (void) nth;
    if (j->npart > 0)
    {
        double *part = j->acc + (size_t) (1 + tid) * 4 * j->nout;
        memset(part, 0, (size_t) (j->all ? 4 : 1) * j->nout * sizeof(double));
        my.a0   = part;
        my.a1   = part + j->nout;
        my.aref = part + 2 * j->nout;
        my.acnt = part + 3 * j->nout;
    }
    const uint64_t nx = j->nx, ny = j->ny, nz = j->nz, rpc = j->rpc;
    for (uint64_t c = lo; c < hi; c++)
    {
        if (j->axis == 0)
        {
            uint64_t r0 = c * rpc;
            uint64_t r1 = (r0 + rpc < ny * nz) ? r0 + rpc : ny * nz;
            accum_range(&my, r0 * nx, r1 * nx, stk);
        }
        else if (j->axis == 2 || j->twod_ax1)
        {
            uint64_t y0 = c * rpc;
            uint64_t y1 = (y0 + rpc < ny) ? y0 + rpc : ny;
            for (uint64_t z = 0; z < nz; z++)
            {
                accum_range(&my, z * nx * ny + y0 * nx, z * nx * ny + y1 * nx, stk);
            }
        }
        else
        {
            accum_range(&my, c * nx * ny, (c + 1) * nx * ny, stk);
        }
    }
    if (j->npart > 0)
    {
        cli_calc_team_lock();
        merge_axis_thread_acc(j->nout, my.op, my.a0, my.a1, my.aref, my.acnt, j->ac.a0, j->ac.a1,
                              j->ac.aref, j->ac.acnt);
        cli_calc_team_unlock();
    }
}

typedef struct
{
    const double *a0;
    const double *a1;
    const double *acnt;
    reduce_op     op;
    double        L;
    IMAGE        *res;
    int           is_flt;
} axis_fin_t;

/** axis_fin_job() - team job: outputs [lo, hi) from their accumulators. */
static void axis_fin_job(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)
{
    const axis_fin_t *f = (const axis_fin_t *) ctx;
    (void) tid;
    (void) nth;
    for (uint64_t o = lo; o < hi; o++)
    {
        double r = f->a0[o];
        if ((f->op == RED_MIN || f->op == RED_MAX) && f->acnt[o] == 0.0)
        {
            r = NAN;
        }
        else if (f->op == RED_MEAN)
        {
            r /= f->L;
        }
        else if (f->op == RED_NORM)
        {
            r = sqrt(r);
        }
        else if (f->op == RED_VAR || f->op == RED_STD)
        {
            double m   = f->a0[o] / f->L;
            double var = (f->a1[o] / f->L) - m * m;
            if (var < 0.0)
            {
                var = 0.0;
            }
            r = (f->op == RED_STD) ? sqrt(var) : var;
        }
        if (f->is_flt)
        {
            f->res->array.F[o] = (float) r;
        }
        else
        {
            f->res->array.D[o] = r;
        }
    }
}

/** run_axis_reduce - Evaluate @sub over its iteration space and reduce it along @axis with @op. */
static int run_axis_reduce(fprog *sub, int axis, reduce_op op, IMAGE *res)
{
    double sval[FUSE_MAXINS];
    if (cli_calc_fuse_eval_scalars(sub, sval) != 0 ||
        cli_calc_fuse_refresh_views(sub, sval, NULL) != 0 ||
        cli_calc_fuse_finalize_views(sub) != 0 || cli_calc_fuse_run_axes(sub) != 0)
    {
        return -1;
    }
    const uint64_t nx = sub->ishape[0], ny = sub->ishape[1], nz = sub->ishape[2], nel = sub->inel;
    const uint64_t nout = res->md[0].nelement;
    const double   L    = (double) sub->ishape[axis];
    if (nout == 0 || nout * sub->ishape[axis] != nel)
    {
        parse_errmsg("axis reduction: the image changed shape");
        return -1;
    }
    const int use_flt  = cli_calc_fuse_all_float_prog(sub);
    const int all      = axis_needs_all(op);
    const int twod_ax1 = (axis == 1 && nz == 1); /* every row hits the same outputs: partials */
    int       nth      = milk_omp_team_size(fprog_effective_nel(sub), FUSE_OMP_MIN);
    const uint64_t rpc = (nx >= FUSE_BLK) ? 1 : (FUSE_BLK / nx); /* rows per chunk */
    uint64_t       nchunk;
    if (axis == 0)
    {
        nchunk = (ny * nz + rpc - 1) / rpc;
    }
    else if (axis == 2 || twod_ax1)
    {
        nchunk = (ny + rpc - 1) / rpc;
    }
    else
    {
        nchunk = nz;
    }
    if ((uint64_t) nth > nchunk)
    {
        nth = (int) nchunk;
    }
    const int npart      = twod_ax1 ? nth : 0; /* per-thread partial accumulators */
    size_t    need_bytes = (size_t) (4 * nout) * sizeof(double) * (size_t) (1 + npart);
    if (sub->alias_cap < need_bytes)
    {
        void *na = realloc(sub->alias_buf, need_bytes);
        if (na == NULL)
        {
            return -1;
        }
        sub->alias_buf = na;
        sub->alias_cap = need_bytes;
    }
    double *acc  = (double *) sub->alias_buf;
    double *a0   = acc;
    double *a1   = acc + nout;
    double *aref = acc + 2 * nout;
    double *acnt = acc + 3 * nout;
    memset(acc, 0, (size_t) (all ? 4 : 1) * nout * sizeof(double));
    const fview wv = {
        .base   = 0,
        .stride = { (axis == 0) ? 0 : 1,
                    (axis == 0)   ? 1
                    : (axis == 1) ? 0
                                  : (int64_t) nx,
                    (axis == 0)   ? (int64_t) ny
                    : (axis == 1) ? (int64_t) nx
                                  : 0 },
    };
    axis_acc_t ac = { sub, &wv, (axis == 0) ? 0 : 1, op, use_flt, a0, a1, aref, acnt };

    axis_job_t job = { ac, axis, twod_ax1, nchunk, rpc, nx, ny, nz, npart, acc, nout, all };
    cli_calc_team_run(nth, nchunk, axis_job, &job);

    axis_fin_t fin = { a0, a1, acnt, op, L, res, (res->md[0].datatype == _DATATYPE_FLOAT) };
    cli_calc_team_run((nout >= 65536) ? nth : 1, nout, axis_fin_job, &fin);
    return 0;
}

#undef AXIS_ACCUM_ROW

/**
 * cli_calc_fuse_run_gauss_prestage - Evaluate and convolve separable Gaussian pre-stage @slot
 * @p:    Parent fused program
 * @slot: Pre-stage slot index
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_run_gauss_prestage(fprog *p, int slot)
{
    fprog *sub = p->ax_p[slot];
    IMAGE *res = &p->ax_im[slot];
    if (cli_calc_fuse_run_program(sub, res) != 0)
    {
        return -1;
    }
    return arith_image_gauss2d_im(res, res, p->ax_c[slot][0], p->ax_c[slot][1], 0, &sub->alias_buf,
                                  &sub->alias_cap);
}

/**
 * cli_calc_fuse_run_cumsum_prestage - Evaluate pre-stage @slot and sum cumulatively along its axis
 * @p:    Parent fused program
 * @slot: Pre-stage slot index (`ax_c[slot][0]` is the axis)
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_run_cumsum_prestage(fprog *p, int slot)
{
    fprog *sub = p->ax_p[slot];
    IMAGE *res = &p->ax_im[slot];
    if (cli_calc_fuse_run_program(sub, res) != 0)
    {
        return -1;
    }
    int axis = (int) llround(p->ax_c[slot][0]);
    if (arith_image_cumsum_inplace_im(res, axis) != 0)
    {
        parse_errmsg("cumsum: axis out of range");
        return -1;
    }
    return 0;
}

/**
 * cli_calc_fuse_run_axes - Execute all registered pre-stage sub-programs of @p
 * @p: Compiled fused program
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_run_axes(fprog *p)
{
    for (int i = 0; i < p->nax; i++)
    {
        int rc = 0;
        switch (p->ax_kind[i])
        {
        case AX_KIND_AXIS:
            rc = run_axis_reduce(p->ax_p[i], p->ax_axis[i], p->ax_op[i], &p->ax_im[i]);
            break;
        case AX_KIND_SEG:
        case AX_KIND_RPROF:
            rc = cli_calc_fuse_run_seg_reduce(p, i);
            break;
        case AX_KIND_GAUSS:
            rc = cli_calc_fuse_run_gauss_prestage(p, i);
            break;
        case AX_KIND_CUMSUM:
            rc = cli_calc_fuse_run_cumsum_prestage(p, i);
            break;
        case AX_KIND_EVAL:
            rc = cli_calc_fuse_run_program(p->ax_p[i], &p->ax_im[i]);
            break;
        case AX_KIND_USER:
            rc = cli_calc_fuse_run_user_prestage(p, i);
            break;
        default:
            rc = -1;
            break;
        }
        if (rc != 0)
        {
            return -1;
        }
    }
    return 0;
}
