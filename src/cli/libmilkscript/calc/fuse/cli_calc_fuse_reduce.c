// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_fuse_reduce.c
 * @brief Full-image reductions, flux-weighted centroiding, and one-shot fused evaluators.
 *
 * Role & Goal:
 *   Evaluates fused whole-image statistical reductions (`cli_calc_fuse_run_reduce`),
 *   flux-weighted centroiding (`cli_calc_fuse_run_centroid`), one-shot fused expression
 *   evaluation (`cli_calc_fuse_expr`), and standalone single-op helpers (`cli_calc_select`,
 *   `cli_calc_eval_binop_im`, `cli_calc_eval_fn_im`, `cli_calc_shift`, etc.).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_fuse_internal.h"

typedef struct
{
    const fprog *p1;
    const fprog *p2;
    reduce_op    op;
    uint64_t     nel;
    uint64_t     blk_sz;
    fred_acc    *out_acc;
} reduce_job_t;

/** REDUCE_JOB - team job of one element type: blocks [lo, hi) into a local accumulator, then merge. */
#define REDUCE_JOB(NAME, FT, FK_DT, EVAL, REDUCE)                                                   \
    static void NAME(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)                         \
    {                                                                                               \
        const reduce_job_t *j   = (const reduce_job_t *) ctx;                                       \
        uint64_t            bsz = j->blk_sz;                                                        \
        MILK_ALIGNED(64) FT stk1[FUSE_MAXSTK][FUSE_BLK];                                            \
        MILK_ALIGNED(64) FT stk2[FUSE_MAXSTK][FUSE_BLK];                                            \
        fred_acc            loc   = { 0, 0.0, 0.0, 0.0, NAN, NAN };                                 \
        const FT           *base1 = (j->p1->n == 1 && j->p1->ins[0].k == FI_IMG &&                  \
                                     !j->p1->view[j->p1->ins[0].img].is_view &&                     \
                                     j->p1->img[j->p1->ins[0].img]->md[0].datatype == (FK_DT))      \
                                        ? (const FT *) j->p1->img[j->p1->ins[0].img]->array.raw     \
                                        : NULL;                                                     \
        const FT           *base2 = (j->p2 != NULL && j->p2->n == 1 && j->p2->ins[0].k == FI_IMG && \
                                     !j->p2->view[j->p2->ins[0].img].is_view &&                     \
                                     j->p2->img[j->p2->ins[0].img]->md[0].datatype == (FK_DT))      \
                                        ? (const FT *) j->p2->img[j->p2->ins[0].img]->array.raw     \
                                        : NULL;                                                     \
        (void) tid;                                                                                 \
        (void) nth;                                                                                 \
        for (uint64_t b = lo; b < hi; b++)                                                          \
        {                                                                                           \
            uint64_t  off = b * bsz;                                                                \
            int       len = (int) ((j->nel - off < bsz) ? j->nel - off : bsz);                      \
            const FT *v1  = (base1 != NULL) ? (base1 + off) : stk1[0];                              \
            if (base1 == NULL)                                                                      \
            {                                                                                       \
                EVAL(j->p1, off, len, stk1);                                                        \
            }                                                                                       \
            if (j->p2 != NULL)                                                                      \
            {                                                                                       \
                const FT *v2 = (base2 != NULL) ? (base2 + off) : stk2[0];                           \
                if (base2 == NULL)                                                                  \
                {                                                                                   \
                    EVAL(j->p2, off, len, stk2);                                                    \
                }                                                                                   \
                REDUCE(j->op, v1, v2, len, &loc);                                                   \
            }                                                                                       \
            else                                                                                    \
            {                                                                                       \
                REDUCE(j->op, v1, NULL, len, &loc);                                                 \
            }                                                                                       \
        }                                                                                           \
        cli_calc_team_lock();                                                                       \
        fred_merge(j->out_acc, &loc);                                                               \
        cli_calc_team_unlock();                                                                     \
    }

REDUCE_JOB(reduce_job_f, float, _DATATYPE_FLOAT, cli_calc_fuse_eval_block_f, reduce_block_f)
REDUCE_JOB(reduce_job_d, double, _DATATYPE_DOUBLE, cli_calc_fuse_eval_block_d, reduce_block_d)
#undef REDUCE_JOB

typedef struct
{
    const fprog *sub;
    reduce_op    op;
    uint64_t     nel;
    double       best_v;
    int64_t      best_i;
} argext_job_t;

/** ARGEXT_JOB - team job of one element type: the first extremum of blocks [lo, hi), merged under the lock. */
#define ARGEXT_JOB(NAME, FT, EVAL)                                                               \
    static void NAME(void *ctx, int tid, int nth, uint64_t lo, uint64_t hi)                      \
    {                                                                                            \
        argext_job_t       *j = (argext_job_t *) ctx;                                            \
        MILK_ALIGNED(64) FT stk[FUSE_MAXSTK][FUSE_BLK];                                          \
        double              bv = 0.0;                                                            \
        int64_t             bi = -1;                                                             \
        (void) tid;                                                                              \
        (void) nth;                                                                              \
        for (uint64_t b = lo; b < hi; b++)                                                       \
        {                                                                                        \
            uint64_t off = b * FUSE_BLK;                                                         \
            int      len = (int) ((j->nel - off < FUSE_BLK) ? j->nel - off : FUSE_BLK);          \
            EVAL(j->sub, off, len, stk);                                                         \
            for (int q = 0; q < len; q++)                                                        \
            {                                                                                    \
                double raw = (double) stk[0][q];                                                 \
                if (arith_isnan_d(raw))                                                          \
                {                                                                                \
                    continue;                                                                    \
                }                                                                                \
                double v = (j->op == RED_AMAX) ? fabs(raw) : (j->op == RED_ARGMIN) ? -raw : raw; \
                if (bi < 0 || v > bv)                                                            \
                {                                                                                \
                    bv = v;                                                                      \
                    bi = (int64_t) (off + (uint64_t) q);                                         \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
        cli_calc_team_lock();                                                                    \
        if (bi >= 0 && (j->best_i < 0 || bv > j->best_v || (bv == j->best_v && bi < j->best_i))) \
        {                                                                                        \
            j->best_v = bv;                                                                      \
            j->best_i = bi;                                                                      \
        }                                                                                        \
        cli_calc_team_unlock();                                                                  \
    }

ARGEXT_JOB(argext_job_f, float, cli_calc_fuse_eval_block_f)
ARGEXT_JOB(argext_job_d, double, cli_calc_fuse_eval_block_d)
#undef ARGEXT_JOB

/**
 * cli_calc_fuse_run_argext - Flat index of the first max, min or max-|x| element of prepared @sub
 * @sub:     Fused program, its scalars, views and pre-stages already evaluated for this run
 * @op:      `RED_ARGMAX`, `RED_ARGMIN` or `RED_AMAX`
 * @out_idx: Output index, -1 when every element is NaN; NaNs are skipped, ties go to the first
 *
 * Return: 0 on success.
 */
int cli_calc_fuse_run_argext(fprog *sub, reduce_op op, int64_t *out_idx)
{
    uint64_t nel  = sub->inel;
    uint64_t nblk = (nel + FUSE_BLK - 1) / FUSE_BLK;
    int      nth  = milk_omp_team_size(fprog_effective_nel(sub), FUSE_OMP_MIN);
    if ((uint64_t) nth > nblk)
    {
        nth = (int) nblk;
    }
    argext_job_t job = { sub, op, nel, 0.0, -1 };
    cli_calc_team_run(nth, nblk, cli_calc_fuse_all_float_prog(sub) ? argext_job_f : argext_job_d,
                      &job);
    *out_idx = job.best_i;
    return 0;
}

/**
 * cli_calc_fuse_run_reduce - Evaluate one or two fused programs block-by-block into @out_acc
 * @p1:      Primary fused program
 * @p2:      Optional secondary fused program (mask or second dot-product operand)
 * @op:      Reduction operator
 * @out_acc: Output statistical accumulator
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_fuse_run_reduce(fprog *p1, fprog *p2, reduce_op op, fred_acc *out_acc)
{
    double sval[FUSE_MAXINS];
    if (cli_calc_fuse_eval_scalars(p1, sval) != 0 ||
        cli_calc_fuse_refresh_views(p1, sval, NULL) != 0 || cli_calc_fuse_finalize_views(p1) != 0 ||
        cli_calc_fuse_run_axes(p1) != 0 ||
        (p2 != NULL && (cli_calc_fuse_eval_scalars(p2, sval) != 0 ||
                        cli_calc_fuse_refresh_views(p2, sval, NULL) != 0 ||
                        cli_calc_fuse_finalize_views(p2) != 0 ||
                        memcmp(p2->ishape, p1->ishape, sizeof(p1->ishape)) != 0 ||
                        cli_calc_fuse_run_axes(p2) != 0)))
    {
        return -1;
    }
    uint64_t nel     = p1->inel;
    uint64_t nblk    = 0;
    uint64_t blk_sz  = FUSE_BLK;
    uint64_t eff_nel = fprog_effective_nel(p1) + ((p2 != NULL) ? fprog_effective_nel(p2) : 0);
    int      nth     = 1;
    fprog_choose_grid(nel, eff_nel, &nth, &nblk, &blk_sz);
    int has_mm = (p1->nmatmul > 0 || (p2 != NULL && p2->nmatmul > 0));
    if (has_mm)
    {
        arith_blas_set_threads(1);
        if (cli_calc_fuse_prep_matmuls(p1, nth) != 0 ||
            (p2 != NULL && cli_calc_fuse_prep_matmuls(p2, nth) != 0))
        {
            return -1;
        }
    }
    *out_acc = (fred_acc) { 0, 0.0, 0.0, 0.0, NAN, NAN };
    if (red_is_argext(op))
    {
        int64_t idx = -1;
        if (p2 != NULL || cli_calc_fuse_run_argext(p1, op, &idx) != 0)
        {
            return -1;
        }
        out_acc->sum = (double) idx;
        out_acc->n   = 1;
        return 0;
    }

    reduce_job_t job = { p1, p2, op, nel, blk_sz, out_acc };
    int flt = cli_calc_fuse_all_float_prog(p1) && (p2 == NULL || cli_calc_fuse_all_float_prog(p2));
    cli_calc_team_run(nth, nblk, flt ? reduce_job_f : reduce_job_d, &job);
    return 0;
}

/**
 * cli_calc_fuse_reduce - Parse and evaluate a fused reduction call at the current token cursor
 * @op:      Reduction operator
 * @is_long: 1 if the reduction returns `long`, 0 for `double`
 * @out:     Output value
 *
 * Return: 1 if fused reduction succeeded, 0 to fall back to standard evaluator.
 */
int cli_calc_fuse_reduce(reduce_op op, int is_long, val_t *out)
{
    if (parse_mode == 0 || !cli_calc_tokens_have_image())
    {
        return 0;
    }
    cli_token *(*cur)(void) = cli_calc_cur_func();
    cli_token *(*adv)(void) = cli_calc_advance_func();
    fprog *p1               = (fprog *) calloc(1, sizeof(*p1));
    fprog *p2               = NULL;
    fval   v1;
    fval   v2;
    int    pos = cli_calc_tok_pos();

    if (p1 == NULL)
    {
        return 0;
    }
    if (cli_calc_fuse_compile(p1, 0, &v1) != 0 || !v1.vec)
    {
        cli_calc_fuse_free_fprog_children(p1);
        free(p1);
        cli_calc_tok_seek(pos);
        return 0;
    }

    if (op == RED_TRACE)
    {
        if (cur()->type != TOK_RPAREN || cli_calc_fuse_restrict_diag(p1, 0) != 0)
        {
            cli_calc_fuse_free_fprog_children(p1);
            free(p1);
            cli_calc_tok_seek(pos);
            return 0;
        }
        op = RED_SUM;
    }
    else if (op == RED_DOT || cur()->type == TOK_COMMA)
    {
        if (cur()->type != TOK_COMMA || (op != RED_DOT && !axis_op_ok(op)))
        {
            cli_calc_fuse_free_fprog_children(p1);
            free(p1);
            cli_calc_tok_seek(pos);
            return 0;
        }
        adv();
        p2 = (fprog *) calloc(1, sizeof(*p2));
        if (p2 == NULL || cli_calc_fuse_compile(p2, 0, &v2) != 0 || !v2.vec ||
            !cli_calc_fuse_reduce_shapes_match(p1, p2))
        {
            cli_calc_fuse_free_fprog_children(p1);
            free(p1);
            if (p2 != NULL)
            {
                cli_calc_fuse_free_fprog_children(p2);
                free(p2);
            }
            cli_calc_tok_seek(pos);
            return 0;
        }
    }

    if (cur()->type != TOK_RPAREN)
    {
        cli_calc_fuse_free_fprog_children(p1);
        free(p1);
        if (p2 != NULL)
        {
            cli_calc_fuse_free_fprog_children(p2);
            free(p2);
        }
        cli_calc_tok_seek(pos);
        return 0;
    }
    adv();

    fred_acc acc;
    int      rc = cli_calc_fuse_run_reduce(p1, p2, op, &acc);
    cli_calc_fuse_free_fprog_children(p1);
    free(p1);
    if (p2 != NULL)
    {
        cli_calc_fuse_free_fprog_children(p2);
        free(p2);
    }
    if (rc != 0)
    {
        return 0;
    }
    double r = fuse_finish(&acc, op, is_long);
    *out     = is_long ? mk_long((long) r) : mk_double(r);
    return 1;
}

/**
 * cli_calc_fuse_expr - Try to compile and evaluate a multi-op image sub-expression into a temp image
 * @min_prec: Minimum operator precedence
 * @out:      Output value receiving the temporary image name
 *
 * Return: 1 if fused evaluation succeeded, 0 if not fusible, -1 on runtime error.
 */
int cli_calc_fuse_expr(int min_prec, val_t *out)
{
    if (parse_mode == 0 || !cli_calc_tokens_have_image())
    {
        return 0;
    }
    fprog *p = (fprog *) calloc(1, sizeof(*p));
    if (p == NULL)
    {
        return 0;
    }
    fval v;
    int  pos = cli_calc_tok_pos();
    if (cli_calc_fuse_compile(p, min_prec, &v) != 0)
    {
        free(p);
        return 0;
    }
    if (p->nops == 0 || !v.vec || cli_calc_fuse_finalize_views(p) != 0)
    {
        cli_calc_fuse_free_fprog_children(p);
        free(p);
        cli_calc_tok_seek(pos);
        return 0;
    }

    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    uint8_t out_dt   = p->out_dt ? p->out_dt : (p->out_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT);
    cli_calc_tmp_image(tmpn, p->inaxis, p->ishape, out_dt, 1, &tid);
    if (tid == -1)
    {
        cli_calc_fuse_free_fprog_children(p);
        free(p);
        parse_errmsg("Failed to allocate output image");
        return -1;
    }
    int rc = cli_calc_fuse_run_program(p, &data.core.image[tid]);
    cli_calc_fuse_free_fprog_children(p);
    free(p);
    if (rc != 0)
    {
        return -1;
    }
    *out = mk_string(tmpn);
    return 1;
}

static const char *new_output(const IMAGE *ref, int dbl, IMAGE **out)
{
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[3] = { ref->md[0].size[0], ref->md[0].size[1], ref->md[0].size[2] };

    cli_calc_tmp_image(tmpn, ref->md[0].naxis, sz, dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, 1,
                       &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return NULL;
    }
    *out = &data.core.image[tid];
    return tmpn;
}

static const char *new_output_prog(fprog *p, int dbl, IMAGE **out)
{
    if (cli_calc_fuse_finalize_views(p) != 0)
    {
        return NULL;
    }
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, p->inaxis, p->ishape, dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, 1,
                       &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return NULL;
    }
    *out = &data.core.image[tid];
    return tmpn;
}

static int push_image(fprog *p, const IMAGE *im)
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
 * cli_calc_select - Evaluate elementwise `where(c, t, f)` into a new temporary image
 * @c:  Condition image
 * @t:  True-branch image (or NULL for scalar @tc)
 * @tc: True-branch scalar fallback
 * @f:  False-branch image (or NULL for scalar @fc)
 * @fc: False-branch scalar fallback
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_select(const IMAGE *c, const IMAGE *t, double tc, const IMAGE *f, double fc)
{
    static fprog p;
    IMAGE       *out;

    cli_calc_fuse_reset_fprog(&p);
    if (push_image(&p, c) < 0 || (t != NULL && push_image(&p, t) < 0) ||
        (f != NULL && push_image(&p, f) < 0))
    {
        return NULL;
    }
    p.ins[p.n++] =
        (fins) { .k = FI_SEL, .cpos = (t != NULL) | ((f != NULL) << 1), .c = tc, .c2 = fc };
    const char *tmpn = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL)
    {
        cli_calc_fuse_run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_eval_binop_im - Evaluate a single binary operator on real images/scalars
 * @op: Binary operator token type
 * @li: Left image (or NULL for scalar @lc)
 * @lc: Left scalar
 * @ri: Right image (or NULL for scalar @rc)
 * @rc: Right scalar
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_eval_binop_im(cli_token_type op,
                                   const IMAGE   *li,
                                   double         lc,
                                   const IMAGE   *ri,
                                   double         rc)
{
    static fprog p;
    IMAGE       *out;

    cli_calc_fuse_reset_fprog(&p);
    if ((li != NULL && push_image(&p, li) < 0) || (ri != NULL && push_image(&p, ri) < 0))
    {
        return NULL;
    }
    p.ins[p.n++]     = (fins) { .k    = FI_BIN,
                                .op   = op,
                                .cpos = (li == NULL)   ? 1
                                        : (ri == NULL) ? 2
                                                       : 0,
                                .c    = (li == NULL) ? lc : rc };
    const char *tmpn = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL)
    {
        cli_calc_fuse_run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_eval_fn_im - Evaluate a 1-, 2-, or 3-argument math or cast function on real images
 * @nfn: Number of arguments (1, 2, or 3)
 * @fn:  Function pointer
 * @a:   First image (or NULL for scalar @ac)
 * @ac:  First scalar
 * @b:   Second image (or NULL for scalar @bc)
 * @bc:  Second scalar
 * @c2:  Third scalar
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_eval_fn_im(int nfn,
                                double (*fn)(),
                                const IMAGE *a,
                                double       ac,
                                const IMAGE *b,
                                double       bc,
                                double       c2)
{
    static fprog p;
    IMAGE       *out;

    cli_calc_fuse_reset_fprog(&p);
    if ((a != NULL && push_image(&p, a) < 0) || (b != NULL && push_image(&p, b) < 0))
    {
        return NULL;
    }
    if (nfn == 1)
    {
        uint8_t cdt = cli_calc_cast_datatype(fn);
        if (cdt != 0)
        {
            if (cli_calc_fuse_finalize_views(&p) != 0)
            {
                return NULL;
            }
            const char *tmpn = alloc_tmpname();
            imageID     tid  = -1;
            cli_calc_tmp_image(tmpn, p.inaxis, p.ishape, cdt, 1, &tid);
            if (tid == -1)
            {
                parse_errmsg("Failed to allocate output image");
                return NULL;
            }
            cli_calc_fuse_run_program(&p, &data.core.image[tid]);
            return tmpn;
        }
        p.ins[p.n++] = (fins) { .k = FI_FN1, .fn = fn, .fn1 = cli_calc_fuse_fn1_kind(fn) };
    }
    else if (nfn == 2)
    {
        p.ins[p.n++] = (fins) { .k    = FI_FN2,
                                .fn   = fn,
                                .cpos = (a == NULL)   ? 1
                                        : (b == NULL) ? 2
                                                      : 0,
                                .c    = (a == NULL) ? ac : bc };
    }
    else
    {
        p.ins[p.n++] = (fins) { .k = FI_FN3, .fn = fn, .c = ac, .c2 = c2 };
    }
    const char *tmpn = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL)
    {
        cli_calc_fuse_run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_coord - Generate a coordinate or PRNG image matching @ref's dimensions
 * @ref:  Reference image
 * @kind: Coordinate/PRNG selector (`CALC_COORD_*`)
 * @cx:   Center X or seed
 * @cy:   Center Y or seed-valid flag
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_coord(const IMAGE *ref, int kind, double cx, double cy)
{
    IMAGE      *out;
    const char *tmpn = new_output(ref, ref->md[0].datatype == _DATATYPE_DOUBLE, &out);
    if (tmpn != NULL)
    {
        arith_image_fill_coord(out, kind, cx, cy);
    }
    return tmpn;
}

/**
 * cli_calc_shift - Evaluate a standalone 2D edge-clamped pixel shift into a temporary image
 * @im: Source image
 * @dx: X pixel shift
 * @dy: Y pixel shift
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_shift(const IMAGE *im, int64_t dx, int64_t dy)
{
    static fprog p;
    IMAGE       *out;

    cli_calc_fuse_reset_fprog(&p);
    int k = cli_calc_fuse_add_image(&p, im);
    if (k < 0)
    {
        return NULL;
    }
    p.view[k].is_shift = 1;
    p.ins[p.n++]       = (fins) { .k = FI_SHIFT, .img = k, .c = (double) dx, .c2 = (double) dy };
    const char *tmpn   = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL)
    {
        cli_calc_fuse_run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_lut - Evaluate a standalone 1D linear lookup table interpolation into a temporary image
 * @im:  Source image
 * @tbl: 1D lookup table image
 * @x0:  Table origin coordinate
 * @dx:  Table step size
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_lut(const IMAGE *im, const IMAGE *tbl, double x0, double dx)
{
    static fprog p;
    IMAGE       *out;

    cli_calc_fuse_reset_fprog(&p);
    if (push_image(&p, im) < 0)
    {
        return NULL;
    }
    int k = cli_calc_fuse_add_lut_image(&p, tbl);
    if (k < 0)
    {
        return NULL;
    }
    p.ins[p.n++]     = (fins) { .k = FI_LUT, .img = k, .c = x0, .c2 = dx };
    const char *tmpn = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL)
    {
        cli_calc_fuse_run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_medfilt - Evaluate a standalone KxK 2D median filter (`K` in `{3, 5, 7}`)
 * @im:    Source image
 * @ksize: Window size (3, 5, or 7)
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_medfilt(const IMAGE *im, int ksize)
{
    static fprog p;
    IMAGE       *out;

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        parse_errmsg("medfilt: window size must be 3, 5 or 7");
        return NULL;
    }
    cli_calc_fuse_reset_fprog(&p);
    int k = cli_calc_fuse_add_image(&p, im);
    if (k < 0)
    {
        return NULL;
    }
    p.view[k].is_shift = 1;
    p.ins[p.n++]       = (fins) { .k = FI_MEDFILT, .img = k, .cpos = ksize };
    const char *tmpn   = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL && cli_calc_fuse_run_program(&p, out) != 0)
    {
        return NULL;
    }
    return tmpn;
}

/**
 * cli_calc_pixremap - Evaluate a standalone index-map gather `pixremap(im, map, fill)`
 * @im:   Source image
 * @map:  Index map image
 * @fill: Out-of-range fill value
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_pixremap(const IMAGE *im, const IMAGE *map, double fill)
{
    static fprog p;
    IMAGE       *out;

    cli_calc_fuse_reset_fprog(&p);
    if (push_image(&p, map) < 0)
    {
        return NULL;
    }
    int k = cli_calc_fuse_add_lut_image(&p, im);
    if (k < 0)
    {
        return NULL;
    }
    if (im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p.out_double = 1;
    }
    p.ins[p.n++]     = (fins) { .k = FI_GATHER, .img = k, .c = fill };
    const char *tmpn = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL)
    {
        cli_calc_fuse_run_program(&p, out);
    }
    return tmpn;
}

/**
 * cli_calc_interp2 - Evaluate a standalone 2D bilinear resampling `interp2(im, xmap, ymap)`
 * @im:   Source 2D image
 * @xmap: X coordinate map image
 * @ymap: Y coordinate map image
 *
 * Return: Temporary image name, or NULL on error.
 */
const char *cli_calc_interp2(const IMAGE *im, const IMAGE *xmap, const IMAGE *ymap)
{
    static fprog p;
    IMAGE       *out;

    cli_calc_fuse_reset_fprog(&p);
    if (push_image(&p, xmap) < 0 || push_image(&p, ymap) < 0)
    {
        return NULL;
    }
    int k = cli_calc_fuse_add_lut_image(&p, im);
    if (k < 0)
    {
        return NULL;
    }
    if (im->md[0].datatype == _DATATYPE_DOUBLE)
    {
        p.out_double = 1;
    }
    p.ins[p.n++]     = (fins) { .k = FI_INTERP2, .img = k };
    const char *tmpn = new_output_prog(&p, p.out_double, &out);
    if (tmpn != NULL)
    {
        cli_calc_fuse_run_program(&p, out);
    }
    return tmpn;
}
