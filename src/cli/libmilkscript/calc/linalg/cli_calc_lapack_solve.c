// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_lapack_solve.c
 * @brief Shared LAPACK solver preparation and square solvers (`gesv`, `posv`, `inv`).
 */

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MILK_CMAKE_REQUEST_BLAS
// MILK_CMAKE_REQUEST_LAPACKE
#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "milk_blas_lapacke.h"

/**
 * cli_calc_copy_im_to_dmat - Copy an image into a newly allocated column-major `double` matrix
 * @im:    Source image
 * @rows:  Matrix row count (before optional transpose)
 * @cols:  Matrix column count (before optional transpose)
 * @trans: Non-zero to transpose during copy (producing a `cols x rows` matrix)
 *
 * Return: Newly allocated `double` buffer of `rows * cols` elements, or NULL on failure.
 */
double *cli_calc_copy_im_to_dmat(const IMAGE *im, uint64_t rows, uint64_t cols, int trans)
{
    uint64_t nel = rows * cols;
    double  *buf = (double *) malloc((size_t) (nel > 0 ? nel : 1) * sizeof(double));
    if (!buf)
    {
        return NULL;
    }
    uint8_t dt = im->md[0].datatype;
    if (!trans)
    {
        if (dt == _DATATYPE_FLOAT)
        {
            const float *restrict src = im->array.F;
            for (uint64_t i = 0; i < nel; i++)
            {
                buf[i] = (double) src[i];
            }
        }
        else if (dt == _DATATYPE_DOUBLE)
        {
            __builtin_memcpy(buf, im->array.D, (size_t) nel * sizeof(double));
        }
        else
        {
            for (uint64_t i = 0; i < nel; i++)
            {
                buf[i] = cli_calc_read_im_pix(im, dt, i);
            }
        }
    }
    else
    {
        for (uint64_t r = 0; r < rows; r++)
        {
            for (uint64_t c = 0; c < cols; c++)
            {
                buf[r * cols + c] = cli_calc_read_im_pix(im, dt, c * rows + r);
            }
        }
    }
    return buf;
}

/**
 * cli_calc_write_dmat_to_new_im - Create a temporary output image and populate it from `double`
 * @src:     Source `double` array of `nel` elements
 * @nel:     Total number of elements
 * @naxis:   Output image `naxis` (1, 2, or 3)
 * @outsz:   Output image dimensions `size[3]`
 * @out_dt:  Output datatype (`_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`)
 * @fn_name: Function name for error reporting
 *
 * Return: `val_t` holding the new temporary image name, or empty string on error.
 */
val_t cli_calc_write_dmat_to_new_im(const double  *src,
                                    uint64_t       nel,
                                    long           naxis,
                                    const uint32_t outsz[3],
                                    uint8_t        out_dt,
                                    const char    *fn_name)
{
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[3] = { outsz[0], outsz[1], outsz[2] };
    cli_calc_tmp_image(tmpn, naxis, sz, out_dt, 1, &tid);
    if (tid == -1)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s: failed to allocate output image", fn_name);
        parse_errmsg(msg);
        return mk_string("");
    }
    IMAGE *out = &data.core.image[tid];
    if (out_dt == _DATATYPE_DOUBLE)
    {
        __builtin_memcpy(out->array.D, src, (size_t) nel * sizeof(double));
    }
    else
    {
        float *restrict dst = out->array.F;
        for (uint64_t i = 0; i < nel; i++)
        {
            dst[i] = (float) src[i];
        }
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_resolve_rhs_dims - Validate and extract `(M, NRHS)` and output shape for solver RHS `B`
 */
int cli_calc_resolve_rhs_dims(const IMAGE *imA,
                              int          is_3d_a,
                              uint64_t     M,
                              uint64_t     N,
                              const IMAGE *imB,
                              uint64_t    *out_rhs,
                              long        *out_nax,
                              uint32_t     out_sz[3])
{
    uint64_t b0      = 0;
    uint64_t b1      = 0;
    int      is_3d_b = 0;
    cli_calc_flatten2d(imB, &b0, &b1, &is_3d_b);

    if (is_3d_a && imB->md[0].naxis == 2 && M == (uint64_t) imB->md[0].size[0] * imB->md[0].size[1])
    {
        b0 = M;
        b1 = 1;
    }
    if (b0 != M || M == 0 || N == 0 || b1 == 0)
    {
        return -1;
    }
    *out_rhs  = b1;
    out_sz[0] = (uint32_t) N;
    out_sz[1] = (uint32_t) b1;
    out_sz[2] = 1;
    if (b1 == 1 && (imB->md[0].naxis == 1 || (is_3d_a && imB->md[0].naxis == 2)))
    {
        *out_nax = 1;
    }
    else
    {
        *out_nax = 2;
    }
    (void) imA;
    return 0;
}

/**
 * parse_ab_first_arg - Parse first matrix argument @argA and comma separator for `(A, B)` solvers
 */
static int parse_ab_first_arg(const char *fn, int allow_trans, val_t *argA, int *transA)
{
    char msg[96];
    *transA = 0;
    if (allow_trans)
    {
        if (cli_calc_parse_matmul_arg(argA, transA) != 0)
        {
            return -1;
        }
    }
    else
    {
        *argA = parse_expr(0);
        if (parse_error || eval_error)
        {
            return -1;
        }
    }
    if (cli_calc_cur_func()()->type != TOK_COMMA)
    {
        snprintf(msg, sizeof(msg), "%s: expected ','", fn);
        parse_errmsg(msg);
        return -1;
    }
    cli_calc_advance_func()();
    return 0;
}

/**
 * cli_calc_prepare_ab_solver - Parse and validate `(A, B[, extra])` arguments for solvers
 */
int cli_calc_prepare_ab_solver(const char             *fn,
                               int                     allow_trans,
                               int                     req_square,
                               int                     max_extra,
                               const char             *sq_err,
                               const char             *ex_err,
                               cli_calc_ab_solver_ctx *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    val_t argA;
    int   transA = 0;
    char  msg[96];

    if (parse_ab_first_arg(fn, allow_trans, &argA, &transA) != 0)
    {
        return -1;
    }

    val_t rest[2];
    snprintf(msg, sizeof(msg), "%s: expected '%c'", fn, max_extra > 0 ? ',' : ')');
    int nrest = cli_calc_parse_func_argv(1, 1 + max_extra, rest, msg);
    if (nrest < 0 || cli_calc_reject_complex(&argA, 1) || cli_calc_reject_complex(rest, nrest))
    {
        return -1;
    }
    if (argA.type != VAL_STRING || rest[0].type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(rest[0].sval))
    {
        if (argA.type != VAL_STRING || rest[0].type != VAL_STRING)
        {
            snprintf(msg, sizeof(msg), "%s: A and B must be images", fn);
            parse_errmsg(msg);
        }
        return -1;
    }
    if (nrest == 2)
    {
        if (rest[1].type == VAL_STRING)
        {
            parse_errmsg(ex_err);
            return -1;
        }
        ctx->n_extra      = 1;
        ctx->extra_scalar = to_double(rest[1]);
    }

    ctx->imA    = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    ctx->imB    = &data.core.image[image_ID(rest[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0 = 0;
    uint64_t a1 = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(ctx->imA, &a0, &a1, &is_3d_a);
    if (req_square && (is_3d_a || a0 != a1 || a0 == 0))
    {
        parse_errmsg(sq_err);
        return -1;
    }

    ctx->M = transA ? a1 : a0;
    ctx->N = transA ? a0 : a1;
    if (cli_calc_resolve_rhs_dims(ctx->imA, !req_square && is_3d_a && !transA, ctx->M, ctx->N,
                                  ctx->imB, &ctx->NRHS, &ctx->out_nax, ctx->out_sz) != 0)
    {
        snprintf(msg, sizeof(msg), "%s: incompatible A and B dimensions", fn);
        parse_errmsg(msg);
        return -1;
    }

    ctx->A = cli_calc_copy_im_to_dmat(ctx->imA, a0, a1, transA);
    ctx->B = cli_calc_copy_im_to_dmat(ctx->imB, ctx->M, ctx->NRHS, 0);
    if (!ctx->A || !ctx->B)
    {
        free(ctx->A);
        free(ctx->B);
        snprintf(msg, sizeof(msg), "%s: out of memory", fn);
        parse_errmsg(msg);
        return -1;
    }
    ctx->out_dt = (ctx->imA->md[0].datatype == _DATATYPE_DOUBLE ||
                   ctx->imB->md[0].datatype == _DATATYPE_DOUBLE)
                      ? _DATATYPE_DOUBLE
                      : _DATATYPE_FLOAT;
    return 0;
}

#ifndef HAVE_LAPACKE
/**
 * lu_factor_fallback - In-place LU factorization with partial pivoting when LAPACKE is absent
 */
static int lu_factor_fallback(double *restrict A, uint64_t N, double tol, uint64_t *restrict piv)
{
    for (uint64_t k = 0; k < N; k++)
    {
        uint64_t p                   = k;
        const double *restrict col_k = A + k * N;
        double max_v                 = fabs(col_k[k]);
        for (uint64_t i = k + 1; i < N; i++)
        {
            double v = fabs(col_k[i]);
            if (v > max_v)
            {
                max_v = v;
                p     = i;
            }
        }
        if (max_v <= tol)
        {
            return -1;
        }
        piv[k] = p;
        if (p != k)
        {
            for (uint64_t j = 0; j < N; j++)
            {
                double tmp   = A[j * N + k];
                A[j * N + k] = A[j * N + p];
                A[j * N + p] = tmp;
            }
        }

        double inv_piv = 1.0 / A[k * N + k];
        for (uint64_t i = k + 1; i < N; i++)
        {
            A[k * N + i] *= inv_piv;
        }

        for (uint64_t j = k + 1; j < N; j++)
        {
            double *restrict col_j       = A + j * N;
            const double *restrict col_l = A + k * N;
            double akj                   = col_j[k];
            if (akj != 0.0)
            {
                for (uint64_t i = k + 1; i < N; i++)
                {
                    col_j[i] -= col_l[i] * akj;
                }
            }
        }
    }
    return 0;
}
#endif

/**
 * cli_calc_solve_lu_inplace - Solve general square system $A X = B$ via LU with partial pivoting
 */
int cli_calc_solve_lu_inplace(double *restrict A, uint64_t N, double *restrict X, uint64_t NRHS)
{
    double max_norm = 0.0;
    for (uint64_t i = 0; i < N * N; i++)
    {
        double v = fabs(A[i]);
        if (v > max_norm)
        {
            max_norm = v;
        }
    }
    if (max_norm == 0.0)
    {
        return -1;
    }
    double tol = 1e-14 * max_norm;

#ifdef HAVE_LAPACKE
    lapack_int *ipiv = (lapack_int *) malloc((size_t) N * sizeof(lapack_int));
    if (!ipiv)
    {
        return -1;
    }
    lapack_int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, (lapack_int) N, (lapack_int) NRHS, A,
                                    (lapack_int) N, ipiv, X, (lapack_int) N);
    free(ipiv);
    if (info != 0)
    {
        return -1;
    }
    for (uint64_t k = 0; k < N; k++)
    {
        if (fabs(A[k * N + k]) <= tol)
        {
            return -1;
        }
    }
    return 0;
#else
    uint64_t *piv = (uint64_t *) malloc((size_t) N * sizeof(uint64_t));
    if (!piv)
    {
        return -1;
    }
    if (lu_factor_fallback(A, N, tol, piv) != 0)
    {
        free(piv);
        return -1;
    }
    for (uint64_t k = 0; k < N; k++)
    {
        uint64_t p = piv[k];
        if (p != k)
        {
            for (uint64_t r = 0; r < NRHS; r++)
            {
                double tmp   = X[r * N + k];
                X[r * N + k] = X[r * N + p];
                X[r * N + p] = tmp;
            }
        }
    }
    free(piv);

    if (cli_calc_solve_triangular_inplace(A, N, 0, 0, 1, 1.0, X, NRHS) != 0)
    {
        return -1;
    }
    return cli_calc_solve_triangular_inplace(A, N, 0, 1, 0, 1.0, X, NRHS);
#endif
}

/**
 * cli_calc_solve_cholesky_inplace - Solve SPD system $A X = B$ via Cholesky $A = L L^T$
 */
int cli_calc_solve_cholesky_inplace(double *restrict A,
                                    uint64_t N,
                                    double *restrict X,
                                    uint64_t NRHS)
{
    double max_diag = 0.0;
    for (uint64_t j = 0; j < N; j++)
    {
        double d = A[j * N + j];
        if (d > max_diag)
        {
            max_diag = d;
        }
    }
    if (max_diag <= 0.0)
    {
        return -1;
    }
    double sym_tol = 1e-5 * max_diag;
    for (uint64_t j = 0; j < N; j++)
    {
        for (uint64_t i = j + 1; i < N; i++)
        {
            double aij = A[j * N + i];
            double aji = A[i * N + j];
            if (fabs(aij - aji) > sym_tol)
            {
                return -1;
            }
            A[j * N + i] = 0.5 * (aij + aji);
            A[i * N + j] = 0.0;
        }
    }

#ifdef HAVE_LAPACKE
    lapack_int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (lapack_int) N, A, (lapack_int) N);
    if (info != 0)
    {
        return -1;
    }
    info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', (lapack_int) N, (lapack_int) NRHS, A,
                          (lapack_int) N, X, (lapack_int) N);
    return (info == 0) ? 0 : -1;
#else
    for (uint64_t j = 0; j < N; j++)
    {
        double *restrict col_j = A + j * N;
        double sum             = col_j[j];
        for (uint64_t k = 0; k < j; k++)
        {
            double ljk = A[k * N + j];
            sum -= ljk * ljk;
        }
        if (sum <= 0.0 || cli_calc_isnan(sum) != 0.0)
        {
            return -1;
        }
        double ljj = sqrt(sum);
        col_j[j]   = ljj;
        double inv = 1.0 / ljj;

        for (uint64_t k = 0; k < j; k++)
        {
            const double *restrict col_k = A + k * N;
            double ljk                   = col_k[j];
            if (ljk != 0.0)
            {
                for (uint64_t i = j + 1; i < N; i++)
                {
                    col_j[i] -= col_k[i] * ljk;
                }
            }
        }
        for (uint64_t i = j + 1; i < N; i++)
        {
            col_j[i] *= inv;
        }
    }

    if (cli_calc_solve_triangular_inplace(A, N, 0, 0, 0, 1.0, X, NRHS) != 0)
    {
        return -1;
    }
    return cli_calc_solve_triangular_inplace(A, N, 1, 0, 0, 1.0, X, NRHS);
#endif
}

/**
 * cli_calc_func_gesv - Evaluate `gesv(A, B)` to solve general square system $A X = B$
 */
val_t cli_calc_func_gesv(void)
{
    cli_calc_ab_solver_ctx ctx;
    if (cli_calc_prepare_ab_solver("gesv", 1, 1, 0, "gesv: A must be a square 2D matrix", NULL,
                                   &ctx) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_solve_lu_inplace(ctx.A, ctx.N, ctx.B, ctx.NRHS) != 0)
    {
        free(ctx.A);
        free(ctx.B);
        parse_errmsg("gesv: singular matrix");
        return mk_string("");
    }
    val_t res = cli_calc_write_dmat_to_new_im(ctx.B, ctx.N * ctx.NRHS, ctx.out_nax, ctx.out_sz,
                                              ctx.out_dt, "gesv");
    free(ctx.A);
    free(ctx.B);
    return res;
}

/**
 * cli_calc_func_posv - Evaluate `posv(A, B)` to solve SPD system $A X = B$ via Cholesky
 */
val_t cli_calc_func_posv(void)
{
    cli_calc_ab_solver_ctx ctx;
    if (cli_calc_prepare_ab_solver("posv", 0, 1, 0,
                                   "posv: A must be a square symmetric positive-definite matrix",
                                   NULL, &ctx) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_solve_cholesky_inplace(ctx.A, ctx.N, ctx.B, ctx.NRHS) != 0)
    {
        free(ctx.A);
        free(ctx.B);
        parse_errmsg("posv: matrix is not symmetric positive-definite");
        return mk_string("");
    }
    val_t res = cli_calc_write_dmat_to_new_im(ctx.B, ctx.N * ctx.NRHS, ctx.out_nax, ctx.out_sz,
                                              ctx.out_dt, "posv");
    free(ctx.A);
    free(ctx.B);
    return res;
}

/**
 * cli_calc_func_inv - Evaluate `inv(A)` to compute the inverse $A^{-1}$ of a square matrix
 */
val_t cli_calc_func_inv(void)
{
    val_t argA;
    int   transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_cur_func()()->type != TOK_RPAREN)
    {
        parse_errmsg("inv: expected ')'");
        return mk_string("");
    }
    cli_calc_advance_func()();

    if (cli_calc_reject_complex(&argA, 1))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || !check_image(argA.sval))
    {
        if (argA.type != VAL_STRING)
        {
            parse_errmsg("inv: argument must be an image");
        }
        return mk_string("");
    }

    IMAGE   *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0  = 0;
    uint64_t a1  = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    if (is_3d_a || a0 != a1 || a0 == 0)
    {
        parse_errmsg("inv: A must be a square 2D matrix");
        return mk_string("");
    }

    uint64_t N = a0;
    double  *A = cli_calc_copy_im_to_dmat(imA, N, N, transA);
    double  *X = (double *) calloc((size_t) (N * N), sizeof(double));
    if (!A || !X)
    {
        free(A);
        free(X);
        parse_errmsg("inv: out of memory");
        return mk_string("");
    }
    for (uint64_t i = 0; i < N; i++)
    {
        X[i * N + i] = 1.0;
    }
    if (cli_calc_solve_lu_inplace(A, N, X, N) != 0)
    {
        free(A);
        free(X);
        parse_errmsg("inv: singular matrix");
        return mk_string("");
    }

    uint32_t out_sz[3] = { (uint32_t) N, (uint32_t) N, 1 };
    uint8_t out_dt = (imA->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    val_t   res    = cli_calc_write_dmat_to_new_im(X, N * N, 2, out_sz, out_dt, "inv");
    free(A);
    free(X);
    return res;
}
