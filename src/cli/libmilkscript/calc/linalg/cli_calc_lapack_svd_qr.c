// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_lapack_svd_qr.c
 * @brief Householder QR/LQ least-squares (`gels`), unified ridge solver (`solve`),
 *        and SVD Moore-Penrose pseudo-inverse (`pinv`).
 *
 * Role & Goal:
 *   Implements overdetermined and underdetermined linear least-squares (`gels`),
 *   Tikhonov-regularized and general linear system solving (`solve`), and
 *   SVD-based Moore-Penrose pseudo-inversion (`pinv` with hard or Tikhonov
 *   singular-value filtering).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// MILK_CMAKE_REQUEST_BLAS
// MILK_CMAKE_REQUEST_LAPACKE
#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "cli_calc_internal.h"
#include "milk_blas_lapacke.h"

#ifndef HAVE_LAPACKE
/**
 * householder_qr_cols - Compute Householder QR factorization of `rows x cols` matrix @A in place
 */
static int householder_qr_cols(double *restrict A,
                               uint64_t rows,
                               uint64_t cols,
                               double *restrict r_diag,
                               double *restrict beta)
{
    double max_r = 0.0;
    for (uint64_t k = 0; k < cols; k++)
    {
        double *restrict col_k = A + k * rows;
        double norm2           = 0.0;
        for (uint64_t i = k; i < rows; i++)
        {
            norm2 += col_k[i] * col_k[i];
        }
        double sigma = sqrt(norm2);
        if (sigma == 0.0)
        {
            return -1;
        }
        double akk    = col_k[k];
        double alpha  = (akk >= 0.0) ? -sigma : sigma;
        double vk     = akk - alpha;
        double beta_k = -1.0 / (alpha * vk);
        col_k[k]      = vk;
        r_diag[k]     = alpha;
        beta[k]       = beta_k;
        if (fabs(alpha) > max_r)
        {
            max_r = fabs(alpha);
        }

        for (uint64_t j = k + 1; j < cols; j++)
        {
            double *restrict col_j = A + j * rows;
            double dot             = 0.0;
            for (uint64_t i = k; i < rows; i++)
            {
                dot += col_k[i] * col_j[i];
            }
            double s = beta_k * dot;
            for (uint64_t i = k; i < rows; i++)
            {
                col_j[i] -= s * col_k[i];
            }
        }
    }

    double tol = 1e-13 * max_r;
    for (uint64_t k = 0; k < cols; k++)
    {
        if (fabs(r_diag[k]) <= tol)
        {
            return -1;
        }
    }
    return 0;
}

/**
 * solve_qr_overdet - Householder QR least-squares fallback when $M \ge N$
 */
static int solve_qr_overdet(double *restrict A,
                            uint64_t M,
                            uint64_t N,
                            const double *restrict B,
                            uint64_t NRHS,
                            double *restrict X)
{
    double *W      = (double *) malloc((size_t) (M * NRHS) * sizeof(double));
    double *r_diag = (double *) malloc((size_t) N * sizeof(double));
    double *beta   = (double *) malloc((size_t) N * sizeof(double));
    if (!W || !r_diag || !beta)
    {
        free(W);
        free(r_diag);
        free(beta);
        return -1;
    }
    __builtin_memcpy(W, B, (size_t) (M * NRHS) * sizeof(double));

    if (householder_qr_cols(A, M, N, r_diag, beta) != 0)
    {
        free(W);
        free(r_diag);
        free(beta);
        return -1;
    }
    for (uint64_t k = 0; k < N; k++)
    {
        const double *restrict col_k = A + k * M;
        double beta_k                = beta[k];
        for (uint64_t r = 0; r < NRHS; r++)
        {
            double *restrict col_w = W + r * M;
            double dot             = 0.0;
            for (uint64_t i = k; i < M; i++)
            {
                dot += col_k[i] * col_w[i];
            }
            double s = beta_k * dot;
            for (uint64_t i = k; i < M; i++)
            {
                col_w[i] -= s * col_k[i];
            }
        }
    }

    for (uint64_t r = 0; r < NRHS; r++)
    {
        const double *restrict w = W + r * M;
        double *restrict x       = X + r * N;
        for (uint64_t i = 0; i < N; i++)
        {
            x[i] = w[i];
        }
        for (uint64_t jj = N; jj > 0; jj--)
        {
            uint64_t j                   = jj - 1;
            const double *restrict col_j = A + j * M;
            x[j] /= r_diag[j];
            double xj = x[j];
            for (uint64_t i = 0; i < j; i++)
            {
                x[i] -= col_j[i] * xj;
            }
        }
    }

    free(W);
    free(r_diag);
    free(beta);
    return 0;
}

/**
 * solve_qr_underdet - Minimum 2-norm Householder QR of $A^T$ fallback when $M < N$
 */
static int solve_qr_underdet(const double *restrict A,
                             uint64_t M,
                             uint64_t N,
                             const double *restrict B,
                             uint64_t NRHS,
                             double *restrict X)
{
    double *AT     = (double *) malloc((size_t) (N * M) * sizeof(double));
    double *r_diag = (double *) malloc((size_t) M * sizeof(double));
    double *beta   = (double *) malloc((size_t) M * sizeof(double));
    if (!AT || !r_diag || !beta)
    {
        free(AT);
        free(r_diag);
        free(beta);
        return -1;
    }
    for (uint64_t r = 0; r < M; r++)
    {
        for (uint64_t c = 0; c < N; c++)
        {
            AT[r * N + c] = A[c * M + r];
        }
    }

    if (householder_qr_cols(AT, N, M, r_diag, beta) != 0)
    {
        free(AT);
        free(r_diag);
        free(beta);
        return -1;
    }

    for (uint64_t r = 0; r < NRHS; r++)
    {
        const double *restrict b = B + r * M;
        double *restrict x       = X + r * N;
        for (uint64_t i = 0; i < M; i++)
        {
            const double *restrict col_i = AT + i * N;
            double sum                   = b[i];
            for (uint64_t j = 0; j < i; j++)
            {
                sum -= col_i[j] * x[j];
            }
            x[i] = sum / r_diag[i];
        }
        for (uint64_t i = M; i < N; i++)
        {
            x[i] = 0.0;
        }
        for (uint64_t kk = M; kk > 0; kk--)
        {
            uint64_t k                   = kk - 1;
            const double *restrict col_k = AT + k * N;
            double beta_k                = beta[k];
            double dot                   = 0.0;
            for (uint64_t i = k; i < N; i++)
            {
                dot += col_k[i] * x[i];
            }
            double s = beta_k * dot;
            for (uint64_t i = k; i < N; i++)
            {
                x[i] -= s * col_k[i];
            }
        }
    }

    free(AT);
    free(r_diag);
    free(beta);
    return 0;
}
#endif

/**
 * solve_qr_ls - Solve least-squares or minimum-norm system $\min \|A X - B\|_F$ via QR/LQ
 */
static int solve_qr_ls(double *restrict A,
                       uint64_t M,
                       uint64_t N,
                       const double *restrict B,
                       uint64_t NRHS,
                       double *restrict X)
{
#ifdef HAVE_LAPACKE
    uint64_t ldb = (M >= N) ? M : N;
    uint64_t mn  = (M <= N) ? M : N;
    double  *W   = (double *) calloc((size_t) (ldb * NRHS), sizeof(double));
    if (!W)
    {
        return -1;
    }
    for (uint64_t r = 0; r < NRHS; r++)
    {
        __builtin_memcpy(W + r * ldb, B + r * M, (size_t) M * sizeof(double));
    }
    lapack_int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', (lapack_int) M, (lapack_int) N,
                                    (lapack_int) NRHS, A, (lapack_int) M, W, (lapack_int) ldb);
    if (info != 0)
    {
        free(W);
        return -1;
    }
    double max_r = 0.0;
    for (uint64_t k = 0; k < mn; k++)
    {
        double d = fabs(A[k * M + k]);
        if (d > max_r)
        {
            max_r = d;
        }
    }
    double tol = 1e-13 * max_r;
    if (max_r == 0.0)
    {
        free(W);
        return -1;
    }
    for (uint64_t k = 0; k < mn; k++)
    {
        if (fabs(A[k * M + k]) <= tol)
        {
            free(W);
            return -1;
        }
    }
    for (uint64_t r = 0; r < NRHS; r++)
    {
        __builtin_memcpy(X + r * N, W + r * ldb, (size_t) N * sizeof(double));
    }
    free(W);
    return 0;
#else
    return (M >= N) ? solve_qr_overdet(A, M, N, B, NRHS, X)
                    : solve_qr_underdet(A, M, N, B, NRHS, X);
#endif
}

/**
 * solve_ridge - Solve Tikhonov-regularized system $\min \|A X - B\|_F^2 + \lambda \|X\|_F^2$
 */
static int solve_ridge(const double *restrict A,
                       uint64_t M,
                       uint64_t N,
                       const double *restrict B,
                       uint64_t NRHS,
                       double   reg,
                       double *restrict X)
{
    if (M >= N)
    {
        double *G = (double *) malloc((size_t) (N * N) * sizeof(double));
        if (!G)
        {
            return -1;
        }
        arith_matmul_kernel_d(A, 1, A, 0, 1.0, 0.0, G, N, M, N);
        for (uint64_t j = 0; j < N; j++)
        {
            G[j * N + j] += reg;
        }
        arith_matmul_kernel_d(A, 1, B, 0, 1.0, 0.0, X, N, M, NRHS);
        int rc = (reg > 0.0) ? cli_calc_solve_cholesky_inplace(G, N, X, NRHS)
                             : cli_calc_solve_lu_inplace(G, N, X, NRHS);
        free(G);
        return rc;
    }

    double *G = (double *) malloc((size_t) (M * M) * sizeof(double));
    double *Y = (double *) malloc((size_t) (M * NRHS) * sizeof(double));
    if (!G || !Y)
    {
        free(G);
        free(Y);
        return -1;
    }
    __builtin_memcpy(Y, B, (size_t) (M * NRHS) * sizeof(double));
    arith_matmul_kernel_d(A, 0, A, 1, 1.0, 0.0, G, M, N, M);
    for (uint64_t j = 0; j < M; j++)
    {
        G[j * M + j] += reg;
    }
    int rc = (reg > 0.0) ? cli_calc_solve_cholesky_inplace(G, M, Y, NRHS)
                         : cli_calc_solve_lu_inplace(G, M, Y, NRHS);
    if (rc == 0)
    {
        arith_matmul_kernel_d(A, 1, Y, 0, 1.0, 0.0, X, N, M, NRHS);
    }
    free(G);
    free(Y);
    return rc;
}

/**
 * cli_calc_func_gels - Evaluate `gels(A, B)` QR/LQ least-squares and minimum-norm solver
 */
val_t cli_calc_func_gels(void)
{
    cli_calc_ab_solver_ctx ctx;
    if (cli_calc_prepare_ab_solver("gels", 1, 0, 0, NULL, NULL, &ctx) != 0)
    {
        return mk_string("");
    }
    double *X = (double *) malloc((size_t) (ctx.N * ctx.NRHS) * sizeof(double));
    if (!X || solve_qr_ls(ctx.A, ctx.M, ctx.N, ctx.B, ctx.NRHS, X) != 0)
    {
        parse_errmsg(X ? "gels: rank-deficient matrix" : "gels: out of memory");
        free(ctx.A);
        free(ctx.B);
        free(X);
        return mk_string("");
    }
    val_t res = cli_calc_write_dmat_to_new_im(X, ctx.N * ctx.NRHS, ctx.out_nax, ctx.out_sz,
                                              ctx.out_dt, "gels");
    free(ctx.A);
    free(ctx.B);
    free(X);
    return res;
}

/**
 * cli_calc_func_solve - Evaluate `solve(A, B[, reg])` unified linear / ridge solver
 */
val_t cli_calc_func_solve(void)
{
    cli_calc_ab_solver_ctx ctx;
    if (cli_calc_prepare_ab_solver("solve", 1, 0, 1, NULL,
                                   "solve: regularization parameter must be a scalar", &ctx) != 0)
    {
        return mk_string("");
    }
    double *X = (double *) malloc((size_t) (ctx.N * ctx.NRHS) * sizeof(double));
    if (!X)
    {
        free(ctx.A);
        free(ctx.B);
        parse_errmsg("solve: out of memory");
        return mk_string("");
    }
    int rc = 0;
    if (ctx.extra_scalar != 0.0)
    {
        rc = solve_ridge(ctx.A, ctx.M, ctx.N, ctx.B, ctx.NRHS, ctx.extra_scalar, X);
    }
    else if (ctx.M == ctx.N)
    {
        __builtin_memcpy(X, ctx.B, (size_t) (ctx.N * ctx.NRHS) * sizeof(double));
        rc = cli_calc_solve_lu_inplace(ctx.A, ctx.N, X, ctx.NRHS);
    }
    else
    {
        rc = solve_qr_ls(ctx.A, ctx.M, ctx.N, ctx.B, ctx.NRHS, X);
    }
    if (rc != 0)
    {
        free(ctx.A);
        free(ctx.B);
        free(X);
        parse_errmsg("solve: singular or rank-deficient matrix");
        return mk_string("");
    }
    val_t res = cli_calc_write_dmat_to_new_im(X, ctx.N * ctx.NRHS, ctx.out_nax, ctx.out_sz,
                                              ctx.out_dt, "solve");
    free(ctx.A);
    free(ctx.B);
    free(X);
    return res;
}

/**
 * jacobi_svd_sweeps - Orthogonalize columns of `P x Q` matrix @W via one-sided Jacobi rotations
 */
static void jacobi_svd_sweeps(double *restrict W, double *restrict V, uint64_t P, uint64_t Q)
{
    const int max_sweeps = 60;
    for (int sweep = 0; sweep < max_sweeps; sweep++)
    {
        int rotated = 0;
        for (uint64_t p = 0; p < Q; p++)
        {
            for (uint64_t q = p + 1; q < Q; q++)
            {
                double *restrict wp = W + p * P;
                double *restrict wq = W + q * P;
                double app          = 0.0;
                double aqq          = 0.0;
                double apq          = 0.0;
                for (uint64_t i = 0; i < P; i++)
                {
                    double wpi = wp[i];
                    double wqi = wq[i];
                    app += wpi * wpi;
                    aqq += wqi * wqi;
                    apq += wpi * wqi;
                }
                if (fabs(apq) <= 1e-14 * sqrt(app * aqq))
                {
                    continue;
                }
                rotated    = 1;
                double tau = (aqq - app) / (2.0 * apq);
                double t   = copysign(1.0 / (fabs(tau) + sqrt(1.0 + tau * tau)), tau);
                double c   = 1.0 / sqrt(1.0 + t * t);
                double s   = c * t;

                for (uint64_t i = 0; i < P; i++)
                {
                    double wpi = wp[i];
                    double wqi = wq[i];
                    wp[i]      = c * wpi - s * wqi;
                    wq[i]      = s * wpi + c * wqi;
                }
                double *restrict vp = V + p * Q;
                double *restrict vq = V + q * Q;
                for (uint64_t i = 0; i < Q; i++)
                {
                    double vpi = vp[i];
                    double vqi = vq[i];
                    vp[i]      = c * vpi - s * vqi;
                    vq[i]      = s * vpi + c * vqi;
                }
            }
        }
        if (!rotated)
        {
            break;
        }
    }
}

/**
 * scale_svd_cols - Scale `rows x Q` columns of @mat by filtered inverse singular values
 */
static void scale_svd_cols(double *restrict mat,
                           uint64_t rows,
                           uint64_t Q,
                           const double *restrict sigma,
                           double sigma_max,
                           double rtol)
{
    double cutoff = fabs(rtol) * sigma_max;
    double lambda = cutoff * cutoff;
    for (uint64_t j = 0; j < Q; j++)
    {
        double sj = sigma[j];
        double wj = 0.0;
        if (rtol >= 0.0)
        {
            if (sj > cutoff && sj > 0.0)
            {
                wj = 1.0 / sj;
            }
        }
        else if (sj > 0.0)
        {
            wj = sj / (sj * sj + lambda);
        }
        double *restrict col = mat + j * rows;
        for (uint64_t i = 0; i < rows; i++)
        {
            col[i] *= wj;
        }
    }
}

/**
 * compute_pinv_svd - Compute Moore-Penrose pseudo-inverse $A^+$ via SVD
 */
static int compute_pinv_svd(const IMAGE *imA,
                            int          transA,
                            uint64_t     M,
                            uint64_t     N,
                            double       rtol,
                            double      *out)
{
    uint64_t a0 = (imA->md[0].naxis == 3) ? (uint64_t) imA->md[0].size[0] * imA->md[0].size[1]
                                          : imA->md[0].size[0];
    uint64_t a1 = (imA->md[0].naxis == 3)   ? imA->md[0].size[2]
                  : (imA->md[0].naxis >= 2) ? imA->md[0].size[1]
                                            : 1;
    uint64_t Q  = (M <= N) ? M : N;

#ifdef HAVE_LAPACKE
    {
        double *A     = cli_calc_copy_im_to_dmat(imA, a0, a1, transA);
        double *U     = (double *) malloc((size_t) (M * Q) * sizeof(double));
        double *VT    = (double *) malloc((size_t) (Q * N) * sizeof(double));
        double *sigma = (double *) malloc((size_t) Q * sizeof(double));
        if (!A || !U || !VT || !sigma)
        {
            free(A);
            free(U);
            free(VT);
            free(sigma);
            return -1;
        }
        lapack_int info =
            LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', (lapack_int) M, (lapack_int) N, A, (lapack_int) M,
                           sigma, U, (lapack_int) M, VT, (lapack_int) Q);
        free(A);
        if (info == 0)
        {
            scale_svd_cols(U, M, Q, sigma, (Q > 0) ? sigma[0] : 0.0, rtol);
            arith_matmul_kernel_d(VT, 1, U, 1, 1.0, 0.0, out, N, Q, M);
            free(U);
            free(VT);
            free(sigma);
            return 0;
        }
        free(U);
        free(VT);
        free(sigma);
    }
#endif

    int      tall  = (M >= N);
    uint64_t P     = tall ? M : N;
    double  *W     = cli_calc_copy_im_to_dmat(imA, a0, a1, tall ? transA : !transA);
    double  *V     = (double *) calloc((size_t) (Q * Q), sizeof(double));
    double  *sigma = (double *) malloc((size_t) Q * sizeof(double));
    if (!W || !V || !sigma)
    {
        free(W);
        free(V);
        free(sigma);
        return -1;
    }
    for (uint64_t j = 0; j < Q; j++)
    {
        V[j * Q + j] = 1.0;
    }

    jacobi_svd_sweeps(W, V, P, Q);

    double sigma_max = 0.0;
    for (uint64_t j = 0; j < Q; j++)
    {
        double *restrict wj = W + j * P;
        double norm2        = 0.0;
        for (uint64_t i = 0; i < P; i++)
        {
            norm2 += wj[i] * wj[i];
        }
        double sj = sqrt(norm2);
        sigma[j]  = sj;
        if (sj > sigma_max)
        {
            sigma_max = sj;
        }
        if (sj > 0.0)
        {
            double inv_s = 1.0 / sj;
            for (uint64_t i = 0; i < P; i++)
            {
                wj[i] *= inv_s;
            }
        }
    }

    scale_svd_cols(V, Q, Q, sigma, sigma_max, rtol);
    arith_matmul_kernel_d(tall ? V : W, 0, tall ? W : V, 1, 1.0, 0.0, out, N, Q, M);
    free(W);
    free(V);
    free(sigma);
    return 0;
}

/**
 * cli_calc_func_pinv - Evaluate `pinv(A[, rtol])` Moore-Penrose pseudo-inverse via SVD
 */
val_t cli_calc_func_pinv(void)
{
    val_t argA;
    int   transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    double rtol = 1e-6;
    if (cli_calc_cur_func()()->type == TOK_COMMA)
    {
        cli_calc_advance_func()();
        val_t arg_tol;
        if (cli_calc_parse_func_argv(1, 1, &arg_tol, "pinv: expected ')'") < 0 ||
            cli_calc_reject_complex(&arg_tol, 1))
        {
            return mk_string("");
        }
        if (arg_tol.type == VAL_STRING)
        {
            parse_errmsg("pinv: rtol must be a scalar");
            return mk_string("");
        }
        rtol = to_double(arg_tol);
    }
    else if (cli_calc_cur_func()()->type == TOK_RPAREN)
    {
        cli_calc_advance_func()();
    }
    else
    {
        parse_errmsg("pinv: expected ')'");
        return mk_string("");
    }

    if (cli_calc_reject_complex(&argA, 1))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || !check_image(argA.sval))
    {
        if (argA.type != VAL_STRING)
        {
            parse_errmsg("pinv: first argument must be an image");
        }
        return mk_string("");
    }

    IMAGE   *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0  = 0;
    uint64_t a1  = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    uint64_t M = transA ? a1 : a0;
    uint64_t N = transA ? a0 : a1;
    if (M == 0 || N == 0)
    {
        parse_errmsg("pinv: empty matrix");
        return mk_string("");
    }

    double *out = (double *) malloc((size_t) (N * M) * sizeof(double));
    if (!out || compute_pinv_svd(imA, transA, M, N, rtol, out) != 0)
    {
        free(out);
        parse_errmsg("pinv: out of memory");
        return mk_string("");
    }

    uint32_t out_sz[3] = { (uint32_t) N, (uint32_t) M, 1 };
    uint8_t out_dt = (imA->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    val_t   res    = cli_calc_write_dmat_to_new_im(out, N * M, 2, out_sz, out_dt, "pinv");
    free(out);
    return res;
}
