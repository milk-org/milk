// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_linalg_lapack.c
 * @brief LAPACK-level linear solvers, matrix inverses, SVD, and symmetric eigendecomposition
 *        for `libmilkcalc`.
 *
 * This module implements the higher-level matrix decompositions and linear system solvers for
 * the `milk.calc` expression calculator (`libmilkcalc`):
 *
 * - **Triangular solvers (`trsv`, `trsm`)**: solves `op(A) * x = b` or `op(A) * X = alpha * B`
 *   (and right-sided `X * op(A) = alpha * B`) using `cblas_dtrsv()` / `cblas_dtrsm()` when
 *   compiled with BLAS, or an internal forward/backward substitution fallback
 *   (`solve_tri_col_d()`).
 * - **Linear system solvers (`gesv`, `posv`, `gels`, `solve`)**: dispatches square LU
 *   (`LAPACKE_dgesv`), Cholesky (`LAPACKE_dposv`), and QR/LQ least-squares (`LAPACKE_dgels`)
 *   solvers, with an automatic SVD-based minimum-norm solver fallback when compiled without
 *   LAPACKE. Also supports Tikhonov-regularized solving (`solve(A, B, lambda)`), forming the
 *   smaller normal system (`(A^T A + lambda I) X = A^T B` when `M >= N`, or dual
 *   `(A A^T + lambda I) Y = B, X = A^T Y` when `M < N`) and solving via Cholesky.
 * - **Matrix inverse and Moore-Penrose pseudoinverse (`inv`, `pinv`)**: computes exact square
 *   inverses via `gesv(A, I)` and SVD-truncated pseudoinverses `V * diag(1/s_i) * U^T` with
 *   user-supplied or default machine-epsilon `rcond` cutoff.
 * - **Singular value and symmetric eigenvalue decompositions (`svd`, `eigh`)**: supports both
 *   single-LHS value-only calls (`S = svd(A)`, `w = eigh(C)`) and multi-LHS factorizations
 *   (`U, S, Vt = svd(A)`, `w, V = eigh(C)`) via `LAPACKE_dgesdd` / `LAPACKE_dsyevd` with
 *   self-contained one-sided and two-sided Jacobi rotation fallbacks.
 */

// MILK_CMAKE_REQUEST_BLAS
// MILK_CMAKE_REQUEST_LAPACKE

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_blas_lapacke.h"
#include "libmilkdata/milkdata.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/image_ID.h"
#include "linalg/mc_linalg.h"

/* ============================================================================
 * Error Reporting and Matrix Buffer Utilities
 * ============================================================================ */

/**
 * lap_err - Format a diagnostic message into @err (if still empty) and return -1
 * @err:   Destination error buffer (may be NULL)
 * @errsz: Capacity of @err in bytes
 * @fmt:   printf-style format string
 *
 * Preserves the first error recorded during execution by only writing into @err
 * when `err[0] == '\0'`.
 *
 * Return: Always -1
 */
__attribute__((format(printf, 3, 4))) static int lap_err(char       *err,
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
 * unwrap_trans - Follow user-function parameter bindings and strip an outer `transpose(X)`
 * @e:         Operand expression to inspect
 * @out_trans: Output flag set to 1 if an outer `transpose(...)` wrapper was stripped, else 0
 *
 * Return: Underlying operand expression inside `transpose(...)`, or @e (after param resolution)
 */
static const mc_expr_t *unwrap_trans(const mc_expr_t *e, int *out_trans)
{
    *out_trans = 0;
    if (e != NULL && e->kind == MC_EXPR_CALL && e->u.call.fn != NULL &&
        strcmp(e->u.call.fn->name, "transpose") == 0 && e->u.call.nargs == 1)
    {
        *out_trans = 1;
        return e->u.call.args[0];
    }
    return e;
}

/**
 * eval_flag_char - Evaluate a string argument and extract a 1-character uppercase LAPACK option
 * @e:         Expression node expected to evaluate to a single-character option string
 * @c1:        First allowed uppercase option character (e.g. `'U'`, `'N'`, `'L'`)
 * @c2:        Second allowed uppercase option character (e.g. `'L'`, `'U'`, `'R'`)
 * @fn:        Calling function name for diagnostic messages
 * @out_c:     Output uppercase option character (`@c1` or `@c2`)
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Return: 0 on success, -1 if @e is not a string matching @c1 or @c2 (case-insensitive)
 */
static int eval_flag_char(const mc_expr_t *e,
                          char             c1,
                          char             c2,
                          const char      *fn,
                          char            *out_c,
                          char            *err,
                          size_t           errsz)
{
    mc_val_t v  = { 0 };
    int      tr = 0;
    e           = unwrap_trans(e, &tr);
    if (tr || mc_scalar_eval_expr(e, &v, err, errsz) != 0 || v.type != MC_VAL_STRING ||
        v.sval[0] == '\0')
    {
        return lap_err(err, errsz, "%s: expected string flag \"%c\" or \"%c\"", fn, c1, c2);
    }
    char c = (char) toupper((unsigned char) v.sval[0]);
    if (c != c1 && c != c2)
    {
        return lap_err(err, errsz, "%s: flag must be \"%c\" or \"%c\"", fn, c1, c2);
    }
    *out_c = c;
    return 0;
}

/**
 * mat_dims_2d - Compute the effective 2D matrix dimensions `(M, N)` of an image
 * @im:    Input image (1D vector, 2D matrix, or 3D cube treated as `(nx*ny) x nz`)
 * @trans: Non-zero if the logical matrix is transposed (`M` and `N` swapped)
 * @M:     Output row count (axis 0 in column-major convention)
 * @N:     Output column count (axis 1 in column-major convention)
 *
 * Return: void
 */
static void mat_dims_2d(const IMAGE *im, int trans, uint64_t *M, uint64_t *N)
{
    uint64_t r = (im->md[0].naxis == 3)
                     ? ((uint64_t) im->md[0].size[0] * (uint64_t) im->md[0].size[1])
                     : im->md[0].size[0];
    uint64_t c = (im->md[0].naxis == 3)   ? im->md[0].size[2]
                 : (im->md[0].naxis == 2) ? im->md[0].size[1]
                                          : 1;
    *M         = trans ? c : r;
    *N         = trans ? r : c;
}

/**
 * alloc_mat_d - Allocate and populate a column-major `double` matrix buffer from @im
 * @im:    Source image of any real numeric datatype
 * @trans: If non-zero, transpose elements while copying into the `M x N` buffer
 * @M:     Row count of the returned column-major buffer
 * @N:     Column count of the returned column-major buffer
 *
 * Return: Heap-allocated `double` array of `M * N` elements, or NULL on allocation failure
 */
static double *alloc_mat_d(const IMAGE *im, int trans, uint64_t M, uint64_t N)
{
    double *buf = malloc(sizeof(double) * (size_t) (M * N > 0 ? M * N : 1));
    uint8_t dt  = im->md[0].datatype;
    for (uint64_t j = 0; buf != NULL && j < N; j++)
    {
        for (uint64_t i = 0; i < M; i++)
        {
            buf[j * M + i] = arith_read_im_pix(im, dt, trans ? (i * N + j) : (j * M + i));
        }
    }
    return buf;
}

/**
 * write_mat_slot_d - Prepare destination @slot and copy a `double` buffer into it
 * @slot:    Destination program slot to prepare and populate
 * @naxis:   Output image dimensionality (1 or 2)
 * @sz:      Output dimension sizes `[nx, ny, nz]`
 * @pref_dt: Preferred output datatype (`_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`)
 * @buf:     Source column-major `double` buffer of `sz[0] * sz[1] * sz[2]` elements
 * @in_unit: Non-zero when executing inside a persistent FPS compute unit
 * @err:     Error message buffer
 * @errsz:   Capacity of @err in bytes
 *
 * Converts @buf to `float32` if the resolved destination slot datatype is
 * `_DATATYPE_FLOAT`, or copies directly when `_DATATYPE_DOUBLE`, then posts
 * the output image semaphores.
 *
 * Return: 0 on success, -1 on output preparation failure
 */
static int write_mat_slot_d(mc_slot_t     *slot,
                            uint8_t        naxis,
                            const uint32_t sz[3],
                            uint8_t        pref_dt,
                            const double  *buf,
                            int            in_unit,
                            char          *err,
                            size_t         errsz)
{
    uint8_t out_dt = mc_select_ctor_out_dt(slot, pref_dt);
    imageID oid    = mc_out_prepare(slot, naxis, sz, out_dt, in_unit, err, errsz);
    if (oid < 0)
    {
        return -1;
    }
    IMAGE   *out = &milk_data.image[oid];
    uint64_t nel = (uint64_t) sz[0] * (uint64_t) sz[1] * (uint64_t) sz[2];
    if (out_dt == _DATATYPE_DOUBLE)
    {
        memcpy(out->array.D, buf, sizeof(double) * (size_t) nel);
    }
    else
    {
        for (uint64_t i = 0; i < nel; i++)
        {
            out->array.F[i] = (float) buf[i];
        }
    }
    mc_out_post(oid);
    return 0;
}

/* ============================================================================
 * Triangular Solvers (trsv, trsm)
 * ============================================================================ */

/**
 * solve_tri_col_d - Plain-C fallback to solve a single triangular column `op(A) * x = b` in place
 * @N:      Matrix order (number of rows and columns of @A)
 * @A:      Column-major `N x N` triangular matrix
 * @transA: Non-zero to solve with `A^T` instead of `A`
 * @uplo:   `'U'` if @A is upper triangular, `'L'` if lower triangular
 * @diag:   `'U'` if @A is unit triangular (diagonal assumed 1.0), `'N'` otherwise
 * @x:      In/out vector of length @N (enters as RHS `b`, exits as solution `x`)
 *
 * Performs forward substitution when the effective matrix (accounting for @uplo
 * and @transA) is lower triangular, or backward substitution when upper triangular.
 *
 * Return: 0 on success, -1 if a zero diagonal pivot is encountered when `diag != 'U'`
 */
static int solve_tri_col_d(uint64_t N, const double *A, int transA, char uplo, char diag, double *x)
{
    int eff_upper = (uplo == 'U') ? !transA : transA;
    for (uint64_t step = 0; step < N; step++)
    {
        uint64_t i  = eff_upper ? (N - 1 - step) : step;
        double   s  = x[i];
        uint64_t lo = eff_upper ? (i + 1) : 0;
        uint64_t hi = eff_upper ? N : i;
        for (uint64_t j = lo; j < hi; j++)
        {
            s -= (transA ? A[i * N + j] : A[j * N + i]) * x[j];
        }
        if (diag != 'U' && A[i * N + i] == 0.0)
        {
            return -1;
        }
        x[i] = (diag != 'U') ? (s / A[i * N + i]) : s;
    }
    return 0;
}

/**
 * exec_trsv_or_trsm - Execute `trsv(A, b[, uplo, diag])` or `trsm(A, B[, uplo, diag, alpha, side])`
 * @slot:      Destination slot to receive the solved vector or matrix
 * @rhs:       Call expression node for `trsv` or `trsm`
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Evaluates optional flag arguments (`uplo`: `"U"`/`"L"`, `diag`: `"N"`/`"U"`,
 * `alpha`, `side`: `"L"`/`"R"`), validates dimensions, and solves the triangular
 * system via `cblas_dtrsv` / `cblas_dtrsm` (or `solve_tri_col_d()` when built
 * without BLAS).
 *
 * Return: 0 on success, -1 on invalid flags, dimension mismatch, or allocation failure
 */
static int exec_trsv_or_trsm(mc_slot_t       *slot,
                             const mc_expr_t *rhs,
                             imageID          avoid_id,
                             int              in_unit,
                             char            *err,
                             size_t           errsz)
{
    const char      *fn     = rhs->u.call.fn->name;
    int              is_sm  = (strcmp(fn, "trsm") == 0);
    int              nargs  = rhs->u.call.nargs;
    int              transA = 0, transB = 0;
    const mc_expr_t *eA = unwrap_trans(rhs->u.call.args[0], &transA);
    const mc_expr_t *eB = is_sm ? unwrap_trans(rhs->u.call.args[1], &transB) : rhs->u.call.args[1];
    char             uplo  = 'U';
    char             diag  = 'N';
    char             side  = 'L';
    double           alpha = 1.0;

    if ((nargs >= 3 && eval_flag_char(rhs->u.call.args[2], 'U', 'L', fn, &uplo, err, errsz) != 0) ||
        (nargs >= 4 && eval_flag_char(rhs->u.call.args[3], 'N', 'U', fn, &diag, err, errsz) != 0) ||
        (is_sm && nargs >= 5 && mc_eval_scalar_d(rhs->u.call.args[4], &alpha, err, errsz) != 0) ||
        (is_sm && nargs >= 6 &&
         eval_flag_char(rhs->u.call.args[5], 'L', 'R', fn, &side, err, errsz) != 0))
    {
        return -1;
    }

    imageID idA = -1, idB = -1;
    int     tmpA = 0, tmpB = 0, rc = 0;
    if (mc_resolve_or_eval_im(eA, avoid_id, in_unit, &idA, &tmpA, err, errsz) != 0 ||
        mc_resolve_or_eval_im(eB, avoid_id, in_unit, &idB, &tmpB, err, errsz) != 0)
    {
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA];
    const IMAGE *imB = &milk_data.image[idB];
    if (imA->md[0].naxis != 2 || imA->md[0].size[0] != imA->md[0].size[1])
    {
        rc = lap_err(err, errsz, "%s: A must be a 2D square matrix", fn);
        goto done;
    }
    uint64_t K = imA->md[0].size[0], M = 0, N = 1;
    if (!is_sm && (!arith_image_is_1d_vec(imB, &M) || M != K))
    {
        rc = lap_err(err, errsz, "trsv: b must be a 1D vector of length %llu",
                     (unsigned long long) K);
        goto done;
    }
    if (is_sm)
    {
        mat_dims_2d(imB, transB, &M, &N);
        if (((side == 'L') ? M : N) != K)
        {
            rc = lap_err(err, errsz, "trsm: incompatible dimensions (%llu vs %llu)",
                         (unsigned long long) K, (unsigned long long) ((side == 'L') ? M : N));
            goto done;
        }
    }

    double *A_d = alloc_mat_d(imA, 0, K, K);
    double *B_d = alloc_mat_d(imB, transB, M, N);
    if (A_d == NULL || B_d == NULL)
    {
        free(A_d);
        free(B_d);
        rc = lap_err(err, errsz, "%s: out of memory", fn);
        goto done;
    }
    for (uint64_t i = 0; i < M * N; i++)
    {
        B_d[i] *= alpha;
    }

#ifdef HAVE_BLAS
    CBLAS_UPLO      cb_uplo  = (uplo == 'U') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cb_trans = transA ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG      cb_diag  = (diag == 'U') ? CblasUnit : CblasNonUnit;
    CBLAS_SIDE      cb_side  = (side == 'L') ? CblasLeft : CblasRight;
    if (!is_sm)
    {
        cblas_dtrsv(CblasColMajor, cb_uplo, cb_trans, cb_diag, (int) K, A_d, (int) K, B_d, 1);
    }
    else
    {
        cblas_dtrsm(CblasColMajor, cb_side, cb_uplo, cb_trans, cb_diag, (int) M, (int) N, 1.0, A_d,
                    (int) K, B_d, (int) M);
    }
#else
    for (uint64_t j = 0; j < N && rc == 0; j++)
    {
        rc = solve_tri_col_d(K, A_d, transA, uplo, diag, &B_d[j * M]);
    }
    (void) side;
#endif

    uint8_t pref_dt =
        (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    uint32_t out_sz[3] = { (uint32_t) M, (uint32_t) N, 1 };
    if (rc == 0)
    {
        rc = write_mat_slot_d(slot, (!is_sm || (imB->md[0].naxis == 1 && !transB)) ? 1 : 2, out_sz,
                              pref_dt, B_d, in_unit, err, errsz);
    }
    free(A_d);
    free(B_d);

done:
    if (tmpA)
    {
        mc_tmp_release(idA);
    }
    if (tmpB)
    {
        mc_tmp_release(idB);
    }
    return rc;
}

/* ============================================================================
 * SVD, Symmetric Eigensolver, and Linear System Kernels
 * ============================================================================ */

/**
 * jacobi_svd_d - Compute thin SVD `A = U * diag(S) * Vt` via one-sided Jacobi orthogonalization
 * @m:      Number of rows of @A
 * @n:      Number of columns of @A
 * @A:      Input column-major `m x n` matrix
 * @U_out:  Optional output column-major `m x min(m, n)` left singular vectors (may be NULL)
 * @S_out:  Output array of `min(m, n)` singular values sorted in descending order
 * @Vt_out: Optional output column-major `min(m, n) x n` right singular vectors `V^T` (may be NULL)
 *
 * Applies up to 60 sweeps of plane rotations to orthogonalize the columns of
 * `W = A * V`, then normalizes the orthogonal columns to obtain `S` and `U` and
 * sorts the singular triplets in descending order of singular value. Used as a
 * self-contained fallback when LAPACKE is unavailable or fails.
 *
 * Return: 0 on success, -1 on memory allocation failure
 */
static int jacobi_svd_d(int m, int n, double *A, double *U_out, double *S_out, double *Vt_out)
{
    int     q = (m < n) ? m : n;
    double *V = calloc((size_t) (n * n), sizeof(double));
    double *W = calloc((size_t) (m * n), sizeof(double));
    if (V == NULL || W == NULL)
    {
        free(V);
        free(W);
        return -1;
    }
    memcpy(W, A, sizeof(double) * (size_t) (m * n));
    for (int i = 0; i < n; i++)
    {
        V[i * n + i] = 1.0;
    }

    /* Orthogonalize columns of W = A * V using one-sided Jacobi sweeps */
    for (int sweep = 0; sweep < 60; sweep++)
    {
        double max_off = 0.0;
        for (int p = 0; p < n - 1; p++)
        {
            for (int k = p + 1; k < n; k++)
            {
                double app = 0.0, akk = 0.0, apk = 0.0;
                for (int i = 0; i < m; i++)
                {
                    double wp = W[p * m + i];
                    double wk = W[k * m + i];
                    app += wp * wp;
                    akk += wk * wk;
                    apk += wp * wk;
                }
                max_off = (fabs(apk) > max_off) ? fabs(apk) : max_off;
                if (fabs(apk) <= 1e-14 * sqrt(app * akk))
                {
                    continue;
                }
                double tau = (akk - app) / (2.0 * apk);
                double t   = (tau >= 0.0) ? (1.0 / (tau + hypot(1.0, tau)))
                                          : (-1.0 / (-tau + hypot(1.0, tau)));
                double c   = 1.0 / hypot(1.0, t);
                double s   = t * c;
                for (int i = 0; i < m; i++)
                {
                    double wp    = W[p * m + i];
                    double wk    = W[k * m + i];
                    W[p * m + i] = c * wp - s * wk;
                    W[k * m + i] = s * wp + c * wk;
                }
                for (int i = 0; i < n; i++)
                {
                    double vp    = V[p * n + i];
                    double vk    = V[k * n + i];
                    V[p * n + i] = c * vp - s * vk;
                    V[k * n + i] = s * vp + c * vk;
                }
            }
        }
        if (max_off < 1e-14)
        {
            break;
        }
    }

    /* Extract column norms as singular values and sort in descending order */
    double *nrms = calloc((size_t) n, sizeof(double));
    int    *ord  = calloc((size_t) n, sizeof(int));
    for (int j = 0; j < n; j++)
    {
        double s = 0.0;
        for (int i = 0; i < m; i++)
        {
            s += W[j * m + i] * W[j * m + i];
        }
        nrms[j] = sqrt(s);
        ord[j]  = j;
    }
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (nrms[ord[j]] > nrms[ord[i]])
            {
                int t  = ord[i];
                ord[i] = ord[j];
                ord[j] = t;
            }
        }
    }
    for (int j = 0; j < q; j++)
    {
        int    col = ord[j];
        double sig = nrms[col];
        S_out[j]   = sig;
        for (int i = 0; U_out != NULL && i < m; i++)
        {
            U_out[j * m + i] = (sig > 1e-15) ? (W[col * m + i] / sig) : ((i == j) ? 1.0 : 0.0);
        }
        for (int i = 0; Vt_out != NULL && i < n; i++)
        {
            Vt_out[i * q + j] = V[col * n + i];
        }
    }
    free(nrms);
    free(ord);
    free(V);
    free(W);
    return 0;
}

/**
 * run_svd_d - Compute thin SVD `A = U * diag(S) * Vt` via LAPACKE_dgesdd or Jacobi fallback
 * @m:      Number of rows of @A
 * @n:      Number of columns of @A
 * @A:      Column-major `m x n` matrix (overwritten on exit when LAPACKE is used)
 * @U_out:  Optional output `m x min(m, n)` left singular vectors (NULL for singular values only)
 * @S_out:  Output array of `min(m, n)` singular values in descending order
 * @Vt_out: Optional output `min(m, n) x n` transposed right singular vectors (NULL for S only)
 *
 * Return: 0 on success, -1 on failure
 */
static int run_svd_d(int m, int n, double *A, double *U_out, double *S_out, double *Vt_out)
{
#ifdef HAVE_LAPACKE
    int q = (m < n) ? m : n;
    if ((U_out == NULL && Vt_out == NULL &&
         LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'N', m, n, A, m, S_out, NULL, 1, NULL, 1) == 0) ||
        (U_out != NULL &&
         LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', m, n, A, m, S_out, U_out, m, Vt_out, q) == 0))
    {
        return 0;
    }
#endif
    return jacobi_svd_d(m, n, A, U_out, S_out, Vt_out);
}

/**
 * run_eigh_d - Compute eigenvalues (and optionally eigenvectors) of a real symmetric matrix
 * @n:      Order of the square matrix @C
 * @C:      In/out `n x n` column-major symmetric matrix; overwritten with eigenvectors if @want_v
 * @want_v: Non-zero to compute orthonormal eigenvectors into @C
 * @w_out:  Output array of @n eigenvalues sorted in ascending order
 *
 * Symmetrizes @C in place (`0.5 * (C + C^T)`) before solving via `LAPACKE_dsyevd`,
 * falling back to classical two-sided Jacobi diagonalization if LAPACKE is unavailable.
 *
 * Return: 0 on success, -1 on allocation failure
 */
static int run_eigh_d(int n, double *C, int want_v, double *w_out)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = j + 1; i < n; i++)
        {
            C[j * n + i] = C[i * n + j] = 0.5 * (C[j * n + i] + C[i * n + j]);
        }
    }
#ifdef HAVE_LAPACKE
    if (LAPACKE_dsyevd(LAPACK_COL_MAJOR, want_v ? 'V' : 'N', 'L', n, C, n, w_out) == 0)
    {
        return 0;
    }
#endif
    double *V = calloc((size_t) (n * n), sizeof(double));
    if (V == NULL)
    {
        return -1;
    }
    for (int i = 0; i < n; i++)
    {
        V[i * n + i] = 1.0;
    }
    for (int sweep = 0; sweep < 60; sweep++)
    {
        double max_off = 0.0;
        for (int p = 0; p < n - 1; p++)
        {
            for (int q = p + 1; q < n; q++)
            {
                double apq = C[q * n + p];
                max_off    = (fabs(apq) > max_off) ? fabs(apq) : max_off;
                if (fabs(apq) < 1e-14)
                {
                    continue;
                }
                double tau = (C[q * n + q] - C[p * n + p]) / (2.0 * apq);
                double t   = (tau >= 0.0) ? (1.0 / (tau + hypot(1.0, tau)))
                                          : (-1.0 / (-tau + hypot(1.0, tau)));
                double c   = 1.0 / hypot(1.0, t);
                double s   = t * c;
                for (int k = 0; k < n; k++)
                {
                    double ckp   = C[p * n + k];
                    double ckq   = C[q * n + k];
                    C[p * n + k] = c * ckp - s * ckq;
                    C[q * n + k] = s * ckp + c * ckq;
                }
                for (int k = 0; k < n; k++)
                {
                    double cpk   = C[k * n + p];
                    double cqk   = C[k * n + q];
                    C[k * n + p] = c * cpk - s * cqk;
                    C[k * n + q] = s * cpk + c * cqk;
                    double vkp   = V[p * n + k];
                    double vkq   = V[q * n + k];
                    V[p * n + k] = c * vkp - s * vkq;
                    V[q * n + k] = s * vkp + c * vkq;
                }
            }
        }
        if (max_off < 1e-14)
        {
            break;
        }
    }
    for (int i = 0; i < n; i++)
    {
        w_out[i] = C[i * n + i];
    }
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (w_out[j] < w_out[i])
            {
                double tw = w_out[i];
                w_out[i]  = w_out[j];
                w_out[j]  = tw;
                for (int k = 0; k < n; k++)
                {
                    double tv    = V[i * n + k];
                    V[i * n + k] = V[j * n + k];
                    V[j * n + k] = tv;
                }
            }
        }
    }
    if (want_v)
    {
        memcpy(C, V, sizeof(double) * (size_t) (n * n));
    }
    free(V);
    return 0;
}

/**
 * solve_linear_d - Solve `A * X = B` via Cholesky (`posv`), LU (`gesv`), QR/LQ (`gels`), or SVD
 * @fn:    Solver name (`"posv"`, `"gesv"`, `"gels"`, or `"solve"`)
 * @uplo:  Triangle selector (`'L'` or `'U'`) for `"posv"`
 * @M:     Row count of @A (and @B)
 * @N:     Column count of @A (and row count of @X_out)
 * @NRHS:  Number of right-hand-side columns in @B
 * @A:     Column-major `M x N` coefficient matrix (modified in place)
 * @B:     Column-major `M x NRHS` right-hand-side matrix (modified in place)
 * @X_out: Output column-major `N x NRHS` solution matrix
 * @err:   Error message buffer
 * @errsz: Capacity of @err in bytes
 *
 * When built with `HAVE_LAPACKE`, dispatches to `LAPACKE_dposv`, `LAPACKE_dgesv`,
 * or `LAPACKE_dgels`. Otherwise falls back to a truncated SVD pseudoinverse solve.
 *
 * Return: 0 on success, -1 if @A is singular, non-SPD, rank-deficient, or out of memory
 */
static int solve_linear_d(const char *fn,
                          char        uplo,
                          int         M,
                          int         N,
                          int         NRHS,
                          double     *A,
                          double     *B,
                          double     *X_out,
                          char       *err,
                          size_t      errsz)
{
#ifdef HAVE_LAPACKE
    if (strcmp(fn, "posv") == 0)
    {
        if (LAPACKE_dposv(LAPACK_COL_MAJOR, uplo, N, NRHS, A, N, B, N) != 0)
        {
            return lap_err(err, errsz, "posv: matrix is not positive definite");
        }
        memcpy(X_out, B, sizeof(double) * (size_t) (N * NRHS));
        return 0;
    }
    if (strcmp(fn, "gesv") == 0 || (strcmp(fn, "solve") == 0 && M == N))
    {
        int *ipiv = calloc((size_t) N, sizeof(int));
        int  info = ipiv ? LAPACKE_dgesv(LAPACK_COL_MAJOR, N, NRHS, A, N, ipiv, B, N) : -1;
        free(ipiv);
        if (info != 0)
        {
            return lap_err(err, errsz, "%s: singular matrix", fn);
        }
        memcpy(X_out, B, sizeof(double) * (size_t) (N * NRHS));
        return 0;
    }
    if (strcmp(fn, "gels") == 0 || strcmp(fn, "solve") == 0)
    {
        int     ldb  = (M > N) ? M : N;
        double *Bpad = calloc((size_t) (ldb * NRHS), sizeof(double));
        if (Bpad == NULL)
        {
            return lap_err(err, errsz, "%s: out of memory", fn);
        }
        for (int j = 0; j < NRHS; j++)
        {
            memcpy(&Bpad[j * ldb], &B[j * M], sizeof(double) * (size_t) M);
        }
        int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', M, N, NRHS, A, M, Bpad, ldb);
        for (int j = 0; info == 0 && j < NRHS; j++)
        {
            memcpy(&X_out[j * N], &Bpad[j * ldb], sizeof(double) * (size_t) N);
        }
        free(Bpad);
        return (info == 0) ? 0 : lap_err(err, errsz, "%s: rank-deficient matrix", fn);
    }
#endif
    int     q  = (M < N) ? M : N;
    double *U  = calloc((size_t) (M * q), sizeof(double));
    double *S  = calloc((size_t) q, sizeof(double));
    double *Vt = calloc((size_t) (q * N), sizeof(double));
    if (!U || !S || !Vt || run_svd_d(M, N, A, U, S, Vt) != 0)
    {
        free(U);
        free(S);
        free(Vt);
        return lap_err(err, errsz, "%s: solver failed", fn);
    }
    double tol = 1e-12 * ((M > N) ? M : N) * (S[0] > 0.0 ? S[0] : 1.0);
    memset(X_out, 0, sizeof(double) * (size_t) (N * NRHS));
    for (int r = 0; r < NRHS; r++)
    {
        for (int k = 0; k < q; k++)
        {
            double utb = 0.0;
            for (int i = 0; S[k] > tol && i < M; i++)
            {
                utb += U[k * M + i] * B[r * M + i];
            }
            for (int j = 0; S[k] > tol && j < N; j++)
            {
                X_out[r * N + j] += Vt[j * q + k] * (utb / S[k]);
            }
        }
    }
    free(U);
    free(S);
    free(Vt);
    (void) uplo;
    return 0;
}

/**
 * exec_pinv_or_inv - Execute `inv(A)` or `pinv(A[, rcond])` and write the inverse into @slot
 * @slot:      Destination slot to receive the `N x M` inverse or pseudoinverse
 * @rhs:       Call expression node for `inv` or `pinv`
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * For `inv(A)`, solves `A * X = I` via LU factorization (`solve_linear_d("gesv", ...)`).
 * For `pinv(A[, rcond])`, computes the thin SVD `A = U * diag(S) * Vt`, zeroes singular
 * values at or below `rcond * S[0]` (defaulting to `eps * max(M, N) * S[0]`), scales the
 * rows of `Vt` by `1 / S[k]`, and forms `Out = Vt^T * U^T`.
 *
 * Return: 0 on success, -1 on dimension mismatch, singular matrix, or allocation failure
 */
static int exec_pinv_or_inv(mc_slot_t       *slot,
                            const mc_expr_t *rhs,
                            imageID          avoid_id,
                            int              in_unit,
                            char            *err,
                            size_t           errsz)
{
    const char      *fn     = rhs->u.call.fn->name;
    int              is_inv = (strcmp(fn, "inv") == 0);
    int              transA = 0, tmpA = 0, rc = 0;
    const mc_expr_t *eA    = unwrap_trans(rhs->u.call.args[0], &transA);
    double           rcond = -1.0;
    imageID          idA   = -1;
    if ((!is_inv && rhs->u.call.nargs == 2 &&
         mc_eval_scalar_d(rhs->u.call.args[1], &rcond, err, errsz) != 0) ||
        mc_resolve_or_eval_im(eA, avoid_id, in_unit, &idA, &tmpA, err, errsz) != 0)
    {
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA];
    uint64_t     M = 0, N = 0;
    mat_dims_2d(imA, transA, &M, &N);
    if (is_inv && (imA->md[0].naxis != 2 || M != N))
    {
        rc = lap_err(err, errsz, "inv: expected 2D square matrix");
        goto done;
    }

    double *A_d   = alloc_mat_d(imA, transA, M, N);
    double *Out_d = calloc((size_t) (N * M > 0 ? N * M : 1), sizeof(double));
    if (A_d == NULL || Out_d == NULL)
    {
        free(A_d);
        free(Out_d);
        rc = lap_err(err, errsz, "%s: out of memory", fn);
        goto done;
    }

    if (is_inv)
    {
        double *I_d = calloc((size_t) (N * N > 0 ? N * N : 1), sizeof(double));
        for (uint64_t i = 0; I_d != NULL && i < N; i++)
        {
            I_d[i * N + i] = 1.0;
        }
        rc = (I_d == NULL) ? lap_err(err, errsz, "inv: out of memory")
                           : solve_linear_d("gesv", 'L', (int) N, (int) N, (int) N, A_d, I_d, Out_d,
                                            err, errsz);
        free(I_d);
    }
    else
    {
        int     q  = (M < N) ? (int) M : (int) N;
        double *U  = calloc((size_t) (M * (uint64_t) q), sizeof(double));
        double *S  = calloc((size_t) q, sizeof(double));
        double *Vt = calloc((size_t) ((uint64_t) q * N), sizeof(double));
        if (!U || !S || !Vt || run_svd_d((int) M, (int) N, A_d, U, S, Vt) != 0)
        {
            rc = lap_err(err, errsz, "pinv: SVD failed");
        }
        else
        {
            double cutoff = (rcond >= 0.0)
                                ? (rcond * S[0])
                                : (2.220446049250313e-16 * (double) ((M > N) ? M : N) * S[0]);
            for (int k = 0; k < q; k++)
            {
                double inv_s = (S[k] > cutoff) ? (1.0 / S[k]) : 0.0;
                for (uint64_t i = 0; i < N; i++)
                {
                    Vt[i * (uint64_t) q + (uint64_t) k] *= inv_s;
                }
            }
#ifdef HAVE_BLAS
            cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, (int) N, (int) M, q, 1.0, Vt, q, U,
                        (int) M, 0.0, Out_d, (int) N);
#else
            for (uint64_t j = 0; j < M; j++)
            {
                for (uint64_t i = 0; i < N; i++)
                {
                    for (int k = 0; k < q; k++)
                    {
                        Out_d[j * N + i] +=
                            Vt[i * (uint64_t) q + (uint64_t) k] * U[(uint64_t) k * M + j];
                    }
                }
            }
#endif
        }
        free(U);
        free(S);
        free(Vt);
    }

    if (rc == 0)
    {
        uint32_t out_sz[3] = { (uint32_t) N, (uint32_t) M, 1 };
        rc = write_mat_slot_d(slot, 2, out_sz,
                              (imA->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE
                                                                        : _DATATYPE_FLOAT,
                              Out_d, in_unit, err, errsz);
    }
    free(A_d);
    free(Out_d);

done:
    if (tmpA)
    {
        mc_tmp_release(idA);
    }
    return rc;
}

/* ============================================================================
 * Public LAPACK Execution Entry Points
 * ============================================================================ */

/**
 * mc_exec_lapack_call - Execute a single-LHS LAPACK solver, inverse, SVD, or eigensolver step
 * @prog:      Compiled calculator program holding the slot table
 * @cs:        Constructor step descriptor to execute
 * @avoid_id:  Image ID to avoid clobbering when evaluating operand sub-expressions
 * @in_unit:   Non-zero when executing inside a persistent FPS compute unit
 * @err:       Error message buffer
 * @errsz:     Capacity of @err in bytes
 *
 * Dispatches `trsv`, `trsm`, `inv`, `pinv`, 1-LHS `svd` / `eigh` (returning singular
 * values or eigenvalues only), and linear system solvers `gesv`, `posv`, `gels`, and
 * `solve(A, B[, lambda])`.
 *
 * Return: 0 on success, -1 on error
 */
int mc_exec_lapack_call(mcalc_prog_t   *prog,
                        mc_ctor_step_t *cs,
                        imageID         avoid_id,
                        int             in_unit,
                        char           *err,
                        size_t          errsz)
{
    mc_slot_t       *slot = &prog->slots[cs->slot_idx];
    const mc_expr_t *rhs  = cs->rhs_expr;
    const char      *fn   = rhs->u.call.fn->name;

    if (strcmp(fn, "trsv") == 0 || strcmp(fn, "trsm") == 0)
    {
        return exec_trsv_or_trsm(slot, rhs, avoid_id, in_unit, err, errsz);
    }
    if (strcmp(fn, "inv") == 0 || strcmp(fn, "pinv") == 0)
    {
        return exec_pinv_or_inv(slot, rhs, avoid_id, in_unit, err, errsz);
    }
    if (strcmp(fn, "svd") == 0 || strcmp(fn, "eigh") == 0)
    {
        int     is_eigh = (strcmp(fn, "eigh") == 0);
        int     tmpA = 0, rc = 0;
        imageID idA = -1;
        if (mc_resolve_or_eval_im(rhs->u.call.args[0], avoid_id, in_unit, &idA, &tmpA, err,
                                  errsz) != 0)
        {
            return -1;
        }
        const IMAGE *imA = &milk_data.image[idA];
        uint64_t     M = 0, N = 0;
        mat_dims_2d(imA, 0, &M, &N);
        if (imA->md[0].naxis != 2 || (is_eigh && M != N))
        {
            if (tmpA)
            {
                mc_tmp_release(idA);
            }
            return lap_err(err, errsz, "%s: expected 2D %smatrix", fn, is_eigh ? "square " : "");
        }
        uint64_t Q   = (M < N) ? M : N;
        double  *A_d = alloc_mat_d(imA, 0, M, N);
        double  *S_d = calloc((size_t) (Q > 0 ? Q : 1), sizeof(double));
        rc           = (!A_d || !S_d) ? lap_err(err, errsz, "%s: out of memory", fn)
                       : is_eigh      ? run_eigh_d((int) N, A_d, 0, S_d)
                                      : run_svd_d((int) M, (int) N, A_d, NULL, S_d, NULL);
        if (rc == 0)
        {
            uint32_t sz[3] = { (uint32_t) Q, 1, 1 };
            rc = write_mat_slot_d(slot, 1, sz,
                                  (imA->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE
                                                                            : _DATATYPE_FLOAT,
                                  S_d, in_unit, err, errsz);
        }
        free(A_d);
        free(S_d);
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        return rc;
    }

    /* Linear system solvers: gesv, posv, gels, solve */
    int              transA = 0, transB = 0;
    const mc_expr_t *eA   = unwrap_trans(rhs->u.call.args[0], &transA);
    const mc_expr_t *eB   = unwrap_trans(rhs->u.call.args[1], &transB);
    char             uplo = 'L';
    double           reg  = 0.0;
    if ((strcmp(fn, "posv") == 0 && rhs->u.call.nargs == 3 &&
         eval_flag_char(rhs->u.call.args[2], 'L', 'U', fn, &uplo, err, errsz) != 0) ||
        (strcmp(fn, "solve") == 0 && rhs->u.call.nargs == 3 &&
         mc_eval_scalar_d(rhs->u.call.args[2], &reg, err, errsz) != 0) ||
        reg < 0.0)
    {
        return (reg < 0.0) ? lap_err(err, errsz, "solve: regularization lambda must be >= 0") : -1;
    }

    imageID idA = -1, idB = -1;
    int     tmpA = 0, tmpB = 0, rc = 0;
    if (mc_resolve_or_eval_im(eA, avoid_id, in_unit, &idA, &tmpA, err, errsz) != 0 ||
        mc_resolve_or_eval_im(eB, avoid_id, in_unit, &idB, &tmpB, err, errsz) != 0)
    {
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA];
    const IMAGE *imB = &milk_data.image[idB];
    uint64_t     M = 0, N = 0, MB = 0, NRHS = 1;
    mat_dims_2d(imA, transA, &M, &N);
    mat_dims_2d(imB, transB, &MB, &NRHS);
    if (MB != M || ((strcmp(fn, "gesv") == 0 || strcmp(fn, "posv") == 0) && M != N))
    {
        rc = lap_err(err, errsz, "%s: incompatible matrix dimensions (%llu x %llu vs %llu x %llu)",
                     fn, (unsigned long long) M, (unsigned long long) N, (unsigned long long) MB,
                     (unsigned long long) NRHS);
        goto done_solve;
    }

    double *A_d = alloc_mat_d(imA, transA, M, N);
    double *B_d = alloc_mat_d(imB, transB, M, NRHS);
    double *X_d = calloc((size_t) (N * NRHS > 0 ? N * NRHS : 1), sizeof(double));
    if (!A_d || !B_d || !X_d)
    {
        free(A_d);
        free(B_d);
        free(X_d);
        rc = lap_err(err, errsz, "%s: out of memory", fn);
        goto done_solve;
    }

    /* Tikhonov-regularized solve: form smaller Gram matrix (N x N or M x M) + reg * I */
    if (strcmp(fn, "solve") == 0 && reg > 0.0)
    {
        uint64_t K = (M >= N) ? N : M;
        uint64_t L = (M >= N) ? M : N;
        double  *G = calloc((size_t) (K * K), sizeof(double));
        double  *R = calloc((size_t) (K * NRHS), sizeof(double));
#ifdef HAVE_BLAS
        cblas_dsyrk(CblasColMajor, CblasLower, (M >= N) ? CblasTrans : CblasNoTrans, (int) K,
                    (int) L, 1.0, A_d, (int) M, 0.0, G, (int) K);
        if (M >= N)
        {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) N, (int) NRHS, (int) M, 1.0,
                        A_d, (int) M, B_d, (int) M, 0.0, R, (int) N);
        }
#else
        for (uint64_t j = 0; j < K; j++)
        {
            for (uint64_t i = j; i < K; i++)
            {
                for (uint64_t k = 0; k < L; k++)
                {
                    G[j * K + i] += (M >= N) ? (A_d[i * M + k] * A_d[j * M + k])
                                             : (A_d[k * M + i] * A_d[k * M + j]);
                }
            }
            for (uint64_t r = 0; M >= N && r < NRHS; r++)
            {
                for (uint64_t k = 0; k < M; k++)
                {
                    R[r * N + j] += A_d[j * M + k] * B_d[r * M + k];
                }
            }
        }
#endif
        for (uint64_t i = 0; i < K; i++)
        {
            for (uint64_t j = 0; j < i; j++)
            {
                G[i * K + j] = G[j * K + i];
            }
            G[i * K + i] += reg;
        }
        rc = solve_linear_d("posv", 'L', (int) K, (int) K, (int) NRHS, G, (M >= N) ? R : B_d,
                            (M >= N) ? X_d : R, err, errsz);
        if (rc == 0 && M < N)
        {
#ifdef HAVE_BLAS
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) N, (int) NRHS, (int) M, 1.0,
                        A_d, (int) M, R, (int) M, 0.0, X_d, (int) N);
#else
            for (uint64_t r = 0; r < NRHS; r++)
            {
                for (uint64_t i = 0; i < N; i++)
                {
                    for (uint64_t k = 0; k < M; k++)
                    {
                        X_d[r * N + i] += A_d[i * M + k] * R[r * M + k];
                    }
                }
            }
#endif
        }
        free(G);
        free(R);
    }
    else
    {
        rc = solve_linear_d(fn, uplo, (int) M, (int) N, (int) NRHS, A_d, B_d, X_d, err, errsz);
    }

    if (rc == 0)
    {
        uint8_t pref_dt =
            (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
                ? _DATATYPE_DOUBLE
                : _DATATYPE_FLOAT;
        uint32_t out_sz[3] = { (uint32_t) N, (uint32_t) NRHS, 1 };
        rc = write_mat_slot_d(slot, (imB->md[0].naxis == 1 && !transB) ? 1 : 2, out_sz, pref_dt,
                              X_d, in_unit, err, errsz);
    }
    free(A_d);
    free(B_d);
    free(X_d);

done_solve:
    if (tmpA)
    {
        mc_tmp_release(idA);
    }
    if (tmpB)
    {
        mc_tmp_release(idB);
    }
    return rc;
}

/**
 * mc_exec_multi_svd_or_eigh - Execute a multi-LHS `U, S, Vt = svd(A)` or `w, V = eigh(C)` statement
 * @prog:    Compiled calculator program holding the slot table (may be NULL in one-shot mode)
 * @s:       Multi-assignment statement AST node
 * @in_unit: Non-zero when executing inside a persistent FPS compute unit
 * @err:     Error message buffer
 * @errsz:   Capacity of @err in bytes
 *
 * Evaluates the input matrix, computes the full thin SVD (`U`, `S`, `Vt`) or
 * symmetric eigendecomposition (`w`, `V`), and writes each factor into its
 * corresponding LHS slot.
 *
 * Return: 0 on success, -1 on dimension/type mismatch, decomposition failure, or OOM
 */
int mc_exec_multi_svd_or_eigh(mcalc_prog_t    *prog,
                              const mc_stmt_t *s,
                              int              in_unit,
                              char            *err,
                              size_t           errsz)
{
    const mc_expr_t *rhs     = s->u.multi.rhs[0];
    const char      *fn      = rhs->u.call.fn->name;
    int              is_eigh = (strcmp(fn, "eigh") == 0);
    int              tmpA = 0, rc = 0;
    imageID          idA = -1;

    if (mc_resolve_or_eval_im(rhs->u.call.args[0], -1, in_unit, &idA, &tmpA, err, errsz) != 0)
    {
        return -1;
    }
    const IMAGE *imA = &milk_data.image[idA];
    uint64_t     M = 0, N = 0;
    mat_dims_2d(imA, 0, &M, &N);
    if (imA->md[0].naxis != 2 || (is_eigh && M != N))
    {
        if (tmpA)
        {
            mc_tmp_release(idA);
        }
        return lap_err(err, errsz, "%s: expected 2D %smatrix", fn, is_eigh ? "square " : "");
    }
    uint8_t out_dt = (imA->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    uint64_t Q     = (M < N) ? M : N;
    double  *A_d   = alloc_mat_d(imA, 0, M, N);
    double  *S_d   = calloc((size_t) (Q > 0 ? Q : 1), sizeof(double));
    double  *U_d   = is_eigh ? NULL : calloc((size_t) (M * Q), sizeof(double));
    double  *Vt_d  = is_eigh ? NULL : calloc((size_t) (Q * N), sizeof(double));

    rc = (!A_d || !S_d || (!is_eigh && (!U_d || !Vt_d)))
             ? lap_err(err, errsz, "%s: out of memory", fn)
         : is_eigh ? run_eigh_d((int) N, A_d, 1, S_d)
                   : run_svd_d((int) M, (int) N, A_d, U_d, S_d, Vt_d);

    int            nlhs      = is_eigh ? 2 : 3;
    const double  *bufs[3]   = { is_eigh ? S_d : U_d, is_eigh ? A_d : S_d, Vt_d };
    const uint8_t  nax[3]    = { is_eigh ? 1 : 2, is_eigh ? 2 : 1, 2 };
    const uint32_t szs[3][3] = {
        { (uint32_t) (is_eigh ? N : M), (uint32_t) (is_eigh ? 1 : Q), 1 },
        { (uint32_t) (is_eigh ? N : Q), (uint32_t) (is_eigh ? N : 1), 1 },
        { (uint32_t) Q, (uint32_t) N, 1 },
    };

    for (int j = 0; rc == 0 && j < nlhs; j++)
    {
        const mc_expr_t *lhs      = s->u.multi.lhs[j];
        mc_slot_t        tmp_slot = { 0 };
        mc_slot_t       *sl       = NULL;
        for (int k = 0; prog != NULL && k < prog->nslots && sl == NULL; k++)
        {
            if (strcmp(prog->slots[k].bare, lhs->u.ident.bare) == 0)
            {
                sl = &prog->slots[k];
            }
        }
        if (sl == NULL)
        {
            snprintf(tmp_slot.name, sizeof(tmp_slot.name), "%s", lhs->u.ident.name);
            snprintf(tmp_slot.bare, sizeof(tmp_slot.bare), "%s", lhs->u.ident.bare);
            tmp_slot.loc   = lhs->u.ident.loc;
            tmp_slot.im_id = mc_resolve_or_autoconnect_image(lhs->u.ident.name);
            sl             = &tmp_slot;
        }
        rc = write_mat_slot_d(sl, nax[j], szs[j], out_dt, bufs[j], in_unit, err, errsz);
    }

    free(A_d);
    free(S_d);
    free(U_d);
    free(Vt_d);
    if (tmpA)
    {
        mc_tmp_release(idA);
    }
    return rc;
}
