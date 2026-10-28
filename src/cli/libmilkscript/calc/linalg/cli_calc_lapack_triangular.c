// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_lapack_triangular.c
 * @brief BLAS triangular solvers (`trsv`, `trsm`) and triangular back/forward substitution.
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
 * detect_uplo - Auto-detect whether a square `N x N` matrix is upper or lower triangular
 */
static int detect_uplo(const double *A, uint64_t N)
{
    double upper_sum = 0.0;
    double lower_sum = 0.0;
    for (uint64_t c = 0; c < N; c++)
    {
        const double *col = A + c * N;
        for (uint64_t r = 0; r < c; r++)
        {
            upper_sum += fabs(col[r]);
        }
        for (uint64_t r = c + 1; r < N; r++)
        {
            lower_sum += fabs(col[r]);
        }
    }
    return (lower_sum <= upper_sum) ? 1 : 0;
}

/**
 * parse_uplo_arg - Parse an optional BLAS `uplo` argument (`"U"`/`"L"` or numeric)
 */
static int parse_uplo_arg(val_t v, int *out_uplo)
{
    if (v.type == VAL_STRING)
    {
        char c = (char) toupper((unsigned char) v.sval[0]);
        if (c == 'U')
        {
            *out_uplo = 1;
            return 0;
        }
        if (c == 'L')
        {
            *out_uplo = 0;
            return 0;
        }
        return -1;
    }
    double d  = to_double(v);
    *out_uplo = (d > 0.0) ? 1 : 0;
    return 0;
}

/**
 * parse_diag_arg - Parse an optional BLAS `diag` argument (`"U"`/`"N"` or numeric)
 */
static int parse_diag_arg(val_t v, int *out_unit)
{
    if (v.type == VAL_STRING)
    {
        char c = (char) toupper((unsigned char) v.sval[0]);
        if (c == 'U')
        {
            *out_unit = 1;
            return 0;
        }
        if (c == 'N')
        {
            *out_unit = 0;
            return 0;
        }
        return -1;
    }
    double d  = to_double(v);
    *out_unit = (d != 0.0) ? 1 : 0;
    return 0;
}

/**
 * solve_tri_vec - Solve $op(A) x = x$ in-place for a single RHS vector @x of length @N
 */
static void solve_tri_vec(const double *restrict A,
                          uint64_t N,
                          int      transA,
                          int      eff_upper,
                          int      is_unit,
                          double *restrict x)
{
    if (!eff_upper)
    {
        if (!transA)
        {
            for (uint64_t j = 0; j < N; j++)
            {
                if (!is_unit)
                {
                    x[j] /= A[j * N + j];
                }
                double xj                  = x[j];
                const double *restrict col = A + j * N;
                if (xj != 0.0)
                {
                    for (uint64_t i = j + 1; i < N; i++)
                    {
                        x[i] -= col[i] * xj;
                    }
                }
            }
        }
        else
        {
            for (uint64_t i = 0; i < N; i++)
            {
                const double *restrict col_i = A + i * N;
                double sum                   = x[i];
                for (uint64_t j = 0; j < i; j++)
                {
                    sum -= col_i[j] * x[j];
                }
                x[i] = is_unit ? sum : (sum / col_i[i]);
            }
        }
    }
    else if (!transA)
    {
        for (uint64_t jj = N; jj > 0; jj--)
        {
            uint64_t j = jj - 1;
            if (!is_unit)
            {
                x[j] /= A[j * N + j];
            }
            double xj                  = x[j];
            const double *restrict col = A + j * N;
            if (xj != 0.0)
            {
                for (uint64_t i = 0; i < j; i++)
                {
                    x[i] -= col[i] * xj;
                }
            }
        }
    }
    else
    {
        for (uint64_t ii = N; ii > 0; ii--)
        {
            uint64_t i                   = ii - 1;
            const double *restrict col_i = A + i * N;
            double sum                   = x[i];
            for (uint64_t j = i + 1; j < N; j++)
            {
                sum -= col_i[j] * x[j];
            }
            x[i] = is_unit ? sum : (sum / col_i[i]);
        }
    }
}

/**
 * cli_calc_solve_triangular_inplace - Solve $op(A) X = \alpha B$ in place in `X` (`N x NRHS`)
 */
int cli_calc_solve_triangular_inplace(const double *restrict A,
                                      uint64_t N,
                                      int      transA,
                                      int      is_upper,
                                      int      is_unit,
                                      double   alpha,
                                      double *restrict X,
                                      uint64_t NRHS)
{
    if (alpha != 1.0)
    {
        uint64_t total = N * NRHS;
        for (uint64_t i = 0; i < total; i++)
        {
            X[i] *= alpha;
        }
    }
    if (!is_unit)
    {
        for (uint64_t i = 0; i < N; i++)
        {
            if (A[i * N + i] == 0.0)
            {
                return -1;
            }
        }
    }

    int eff_upper = transA ? !is_upper : is_upper;
    for (uint64_t rhs = 0; rhs < NRHS; rhs++)
    {
        solve_tri_vec(A, N, transA, eff_upper, is_unit, X + rhs * N);
    }
    return 0;
}

/**
 * parse_trsv_opts - Parse optional `(uplo, diag, alpha)` arguments for `trsv` / `trsm`
 */
static int parse_trsv_opts(const double *A,
                           uint64_t      N,
                           int           nrest,
                           const val_t   rest[4],
                           int          *is_upper,
                           int          *is_unit,
                           double       *alpha)
{
    *is_upper = (nrest >= 2) ? 0 : detect_uplo(A, N);
    if (nrest >= 2 && parse_uplo_arg(rest[1], is_upper) != 0)
    {
        parse_errmsg("trsv: uplo must be \"U\" or \"L\"");
        return -1;
    }
    *is_unit = 0;
    if (nrest >= 3 && parse_diag_arg(rest[2], is_unit) != 0)
    {
        parse_errmsg("trsv: diag must be \"U\" or \"N\"");
        return -1;
    }
    *alpha = 1.0;
    if (nrest >= 4)
    {
        if (rest[3].type == VAL_STRING)
        {
            parse_errmsg("trsm: alpha must be a scalar");
            return -1;
        }
        *alpha = to_double(rest[3]);
    }
    return 0;
}

/**
 * cli_calc_func_trsv - Evaluate `trsv(A, b[, uplo[, diag]])` and `trsm(A, B[, uplo[, diag[, alpha]]])`
 * @is_trsm: Non-zero when invoked as `trsm`, 0 when invoked as `trsv`
 *
 * Return: `val_t` holding the solution image name, or empty string on error.
 */
val_t cli_calc_func_trsv(int is_trsm)
{
    const char *fn = is_trsm ? "trsm" : "trsv";
    val_t       argA;
    int         transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_cur_func()()->type != TOK_COMMA)
    {
        parse_errmsg(is_trsm ? "trsm: expected ','" : "trsv: expected ','");
        return mk_string("");
    }
    cli_calc_advance_func()();

    val_t rest[4];
    int   max_rest = is_trsm ? 4 : 3;
    int   nrest    = cli_calc_parse_func_argv(1, max_rest, rest, "Expected ','");
    if (nrest < 0 || cli_calc_reject_complex(&argA, 1) || cli_calc_reject_complex(rest, nrest))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || rest[0].type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(rest[0].sval))
    {
        if (argA.type != VAL_STRING || rest[0].type != VAL_STRING)
        {
            parse_errmsg(is_trsm ? "trsm: A and B must be images" : "trsv: A and b must be images");
        }
        return mk_string("");
    }

    IMAGE *imA  = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imB  = &data.core.image[image_ID(rest[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0 = 0;
    uint64_t a1 = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    if (is_3d_a || a0 != a1 || a0 == 0)
    {
        parse_errmsg(is_trsm ? "trsm: A must be a square 2D matrix"
                             : "trsv: A must be a square 2D matrix");
        return mk_string("");
    }

    uint64_t N       = a0;
    uint64_t NRHS    = 0;
    long     out_nax = 1;
    uint32_t out_sz[3];
    if (cli_calc_resolve_rhs_dims(imA, 0, N, N, imB, &NRHS, &out_nax, out_sz) != 0)
    {
        parse_errmsg(is_trsm ? "trsm: incompatible A and B dimensions"
                             : "trsv: incompatible A and b dimensions");
        return mk_string("");
    }
    if (!is_trsm && NRHS != 1)
    {
        parse_errmsg("trsv: right-hand side b must be a vector (use trsm for matrix RHS)");
        return mk_string("");
    }

    double *A = cli_calc_copy_im_to_dmat(imA, N, N, 0);
    double *X = cli_calc_copy_im_to_dmat(imB, N, NRHS, 0);
    if (!A || !X)
    {
        free(A);
        free(X);
        parse_errmsg("trsv: out of memory");
        return mk_string("");
    }

    int    is_upper = 0;
    int    is_unit  = 0;
    double alpha    = 1.0;
    if (parse_trsv_opts(A, N, nrest, rest, &is_upper, &is_unit, &alpha) != 0)
    {
        free(A);
        free(X);
        return mk_string("");
    }

    if (cli_calc_solve_triangular_inplace(A, N, transA, is_upper, is_unit, alpha, X, NRHS) != 0)
    {
        free(A);
        free(X);
        parse_errmsg(is_trsm ? "trsm: singular triangular matrix (zero diagonal)"
                             : "trsv: singular triangular matrix (zero diagonal)");
        return mk_string("");
    }

    uint8_t out_dt =
        (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    val_t res = cli_calc_write_dmat_to_new_im(X, N * NRHS, out_nax, out_sz, out_dt, fn);
    free(A);
    free(X);
    return res;
}

/**
 * cli_calc_func_trsv_op - Zero-argument builtin entry point for `trsv(...)`
 */
val_t cli_calc_func_trsv_op(void)
{
    return cli_calc_func_trsv(0);
}

/**
 * cli_calc_func_trsm_op - Zero-argument builtin entry point for `trsm(...)`
 */
val_t cli_calc_func_trsm_op(void)
{
    return cli_calc_func_trsv(1);
}
