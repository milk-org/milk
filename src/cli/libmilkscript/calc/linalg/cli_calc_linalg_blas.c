// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_linalg_blas.c
 * @brief Calculator immediate BLAS Level 1-3 builtins (`matmul`, `gemm`, `gemv`,
 *        `ger`, `outer`, `syrk`, `symv`, `axpy`, `scal`).
 */

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
 * cli_calc_func_matmul - Evaluate calculator `matmul(A, B)` 2-argument product
 *
 * Allocates a new temporary product image `op(A) * op(B)` (unwrapping top-level
 * `transpose(A)` and `transpose(B)` calls zero-copy) and populates it via
 * `arith_image_gemm_im()`.
 *
 * Return: `val_t` holding the temporary product image name, or empty string on error.
 */
val_t cli_calc_func_matmul(void)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    val_t args[2];
    int   transA = 0;
    int   transB = 0;

    if (cli_calc_parse_matmul_arg(&args[0], &transA) != 0)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_double(0);
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&args[1], &transB) != 0)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_double(0);
    }
    advance_func();

    if (cli_calc_reject_complex(args, 2))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING)
    {
        parse_errmsg("matmul: expected image arguments");
        return mk_string("");
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_string("");
    }

    IMAGE *im1 = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *im2 = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    int    tA  = 0;
    int    tB  = 0;
    uint64_t M = 0;
    uint64_t K = 0;
    uint64_t N = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (arith_image_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        parse_errmsg("matmul: incompatible image dimensions");
        return mk_string("");
    }

    uint8_t out_dt =
        (im1->md[0].datatype == _DATATYPE_DOUBLE || im2->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, onax, outsz, out_dt, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("matmul: failed to allocate output image");
        return mk_string("");
    }
    IMAGE *out = &data.core.image[tid];
    if (arith_image_gemm_im(im1, transA, im2, transB, 1.0, 0.0, out) != 0)
    {
        parse_errmsg("matmul: memory allocation failed");
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * parse_gemm_args - Parse `(alpha, A, B, beta, C)` arguments for `gemm` or `gemv`
 */
static int parse_gemm_args(int     is_gemv,
                           double *alpha,
                           double *beta,
                           val_t   args[3],
                           int    *transA,
                           int    *transB)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();

    val_t av = parse_expr(0);
    if (parse_error || eval_error || av.type == VAL_STRING)
    {
        parse_errmsg(is_gemv ? "gemv: expected gemv(alpha, A, x, beta, y)"
                             : "gemm: expected gemm(alpha, A, B, beta, C)");
        return -1;
    }
    *alpha = to_double(av);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return -1;
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&args[0], transA) != 0)
    {
        return -1;
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return -1;
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&args[1], transB) != 0)
    {
        return -1;
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg(is_gemv ? "gemv: expected gemv(alpha, A, x, beta, y)"
                             : "gemm: expected gemm(alpha, A, B, beta, C)");
        return -1;
    }
    advance_func();
    val_t bv = parse_expr(0);
    if (parse_error || eval_error || bv.type == VAL_STRING)
    {
        parse_errmsg(is_gemv ? "gemv: beta must be a scalar" : "gemm: beta must be a scalar");
        return -1;
    }
    *beta = to_double(bv);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg(is_gemv ? "gemv: expected destination y after beta"
                             : "gemm: expected destination C after beta");
        return -1;
    }
    advance_func();
    args[2] = parse_expr(0);
    if (parse_error || eval_error)
    {
        return -1;
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return -1;
    }
    advance_func();
    return 0;
}

/**
 * cli_calc_func_gemm - Evaluate strict BLAS `gemm(alpha, A, B, beta, C)` or `gemv(alpha, A, x, beta, y)`
 * @is_gemv: Non-zero when invoked as `gemv(...)` (requires vector output `N == 1`)
 *
 * Updates destination matrix `C` (or vector `y`) in-place with
 * `alpha * op(A) * op(B) + beta * C` and posts output stream updates.
 *
 * Return: `val_t` holding the destination image name, or empty string on error.
 */
val_t cli_calc_func_gemm(int is_gemv)
{
    val_t  args[3];
    int    transA = 0;
    int    transB = 0;
    double alpha  = 0.0;
    double beta   = 0.0;

    if (parse_gemm_args(is_gemv, &alpha, &beta, args, &transA, &transB) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_reject_complex(args, 3))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING)
    {
        parse_errmsg(is_gemv ? "gemv: A, x, and y must be images"
                             : "gemm: A, B, and C must be images");
        return mk_string("");
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval) || !check_image(args[2].sval))
    {
        return mk_string("");
    }

    IMAGE *imA = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imB = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imC = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imC == imA || imC == imB)
    {
        parse_errmsg(is_gemv ? "gemv: destination y must not alias A or x"
                             : "gemm: destination C must not alias A or B");
        return mk_string("");
    }

    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (arith_image_matmul_dims(imA, transA, imB, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        parse_errmsg(is_gemv ? "gemv: incompatible image dimensions"
                             : "gemm: incompatible image dimensions");
        return mk_string("");
    }
    if (is_gemv && N != 1)
    {
        parse_errmsg("gemv: x must be a vector");
        return mk_string("");
    }
    if (arith_image_matmul_check(imA, transA, imB, transB, imC, 0) != 0)
    {
        parse_errmsg(is_gemv ? "gemv: y dimensions or datatype do not match output vector"
                             : "gemm: C dimensions or datatype do not match product shape");
        return mk_string("");
    }

    imC->md[0].write = 1;
    if (arith_image_gemm_im(imA, transA, imB, transB, alpha, beta, imC) != 0)
    {
        imC->md[0].write = 0;
        parse_errmsg("matmul: memory allocation failed");
        return mk_string("");
    }
    cli_calc_post_output(imC);
    return mk_string(imC->name);
}

/**
 * cli_calc_func_outer - Evaluate 2-argument outer product `outer(x, y)` returning `x * y^T`
 *
 * Return: `val_t` holding the temporary `M x N` outer-product image name, or empty string on error.
 */
val_t cli_calc_func_outer(void)
{
    val_t args[2];
    int   nargs = cli_calc_parse_func_argv(2, 2, args, "outer: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, 2))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING || !check_image(args[0].sval) ||
        !check_image(args[1].sval))
    {
        parse_errmsg("outer: expected two 1D vector images");
        return mk_string("");
    }
    IMAGE *imX = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t M = 0;
    uint64_t N = 0;
    if (!arith_image_is_1d_vec(imX, &M) || !arith_image_is_1d_vec(imY, &N))
    {
        parse_errmsg("outer: expected 1D vector arguments");
        return mk_string("");
    }
    uint8_t out_dt =
        (imX->md[0].datatype == _DATATYPE_DOUBLE || imY->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[2] = { (uint32_t) M, (uint32_t) N };
    cli_calc_tmp_image(tmpn, 2, sz, out_dt, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("outer: failed to allocate output image");
        return mk_string("");
    }
    IMAGE *out = &data.core.image[tid];
    if (arith_image_ger_im(imX, imY, 1.0, 0.0, out) != 0)
    {
        parse_errmsg("ger: expected two 1D vector arguments");
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_ger - Evaluate strict BLAS rank-1 update `ger(alpha, x, y, A)`
 *
 * Updates `M x N` matrix @A in-place with `alpha * x * y^T + A`.
 *
 * Return: `val_t` holding the destination matrix @A name, or empty string on error.
 */
val_t cli_calc_func_ger(void)
{
    val_t args[4];
    int   nargs = cli_calc_parse_func_argv(4, 4, args, "ger: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, 4))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING ||
        args[3].type != VAL_STRING || !check_image(args[1].sval) || !check_image(args[2].sval) ||
        !check_image(args[3].sval))
    {
        parse_errmsg("ger: expected ger(alpha, x, y, A)");
        return mk_string("");
    }
    double alpha = to_double(args[0]);
    IMAGE *imX = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imA = &data.core.image[image_ID(args[3].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imA == imX || imA == imY)
    {
        parse_errmsg("ger: destination matrix A must not alias x or y");
        return mk_string("");
    }
    if (arith_image_ger_check(imX, imY, imA) != 0)
    {
        parse_errmsg("ger: expected 1D vectors x (M), y (N) and M x N float/double matrix A");
        return mk_string("");
    }
    imA->md[0].write = 1;
    if (arith_image_ger_im(imX, imY, alpha, 1.0, imA) != 0)
    {
        imA->md[0].write = 0;
        parse_errmsg("ger: expected two 1D vector arguments");
        return mk_string("");
    }
    cli_calc_post_output(imA);
    return mk_string(imA->name);
}

/**
 * cli_calc_func_syrk - Evaluate strict BLAS symmetric rank-k update `syrk(alpha, A, beta, C)`
 *
 * Updates `N x N` symmetric matrix @C in-place with `alpha * A * A^T + beta * C`,
 * or `alpha * A^T * A + beta * C` when @A is wrapped in `transpose(A)`.
 *
 * Return: `val_t` holding the destination matrix @C name, or empty string on error.
 */
val_t cli_calc_func_syrk(void)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    val_t argA;
    val_t argC;
    int   transA = 0;

    val_t av = parse_expr(0);
    if (parse_error || eval_error || av.type == VAL_STRING)
    {
        parse_errmsg("syrk: expected syrk(alpha, A, beta, C)");
        return mk_string("");
    }
    double alpha = to_double(av);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_string("");
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("syrk: expected syrk(alpha, A, beta, C)");
        return mk_string("");
    }
    advance_func();
    val_t bv = parse_expr(0);
    if (parse_error || eval_error || bv.type == VAL_STRING)
    {
        parse_errmsg("syrk: beta must be a scalar");
        return mk_string("");
    }
    double beta = to_double(bv);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("syrk: expected destination C after beta");
        return mk_string("");
    }
    advance_func();
    argC = parse_expr(0);
    if (parse_error || eval_error)
    {
        return mk_string("");
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_string("");
    }
    advance_func();

    if (argA.type != VAL_STRING || argC.type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(argC.sval) || cli_calc_reject_complex(&argA, 1) ||
        cli_calc_reject_complex(&argC, 1))
    {
        parse_errmsg("syrk: A and C must be real matrix images");
        return mk_string("");
    }
    IMAGE *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imC = &data.core.image[image_ID(argC.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imC == imA)
    {
        parse_errmsg("syrk: destination C must not alias A");
        return mk_string("");
    }
    if (arith_image_syrk_check(imA, transA, imC) != 0)
    {
        parse_errmsg("syrk: C must be an N x N float/double matrix");
        return mk_string("");
    }
    imC->md[0].write = 1;
    if (arith_image_syrk_im(imA, transA, alpha, beta, imC) != 0)
    {
        imC->md[0].write = 0;
        return mk_string("");
    }
    cli_calc_post_output(imC);
    return mk_string(imC->name);
}

/**
 * cli_calc_func_symv - Evaluate strict BLAS symmetric matrix-vector product `symv(alpha, A, x, beta, y)`
 *
 * Updates length-`N` vector @y in-place with `alpha * A * x + beta * y` using
 * the upper triangle of symmetric `N x N` matrix @A.
 *
 * Return: `val_t` holding the destination vector @y name, or empty string on error.
 */
val_t cli_calc_func_symv(void)
{
    val_t args[5];
    int   nargs = cli_calc_parse_func_argv(5, 5, args, "symv: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, 5))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING ||
        args[3].type == VAL_STRING || args[4].type != VAL_STRING || !check_image(args[1].sval) ||
        !check_image(args[2].sval) || !check_image(args[4].sval))
    {
        parse_errmsg("symv: expected symv(alpha, A, x, beta, y)");
        return mk_string("");
    }
    double alpha = to_double(args[0]);
    double beta  = to_double(args[3]);
    IMAGE *imA = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imX = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[4].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imY == imA || imY == imX)
    {
        parse_errmsg("symv: destination y must not alias A or x");
        return mk_string("");
    }
    uint64_t N   = imA->md[0].size[0];
    uint64_t lx  = 0;
    uint64_t ly  = 0;
    uint8_t  dty = imY->md[0].datatype;
    if (imA->md[0].naxis != 2 || imA->md[0].size[1] != N || !arith_image_is_1d_vec(imX, &lx) ||
        lx != N || !arith_image_is_1d_vec(imY, &ly) || ly != N ||
        (dty != _DATATYPE_FLOAT && dty != _DATATYPE_DOUBLE))
    {
        parse_errmsg("symv: expected N x N matrix A and length-N vectors x, y (float/double)");
        return mk_string("");
    }
    imY->md[0].write = 1;
    arith_image_symv_im(imA, imX, alpha, beta, imY);
    cli_calc_post_output(imY);
    return mk_string(imY->name);
}

/**
 * cli_calc_func_axpy - Evaluate strict BLAS `axpy(alpha, x, y)` updating `y = alpha * x + y` in-place
 *
 * Return: `val_t` holding the destination image @y name, or empty string on error.
 */
val_t cli_calc_func_axpy(void)
{
    val_t args[3];
    if (cli_calc_parse_func_argv(3, 3, args, "axpy: expected ','") != 3 ||
        cli_calc_reject_complex(args, 3))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING ||
        !check_image(args[1].sval) || !check_image(args[2].sval))
    {
        parse_errmsg("axpy: expected axpy(alpha, x, y)");
        return mk_string("");
    }
    double alpha = to_double(args[0]);
    IMAGE *imX = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imY == imX)
    {
        parse_errmsg("axpy: destination y must not alias x");
        return mk_string("");
    }
    uint64_t nel = imY->md[0].nelement;
    uint8_t  dty = imY->md[0].datatype;
    if (imX->md[0].nelement != nel || (dty != _DATATYPE_FLOAT && dty != _DATATYPE_DOUBLE))
    {
        parse_errmsg("axpy: x and y must have matching element counts and float/double y");
        return mk_string("");
    }
    imY->md[0].write = 1;
    arith_image_axpy_im(alpha, imX, imY);
    cli_calc_post_output(imY);
    return mk_string(imY->name);
}

/**
 * cli_calc_func_scal - Evaluate strict BLAS `scal(alpha, x)` scaling @x in-place by @alpha
 *
 * Return: `val_t` holding the destination image @x name, or empty string on error.
 */
val_t cli_calc_func_scal(void)
{
    val_t args[2];
    if (cli_calc_parse_func_argv(2, 2, args, "scal: expected ','") != 2 ||
        cli_calc_reject_complex(args, 2))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || !check_image(args[1].sval))
    {
        parse_errmsg("scal: expected scal(alpha, x)");
        return mk_string("");
    }
    double  alpha = to_double(args[0]);
    IMAGE  *imX = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint8_t dtx = imX->md[0].datatype;
    if (dtx != _DATATYPE_FLOAT && dtx != _DATATYPE_DOUBLE)
    {
        parse_errmsg("scal: x must be a float or double image");
        return mk_string("");
    }
    imX->md[0].write = 1;
    arith_image_scal_im(alpha, imX);
    cli_calc_post_output(imX);
    return mk_string(imX->name);
}

/**
 * cli_calc_func_gemm_op - Zero-argument builtin entry point for `gemm(alpha, A, B, beta, C)`
 */
val_t cli_calc_func_gemm_op(void)
{
    return cli_calc_func_gemm(0);
}

/**
 * cli_calc_func_gemv_op - Zero-argument builtin entry point for `gemv(alpha, A, x, beta, y)`
 */
val_t cli_calc_func_gemv_op(void)
{
    return cli_calc_func_gemm(1);
}
