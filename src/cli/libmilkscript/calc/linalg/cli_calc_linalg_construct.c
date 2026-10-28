// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_linalg_construct.c
 * @brief Calculator tensor shape and constructor builtins (`transpose`, `reshape`,
 *        `vec`, `zeros`, `eye`, `diag`).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"

/**
 * cli_calc_func_transpose - Evaluate calculator `transpose(im)` call
 *
 * Parses a single 1-D or 2-D image argument, allocates a transposed temporary
 * image (`N x M`), and populates it via `arith_image_transpose_im()`.
 *
 * Return: `val_t` holding the temporary image name, or empty string on error.
 */
val_t cli_calc_func_transpose(void)
{
    val_t arg;
    if (cli_calc_parse_func_argv(1, 1, &arg, NULL) < 0)
    {
        return mk_double(0);
    }
    if (cli_calc_reject_complex(&arg, 1))
    {
        return mk_string("");
    }
    if (arg.type != VAL_STRING || !check_image(arg.sval))
    {
        if (arg.type != VAL_STRING)
        {
            parse_errmsg("transpose: expected image argument");
        }
        return mk_string("");
    }
    IMAGE *src = &data.core.image[image_ID(arg.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (src->md[0].naxis != 1 && src->md[0].naxis != 2)
    {
        parse_errmsg("transpose: expected 1D or 2D image");
        return mk_string("");
    }
    uint32_t outsz[3] = {
        (src->md[0].naxis == 2) ? src->md[0].size[1] : 1,
        src->md[0].size[0],
        1,
    };
    uint8_t out_dt = (src->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, 2, outsz, out_dt, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("transpose: failed to allocate output image");
        return mk_string("");
    }
    if (arith_image_transpose_im(src, &data.core.image[tid]) != 0)
    {
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_reshape - Evaluate calculator `reshape(im, d0[, d1[, d2]])` call
 *
 * Supports 1-D, 2-D, or 3-D target shapes with at most one inferred `-1` dimension,
 * returning a newly allocated temporary image with the same elements as `im`.
 *
 * Return: `val_t` holding the temporary image name, or empty string on error.
 */
val_t cli_calc_func_reshape(void)
{
    val_t args[4];
    int   n = cli_calc_parse_func_argv(2, 4, args, "Expected ','");
    if (n < 0)
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || !check_image(args[0].sval))
    {
        if (args[0].type != VAL_STRING)
        {
            parse_errmsg("reshape: first argument must be an image");
        }
        return mk_string("");
    }
    IMAGE *src = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t nel      = src->md[0].nelement;
    int      ndims    = n - 1;
    long     dims[3]  = { 1, 1, 1 };
    int      neg_idx  = -1;
    uint64_t prod_pos = 1;

    for (int i = 0; i < ndims; i++)
    {
        if (args[i + 1].type == VAL_STRING)
        {
            parse_errmsg("reshape: dimensions must be integers");
            return mk_string("");
        }
        long d =
            (args[i + 1].type == VAL_LONG) ? args[i + 1].lval : (long) llround(args[i + 1].dval);
        if (d == -1)
        {
            if (neg_idx != -1)
            {
                parse_errmsg("reshape: at most one dimension may be -1");
                return mk_string("");
            }
            neg_idx = i;
        }
        else if (d <= 0)
        {
            parse_errmsg("reshape: dimensions must be positive");
            return mk_string("");
        }
        else
        {
            dims[i] = d;
            prod_pos *= (uint64_t) d;
        }
    }
    if (neg_idx != -1)
    {
        if (prod_pos == 0 || nel % prod_pos != 0)
        {
            parse_errmsg("reshape: element count mismatch");
            return mk_string("");
        }
        dims[neg_idx] = (long) (nel / prod_pos);
        prod_pos      = nel;
    }
    if (prod_pos != nel)
    {
        parse_errmsg("reshape: element count mismatch");
        return mk_string("");
    }
    uint32_t    outsz[3] = { (uint32_t) dims[0], (uint32_t) dims[1], (uint32_t) dims[2] };
    const char *tmpn     = alloc_tmpname();
    imageID     tid      = -1;
    cli_calc_tmp_image(tmpn, ndims, outsz, src->md[0].datatype, 1, &tid);
    if (tid == -1 || arith_image_reshape_im(src, &data.core.image[tid]) != 0)
    {
        if (tid == -1)
        {
            parse_errmsg("reshape: failed to allocate output image");
        }
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_vec - Evaluate calculator `vec(v0, v1, ...)` 1-D float vector constructor
 *
 * Parses up to 256 numeric scalar arguments and packs them into a 1-D `_DATATYPE_FLOAT`
 * temporary image.
 *
 * Return: `val_t` holding the temporary image name, or empty string on error.
 */
val_t cli_calc_func_vec(void)
{
    val_t args[256];
    int   n = cli_calc_parse_func_argv(1, 256, args, "Expected ','");
    if (n < 0)
    {
        return mk_string("");
    }
    for (int i = 0; i < n; i++)
    {
        if (args[i].type == VAL_STRING)
        {
            parse_errmsg("vec: arguments must be numbers");
            return mk_string("");
        }
    }
    uint32_t    outsz[3] = { (uint32_t) n, 1, 1 };
    const char *tmpn     = alloc_tmpname();
    imageID     tid      = -1;
    cli_calc_tmp_image(tmpn, 1, outsz, _DATATYPE_FLOAT, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("vec: failed to allocate output image");
        return mk_string("");
    }
    IMAGE *out = &data.core.image[tid];
    for (int i = 0; i < n; i++)
    {
        out->array.F[i] = (float) to_double(args[i]);
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_zeros - Evaluate `zeros(nx[, ny[, nz]])`: a float32 image of that shape, all zero
 *
 * Return: `val_t` holding the temporary image name, or empty string on error.
 */
val_t cli_calc_func_zeros(void)
{
    val_t args[3];
    int   n = cli_calc_parse_func_argv(1, 3, args, "zeros: expected ','");
    if (n < 0)
    {
        return mk_string("");
    }
    uint32_t sz[3] = { 1, 1, 1 };
    for (int i = 0; i < n; i++)
    {
        long d = (args[i].type == VAL_STRING) ? 0 : llround(to_double(args[i]));
        if (d <= 0)
        {
            parse_errmsg("zeros: dimensions must be positive numbers");
            return mk_string("");
        }
        sz[i] = (uint32_t) d;
    }
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, n, sz, _DATATYPE_FLOAT, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("zeros: failed to allocate output image");
        return mk_string("");
    }
    memset(data.core.image[tid].array.F, 0, sizeof(float) * data.core.image[tid].md[0].nelement);
    return mk_string(tmpn);
}

/**
 * cli_calc_func_eye - Evaluate `eye(N[, M])` to construct an `N x M` float32 identity matrix
 *
 * Return: `val_t` holding the temporary identity matrix image name, or empty string on error.
 */
val_t cli_calc_func_eye(void)
{
    val_t args[2];
    int   nargs = cli_calc_parse_func_argv(1, 2, args, "eye: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, nargs))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || (nargs == 2 && args[1].type == VAL_STRING))
    {
        parse_errmsg("eye: expected integer dimension arguments");
        return mk_string("");
    }
    long n = llround(to_double(args[0]));
    long m = (nargs == 2) ? llround(to_double(args[1])) : n;
    if (n <= 0 || m <= 0)
    {
        parse_errmsg("eye: dimensions must be positive");
        return mk_string("");
    }
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[2] = { (uint32_t) n, (uint32_t) m };
    cli_calc_tmp_image(tmpn, 2, sz, _DATATYPE_FLOAT, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("eye: failed to allocate output image");
        return mk_string("");
    }
    IMAGE   *out = &data.core.image[tid];
    uint64_t nel = (uint64_t) n * (uint64_t) m;
    memset(out->array.F, 0, sizeof(float) * nel);
    uint64_t dlen = (n < m) ? (uint64_t) n : (uint64_t) m;
    for (uint64_t i = 0; i < dlen; i++)
    {
        out->array.F[i * (uint64_t) n + i] = 1.0f;
    }
    return mk_string(tmpn);
}

/**
 * diag_from_1d - Build a 2-D diagonal matrix from a 1-D vector @src at offset @k
 */
static val_t diag_from_1d(const IMAGE *src, uint8_t src_dt, uint8_t out_dt, long k, long absk)
{
    uint64_t    L     = src->md[0].size[0];
    uint64_t    N     = L + (uint64_t) absk;
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[2] = { (uint32_t) N, (uint32_t) N };
    cli_calc_tmp_image(tmpn, 2, sz, out_dt, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("diag: failed to allocate output image");
        return mk_string("");
    }
    IMAGE   *out = &data.core.image[tid];
    uint64_t r0  = (k < 0) ? (uint64_t) absk : 0;
    uint64_t c0  = (k > 0) ? (uint64_t) absk : 0;
    if (out_dt == _DATATYPE_DOUBLE)
    {
        memset(out->array.D, 0, sizeof(double) * N * N);
        for (uint64_t i = 0; i < L; i++)
        {
            out->array.D[(c0 + i) * N + (r0 + i)] = cli_calc_read_im_pix(src, src_dt, i);
        }
    }
    else
    {
        memset(out->array.F, 0, sizeof(float) * N * N);
        for (uint64_t i = 0; i < L; i++)
        {
            out->array.F[(c0 + i) * N + (r0 + i)] = (float) cli_calc_read_im_pix(src, src_dt, i);
        }
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_diag - Evaluate `diag(X[, k])` to extract or construct a diagonal matrix
 *
 * Return: `val_t` holding the temporary output image name, or empty string on error.
 */
val_t cli_calc_func_diag(void)
{
    val_t args[2];
    int   nargs = cli_calc_parse_func_argv(1, 2, args, "diag: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, nargs))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || !check_image(args[0].sval))
    {
        parse_errmsg("diag: first argument must be an image");
        return mk_string("");
    }
    long k = 0;
    if (nargs == 2)
    {
        if (args[1].type == VAL_STRING)
        {
            parse_errmsg("diag: offset k must be a number");
            return mk_string("");
        }
        k = llround(to_double(args[1]));
    }
    const IMAGE *src =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint8_t src_dt = src->md[0].datatype;
    uint8_t out_dt = (src_dt == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    long    absk   = (k < 0) ? -k : k;

    if (src->md[0].naxis == 1)
    {
        return diag_from_1d(src, src_dt, out_dt, k, absk);
    }

    if (src->md[0].naxis == 2)
    {
        uint64_t M  = src->md[0].size[0];
        uint64_t N  = src->md[0].size[1];
        uint64_t r0 = (k < 0) ? (uint64_t) absk : 0;
        uint64_t c0 = (k > 0) ? (uint64_t) absk : 0;
        if (r0 >= M || c0 >= N)
        {
            parse_errmsg("diag: diagonal offset out of range");
            return mk_string("");
        }
        uint64_t    rm   = M - r0;
        uint64_t    cn   = N - c0;
        uint64_t    L    = (rm < cn) ? rm : cn;
        const char *tmpn = alloc_tmpname();
        imageID     tid  = -1;
        uint32_t    sz   = (uint32_t) L;
        cli_calc_tmp_image(tmpn, 1, &sz, out_dt, 1, &tid);
        if (tid == -1)
        {
            parse_errmsg("diag: failed to allocate output image");
            return mk_string("");
        }
        IMAGE *out = &data.core.image[tid];
        for (uint64_t i = 0; i < L; i++)
        {
            double v = cli_calc_read_im_pix(src, src_dt, (c0 + i) * M + (r0 + i));
            if (out_dt == _DATATYPE_DOUBLE)
            {
                out->array.D[i] = v;
            }
            else
            {
                out->array.F[i] = (float) v;
            }
        }
        return mk_string(tmpn);
    }

    parse_errmsg("diag: expected 1D or 2D image");
    return mk_string("");
}

enum
{
    FKEPT_ELEMWISE  = 0,
    FKEPT_MATMUL    = 1,
    FKEPT_TRANSPOSE = 2,
    FKEPT_RESHAPE   = 3,
    FKEPT_GER       = 4,
    FKEPT_SYRK      = 5
};

/** is_top_level_transpose - Peek whether the current argument is wrapped in `transpose(...)`. */
static int is_top_level_transpose(void)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    if (cur_func()->type != TOK_FUNC_IMIM_IM || strcmp(cur_func()->sval, "transpose(") != 0)
    {
        return 0;
    }
    int saved = cli_calc_tok_pos();
    advance_func();
    int depth = 1;
    while (depth > 0 && cur_func()->type != TOK_EOF && cur_func()->type != TOK_NEWLINE)
    {
        cli_token_type t = cur_func()->type;
        if (t == TOK_LPAREN ||
            (t >= TOK_FUNC_D_D && t <= TOK_FUNC_COORD && strchr(cur_func()->sval, '(') != NULL))
        {
            depth++;
        }
        else if (t == TOK_RPAREN)
        {
            depth--;
        }
        advance_func();
    }
    int is_top = (depth == 0 && (cur_func()->type == TOK_COMMA || cur_func()->type == TOK_RPAREN));
    cli_calc_tok_seek(saved);
    return is_top;
}

/**
 * cli_calc_parse_matmul_arg - Parse one matrix argument, unwrapping a top-level `transpose()`
 * @out:       Receives the evaluated operand value
 * @out_trans: Receives 1 if an outer `transpose(...)` was unwrapped, 0 otherwise
 *
 * Return: 0 on success, -1 on syntax or evaluation error.
 */
int cli_calc_parse_matmul_arg(val_t *out, int *out_trans)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    int trans                        = 0;

    if (is_top_level_transpose())
    {
        advance_func();
        trans = 1;
    }
    *out = parse_expr(0);
    if (parse_error || eval_error)
    {
        return -1;
    }
    if (trans)
    {
        if (cur_func()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return -1;
        }
        advance_func();
    }
    *out_trans = trans;
    return 0;
}

static int read_kept_var_checked(int16_t s1, double *out)
{
    int vid = (int) s1 - 1;
    if (vid < 0 || vid >= data.core.NB_MAX_VARIABLE)
    {
        return -1;
    }
    const VARIABLE *v = &data.core.variable[vid];
    if (!v->used || (v->type != 0 && v->type != 1))
    {
        return -1;
    }
    if (image_ID(v->name, data.core.image, data.core.NB_MAX_IMAGE) != -1)
    {
        return -1;
    }
    *out = (v->type == 1) ? (double) v->value.l : v->value.f;
    return 0;
}

/**
 * cli_calc_exec_kept_matrix - Execute a compiled matrix or reshape operation directly into @dst
 * @m:   Compiled `cli_calc_kept_matrix` descriptor
 * @dst: Destination image to write
 *
 * Return: 0 on success, -1 if an image changed shape/type or execution failed.
 */
int cli_calc_exec_kept_matrix(const cli_calc_kept_matrix *m, IMAGE *dst)
{
    double alpha = m->alpha_c;
    if (m->alpha_s1 > 0)
    {
        double av = 0.0;
        if (read_kept_var_checked(m->alpha_s1, &av) != 0)
        {
            return -1;
        }
        alpha = av * m->alpha_c;
    }
    double beta = m->beta_c;
    if (m->beta_s1 > 0)
    {
        double bv = 0.0;
        if (read_kept_var_checked(m->beta_s1, &bv) != 0)
        {
            return -1;
        }
        beta = bv * m->beta_c;
    }
    const char *opname = "matrix";
    int         chk    = -1;

    switch (m->kind)
    {
    case FKEPT_MATMUL:
        opname = "matmul";
        chk    = arith_image_matmul_check(m->img[0], m->transA, m->img[1], m->transB, dst,
                                          m->allow_reshape);
        break;
    case FKEPT_GER:
        opname = "ger";
        chk    = arith_image_ger_check(m->img[0], m->img[1], dst);
        break;
    case FKEPT_SYRK:
        opname = "syrk";
        chk    = arith_image_syrk_check(m->img[0], m->transA, dst);
        break;
    case FKEPT_TRANSPOSE:
        opname = "transpose";
        chk    = arith_image_transpose_check(m->img[0], dst);
        break;
    case FKEPT_RESHAPE:
        opname = "reshape";
        chk    = arith_image_reshape_check(m->img[0], dst);
        break;
    default:
        return -1;
    }
    if (chk != 0)
    {
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s: %s or input changed size or type",
                 opname, dst->name);
        return -1;
    }

    dst->md[0].write = 1;
    int rc           = -1;
    switch (m->kind)
    {
    case FKEPT_MATMUL:
        rc = arith_image_gemm_im(m->img[0], m->transA, m->img[1], m->transB, alpha, beta, dst);
        if (rc != 0)
        {
            parse_errmsg("matmul: memory allocation failed");
        }
        break;
    case FKEPT_GER:
        rc = arith_image_ger_im(m->img[0], m->img[1], alpha, beta, dst);
        if (rc != 0)
        {
            parse_errmsg("ger: expected two 1D vector arguments");
        }
        break;
    case FKEPT_SYRK:
        rc = arith_image_syrk_im(m->img[0], m->transA, alpha, beta, dst);
        break;
    case FKEPT_TRANSPOSE:
        rc = arith_image_transpose_im(m->img[0], dst);
        break;
    case FKEPT_RESHAPE:
        rc = arith_image_reshape_im(m->img[0], dst);
        break;
    default:
        break;
    }
    if (rc != 0)
    {
        dst->md[0].write = 0;
        return -1;
    }
    return 0;
}
