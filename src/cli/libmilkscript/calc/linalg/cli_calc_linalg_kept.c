// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_linalg_kept.c
 * @brief Kept matrix and reshape assignment lowering for calculator programs.
 *
 * Role & Goal:
 *   Compiles and executes direct destination-buffered BLAS Level 2/3 and shape
 *   assignments (`C = alpha * matmul(A, B) + beta * C`, `C += outer(x, y)`,
 *   `gemm`, `gemv`, `ger`, `syrk`, `transpose`, `reshape`) without intermediate
 *   temporary image allocations across repeated ticks.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// MILK_CMAKE_REQUEST_BLAS
// MILK_CMAKE_REQUEST_LAPACKE
#include "CLIcore.h"
#include "COREMOD_arith/image_linalg.h"
#include "cli_calc_internal.h"
#include "milk_blas_lapacke.h"

enum
{
    FKEPT_ELEMWISE  = 0,
    FKEPT_MATMUL    = 1,
    FKEPT_TRANSPOSE = 2,
    FKEPT_RESHAPE   = 3,
    FKEPT_GER       = 4,
    FKEPT_SYRK      = 5
};

/** parse_kept_img_or_trans - Parse `im` or `transpose(im)` as an operand of a kept matrix op. */
static int parse_kept_img_or_trans(IMAGE **out_im, int *out_trans)
{
    int trans = 0;
    if (cur_eval()->type == TOK_FUNC_IMIM_IM && strcmp(cur_eval()->sval, "transpose(") == 0)
    {
        advance_eval();
        trans = 1;
    }
    if (cur_eval()->type != TOK_IMAGE || strchr(cur_eval()->sval, '[') != NULL)
    {
        return -1;
    }
    imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
    if (id == -1)
    {
        return -1;
    }
    advance_eval();
    if (trans)
    {
        if (cur_eval()->type != TOK_RPAREN)
        {
            return -1;
        }
        advance_eval();
    }
    *out_im    = &data.core.image[id];
    *out_trans = trans;
    return 0;
}

/** parse_kept_scalar_factor - Parse a scalar constant or live variable factor for BLAS alpha/beta. */
static int parse_kept_scalar_factor(int keep_vars, double *out_c, int16_t *out_s1)
{
    int    pos  = cli_calc_tok_pos();
    double sign = 1.0;
    while (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        sign = -sign;
    }
    if (cur_eval()->type == TOK_LONG)
    {
        *out_c  = sign * (double) cur_eval()->val_l;
        *out_s1 = 0;
        advance_eval();
        return 0;
    }
    if (cur_eval()->type == TOK_DOUBLE)
    {
        *out_c  = sign * cur_eval()->val_d;
        *out_s1 = 0;
        advance_eval();
        return 0;
    }
    if ((cur_eval()->type == TOK_VAR || cur_eval()->type == TOK_NVAR) && !cur_eval()->literal &&
        image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE) == -1)
    {
        if (keep_vars)
        {
            long vid = variable_ID(cur_eval()->sval);
            if (vid >= 0 && data.core.variable[vid].type != 2)
            {
                *out_c  = sign;
                *out_s1 = (int16_t) (vid + 1);
                advance_eval();
                return 0;
            }
        }
        if (!(cur_eval()[1].type >= TOK_EQUAL && cur_eval()[1].type <= TOK_OP_SLASH_EQ))
        {
            val_t v = eval_existing_var_ref(cur_eval()->sval);
            if (v.type == VAL_LONG || v.type == VAL_DOUBLE)
            {
                *out_c  = sign * to_double(v);
                *out_s1 = 0;
                advance_eval();
                return 0;
            }
        }
    }
    cli_calc_tok_seek(pos);
    return -1;
}

/** parse_kept_dst_term - Parse `[scalar *] dst [* scalar]` representing `beta * dst`. */
static int parse_kept_dst_term(int          keep_vars,
                               const IMAGE *dst,
                               double       sign,
                               double      *beta_c,
                               int16_t     *beta_s1)
{
    int pos = cli_calc_tok_pos();
    if (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        sign = -sign;
    }
    if (cur_eval()->type == TOK_IMAGE && strcmp(cur_eval()->sval, dst->name) == 0)
    {
        advance_eval();
        double  c  = 1.0;
        int16_t s1 = 0;
        if (cur_eval()->type == TOK_OP_STAR)
        {
            advance_eval();
            if (parse_kept_scalar_factor(keep_vars, &c, &s1) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
        }
        *beta_c  = sign * c;
        *beta_s1 = s1;
        return 0;
    }
    double  c  = 1.0;
    int16_t s1 = 0;
    if (parse_kept_scalar_factor(keep_vars, &c, &s1) == 0 && cur_eval()->type == TOK_OP_STAR)
    {
        advance_eval();
        if (cur_eval()->type == TOK_IMAGE && strcmp(cur_eval()->sval, dst->name) == 0)
        {
            advance_eval();
            *beta_c  = sign * c;
            *beta_s1 = s1;
            return 0;
        }
    }
    cli_calc_tok_seek(pos);
    return -1;
}

/** parse_kept_alpha_arg - Parse leading `alpha,` argument inside `gemm(`, `gemv(`, `ger(`, or `syrk(`. */
static int parse_kept_alpha_arg(int keep_vars, double *alpha_c, int16_t *alpha_s1)
{
    double  ac  = 1.0;
    int16_t as1 = 0;
    if (parse_kept_scalar_factor(keep_vars, &ac, &as1) != 0 || (*alpha_s1 != 0 && as1 != 0) ||
        cur_eval()->type != TOK_COMMA)
    {
        return -1;
    }
    *alpha_c *= ac;
    if (as1 != 0)
    {
        *alpha_s1 = as1;
    }
    advance_eval();
    return 0;
}

/** parse_kept_beta_dst_args - Parse trailing `[beta,] dst)` arguments of `gemm`, `gemv`, `ger`, or `syrk`. */
static int parse_kept_beta_dst_args(cli_calc_kept_matrix *m,
                                    const IMAGE          *dst,
                                    int                   keep_vars,
                                    int                   has_beta)
{
    double  bc  = 1.0;
    int16_t bs1 = 0;
    if (has_beta)
    {
        if (parse_kept_scalar_factor(keep_vars, &bc, &bs1) != 0 || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
    }
    if (cur_eval()->type != TOK_IMAGE || strcmp(cur_eval()->sval, dst->name) != 0 ||
        m->beta_c != 0.0 || m->beta_s1 != 0)
    {
        return -1;
    }
    advance_eval();
    if (cur_eval()->type != TOK_RPAREN)
    {
        return -1;
    }
    advance_eval();
    m->beta_c  = bc;
    m->beta_s1 = bs1;
    return 0;
}

/** parse_kept_matmul_or_outer - Match and validate `matmul(...)` or `outer(...)` in a kept term. */
static int parse_kept_matmul_or_outer(const char *fn, cli_calc_kept_matrix *m, const IMAGE *dst)
{
    if (strcmp(fn, "matmul(") == 0)
    {
        advance_eval();
        IMAGE *imA    = NULL;
        IMAGE *imB    = NULL;
        int    transA = 0;
        int    transB = 0;
        if (parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_RPAREN)
        {
            return -1;
        }
        advance_eval();
        if (imA == imB && transA != transB && arith_image_syrk_check(imA, transA, dst) == 0)
        {
            m->kind   = FKEPT_SYRK;
            m->transA = transA;
            m->img[0] = imA;
            m->nimg   = 1;
            return 0;
        }
        if (arith_image_matmul_check(imA, transA, imB, transB, dst, 0) == 0)
        {
            m->kind          = FKEPT_MATMUL;
            m->transA        = transA;
            m->transB        = transB;
            m->allow_reshape = 0;
            m->img[0]        = imA;
            m->img[1]        = imB;
            m->nimg          = 2;
            return 0;
        }
        return -1;
    }
    if (strcmp(fn, "outer(") == 0)
    {
        advance_eval();
        IMAGE *imX = NULL;
        IMAGE *imY = NULL;
        int    tX  = 0;
        int    tY  = 0;
        if (parse_kept_img_or_trans(&imX, &tX) != 0 || tX || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imY, &tY) != 0 || tY || cur_eval()->type != TOK_RPAREN ||
            arith_image_ger_check(imX, imY, dst) != 0)
        {
            return -1;
        }
        advance_eval();
        m->kind   = FKEPT_GER;
        m->img[0] = imX;
        m->img[1] = imY;
        m->nimg   = 2;
        return 0;
    }
    return -1;
}

/** parse_kept_blas_dispatch - Match and validate the inner BLAS function call in a kept term. */
static int parse_kept_blas_dispatch(cli_calc_kept_matrix *m,
                                    const IMAGE          *dst,
                                    int                   keep_vars,
                                    double               *alpha_c,
                                    int16_t              *alpha_s1)
{
    const char *fn = cur_eval()->sval;
    if (strcmp(fn, "matmul(") == 0 || strcmp(fn, "outer(") == 0)
    {
        return parse_kept_matmul_or_outer(fn, m, dst);
    }
    if (strcmp(fn, "gemm(") == 0 || strcmp(fn, "gemv(") == 0)
    {
        int is_gemv = (strcmp(fn, "gemv(") == 0);
        advance_eval();
        IMAGE *imA    = NULL;
        IMAGE *imB    = NULL;
        int    transA = 0;
        int    transB = 0;
        if (parse_kept_alpha_arg(keep_vars, alpha_c, alpha_s1) != 0 ||
            parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
        if (parse_kept_beta_dst_args(m, dst, keep_vars, 1) != 0 ||
            arith_image_matmul_check(imA, transA, imB, transB, dst, 0) != 0 ||
            (is_gemv && dst->md[0].naxis == 2 && dst->md[0].size[1] != 1 && imA->md[0].naxis != 3))
        {
            return -1;
        }
        m->kind          = FKEPT_MATMUL;
        m->transA        = transA;
        m->transB        = transB;
        m->allow_reshape = 0;
        m->img[0]        = imA;
        m->img[1]        = imB;
        m->nimg          = 2;
        return 0;
    }
    if (strcmp(fn, "ger(") == 0)
    {
        advance_eval();
        IMAGE *imX = NULL;
        IMAGE *imY = NULL;
        int    tX  = 0;
        int    tY  = 0;
        if (parse_kept_alpha_arg(keep_vars, alpha_c, alpha_s1) != 0 ||
            parse_kept_img_or_trans(&imX, &tX) != 0 || tX || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imY, &tY) != 0 || tY || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
        if (parse_kept_beta_dst_args(m, dst, keep_vars, 0) != 0 ||
            arith_image_ger_check(imX, imY, dst) != 0)
        {
            return -1;
        }
        m->kind   = FKEPT_GER;
        m->img[0] = imX;
        m->img[1] = imY;
        m->nimg   = 2;
        return 0;
    }
    if (strcmp(fn, "syrk(") == 0)
    {
        advance_eval();
        IMAGE *imA    = NULL;
        int    transA = 0;
        if (parse_kept_alpha_arg(keep_vars, alpha_c, alpha_s1) != 0 ||
            parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
        if (parse_kept_beta_dst_args(m, dst, keep_vars, 1) != 0 ||
            arith_image_syrk_check(imA, transA, dst) != 0)
        {
            return -1;
        }
        m->kind   = FKEPT_SYRK;
        m->transA = transA;
        m->img[0] = imA;
        m->nimg   = 1;
        return 0;
    }
    return -1;
}

/** parse_kept_blas_term - Parse `[alpha *] (matmul|gemm|gemv|ger|outer|syrk)(...) [* alpha2]`. */
static int parse_kept_blas_term(cli_calc_kept_matrix *m,
                                const IMAGE          *dst,
                                double                sign,
                                int                   keep_vars)
{
    int pos = cli_calc_tok_pos();
    while (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        sign = -sign;
    }
    double  alpha_c  = sign;
    int16_t alpha_s1 = 0;
    int     p2       = cli_calc_tok_pos();
    double  fc       = 1.0;
    int16_t fs1      = 0;
    if (parse_kept_scalar_factor(keep_vars, &fc, &fs1) == 0)
    {
        if (cur_eval()->type == TOK_OP_STAR)
        {
            advance_eval();
            alpha_c *= fc;
            alpha_s1 = fs1;
        }
        else
        {
            cli_calc_tok_seek(p2);
        }
    }
    while (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        alpha_c = -alpha_c;
    }
    if (cur_eval()->type != TOK_FUNC_IMIM_IM ||
        parse_kept_blas_dispatch(m, dst, keep_vars, &alpha_c, &alpha_s1) != 0)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }
    if (cur_eval()->type == TOK_OP_STAR)
    {
        advance_eval();
        double  tc  = 1.0;
        int16_t ts1 = 0;
        if (parse_kept_scalar_factor(keep_vars, &tc, &ts1) != 0 || (alpha_s1 != 0 && ts1 != 0))
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        alpha_c *= tc;
        if (ts1 != 0)
        {
            alpha_s1 = ts1;
        }
    }
    m->alpha_c  = alpha_c;
    m->alpha_s1 = alpha_s1;
    return 0;
}

/** parse_kept_reshape_dims - Parse and validate target dimensions of a kept `reshape(...)` call. */
static int parse_kept_reshape_dims(uint64_t nel, const IMAGE *dst)
{
    int      ndims    = 0;
    long     dims[3]  = { 1, 1, 1 };
    int      neg_idx  = -1;
    uint64_t prod_pos = 1;

    while (cur_eval()->type == TOK_COMMA && ndims < 3)
    {
        advance_eval();
        double dv = 0.0;
        if (cli_calc_eval_const_scalar(&dv) != 0)
        {
            return -1;
        }
        long d = llround(dv);
        if (d == -1)
        {
            if (neg_idx != -1)
            {
                return -1;
            }
            neg_idx = ndims;
        }
        else if (d <= 0)
        {
            return -1;
        }
        else
        {
            dims[ndims] = d;
            prod_pos *= (uint64_t) d;
        }
        ndims++;
    }
    if (ndims == 0 || cur_eval()->type != TOK_RPAREN)
    {
        return -1;
    }
    advance_eval();
    if (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        return -1;
    }
    if (neg_idx != -1)
    {
        if (prod_pos == 0 || nel % prod_pos != 0)
        {
            return -1;
        }
        dims[neg_idx] = (long) (nel / prod_pos);
        prod_pos      = nel;
    }
    if (prod_pos != nel || dst->md[0].naxis != ndims)
    {
        return -1;
    }
    for (int i = 0; i < ndims; i++)
    {
        if (dst->md[0].size[i] != (uint32_t) dims[i])
        {
            return -1;
        }
    }
    return 0;
}

/** compile_kept_transpose_or_reshape - Try to compile `dst = transpose(im)` or `dst = reshape(...)`. */
static int compile_kept_transpose_or_reshape(cli_calc_kept_matrix *m,
                                             IMAGE                *dst,
                                             int                   strict_local_dt)
{
    int pos = cli_calc_tok_pos();
    if (strcmp(cur_eval()->sval, "transpose(") == 0)
    {
        advance_eval();
        if (cur_eval()->type != TOK_IMAGE || strchr(cur_eval()->sval, '[') != NULL)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
        advance_eval();
        if (id == -1 || cur_eval()->type != TOK_RPAREN)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        IMAGE  *src = &data.core.image[id];
        uint8_t exp_dt =
            (src->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
        if ((cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE) ||
            arith_image_transpose_check(src, dst) != 0 ||
            (strict_local_dt && dst->md[0].datatype != exp_dt))
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        m->kind   = FKEPT_TRANSPOSE;
        m->img[0] = src;
        m->nimg   = 1;
        return 0;
    }
    if (strcmp(cur_eval()->sval, "reshape(") == 0)
    {
        advance_eval();
        if (cur_eval()->type == TOK_FUNC_IMIM_IM && strcmp(cur_eval()->sval, "matmul(") == 0)
        {
            advance_eval();
            IMAGE *imA    = NULL;
            IMAGE *imB    = NULL;
            int    transA = 0;
            int    transB = 0;
            if (parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            advance_eval();
            if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_RPAREN)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            advance_eval();
            uint8_t exp_dt =
                (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
                    ? _DATATYPE_DOUBLE
                    : _DATATYPE_FLOAT;
            if (arith_image_matmul_check(imA, transA, imB, transB, dst, 1) != 0 ||
                parse_kept_reshape_dims(dst->md[0].nelement, dst) != 0 ||
                (strict_local_dt && dst->md[0].datatype != exp_dt))
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            m->kind          = FKEPT_MATMUL;
            m->transA        = transA;
            m->transB        = transB;
            m->allow_reshape = 1;
            m->alpha_c       = 1.0;
            m->beta_c        = 0.0;
            m->img[0]        = imA;
            m->img[1]        = imB;
            m->nimg          = 2;
            return 0;
        }
        if (cur_eval()->type == TOK_IMAGE && strchr(cur_eval()->sval, '[') == NULL)
        {
            imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
            advance_eval();
            if (id == -1 || arith_image_reshape_check(&data.core.image[id], dst) != 0 ||
                parse_kept_reshape_dims(dst->md[0].nelement, dst) != 0 ||
                (strict_local_dt && dst->md[0].datatype != data.core.image[id].md[0].datatype))
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            m->kind   = FKEPT_RESHAPE;
            m->img[0] = &data.core.image[id];
            m->nimg   = 1;
            return 0;
        }
    }
    cli_calc_tok_seek(pos);
    return -1;
}

/**
 * cli_calc_compile_kept_matrix - Try to compile a top-level matrix or reshape assignment into @m
 * @m:               Kept-matrix descriptor to populate
 * @dst:             Destination image assigned by the statement
 * @aop:             Assignment operator (`=`, `+=`, or `-=`)
 * @keep_vars:       Non-zero to record live scalar variable indices for `alpha` / `beta`
 * @strict_local_dt: Non-zero to require local @dst datatype to match natural output precision
 *
 * Return: 0 on successful compilation, -1 if the RHS is not a supported matrix/reshape op
 */
int cli_calc_compile_kept_matrix(cli_calc_kept_matrix *m,
                                 IMAGE                *dst,
                                 cli_token_type        aop,
                                 int                   keep_vars,
                                 int                   strict_local_dt)
{
    memset(m, 0, sizeof(*m));
    int pos = cli_calc_tok_pos();
    if (aop != TOK_EQUAL && aop != TOK_OP_PLUS_EQ && aop != TOK_OP_MINUS_EQ)
    {
        return -1;
    }
    if (aop == TOK_EQUAL && cur_eval()->type == TOK_FUNC_IMIM_IM &&
        compile_kept_transpose_or_reshape(m, dst, strict_local_dt) == 0)
    {
        return 0;
    }

    cli_calc_tok_seek(pos);
    m->alpha_c  = 1.0;
    m->alpha_s1 = 0;
    m->beta_c   = (aop == TOK_EQUAL) ? 0.0 : 1.0;
    m->beta_s1  = 0;
    double sign = (aop == TOK_OP_MINUS_EQ) ? -1.0 : 1.0;

    if (aop == TOK_EQUAL)
    {
        double  bc  = 0.0;
        int16_t bs1 = 0;
        if (parse_kept_dst_term(keep_vars, dst, 1.0, &bc, &bs1) == 0)
        {
            if (cur_eval()->type != TOK_OP_PLUS && cur_eval()->type != TOK_OP_MINUS)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            sign = (cur_eval()->type == TOK_OP_MINUS) ? -1.0 : 1.0;
            advance_eval();
            m->beta_c  = bc;
            m->beta_s1 = bs1;
            if (parse_kept_blas_term(m, dst, sign, keep_vars) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
        }
        else
        {
            if (parse_kept_blas_term(m, dst, 1.0, keep_vars) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            if ((cur_eval()->type == TOK_OP_PLUS || cur_eval()->type == TOK_OP_MINUS) &&
                m->beta_c == 0.0 && m->beta_s1 == 0)
            {
                double bsign = (cur_eval()->type == TOK_OP_MINUS) ? -1.0 : 1.0;
                advance_eval();
                if (parse_kept_dst_term(keep_vars, dst, bsign, &m->beta_c, &m->beta_s1) != 0)
                {
                    cli_calc_tok_seek(pos);
                    return -1;
                }
            }
        }
    }
    else if (parse_kept_blas_term(m, dst, sign, keep_vars) != 0)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }

    if (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }
    if (strict_local_dt)
    {
        int any_d = 0;
        for (int i = 0; i < m->nimg; i++)
        {
            if (m->img[i]->md[0].datatype == _DATATYPE_DOUBLE)
            {
                any_d = 1;
            }
        }
        uint8_t exp_dt = any_d ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
        if (dst->md[0].datatype != exp_dt)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
    }
    return 0;
}
