// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_functions.c
 * @brief Calculator built-in function call evaluation (`sin`, `where`, `matmul`, string ops).
 *
 * Role & Goal:
 *   Evaluates built-in mathematical, image-reduction, matrix-multiplication, and
 *   string manipulation functions inside calculator expressions (`parse_funccall()`).
 *   Uses a shared `parse_func_args()` helper to parse comma-separated argument lists
 *   and dispatches each function signature class to a dedicated handler.
 */

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"
#include "cli_calc_internal.h"
#include "libmilkdata/milk_type_dispatch.h"

/**
 * @brief Parse @nargs comma-separated expression arguments followed by `)`.
 *
 * @param nargs     Number of arguments expected (1..3)
 * @param args      Output array of `val_t` receiving parsed arguments
 * @param comma_msg Error message if a comma is missing
 * @return 0 on success, -1 on syntax/evaluation error
 */
static int parse_func_args(int nargs, val_t *args, const char *comma_msg)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();

    for (int i = 0; i < nargs; i++)
    {
        if (i > 0)
        {
            if (cur_func()->type != TOK_COMMA)
            {
                parse_errmsg(comma_msg != NULL ? comma_msg : "Expected ','");
                return -1;
            }
            advance_func();
        }
        args[i] = parse_expr(0);
        if (parse_error || eval_error)
        {
            return -1;
        }
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
 * @brief Evaluate `where(cond, T, F)` for scalar or image condition.
 */
static val_t func_where(void)
{
    val_t args[3];
    if (parse_func_args(3, args, "Expected ','") != 0)
    {
        return mk_double(0);
    }

    if (args[0].type != VAL_STRING)
    {
        double c = to_double(args[0]);
        return (c != 0.0) ? args[1] : args[2];
    }

    char tmp_mask[200];
    char tmp_imask[200];
    char tmp_tpart[200];
    char tmp_fpart[200];
    char tmpn[200];
    snprintf(tmp_mask, sizeof(tmp_mask), "%s", alloc_tmpname());
    snprintf(tmp_imask, sizeof(tmp_imask), "%s", alloc_tmpname());
    snprintf(tmp_tpart, sizeof(tmp_tpart), "%s", alloc_tmpname());
    snprintf(tmp_fpart, sizeof(tmp_fpart), "%s", alloc_tmpname());
    snprintf(tmpn, sizeof(tmpn), "%s", alloc_tmpname());

    if (!check_image(args[0].sval))
    {
        return mk_string("");
    }

    arith_image_csttestne(args[0].sval, 0.0, tmp_mask);
    arith_image_cstteste(args[0].sval, 0.0, tmp_imask);

    if (args[1].type == VAL_STRING)
    {
        if (!check_image(args[1].sval))
        {
            return mk_string("");
        }
        arith_image_mult(args[1].sval, tmp_mask, tmp_tpart);
    }
    else
    {
        arith_image_cstmult(tmp_mask, to_double(args[1]), tmp_tpart);
    }

    if (args[2].type == VAL_STRING)
    {
        if (!check_image(args[2].sval))
        {
            return mk_string("");
        }
        arith_image_mult(args[2].sval, tmp_imask, tmp_fpart);
    }
    else
    {
        arith_image_cstmult(tmp_imask, to_double(args[2]), tmp_fpart);
    }

    arith_image_add(tmp_tpart, tmp_fpart, tmpn);
    return mk_string(tmpn);
}

/**
 * @brief Evaluate `replace(s, old, new)` returning a temporary string variable.
 */
static val_t func_replace(void)
{
    val_t args[3];
    if (parse_func_args(3, args, "replace: need 3 args") != 0)
    {
        return mk_double(0);
    }

    const char *s[3] = { NULL, NULL, NULL };
    for (int i = 0; i < 3; i++)
    {
        if (args[i].type == VAL_STRING)
        {
            const char *cv = cli_var_get(args[i].sval);
            s[i]           = cv ? cv : args[i].sval;
        }
    }
    if (!s[0] || !s[1] || !s[2])
    {
        parse_errmsg("replace: all args must be strings");
        return mk_double(0);
    }

    char result[CLI_CALC_TOKEN_MAXLEN];
    result[0] = '\0';
    int s2len = (int) strlen(s[1]);
    if (s2len == 0)
    {
        strncpy(result, s[0], sizeof(result) - 1);
        result[sizeof(result) - 1] = '\0';
    }
    else
    {
        const char *p   = s[0];
        char       *w   = result;
        char       *end = result + sizeof(result) - 1;
        while (*p && w < end)
        {
            if (strncmp(p, s[1], (size_t) s2len) == 0)
            {
                int rlen = (int) strlen(s[2]);
                if (w + rlen < end)
                {
                    memcpy(w, s[2], (size_t) rlen);
                    w += rlen;
                }
                p += s2len;
            }
            else
            {
                *w++ = *p++;
            }
        }
        *w = '\0';
    }

    return mk_string(result);
}

/**
 * @brief Evaluate unary `double -> double` or `image -> image` function (`TOK_FUNC_D_D`).
 */
static val_t func_eval_d_d(const cli_token *ftok)
{
    val_t arg;
    if (parse_func_args(1, &arg, NULL) != 0)
    {
        return mk_double(0);
    }

    if (arg.type == VAL_STRING)
    {
        if (!check_image(arg.sval))
        {
            return mk_string("");
        }
        const char *tmpn = alloc_tmpname();
        arith_image_function_im_im__d_d(arg.sval, tmpn, ftok->fnctptr);
        if (data.core.Debug > 0)
        {
            printf("double_func(double)\n");
        }
        return mk_string(tmpn);
    }

    double darg = to_double(arg);
    double res  = ((double (*)(double)) ftok->fnctptr)(darg);
    if (data.core.Debug > 0)
    {
        printf("double=func(double)\n");
    }
    return mk_double(res);
}

/**
 * @brief Evaluate binary `(d, d) -> d`, `(im, d) -> im`, or `(im, im) -> im` (`TOK_FUNC_DD_D`).
 */
static val_t func_eval_dd_d(const cli_token *ftok)
{
    val_t args[2];
    if (parse_func_args(2, args, "Expected ','") != 0)
    {
        return mk_double(0);
    }

    if (args[0].type == VAL_STRING && args[1].type == VAL_STRING)
    {
        if (!check_image(args[0].sval) || !check_image(args[1].sval))
        {
            return mk_string("");
        }
        const char *tmpn = alloc_tmpname();
        if (ftok->fnctptr == fmin)
        {
            arith_image_minv(args[0].sval, args[1].sval, tmpn);
        }
        else if (ftok->fnctptr == fmax)
        {
            arith_image_maxv(args[0].sval, args[1].sval, tmpn);
        }
        else if (ftok->fnctptr == fmod)
        {
            arith_image_fmod(args[0].sval, args[1].sval, tmpn);
        }
        else if (ftok->fnctptr == pow)
        {
            arith_image_pow(args[0].sval, args[1].sval, tmpn);
        }
        else
        {
            parse_errmsg("Unsupported (image, image) function");
            return mk_string("");
        }
        return mk_string(tmpn);
    }

    if (args[0].type == VAL_STRING)
    {
        if (!check_image(args[0].sval))
        {
            return mk_string("");
        }
        const char *tmpn = alloc_tmpname();
        arith_image_function_imd_im__dd_d(args[0].sval, to_double(args[1]), tmpn, ftok->fnctptr);
        return mk_string(tmpn);
    }

    if (args[1].type == VAL_STRING) // (scalar, image)
    {
        if (!check_image(args[1].sval))
        {
            return mk_string("");
        }
        const char *tmpn = alloc_tmpname();
        if (ftok->fnctptr == fmin || ftok->fnctptr == fmax) // commutative
        {
            arith_image_function_imd_im__dd_d(args[1].sval, to_double(args[0]), tmpn,
                                              ftok->fnctptr);
        }
        else if (ftok->fnctptr == fmod)
        {
            char cst[32];
            snprintf(cst, sizeof(cst), "%s", alloc_tmpname()); // constant image shaped like arg 2
            char zero[32];
            snprintf(zero, sizeof(zero), "%s", alloc_tmpname());
            arith_image_cstmult(args[1].sval, 0.0, zero);
            arith_image_cstadd(zero, to_double(args[0]), cst);
            arith_image_fmod(cst, args[1].sval, tmpn);
        }
        else
        {
            parse_errmsg("Unsupported (scalar, image) function");
            return mk_string("");
        }
        return mk_string(tmpn);
    }

    double d1 = to_double(args[0]);
    double d2 = to_double(args[1]);
    return mk_double(((double (*)(double, double)) ftok->fnctptr)(d1, d2));
}

/**
 * @brief Evaluate ternary `(d, d, d) -> d` or `(im, d, d) -> im` (`TOK_FUNC_DDD_D`).
 */
static val_t func_eval_ddd_d(const cli_token *ftok)
{
    val_t args[3];
    if (parse_func_args(3, args, "Expected ','") != 0)
    {
        return mk_double(0);
    }

    if (args[0].type == VAL_STRING)
    {
        if (!check_image(args[0].sval))
        {
            return mk_string("");
        }
        const char *tmpn = alloc_tmpname();
        arith_image_function_imdd_im__ddd_d(args[0].sval, to_double(args[1]), to_double(args[2]),
                                            tmpn, ftok->fnctptr);
        return mk_string(tmpn);
    }
    if (args[1].type == VAL_STRING || args[2].type == VAL_STRING)
    {
        parse_errmsg("Only the first argument may be an image");
        return mk_double(0);
    }

    double d1 = to_double(args[0]);
    double d2 = to_double(args[1]);
    double d3 = to_double(args[2]);
    return mk_double(((double (*)(double, double, double)) ftok->fnctptr)(d1, d2, d3));
}

static void copy_im_to_double_buf(const IMAGE *im, double *dst)
{
    uint64_t nel = im->md[0].nelement;
    uint8_t  dt  = im->md[0].datatype;
    if (dt == _DATATYPE_FLOAT)
    {
        for (uint64_t i = 0; i < nel; i++)
        {
            dst[i] = (double) im->array.F[i];
        }
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        memcpy(dst, im->array.D, sizeof(double) * nel);
    }
    else
    {
        memset(dst, 0, sizeof(double) * nel);
    }
}

/**
 * @brief Evaluate `matmul(imA, imB)` (`TOK_FUNC_IMIM_IM`).
 */
static val_t func_eval_matmul(void)
{
    val_t args[2];
    if (parse_func_args(2, args, "Expected ','") != 0)
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING)
    {
        parse_errmsg("matmul: expected two image arguments");
        return mk_string("");
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_string("");
    }

    imageID id1 = image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE);
    imageID id2 = image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE);
    IMAGE  *im1 = &data.core.image[id1];
    IMAGE  *im2 = &data.core.image[id2];

    uint64_t a0      = im1->md[0].size[0];
    uint64_t a1      = 1;
    int      is_3d_a = 0;
    if (im1->md[0].naxis == 3)
    {
        a0      = (uint64_t) im1->md[0].size[0] * im1->md[0].size[1];
        a1      = im1->md[0].size[2];
        is_3d_a = 1;
    }
    else if (im1->md[0].naxis >= 2)
    {
        a1 = im1->md[0].size[1];
    }

    uint64_t b0 = im2->md[0].size[0];
    uint64_t b1 = 1;
    if (im2->md[0].naxis == 3)
    {
        b0 = (uint64_t) im2->md[0].size[0] * im2->md[0].size[1];
        b1 = im2->md[0].size[2];
    }
    else if (im2->md[0].naxis >= 2 &&
             !(is_3d_a && a0 == (uint64_t) im2->md[0].size[0] * im2->md[0].size[1]))
    {
        b1 = im2->md[0].size[1];
    }
    else if (is_3d_a && im2->md[0].naxis == 2 &&
             a0 == (uint64_t) im2->md[0].size[0] * im2->md[0].size[1])
    {
        b0 = a0;
        b1 = 1;
    }

    int      mode = 0;
    uint64_t M    = 0;
    uint64_t N    = 0;
    uint64_t K    = 0;
    if (a1 == b0)
    {
        mode = 1;
        M    = a0;
        K    = a1;
        N    = b1;
    }
    else if (a0 == b0)
    {
        mode = 2;
        M    = a1;
        K    = a0;
        N    = b1;
    }
    else
    {
        parse_errmsg("matmul: incompatible image dimensions");
        return mk_string("");
    }

    uint32_t outsz[3] = { 1, 1, 1 };
    long     onax     = 2;
    if (mode == 1 && is_3d_a)
    {
        outsz[0] = im1->md[0].size[0];
        outsz[1] = im1->md[0].size[1];
        onax     = (N == 1) ? 2 : 3;
        outsz[2] = (uint32_t) N;
    }
    else if (N == 1 && (im2->md[0].naxis == 1 || mode == 2))
    {
        onax     = 1;
        outsz[0] = (uint32_t) M;
    }
    else
    {
        onax     = 2;
        outsz[0] = (uint32_t) M;
        outsz[1] = (uint32_t) N;
    }

    uint8_t out_dt =
        (im1->md[0].datatype == _DATATYPE_DOUBLE || im2->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;

    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    create_image_ID(tmpn, onax, outsz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("matmul: failed to allocate output image");
        return mk_string("");
    }
    IMAGE *outim = &data.core.image[tid];

    double *bufA = (double *) malloc(sizeof(double) * im1->md[0].nelement);
    double *bufB = (double *) malloc(sizeof(double) * im2->md[0].nelement);
    if (bufA == NULL || bufB == NULL)
    {
        free(bufA);
        free(bufB);
        parse_errmsg("matmul: memory allocation failed");
        return mk_string("");
    }
    copy_im_to_double_buf(im1, bufA);
    copy_im_to_double_buf(im2, bufB);

    for (uint64_t n = 0; n < N; n++)
    {
        for (uint64_t m = 0; m < M; m++)
        {
            double acc = 0.0;
            if (mode == 1)
            {
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += bufA[k * M + m] * bufB[n * K + k];
                }
            }
            else
            {
                const double *rowA = &bufA[m * K];
                const double *colB = &bufB[n * K];
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * colB[k];
                }
            }
            if (out_dt == _DATATYPE_DOUBLE)
            {
                outim->array.D[n * M + m] = acc;
            }
            else
            {
                outim->array.F[n * M + m] = (float) acc;
            }
        }
    } // for (uint64_t n = 0; n < N; n++)

    free(bufA);
    free(bufB);
    return mk_string(tmpn);
}

/**
 * @brief Evaluate image-to-scalar reduction functions (`TOK_FUNC_IM_D`, `TOK_FUNC_IMD_D`, `TOK_FUNC_IMIM_D`).
 */
static val_t func_eval_image_reduction(const cli_token *ftok)
{
    if (ftok->type == TOK_FUNC_IM_D)
    {
        val_t arg;
        if (parse_func_args(1, &arg, NULL) != 0)
        {
            return mk_double(0);
        }
        if (arg.type != VAL_STRING)
        {
            parse_errmsg("Expected image argument");
            return mk_double(0);
        }
        if (!check_image(arg.sval))
        {
            return mk_double(0);
        }
        double res = ((double (*)(const char *)) ftok->fnctptr)(arg.sval);
        if (data.core.Debug > 0)
        {
            printf("double=func(image)\n");
        }
        return mk_double(res);
    }

    val_t args[2];
    if (parse_func_args(2, args, "Expected ','") != 0)
    {
        return mk_double(0);
    }
    if (ftok->type == TOK_FUNC_IMD_D)
    {
        if (args[0].type != VAL_STRING)
        {
            parse_errmsg("Expected image argument");
            return mk_double(0);
        }
        if (!check_image(args[0].sval))
        {
            return mk_double(0);
        }
        double res =
            ((double (*)(const char *, double)) ftok->fnctptr)(args[0].sval, to_double(args[1]));
        return mk_double(res);
    }

    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING)
    {
        parse_errmsg("Expected two image arguments");
        return mk_double(0);
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_double(0);
    }
    double res =
        ((double (*)(const char *, const char *)) ftok->fnctptr)(args[0].sval, args[1].sval);
    return mk_double(res);
}

/**
 * @brief Evaluate string & format functions (`strlen`, `toupper`, `tolower`, `substr`, `hex`, `oct`, `bin`).
 */
static val_t func_eval_string_or_format(const cli_token *ftok)
{
    if (ftok->type == TOK_FUNC_S_D)
    {
        val_t arg;
        if (parse_func_args(1, &arg, NULL) != 0)
        {
            return mk_double(0);
        }
        if (arg.type == VAL_STRING)
        {
            const char *cv = cli_var_get(arg.sval);
            return mk_long((long) strlen(cv != NULL ? cv : arg.sval));
        }
        char numstr[64];
        if (arg.type == VAL_LONG)
        {
            snprintf(numstr, sizeof(numstr), "%ld", arg.lval);
        }
        else
        {
            snprintf(numstr, sizeof(numstr), "%g", arg.dval);
        }
        return mk_long((long) strlen(numstr));
    }

    if (ftok->type == TOK_FUNC_S_S)
    {
        val_t arg;
        if (parse_func_args(1, &arg, NULL) != 0)
        {
            return mk_double(0);
        }
        const char *sv = NULL;
        char        numbuf[64];
        if (arg.type == VAL_STRING)
        {
            const char *cv = cli_var_get(arg.sval);
            sv             = cv ? cv : arg.sval;
        }
        else if (arg.type == VAL_LONG)
        {
            snprintf(numbuf, sizeof(numbuf), "%ld", arg.lval);
            sv = numbuf;
        }
        else
        {
            snprintf(numbuf, sizeof(numbuf), "%g", arg.dval);
            sv = numbuf;
        }

        char result[CLI_CALC_TOKEN_MAXLEN];
        strncpy(result, sv, sizeof(result) - 1);
        result[sizeof(result) - 1] = '\0';

        int is_upper = (strncmp(ftok->sval, "toupper(", 8) == 0);
        for (int i = 0; result[i]; i++)
        {
            result[i] = is_upper ? (char) toupper((unsigned char) result[i])
                                 : (char) tolower((unsigned char) result[i]);
        }
        return mk_string(result);
    }

    if (ftok->type == TOK_FUNC_SDD_S)
    {
        val_t args[3];
        if (parse_func_args(3, args, "substr: need 3 args") != 0)
        {
            return mk_double(0);
        }
        if (args[0].type != VAL_STRING)
        {
            parse_errmsg("substr: first arg must be string");
            return mk_double(0);
        }
        const char *cv   = cli_var_get(args[0].sval);
        const char *sv   = cv ? cv : args[0].sval;
        int         slen = (int) strlen(sv);
        int         off  = (int) to_double(args[1]);
        int         len  = (int) to_double(args[2]);
        if (off < 0)
        {
            off = 0;
        }
        if (off > slen)
        {
            off = slen;
        }
        if (len < 0)
        {
            len = 0;
        }
        if (off + len > slen)
        {
            len = slen - off;
        }
        char result[CLI_CALC_TOKEN_MAXLEN];
        memcpy(result, sv + off, (size_t) len);
        result[len] = '\0';
        return mk_string(result);
    }

    /* TOK_FUNC_D_S: hex(n), oct(n), bin(n) */
    val_t arg;
    if (parse_func_args(1, &arg, NULL) != 0)
    {
        return mk_double(0);
    }
    long        iv = (arg.type == VAL_LONG) ? arg.lval : (long) to_double(arg);
    char        result[CLI_CALC_TOKEN_MAXLEN];
    const char *fn = ftok->sval;
    if (strncmp(fn, "hex(", 4) == 0)
    {
        snprintf(result, sizeof(result), "0x%lx", iv);
    }
    else if (strncmp(fn, "oct(", 4) == 0)
    {
        snprintf(result, sizeof(result), "0o%lo", iv);
    }
    else
    {
        char *w = result;
        *w++    = '0';
        *w++    = 'b';
        if (iv == 0)
        {
            *w++ = '0';
        }
        else
        {
            unsigned long v = (unsigned long) iv; // two's complement for negative values
            char          bits[65];
            int           bi = 0;
            while (v > 0 && bi < 64)
            {
                bits[bi++] = (v & 1) ? '1' : '0';
                v >>= 1;
            }
            for (int i = bi - 1; i >= 0; i--)
            {
                *w++ = bits[i];
            }
        }
        *w = '\0';
    }
    return mk_string(result);
}

/**
 * @brief Dispatch a built-in calculator function call token @ftok.
 */
val_t parse_funccall(cli_token *ftok)
{
    if (parse_error || eval_error)
    {
        return mk_double(0);
    }

    switch (ftok->type)
    {
    case TOK_FUNC_D_D:
        return func_eval_d_d(ftok);
    case TOK_FUNC_DD_D:
        return func_eval_dd_d(ftok);
    case TOK_FUNC_DDD_D:
        return func_eval_ddd_d(ftok);
    case TOK_FUNC_WHERE:
        return func_where();
    case TOK_FUNC_IMIM_IM:
        return func_eval_matmul();
    case TOK_FUNC_IM_D:
    case TOK_FUNC_IMD_D:
    case TOK_FUNC_IMIM_D:
        return func_eval_image_reduction(ftok);
    case TOK_FUNC_S_D:
    case TOK_FUNC_S_S:
    case TOK_FUNC_SDD_S:
    case TOK_FUNC_D_S:
        return func_eval_string_or_format(ftok);
    case TOK_FUNC_SSS_S:
        return func_replace();
    default:
        parse_errmsg("Unknown function type");
        return mk_double(0);
    }
}
