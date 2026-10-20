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

// MILK_CMAKE_REQUEST_BLAS
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"
#include "cli_calc_internal.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "milk_blas_lapacke.h"

/** parse_func_argv() - @nmin to @nmax comma-separated arguments and `)`; how many, or -1 on error. */
static int parse_func_argv(int nmin, int nmax, val_t *args, const char *comma_msg)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    int n                            = 0;

    for (; n < nmax; n++)
    {
        if (n > 0)
        {
            if (n >= nmin && cur_func()->type == TOK_RPAREN)
            {
                break;
            }
            if (cur_func()->type != TOK_COMMA)
            {
                parse_errmsg(comma_msg != NULL ? comma_msg : "Expected ','");
                return -1;
            }
            advance_func();
        }
        args[n] = parse_expr(0);
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
    return n;
}

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
    return (parse_func_argv(nargs, nargs, args, comma_msg) < 0) ? -1 : 0;
}

/** reject_complex() - 1 (with an error) if one of the @n values is a complex image. */
static int reject_complex(const val_t *v, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (cli_calc_complex_image(&v[i]) != NULL)
        {
            parse_errmsg("Function of real pixels applied to a complex image; use real(), imag() "
                         "or abs() first");
            return 1;
        }
    }
    return 0;
}

/** func_eval_cplx() - real(z), imag(z), arg(z), conj(z), cplx(re, im). */
static val_t func_eval_cplx(const cli_token *ftok)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    val_t args[2];
    int   n = 0;
    do
    {
        if (n == 2)
        {
            parse_errmsg("Too many arguments");
            return mk_double(0);
        }
        args[n++] = parse_expr(0);
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }
    } while (cur_func()->type == TOK_COMMA && advance_func());
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_double(0);
    }
    advance_func();
    return cli_calc_complex_func(ftok->sval, args, n);
}

val_t cli_calc_where(val_t c, val_t t, val_t f)
{
    if (!check_image(c.sval) || (t.type == VAL_STRING && !check_image(t.sval)) ||
        (f.type == VAL_STRING && !check_image(f.sval)))
    {
        return mk_string("");
    }
    const IMAGE *ci = &data.core.image[image_ID(c.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *ti =
        (t.type == VAL_STRING)
            ? &data.core.image[image_ID(t.sval, data.core.image, data.core.NB_MAX_IMAGE)]
            : NULL;
    const IMAGE *fi =
        (f.type == VAL_STRING)
            ? &data.core.image[image_ID(f.sval, data.core.image, data.core.NB_MAX_IMAGE)]
            : NULL;
    const char *sel = cli_calc_select(ci, ti, (ti == NULL) ? to_double(t) : 0.0, fi,
                                      (fi == NULL) ? to_double(f) : 0.0);
    if (sel != NULL || parse_error)
    {
        return mk_string((sel != NULL) ? sel : "");
    }

    /* broadcasting (sizes differ): mask arithmetic, as the binary operators broadcast */
    const char *tmask  = alloc_tmpname();
    const char *timask = alloc_tmpname();
    const char *tpart  = alloc_tmpname();
    const char *fpart  = alloc_tmpname();
    const char *tmpn   = alloc_tmpname();
    arith_image_csttestne(c.sval, 0.0, tmask);
    arith_image_cstteste(c.sval, 0.0, timask);
    if (ti != NULL)
    {
        arith_image_mult(t.sval, tmask, tpart);
    }
    else
    {
        arith_image_cstmult(tmask, to_double(t), tpart);
    }
    if (fi != NULL)
    {
        arith_image_mult(f.sval, timask, fpart);
    }
    else
    {
        arith_image_cstmult(timask, to_double(f), fpart);
    }
    arith_image_add(tpart, fpart, tmpn);
    return mk_string(tmpn);
}

/** func_where() - `where(cond, T, F)`: a number picks one, an image selects per pixel. */
static val_t func_where(void)
{
    val_t args[3];
    if (parse_func_args(3, args, "Expected ','") != 0 || reject_complex(args, 3))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING)
    {
        return (to_double(args[0]) != 0.0) ? args[1] : args[2];
    }
    return cli_calc_where(args[0], args[1], args[2]);
}

/** func_coord() - ix/iy/ir/itheta(im[, cx, cy]): a new image of im's size, centred on (xsize/2, ysize/2). */
static val_t func_coord(const cli_token *ftok)
{
    val_t args[3];
    int   n = parse_func_argv(1, 3, args, "Expected ','");
    if (n < 0)
    {
        return mk_double(0);
    }
    if (n == 2 || args[0].type != VAL_STRING ||
        (n == 3 && (args[1].type == VAL_STRING || args[2].type == VAL_STRING)))
    {
        parse_errmsg("Expected an image, then optionally the centre: x, y");
        return mk_double(0);
    }
    if (!check_image(args[0].sval))
    {
        return mk_string("");
    }
    const IMAGE *im =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    double      cx   = (n == 3) ? to_double(args[1]) : (double) (im->md[0].size[0] / 2);
    double      cy   = (n == 3) ? to_double(args[2])
                                : (double) ((im->md[0].naxis > 1) ? im->md[0].size[1] / 2 : 0);
    const char *tmpn = cli_calc_coord(im, cli_calc_coord_kind(ftok), cx, cy);
    return mk_string((tmpn != NULL) ? tmpn : "");
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

    const IMAGE *z = cli_calc_complex_image(&arg);
    if (z != NULL && ftok->fnctptr == (double (*)()) fabs)
    {
        return cli_calc_complex_part(z, 2); /* |z| */
    }
    if (reject_complex(&arg, 1))
    {
        return mk_double(0);
    }

    if (arg.type == VAL_STRING)
    {
        if (!check_image(arg.sval))
        {
            return mk_string("");
        }
        uint8_t cdt = cli_calc_cast_datatype(ftok->fnctptr);
        if (cdt != 0)
        {
            const IMAGE *src =
                &data.core.image[image_ID(arg.sval, data.core.image, data.core.NB_MAX_IMAGE)];
            uint32_t    sz[3] = { src->md[0].size[0], src->md[0].size[1], src->md[0].size[2] };
            const char *tmpn  = alloc_tmpname();
            imageID     tid   = -1;
            create_image_ID(tmpn, src->md[0].naxis, sz, cdt, 0, 10, 0, &tid);
            if (tid == -1 || cli_calc_inplace(&data.core.image[tid], TOK_EQUAL, src, 0.0) != 0)
            {
                if (tid == -1)
                {
                    parse_errmsg("Failed to allocate output image");
                }
                return mk_string("");
            }
            return mk_string(tmpn);
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
    if (parse_func_args(2, args, "Expected ','") != 0 || reject_complex(args, 2))
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
    if (parse_func_args(3, args, "Expected ','") != 0 || reject_complex(args, 3))
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

static inline double read_im_pix(const IMAGE *im, uint8_t dt, uint64_t idx)
{
    switch (dt)
    {
    case _DATATYPE_FLOAT:
        return (double) im->array.F[idx];
    case _DATATYPE_DOUBLE:
        return im->array.D[idx];
    case _DATATYPE_UINT8:
        return (double) im->array.UI8[idx];
    case _DATATYPE_INT8:
        return (double) im->array.SI8[idx];
    case _DATATYPE_UINT16:
        return (double) im->array.UI16[idx];
    case _DATATYPE_INT16:
        return (double) im->array.SI16[idx];
    case _DATATYPE_UINT32:
        return (double) im->array.UI32[idx];
    case _DATATYPE_INT32:
        return (double) im->array.SI32[idx];
    case _DATATYPE_UINT64:
        return (double) im->array.UI64[idx];
    case _DATATYPE_INT64:
        return (double) im->array.SI64[idx];
    default:
        return 0.0;
    }
}

int cli_calc_transpose_check(const IMAGE *src, const IMAGE *dst)
{
    if (src == NULL || dst == NULL || !src->used || !dst->used || src == dst)
    {
        return -1;
    }
    if (cli_calc_is_complex_dt(src->md[0].datatype) ||
        (src->md[0].naxis != 1 && src->md[0].naxis != 2))
    {
        return -1;
    }
    if (dst->md[0].datatype != _DATATYPE_FLOAT && dst->md[0].datatype != _DATATYPE_DOUBLE)
    {
        return -1;
    }
    uint64_t M = src->md[0].size[0];
    uint64_t N = (src->md[0].naxis == 2) ? src->md[0].size[1] : 1;
    if (dst->md[0].naxis != 2 || dst->md[0].size[0] != N || dst->md[0].size[1] != M)
    {
        return -1;
    }
    return 0;
}

int cli_calc_transpose_into(const IMAGE *src, IMAGE *dst)
{
    uint64_t M       = src->md[0].size[0];
    uint64_t N       = (src->md[0].naxis == 2) ? src->md[0].size[1] : 1;
    uint8_t  src_dt  = src->md[0].datatype;
    uint8_t  dst_dt  = dst->md[0].datatype;
    uint64_t nel     = M * N;
    uint64_t tile_sz = 32;

    if (src_dt == _DATATYPE_FLOAT && dst_dt == _DATATYPE_FLOAT)
    {
        const float *restrict s = src->array.F;
        float *restrict d       = dst->array.F;
        if (M == 1 || N == 1)
        {
            __builtin_memcpy(d, s, sizeof(float) * nel);
            return 0;
        }
#pragma omp parallel for collapse(2) schedule(static) if (nel >= 65536)
        for (uint64_t m0 = 0; m0 < M; m0 += tile_sz)
        {
            for (uint64_t n0 = 0; n0 < N; n0 += tile_sz)
            {
                uint64_t mend = (m0 + tile_sz < M) ? m0 + tile_sz : M;
                uint64_t nend = (n0 + tile_sz < N) ? n0 + tile_sz : N;
                for (uint64_t n = n0; n < nend; n++)
                {
                    const float *restrict scol = &s[n * M];
                    for (uint64_t m = m0; m < mend; m++)
                    {
                        d[m * N + n] = scol[m];
                    }
                }
            }
        }
        return 0;
    }

    if (src_dt == _DATATYPE_DOUBLE && dst_dt == _DATATYPE_DOUBLE)
    {
        const double *restrict s = src->array.D;
        double *restrict d       = dst->array.D;
        if (M == 1 || N == 1)
        {
            __builtin_memcpy(d, s, sizeof(double) * nel);
            return 0;
        }
#pragma omp parallel for collapse(2) schedule(static) if (nel >= 65536)
        for (uint64_t m0 = 0; m0 < M; m0 += tile_sz)
        {
            for (uint64_t n0 = 0; n0 < N; n0 += tile_sz)
            {
                uint64_t mend = (m0 + tile_sz < M) ? m0 + tile_sz : M;
                uint64_t nend = (n0 + tile_sz < N) ? n0 + tile_sz : N;
                for (uint64_t n = n0; n < nend; n++)
                {
                    const double *restrict scol = &s[n * M];
                    for (uint64_t m = m0; m < mend; m++)
                    {
                        d[m * N + n] = scol[m];
                    }
                }
            }
        }
        return 0;
    }

    for (uint64_t m = 0; m < M; m++)
    {
        for (uint64_t n = 0; n < N; n++)
        {
            double v = read_im_pix(src, src_dt, n * M + m);
            if (dst_dt == _DATATYPE_DOUBLE)
            {
                dst->array.D[m * N + n] = v;
            }
            else
            {
                dst->array.F[m * N + n] = (float) v;
            }
        }
    }
    return 0;
}

static val_t func_eval_transpose(void)
{
    val_t arg;
    if (parse_func_args(1, &arg, NULL) != 0)
    {
        return mk_double(0);
    }
    if (reject_complex(&arg, 1))
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
    uint32_t outsz[3] = { (src->md[0].naxis == 2) ? src->md[0].size[1] : 1, src->md[0].size[0], 1 };
    uint8_t out_dt = (src->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    create_image_ID(tmpn, 2, outsz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("transpose: failed to allocate output image");
        return mk_string("");
    }
    if (cli_calc_transpose_into(src, &data.core.image[tid]) != 0)
    {
        return mk_string("");
    }
    return mk_string(tmpn);
}

int cli_calc_reshape_check(const IMAGE *src, const IMAGE *dst)
{
    if (src == NULL || dst == NULL || !src->used || !dst->used || src == dst ||
        src->md[0].nelement != dst->md[0].nelement)
    {
        return -1;
    }
    if (src->md[0].datatype != dst->md[0].datatype &&
        (cli_calc_is_complex_dt(src->md[0].datatype) ||
         cli_calc_is_complex_dt(dst->md[0].datatype)))
    {
        return -1;
    }
    return 0;
}

int cli_calc_reshape_into(const IMAGE *src, IMAGE *dst)
{
    if (src->md[0].datatype == dst->md[0].datatype)
    {
        int tsz = ImageStreamIO_typesize(src->md[0].datatype);
        if (tsz <= 0)
        {
            return -1;
        }
        __builtin_memcpy(dst->array.raw, src->array.raw,
                         (size_t) src->md[0].nelement * (size_t) tsz);
        return 0;
    }
    return cli_calc_inplace(dst, TOK_EQUAL, src, 0.0);
}

static val_t func_eval_reshape(void)
{
    val_t args[4];
    int   n = parse_func_argv(2, 4, args, "Expected ','");
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
    create_image_ID(tmpn, ndims, outsz, src->md[0].datatype, 0, 10, 0, &tid);
    if (tid == -1 || cli_calc_reshape_into(src, &data.core.image[tid]) != 0)
    {
        if (tid == -1)
        {
            parse_errmsg("reshape: failed to allocate output image");
        }
        return mk_string("");
    }
    return mk_string(tmpn);
}

static val_t func_eval_vec(void)
{
    val_t args[256];
    int   n = parse_func_argv(1, 256, args, "Expected ','");
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
    create_image_ID(tmpn, 1, outsz, _DATATYPE_FLOAT, 0, 10, 0, &tid);
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

static int cli_calc_matmul_dims(const IMAGE *im1,
                                int          transA,
                                const IMAGE *im2,
                                int          transB,
                                int         *out_tA,
                                int         *out_tB,
                                uint64_t    *out_M,
                                uint64_t    *out_K,
                                uint64_t    *out_N,
                                long        *out_nax,
                                uint32_t     out_sz[3])
{
    uint64_t a0      = im1->md[0].size[0];
    uint64_t a1      = 1;
    int      is_3d_a = 0;
    if (im1->md[0].naxis == 3)
    {
        if (transA)
        {
            return -1;
        }
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
        if (transB)
        {
            return -1;
        }
        b0 = (uint64_t) im2->md[0].size[0] * im2->md[0].size[1];
        b1 = im2->md[0].size[2];
    }
    else if (im2->md[0].naxis >= 2 && !(is_3d_a && !transA && !transB &&
                                        a0 == (uint64_t) im2->md[0].size[0] * im2->md[0].size[1]))
    {
        b1 = im2->md[0].size[1];
    }
    else if (is_3d_a && !transA && !transB && im2->md[0].naxis == 2 &&
             a0 == (uint64_t) im2->md[0].size[0] * im2->md[0].size[1])
    {
        b0 = a0;
        b1 = 1;
    }

    int      tA = 0;
    int      tB = 0;
    uint64_t M  = 0;
    uint64_t K  = 0;
    uint64_t N  = 0;
    if (!transA && !transB)
    {
        if (a1 == b0)
        {
            tA = 0;
            tB = 0;
            M  = a0;
            K  = a1;
            N  = b1;
        }
        else if (a0 == b0)
        {
            tA = 1;
            tB = 0;
            M  = a1;
            K  = a0;
            N  = b1;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        uint64_t eff_a0 = transA ? a1 : a0;
        uint64_t eff_a1 = transA ? a0 : a1;
        uint64_t eff_b0 = transB ? b1 : b0;
        uint64_t eff_b1 = transB ? b0 : b1;
        if (eff_a1 != eff_b0)
        {
            return -1;
        }
        tA = transA;
        tB = transB;
        M  = eff_a0;
        K  = eff_a1;
        N  = eff_b1;
    }

    out_sz[0] = 1;
    out_sz[1] = 1;
    out_sz[2] = 1;
    if (tA == 0 && tB == 0 && is_3d_a)
    {
        out_sz[0] = im1->md[0].size[0];
        out_sz[1] = im1->md[0].size[1];
        *out_nax  = (N == 1) ? 2 : 3;
        out_sz[2] = (uint32_t) N;
    }
    else if (N == 1 && ((im2->md[0].naxis == 1 && !transB) || (tA == 1 && tB == 0)))
    {
        *out_nax  = 1;
        out_sz[0] = (uint32_t) M;
    }
    else
    {
        *out_nax  = 2;
        out_sz[0] = (uint32_t) M;
        out_sz[1] = (uint32_t) N;
    }
    *out_tA = tA;
    *out_tB = tB;
    *out_M  = M;
    *out_K  = K;
    *out_N  = N;
    return 0;
}

int cli_calc_matmul_check(const IMAGE *im1,
                          int          transA,
                          const IMAGE *im2,
                          int          transB,
                          const IMAGE *dst,
                          int          allow_reshape)
{
    if (im1 == NULL || im2 == NULL || dst == NULL || !im1->used || !im2->used || !dst->used ||
        dst == im1 || dst == im2)
    {
        return -1;
    }
    if (cli_calc_is_complex_dt(im1->md[0].datatype) || cli_calc_is_complex_dt(im2->md[0].datatype))
    {
        return -1;
    }
    if (dst->md[0].datatype != _DATATYPE_FLOAT && dst->md[0].datatype != _DATATYPE_DOUBLE)
    {
        return -1;
    }
    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (cli_calc_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        return -1;
    }
    if (dst->md[0].nelement != M * N)
    {
        return -1;
    }
    if (allow_reshape)
    {
        return 0;
    }
    if (N == 1)
    {
        if (dst->md[0].size[0] != outsz[0])
        {
            return -1;
        }
    }
    else if (dst->md[0].naxis != onax || dst->md[0].size[0] != outsz[0] ||
             dst->md[0].size[1] != outsz[1] || (onax == 3 && dst->md[0].size[2] != outsz[2]))
    {
        return -1;
    }
    return 0;
}

static void matmul_kernel_f(const float *restrict A,
                            int tA,
                            const float *restrict B,
                            int tB,
                            float *restrict C,
                            uint64_t M,
                            uint64_t K,
                            uint64_t N)
{
#ifdef HAVE_BLAS
    if (N == 1)
    {
        cblas_sgemv(CblasColMajor, tA ? CblasTrans : CblasNoTrans, (int) (tA ? K : M),
                    (int) (tA ? M : K), 1.0f, A, (int) (tA ? K : M), B, 1, 0.0f, C, 1);
        return;
    }
    cblas_sgemm(CblasColMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans,
                (int) M, (int) N, (int) K, 1.0f, A, (int) (tA ? K : M), B, (int) (tB ? N : K), 0.0f,
                C, (int) M);
#else
    if (!tA)
    {
        const uint64_t mblk  = 128;
        uint64_t       nmblk = (M + mblk - 1) / mblk;
#    pragma omp parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t mb = 0; mb < nmblk; mb++)
            {
                uint64_t m0          = mb * mblk;
                uint64_t mend        = (m0 + mblk < M) ? m0 + mblk : M;
                uint64_t mlen        = mend - m0;
                float *restrict cblk = &C[n * M + m0];
                memset(cblk, 0, sizeof(float) * mlen);
                for (uint64_t k = 0; k < K; k++)
                {
                    float b_kn                 = tB ? B[k * N + n] : B[n * K + k];
                    const float *restrict colA = &A[k * M + m0];
#    pragma omp simd
                    for (uint64_t m = 0; m < mlen; m++)
                    {
                        cblk[m] += colA[m] * b_kn;
                    }
                }
            }
        }
    }
    else if (!tB)
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const float *restrict rowA = &A[m * K];
                const float *restrict colB = &B[n * K];
                float acc                  = 0.0f;
#    pragma omp simd reduction(+ : acc)
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * colB[k];
                }
                C[n * M + m] = acc;
            }
        }
    }
    else
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const float *restrict rowA = &A[m * K];
                float acc                  = 0.0f;
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * B[k * N + n];
                }
                C[n * M + m] = acc;
            }
        }
    }
#endif
}

static void matmul_kernel_d(const double *restrict A,
                            int tA,
                            const double *restrict B,
                            int tB,
                            double *restrict C,
                            uint64_t M,
                            uint64_t K,
                            uint64_t N)
{
#ifdef HAVE_BLAS
    if (N == 1)
    {
        cblas_dgemv(CblasColMajor, tA ? CblasTrans : CblasNoTrans, (int) (tA ? K : M),
                    (int) (tA ? M : K), 1.0, A, (int) (tA ? K : M), B, 1, 0.0, C, 1);
        return;
    }
    cblas_dgemm(CblasColMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans,
                (int) M, (int) N, (int) K, 1.0, A, (int) (tA ? K : M), B, (int) (tB ? N : K), 0.0,
                C, (int) M);
#else
    if (!tA)
    {
        const uint64_t mblk  = 64;
        uint64_t       nmblk = (M + mblk - 1) / mblk;
#    pragma omp parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t mb = 0; mb < nmblk; mb++)
            {
                uint64_t m0           = mb * mblk;
                uint64_t mend         = (m0 + mblk < M) ? m0 + mblk : M;
                uint64_t mlen         = mend - m0;
                double *restrict cblk = &C[n * M + m0];
                memset(cblk, 0, sizeof(double) * mlen);
                for (uint64_t k = 0; k < K; k++)
                {
                    double b_kn                 = tB ? B[k * N + n] : B[n * K + k];
                    const double *restrict colA = &A[k * M + m0];
#    pragma omp simd
                    for (uint64_t m = 0; m < mlen; m++)
                    {
                        cblk[m] += colA[m] * b_kn;
                    }
                }
            }
        }
    }
    else if (!tB)
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const double *restrict rowA = &A[m * K];
                const double *restrict colB = &B[n * K];
                double acc                  = 0.0;
#    pragma omp simd reduction(+ : acc)
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * colB[k];
                }
                C[n * M + m] = acc;
            }
        }
    }
    else
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const double *restrict rowA = &A[m * K];
                double acc                  = 0.0;
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * B[k * N + n];
                }
                C[n * M + m] = acc;
            }
        }
    }
#endif
}

int cli_calc_matmul_into(const IMAGE *im1, int transA, const IMAGE *im2, int transB, IMAGE *outim)
{
    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (cli_calc_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        parse_errmsg("matmul: incompatible image dimensions");
        return -1;
    }

    uint8_t dt1    = im1->md[0].datatype;
    uint8_t dt2    = im2->md[0].datatype;
    uint8_t out_dt = outim->md[0].datatype;

    if (dt1 == _DATATYPE_FLOAT && dt2 == _DATATYPE_FLOAT && out_dt == _DATATYPE_FLOAT)
    {
        matmul_kernel_f(im1->array.F, tA, im2->array.F, tB, outim->array.F, M, K, N);
        return 0;
    }
    if (dt1 == _DATATYPE_DOUBLE && dt2 == _DATATYPE_DOUBLE && out_dt == _DATATYPE_DOUBLE)
    {
        matmul_kernel_d(im1->array.D, tA, im2->array.D, tB, outim->array.D, M, K, N);
        return 0;
    }

    double *bufA = (double *) malloc(sizeof(double) * im1->md[0].nelement);
    double *bufB = (double *) malloc(sizeof(double) * im2->md[0].nelement);
    if (bufA == NULL || bufB == NULL)
    {
        free(bufA);
        free(bufB);
        parse_errmsg("matmul: memory allocation failed");
        return -1;
    }
    for (uint64_t i = 0; i < im1->md[0].nelement; i++)
    {
        bufA[i] = read_im_pix(im1, dt1, i);
    }
    for (uint64_t i = 0; i < im2->md[0].nelement; i++)
    {
        bufB[i] = read_im_pix(im2, dt2, i);
    }

    if (out_dt == _DATATYPE_DOUBLE)
    {
        matmul_kernel_d(bufA, tA, bufB, tB, outim->array.D, M, K, N);
    }
    else
    {
        double *bufC = (double *) malloc(sizeof(double) * M * N);
        if (bufC == NULL)
        {
            free(bufA);
            free(bufB);
            parse_errmsg("matmul: memory allocation failed");
            return -1;
        }
        matmul_kernel_d(bufA, tA, bufB, tB, bufC, M, K, N);
        for (uint64_t i = 0; i < M * N; i++)
        {
            outim->array.F[i] = (float) bufC[i];
        }
        free(bufC);
    }

    free(bufA);
    free(bufB);
    return 0;
}

static int is_top_level_transpose(cli_token_type end_tok)
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
    int is_top = (depth == 0 && cur_func()->type == end_tok);
    cli_calc_tok_seek(saved);
    return is_top;
}

static int parse_matmul_arg(val_t *out, int *out_trans, cli_token_type end_tok)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    int trans                        = 0;

    if (is_top_level_transpose(end_tok))
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

static val_t func_eval_matmul(void)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    val_t args[2];
    int   transA = 0;
    int   transB = 0;

    if (parse_matmul_arg(&args[0], &transA, TOK_COMMA) != 0)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_double(0);
    }
    advance_func();
    if (parse_matmul_arg(&args[1], &transB, TOK_RPAREN) != 0)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_double(0);
    }
    advance_func();

    if (reject_complex(args, 2))
    {
        return mk_string("");
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

    IMAGE *im1 = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *im2 = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];

    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (cli_calc_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
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
    create_image_ID(tmpn, onax, outsz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("matmul: failed to allocate output image");
        return mk_string("");
    }
    if (cli_calc_matmul_into(im1, transA, im2, transB, &data.core.image[tid]) != 0)
    {
        return mk_string("");
    }
    return mk_string(tmpn);
}

typedef enum
{
    RED_SUM,
    RED_MEAN,
    RED_MIN,
    RED_MAX,
    RED_STD,
    RED_VAR,
    RED_PERC /* masked only: imedian, perc */
} reduce_op;

/** reduce_lane_im() - reduce @n values of @im starting at @base with @stride using @op. */
static double reduce_lane_im(const IMAGE *im,
                             uint8_t      dt,
                             uint64_t     base,
                             uint64_t     n,
                             uint64_t     stride,
                             reduce_op    op)
{
    double acc = (op == RED_MIN) ? INFINITY : (op == RED_MAX) ? -INFINITY : 0.0;
    if (dt == _DATATYPE_FLOAT)
    {
        const float *restrict p = im->array.F + base;
        for (uint64_t k = 0; k < n; k++)
        {
            double v = (double) p[k * stride];
            acc      = (op == RED_MIN) ? fmin(acc, v) : (op == RED_MAX) ? fmax(acc, v) : acc + v;
        }
        if (op == RED_SUM || op == RED_MIN || op == RED_MAX)
        {
            return acc;
        }
        double mean = acc / (double) n;
        if (op == RED_MEAN)
        {
            return mean;
        }
        double s2 = 0.0;
        for (uint64_t k = 0; k < n; k++)
        {
            double d = (double) p[k * stride] - mean;
            s2 += d * d;
        }
        return (op == RED_VAR) ? s2 / (double) n : sqrt(s2 / (double) n);
    }
    if (dt == _DATATYPE_DOUBLE)
    {
        const double *restrict p = im->array.D + base;
        for (uint64_t k = 0; k < n; k++)
        {
            double v = p[k * stride];
            acc      = (op == RED_MIN) ? fmin(acc, v) : (op == RED_MAX) ? fmax(acc, v) : acc + v;
        }
        if (op == RED_SUM || op == RED_MIN || op == RED_MAX)
        {
            return acc;
        }
        double mean = acc / (double) n;
        if (op == RED_MEAN)
        {
            return mean;
        }
        double s2 = 0.0;
        for (uint64_t k = 0; k < n; k++)
        {
            double d = p[k * stride] - mean;
            s2 += d * d;
        }
        return (op == RED_VAR) ? s2 / (double) n : sqrt(s2 / (double) n);
    }
    for (uint64_t k = 0; k < n; k++)
    {
        double v = read_im_pix(im, dt, base + k * stride);
        acc      = (op == RED_MIN) ? fmin(acc, v) : (op == RED_MAX) ? fmax(acc, v) : acc + v;
    }
    if (op == RED_SUM || op == RED_MIN || op == RED_MAX)
    {
        return acc;
    }
    double mean = acc / (double) n;
    if (op == RED_MEAN)
    {
        return mean;
    }
    double s2 = 0.0;
    for (uint64_t k = 0; k < n; k++)
    {
        double d = read_im_pix(im, dt, base + k * stride) - mean;
        s2 += d * d;
    }
    return (op == RED_VAR) ? s2 / (double) n : sqrt(s2 / (double) n);
}

/** reduce_axis() - reduce image @name along @axis into a temporary image with that axis removed. */
static val_t reduce_axis(const char *name, long axis, reduce_op op)
{
    IMAGE  *im  = &data.core.image[image_ID(name, data.core.image, data.core.NB_MAX_IMAGE)];
    int     nax = im->md[0].naxis;
    uint8_t dt  = im->md[0].datatype;
    if (axis < 0 || axis >= nax)
    {
        parse_errmsg("Axis out of range");
        return mk_double(0);
    }
    if (dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        parse_errmsg("Axis reduction of a complex image is not supported");
        return mk_double(0);
    }

    uint64_t inner = 1;
    uint64_t outer = 1;
    uint64_t n     = im->md[0].size[axis];
    uint32_t osz[3];
    int      onax = 0;
    for (int a = 0; a < nax; a++)
    {
        if (a < axis)
        {
            inner *= im->md[0].size[a];
        }
        else if (a > axis)
        {
            outer *= im->md[0].size[a];
        }
        if (a != axis)
        {
            osz[onax++] = im->md[0].size[a];
        }
    }

    if (onax == 0) // 1-D input: the result is a scalar
    {
        return mk_double(reduce_lane_im(im, dt, 0, n, 1, op));
    }

    int         out_d = !(dt == _DATATYPE_FLOAT || dt == _DATATYPE_HALF || dt == _DATATYPE_UINT8 ||
                          dt == _DATATYPE_INT8 || dt == _DATATYPE_UINT16 || dt == _DATATYPE_INT16);
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    create_image_ID(tmpn, onax, osz, out_d ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return mk_string("");
    }
    IMAGE *oim = &data.core.image[tid];
    for (uint64_t o = 0; o < outer; o++)
    {
        for (uint64_t i = 0; i < inner; i++)
        {
            double r = reduce_lane_im(im, dt, o * n * inner + i, n, inner, op);
            if (out_d)
            {
                oim->array.D[o * inner + i] = r;
            }
            else
            {
                oim->array.F[o * inner + i] = (float) r;
            }
        }
    }
    return mk_string(tmpn);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *) a;
    double y = *(const double *) b;
    return (x > y) - (x < y);
}

/**
 * reduce_masked() - reduce the pixels of image @name where image @mask is not zero; RED_PERC takes the
 * value at fraction @f of the sorted pixels. NaN when the mask selects no pixel (a sum is 0).
 */
static val_t reduce_masked(const char *name, const char *mask, reduce_op op, double f)
{
    const IMAGE *im  = &data.core.image[image_ID(name, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *mk  = &data.core.image[image_ID(mask, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t     nel = im->md[0].nelement;
    uint8_t      idt = im->md[0].datatype;
    uint8_t      mdt = mk->md[0].datatype;
    uint64_t     n   = 0;

    int same = (mk->md[0].naxis == im->md[0].naxis);
    for (int a = 0; same && a < im->md[0].naxis; a++)
    {
        same = (mk->md[0].size[a] == im->md[0].size[a]);
    }
    if (!same)
    {
        parse_errmsg("The mask must have the image's size");
        return mk_double(0);
    }

    if (op == RED_PERC)
    {
        double *v = (double *) malloc(sizeof(double) * nel);
        if (v == NULL)
        {
            parse_errmsg("Memory allocation failed");
            return mk_double(0);
        }
        for (uint64_t i = 0; i < nel; i++)
        {
            if (read_im_pix(mk, mdt, i) != 0.0)
            {
                v[n++] = read_im_pix(im, idt, i);
            }
        }
        double r = NAN;
        if (n > 0)
        {
            long k = (long) (f * (double) n);
            qsort(v, n, sizeof(double), cmp_double);
            r = v[(k < 0) ? 0 : ((uint64_t) k >= n) ? (long) n - 1 : k];
        }
        free(v);
        return mk_double(r);
    }

    double acc = (op == RED_MIN) ? INFINITY : (op == RED_MAX) ? -INFINITY : 0.0;
    for (uint64_t i = 0; i < nel; i++)
    {
        if (read_im_pix(mk, mdt, i) != 0.0)
        {
            double v = read_im_pix(im, idt, i);
            acc      = (op == RED_MIN) ? fmin(acc, v) : (op == RED_MAX) ? fmax(acc, v) : acc + v;
            n++;
        }
    }
    if (n == 0)
    {
        return mk_double((op == RED_SUM) ? 0.0 : NAN);
    }
    if (op == RED_SUM || op == RED_MIN || op == RED_MAX)
    {
        return mk_double(acc);
    }
    double mean = acc / (double) n;
    if (op == RED_MEAN)
    {
        return mk_double(mean);
    }
    double s2 = 0.0;
    for (uint64_t i = 0; i < nel; i++)
    {
        if (read_im_pix(mk, mdt, i) != 0.0)
        {
            double d = read_im_pix(im, idt, i) - mean;
            s2 += d * d;
        }
    }
    return mk_double((op == RED_VAR) ? s2 / (double) n : sqrt(s2 / (double) n));
}

/** axis_reduce_op() - the reduce_op of reduction @fn, or -1 if it has no axis form. */
static int axis_reduce_op(double (*fn)())
{
    if (fn == (double (*)()) arith_image_total)
    {
        return RED_SUM;
    }
    if (fn == (double (*)()) arith_image_mean)
    {
        return RED_MEAN;
    }
    if (fn == (double (*)()) arith_image_min)
    {
        return RED_MIN;
    }
    if (fn == (double (*)()) arith_image_max)
    {
        return RED_MAX;
    }
    if (fn == (double (*)()) arith_image_rms)
    {
        return RED_STD;
    }
    if (fn == (double (*)()) arith_image_var)
    {
        return RED_VAR;
    }
    return -1;
}

/**
 * func_eval_image_reduction() - image-to-scalar reductions (`TOK_FUNC_IM_D`, `_IM_L`, `_IMD_D`, `_IMIM_D`);
 * `f(im, axis)` reduces along one axis instead and returns an image.
 */
static val_t func_eval_image_reduction(const cli_token *ftok)
{
    if (ftok->type == TOK_FUNC_IM_D || ftok->type == TOK_FUNC_IM_L)
    {
        cli_token *(*cur_func)(void)     = cli_calc_cur_func();
        cli_token *(*advance_func)(void) = cli_calc_advance_func();

        val_t arg      = parse_expr(0);
        val_t axisv    = mk_long(-1);
        int   has_axis = 0;
        if (!parse_error && !eval_error && cur_func()->type == TOK_COMMA)
        {
            advance_func();
            axisv    = parse_expr(0);
            has_axis = 1;
        }
        if (parse_error || eval_error)
        {
            return mk_double(0);
        }
        if (cur_func()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return mk_double(0);
        }
        advance_func();
        if (arg.type != VAL_STRING)
        {
            parse_errmsg("Expected image argument");
            return mk_double(0);
        }
        if (!check_image(arg.sval) || reject_complex(&arg, 1))
        {
            return mk_double(0);
        }
        if (has_axis)
        {
            int op = (ftok->type == TOK_FUNC_IM_D) ? axis_reduce_op(ftok->fnctptr) : -1;
            if (axisv.type == VAL_STRING) /* f(im, mask) */
            {
                op = (ftok->fnctptr == (double (*)()) arith_image_median) ? RED_PERC : op;
                if (op < 0)
                {
                    parse_errmsg("This function takes no mask argument");
                    return mk_double(0);
                }
                if (!check_image(axisv.sval) || reject_complex(&axisv, 1))
                {
                    return mk_double(0);
                }
                return reduce_masked(arg.sval, axisv.sval, (reduce_op) op, 0.5);
            }
            if (op < 0)
            {
                parse_errmsg("This function takes no axis argument");
                return mk_double(0);
            }
            return reduce_axis(arg.sval, (long) to_double(axisv), (reduce_op) op);
        }
        double res = ((double (*)(const char *)) ftok->fnctptr)(arg.sval);
        return (ftok->type == TOK_FUNC_IM_L) ? mk_long((long) res) : mk_double(res);
    }

    val_t args[3];
    int   nargs = parse_func_argv(2, (ftok->type == TOK_FUNC_IMD_D) ? 3 : 2, args, "Expected ','");
    if (nargs < 0 || reject_complex(args, nargs))
    {
        return mk_double(0);
    }
    if (ftok->type == TOK_FUNC_IMD_D)
    {
        if (args[0].type != VAL_STRING || (nargs == 3 && args[2].type != VAL_STRING))
        {
            parse_errmsg(nargs == 3 ? "Expected perc(image, fraction, mask)"
                                    : "Expected image argument");
            return mk_double(0);
        }
        if (!check_image(args[0].sval) || (nargs == 3 && !check_image(args[2].sval)))
        {
            return mk_double(0);
        }
        if (nargs == 3)
        {
            return reduce_masked(args[0].sval, args[2].sval, RED_PERC, to_double(args[1]));
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
    case TOK_FUNC_COORD:
        return func_coord(ftok);
    case TOK_FUNC_IMIM_IM:
        if (strncmp(ftok->sval, "transpose(", 10) == 0)
        {
            return func_eval_transpose();
        }
        if (strncmp(ftok->sval, "reshape(", 8) == 0)
        {
            return func_eval_reshape();
        }
        if (strncmp(ftok->sval, "vec(", 4) == 0)
        {
            return func_eval_vec();
        }
        return func_eval_matmul();
    case TOK_FUNC_IM_D:
    case TOK_FUNC_IM_L:
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
    case TOK_FUNC_CPLX:
        return func_eval_cplx(ftok);
    default:
        parse_errmsg("Unknown function type");
        return mk_double(0);
    }
}
