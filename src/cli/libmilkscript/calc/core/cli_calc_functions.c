// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_functions.c
 * @brief Calculator built-in math, conditional, coordinate, stencil, and spatial function calls.
 *
 * Role & Goal:
 *   Evaluates built-in mathematical, conditional (`where`), coordinate (`ix`, `iy`,
 *   `ir`, `itheta`, `rand`, `randn`), stencil/LUT (`shift`, `lut`), and spatial/remap
 *   (`medfilt3/5/7`, `pixremap`, `pixunmap`, `rprof`, `gauss`, `interp2`) functions
 *   inside calculator expressions (`parse_funccall()`). Image reductions, string
 *   functions, and user-defined functions live in `cli_calc_functions_red_str.c`.
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
 * cli_calc_parse_func_argv - Parse @nmin to @nmax comma-separated arguments and closing `)`
 * @nmin:      Minimum required arguments
 * @nmax:      Maximum allowed arguments
 * @args:      Output argument array of size @nmax
 * @comma_msg: Optional custom error message when a comma is missing
 *
 * Return: Number of arguments parsed, or -1 on error
 */
int cli_calc_parse_func_argv(int nmin, int nmax, val_t *args, const char *comma_msg)
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

/** parse_func_args() - parse @nargs comma-separated expression arguments followed by ')'. */
static int parse_func_args(int nargs, val_t *args, const char *comma_msg)
{
    return (cli_calc_parse_func_argv(nargs, nargs, args, comma_msg) < 0) ? -1 : 0;
}

/**
 * cli_calc_reject_complex - Report an error if any of the @n values is a complex image
 * @v: Argument array
 * @n: Number of arguments
 *
 * Return: 1 if a complex image was rejected, 0 otherwise
 */
int cli_calc_reject_complex(const val_t *v, int n)
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

/** func_eval_cplx() - evaluate `real(z)`, `imag(z)`, `arg(z)`, `conj(z)`, `cplx(re, im)`. */
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

/**
 * cli_calc_where - Select pixels element-wise from @t or @f according to image mask @c
 * @c: Condition image value
 * @t: True-branch value (image or scalar)
 * @f: False-branch value (image or scalar)
 *
 * Return: `VAL_STRING` naming the temporary output image
 */
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

/** func_where() - evaluate `where(cond, T, F)`. */
static val_t func_where(void)
{
    val_t args[3];
    if (parse_func_args(3, args, "Expected ','") != 0 || cli_calc_reject_complex(args, 3))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING)
    {
        return (to_double(args[0]) != 0.0) ? args[1] : args[2];
    }
    return cli_calc_where(args[0], args[1], args[2]);
}

/** func_coord() - evaluate `ix/iy/ir/itheta(im[, cx, cy])` or `rand/randn([im[, seed]])`. */
static val_t func_coord(const cli_token *ftok)
{
    int   kind = cli_calc_coord_kind(ftok);
    val_t args[3];

    if (kind == CALC_COORD_RAND || kind == CALC_COORD_RANDN)
    {
        cli_token *(*cur_func)(void)     = cli_calc_cur_func();
        cli_token *(*advance_func)(void) = cli_calc_advance_func();
        if (cur_func()->type == TOK_RPAREN)
        {
            advance_func();
            return mk_double(cli_calc_rng_draw(kind, cli_calc_next_rng_seed()));
        }
        int n = cli_calc_parse_func_argv(1, 2, args, "Expected ','");
        if (n < 0 || cli_calc_reject_complex(args, n))
        {
            return mk_double(0);
        }
        if (n == 1 && args[0].type != VAL_STRING)
        {
            uint32_t seed = (uint32_t) (int64_t) llround(to_double(args[0]));
            return mk_double(cli_calc_rng_draw(kind, seed));
        }
        if (args[0].type != VAL_STRING || (n == 2 && args[1].type == VAL_STRING))
        {
            parse_errmsg("rand: expected rand([im[, seed]])");
            return mk_double(0);
        }
        if (!check_image(args[0].sval))
        {
            return mk_string("");
        }
        const IMAGE *im =
            &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
        uint32_t seed =
            (n == 2) ? (uint32_t) (int64_t) llround(to_double(args[1])) : cli_calc_next_rng_seed();
        const char *tmpn = cli_calc_coord(im, kind, (double) seed, 1.0);
        return mk_string((tmpn != NULL) ? tmpn : "");
    }

    int n = cli_calc_parse_func_argv(1, 3, args, "Expected ','");
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
    const char *tmpn = cli_calc_coord(im, kind, cx, cy);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/** func_eval_d_d() - evaluate unary `double -> double` or `image -> image` (`TOK_FUNC_D_D`). */
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
        return cli_calc_complex_part(z, 2);
    }
    if (cli_calc_reject_complex(&arg, 1))
    {
        return mk_double(0);
    }

    if (arg.type == VAL_STRING)
    {
        if (!check_image(arg.sval))
        {
            return mk_string("");
        }
        const IMAGE *src =
            &data.core.image[image_ID(arg.sval, data.core.image, data.core.NB_MAX_IMAGE)];
        const char *tmpn = cli_calc_eval_fn_im(1, ftok->fnctptr, src, 0.0, NULL, 0.0, 0.0);
        return mk_string((tmpn != NULL) ? tmpn : "");
    }

    double darg = to_double(arg);
    double res  = ((double (*)(double)) ftok->fnctptr)(darg);
    return mk_double(res);
}

/** func_eval_dd_d() - evaluate binary `(d, d) -> d`, `(im, d) -> im`, or `(im, im) -> im`. */
static val_t func_eval_dd_d(const cli_token *ftok)
{
    val_t args[2];
    if (parse_func_args(2, args, "Expected ','") != 0 || cli_calc_reject_complex(args, 2))
    {
        return mk_double(0);
    }

    if (args[0].type == VAL_STRING || args[1].type == VAL_STRING)
    {
        const IMAGE *ia = NULL;
        const IMAGE *ib = NULL;
        if (args[0].type == VAL_STRING)
        {
            if (!check_image(args[0].sval))
            {
                return mk_string("");
            }
            ia = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
        }
        if (args[1].type == VAL_STRING)
        {
            if (!check_image(args[1].sval))
            {
                return mk_string("");
            }
            ib = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
        }
        double      ac   = (ia == NULL) ? to_double(args[0]) : 0.0;
        double      bc   = (ib == NULL) ? to_double(args[1]) : 0.0;
        const char *tmpn = cli_calc_eval_fn_im(2, ftok->fnctptr, ia, ac, ib, bc, 0.0);
        if (tmpn != NULL)
        {
            return mk_string(tmpn);
        }
        if (ia != NULL && ib != NULL)
        {
            tmpn = alloc_tmpname();
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
        return mk_string("");
    }

    double d1 = to_double(args[0]);
    double d2 = to_double(args[1]);
    return mk_double(((double (*)(double, double)) ftok->fnctptr)(d1, d2));
}

/** func_eval_ddd_d() - evaluate ternary `(d, d, d) -> d` or `(im, d, d) -> im`. */
static val_t func_eval_ddd_d(const cli_token *ftok)
{
    val_t args[3];
    if (parse_func_args(3, args, "Expected ','") != 0 || cli_calc_reject_complex(args, 3))
    {
        return mk_double(0);
    }

    if (args[0].type == VAL_STRING)
    {
        if (!check_image(args[0].sval))
        {
            return mk_string("");
        }
        const IMAGE *src =
            &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
        const char *tmpn = cli_calc_eval_fn_im(3, ftok->fnctptr, src, to_double(args[1]), NULL, 0.0,
                                               to_double(args[2]));
        return mk_string((tmpn != NULL) ? tmpn : "");
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

/** func_shift() - evaluate `shift(im, dx[, dy])` 2D edge-clamped stencil. */
static val_t func_shift(void)
{
    val_t args[3];
    int   n = cli_calc_parse_func_argv(2, 3, args, "Expected ','");
    if (n < 0 || cli_calc_reject_complex(args, n))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING || args[1].type == VAL_STRING ||
        (n == 3 && args[2].type == VAL_STRING))
    {
        parse_errmsg("shift: expected shift(im, dx[, dy])");
        return mk_double(0);
    }
    if (!check_image(args[0].sval))
    {
        return mk_string("");
    }
    const IMAGE *im =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    int64_t     dx   = (int64_t) llround(to_double(args[1]));
    int64_t     dy   = (n == 3) ? (int64_t) llround(to_double(args[2])) : 0;
    const char *tmpn = cli_calc_shift(im, dx, dy);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/** func_lut() - evaluate `lut(x, tbl[, x0, dx])` 1D linear interpolation lookup. */
static val_t func_lut(void)
{
    val_t args[4];
    int   n = cli_calc_parse_func_argv(2, 4, args, "Expected ','");
    if (n < 0 || cli_calc_reject_complex(args, n))
    {
        return mk_double(0);
    }
    if (n == 3 || args[1].type != VAL_STRING ||
        (n == 4 && (args[2].type == VAL_STRING || args[3].type == VAL_STRING)))
    {
        parse_errmsg("lut: expected lut(x, tbl[, x0, dx])");
        return mk_double(0);
    }
    if (!check_image(args[1].sval))
    {
        return mk_string("");
    }
    const IMAGE *tbl =
        &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    double x0 = (n == 4) ? to_double(args[2]) : 0.0;
    double dx = (n == 4) ? to_double(args[3]) : 1.0;
    if (dx == 0.0)
    {
        parse_errmsg("lut: dx must be non-zero");
        return mk_double(0);
    }
    if (args[0].type == VAL_STRING)
    {
        if (!check_image(args[0].sval))
        {
            return mk_string("");
        }
        const IMAGE *im =
            &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
        const char *tmpn = cli_calc_lut(im, tbl, x0, dx);
        return mk_string((tmpn != NULL) ? tmpn : "");
    }
    int64_t N = (int64_t) tbl->md[0].nelement;
    if (N < 1)
    {
        return mk_double(0.0);
    }
    double u = (to_double(args[0]) - x0) / dx;
    if (cli_calc_isnan(u))
    {
        return mk_double(u);
    }
    if (u <= 0.0)
    {
        return mk_double(cli_calc_read_im_pix(tbl, tbl->md[0].datatype, 0));
    }
    if (u >= (double) (N - 1))
    {
        return mk_double(cli_calc_read_im_pix(tbl, tbl->md[0].datatype, (uint64_t) (N - 1)));
    }
    int64_t idx = (int64_t) u;
    if (idx >= 0 && idx < N - 1)
    {
        double frac = u - (double) idx;
        double v0   = cli_calc_read_im_pix(tbl, tbl->md[0].datatype, (uint64_t) idx);
        double v1   = cli_calc_read_im_pix(tbl, tbl->md[0].datatype, (uint64_t) (idx + 1));
        return mk_double(v0 + frac * (v1 - v0));
    }
    return mk_double(NAN);
}

/** func_medfilt() - evaluate `medfilt3/5/7(im)` exact 2D median filter. */
static val_t func_medfilt(const cli_token *ftok)
{
    val_t arg;
    if (parse_func_args(1, &arg, NULL) != 0 || cli_calc_reject_complex(&arg, 1))
    {
        return mk_double(0);
    }
    if (arg.type != VAL_STRING)
    {
        parse_errmsg("medfilt: expected image argument");
        return mk_double(0);
    }
    if (!check_image(arg.sval))
    {
        return mk_string("");
    }
    const IMAGE *im = &data.core.image[image_ID(arg.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    int          K  = (strncmp(ftok->sval, "medfilt3(", 9) == 0)   ? 3
                      : (strncmp(ftok->sval, "medfilt5(", 9) == 0) ? 5
                                                                   : 7;
    const char  *tmpn = cli_calc_medfilt(im, K);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/** func_pixremap() - evaluate `pixremap(im, map[, fill])` indexed gather. */
static val_t func_pixremap(void)
{
    val_t args[3];
    int   n = cli_calc_parse_func_argv(2, 3, args, "Expected ','");
    if (n < 0 || cli_calc_reject_complex(args, n))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING ||
        (n == 3 && args[2].type == VAL_STRING))
    {
        parse_errmsg("pixremap: expected pixremap(im, map[, fill])");
        return mk_double(0);
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_string("");
    }
    const IMAGE *im =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *map =
        &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    double      fill = (n == 3) ? to_double(args[2]) : 0.0;
    const char *tmpn = cli_calc_pixremap(im, map, fill);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/** func_pixunmap() - evaluate `pixunmap(im, map[, n])` or `rprof(im[, cx, cy])` segmented mean. */
static val_t func_pixunmap(const cli_token *ftok)
{
    val_t args[3];
    if (strncmp(ftok->sval, "rprof(", 6) ==
        0) /* the tokenizer shares the token kind with pixunmap */
    {
        int n = cli_calc_parse_func_argv(1, 3, args, "Expected ','");
        if (n < 0 || cli_calc_reject_complex(args, n))
        {
            return mk_double(0);
        }
        if (n == 2 || args[0].type != VAL_STRING ||
            (n == 3 && (args[1].type == VAL_STRING || args[2].type == VAL_STRING)))
        {
            parse_errmsg("rprof: expected rprof(im[, cx, cy])");
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
        const char *tmpn = cli_calc_rprof(im, cx, cy);
        return mk_string((tmpn != NULL) ? tmpn : "");
    }
    int n = cli_calc_parse_func_argv(2, 3, args, "Expected ','");
    if (n < 0 || cli_calc_reject_complex(args, n))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING ||
        (n == 3 && args[2].type == VAL_STRING))
    {
        parse_errmsg("pixunmap: expected pixunmap(im, map[, n])");
        return mk_double(0);
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_string("");
    }
    const IMAGE *im =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *map =
        &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    int64_t     nbins = (n == 3) ? (int64_t) llround(to_double(args[2])) : -1;
    const char *tmpn  = cli_calc_pixunmap(im, map, nbins);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/** func_gauss() - evaluate `gauss(im, sigma[, sigmay])` separable 2D Gaussian blur. */
static val_t func_gauss(void)
{
    val_t args[3];
    int   n = cli_calc_parse_func_argv(2, 3, args, "Expected ','");
    if (n < 0 || cli_calc_reject_complex(args, n))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING || args[1].type == VAL_STRING ||
        (n == 3 && args[2].type == VAL_STRING))
    {
        parse_errmsg("gauss: expected gauss(im, sigma[, sigmay])");
        return mk_double(0);
    }
    if (!check_image(args[0].sval))
    {
        return mk_string("");
    }
    const IMAGE *im =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    double      sx   = to_double(args[1]);
    double      sy   = (n == 3) ? to_double(args[2]) : sx;
    const char *tmpn = cli_calc_gauss(im, sx, sy);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/** func_cumsum() - evaluate `cumsum(im[, axis])` cumulative sum along an axis (0 when omitted). */
static val_t func_cumsum(void)
{
    val_t args[2];
    int   n = cli_calc_parse_func_argv(1, 2, args, "Expected ','");
    if (n < 0 || cli_calc_reject_complex(args, n))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING || (n == 2 && args[1].type == VAL_STRING))
    {
        parse_errmsg("cumsum: expected cumsum(im[, axis])");
        return mk_double(0);
    }
    if (!check_image(args[0].sval))
    {
        return mk_string("");
    }
    const IMAGE *im =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    int         axis = (n == 2) ? (int) llround(to_double(args[1])) : 0;
    const char *tmpn = cli_calc_cumsum(im, axis);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/** func_interp2() - evaluate `interp2(im, xmap, ymap)` 2D sub-pixel bilinear gather. */
static val_t func_interp2(void)
{
    val_t args[3];
    if (parse_func_args(3, args, "Expected ','") != 0 || cli_calc_reject_complex(args, 3))
    {
        return mk_double(0);
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING)
    {
        parse_errmsg("interp2: expected interp2(im, xmap, ymap)");
        return mk_double(0);
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval) || !check_image(args[2].sval))
    {
        return mk_string("");
    }
    const IMAGE *im =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *xm =
        &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    const IMAGE *ym =
        &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    const char *tmpn = cli_calc_interp2(im, xm, ym);
    return mk_string((tmpn != NULL) ? tmpn : "");
}

/**
 * parse_funccall - Dispatch a built-in calculator function call token @ftok
 * @ftok: Function token
 *
 * Return: Evaluated `val_t`
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
    case TOK_FUNC_SHIFT:
        return func_shift();
    case TOK_FUNC_LUT:
        return func_lut();
    case TOK_FUNC_MEDFILT:
        return func_medfilt(ftok);
    case TOK_FUNC_PIXREMAP:
        return func_pixremap();
    case TOK_FUNC_PIXUNMAP:
        return func_pixunmap(ftok);
    case TOK_FUNC_GAUSS:
        return func_gauss();
    case TOK_FUNC_CUMSUM:
        return func_cumsum();
    case TOK_FUNC_INTERP2:
        return func_interp2();
    case TOK_FUNC_UDEF:
        return cli_calc_func_udef(ftok);
    case TOK_FUNC_USER:
        return cli_calc_eval_user_call(ftok);
    case TOK_FUNC_IMIM_IM:
        return ((val_t (*)(void)) ftok->fnctptr)();
    case TOK_FUNC_IM_D:
    case TOK_FUNC_IM_L:
    case TOK_FUNC_IMD_D:
    case TOK_FUNC_IMIM_D:
        return cli_calc_func_eval_image_reduction(ftok);
    case TOK_FUNC_S_D:
    case TOK_FUNC_S_S:
    case TOK_FUNC_SDD_S:
    case TOK_FUNC_D_S:
        return cli_calc_func_eval_string_or_format(ftok);
    case TOK_FUNC_SSS_S:
        return cli_calc_func_replace();
    case TOK_FUNC_CPLX:
        return func_eval_cplx(ftok);
    default:
        parse_errmsg("Unknown function type");
        return mk_double(0);
    }
}
