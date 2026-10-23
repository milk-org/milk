// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_complex.c
 * @brief Complex images in calculator expressions (delegating kernels to `COREMOD_arith`).
 *
 * Role & Goal:
 *   Calculator values are real, so complex images (CF/CD) enter and leave expressions through
 *   `real()`, `imag()`, `abs()`, `arg()`, `conj()`, `cplx(re, im)` and `expi(phi)`. Between them,
 *   `+ - * /` and unary `-` work element-wise with real images and numbers (taken as `x + 0i`),
 *   and in-place assignment writes complex pixels via `COREMOD_arith` complex kernels.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_arith/image_arith__Cim_Cim__Cim.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"

#define CPLX_HINT "use real(), imag() or abs() first"

/**
 * cli_calc_is_complex_dt - Check whether an ImageStreamIO datatype is complex
 * @dt: ImageStreamIO `_DATATYPE_*` code
 *
 * Return: 1 if @dt is `_DATATYPE_COMPLEX_FLOAT` or `_DATATYPE_COMPLEX_DOUBLE`, 0 otherwise
 */
int cli_calc_is_complex_dt(uint8_t dt)
{
    return arith_is_complex_dt(dt);
}

/**
 * cli_calc_complex_image - Resolve a calculator value to a complex IMAGE pointer
 * @v: Calculator value to inspect
 *
 * Return: Pointer to the complex `IMAGE`, or NULL if @v is not a complex image
 */
const IMAGE *cli_calc_complex_image(const val_t *v)
{
    if (v->type != VAL_STRING)
    {
        return NULL;
    }
    imageID id = image_ID(v->sval, data.core.image, data.core.NB_MAX_IMAGE);
    return (id != -1 && arith_is_complex_dt(data.core.image[id].md[0].datatype))
               ? &data.core.image[id]
               : NULL;
}

/** Return 1 if @dt is `_DATATYPE_DOUBLE` or `_DATATYPE_COMPLEX_DOUBLE`, 0 otherwise. */
static int is_double_dt(uint8_t dt)
{
    return dt == _DATATYPE_DOUBLE || dt == _DATATYPE_COMPLEX_DOUBLE;
}

/** Allocate a temporary image shaped like @like with datatype @dt; return NULL on failure. */
static IMAGE *new_image(const IMAGE *like, uint8_t dt, const char **name)
{
    uint32_t sz[3] = { like->md[0].size[0], like->md[0].size[1], like->md[0].size[2] };
    imageID  tid   = -1;
    *name          = alloc_tmpname();
    create_image_ID(*name, like->md[0].naxis, sz, dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate output image");
        return NULL;
    }
    return &data.core.image[tid];
}

/** Resolve `v->sval` in `data.core.image` when @v has type `VAL_STRING`, else NULL. */
static const IMAGE *val_image(const val_t *v)
{
    if (v->type != VAL_STRING)
    {
        return NULL;
    }
    imageID id = image_ID(v->sval, data.core.image, data.core.NB_MAX_IMAGE);
    return (id == -1) ? NULL : &data.core.image[id];
}

/**
 * cli_calc_complex_binop - Evaluate a binary arithmetic operator involving a complex image
 * @op:    Operator token (`+`, `-`, `*`, or `/`)
 * @left:  Left operand (complex image, real image, or scalar)
 * @right: Right operand (complex image, real image, or scalar)
 *
 * Return: `VAL_STRING` naming the temporary output image, or empty string on error
 */
val_t cli_calc_complex_binop(cli_token_type op, val_t left, val_t right)
{
    if (op != TOK_OP_PLUS && op != TOK_OP_MINUS && op != TOK_OP_STAR && op != TOK_OP_SLASH)
    {
        parse_errmsg("Operator not supported for complex images; " CPLX_HINT);
        return mk_string("");
    }
    const IMAGE *li = val_image(&left);
    const IMAGE *ri = val_image(&right);
    if ((left.type == VAL_STRING && li == NULL) || (right.type == VAL_STRING && ri == NULL))
    {
        check_image(li == NULL ? left.sval : right.sval);
        return mk_string("");
    }
    if (li != NULL && ri != NULL && li->md[0].nelement != ri->md[0].nelement)
    {
        parse_errmsg("Complex image operands must have the same size");
        return mk_string("");
    }
    if (ri == NULL && op == TOK_OP_SLASH && to_double(right) == 0.0)
    {
        parse_errmsg("Division by zero");
        return mk_string("");
    }
    const IMAGE *like   = (li != NULL && arith_is_complex_dt(li->md[0].datatype)) ? li : ri;
    int          dbl    = (li != NULL && is_double_dt(li->md[0].datatype)) ||
                          (ri != NULL && is_double_dt(ri->md[0].datatype));
    uint8_t      out_dt = dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT;
    const char  *tmpn   = NULL;
    IMAGE       *out    = NULL;
    if (li != NULL && li->md[0].datatype == out_dt && strncmp(left.sval, "_tmpcalc", 8) == 0)
    {
        out  = (IMAGE *) li;
        tmpn = left.sval;
    }
    else if (ri != NULL && ri->md[0].datatype == out_dt && strncmp(right.sval, "_tmpcalc", 8) == 0)
    {
        out  = (IMAGE *) ri;
        tmpn = right.sval;
    }
    else
    {
        out = new_image(like, out_dt, &tmpn);
    }
    if (out == NULL)
    {
        return mk_string("");
    }
    int cop_code = (op == TOK_OP_PLUS)    ? ARITH_CPLX_ADD
                   : (op == TOK_OP_MINUS) ? ARITH_CPLX_SUB
                   : (op == TOK_OP_STAR)  ? ARITH_CPLX_MUL
                                          : ARITH_CPLX_DIV;
    arith_complex_binop_im(cop_code, li, (li == NULL) ? to_double(left) : 0.0, ri,
                           (ri == NULL) ? to_double(right) : 0.0, out);
    return mk_string(tmpn);
}

/**
 * cli_calc_complex_neg - Negate every pixel of a complex image
 * @z: Source complex image
 *
 * Return: `VAL_STRING` naming the temporary negated image, or empty string on error
 */
val_t cli_calc_complex_neg(const IMAGE *z)
{
    const char *tmpn = z->name;
    IMAGE      *out =
        (strncmp(tmpn, "_tmpcalc", 8) == 0) ? (IMAGE *) z : new_image(z, z->md[0].datatype, &tmpn);
    if (out == NULL)
    {
        return mk_string("");
    }
    arith_complex_neg_im(z, out);
    return mk_string(tmpn);
}

/**
 * cli_calc_inplace_complex - Perform in-place assignment or compound update on a complex image
 * @dst: Destination image (must have a complex datatype)
 * @aop: Assignment operator (`=`, `+=`, `-=`, `*=`, or `/=`)
 * @src: Optional source image operand, or NULL when @val is used
 * @val: Scalar real operand used when @src is NULL
 *
 * Return: 0 on success, -1 on error (with parse error message set)
 */
int cli_calc_inplace_complex(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val)
{
    if (!arith_is_complex_dt(dst->md[0].datatype))
    {
        parse_errmsg((src != NULL && arith_is_complex_dt(src->md[0].datatype))
                         ? "Cannot store a complex image in a real one; " CPLX_HINT
                         : "Unsupported image datatype");
        return -1;
    }
    if (src != NULL && !arith_is_complex_dt(src->md[0].datatype) &&
        ImageStreamIO_typesize(src->md[0].datatype) <= 0)
    {
        parse_errmsg("Unsupported image datatype");
        return -1;
    }
    int cop_code = (aop == TOK_EQUAL)         ? ARITH_CPLX_ASSIGN
                   : (aop == TOK_OP_PLUS_EQ)  ? ARITH_CPLX_ADD
                   : (aop == TOK_OP_MINUS_EQ) ? ARITH_CPLX_SUB
                   : (aop == TOK_OP_STAR_EQ)  ? ARITH_CPLX_MUL
                                              : ARITH_CPLX_DIV;
    if (arith_complex_inplace_im(cop_code, dst, src, val) != RETURN_SUCCESS)
    {
        parse_errmsg("Division by zero");
        return -1;
    }
    return 0;
}

/**
 * cli_calc_complex_part - Extract `real()`, `imag()`, `abs()`, or `arg()` of complex image @z
 * @z:     Source complex image
 * @which: `ARITH_CPLX_REAL` (0), `ARITH_CPLX_IMAG` (1), `ARITH_CPLX_ABS` (2), or `ARITH_CPLX_ARG` (3)
 *
 * Return: `VAL_DOUBLE` if @z has 1 pixel, else `VAL_STRING` naming the temporary real image
 */
val_t cli_calc_complex_part(const IMAGE *z, int which)
{
    if (z->md[0].nelement == 1)
    {
        double re, im;
        arith_complex_read_pix(z, 0, &re, &im);
        double v = (which == ARITH_CPLX_REAL)   ? re
                   : (which == ARITH_CPLX_IMAG) ? im
                   : (which == ARITH_CPLX_ABS)  ? hypot(re, im)
                                                : atan2(im, re);
        return mk_double(v);
    }
    const char *tmpn;
    int         dbl = is_double_dt(z->md[0].datatype);
    IMAGE      *out = new_image(z, dbl ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT, &tmpn);
    if (out == NULL)
    {
        return mk_string("");
    }
    arith_complex_part_im(z, which, out);
    return mk_string(tmpn);
}

/**
 * cli_calc_complex_func - Evaluate `real`, `imag`, `arg`, `conj`, `cplx`, or `expi` on @args
 * @fname: Function token string (`"real("`, `"imag("`, `"arg("`, `"conj("`, `"cplx("`, `"expi("`)
 * @args:  Evaluated argument array
 * @nargs: Number of arguments
 *
 * Return: Evaluated `val_t` result
 */
val_t cli_calc_complex_func(const char *fname, const val_t *args, int nargs)
{
    int which = (strncmp(fname, "real(", 5) == 0)   ? ARITH_CPLX_REAL
                : (strncmp(fname, "imag(", 5) == 0) ? ARITH_CPLX_IMAG
                                                    : ARITH_CPLX_ARG;
    if (strncmp(fname, "cplx(", 5) == 0)
    {
        if (nargs != 2)
        {
            parse_errmsg("cplx: need 2 arguments (real part, imaginary part)");
            return mk_double(0);
        }
        const IMAGE *a = val_image(&args[0]);
        const IMAGE *b = val_image(&args[1]);
        if ((args[0].type == VAL_STRING && a == NULL) ||
            (args[1].type == VAL_STRING && b == NULL) || (a == NULL && b == NULL))
        {
            parse_errmsg("cplx: needs at least one image (there are no complex numbers)");
            return mk_double(0);
        }
        if ((a != NULL && arith_is_complex_dt(a->md[0].datatype)) ||
            (b != NULL && arith_is_complex_dt(b->md[0].datatype)) ||
            (a != NULL && b != NULL && a->md[0].nelement != b->md[0].nelement))
        {
            parse_errmsg("cplx: needs two real operands of the same size");
            return mk_double(0);
        }
        int          dbl  = (a != NULL && is_double_dt(a->md[0].datatype)) ||
                            (b != NULL && is_double_dt(b->md[0].datatype));
        const IMAGE *like = (a != NULL) ? a : b;
        const char  *tmpn;
        IMAGE       *out =
            new_image(like, dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT, &tmpn);
        if (out == NULL)
        {
            return mk_double(0);
        }
        arith_complex_cplx_im(a, (a == NULL) ? to_double(args[0]) : 0.0, b,
                              (b == NULL) ? to_double(args[1]) : 0.0, out);
        return mk_string(tmpn);
    }

    if (nargs != 1)
    {
        parse_errmsg("Expected one argument");
        return mk_double(0);
    }
    const IMAGE *z = val_image(&args[0]);
    if (args[0].type == VAL_STRING && z == NULL)
    {
        check_image(args[0].sval);
        return mk_double(0);
    }
    if (strncmp(fname, "expi(", 5) == 0)
    {
        if (z == NULL || arith_is_complex_dt(z->md[0].datatype))
        {
            parse_errmsg("expi: expected a real phase image");
            return mk_double(0);
        }
        int         dbl = is_double_dt(z->md[0].datatype);
        const char *tmpn;
        IMAGE *out = new_image(z, dbl ? _DATATYPE_COMPLEX_DOUBLE : _DATATYPE_COMPLEX_FLOAT, &tmpn);
        if (out == NULL)
        {
            return mk_double(0);
        }
        arith_complex_expi_im(z, out);
        return mk_string(tmpn);
    }
    if (z == NULL || !arith_is_complex_dt(z->md[0].datatype))
    {
        if (strncmp(fname, "conj(", 5) == 0 || which == ARITH_CPLX_REAL)
        {
            return args[0];
        }
        if (z == NULL)
        {
            double x = to_double(args[0]);
            return mk_double((which == ARITH_CPLX_IMAG) ? 0.0 : atan2(0.0, x));
        }
    }
    if (strncmp(fname, "conj(", 5) == 0)
    {
        const char *tmpn = z->name;
        IMAGE *out = (strncmp(tmpn, "_tmpcalc", 8) == 0) ? (IMAGE *) z
                                                         : new_image(z, z->md[0].datatype, &tmpn);
        if (out == NULL)
        {
            return mk_double(0);
        }
        arith_complex_conj_im(z, out);
        return mk_string(tmpn);
    }
    return cli_calc_complex_part(z, which);
}
