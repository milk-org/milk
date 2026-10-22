// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_complex.c
 * @brief Complex images in calculator expressions.
 *
 * Role & Goal:
 *   Calculator values are real, so complex images (CF/CD) enter and leave expressions through
 *   `real()`, `imag()`, `abs()`, `arg()`, `conj()` and `cplx(re, im)`. Between them, `+ - * /` and
 *   unary `-` work element-wise with real images and numbers (taken as `x + 0i`), and in-place
 *   assignment writes complex pixels. Results are COMPLEX_DOUBLE if any operand is double, else
 *   COMPLEX_FLOAT. Other operators and real-only functions on a complex image are errors.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "libmilkdata/milk_type_dispatch.h"

#define CPLX_HINT "use real(), imag() or abs() first"

/**
 * cli_calc_is_complex_dt - Check whether an ImageStreamIO datatype is complex
 * @dt: ImageStreamIO `_DATATYPE_*` code
 *
 * Identifies single- and double-precision complex image datatypes so the
 * calculator can dispatch operations to complex-aware routines.
 *
 * Return: 1 if @dt is `_DATATYPE_COMPLEX_FLOAT` or `_DATATYPE_COMPLEX_DOUBLE`, 0 otherwise
 */
int cli_calc_is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

/**
 * cli_calc_complex_image - Resolve a calculator value to a complex IMAGE pointer
 * @v: Calculator value to inspect
 *
 * Checks whether @v names an existing image in `data.core.image` whose
 * datatype is complex (`CF` or `CD`), returning NULL for scalar values,
 * unknown names, or real images.
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
    return (id != -1 && cli_calc_is_complex_dt(data.core.image[id].md[0].datatype))
               ? &data.core.image[id]
               : NULL;
}

/** pix() - pixel @i of any image as (re, im); real pixels have im = 0. */
static inline void pix(const IMAGE *im, uint64_t i, double *re, double *imv)
{
    uint8_t dt = im->md[0].datatype;
    *imv       = 0.0;
    if (dt == _DATATYPE_COMPLEX_FLOAT)
    {
        *re  = im->array.CF[i].re;
        *imv = im->array.CF[i].im;
        return;
    }
    if (dt == _DATATYPE_COMPLEX_DOUBLE)
    {
        *re  = im->array.CD[i].re;
        *imv = im->array.CD[i].im;
        return;
    }
#define PIX_BODY(MBR)                    \
    {                                    \
        *re = (double) im->array.MBR[i]; \
        return;                          \
    }
    MILK_FOR_EACH_DATATYPE(dt, PIX_BODY, PIX_BODY(D))
#undef PIX_BODY
    *re = 0.0;
}

/**
 * set_pix - Store a complex value into pixel @i of a complex image
 * @im:  Destination complex image (`COMPLEX_FLOAT` or `COMPLEX_DOUBLE`)
 * @i:   Linear pixel index
 * @re:  Real component to store
 * @imv: Imaginary component to store
 *
 * Casts @re and @imv to `float` when @im is `COMPLEX_FLOAT` or writes `double`
 * directly for `COMPLEX_DOUBLE`.
 *
 * Return: void
 */
static inline void set_pix(IMAGE *im, uint64_t i, double re, double imv)
{
    if (im->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        im->array.CF[i].re = (float) re;
        im->array.CF[i].im = (float) imv;
    }
    else
    {
        im->array.CD[i].re = re;
        im->array.CD[i].im = imv;
    }
}

/** cop() - (ar + i ai) op (br + i bi) for + - * /. */
static inline void
cop(cli_token_type op, double ar, double ai, double br, double bi, double *r, double *i)
{
    switch (op)
    {
    case TOK_OP_PLUS:
        *r = ar + br;
        *i = ai + bi;
        break;
    case TOK_OP_MINUS:
        *r = ar - br;
        *i = ai - bi;
        break;
    case TOK_OP_STAR:
        *r = ar * br - ai * bi;
        *i = ar * bi + ai * br;
        break;
    default:
    {
        double d = br * br + bi * bi;
        *r       = (ar * br + ai * bi) / d;
        *i       = (ai * br - ar * bi) / d;
        break;
    }
    }
}

/**
 * is_double_dt - Check whether an image datatype uses 64-bit floating-point precision
 * @dt: ImageStreamIO `_DATATYPE_*` code
 *
 * Determines whether a complex operation should promote its output to
 * `_DATATYPE_COMPLEX_DOUBLE` or `_DATATYPE_DOUBLE`.
 *
 * Return: 1 if @dt is `_DATATYPE_DOUBLE` or `_DATATYPE_COMPLEX_DOUBLE`, 0 otherwise
 */
static int is_double_dt(uint8_t dt)
{
    return dt == _DATATYPE_DOUBLE || dt == _DATATYPE_COMPLEX_DOUBLE;
}

/** new_image() - temporary image shaped like @like, of type @dt; NULL on failure. */
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

/**
 * val_image - Look up the IMAGE referenced by a string calculator value
 * @v: Calculator value to resolve
 *
 * Resolves `v->sval` in `data.core.image` when @v has type `VAL_STRING`,
 * allowing binary and function evaluators to distinguish image operands from
 * numeric scalars.
 *
 * Return: Pointer to the resolved `IMAGE`, or NULL if @v is not an existing image
 */
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
 * Allocates a temporary complex image (`COMPLEX_DOUBLE` if either image uses
 * double precision, otherwise `COMPLEX_FLOAT`) and evaluates `+`, `-`, `*`, or
 * `/` element-wise, treating real operands as having zero imaginary part.
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
    const IMAGE *like   = (li != NULL && cli_calc_is_complex_dt(li->md[0].datatype)) ? li : ri;
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
    double   lc  = (li == NULL) ? to_double(left) : 0.0;
    double   rc  = (ri == NULL) ? to_double(right) : 0.0;
    uint64_t nel = out->md[0].nelement;

    if (!dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && ri != NULL &&
        ri->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const complex_float *a   = li->array.CF;
        const complex_float *b   = ri->array.CF;
        complex_float       *dst = out->array.CF;
        if (op == TOK_OP_PLUS)
        {
            const float *af = (const float *) a;
            const float *bf = (const float *) b;
            float       *df = (float *) dst;
            uint64_t     n2 = nel * 2;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t k = 0; k < n2; k++)
            {
                df[k] = af[k] + bf[k];
            }
            return mk_string(tmpn);
        }
        if (op == TOK_OP_MINUS)
        {
            const float *af = (const float *) a;
            const float *bf = (const float *) b;
            float       *df = (float *) dst;
            uint64_t     n2 = nel * 2;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t k = 0; k < n2; k++)
            {
                df[k] = af[k] - bf[k];
            }
            return mk_string(tmpn);
        }
        if (op == TOK_OP_STAR)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                float ar  = a[i].re;
                float ai  = a[i].im;
                float br  = b[i].re;
                float bi  = b[i].im;
                dst[i].re = ar * br - ai * bi;
                dst[i].im = ar * bi + ai * br;
            }
            return mk_string(tmpn);
        }
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t i = 0; i < nel; i++)
        {
            float ar  = a[i].re;
            float ai  = a[i].im;
            float br  = b[i].re;
            float bi  = b[i].im;
            float d   = br * br + bi * bi;
            dst[i].re = (ar * br + ai * bi) / d;
            dst[i].im = (ai * br - ar * bi) / d;
        }
        return mk_string(tmpn);
    }

    if (!dbl && li != NULL && li->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && ri == NULL)
    {
        const float *af = (const float *) li->array.CF;
        float       *df = (float *) out->array.CF;
        uint64_t     n2 = nel * 2;
        if (op == TOK_OP_STAR || op == TOK_OP_SLASH)
        {
            float s = (op == TOK_OP_STAR) ? (float) rc : (float) (1.0 / rc);
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t k = 0; k < n2; k++)
            {
                df[k] = af[k] * s;
            }
            return mk_string(tmpn);
        }
    }

#pragma omp parallel for schedule(static) if (nel >= 131072)
    for (uint64_t i = 0; i < nel; i++)
    {
        double ar = lc, ai = 0.0, br = rc, bi = 0.0, r, im;
        if (li != NULL)
        {
            pix(li, i, &ar, &ai);
        }
        if (ri != NULL)
        {
            pix(ri, i, &br, &bi);
        }
        cop(op, ar, ai, br, bi, &r, &im);
        set_pix(out, i, r, im);
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_complex_neg - Negate every pixel of a complex image
 * @z: Source complex image
 *
 * Creates a temporary complex image of the same shape and datatype as @z with
 * both real and imaginary parts negated.
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
    uint64_t nel = z->md[0].nelement;
    uint64_t n2  = nel * 2;
    if (z->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const float *src = (const float *) z->array.CF;
        float       *dst = (float *) out->array.CF;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t k = 0; k < n2; k++)
        {
            dst[k] = 0.0f - src[k];
        }
        return mk_string(tmpn);
    }
    const double *src = (const double *) z->array.CD;
    double       *dst = (double *) out->array.CD;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
    for (uint64_t k = 0; k < n2; k++)
    {
        dst[k] = 0.0 - src[k];
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_inplace_complex - Perform in-place assignment or compound update on a complex image
 * @dst: Destination image (must have a complex datatype)
 * @aop: Assignment operator (`=`, `+=`, `-=`, `*=`, or `/=`)
 * @src: Optional source image operand, or NULL when @val is used
 * @val: Scalar real operand used when @src is NULL
 *
 * Applies `dst aop src` or `dst aop val` element-wise on complex pixels,
 * rejecting attempts to store complex data into a real destination image or
 * divide by a zero pixel.
 *
 * Return: 0 on success, -1 on error (with parse error message set)
 */
int cli_calc_inplace_complex(IMAGE *dst, cli_token_type aop, const IMAGE *src, double val)
{
    if (!cli_calc_is_complex_dt(dst->md[0].datatype))
    {
        parse_errmsg((src != NULL && cli_calc_is_complex_dt(src->md[0].datatype))
                         ? "Cannot store a complex image in a real one; " CPLX_HINT
                         : "Unsupported image datatype");
        return -1;
    }
    if (src != NULL && !cli_calc_is_complex_dt(src->md[0].datatype) &&
        ImageStreamIO_typesize(src->md[0].datatype) <= 0)
    {
        parse_errmsg("Unsupported image datatype");
        return -1;
    }
    uint64_t       nel = dst->md[0].nelement;
    cli_token_type op  = (aop == TOK_OP_PLUS_EQ)    ? TOK_OP_PLUS
                         : (aop == TOK_OP_MINUS_EQ) ? TOK_OP_MINUS
                         : (aop == TOK_OP_STAR_EQ)  ? TOK_OP_STAR
                                                    : TOK_OP_SLASH;
    if (src != NULL && aop == TOK_OP_SLASH_EQ)
    {
        for (uint64_t i = 0; i < nel; i++)
        {
            double re, im;
            pix(src, i, &re, &im);
            if (re == 0.0 && im == 0.0)
            {
                parse_errmsg("Division by zero");
                return -1;
            }
        }
    }
    if (dst->md[0].datatype == _DATATYPE_COMPLEX_FLOAT && src != NULL &&
        src->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        complex_float       *d  = dst->array.CF;
        const complex_float *s  = src->array.CF;
        uint64_t             n2 = nel * 2;
        if (aop == TOK_EQUAL)
        {
            if (d != s)
            {
                __builtin_memcpy(d, s, nel * sizeof(complex_float));
            }
            return 0;
        }
        if (aop == TOK_OP_PLUS_EQ)
        {
            float       *df = (float *) d;
            const float *sf = (const float *) s;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t k = 0; k < n2; k++)
            {
                df[k] += sf[k];
            }
            return 0;
        }
        if (aop == TOK_OP_MINUS_EQ)
        {
            float       *df = (float *) d;
            const float *sf = (const float *) s;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t k = 0; k < n2; k++)
            {
                df[k] -= sf[k];
            }
            return 0;
        }
        if (aop == TOK_OP_STAR_EQ)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                float ar = d[i].re;
                float ai = d[i].im;
                float br = s[i].re;
                float bi = s[i].im;
                d[i].re  = ar * br - ai * bi;
                d[i].im  = ar * bi + ai * br;
            }
            return 0;
        }
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t i = 0; i < nel; i++)
        {
            float ar = d[i].re;
            float ai = d[i].im;
            float br = s[i].re;
            float bi = s[i].im;
            float dn = br * br + bi * bi;
            d[i].re  = (ar * br + ai * bi) / dn;
            d[i].im  = (ai * br - ar * bi) / dn;
        }
        return 0;
    }
    if (dst->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE && src != NULL &&
        src->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE && aop == TOK_EQUAL)
    {
        if (dst->array.CD != src->array.CD)
        {
            __builtin_memcpy(dst->array.CD, src->array.CD, nel * sizeof(complex_double));
        }
        return 0;
    }
#pragma omp                   parallel for schedule(static) if (nel >= 131072)
    for (uint64_t i = 0; i < nel; i++)
    {
        double br = val, bi = 0.0, ar, ai, r, im;
        if (src != NULL)
        {
            pix(src, i, &br, &bi);
        }
        if (aop == TOK_EQUAL)
        {
            set_pix(dst, i, br, bi);
            continue;
        }
        pix(dst, i, &ar, &ai);
        cop(op, ar, ai, br, bi, &r, &im);
        set_pix(dst, i, r, im);
    }
    return 0;
}

/** cli_calc_complex_part() - real(), imag(), abs(), arg() of @z: a real image, or a number if @z has one pixel. */
val_t cli_calc_complex_part(const IMAGE *z, int which)
{
    uint64_t nel = z->md[0].nelement;
    if (nel == 1)
    {
        double re, im;
        pix(z, 0, &re, &im);
        double v = (which == 0)   ? re
                   : (which == 1) ? im
                   : (which == 2) ? hypot(re, im)
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
    if (!dbl && z->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        const complex_float *restrict src = z->array.CF;
        float *restrict dst               = out->array.F;
        if (which == 0)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                dst[i] = src[i].re;
            }
        }
        else if (which == 1)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                dst[i] = src[i].im;
            }
        }
        else if (which == 2)
        {
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                float re = src[i].re;
                float im = src[i].im;
                dst[i]   = sqrtf(re * re + im * im);
            }
        }
        else
        {
            uint64_t nblk = (nel + 255) / 256;
#pragma omp                   parallel for schedule(static) if (nel >= 131072)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t               off = b * 256;
                int                    len = (int) ((nel - off < 256) ? (nel - off) : 256);
                MILK_ALIGNED(64) float rb[256];
                MILK_ALIGNED(64) float ib[256];
                for (int j = 0; j < len; j++)
                {
                    rb[j] = src[off + (uint64_t) j].re;
                    ib[j] = src[off + (uint64_t) j].im;
                }
                for (int j = 0; j < len; j++)
                {
                    dst[off + (uint64_t) j] = atan2f(ib[j], rb[j]);
                }
            }
        }
        return mk_string(tmpn);
    }

#pragma omp parallel for schedule(static) if (nel >= 131072)
    for (uint64_t i = 0; i < nel; i++)
    {
        double re, im;
        pix(z, i, &re, &im);
        double v = (which == 0)   ? re
                   : (which == 1) ? im
                   : (which == 2) ? sqrt(re * re + im * im)
                                  : atan2(im, re);
        if (dbl)
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

/** cli_calc_complex_func() - real/imag/arg/conj/cplx(...) once its arguments are evaluated. */
val_t cli_calc_complex_func(const char *fname, const val_t *args, int nargs)
{
    int which = (strncmp(fname, "real(", 5) == 0) ? 0 : (strncmp(fname, "imag(", 5) == 0) ? 1 : 3;
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
        if ((a != NULL && cli_calc_is_complex_dt(a->md[0].datatype)) ||
            (b != NULL && cli_calc_is_complex_dt(b->md[0].datatype)) ||
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
        double   ac  = (a == NULL) ? to_double(args[0]) : 0.0;
        double   bc  = (b == NULL) ? to_double(args[1]) : 0.0;
        uint64_t nel = out->md[0].nelement;
        if (!dbl && (a == NULL || a->md[0].datatype == _DATATYPE_FLOAT) &&
            (b == NULL || b->md[0].datatype == _DATATYPE_FLOAT))
        {
            const float *restrict af  = (a != NULL) ? a->array.F : NULL;
            const float *restrict bf  = (b != NULL) ? b->array.F : NULL;
            complex_float *restrict d = out->array.CF;
            const float fac           = (float) ac;
            const float fbc           = (float) bc;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                d[i].re = (af != NULL) ? af[i] : fac;
                d[i].im = (bf != NULL) ? bf[i] : fbc;
            }
            return mk_string(tmpn);
        }
#pragma omp                   parallel for schedule(static) if (nel >= 131072)
        for (uint64_t i = 0; i < nel; i++)
        {
            double re = ac, im = bc, unused;
            if (a != NULL)
            {
                pix(a, i, &re, &unused);
            }
            if (b != NULL)
            {
                pix(b, i, &im, &unused);
            }
            set_pix(out, i, re, im);
        }
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
        if (z == NULL || cli_calc_is_complex_dt(z->md[0].datatype))
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
        uint64_t nel = out->md[0].nelement;
        if (z->md[0].datatype == _DATATYPE_FLOAT)
        {
            const float *restrict src   = z->array.F;
            complex_float *restrict dst = out->array.CF;
            uint64_t nblk               = (nel + 255) / 256;
#pragma omp                   parallel for schedule(static) if (nel >= 131072)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t off              = b * 256;
                int      len              = (int) ((nel - off < 256) ? (nel - off) : 256);
                const float *restrict s   = src + off;
                complex_float *restrict d = dst + off;
                MILK_ALIGNED(64) float cr[256];
                MILK_ALIGNED(64) float ci[256];
                for (int j = 0; j < len; j++)
                {
                    cr[j] = cosf(s[j]);
                }
                for (int j = 0; j < len; j++)
                {
                    ci[j] = sinf(s[j]);
                }
                for (int j = 0; j < len; j++)
                {
                    d[j].re = cr[j];
                    d[j].im = ci[j];
                }
            }
        }
        else if (z->md[0].datatype == _DATATYPE_DOUBLE)
        {
            const double *restrict src   = z->array.D;
            complex_double *restrict dst = out->array.CD;
            uint64_t nblk                = (nel + 255) / 256;
#pragma omp                   parallel for schedule(static) if (nel >= 131072)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t off               = b * 256;
                int      len               = (int) ((nel - off < 256) ? (nel - off) : 256);
                const double *restrict s   = src + off;
                complex_double *restrict d = dst + off;
                MILK_ALIGNED(64) double cr[256];
                MILK_ALIGNED(64) double ci[256];
                for (int j = 0; j < len; j++)
                {
                    cr[j] = cos(s[j]);
                }
                for (int j = 0; j < len; j++)
                {
                    ci[j] = sin(s[j]);
                }
                for (int j = 0; j < len; j++)
                {
                    d[j].re = cr[j];
                    d[j].im = ci[j];
                }
            }
        }
        else
        {
#pragma omp                   parallel for schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                double phi = 0.0, unused = 0.0;
                pix(z, i, &phi, &unused);
                set_pix(out, i, cos(phi), sin(phi));
            }
        }
        return mk_string(tmpn);
    }
    if (z == NULL || !cli_calc_is_complex_dt(z->md[0].datatype)) /* a real value: x + 0i */
    {
        if (strncmp(fname, "conj(", 5) == 0 || which == 0)
        {
            return args[0];
        }
        if (z == NULL)
        {
            double x = to_double(args[0]);
            return mk_double((which == 1) ? 0.0 : atan2(0.0, x));
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
        uint64_t nel = z->md[0].nelement;
        if (z->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            const complex_float *src = z->array.CF;
            complex_float       *dst = out->array.CF;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
            for (uint64_t i = 0; i < nel; i++)
            {
                dst[i].re = src[i].re;
                dst[i].im = 0.0f - src[i].im;
            }
            return mk_string(tmpn);
        }
        const complex_double *src = z->array.CD;
        complex_double       *dst = out->array.CD;
#pragma omp parallel for simd schedule(static) if (nel >= 131072)
        for (uint64_t i = 0; i < nel; i++)
        {
            dst[i].re = src[i].re;
            dst[i].im = 0.0 - src[i].im;
        }
        return mk_string(tmpn);
    }
    return cli_calc_complex_part(z, which);
}
