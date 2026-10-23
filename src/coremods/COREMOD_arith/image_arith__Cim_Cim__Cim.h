// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__Cim_Cim__Cim.h
 * @brief   Complex image arithmetic and component extraction routines
 */

#ifndef COREMOD_ARITH_IMAGE_ARITH_CIM_CIM_CIM_H
#define COREMOD_ARITH_IMAGE_ARITH_CIM_CIM_CIM_H

#include <stdint.h>
#include "ImageStreamIO/ImageStruct.h"
#include "libmilkcommon/milk_compiler.h"

/** Complex binary and in-place operation selectors for `arith_complex_*_im`. */
enum
{
    ARITH_CPLX_ADD,
    ARITH_CPLX_SUB,
    ARITH_CPLX_MUL,
    ARITH_CPLX_DIV,
    ARITH_CPLX_ASSIGN
};

/** Complex component selectors for `arith_complex_part_im`. */
enum
{
    ARITH_CPLX_REAL = 0,
    ARITH_CPLX_IMAG = 1,
    ARITH_CPLX_ABS  = 2,
    ARITH_CPLX_ARG  = 3
};

/** Return 1 if @dt is `_DATATYPE_COMPLEX_FLOAT` or `_DATATYPE_COMPLEX_DOUBLE`, 0 otherwise. */
static inline int arith_is_complex_dt(uint8_t dt)
{
    return dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE;
}

/** Read pixel @i of any real or complex image @im as `(re, imv)`. */
void arith_complex_read_pix(const IMAGE *im, uint64_t i, double *re, double *imv);

/** Evaluate `li/lc op ri/rc` into complex image @out (`ARITH_CPLX_ADD`..`ARITH_CPLX_DIV`). */
errno_t arith_complex_binop_im(int          op,
                               const IMAGE *li,
                               double       lc,
                               const IMAGE *ri,
                               double       rc,
                               IMAGE       *out);

/** Evaluate `dst op= src/val` in place on complex image @dst (`ARITH_CPLX_ADD`..`ARITH_CPLX_ASSIGN`). */
errno_t arith_complex_inplace_im(int op, IMAGE *dst, const IMAGE *src, double val);

/** Negate every pixel of complex image @src into @dst (may alias @src). */
errno_t arith_complex_neg_im(const IMAGE *src, IMAGE *dst);

/** Conjugate every pixel of complex image @src into @dst (may alias @src). */
errno_t arith_complex_conj_im(const IMAGE *src, IMAGE *dst);

/** Extract `real` (0), `imag` (1), `abs` (2), or `arg` (3) of @src into `FLOAT`/`DOUBLE` image @dst. */
errno_t arith_complex_part_im(const IMAGE *src, int which, IMAGE *dst);

/** Pack real part (`@a` or `@ac`) and imaginary part (`@b` or `@bc`) into complex image @out. */
errno_t arith_complex_cplx_im(const IMAGE *a, double ac, const IMAGE *b, double bc, IMAGE *out);

/** Evaluate `cos(phi) + i*sin(phi)` of real image @phi into complex image @out. */
errno_t arith_complex_expi_im(const IMAGE *phi, IMAGE *out);

/** Add two complex images by name. */
int arith_image_Cadd(const char *ID1_name, const char *ID2_name, const char *ID_out);

/** Subtract two complex images by name. */
int arith_image_Csub(const char *ID1_name, const char *ID2_name, const char *ID_out);

/** Multiply two complex images by name. */
int arith_image_Cmult(const char *ID1_name, const char *ID2_name, const char *ID_out);

/** Divide two complex images by name. */
int arith_image_Cdiv(const char *ID1_name, const char *ID2_name, const char *ID_out);

#endif // COREMOD_ARITH_IMAGE_ARITH_CIM_CIM_CIM_H
