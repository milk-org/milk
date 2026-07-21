// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file  image_arith__im__im.h
 * @brief Unary image arithmetic prototypes (image → image).
 *
 * All prototypes are generated from the MILK_UNARY_OPS X-macro
 * table.  Adding a new operation requires only one edit:
 * add a row to MILK_UNARY_OPS in this header.
 *
 * Three call surfaces per operation:
 *   arith_image_<op>_IMGID(imgin, imgout)
 *   arith_image_<op>(name_in, name_out)
 *   arith_image_<op>_inplace(name)
 */

#ifndef IMAGE_ARITH_IM_IM_H
#define IMAGE_ARITH_IM_IM_H

#include <libfps/IMGID.h>

double Ppositive(double a);


/* ==========================================================
 * X-macro table: all unary image operations.
 *
 * X(op, fptr)
 *   op   — operation suffix
 *   fptr — function-pointer (from mathfuncs.h)
 * ========================================================== */

#define MILK_UNARY_OPS(X) \
    X(acos, Pacos)        \
    X(asin, Pasin)        \
    X(atan, Patan)        \
    X(ceil, Pceil)        \
    X(cos, Pcos)          \
    X(cosh, Pcosh)        \
    X(exp, Pexp)          \
    X(fabs, Pfabs)        \
    X(floor, Pfloor)      \
    X(ln, Pln)            \
    X(log, Plog)          \
    X(sqrt, Psqrt)        \
    X(sin, Psin)          \
    X(sinh, Psinh)        \
    X(tan, Ptan)          \
    X(tanh, Ptanh)        \
    X(positive, Ppositive)


/* ----------------------------------------------------------
 * Generated prototypes — IMGID, string, and inplace APIs
 * ---------------------------------------------------------- */

#define IMAGE_ARITH_DECLARE_UNARY(op, fptr)                        \
    int arith_image_##op##_IMGID(IMGID *imgin, IMGID *imgout);     \
    int arith_image_##op(const char *ID_name, const char *ID_out); \
    int arith_image_##op##_inplace(const char *ID_name);

MILK_UNARY_OPS(IMAGE_ARITH_DECLARE_UNARY)
#undef IMAGE_ARITH_DECLARE_UNARY

#endif /* IMAGE_ARITH_IM_IM_H */
