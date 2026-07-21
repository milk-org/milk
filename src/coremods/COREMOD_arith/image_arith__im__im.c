// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im__im.c
 * @brief   Unary image arithmetic (image → image)
 *
 * Applies a single math function to every pixel of
 * an input image, producing an output image.
 *
 * All call surfaces are generated from the MILK_UNARY_OPS
 * X-macro table:
 *
 *  - arith_image_<op>_IMGID(imgin, imgout)
 *    Modern IMGID API.
 *  - arith_image_<op>(name_in, name_out)
 *    String-based API for CLI use.
 *  - arith_image_<op>_inplace(name)
 *    In-place via string name.
 */


#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__im__im.h"


/* MILK_UNARY_OPS table is defined in image_arith__im__im.h */


/* ----------------------------------------------------------
 * 1. Modern IMGID wrappers
 * ---------------------------------------------------------- */

#define DEFINE_IMGID(op, fptr)                                    \
    int arith_image_##op##_IMGID(IMGID *imgin, IMGID *imgout)     \
    {                                                             \
        return arith_image_##op##_optimized_IMGID(imgin, imgout); \
    }

MILK_UNARY_OPS(DEFINE_IMGID)
#undef DEFINE_IMGID


/* ----------------------------------------------------------
 * 2. String-based wrappers  (name → name)
 * ---------------------------------------------------------- */

#define DEFINE_STRING(op, fptr)                                   \
    int arith_image_##op(const char *ID_name, const char *ID_out) \
    {                                                             \
        IMGID imgin  = imgid_make_from_name(ID_name);             \
        IMGID imgout = imgid_make_from_name(ID_out);              \
        return arith_image_##op##_IMGID(&imgin, &imgout);         \
    }

MILK_UNARY_OPS(DEFINE_STRING)
#undef DEFINE_STRING


/* ----------------------------------------------------------
 * 3. In-place wrappers  (name → name modified)
 * ---------------------------------------------------------- */

#define DEFINE_INPLACE(op, fptr)                          \
    int arith_image_##op##_inplace(const char *ID_name)   \
    {                                                     \
        arith_image_function_1_1_inplace(ID_name, &fptr); \
        return 0;                                         \
    }

MILK_UNARY_OPS(DEFINE_INPLACE)
#undef DEFINE_INPLACE
