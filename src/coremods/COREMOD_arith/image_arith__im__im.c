/**
 * @file    image_arith__im__im.c
 * @brief   Unary image arithmetic (image → image)
 *
 * Applies a single math function to every pixel of
 * an input image, producing an output image.
 *
 * All call surfaces are generated from the UNARY_OPS
 * X-macro table:
 *

 *    Legacy API using raw image slot indices.
 *  - arith_image_<op>_IMGID(imgin, imgout)
 *    Modern IMGID API.
 *  - arith_image_<op>(name_in, name_out)
 *    String-based API for CLI use.
 *  - arith_image_<op>_inplace_byID(ID)
 *    In-place via legacy image ID.
 *  - arith_image_<op>_inplace(name)
 *    In-place via string name.
 */


#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__im__im.h"


/* ==========================================================
 * X-macro table: all unary image operations.
 *
 * Columns:  X(op, fptr)
 *   op   — operation suffix
 *   fptr — function-pointer variable from mathfuncs.h
 * ========================================================== */

#define UNARY_OPS(X) \
    X(acos,     Pacos)     \
    X(asin,     Pasin)     \
    X(atan,     Patan)     \
    X(ceil,     Pceil)     \
    X(cos,      Pcos)      \
    X(cosh,     Pcosh)     \
    X(exp,      Pexp)      \
    X(fabs,     Pfabs)     \
    X(floor,    Pfloor)    \
    X(ln,       Pln)       \
    X(log,      Plog)      \
    X(sqrt,     Psqrt)     \
    X(sin,      Psin)      \
    X(sinh,     Psinh)     \
    X(tan,      Ptan)      \
    X(tanh,     Ptanh)     \
    X(positive, Ppositive)


/* ----------------------------------------------------------
 * 1. Legacy by-ID wrappers  (ID, IDout)
 * ---------------------------------------------------------- */







/* ----------------------------------------------------------
 * 2. Modern IMGID wrappers
 * ---------------------------------------------------------- */

#define DEFINE_IMGID(op, fptr) \
int arith_image_##op##_IMGID(  \
    IMGID *imgin,              \
    IMGID *imgout)             \
{                              \
    return arith_image_##op##_optimized_IMGID( \
        imgin, imgout);        \
}

UNARY_OPS(DEFINE_IMGID)
#undef DEFINE_IMGID


/* ----------------------------------------------------------
 * 3. String-based wrappers  (name → name)
 * ---------------------------------------------------------- */

#define DEFINE_STRING(op, fptr) \
int arith_image_##op(                        \
    const char *ID_name,                     \
    const char *ID_out)                      \
{                                            \
    IMGID imgin =                            \
        imgid_make_from_name(ID_name);       \
    IMGID imgout =                           \
        imgid_make_from_name(ID_out);        \
    return arith_image_##op##_IMGID(         \
        &imgin, &imgout);                    \
}

UNARY_OPS(DEFINE_STRING)
#undef DEFINE_STRING


/* ----------------------------------------------------------
 * 4. In-place-by-ID wrappers
 * ---------------------------------------------------------- */







/* ----------------------------------------------------------
 * 5. In-place wrappers  (name → name modified)
 * ---------------------------------------------------------- */

#define DEFINE_INPLACE(op, fptr) \
int arith_image_##op##_inplace(   \
    const char *ID_name)          \
{                                 \
    arith_image_function_1_1_inplace( \
        ID_name, &fptr);          \
    return 0;                     \
}

UNARY_OPS(DEFINE_INPLACE)
#undef DEFINE_INPLACE
