/**
 * @file    image_total.c
 * @brief   sum image pixels
 *
 *
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#endif
#include "image_total.h"

#include <math.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"

#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 100000
#define OMP_FOR_SIMD _Pragma("omp for simd")
#else
#define OMP_FOR_SIMD
#endif

/**
 * @brief Typed reduction over all elements
 *
 * Generates one else-if branch per real datatype.
 * Each branch creates a restrict+aligned typed pointer
 * and runs an OMP-simd reduction loop.
 *
 * @param DTYPE   _DATATYPE_* constant
 * @param ACC     Union accessor (F, D, UI8, ...)
 * @param CTYPE   C type (float, double, uint8_t, ...)
 *
 * Captured from enclosing scope:
 *   imgin, nelement, lvalue, datatype
 *
 * ACCUM_CAST and ELEM_EXPR(v) must be #defined
 * before invoking FOREACH_REAL_DATATYPE.
 */
#define REDUCE_CASE(DTYPE, ACC, CTYPE)                  \
    else if (datatype == DTYPE)                         \
    {                                                   \
        CTYPE * MILK_RESTRICT ptr =                     \
            MILK_ASSUME_ALIGNED(imgin->im->array.ACC);  \
        OMP_FOR_SIMD                                    \
        for (uint64_t ii = 0; ii < nelement; ii++)      \
        {                                               \
            lvalue += (ACCUM_CAST) ELEM_EXPR(ptr[ii]);  \
        }                                               \
    }


double MILK_HOT arith_image_total_IMGID(IMGID *imgin)
{
    long double lvalue; // uses long double internally
    uint64_t    nelement;
    uint8_t     datatype;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    if (imgin->ID == -1) {
        return RETURN_FAILURE;
    }

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

#define ACCUM_CAST long double
#define ELEM_EXPR(v) (v)

#ifdef _OPENMP
    #pragma omp parallel reduction(+:lvalue) \
        if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if (0) {}  // anchor for else-if chain
    FOREACH_REAL_DATATYPE(REDUCE_CASE)
    else
    {
        PRINT_ERROR("invalid data type");
    }

#ifdef _OPENMP
    } // omp parallel
#endif

#undef ACCUM_CAST
#undef ELEM_EXPR

    return (double) lvalue;
}

double arith_image_total(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_total_IMGID(&imgin);
}

double MILK_HOT arith_image_sumsquare_IMGID(IMGID *imgin)
{
    double   lvalue; // uses double internally
    uint64_t nelement;
    uint8_t  datatype;

    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    if (imgin->ID == -1) {
        return RETURN_FAILURE;
    }

    nelement = imgin->md[0].nelement;

    lvalue = 0.0;

#define ACCUM_CAST double
#define ELEM_EXPR(v) ((v) * (v))

#ifdef _OPENMP
    #pragma omp parallel reduction(+:lvalue) \
        if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if (0) {}  // anchor for else-if chain
    FOREACH_REAL_DATATYPE(REDUCE_CASE)
    else
    {
        PRINT_ERROR("invalid data type");
    }

#ifdef _OPENMP
    } // omp parallel
#endif

#undef ACCUM_CAST
#undef ELEM_EXPR

    return (double) lvalue;
}

#undef REDUCE_CASE

double arith_image_sumsquare(const char *ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    return arith_image_sumsquare_IMGID(&imgin);
}
