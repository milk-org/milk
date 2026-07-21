// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    im_math_unary.c
 * @brief   Unary optimized image math functions
 */

#include <math.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "libmilkcommon/milk_compiler.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "imgid_arith_helpers.h"

#include "imfunctions.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

/* ---------------------------------------------------------- */
/* Unary optimized: calls float/double math directly          */
/* ---------------------------------------------------------- */
#define ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(name, funcname, funcname_f)                            \
    errno_t arith_image_##name##_optimized_IMGID(IMGID *imgin, IMGID *imgout)                      \
    {                                                                                              \
        DEBUG_TRACE_FSTART();                                                                      \
        if (imgin->im == NULL)                                                                     \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        imgid_ensure_output(imgin, imgout);                                                        \
        uint64_t nelement = imgout->md->nelement;                                                  \
        if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)    \
        {                                                                                          \
            float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.F);                     \
            float *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(imgout->im->array.F);                    \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = funcname_f(p1[i]);                                                         \
            }                                                                                      \
        }                                                                                          \
        else if (imgin->md->datatype == _DATATYPE_DOUBLE &&                                        \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.D);                    \
            double *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(imgout->im->array.D);                   \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = funcname(p1[i]);                                                           \
            }                                                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_1_1_IMGID(imgin, imgout, &P##name);                               \
        }                                                                                          \
        DEBUG_TRACE_FEXIT();                                                                       \
        return RETURN_SUCCESS;                                                                     \
    }

ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(acos, acos, acosf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(asin, asin, asinf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(atan, atan, atanf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(ceil, ceil, ceilf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(cos, cos, cosf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(cosh, cosh, coshf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(exp, exp, expf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(fabs, fabs, fabsf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(floor, floor, floorf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(ln, log, logf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(log, log10, log10f)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(sqrt, sqrt, sqrtf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(sin, sin, sinf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(sinh, sinh, sinhf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(tan, tan, tanf)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(tanh, tanh, tanhf)
