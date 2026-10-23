// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im__im.c
 * @brief   Unary image arithmetic (image -> image)
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "image_arith__im__im.h"
#include "imfunctions.h"
#include "imgid_arith_helpers.h"
#include "libmilkcommon/milk_compiler.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

#define DEFINE_UNARY_OPS(op, fptr)                                                                 \
    int arith_image_##op##_IMGID(IMGID *imgin, IMGID *imgout)                                      \
    {                                                                                              \
        if (imgin->im == NULL)                                                                     \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        imgid_ensure_output(imgin, imgout);                                                        \
        uint64_t nelement = imgout->md->nelement;                                                  \
        if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)    \
        {                                                                                          \
            const float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.F);               \
            float *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.F);              \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = fptr##f(p1[i]);                                                            \
            }                                                                                      \
        }                                                                                          \
        else if (imgin->md->datatype == _DATATYPE_DOUBLE &&                                        \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.D);              \
            double *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.D);             \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = fptr(p1[i]);                                                               \
            }                                                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_1_1_IMGID(imgin, imgout, &fptr);                                  \
        }                                                                                          \
        return RETURN_SUCCESS;                                                                     \
    }                                                                                              \
    int arith_image_##op(const char *ID_name, const char *ID_out)                                  \
    {                                                                                              \
        IMGID imgin  = imgid_make_from_name(ID_name);                                              \
        IMGID imgout = imgid_make_from_name(ID_out);                                               \
        return arith_image_##op##_IMGID(&imgin, &imgout);                                          \
    }                                                                                              \
    int arith_image_##op##_inplace(const char *ID_name)                                            \
    {                                                                                              \
        arith_image_function_1_1_inplace(ID_name, &fptr);                                          \
        return 0;                                                                                  \
    }

MILK_UNARY_OPS(DEFINE_UNARY_OPS)
#undef DEFINE_UNARY_OPS
