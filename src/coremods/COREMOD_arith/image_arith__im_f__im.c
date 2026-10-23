// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im_f__im.c
 * @brief   Image + scalar arithmetic (img, float -> img)
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "image_arith__im_f__im.h"
#include "imfunctions.h"
#include "imgid_arith_helpers.h"
#include "libmilkcommon/milk_compiler.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

#define DEFINE_CST_IMGID_AND_STR(op, tag)                                                          \
    int arith_image_cst##op##_IMGID(IMGID *imgin, double f1, IMGID *imgout)                        \
    {                                                                                              \
        if (imgin->im == NULL)                                                                     \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        imgid_ensure_output(imgin, imgout);                                                        \
        uint64_t nelement = imgout->md->nelement;                                                  \
        if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)    \
        {                                                                                          \
            const float *MILK_RESTRICT p1  = MILK_ASSUME_ALIGNED(imgin->im->array.F);              \
            float *MILK_RESTRICT       po  = MILK_ASSUME_ALIGNED(imgout->im->array.F);             \
            float                      cf1 = (float) f1;                                           \
            _Pragma("omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)") for (uint64_t i =  \
                                                                                         0;        \
                                                                                     i < nelement; \
                                                                                     i++)          \
            {                                                                                      \
                po[i] = P##op##f(p1[i], cf1);                                                      \
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
                po[i] = P##op(p1[i], f1);                                                          \
            }                                                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            arith_image_function_1f_1_IMGID(imgin, f1, imgout, &P##op);                            \
        }                                                                                          \
        return RETURN_SUCCESS;                                                                     \
    }                                                                                              \
    int arith_image_cst##op(const char *ID_name, double f1, const char *ID_out)                    \
    {                                                                                              \
        IMGID imgin  = imgid_make_from_name(ID_name);                                              \
        IMGID imgout = imgid_make_from_name(ID_out);                                               \
        resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);                                        \
        resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);                                        \
        if (imgout.ID == -1)                                                                       \
        {                                                                                          \
            imgout.mdt->shared = dcshareddft;                                                      \
            imgout.mdt->NBkw   = NB_KEYWNODE_MAX;                                                  \
        }                                                                                          \
        int ret = arith_image_cst##op##_IMGID(&imgin, f1, &imgout);                                \
        if (imgout.ID == -1 && imgout.im != NULL)                                                  \
        {                                                                                          \
            RegisterIMGID(&imgout, dcimg, dcnimg);                                                 \
        }                                                                                          \
        imgid_free(&imgin);                                                                        \
        imgid_free(&imgout);                                                                       \
        return ret;                                                                                \
    }

MILK_CST_OPS_ALL(DEFINE_CST_IMGID_AND_STR)
#undef DEFINE_CST_IMGID_AND_STR

#define DEFINE_CST_INPLACE(op, tag)                                   \
    int arith_image_cst##op##_inplace(const char *ID_name, double f1) \
    {                                                                 \
        arith_image_function_1f_1_inplace(ID_name, f1, &P##op);       \
        return 0;                                                     \
    }

MILK_CST_OPS_OPT_FULL(DEFINE_CST_INPLACE)
MILK_CST_OPS_FPTR_FULL(DEFINE_CST_INPLACE)
#undef DEFINE_CST_INPLACE
