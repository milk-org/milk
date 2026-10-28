// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im_im__im.c
 * @brief   Binary image x image arithmetic (img, img -> img)
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "image_arith__im_im__im.h"
#include "imfunctions.h"
#include "imgid_arith_helpers.h"
#include "libmilkcommon/milk_compiler.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

#define DEFINE_BINARY_IMGID_AND_STR(op, fptr)                                                   \
    int arith_image_##op##_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout)                   \
    {                                                                                           \
        if (imgin1->im == NULL || imgin2->im == NULL)                                           \
        {                                                                                       \
            return RETURN_FAILURE;                                                              \
        }                                                                                       \
        uint8_t dt1 = imgin1->md->datatype;                                                     \
        uint8_t dt2 = imgin2->md->datatype;                                                     \
        if (imgid_same_dims(imgin1, imgin2) && dt1 == dt2 &&                                    \
            (dt1 == _DATATYPE_FLOAT || dt1 == _DATATYPE_DOUBLE))                                \
        {                                                                                       \
            imgid_ensure_output(imgin1, imgout);                                                \
            uint64_t nelement = imgout->md->nelement;                                           \
            int      nth      = milk_omp_team_size(nelement, 262144);                           \
            if (dt1 == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)             \
            {                                                                                   \
                const float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.F);       \
                const float *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.F);       \
                float *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.F);       \
                _Pragma("omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for ( \
                    uint64_t i = 0; i < nelement; i++)                                          \
                {                                                                               \
                    po[i] = fptr##f(p1[i], p2[i]);                                              \
                }                                                                               \
                return RETURN_SUCCESS;                                                          \
            }                                                                                   \
            if (dt1 == _DATATYPE_DOUBLE && imgout->mdt->datatype == _DATATYPE_DOUBLE)           \
            {                                                                                   \
                const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin1->im->array.D);      \
                const double *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(imgin2->im->array.D);      \
                double *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.D);      \
                _Pragma("omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for ( \
                    uint64_t i = 0; i < nelement; i++)                                          \
                {                                                                               \
                    po[i] = fptr(p1[i], p2[i]);                                                 \
                }                                                                               \
                return RETURN_SUCCESS;                                                          \
            }                                                                                   \
        }                                                                                       \
        return arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &fptr);                   \
    }                                                                                           \
    int arith_image_##op(const char *ID1_name, const char *ID2_name, const char *ID_out)        \
    {                                                                                           \
        IMGID imgin1 = imgid_make_from_name(ID1_name);                                          \
        IMGID imgin2 = imgid_make_from_name(ID2_name);                                          \
        IMGID imgout = imgid_make_from_name(ID_out);                                            \
        resolveIMGID(&imgin1, ERRMODE_ABORT, dcimg, dcnimg);                                    \
        resolveIMGID(&imgin2, ERRMODE_ABORT, dcimg, dcnimg);                                    \
        resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);                                     \
        if (imgout.ID == -1)                                                                    \
        {                                                                                       \
            imgout.mdt->shared = dcshareddft;                                                   \
            imgout.mdt->NBkw   = NB_KEYWNODE_MAX;                                               \
        }                                                                                       \
        int ret = arith_image_##op##_IMGID(&imgin1, &imgin2, &imgout);                          \
        if (imgout.ID == -1 && imgout.im != NULL)                                               \
        {                                                                                       \
            RegisterIMGID(&imgout, dcimg, dcnimg);                                              \
        }                                                                                       \
        imgid_free(&imgin1);                                                                    \
        imgid_free(&imgin2);                                                                    \
        imgid_free(&imgout);                                                                    \
        return ret;                                                                             \
    }

MILK_BINARY_OPS_ALL(DEFINE_BINARY_IMGID_AND_STR)
#undef DEFINE_BINARY_IMGID_AND_STR

#define DEFINE_INPLACE(op, fptr)                                                                \
    int arith_image_##op##_inplace(const char *ID1_name, const char *ID2_name)                  \
    {                                                                                           \
        IMGID img1 = imgid_make_from_name(ID1_name);                                            \
        IMGID img2 = imgid_make_from_name(ID2_name);                                            \
        resolveIMGID(&img1, ERRMODE_ABORT, dcimg, dcnimg);                                      \
        resolveIMGID(&img2, ERRMODE_ABORT, dcimg, dcnimg);                                      \
        if (img1.im != NULL && img2.im != NULL && img1.md->nelement == img2.md->nelement &&     \
            img1.md->datatype == img2.md->datatype &&                                           \
            (img1.md->datatype == _DATATYPE_FLOAT || img1.md->datatype == _DATATYPE_DOUBLE))    \
        {                                                                                       \
            uint64_t nelement  = img1.md->nelement;                                             \
            int      nth       = milk_omp_team_size(nelement, 262144);                          \
            img1.im->md->write = 1;                                                             \
            if (img1.md->datatype == _DATATYPE_FLOAT)                                           \
            {                                                                                   \
                float *MILK_RESTRICT       p1 = MILK_ASSUME_ALIGNED(img1.im->array.F);          \
                const float *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(img2.im->array.F);          \
                _Pragma("omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for ( \
                    uint64_t i = 0; i < nelement; i++)                                          \
                {                                                                               \
                    p1[i] = fptr##f(p1[i], p2[i]);                                              \
                }                                                                               \
            }                                                                                   \
            else                                                                                \
            {                                                                                   \
                double *MILK_RESTRICT       p1 = MILK_ASSUME_ALIGNED(img1.im->array.D);         \
                const double *MILK_RESTRICT p2 = MILK_ASSUME_ALIGNED(img2.im->array.D);         \
                _Pragma("omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for ( \
                    uint64_t i = 0; i < nelement; i++)                                          \
                {                                                                               \
                    p1[i] = fptr(p1[i], p2[i]);                                                 \
                }                                                                               \
            }                                                                                   \
            img1.im->md->write = 0;                                                             \
            img1.im->md->cnt0++;                                                                \
            imgid_free(&img1);                                                                  \
            imgid_free(&img2);                                                                  \
            return 0;                                                                           \
        }                                                                                       \
        imgid_free(&img1);                                                                      \
        imgid_free(&img2);                                                                      \
        arith_image_function_2_1_inplace(ID1_name, ID2_name, &fptr);                            \
        return 0;                                                                               \
    }

MILK_BINARY_OPS_FULL(DEFINE_INPLACE)
#undef DEFINE_INPLACE
