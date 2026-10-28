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
#include "image_elem_block.h"
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
        int      nth      = milk_omp_team_size(nelement, 262144);                                  \
        if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)    \
        {                                                                                          \
            const float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.F);               \
            float *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.F);              \
            _Pragma(                                                                               \
                "omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for (uint64_t    \
                                                                                           i = 0;  \
                                                                                       i <         \
                                                                                       nelement;   \
                                                                                       i++)        \
            {                                                                                      \
                po[i] = fptr##f(p1[i]);                                                            \
            }                                                                                      \
        }                                                                                          \
        else if (imgin->md->datatype == _DATATYPE_DOUBLE &&                                        \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                        \
        {                                                                                          \
            const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.D);              \
            double *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.D);             \
            _Pragma(                                                                               \
                "omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for (uint64_t    \
                                                                                           i = 0;  \
                                                                                       i <         \
                                                                                       nelement;   \
                                                                                       i++)        \
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
        IMGID img = imgid_make_from_name(ID_name);                                                 \
        resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);                                           \
        if (img.ID == -1 || img.im == NULL)                                                        \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        uint64_t nelement = img.im->md->nelement;                                                  \
        uint8_t  dt       = img.im->md->datatype;                                                  \
        int      nth      = milk_omp_team_size(nelement, 262144);                                  \
        if (dt == _DATATYPE_FLOAT)                                                                 \
        {                                                                                          \
            float *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(img.im->array.F);                        \
            _Pragma(                                                                               \
                "omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for (uint64_t    \
                                                                                           i = 0;  \
                                                                                       i <         \
                                                                                       nelement;   \
                                                                                       i++)        \
            {                                                                                      \
                po[i] = fptr##f(po[i]);                                                            \
            }                                                                                      \
            img.im->md->write = 0;                                                                 \
            img.im->md->cnt0++;                                                                    \
        }                                                                                          \
        else if (dt == _DATATYPE_DOUBLE)                                                           \
        {                                                                                          \
            double *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(img.im->array.D);                       \
            _Pragma(                                                                               \
                "omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for (uint64_t    \
                                                                                           i = 0;  \
                                                                                       i <         \
                                                                                       nelement;   \
                                                                                       i++)        \
            {                                                                                      \
                po[i] = fptr(po[i]);                                                               \
            }                                                                                      \
            img.im->md->write = 0;                                                                 \
            img.im->md->cnt0++;                                                                    \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            uint64_t nblk     = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;                  \
            img.im->md->write = 1;                                                                 \
            _Pragma(                                                                               \
                "omp parallel for schedule(static) num_threads(nth) if (nth > 1)") for (uint64_t   \
                                                                                            b = 0; \
                                                                                        b < nblk;  \
                                                                                        b++)       \
            {                                                                                      \
                MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];                                       \
                uint64_t                off = b * ARITH_ELEM_BLK;                                  \
                int                     len =                                                      \
                    (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK); \
                arith_load_block_d_inline(img.im, off, len, buf);                                  \
                _Pragma("omp simd") for (int j = 0; j < len; j++)                                  \
                {                                                                                  \
                    buf[j] = fptr(buf[j]);                                                         \
                }                                                                                  \
                arith_store_block_d_inline(img.im, off, len, buf, 0, 0.0);                         \
            }                                                                                      \
            img.im->md->write = 0;                                                                 \
            img.im->md->cnt0++;                                                                    \
        }                                                                                          \
        imgid_free(&img);                                                                          \
        return 0;                                                                                  \
    }

MILK_UNARY_OPS(DEFINE_UNARY_OPS)
#undef DEFINE_UNARY_OPS
