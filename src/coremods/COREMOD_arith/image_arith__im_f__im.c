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
#include "image_elem_block.h"
#include "imfunctions.h"
#include "imgid_arith_helpers.h"
#include "libmilkcommon/milk_compiler.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

#define CST_BLK 1024

#define DEFINE_CST_IMGID_AND_STR(op, tag)                                                           \
    int arith_image_cst##op##_IMGID(IMGID *imgin, double f1, IMGID *imgout)                         \
    {                                                                                               \
        if (imgin->im == NULL)                                                                      \
        {                                                                                           \
            return RETURN_FAILURE;                                                                  \
        }                                                                                           \
        imgid_ensure_output(imgin, imgout);                                                         \
        uint64_t nelement = imgout->md->nelement;                                                   \
        int      nth      = milk_omp_team_size(nelement, 262144);                                   \
        if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)     \
        {                                                                                           \
            const float *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.F);                \
            float *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.F);               \
            if (&P##op == &Ppow)                                                                    \
            {                                                                                       \
                uint64_t nblk = (nelement + CST_BLK - 1) / CST_BLK;                                 \
                _Pragma("omp parallel for schedule(static) num_threads(nth) if (nth > 1)") for (    \
                    uint64_t b = 0; b < nblk; b++)                                                  \
                {                                                                                   \
                    uint64_t off = b * CST_BLK;                                                     \
                    int      len = (int) ((nelement - off < CST_BLK) ? (nelement - off) : CST_BLK); \
                    arith_elem_pow_const_f_inline(po + off, p1 + off, f1, len);                     \
                }                                                                                   \
            }                                                                                       \
            else                                                                                    \
            {                                                                                       \
                float cf1 = (float) f1;                                                             \
                _Pragma("omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for (     \
                    uint64_t i = 0; i < nelement; i++)                                              \
                {                                                                                   \
                    po[i] = P##op##f(p1[i], cf1);                                                   \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
        else if (imgin->md->datatype == _DATATYPE_DOUBLE &&                                         \
                 imgout->mdt->datatype == _DATATYPE_DOUBLE)                                         \
        {                                                                                           \
            const double *MILK_RESTRICT p1 = MILK_ASSUME_ALIGNED(imgin->im->array.D);               \
            double *MILK_RESTRICT       po = MILK_ASSUME_ALIGNED(imgout->im->array.D);              \
            if (&P##op == &Ppow)                                                                    \
            {                                                                                       \
                uint64_t nblk = (nelement + CST_BLK - 1) / CST_BLK;                                 \
                _Pragma("omp parallel for schedule(static) num_threads(nth) if (nth > 1)") for (    \
                    uint64_t b = 0; b < nblk; b++)                                                  \
                {                                                                                   \
                    uint64_t off = b * CST_BLK;                                                     \
                    int      len = (int) ((nelement - off < CST_BLK) ? (nelement - off) : CST_BLK); \
                    arith_elem_pow_const_d_inline(po + off, p1 + off, f1, len);                     \
                }                                                                                   \
            }                                                                                       \
            else                                                                                    \
            {                                                                                       \
                _Pragma("omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for (     \
                    uint64_t i = 0; i < nelement; i++)                                              \
                {                                                                                   \
                    po[i] = P##op(p1[i], f1);                                                       \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            arith_image_function_1f_1_IMGID(imgin, f1, imgout, &P##op);                             \
        }                                                                                           \
        return RETURN_SUCCESS;                                                                      \
    }                                                                                               \
    int arith_image_cst##op(const char *ID_name, double f1, const char *ID_out)                     \
    {                                                                                               \
        IMGID imgin  = imgid_make_from_name(ID_name);                                               \
        IMGID imgout = imgid_make_from_name(ID_out);                                                \
        resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);                                         \
        resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);                                         \
        if (imgout.ID == -1)                                                                        \
        {                                                                                           \
            imgout.mdt->shared = dcshareddft;                                                       \
            imgout.mdt->NBkw   = NB_KEYWNODE_MAX;                                                   \
        }                                                                                           \
        int ret = arith_image_cst##op##_IMGID(&imgin, f1, &imgout);                                 \
        if (imgout.ID == -1 && imgout.im != NULL)                                                   \
        {                                                                                           \
            RegisterIMGID(&imgout, dcimg, dcnimg);                                                  \
        }                                                                                           \
        imgid_free(&imgin);                                                                         \
        imgid_free(&imgout);                                                                        \
        return ret;                                                                                 \
    }

MILK_CST_OPS_ALL(DEFINE_CST_IMGID_AND_STR)
#undef DEFINE_CST_IMGID_AND_STR

#define DEFINE_CST_INPLACE(op, tag)                                                                \
    int arith_image_cst##op##_inplace(const char *ID_name, double f1)                              \
    {                                                                                              \
        IMGID img = imgid_make_from_name(ID_name);                                                 \
        resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);                                           \
        if (img.ID == -1 || img.im == NULL)                                                        \
        {                                                                                          \
            return RETURN_FAILURE;                                                                 \
        }                                                                                          \
        uint64_t nelement = img.im->md[0].nelement;                                                \
        uint8_t  dt       = img.im->md[0].datatype;                                                \
        int      nth      = milk_omp_team_size(nelement, 262144);                                  \
        if (dt == _DATATYPE_FLOAT)                                                                 \
        {                                                                                          \
            float *MILK_RESTRICT po = MILK_ASSUME_ALIGNED(img.im->array.F);                        \
            float                cf = (float) f1;                                                  \
            _Pragma(                                                                               \
                "omp parallel for simd num_threads(nth) if (parallel : nth > 1)") for (uint64_t    \
                                                                                           i = 0;  \
                                                                                       i <         \
                                                                                       nelement;   \
                                                                                       i++)        \
            {                                                                                      \
                po[i] = P##op##f(po[i], cf);                                                       \
            }                                                                                      \
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
                po[i] = P##op(po[i], f1);                                                          \
            }                                                                                      \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            uint64_t nblk = (nelement + CST_BLK - 1) / CST_BLK;                                    \
            _Pragma(                                                                               \
                "omp parallel for schedule(static) num_threads(nth) if (nth > 1)") for (uint64_t   \
                                                                                            b = 0; \
                                                                                        b < nblk;  \
                                                                                        b++)       \
            {                                                                                      \
                MILK_ALIGNED(64) double buf[CST_BLK];                                              \
                uint64_t                off = b * CST_BLK;                                         \
                int len = (int) ((nelement - off < CST_BLK) ? (nelement - off) : CST_BLK);         \
                arith_load_block_d_inline(img.im, off, len, buf);                                  \
                if (&P##op == &Ppow)                                                               \
                {                                                                                  \
                    arith_elem_pow_const_d_inline(buf, buf, f1, len);                              \
                }                                                                                  \
                else                                                                               \
                {                                                                                  \
                    _Pragma("omp simd") for (int j = 0; j < len; j++)                              \
                    {                                                                              \
                        buf[j] = P##op(buf[j], f1);                                                \
                    }                                                                              \
                }                                                                                  \
                arith_store_block_d_inline(img.im, off, len, buf, 0, 0.0);                         \
            }                                                                                      \
        }                                                                                          \
        imgid_free(&img);                                                                          \
        return 0;                                                                                  \
    }

MILK_CST_OPS_OPT_FULL(DEFINE_CST_INPLACE)
MILK_CST_OPS_FPTR_FULL(DEFINE_CST_INPLACE)
#undef DEFINE_CST_INPLACE
