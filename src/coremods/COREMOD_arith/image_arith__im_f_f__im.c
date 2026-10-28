// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im_f_f__im.c
 * @brief   arith functions
 *
 * input : image, float, float
 * output: image
 *
 * Uses FPS V2 framework.
 */


#include <stdint.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "image_arith__im_f_f__im.h"
#include "image_elem_block.h"
#include "imfunctions.h"
#include "imgid_arith_helpers.h"
#include "mathfuncs.h"

#define TRUNC_BLK 1024

int arith_image_trunc_IMGID(IMGID *imgin, double f1, double f2, IMGID *imgout)
{
    if (imgin->im == NULL)
    {
        return RETURN_FAILURE;
    }
    if (imgin->md->datatype == _DATATYPE_FLOAT || imgin->md->datatype == _DATATYPE_DOUBLE)
    {
        imgid_ensure_output(imgin, imgout);
        uint64_t nelement = imgout->md->nelement;
        uint64_t nblk     = (nelement + TRUNC_BLK - 1) / TRUNC_BLK;
        int      nth      = milk_omp_team_size(nelement, 262144);
        if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->mdt->datatype == _DATATYPE_FLOAT)
        {
            const float *MILK_RESTRICT pin  = MILK_ASSUME_ALIGNED(imgin->im->array.F);
            float *MILK_RESTRICT       pout = MILK_ASSUME_ALIGNED(imgout->im->array.F);
#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t off = b * TRUNC_BLK;
                int      len = (int) ((nelement - off < TRUNC_BLK) ? (nelement - off) : TRUNC_BLK);
                arith_elem_clamp_f_inline(pout + off, pin + off, f1, f2, len);
            }
            return 0;
        }
        if (imgin->md->datatype == _DATATYPE_DOUBLE && imgout->mdt->datatype == _DATATYPE_DOUBLE)
        {
            const double *MILK_RESTRICT pin  = MILK_ASSUME_ALIGNED(imgin->im->array.D);
            double *MILK_RESTRICT       pout = MILK_ASSUME_ALIGNED(imgout->im->array.D);
#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
            for (uint64_t b = 0; b < nblk; b++)
            {
                uint64_t off = b * TRUNC_BLK;
                int      len = (int) ((nelement - off < TRUNC_BLK) ? (nelement - off) : TRUNC_BLK);
                arith_elem_clamp_d_inline(pout + off, pin + off, f1, f2, len);
            }
            return 0;
        }
    }
    arith_image_function_1ff_1_IMGID(imgin, f1, f2, imgout, &Ptrunc);
    return (0);
}

int arith_image_trunc(const char *ID_name, double f1, double f2, const char *ID_out)
{
    IMGID imgin  = imgid_make_from_name(ID_name);
    IMGID imgout = imgid_make_from_name(ID_out);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (imgout.ID == -1)
    {
        imgout.mdt->shared = dcshareddft;
        imgout.mdt->NBkw   = NB_KEYWNODE_MAX;
    }
    int ret = arith_image_trunc_IMGID(&imgin, f1, f2, &imgout);
    if (imgout.ID == -1 && imgout.im != NULL)
    {
        RegisterIMGID(&imgout, dcimg, dcnimg);
    }
    imgid_free(&imgin);
    imgid_free(&imgout);
    return ret;
}

int arith_image_trunc_inplace(const char *ID_name, double f1, double f2)
{
    IMGID img = imgid_make_from_name(ID_name);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1 || img.im == NULL)
    {
        return RETURN_FAILURE;
    }
    uint64_t nelement = img.im->md[0].nelement;
    uint64_t nblk     = (nelement + TRUNC_BLK - 1) / TRUNC_BLK;
    uint8_t  dt       = img.im->md[0].datatype;
    int      nth      = milk_omp_team_size(nelement, 262144);
    if (dt == _DATATYPE_FLOAT)
    {
        float *MILK_RESTRICT p = MILK_ASSUME_ALIGNED(img.im->array.F);
#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * TRUNC_BLK;
            int      len = (int) ((nelement - off < TRUNC_BLK) ? (nelement - off) : TRUNC_BLK);
            arith_elem_clamp_f_inline(p + off, p + off, f1, f2, len);
        }
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        double *MILK_RESTRICT p = MILK_ASSUME_ALIGNED(img.im->array.D);
#pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
        for (uint64_t b = 0; b < nblk; b++)
        {
            uint64_t off = b * TRUNC_BLK;
            int      len = (int) ((nelement - off < TRUNC_BLK) ? (nelement - off) : TRUNC_BLK);
            arith_elem_clamp_d_inline(p + off, p + off, f1, f2, len);
        }
    }
    else
    {
        arith_image_function_1ff_1_inplace_IMGID(&img, f1, f2, &Ptrunc);
    }
    imgid_free(&img);
    return (0);
}
