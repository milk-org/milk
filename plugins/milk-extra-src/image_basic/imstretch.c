// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imstretch.c
 * @brief Imstretch module
 */

/** @file imstretch.c
 */

#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

/**
 * Stretch image by coefficient
 * around a center point.
 */
imageID basic_stretch(const char *__restrict name_in,
                      const char *__restrict name_out,
                      float coeff,
                      long  Xcenter,
                      long  Ycenter)
{
    IMGID imgin = imgid_make_from_name(name_in);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t naxes[2];
    naxes[0] = imgin.md->size[0];
    naxes[1] = imgin.md->size[1];

    IMGID imgout       = imgid_make_from_name_2D(name_out, naxes[0], naxes[1]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (uint32_t ii = 0; ii < naxes[0]; ii++)
    {
        for (uint32_t jj = 0; jj < naxes[0]; jj++)
        {
            long i = Xcenter + (long) (1.0 * (ii - Xcenter) * coeff);
            long j = Ycenter + (long) (1.0 * (jj - Ycenter) * coeff);
            if ((i < naxes[0]) && (j < naxes[1]) && (i > -1) && (j > -1))
            {
                imgout.im->array.F[jj * naxes[0] + ii] +=
                    imgin.im->array.F[j * naxes[0] + i] / coeff / coeff;
            }
        }
    }

    arith_image_cstmult_inplace(name_out, arith_image_total(name_in) / arith_image_total(name_out));

    return imgout.ID;
}

/**
 * Stretch image with multi-step range
 * of coefficients and apodization.
 */
imageID basic_stretch_range(const char *__restrict name_in,
                            const char *__restrict name_out,
                            float coeff1,
                            float coeff2,
                            long  Xcenter,
                            long  Ycenter,
                            long  NBstep,
                            float ApoCoeff)
{
    DEBUG_TRACE_FSTART();

    float eps = 1.0e-5;

    IMGID imgin = imgid_make_from_name(name_in);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t naxes[2];
    naxes[0] = imgin.md->size[0];
    naxes[1] = imgin.md->size[1];

    IMGID imgout       = imgid_make_from_name_2D(name_out, naxes[0], naxes[1]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (long step = 0; step < NBstep; step++)
    {
        fprintf(stdout, ".");
        fflush(stdout);
        float coeff = coeff1 + (coeff2 - coeff1) * (1.0 * step / (NBstep - 1));
        float x     = (coeff - (coeff1 + coeff2) / 2.0) / ((coeff2 - coeff1) / 2.0);
        float mcoeff;
        if (ApoCoeff > eps)
        {
            mcoeff = pow((1.0 - pow((fabs(x) - (1.0 - ApoCoeff)) / ApoCoeff, 2.0)), 4.0);
        }
        else
        {
            mcoeff = 1.0;
        }

        if ((1.0 - x * x) < eps)
        {
            mcoeff = 0.0;
        }
        if (fabs(x) < ApoCoeff)
        {
            mcoeff = 1.0;
        }

        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            for (uint32_t jj = 0; jj < naxes[1]; jj++)
            {
                x       = (1.0 * (ii - Xcenter) * coeff) + Xcenter;
                float y = (1.0 * (jj - Ycenter) * coeff) + Ycenter;
                long  i = (long) x;
                long  j = (long) y;
                float u = x - i;
                float t = y - j;
                if ((i < naxes[0] - 1) && (j < naxes[1] - 1) && (i > -1) && (j > -1))
                {
                    float tmp = (1.0 - u) * (1.0 - t) * imgin.im->array.F[j * naxes[0] + i];
                    tmp += (1.0 - u) * t * imgin.im->array.F[(j + 1) * naxes[0] + i];
                    tmp += u * (1.0 - t) * imgin.im->array.F[j * naxes[0] + i + 1];
                    tmp += u * t * imgin.im->array.F[(j + 1) * naxes[0] + i + 1];
                    imgout.im->array.F[jj * naxes[0] + ii] += mcoeff * tmp / coeff / coeff;
                }
            }
        }
    }

    fprintf(stdout, "\n");
    arith_image_cstmult_inplace(name_out, arith_image_total(name_in) / arith_image_total(name_out));

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}

/**
 * Stretch image by coefficient around
 * center of image.
 */
imageID basic_stretchc(const char *__restrict name_in, const char *__restrict name_out, float coeff)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(name_in);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t naxes[2];
    naxes[0]     = imgin.md->size[0];
    naxes[1]     = imgin.md->size[1];
    long Xcenter = naxes[0] / 2;
    long Ycenter = naxes[1] / 2;

    IMGID imgout       = imgid_make_from_name_2D(name_out, naxes[0], naxes[1]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (uint32_t ii = 0; ii < naxes[0]; ii++)
    {
        for (uint32_t jj = 0; jj < naxes[0]; jj++)
        {
            long i = Xcenter + (long) (1.0 * (ii - Xcenter) * coeff);
            long j = Ycenter + (long) (1.0 * (jj - Ycenter) * coeff);
            if ((i < naxes[0]) && (j < naxes[1]) && (i > -1) && (j > -1))
            {
                imgout.im->array.F[jj * naxes[0] + ii] +=
                    imgin.im->array.F[j * naxes[0] + i] / coeff / coeff;
            }
        }
    }

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}
