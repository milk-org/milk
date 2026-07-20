// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_dxdy.c
 * @brief   spatial derivatives
 *
 *
 */
#include <assert.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_dxdy.h"

#include "COREMOD_memory/COREMOD_memory.h"

imageID arith_image_dx_IMGID(IMGID *imgin, IMGID *imgout)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(imgin, ERRMODE_ABORT);
    uint8_t   datatype = imgin->md[0].datatype;
    uint8_t   naxis    = imgin->md[0].naxis;
    if(naxis != 2)
    {
        PRINT_ERROR("Function only supports 2-D images\n");
        abort();
    }

    imgout->naxis = 2;
    imgout->size[0] = imgin->md[0].size[0];
    imgout->size[1] = imgin->md[0].size[1];
    imgout->datatype = datatype;
    imgout->shared = data.SHARED_DFT;
    imgout->NBkw   = NB_KEYWNODE_MAX;

    imcreateIMGID(imgout);

    uint32_t xsize = imgout->size[0];
    uint32_t ysize = imgout->size[1];

    for(uint32_t jj = 0; jj < ysize; jj++)
    {
        for(uint32_t ii = 1; ii < xsize - 1; ii++)
            imgout->im->array.F[jj * xsize + ii] =
                (imgin->im->array.F[jj * xsize + ii + 1] -
                 imgin->im->array.F[jj * xsize + ii - 1]) /
                2.0;
        imgout->im->array.F[jj * xsize] =
            imgin->im->array.F[jj * xsize + 1] -
            imgin->im->array.F[jj * xsize];
        imgout->im->array.F[jj * xsize + xsize - 1] =
            imgin->im->array.F[jj * xsize + xsize - 1] -
            imgin->im->array.F[jj * xsize + xsize - 2];
    }

    DEBUG_TRACE_FEXIT();
    return imgout->ID;
}

imageID arith_image_dx(const char *ID_name, const char *IDout_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(IDout_name);

    return arith_image_dx_IMGID(&imgin, &imgout);
}

imageID arith_image_dy_IMGID(IMGID *imgin, IMGID *imgout)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(imgin, ERRMODE_ABORT);
    uint8_t   datatype = imgin->md[0].datatype;
    uint8_t   naxis    = imgin->md[0].naxis;
    if(naxis != 2)
    {
        PRINT_ERROR("Function only supports 2-D images\n");
        abort();
    }

    imgout->naxis = 2;
    imgout->size[0] = imgin->md[0].size[0];
    imgout->size[1] = imgin->md[0].size[1];
    imgout->datatype = datatype;
    imgout->shared = data.SHARED_DFT;
    imgout->NBkw   = NB_KEYWNODE_MAX;

    imcreateIMGID(imgout);

    uint32_t xsize = imgout->size[0];
    uint32_t ysize = imgout->size[1];

    for(uint32_t ii = 0; ii < xsize; ii++)
    {
        for(uint32_t jj = 1; jj < ysize - 1; jj++)
        {
            imgout->im->array.F[jj * xsize + ii] =
                (imgin->im->array.F[(jj + 1) * xsize + ii] -
                 imgin->im->array.F[(jj - 1) * xsize + ii]) /
                2.0;
        }

        imgout->im->array.F[ii] =
            imgin->im->array.F[1 * xsize + ii] -
            imgin->im->array.F[ii];

        imgout->im->array.F[(ysize - 1) * xsize + ii] =
            imgin->im->array.F[(ysize - 1) * xsize + ii] -
            imgin->im->array.F[(ysize - 2) * xsize + ii];
    }

    DEBUG_TRACE_FEXIT();
    return imgout->ID;
}

imageID arith_image_dy(const char *ID_name, const char *IDout_name)
{
    IMGID imgin = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(IDout_name);

    return arith_image_dy_IMGID(&imgin, &imgout);
}
