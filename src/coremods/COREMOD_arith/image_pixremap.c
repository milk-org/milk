// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_pixremap.c
 * @brief Image pixremap module
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"

// input image
static char insname[FUNCTION_PARAMETER_STRMAXLEN];

// mapping array
static char mapsname[FUNCTION_PARAMETER_STRMAXLEN];

// output image
static char    outimname[FUNCTION_PARAMETER_STRMAXLEN];
static int32_t outshared = 0;

#define FPS_PARAMS(X)                                                                      \
    X(".insname", insname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image name") \
    X(".map", mapsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "mapping image name")  \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image name") \
    X(".out_shared", &outshared, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                    \
      "output shared (1) or not (0)")


MILK_PARAMS_DEFINE(image_pixremap, FPS_PARAMS);


static MILK_HOT errno_t image_pixremap_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    // connect to input
    //
    IMGID imgin = imgid_make_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    int64_t insize = imgin.md->size[0] * imgin.md->size[1];
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgmap = imgid_make_from_name(mapsname);
    resolveIMGID(&imgmap, ERRMODE_WARN, dcimg, dcnimg);

    // read map size
    // Note: currently assumes 2D ... to be updated
    //
    uint32_t xsize = imgmap.md->size[0];
    if (imgmap.ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint32_t ysize = imgmap.md->size[1];

    // link/create output image/stream
    // output size is same as input size (default)

    IMGID imgout       = imgid_make_from_name(outimname);
    imgout.mdt->shared = outshared;
    if (outshared == 1)
    {
        imgid_free(&imgout);
        imgout = stream_connect_create_2D(outimname, xsize, ysize, imgin.md->datatype);
    }
    else
    {
        imgout.mdt->naxis    = 2;
        imgout.mdt->size[0]  = xsize;
        imgout.mdt->size[1]  = ysize;
        imgout.mdt->datatype = imgin.md->datatype;
        createimagefromIMGID(&imgout);
    }
    imcreateIMGID(&imgout);

    // build mapping table
    //
    uint64_t nbpix = 0;
    for (uint64_t ii = 0; ii < (uint64_t) xsize * ysize; ii++)
    {
        int64_t pixindex = imgmap.im->array.SI32[ii];
        if ((pixindex > -1) && (pixindex < insize))
        {
            nbpix++;
        }
    }

    printf("mapping table has %lu elements\n", nbpix);

    uint64_t *MILK_RESTRICT map_outpixindex = (uint64_t *) calloc(nbpix, sizeof(uint64_t));
    uint64_t *MILK_RESTRICT map_inpixindex  = (uint64_t *) calloc(nbpix, sizeof(uint64_t));
    if (map_outpixindex == NULL || map_inpixindex == NULL)
    {
        PRINT_ERROR("calloc failed for pixel map arrays");
        free(map_outpixindex);
        free(map_inpixindex);
        return RETURN_FAILURE;
    }

    nbpix = 0;
    for (uint64_t ii = 0; ii < (uint64_t) xsize * ysize; ii++)
    {
        int64_t pixindex = imgmap.im->array.SI32[ii];
        if ((pixindex > -1) && (pixindex < insize))
        {
            map_outpixindex[nbpix] = ii;
            map_inpixindex[nbpix]  = pixindex;
            nbpix++;
        }
    }

    MILK_PROCINFO_INIT(self)

    MILK_PROCINFO_LOOPSTART(self)
    {
#define REMAP_CASE_(DT, ACC, CT)                           \
    case DT:                                               \
        for (uint64_t pixi = 0; pixi < nbpix; pixi++)      \
        {                                                  \
            imgout.im->array.ACC[map_outpixindex[pixi]] =  \
                imgin.im->array.ACC[map_inpixindex[pixi]]; \
        }                                                  \
        break;

        switch (imgin.md->datatype)
        {
            FOREACH_REAL_DATATYPE(REMAP_CASE_) default : PRINT_ERROR("unsupported datatype");
            break;
        }
#undef REMAP_CASE_

        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    MILK_PROCINFO_END(self)
    free(map_outpixindex);
    free(map_inpixindex);
    imgid_free(&imgin);
    imgid_free(&imgmap);
    imgid_free(&imgout);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


const MILK_RUNNABLE runnable_COREMOD_arith_image_pixremap = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "pixremap",
              .cmdkey           = "pixremap",
              .description      = "pixel remapping of image",
              .description_long = "Remap pixels from one image to another using an index map. "
                                  "Each output pixel is assigned the value of the input pixel at "
                                  "the index specified by the mapping array. Supports arbitrary "
                                  "geometric transformations." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_pixremap),
    .compute = image_pixremap_compute,
};
EXPORT_FPSEXEC(milk, "arith-pixremap", runnable_COREMOD_arith_image_pixremap);
