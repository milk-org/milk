// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_crop2D.c
 * @brief   Crop a 2D rectangular region from stream
 *
 * Uses FPS V2 framework.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     cropinsname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     outsname[FUNCTION_PARAMETER_STRMAXLEN]    = "out";
static uint32_t cropxstart                                = 0;
static uint32_t cropxsize                                 = 100;
static uint32_t cropystart                                = 0;
static uint32_t cropysize                                 = 100;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".insname", cropinsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, \
      "Input stream name")                                                                         \
    X(".outsname", outsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "Output stream name")     \
    X(".cropxstart", &cropxstart, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop x coord start")    \
    X(".cropxsize", &cropxsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop x coord size")       \
    X(".cropystart", &cropystart, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop y coord start")    \
    X(".cropysize", &cropysize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop y coord size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMAGE *input_image, IMAGE *output_image)
{
    uint32_t xs = cropxstart;
    uint32_t xw = cropxsize;
    uint32_t ys = cropystart;
    uint32_t yw = cropysize;
    uint32_t iw = input_image->md[0].size[0];
    uint32_t ih = input_image->md[0].size[1];
    size_t   ts = ImageStreamIO_typesize(input_image->md[0].datatype);

    for (uint32_t j = 0; j < yw; j++)
    {
        uint32_t oj = j + ys;
        if (oj >= ih)
        {
            continue;
        }
        __builtin_memcpy(((char *) output_image->array.raw) + j * xw * ts,
                         ((char *) input_image->array.raw) + (oj * iw + xs) * ts, xw * ts);
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Validate crop parameters against
 *        input stream dimensions.
 */
static errno_t crop2D_validate()
{
    IMAGE im;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(cropinsname, &im) == 0)
    {
        uint32_t w = im.md[0].size[0];
        uint32_t h = im.md[0].size[1];
        if (cropxstart + cropxsize > w)
        {
            if (cropxstart >= w)
            {
                cropxstart = 0;
            }
            if (cropxstart + cropxsize > w)
            {
                cropxsize = w - cropxstart;
            }
        }
        if (cropystart + cropysize > h)
        {
            if (cropystart >= h)
            {
                cropystart = 0;
            }
            if (cropystart + cropysize > h)
            {
                cropysize = h - cropystart;
            }
        }
        ImageStreamIO_closeIm(&im);
    }
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_crop2D, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_crop2D_compute(const MILK_RUNNABLE *self)
{
    IMGID iin = imgid_make_from_name(cropinsname);
    resolveIMGID(&iin, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID iout = stream_connect_create_2D(outsname, cropxsize, cropysize, iin.md->datatype);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec(iin.im, iout.im);
    processinfo_update_output_stream(processinfo, iout.im, iin.im);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_arith_image_crop2D = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "crop2D",
                 .cmdkey           = "crop2D",
                 .description      = "crop 2D image",
                 .description_long = "Extract a rectangular sub-region from a 2D image stream. "
                                     "Specify origin coordinates (x0, y0) and output dimensions "
                                     "(xsize, ysize). The cropped output is written to a shared "
                                     "memory stream." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_crop2D),
    .compute   = image_crop2D_compute,
    .confcheck = crop2D_validate,
};
EXPORT_FPSEXEC(milk, "arith-crop2D", runnable_COREMOD_arith_image_crop2D);
