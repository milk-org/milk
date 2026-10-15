// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_make2D.c
 * @brief Image make2d module
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_make2D.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     outimname[FUNCTION_PARAMETER_STRMAXLEN] = "im2D";
static uint32_t imxsize                                 = 256;
static uint32_t imysize                                 = 256;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                 \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image") \
    X(".xsize", &imxsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "x size")           \
    X(".ysize", &imysize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "y size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

imageID make_2Dimage_IMGID(IMGID *img)
{
    imcreateIMGID(img);
    return (img->ID);
}

imageID make_2Dimage(const char *name, uint32_t xsize, uint32_t ysize)
{
    IMGID img = imgid_make_from_name_2D(name, xsize, ysize);
    return make_2Dimage_IMGID(&img);
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_make2D, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_make2D_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID img = imgid_make_from_name_2D(outimname, imxsize, imysize);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    make_2Dimage_IMGID(&img);

    processinfo_update_output_stream(processinfo, img.im, NULL);
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_image_make2D = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "mk2Dim",
             .cmdkey           = "mk2Dim",
             .description      = "make 2D image",
             .description_long = "Create a new 2D image in shared memory with specified dimensions "
                                 "and data type. Initializes all pixel values to zero." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_make2D),
    .compute = image_make2D_compute,
};
