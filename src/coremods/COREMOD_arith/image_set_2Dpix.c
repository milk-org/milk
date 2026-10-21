// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_set_2Dpix.c
 * @brief   Set a single pixel value in a 2D image
 *
 * Uses FPS V2 framework.
 */

#include <stdint.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *setpix_inimname = NULL;
static float    *setpix_pixval   = NULL;
static uint32_t *setpix_colindex = NULL;
static uint32_t *setpix_rowindex = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                         \
    X(".imname", &setpix_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".pixval", &setpix_pixval, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "pixel value")      \
    X(".col", &setpix_colindex, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "col index")          \
    X(".row", &setpix_rowindex, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "row index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMAGE *inimg)
{
    if (!setpix_pixval || !setpix_colindex || !setpix_rowindex)
    {
        return RETURN_FAILURE;
    }
    float    val   = *setpix_pixval;
    uint32_t col   = *setpix_colindex;
    uint32_t row   = *setpix_rowindex;
    uint32_t xsize = inimg->md[0].size[0];

    if (col >= xsize || row >= inimg->md[0].size[1])
    {
        return RETURN_FAILURE;
    }
#define SET2D_CASE_(DT, ACC, CT)                        \
    case DT:                                            \
        inimg->array.ACC[row * xsize + col] = (CT) val; \
        break;

    switch (inimg->md[0].datatype)
    {
        FOREACH_REAL_DATATYPE(SET2D_CASE_) default : PRINT_ERROR("unsupported datatype");
        return RETURN_FAILURE;
    }
#undef SET2D_CASE_
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_set_2Dpix, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_set_2Dpix_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(setpix_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec(in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_arith_image_set_2Dpix = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "setpix",
                 .cmdkey      = "setpix",
                 .description = "set image pixel value",
                 .description_long =
                     "Set the value of a single pixel at coordinates (x, y) in a 2D image stream. "
                     "Useful for injecting test signals or modifying individual pixel values in "
                     "shared memory." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_set_2Dpix),
    .compute = image_set_2Dpix_compute,
};
EXPORT_FPSEXEC(milk, "arith-setpix", runnable_COREMOD_arith_image_set_2Dpix);
