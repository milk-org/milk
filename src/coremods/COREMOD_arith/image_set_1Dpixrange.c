// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_set_1Dpixrange.c
 * @brief   Set pixels in a 1D index range
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

static char     *setpix1d_inimname = NULL;
static float    *setpix1d_pixval   = NULL;
static uint32_t *setpix1d_minindex = NULL;
static uint32_t *setpix1d_maxindex = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".imname", &setpix1d_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".pixval", &setpix1d_pixval, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "pixel value")      \
    X(".mini", &setpix1d_minindex, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "min index")         \
    X(".maxi", &setpix1d_maxindex, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "max index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMAGE *inimg)
{
    if (!setpix1d_pixval || !setpix1d_minindex || !setpix1d_maxindex)
    {
        return RETURN_FAILURE;
    }
    float    val = *setpix1d_pixval;
    uint32_t mi  = *setpix1d_minindex;
    uint32_t ma  = *setpix1d_maxindex;

    if (ma > inimg->md[0].nelement)
    {
        ma = inimg->md[0].nelement;
    }
    if (mi >= ma)
    {
        return RETURN_FAILURE;
    }
#define SET1D_CASE_(DT, ACC, CT)            \
    case DT:                                \
        for (uint32_t i = mi; i < ma; i++)  \
            inimg->array.ACC[i] = (CT) val; \
        break;

    switch (inimg->md[0].datatype)
    {
        FOREACH_REAL_DATATYPE(SET1D_CASE_) default : PRINT_ERROR("unsupported datatype");
        return RETURN_FAILURE;
    }
#undef SET1D_CASE_
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_set_1Dpixrange, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_set_1Dpixrange_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(setpix1d_inimname);
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

const MILK_RUNNABLE runnable_COREMOD_arith_image_set_1Dpixrange = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "setpix1D",
                 .cmdkey      = "setpix1Drange",
                 .description = "set image pixel value over range",
                 .description_long =
                     "Set a contiguous range of pixels in a 1D image or along the linear memory "
                     "layout of a multi-dimensional image. Specify start index, end index, and the "
                     "value to assign." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_set_1Dpixrange),
    .compute = image_set_1Dpixrange_compute,
};
EXPORT_FPSEXEC(milk, "arith-setpix1Drange", runnable_COREMOD_arith_image_set_1Dpixrange);
