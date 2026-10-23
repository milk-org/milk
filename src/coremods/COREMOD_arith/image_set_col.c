// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_set_col.c
 * @brief   Set image column pixels to a value
 *
 * Sets all pixels in a specified column of an image
 * stream to a given value.
 * Uses FPS V2 framework.
 */

#include <stdint.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     setcol_inimname[256] = "";
static float    setcol_pixval        = 0.0f;
static uint32_t setcol_colindex      = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".imname", setcol_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".pixval", &setcol_pixval, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "pixel value")     \
    X(".col", &setcol_colindex, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "column index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMAGE *inimg)
{
    float    val   = setcol_pixval;
    uint32_t col   = setcol_colindex;
    uint32_t xsize = inimg->md[0].size[0];
    uint32_t ysize = inimg->md[0].size[1];

    if (col >= xsize)
    {
        return RETURN_FAILURE;
    }
#define SETCOL_CASE_(DT, ACC, CT)                         \
    case DT:                                              \
        for (uint32_t j = 0; j < ysize; j++)              \
            inimg->array.ACC[j * xsize + col] = (CT) val; \
        break;

    switch (inimg->md[0].datatype)
    {
        FOREACH_REAL_DATATYPE(SETCOL_CASE_) default : PRINT_ERROR("unsupported datatype");
        return RETURN_FAILURE;
    }
#undef SETCOL_CASE_
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_set_col, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_set_col_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(setcol_inimname);
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

const MILK_RUNNABLE runnable_COREMOD_arith_image_set_col = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "setcol",
              .cmdkey           = "setcol",
              .description      = "set image column pixel values",
              .description_long = "Set all pixels in a specified column of a 2D image to a given "
                                  "value. The column index and target value are specified as "
                                  "parameters." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_set_col),
    .compute = image_set_col_compute,
};
EXPORT_FPSEXEC(milk, "arith-setcol", runnable_COREMOD_arith_image_set_col);
