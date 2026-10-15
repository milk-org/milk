// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_set_row.c
 * @brief   Set image row pixels to a value
 *
 * Sets all pixels in a specified row of an image
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

static char     setrow_inimname[256] = "";
static float    setrow_pixval        = 0.0f;
static uint32_t setrow_rowindex      = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 *
 * Syntax: X(keyword, ptr, type, is_primary, flag, descr)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".imname", setrow_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".pixval", &setrow_pixval, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "pixel value")     \
    X(".row", &setrow_rowindex, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "row index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Set all pixels in a row to a value.
 *
 * Supports FLOAT and DOUBLE datatypes.
 */
static MILK_HOT errno_t fpsexec(IMAGE *inimg)
{
    float    val   = setrow_pixval;
    uint32_t row   = setrow_rowindex;
    uint32_t xsize = inimg->md[0].size[0];

    if (row >= inimg->md[0].size[1])
    {
        return RETURN_FAILURE;
    }
#define SETROW_CASE_(DT, ACC, CT)                         \
    case DT:                                              \
        for (uint32_t i = 0; i < xsize; i++)              \
            inimg->array.ACC[row * xsize + i] = (CT) val; \
        break;

    switch (inimg->md[0].datatype)
    {
        FOREACH_REAL_DATATYPE(SETROW_CASE_) default : PRINT_ERROR("unsupported datatype");
        return RETURN_FAILURE;
    }
#undef SETROW_CASE_
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_set_row, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER (processinfo loop support)
 * ============================================================= */

static MILK_HOT errno_t image_set_row_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(setrow_inimname);
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

const MILK_RUNNABLE runnable_COREMOD_arith_image_set_row = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "setrow",
                 .cmdkey           = "setrow",
                 .description      = "set image row pixel values",
                 .description_long = "Set all pixels in a specified row of a 2D image to a given "
                                     "value. The row index and target value are specified as "
                                     "parameters." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_set_row),
    .compute = image_set_row_compute,
};
EXPORT_FPSEXEC(milk, "arith-setrow", runnable_COREMOD_arith_image_set_row);
