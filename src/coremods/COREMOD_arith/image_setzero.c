// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_setzero.c
 * @brief   Set all image pixels to zero
 *
 * Sets every element of a stream to zero.
 * Uses FPS V2 framework following the POC pattern
 * from examplefunc_fps_cli_poc.c.
 */

#include <stdint.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char imsetzero_imname[FUNCTION_PARAMETER_STRMAXLEN] = "stream";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 *
 * Syntax: X(keyword, ptr, type, is_primary, flag, descr)
 * ============================================================= */

#define IMSETZERO_PARAMS(X) \
    X(".imname", imsetzero_imname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Zero all pixels in the resolved image.
 *
 * Called from within the processinfo loop. The stream
 * is resolved before the loop starts.
 */
static errno_t imsetzero_computation(IMAGE *inimg)
{
    memset(inimg->array.raw, 0,
           ImageStreamIO_typesize(inimg->md[0].datatype) * inimg->md[0].nelement);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_setzero, IMSETZERO_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER (processinfo loop support)
 * ============================================================= */

/**
 * @brief Wraps imsetzero_computation() in the standard
 *        processinfo loop macros.
 *
 * Resolves the stream IMGID before the loop, then
 * calls the computation and updates the output stream
 * on each iteration.
 */
static MILK_HOT errno_t image_setzero_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(imsetzero_imname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    imsetzero_computation(in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_arith_image_setzero = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "imzero",
              .cmdkey           = "imzero",
              .description      = "set all image pixels to zero",
              .description_long = "Zero-fill all pixel values in the target image stream. Operates "
                                  "in-place on the existing shared memory buffer, setting every "
                                  "element to 0." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_setzero),
    .compute = image_setzero_compute,
};
EXPORT_FPSEXEC(milk, "arith-imsetzero", runnable_COREMOD_arith_image_setzero);
