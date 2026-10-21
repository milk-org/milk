// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mkpolygon.c
 * @brief   Create a regular polygon image (fpsexec)
 *
 * Generate a 2D float image containing a regular
 * N-sided polygon (binary mask) at specified center,
 * radius, number of sides, and rotation angle.
 */

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

#include "milk_runnable.h"
#include "image_gen.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_gen/image_gen.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    outim_name[FUNCTION_PARAMETER_STRMAXLEN] = "impoly";
static int64_t outim_xsize                              = 512;
static int64_t outim_ysize                              = 512;
static double  outim_xcenter                            = 256.0;
static double  outim_ycenter                            = 256.0;
static double  outim_radius                             = 100.0;
static int32_t outim_nsides                             = 8;
static double  outim_rotangle                           = 0.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".out_name", outim_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image name") \
    X(".xsize", &outim_xsize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x size")                  \
    X(".ysize", &outim_ysize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y size")                  \
    X(".xcenter", &outim_xcenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x center")          \
    X(".ycenter", &outim_ycenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y center")          \
    X(".radius", &outim_radius, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius")              \
    X(".nsides", &outim_nsides, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT, "number of sides")       \
    X(".rotangle", &outim_rotangle, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "rotation angle [rad]")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(mkpolygon, FPS_PARAMS);


/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */

static MILK_HOT errno_t mkpolygon_compute(const MILK_RUNNABLE *self)
{
    /* Pre-create output as shared memory stream.
     * make_polygon() calls create_2Dimage_ID()
     * which reuses this existing shared image. */
    IMGID img = stream_connect_create_2D(outim_name, (uint32_t) outim_xsize, (uint32_t) outim_ysize,
                                         _DATATYPE_FLOAT);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_polygon(outim_name, (uint32_t) outim_xsize, (uint32_t) outim_ysize, outim_xcenter,
                 outim_ycenter, outim_radius, outim_nsides, outim_rotangle);

    processinfo_update_output_stream(processinfo, img.im, NULL);

    MILK_PROCINFO_END(self)

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_mkpolygon = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "mkpolygon",
                 .cmdkey      = "mkpolygon",
                 .description = "make regular polygon image",
                 .description_long =
                     "Generate a filled regular polygon on a 2D image with configurable center, "
                     "radius, number of sides, and orientation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mkpolygon),
    .compute = mkpolygon_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkpolygon", runnable_image_gen_mkpolygon);
