// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cubecollapse.c
 * @brief   Collapse a cube along z axis
 *
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "cubecollapse.h"
#include "computeunit.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char cubecollapse_inimname[FUNCTION_PARAMETER_STRMAXLEN];
static char cubecollapse_outimname[FUNCTION_PARAMETER_STRMAXLEN];


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".in_name", cubecollapse_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,   \
      "input cube image")                                                              \
    X(".out_name", cubecollapse_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, \
      "output 2D image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static void cube_collapse_step(IMAGE *imgin, IMAGE *imgout)
{
    arith_image_reduce_axis_im(imgin, 2, ARITH_RED_SUM, imgout);
}


/* =========================================
 * Public convenience function
 * ========================================= */

imageID cube_collapse(const char *ID_in_name, const char *ID_out_name)
{
    IMGID in = imgid_make_from_name(ID_in_name);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(ID_out_name, in.md->size[0], in.md->size[1]);
    cube_collapse_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im);
    return out.ID;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(cubecollapse, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t cubecollapse_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(cubecollapse_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(cubecollapse_outimname, in.md->size[0], in.md->size[1]);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    cube_collapse_step(in.im, out.im);
    processinfo_update_output_stream(processinfo, out.im, in.im);

    MILK_PROCINFO_END(self)

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_basic_cubecollapse = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "cubecollapse",
                 .cmdkey           = "cubecollapse",
                 .description      = "collapse a cube along z",
                 .description_long = "Collapse a 3D image cube along the z-axis by computing the "
                                     "mean, median, or sum of all slices. Produces a single 2D "
                                     "output image." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cubecollapse),
    .compute = cubecollapse_compute,
};
EXPORT_FPSEXEC(milk, "imgbasic-cubecollapse", runnable_image_basic_cubecollapse);
