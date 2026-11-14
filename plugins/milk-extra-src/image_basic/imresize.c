// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imresize.c
 * @brief   Resize 2D image
 *
 * Uses FPS V2 framework.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "imresize.h"
#include "computeunit.h"
#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    imresize_inimname[FUNCTION_PARAMETER_STRMAXLEN];
static char    imresize_outimname[FUNCTION_PARAMETER_STRMAXLEN];
static int64_t imresize_xsize = 64;
static int64_t imresize_ysize = 64;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".in_name", imresize_inimname, FPTYPE_STREAMNAME, 1,                                         \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, "input image")                                 \
    X(".out_name", imresize_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image") \
    X(".xsize", &imresize_xsize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "new x size")              \
    X(".ysize", &imresize_ysize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "new y size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static void imresize_step(IMAGE *imgin, IMAGE *imgout)
{
    (void) arith_image_resize2d_im(imgin, imgout);
}


/* =========================================
 * Public convenience function
 * ========================================= */

long basic_resizeim(const char *imname_in, const char *imname_out, long xsizeout, long ysizeout)
{
    IMGID in = imgid_make_from_name(imname_in);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(imname_out, xsizeout, ysizeout);
    imresize_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im);
    return 0;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(imresize, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t imresize_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(imresize_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(imresize_outimname, imresize_xsize, imresize_ysize);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    imresize_step(in.im, out.im);
    processinfo_update_output_stream(processinfo, out.im, in.im);

    MILK_PROCINFO_END(self)

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_basic_imresize = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "resizeim",
                 .cmdkey           = "resizeim",
                 .description      = "resize 2D image",
                 .description_long = "Resize a 2D image to arbitrary dimensions using bilinear "
                                     "interpolation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imresize),
    .compute = imresize_compute,
};
EXPORT_FPSEXEC(milk, "imgbasic-resizeim", runnable_image_basic_imresize);
