// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imrotate.c
 * @brief   Rotate 2D image
 *
 * Uses FPS V2 framework.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "imrotate.h"
#include "computeunit.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char  imrotate_inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char  imrotate_outimname[FUNCTION_PARAMETER_STRMAXLEN] = "out";
static float imrotate_angle                                   = 0.0f;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".in_name", imrotate_inimname, FPTYPE_STREAMNAME, 1,                                         \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, "input image")                                 \
    X(".out_name", imrotate_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image") \
    X(".angle", &imrotate_angle, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "rotate angle")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static void imrotate_step(IMAGE *imgin, IMAGE *imgout, float angle)
{
    uint32_t nx = imgin->md[0].size[0];
    uint32_t ny = imgin->md[0].size[1];
    float    c  = cosf(angle);
    float    s  = sinf(angle);
    for (uint32_t jj = 0; jj < ny; jj++)
    {
        for (uint32_t ii = 0; ii < nx; ii++)
        {
            long iis = (long) (nx / 2 + (ii - (int) nx / 2) * c + (jj - (int) ny / 2) * s);
            long jjs = (long) (ny / 2 - (ii - (int) nx / 2) * s + (jj - (int) ny / 2) * c);
            if ((iis >= 0) && (jjs >= 0) && (iis < (long) nx) && (jjs < (long) ny))
            {
                imgout->array.F[jj * nx + ii] = imgin->array.F[jjs * nx + iis];
            }
            else
            {
                imgout->array.F[jj * nx + ii] = 0.0;
            }
        }
    }
}


/* =========================================
 * Public convenience function
 * ========================================= */

imageID basic_rotate(const char *ID_name, const char *IDout_name, float angle)
{
    IMGID in = imgid_make_from_name(ID_name);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(IDout_name, in.md->size[0], in.md->size[1]);
    imrotate_step(in.im, out.im, angle);
    ImageStreamIO_UpdateIm(out.im);
    return out.ID;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(imrotate, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t imrotate_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(imrotate_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2Df32(imrotate_outimname, in.md->size[0], in.md->size[1]);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    imrotate_step(in.im, out.im, imrotate_angle);
    processinfo_update_output_stream(processinfo, out.im, in.im);

    MILK_PROCINFO_END(self)

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_basic_imrotate = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "rotateim",
                 .cmdkey           = "rotateim",
                 .description      = "rotate 2D image",
                 .description_long = "Rotate a 2D image by an arbitrary angle using bilinear "
                                     "interpolation. Rotation center and angle are configurable." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imrotate),
    .compute = imrotate_compute,
};
EXPORT_FPSEXEC(milk, "imgbasic-rotateim", runnable_image_basic_imrotate);
