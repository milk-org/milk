// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkdist.c
 * @brief   make distance from point image
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    di_n[FUNCTION_PARAMETER_STRMAXLEN] = "imdist";
static int64_t di_xs                              = 512;
static int64_t di_ys                              = 512;
static double  di_cx                              = 256.0;
static double  di_cy                              = 256.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".out_name", di_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")     \
    X(".xsize", &di_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")        \
    X(".ysize", &di_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")        \
    X(".centerx", &di_cx, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "center x") \
    X(".centery", &di_cy, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "center y")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkdist, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkdist_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_dist(di_n, (uint32_t) di_xs, (uint32_t) di_ys, di_cx, di_cy);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkdist = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mkdist",
                 .cmdkey           = "mkdist",
                 .description      = "make distance from point image",
                 .description_long = "Generate a distance map: each pixel value is the "
                                     "Euclidean distance from a specified center point." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkdist),
    .compute = imgen_mkdist_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkdist", runnable_image_gen_imgen_mkdist);
