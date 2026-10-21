// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkrect.c
 * @brief   make rectangle
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    rc_n[FUNCTION_PARAMETER_STRMAXLEN] = "imrect";
static int64_t rc_xs                              = 512;
static int64_t rc_ys                              = 512;
static double  rc_xc                              = 256.0;
static double  rc_yc                              = 256.0;
static double  rc_r1                              = 100.0;
static double  rc_r2                              = 200.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".out_name", rc_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")     \
    X(".xsize", &rc_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")        \
    X(".ysize", &rc_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")        \
    X(".xcenter", &rc_xc, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &rc_yc, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y center") \
    X(".radius1", &rc_r1, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius 1") \
    X(".radius2", &rc_r2, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius 2")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkrect, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkrect_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_rectangle(rc_n, (uint32_t) rc_xs, (uint32_t) rc_ys, rc_xc, rc_yc, rc_r1, rc_r2);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkrect = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mkrect",
                 .cmdkey           = "mkrect",
                 .description      = "make rectangle",
                 .description_long = "Generate a filled rectangle on a 2D image with configurable "
                                     "position, size, and value." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkrect),
    .compute = imgen_mkrect_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkrect", runnable_image_gen_imgen_mkrect);
