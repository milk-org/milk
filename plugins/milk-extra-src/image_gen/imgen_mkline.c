// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkline.c
 * @brief   make line
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    ln_n[FUNCTION_PARAMETER_STRMAXLEN] = "imline";
static int64_t ln_xs                              = 512;
static int64_t ln_ys                              = 512;
static double  ln_x1                              = 256.0;
static double  ln_y1                              = 256.0;
static double  ln_x2                              = 100.0;
static double  ln_y2                              = 200.0;
static double  ln_t                               = 3.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                      \
    X(".out_name", ln_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &ln_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")    \
    X(".ysize", &ln_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")    \
    X(".x1", &ln_x1, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x1")        \
    X(".y1", &ln_y1, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y1")        \
    X(".x2", &ln_x2, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x2")        \
    X(".y2", &ln_y2, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y2")        \
    X(".thickness", &ln_t, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "thickness")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkline, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkline_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_line(ln_n, (uint32_t) ln_xs, (uint32_t) ln_ys, ln_x1, ln_y1, ln_x2, ln_y2, ln_t);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkline = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mkline",
                 .cmdkey           = "mkline",
                 .description      = "make line",
                 .description_long = "Generate a line pattern on a 2D image with configurable "
                                     "endpoints, width, and intensity." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkline),
    .compute = imgen_mkline_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkline", runnable_image_gen_imgen_mkline);
