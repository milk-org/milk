// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mklincoord.c
 * @brief   make linear coordinate
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    lc_n[FUNCTION_PARAMETER_STRMAXLEN] = "imlincoord";
static int64_t lc_xs                              = 512;
static int64_t lc_ys                              = 512;
static double  lc_xc                              = 256.0;
static double  lc_yc                              = 256.0;
static double  lc_a                               = 1.42;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".out_name", lc_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")     \
    X(".xsize", &lc_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")        \
    X(".ysize", &lc_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")        \
    X(".xcenter", &lc_xc, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &lc_yc, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y center") \
    X(".angle", &lc_a, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "angle")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mklincoord, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mklincoord_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_lincoordinate(lc_n, (uint32_t) lc_xs, (uint32_t) lc_ys, lc_xc, lc_yc, lc_a);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mklincoord = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "mklincoord",
                 .cmdkey      = "mklincoord",
                 .description = "make linear coordinate",
                 .description_long =
                     "Generate a linear coordinate ramp image along a specified axis." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mklincoord),
    .compute = imgen_mklincoord_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mklincoord", runnable_image_gen_imgen_mklincoord);
