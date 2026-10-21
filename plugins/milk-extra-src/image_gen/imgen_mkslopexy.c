// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkslopexy.c
 * @brief   make slope image
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    sl_n[FUNCTION_PARAMETER_STRMAXLEN] = "imslope";
static int64_t sl_xs                              = 512;
static int64_t sl_ys                              = 512;
static double  sl_sx                              = 1.2;
static double  sl_sy                              = 1.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                        \
    X(".out_name", sl_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")   \
    X(".xsize", &sl_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")      \
    X(".ysize", &sl_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")      \
    X(".slopex", &sl_sx, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "slope x") \
    X(".slopey", &sl_sy, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "slope y")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkslopexy, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkslopexy_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_slopexy(sl_n, (uint32_t) sl_xs, (uint32_t) sl_ys, sl_sx, sl_sy);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkslopexy = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "mkslopexy",
              .cmdkey      = "mkslopexy",
              .description = "make slope image",
              .description_long =
                  "Generate a planar slope image with configurable gradients along x and y axes." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkslopexy),
    .compute = imgen_mkslopexy_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkslopexy", runnable_image_gen_imgen_mkslopexy);
