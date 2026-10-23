// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkgauss.c
 * @brief   make gaussian spot image
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    gs_n[FUNCTION_PARAMETER_STRMAXLEN] = "imgauss";
static int64_t gs_xs                              = 512;
static int64_t gs_ys                              = 512;
static double  gs_a                               = 12.0;
static double  gs_A                               = 1.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                      \
    X(".out_name", gs_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &gs_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")    \
    X(".ysize", &gs_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")    \
    X(".a", &gs_a, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "width param") \
    X(".amp", &gs_A, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "amplitude")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkgauss, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkgauss_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_gauss(gs_n, (uint32_t) gs_xs, (uint32_t) gs_ys, gs_a, gs_A);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkgauss = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "mkgauss",
                 .cmdkey      = "mkgauss",
                 .description = "make gaussian spot image",
                 .description_long =
                     "Generate a 2D Gaussian intensity pattern with configurable center, width "
                     "(sigma), amplitude, and ellipticity." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkgauss),
    .compute = imgen_mkgauss_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkgauss", runnable_image_gen_imgen_mkgauss);
