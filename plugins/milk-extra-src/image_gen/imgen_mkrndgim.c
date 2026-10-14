// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkrndgim.c
 * @brief   make random gaussian image
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    rg_n[FUNCTION_PARAMETER_STRMAXLEN] = "imrndg";
static int64_t rg_xs                              = 512;
static int64_t rg_ys                              = 512;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                      \
    X(".out_name", rg_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &rg_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")    \
    X(".ysize", &rg_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkrndgim, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkrndgim_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_rnd(rg_n, (uint32_t) rg_xs, (uint32_t) rg_ys, "gauss");

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkrndgim = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mkrndgim",
                 .cmdkey           = "mkrndgim",
                 .description      = "make random gaussian image",
                 .description_long = "Generate a random image with Gaussian-distributed pixel "
                                     "values. Mean and standard deviation are configurable." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkrndgim),
    .compute = imgen_mkrndgim_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkrndgim", runnable_image_gen_imgen_mkrndgim);
