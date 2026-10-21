// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkrndim.c
 * @brief   make random image
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    ri_n[FUNCTION_PARAMETER_STRMAXLEN] = "imrnd";
static int64_t ri_xs                              = 512;
static int64_t ri_ys                              = 512;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                      \
    X(".out_name", ri_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &ri_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")    \
    X(".ysize", &ri_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkrndim, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkrndim_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_rnd(ri_n, (uint32_t) ri_xs, (uint32_t) ri_ys, "");

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkrndim = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mkrndim",
              .cmdkey           = "mkrndim",
              .description      = "make random image",
              .description_long = "Generate a random image with uniformly-distributed pixel values "
                                  "in a configurable range." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkrndim),
    .compute = imgen_mkrndim_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkrndim", runnable_image_gen_imgen_mkrndim);
