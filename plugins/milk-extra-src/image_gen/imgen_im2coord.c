// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_im2coord.c
 * @brief   make coordinate image
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    ic_in[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static int64_t ic_ax                                = 1;
static char    ic_out[FUNCTION_PARAMETER_STRMAXLEN] = "imy";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                         \
    X(".in_name", ic_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input") \
    X(".axis", &ic_ax, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "axis")         \
    X(".out_name", ic_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_im2coord, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_im2coord_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    image_gen_im2coord(ic_in, (uint8_t) ic_ax, ic_out);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_im2coord = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "im2coord",
              .cmdkey           = "im2coord",
              .description      = "make coordinate image",
              .description_long = "Create a coordinate image where each pixel value represents its "
                                  "(x, y) position. Useful "
                                  "for generating spatial masks and geometric transformations." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_im2coord),
    .compute = imgen_im2coord_compute,
};
EXPORT_FPSEXEC(milk, "imggen-im2coord", runnable_image_gen_imgen_im2coord);
