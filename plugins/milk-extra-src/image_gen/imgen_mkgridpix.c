// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkgridpix.c
 * @brief   make regular grid
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    gp_n[FUNCTION_PARAMETER_STRMAXLEN] = "impgrid";
static int64_t gp_xs                              = 512;
static int64_t gp_ys                              = 512;
static double  gp_px                              = 10.0;
static double  gp_py                              = 10.0;
static double  gp_ox                              = 4.5;
static double  gp_oy                              = 2.8;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".out_name", gp_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")     \
    X(".xsize", &gp_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")        \
    X(".ysize", &gp_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")        \
    X(".pitchx", &gp_px, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x pitch")   \
    X(".pitchy", &gp_py, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y pitch")   \
    X(".offsetx", &gp_ox, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x offset") \
    X(".offsety", &gp_oy, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y offset")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkgridpix, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkgridpix_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_2Dgridpix(gp_n, (uint32_t) gp_xs, (uint32_t) gp_ys, gp_px, gp_py, gp_ox, gp_oy);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkgridpix = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "mkgridpix",
              .cmdkey      = "mkgridpix",
              .description = "make regular grid",
              .description_long =
                  "Generate a regular pixel grid pattern with configurable spacing and offset." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkgridpix),
    .compute = imgen_mkgridpix_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkgridpix", runnable_image_gen_imgen_mkgridpix);
