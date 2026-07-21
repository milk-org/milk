// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkgridpix.c
 * @brief   make regular grid
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkgridpix",
    .cmdkey      = "mkgridpix",
    .description = "make regular grid",
    .description_long =
        "Generate a regular pixel grid pattern with configurable spacing and offset."
};

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
FPS_V2_SECTION5(FPS_PARAMS)

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static errno_t compute_function(void)
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_2Dgridpix(gp_n, (uint32_t) gp_xs, (uint32_t) gp_ys, gp_px, gp_py, gp_ox, gp_oy);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */
#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_gen__mkgridpix(void)
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */
#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
