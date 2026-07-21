// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkdist.c
 * @brief   make distance from point image
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = { .fps_name    = "mkdist",
                                     .cmdkey      = "mkdist",
                                     .description = "make distance from point image",
                                     .description_long =
                                         "Generate a distance map: each pixel value is the "
                                         "Euclidean distance from a specified center point." };

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    di_n[FUNCTION_PARAMETER_STRMAXLEN] = "imdist";
static int64_t di_xs                              = 512;
static int64_t di_ys                              = 512;
static double  di_cx                              = 256.0;
static double  di_cy                              = 256.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".out_name", di_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")     \
    X(".xsize", &di_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")        \
    X(".ysize", &di_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")        \
    X(".centerx", &di_cx, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "center x") \
    X(".centery", &di_cy, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "center y")


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

    make_dist(di_n, (uint32_t) di_xs, (uint32_t) di_ys, di_cx, di_cy);

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

errno_t CLIADDCMD_image_gen__mkdist(void)
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
