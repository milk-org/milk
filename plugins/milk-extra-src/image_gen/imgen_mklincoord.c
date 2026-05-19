/**
 * @file    imgen_mklincoord.c
 * @brief   make linear coordinate
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mklincoord",
    .cmdkey      = "mklincoord",
    .description = "make linear coordinate",
    .description_long =
        "Generate a linear coordinate ramp image along a specified axis."
};

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char lc_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imlincoord";
static int64_t lc_xs = 512;
static int64_t lc_ys = 512;
static double lc_xc = 256.0;
static double lc_yc = 256.0;
static double lc_a = 1.42;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X) \
    X(".out_name", lc_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &lc_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &lc_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".xcenter", &lc_xc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &lc_yc, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y center") \
    X(".angle", &lc_a, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "angle")


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

    make_lincoordinate(lc_n,
        (uint32_t)lc_xs, (uint32_t)lc_ys,
        lc_xc, lc_yc, lc_a);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */
#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t CLIADDCMD_image_gen__mklincoord(void)
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */
#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif
