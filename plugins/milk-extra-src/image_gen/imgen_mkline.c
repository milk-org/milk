/**
 * @file    imgen_mkline.c
 * @brief   make line
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkline",
    .cmdkey      = "mkline",
    .description = "make line",
    .description_long =
        "Generate a line pattern on a 2D image with configurable endpoints, width, and intensity."
};

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char ln_n[FUNCTION_PARAMETER_STRMAXLEN]
    = "imline";
static int64_t ln_xs = 512;
static int64_t ln_ys = 512;
static double ln_x1 = 256.0;
static double ln_y1 = 256.0;
static double ln_x2 = 100.0;
static double ln_y2 = 200.0;
static double ln_t = 3.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X) \
    X(".out_name", ln_n, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".xsize", &ln_xs, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "xsize") \
    X(".ysize", &ln_ys, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "ysize") \
    X(".x1", &ln_x1, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x1") \
    X(".y1", &ln_y1, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y1") \
    X(".x2", &ln_x2, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "x2") \
    X(".y2", &ln_y2, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "y2") \
    X(".thickness", &ln_t, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "thickness")


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

    make_line(ln_n,
        (uint32_t)ln_xs, (uint32_t)ln_ys,
        ln_x1, ln_y1, ln_x2, ln_y2, ln_t);

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

errno_t CLIADDCMD_image_gen__mkline(void)
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
