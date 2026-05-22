/**
 * @file    imgen_mkgauss.c
 * @brief   make gaussian spot image
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "mkgauss",
    .cmdkey           = "mkgauss",
    .description      = "make gaussian spot image",
    .description_long = "Generate a 2D Gaussian intensity pattern with configurable center, width "
                        "(sigma), amplitude, and ellipticity."
};

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
FPS_V2_SECTION5(FPS_PARAMS)

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static errno_t compute_function(void)
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_gauss(gs_n, (uint32_t) gs_xs, (uint32_t) gs_ys, gs_a, gs_A);

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

errno_t CLIADDCMD_image_gen__mkgauss(void)
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
