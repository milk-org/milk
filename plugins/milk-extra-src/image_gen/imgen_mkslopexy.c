/**
 * @file    imgen_mkslopexy.c
 * @brief   make slope image
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkslopexy",
    .cmdkey      = "mkslopexy",
    .description = "make slope image",
    .description_long =
        "Generate a planar slope image with configurable gradients along x and y axes."
};

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    sl_n[FUNCTION_PARAMETER_STRMAXLEN] = "imslope";
static int64_t sl_xs                              = 512;
static int64_t sl_ys                              = 512;
static double  sl_sx                              = 1.2;
static double  sl_sy                              = 1.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                        \
    X(".out_name", sl_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")   \
    X(".xsize", &sl_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")      \
    X(".ysize", &sl_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")      \
    X(".slopex", &sl_sx, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "slope x") \
    X(".slopey", &sl_sy, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "slope y")


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

    make_slopexy(sl_n, (uint32_t) sl_xs, (uint32_t) sl_ys, sl_sx, sl_sy);

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

errno_t CLIADDCMD_image_gen__mkslopexy(void)
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
