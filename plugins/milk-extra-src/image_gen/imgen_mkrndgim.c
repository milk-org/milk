/**
 * @file    imgen_mkrndgim.c
 * @brief   make random gaussian image
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = { .fps_name    = "mkrndgim",
                                     .cmdkey      = "mkrndgim",
                                     .description = "make random gaussian image",
                                     .description_long =
                                         "Generate a random image with Gaussian-distributed pixel "
                                         "values. Mean and standard deviation are configurable." };

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
FPS_V2_SECTION5(FPS_PARAMS)

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static errno_t compute_function(void)
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_rnd(rg_n, (uint32_t) rg_xs, (uint32_t) rg_ys, "gauss");

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

errno_t CLIADDCMD_image_gen__mkrndgim(void)
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
