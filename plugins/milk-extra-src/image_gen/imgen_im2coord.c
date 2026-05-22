/**
 * @file    imgen_im2coord.c
 * @brief   make coordinate image
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "im2coord",
    .cmdkey      = "im2coord",
    .description = "make coordinate image",
    .description_long =
        "Create a coordinate image where each pixel value represents its (x, y) position. Useful "
        "for generating spatial masks and geometric transformations."
};

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
FPS_V2_SECTION5(FPS_PARAMS)

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static errno_t compute_function(void)
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_gen_im2coord(ic_in, (uint8_t) ic_ax, ic_out);

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

errno_t CLIADDCMD_image_gen__im2coord(void)
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
