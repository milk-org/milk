/**
 * @file    imgen_mkrect.c
 * @brief   make rectangle
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkrect",
    .cmdkey      = "mkrect",
    .description = "make rectangle",
    .description_long =
        "Generate a filled rectangle on a 2D image with configurable position, size, and value."
};

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    rc_n[FUNCTION_PARAMETER_STRMAXLEN] = "imrect";
static int64_t rc_xs                              = 512;
static int64_t rc_ys                              = 512;
static double  rc_xc                              = 256.0;
static double  rc_yc                              = 256.0;
static double  rc_r1                              = 100.0;
static double  rc_r2                              = 200.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".out_name", rc_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")     \
    X(".xsize", &rc_xs, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "xsize")        \
    X(".ysize", &rc_ys, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "ysize")        \
    X(".xcenter", &rc_xc, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &rc_yc, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y center") \
    X(".radius1", &rc_r1, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius 1") \
    X(".radius2", &rc_r2, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius 2")


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

    make_rectangle(rc_n, (uint32_t) rc_xs, (uint32_t) rc_ys, rc_xc, rc_yc, rc_r1, rc_r2);

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

errno_t CLIADDCMD_image_gen__mkrect(void)
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
