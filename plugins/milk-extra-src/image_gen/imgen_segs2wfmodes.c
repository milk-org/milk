/**
 * @file    imgen_segs2wfmodes.c
 * @brief   segments to WF modes
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = { .fps_name    = "segs2wfmodes",
                                     .cmdkey      = "segs2wfmodes",
                                     .description = "segments to WF modes",
                                     .description_long =
                                         "Convert segment piston/tip/tilt commands into wavefront "
                                         "mode images for segmented-mirror telescope simulation." };

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    sw_pfx[FUNCTION_PARAMETER_STRMAXLEN] = "segim";
static int64_t sw_nd                                = 2;
static char    sw_out[FUNCTION_PARAMETER_STRMAXLEN] = "WFmodes";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".prefix", sw_pfx, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "seg prefix") \
    X(".ndigit", &sw_nd, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb digits")   \
    X(".out_name", sw_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")


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

    IMAGE_gen_segments2WFmodes(sw_pfx, (long) sw_nd, sw_out);

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

errno_t CLIADDCMD_image_gen__segs2wfmodes(void)
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
