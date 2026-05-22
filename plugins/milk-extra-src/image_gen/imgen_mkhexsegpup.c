/**
 * @file    imgen_mkhexsegpup.c
 * @brief   make hex seg pupil
 */

#include "image_gen_internal.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "mkhexsegpup",
    .cmdkey           = "mkhexsegpup",
    .description      = "make hex seg pupil",
    .description_long = "Generate a hexagonally-segmented pupil pattern for simulating "
                        "segmented-mirror telescopes (e.g., JWST, ELT)."
};

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    hx_n[FUNCTION_PARAMETER_STRMAXLEN] = "imhex";
static int64_t hx_sz                              = 4096;
static double  hx_r                               = 200.0;
static double  hx_g                               = 2.0;
static double  hx_s                               = 46.3;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                      \
    X(".out_name", hx_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output") \
    X(".size", &hx_sz, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size")      \
    X(".radius", &hx_r, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius") \
    X(".gap", &hx_g, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "gap")       \
    X(".step", &hx_s, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "step")


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

    make_hexsegpupil(hx_n, (uint32_t) hx_sz, hx_r, hx_g, hx_s);

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

errno_t CLIADDCMD_image_gen__mkhexsegpup(void)
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
