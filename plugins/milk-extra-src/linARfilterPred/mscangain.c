/**
 * @file    mscangain.c
 * @brief   Scan gain
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "linARfilterPred.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "mscangain",
    .cmdkey           = "mscangain",
    .description      = "scan gain",
    .description_long = "Linear autoregressive filter prediction engine. Manages filter training, "
                        "update, and application for real-time prediction."
};

static char   param_in_name[FUNCTION_PARAMETER_STRMAXLEN] = "olwfsmeas";
static double param_multfact                              = 0.98;
static double param_framelag                              = 2.65;

#define FPS_PARAMS(X)                                                                       \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "mode vals")   \
    X(".multfact", &param_multfact, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "mult factor") \
    X(".framelag", &param_framelag, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "frame lag")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_ScanGain(param_in_name, (float) param_multfact, (float) param_framelag);
    return RETURN_SUCCESS;
}

FPS_V2_SECTION5(FPS_PARAMS)
static MILK_HOT errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    fpsexec();
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linARfilterPred__mscangain()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
