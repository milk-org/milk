/**
 * @file    imrepshiftx.c
 * @brief   Repeat and shift image along X
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "linARfilterPred.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imrepshiftx",
    .cmdkey           = "imrepshiftx",
    .description      = "repeat and shift image along X",
    .description_long = "Linear autoregressive filter prediction engine. Manages filter training, "
                        "update, and application for real-time prediction."
};

static char    param_in_name[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static int64_t param_nbstep                                 = 5;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "imout";

#define FPS_PARAMS(X)                                                                 \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input") \
    X(".nbstep", &param_nbstep, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb steps")    \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

static MILK_HOT errno_t fpsexec()
{
    linARfilterPred_repeat_shift_X(param_in_name, (long) param_nbstep, param_out_name);
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

errno_t CLIADDCMD_linARfilterPred__imrepshiftx()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
