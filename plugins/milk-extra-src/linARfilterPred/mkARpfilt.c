// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    mkARpfilt.c
 * @brief   Make linear AR filter
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "linARfilterPred.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "mkARpfilt",
    .cmdkey           = "mkARpfilt",
    .description      = "make linear AR filter",
    .description_long = "Linear autoregressive filter prediction engine. Manages filter training, "
                        "update, and application for real-time prediction."
};

static char    param_in_name[FUNCTION_PARAMETER_STRMAXLEN]  = "indata";
static int64_t param_pforder                                = 5;
static double  param_pflag                                  = 2.4;
static double  param_svdeps                                 = 0.0001;
static double  param_reglambda                              = 0.0;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "outPF";
static int64_t param_loopmode                               = 0;
static double  param_loopgain                               = 0.1;
static int64_t param_testmode                               = 1;

#define FPS_PARAMS(X)                                                                       \
    X(".in_name", param_in_name, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input data")  \
    X(".pforder", &param_pforder, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "PF order")        \
    X(".pflag", &param_pflag, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "PF lag")            \
    X(".svdeps", &param_svdeps, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "SVD eps")         \
    X(".reglambda", &param_reglambda, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "reg param") \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output PF")     \
    X(".loopmode", &param_loopmode, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "loop mode")     \
    X(".loopgain", &param_loopgain, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "loop gain")   \
    X(".testmode", &param_testmode, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "test mode")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_Build_LinPredictor(
        param_in_name, (long) param_pforder, (float) param_pflag, param_svdeps, param_reglambda,
        param_out_name, 1, (int) param_loopmode, (float) param_loopgain, (int) param_testmode);
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

errno_t CLIADDCMD_linARfilterPred__mkARpfilt()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
