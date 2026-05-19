/**
 * @file    linARPFMupdate.c
 * @brief   Update predictive filter matrix
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "linARfilterPred.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name = "linARPFMupdate",
    .cmdkey   = "linARPFMupdate",
    .description = "update predictive filter matrix",
    .description_long = "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};

static char param_pf_name[FUNCTION_PARAMETER_STRMAXLEN] = "outPF";
static char param_pfm_name[FUNCTION_PARAMETER_STRMAXLEN] = "PFMat";
static double param_alpha = 0.1;

#define FPS_PARAMS(X) \
    X(".pf_name", param_pf_name, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "3D predictor") \
    X(".pfm_name", param_pfm_name, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "2D matrix") \
    X(".alpha", &param_alpha, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "update coeff")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_PF_updatePFmatrix(
        param_pf_name, param_pfm_name, (float)param_alpha);
    return RETURN_SUCCESS;
}

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};
#else
static CLICMDDATA CLIcmddata = {
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};
#endif

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)

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
    return safe_fps_generic_CLIfunction(
        &FPS_app_info,
        farg,
        &CLIcmddata,
        my_bindings,
        nb_bindings,
        compute_function);
}

errno_t CLIADDCMD_linARfilterPred__linARPFMupdate()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif
