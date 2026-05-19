/**
 * @file    linARapplyRT.c
 * @brief   RT apply predictive filter
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "linARfilterPred.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name = "linARapplyRT",
    .cmdkey   = "linARapplyRT",
    .description = "RT apply predictive filter",
    .description_long = "Linear autoregressive filter prediction engine. Manages filter training, update, and application for real-time prediction."
};

static char param_in_name[FUNCTION_PARAMETER_STRMAXLEN] = "modevalOL";
static int64_t param_offset = 0;
static int64_t param_semtrig = 2;
static char param_pfm_name[FUNCTION_PARAMETER_STRMAXLEN] = "PFmat";
static int64_t param_pforder = 5;
static char param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "outPFmodeval";
static int64_t param_nbgpu = 0;
static int64_t param_loop = 0;
static int64_t param_nbiter = 0;
static int64_t param_savemode = 0;
static double param_tlag = 1.8;
static int64_t param_pfindex = 0;

#define FPS_PARAMS(X) \
    X(".in_name", param_in_name, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "OL coeffs") \
    X(".offset", &param_offset, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "index off") \
    X(".semtrig", &param_semtrig, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "sem trig") \
    X(".pfm_name", param_pfm_name, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "PF matrix") \
    X(".pforder", &param_pforder, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "filter order") \
    X(".out_name", param_out_name, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output") \
    X(".nbgpu", &param_nbgpu, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb GPUs") \
    X(".loop", &param_loop, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "loop flag") \
    X(".nbiter", &param_nbiter, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "nb iter") \
    X(".savemode", &param_savemode, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "save mode") \
    X(".tlag", &param_tlag, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "time lag") \
    X(".pfindex", &param_pfindex, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "PF index")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_PF_RealTimeApply(
        param_in_name, (long)param_offset, (int)param_semtrig,
        param_pfm_name, (long)param_pforder, param_out_name,
        (int)param_nbgpu, (long)param_loop,
        (long)param_nbiter, (int)param_savemode,
        (float)param_tlag, (long)param_pfindex);
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

errno_t CLIADDCMD_linARfilterPred__linARapplyRT()
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
