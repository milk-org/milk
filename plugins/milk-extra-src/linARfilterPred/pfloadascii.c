/**
 * @file    pfloadascii.c
 * @brief   Load ascii files to PF input
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "linARfilterPred.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "pfloadascii",
    .cmdkey           = "pfloadascii",
    .description      = "load ascii files to PF input",
    .description_long = "Linear autoregressive filter prediction engine. Manages filter training, "
                        "update, and application for real-time prediction."
};

static double  param_tstart                                 = 200.0;
static double  param_dt                                     = 0.001;
static int64_t param_nbpt                                   = 10000;
static int64_t param_nbfr                                   = 4;
static char    param_out_name[FUNCTION_PARAMETER_STRMAXLEN] = "pfin";

#define FPS_PARAMS(X)                                                              \
    X(".tstart", &param_tstart, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "tstart") \
    X(".dt", &param_dt, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "dt")             \
    X(".nbpt", &param_nbpt, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb points")    \
    X(".nbfr", &param_nbfr, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb frames")    \
    X(".out_name", param_out_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

static MILK_HOT errno_t fpsexec()
{
    LINARFILTERPRED_LoadASCIIfiles(param_tstart, param_dt, (long) param_nbpt, (long) param_nbfr,
                                   param_out_name);
    return RETURN_SUCCESS;
}

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int nb_bindings = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
#else
static CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
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
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linARfilterPred__pfloadascii()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
