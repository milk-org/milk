/**
 * @file    examplefunc_fps_cli_poc.c
 * @brief   POC for Unified FPS-CLI Architecture
 *
 * Demonstrates the "FPS-as-Primary" architecture where
 * parameters are defined ONCE in a unified MY_PARAMS macro
 * and the generic infrastructure lives in libfps/libfpsCLI.
 *
 * This file contains ONLY module-specific code:
 *   1. FPS identity (FPS_APP_INFO)
 *   2. Local parameter variables
 *   3. Unified parameter table (MY_PARAMS X-macro)
 *   4. Computation logic
 *   5. Module registration (CLIADDCMD)
 *   6. Standalone entry point (FPS_MAIN_STANDALONE_V2)
 */

#include "CLIcore.h"
#include "fps.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "exfpscli",
    .cmdkey      = "fpsclitest",
    .description = "Test FPS-CLI unification"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static int32_t  param_int32   = 123;
static uint32_t param_uint32  = 456;
static int64_t  param_int64   = 789;
static uint64_t param_uint64  = 101112;
static float    param_float32 = 3.14f;
static double   param_float64 = 2.718;
static pid_t    param_pid     = 1000;

static struct timespec param_timespec =
    {1709424000, 123456789};

static char param_filename[FUNCTION_PARAMETER_STRMAXLEN]
    = "data.txt";
static char param_fitsfilename[FUNCTION_PARAMETER_STRMAXLEN]
    = "image.fits";
static char param_execfilename[FUNCTION_PARAMETER_STRMAXLEN]
    = "run_me.sh";
static char param_dirname[FUNCTION_PARAMETER_STRMAXLEN]
    = "/tmp";
static char param_streamname[FUNCTION_PARAMETER_STRMAXLEN]
    = "cam01";
static char param_string[FUNCTION_PARAMETER_STRMAXLEN]
    = "hello";

static int32_t param_onoff = 0;

static char param_processname[FUNCTION_PARAMETER_STRMAXLEN]
    = "process_a";
static char param_fpsname[FUNCTION_PARAMETER_STRMAXLEN]
    = "otherfps";
static char
    param_string_not_stream[FUNCTION_PARAMETER_STRMAXLEN]
    = "not_a_stream";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 *
 * Syntax: X(keyword, ptr, type, is_primary, flag, descr)
 * ============================================================= */

#define MY_PARAMS(X) \
    X(".p_int32",      &param_int32, \
      FPTYPE_INT32,    1, \
      FPFLAG_DEFAULT_INPUT, "Example INT32") \
    X(".p_uint32",     &param_uint32, \
      FPTYPE_UINT32,   0, \
      FPFLAG_DEFAULT_INPUT, "Example UINT32") \
    X(".p_int64",      &param_int64, \
      FPTYPE_INT64,    0, \
      FPFLAG_DEFAULT_INPUT, "Example INT64") \
    X(".p_uint64",     &param_uint64, \
      FPTYPE_UINT64,   0, \
      FPFLAG_DEFAULT_INPUT, "Example UINT64") \
    X(".p_float32",    &param_float32, \
      FPTYPE_FLOAT32,  0, \
      FPFLAG_DEFAULT_INPUT, "Example FLOAT32") \
    X(".p_float64",    &param_float64, \
      FPTYPE_FLOAT64,  0, \
      FPFLAG_DEFAULT_INPUT, "Example FLOAT64") \
    X(".p_onoff",      &param_onoff, \
      FPTYPE_ONOFF,    0, \
      FPFLAG_DEFAULT_INPUT, "Example ONOFF") \
    X(".p_streamname", param_streamname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "Example STREAMNAME") \
    X(".p_fitsfile",   param_fitsfilename, \
      FPTYPE_FITSFILENAME, 0, \
      FPFLAG_DEFAULT_INPUT, "Example FITSFILENAME") \
    X(".p_string",     param_string, \
      FPTYPE_STRING,   0, \
      FPFLAG_DEFAULT_INPUT, "Example STRING") \
    X(".p_pid",        &param_pid, \
      FPTYPE_PID,      0, \
      FPFLAG_DEFAULT_INPUT, "Example PID") \
    X(".p_timespec",   &param_timespec, \
      FPTYPE_TIMESPEC, 0, \
      FPFLAG_DEFAULT_INPUT, "Example TIMESPEC") \
    X(".p_filename",   param_filename, \
      FPTYPE_FILENAME, 0, \
      FPFLAG_DEFAULT_INPUT, "Example FILENAME") \
    X(".p_execfile",   param_execfilename, \
      FPTYPE_EXECFILENAME, 0, \
      FPFLAG_DEFAULT_INPUT, "Example EXECFILENAME") \
    X(".p_dirname",    param_dirname, \
      FPTYPE_DIRNAME,  0, \
      FPFLAG_DEFAULT_INPUT, "Example DIRNAME") \
    X(".p_process",    param_processname, \
      FPTYPE_PROCESS,  0, \
      FPFLAG_DEFAULT_INPUT, "Example PROCESS") \
    X(".p_fpsname",    param_fpsname, \
      FPTYPE_FPSNAME,  0, \
      FPFLAG_DEFAULT_INPUT, "Example FPSNAME") \
    X(".p_strnotstrm", \
      param_string_not_stream, \
      FPTYPE_STRING_NOT_STREAM, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "Example STRING_NOT_STREAM")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief The actual "heavy lifting" function.
 *
 * Uses the local static variables, which are guaranteed
 * to be synced with FPS shared memory by the framework.
 */
static errno_t example_fps_computation()
{
    printf("\n[COMPUTATION] All FPS parameter types:\n");
    printf("  INT32              = %d\n", param_int32);
    printf("  UINT32             = %u\n", param_uint32);
    printf("  INT64              = %ld\n", param_int64);
    printf("  UINT64             = %lu\n", param_uint64);
    printf("  FLOAT32            = %f\n", param_float32);
    printf("  FLOAT64            = %f\n", param_float64);
    printf("  PID                = %d\n",
           (int) param_pid);
    printf("  TIMESPEC           = %ld.%09ld\n",
           param_timespec.tv_sec,
           param_timespec.tv_nsec);
    printf("  FILENAME           = %s\n", param_filename);
    printf("  FITSFILENAME       = %s\n",
           param_fitsfilename);
    printf("  EXECFILENAME       = %s\n",
           param_execfilename);
    printf("  DIRNAME            = %s\n", param_dirname);
    printf("  STREAMNAME         = %s\n",
           param_streamname);
    printf("  STRING             = %s\n", param_string);
    printf("  ONOFF              = %s\n",
           param_onoff ? "ON" : "OFF");
    printf("  PROCESS            = %s\n",
           param_processname);
    printf("  FPSNAME            = %s\n", param_fpsname);
    printf("  STRING_NOT_STREAM  = %s\n",
           param_string_not_stream);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 *
 * Must appear before compute_function() because the
 * INSERT_STD_PROCINFO macros reference CLIcmddata.
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    MY_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    MY_PARAMS(FPS_X_FARG)
};

CLICMDDATA CLIcmddata = {
    "fpsclitest",
    "Test FPS-CLI unification",
    CLICMD_FIELDS_DEFAULTS
};

/*
 * The INSERT_STD_PROCINFO_COMPUTEFUNC_END macro
 * dereferences CLIcmddata.cmdsettings, which is
 * NULL from CLICMD_FIELDS_DEFAULTS.
 * Provide a valid object for standalone mode.
 */
static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/* ================================================================
 * 6.  COMPUTE WRAPPER (processinfo loop support)
 * ============================================================= */

/**
 * @brief Wraps example_fps_computation() in the standard
 *        processinfo loop macros.
 */
static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    example_fps_computation();

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/**
 * @brief Unified milk CLI entry point.
 *
 * Delegates entirely to the generic library function.
 */
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

/**
 * @brief Module registration function.
 */
errno_t CLIADDCMD_milk_module_example__fpscli()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  STANDALONE ENTRY POINT
 *
 * FPS_MAIN_STANDALONE_V2 generates main() using
 * the generic library lifecycle functions.
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    MY_PARAMS,
    compute_function)
#endif
