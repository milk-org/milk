// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CLIcore.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include <assert.h>

// 1.  FPS COMPONENT IDENTITY
static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "fpssynctest",
    .cmdkey           = "fpssynctest",
    .description      = "Test parameter synchronization post-run",
    .description_long = "Test CU -- check modification of parameters at run time"
};

// 2.  LOCAL PARAMETER VARIABLES
static int32_t  p_int32   = 123;
static uint32_t p_uint32  = 456;
static int64_t  p_int64   = 789;
static uint64_t p_uint64  = 101112;
static float    p_float32 = 3.14f;
static double   p_float64 = 2.718;
static pid_t    p_pid     = 1000;

static struct timespec p_timespec = { 1709424000, 123456789 };

static char p_filename[FUNCTION_PARAMETER_STRMAXLEN]     = "data.txt";
static char p_fitsfilename[FUNCTION_PARAMETER_STRMAXLEN] = "image.fits";
static char p_execfilename[FUNCTION_PARAMETER_STRMAXLEN] = "run_me.sh";
static char p_dirname[FUNCTION_PARAMETER_STRMAXLEN]      = "/tmp";
static char p_streamname[FUNCTION_PARAMETER_STRMAXLEN]   = "cam01";
static char p_string[FUNCTION_PARAMETER_STRMAXLEN]       = "hello";

static int32_t p_onoff = 0;

static char p_processname[FUNCTION_PARAMETER_STRMAXLEN]       = "process_a";
static char p_fpsname[FUNCTION_PARAMETER_STRMAXLEN]           = "otherfps";
static char p_string_not_stream[FUNCTION_PARAMETER_STRMAXLEN] = "not_a_stream";


// 3.  UNIFIED PARAMETER TABLE (X-Macro)
// One entry per FPTYPE_*, none primary: this CU is
// driven through FPS param sync, not CLI positional args.
#define FPS_PARAMS(X)                                                                           \
    X(".p_int32", &p_int32, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT, "Example INT32")             \
    X(".p_uint32", &p_uint32, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT, "Example UINT32")         \
    X(".p_int64", &p_int64, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Example INT64")             \
    X(".p_uint64", &p_uint64, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT, "Example UINT64")         \
    X(".p_float32", &p_float32, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT, "Example FLOAT32")     \
    X(".p_float64", &p_float64, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT, "Example FLOAT64")     \
    X(".p_onoff", &p_onoff, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT, "Example ONOFF")             \
    X(".p_pid", &p_pid, FPTYPE_PID, 0, FPFLAG_DEFAULT_INPUT, "Example PID")                     \
    X(".p_timespec", &p_timespec, FPTYPE_TIMESPEC, 0, FPFLAG_DEFAULT_INPUT, "Example TIMESPEC") \
    X(".p_streamname", p_streamname, FPTYPE_STREAMNAME, 0, FPFLAG_DEFAULT_INPUT,                \
      "Example STREAMNAME")                                                                     \
    X(".p_filename", p_filename, FPTYPE_FILENAME, 0, FPFLAG_DEFAULT_INPUT, "Example FILENAME")  \
    X(".p_fitsfile", p_fitsfilename, FPTYPE_FITSFILENAME, 0, FPFLAG_DEFAULT_INPUT,              \
      "Example FITSFILENAME")                                                                   \
    X(".p_execfile", p_execfilename, FPTYPE_EXECFILENAME, 0, FPFLAG_DEFAULT_INPUT,              \
      "Example EXECFILENAME")                                                                   \
    X(".p_dirname", p_dirname, FPTYPE_DIRNAME, 0, FPFLAG_DEFAULT_INPUT, "Example DIRNAME")      \
    X(".p_string", p_string, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "Example STRING")          \
    X(".p_process", p_processname, FPTYPE_PROCESS, 0, FPFLAG_DEFAULT_INPUT, "Example PROCESS")  \
    X(".p_fpsname", p_fpsname, FPTYPE_FPSNAME, 0, FPFLAG_DEFAULT_INPUT, "Example FPSNAME")      \
    X(".p_strnotstrm", p_string_not_stream, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT,  \
      "Example STRING_NOT_STREAM")

// 4.  COMPUTATION LOGIC

// 5.  BINDINGS, FARG, AND CLI DATA
FPS_V2_SECTION5(FPS_PARAMS)

// 6.  COMPUTE WRAPPER
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        // Do nothing.
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// 7.  MILK MODULE REGISTRATION
#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t FPSTEST_CLIADDCMD_FPSSYNCTEST()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif

// 8.  STANDALONE ENTRY POINT
#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
