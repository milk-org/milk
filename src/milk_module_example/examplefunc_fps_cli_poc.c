// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    examplefunc_fps_cli_poc.c
 * @brief   Template for FPS V2 Compute Units
 *
 * This file is a TEMPLATE for writing compute units
 * (functions that can run as standalone executables
 * or as milk CLI commands) using the unified FPS-CLI
 * V2 architecture.
 *
 * ARCHITECTURE OVERVIEW
 * ---------------------
 * One MILK_RUNNABLE unit serves two entry points:
 *   - the milk CLI command, when the module registry
 *     (milk_module_example.c) lists the unit;
 *   - the standalone executable milk-fpsexec-clitest,
 *     generated from EXPORT_FPSEXEC (section 7).
 *
 * Parameters are defined ONCE in an X-macro
 * (FPS_PARAMS) and the framework automatically:
 *   - Creates FPS shared-memory entries
 *   - Generates CLI argument definitions (farg[])
 *   - Generates FPS-to-local sync bindings
 *   - Produces help text for standalone executables
 *
 * HOW TO CREATE A NEW COMPUTE UNIT
 * ---------------------------------
 * 1. Copy this file and rename it.
 * 2. Update the descriptor (section 7) with the new
 *    FPS name, CLI key, and description.
 * 3. Replace the local variables (section 2) with
 *    your parameters.
 * 4. Replace FPS_PARAMS (section 3) with entries
 *    matching your parameters.
 * 5. Replace fpsexec() (section 4) with your
 *    computation logic.
 * 6. Rename the unit in section 7 and list it in the
 *    module registry (milk_module_example.c).
 *
 * FILE STRUCTURE
 * --------------
 *   1. Identity         - descriptor fields (section 7)
 *   2. Local variables  - C variables for parameters
 *   3. FPS_PARAMS       - X-macro parameter table
 *   4. fpsexec()        - computation logic
 *   5. Bindings         - MILK_PARAMS_DEFINE
 *   6. fpscli_compute   - processinfo loop wrapper
 *   7. MILK_RUNNABLE    - descriptor and fpsexec export
 */

/* ================================================================
 * INCLUDES
 * milk_runnable.h pulls in the FPS and processinfo macros.
 * ============================================================= */

#include "milk_runnable.h"
#include "fps_cli_poc.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 *
 * The identity (fps_name, cmdkey, description) is set
 * once, in the MILK_RUNNABLE descriptor (section 7).
 * The FPS shm name and help text are derived from it.
 *
 * Fields:
 *   .fps_name    FPS shared-memory name (no spaces).
 *                This becomes the name in
 *                /dev/shm/<fps_name>.fps.shm
 *   .cmdkey      CLI command keyword.  Users type
 *                this at the milk prompt.
 *   .description One-line human-readable summary
 *                shown in help and fps-info output.
 * ============================================================= */


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 *
 * These static variables hold the current parameter
 * values in the running process.  The framework
 * automatically syncs them with the FPS shared memory
 * via the bindings defined in section 5.
 *
 * IMPORTANT NOTES ON VARIABLE TYPES:
 *
 * - Scalar types (int32, float, etc.):
 *   Declare as the matching C type.  In FPS_PARAMS,
 *   pass the ADDRESS: &param_variable
 *
 * - String types (STREAMNAME, FILENAME, STRING, etc.)
 *   The sync function (sync_fps_to_local) stores a
 *   POINTER to the FPS shared-memory string buffer.
 *   Therefore:
 *     * If the variable will be read AFTER FPS sync
 *       (standalone/FPS mode), declare as char* and
 *       pass &ptr in FPS_PARAMS.
 *     * If the variable needs a default value for
 *       non-FPS mode, use a char[] buffer.  In
 *       FPS_PARAMS, pass the buffer name directly
 *       (it decays to char*, but the sync overwrites
 *       the first sizeof(char*) bytes with a pointer).
 *
 * - ONOFF type: uses int32_t (0 = OFF, nonzero = ON)
 *
 * - TIMESPEC type: uses struct timespec
 *
 * - PID type: uses pid_t
 *
 * The default values assigned here are used:
 *   - As initial FPS values when creating a new FPS
 *   - As CLI defaults when no argument is provided
 * ============================================================= */

static int32_t  param_int32   = 123;
static uint32_t param_uint32  = 456;
static int64_t  param_int64   = 789;
static uint64_t param_uint64  = 101112;
static float    param_float32 = 3.14f;
static double   param_float64 = 2.718;
static pid_t    param_pid     = 1000;

static struct timespec param_timespec = { 1709424000, 123456789 };

static char param_filename[FUNCTION_PARAMETER_STRMAXLEN]     = "data.txt";
static char param_fitsfilename[FUNCTION_PARAMETER_STRMAXLEN] = "image.fits";
static char param_execfilename[FUNCTION_PARAMETER_STRMAXLEN] = "run_me.sh";
static char param_dirname[FUNCTION_PARAMETER_STRMAXLEN]      = "/tmp";
static char param_streamname[FUNCTION_PARAMETER_STRMAXLEN]   = "cam01";
static char param_string[FUNCTION_PARAMETER_STRMAXLEN]       = "hello";

static int32_t param_onoff = 0;

static char param_processname[FUNCTION_PARAMETER_STRMAXLEN]       = "process_a";
static char param_fpsname[FUNCTION_PARAMETER_STRMAXLEN]           = "otherfps";
static char param_string_not_stream[FUNCTION_PARAMETER_STRMAXLEN] = "not_a_stream";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 *
 * The FPS_PARAMS macro defines ALL parameters in one
 * place using the X-macro pattern.  It is expanded
 * multiple times by different "X" functions to
 * generate:
 *
 *   FPS_X_BINDING  -> FPS_CLI_BINDING array
 *                     (maps FPS keywords to C vars)
 *   FPS_X_FARG     -> CLICMDARGDEF array
 *                     (CLI argument definitions)
 *
 * The standalone --help text is generated from the bindings by fpsexec_main_function().
 *
 * COLUMN REFERENCE:
 *   X(keyword, ptr, type, is_primary, flag, descr)
 *
 *   keyword     FPS parameter keyword, prefixed with
 *               ".".  Full key = fps_name + keyword,
 *               e.g. "exfpscli.p_int32"
 *
 *   ptr         Pointer to the local C variable.
 *               - Scalars: use &variable
 *               - Strings: use buffer_name (decays
 *                 to char*) or &char_ptr_variable
 *
 *   type        FPS parameter type enum:
 *               FPTYPE_INT32, FPTYPE_UINT32,
 *               FPTYPE_INT64, FPTYPE_UINT64,
 *               FPTYPE_FLOAT32, FPTYPE_FLOAT64,
 *               FPTYPE_ONOFF, FPTYPE_PID,
 *               FPTYPE_TIMESPEC, FPTYPE_STREAMNAME,
 *               FPTYPE_FILENAME, FPTYPE_FITSFILENAME,
 *               FPTYPE_EXECFILENAME, FPTYPE_DIRNAME,
 *               FPTYPE_STRING, FPTYPE_PROCESS,
 *               FPTYPE_FPSNAME,
 *               FPTYPE_STRING_NOT_STREAM
 *
 *               For FPTYPE_STREAMNAME, the value
 *               string may carry an @X: prefix to
 *               control load/create behavior:
 *                 @L:name  Local memory only
 *                 @S:name  Force shared memory
 *                 @F:name  Load from FITS conf
 *                 @E:name  Must exist (error if not)
 *                 @N:name  Must not exist (error if)
 *               Modifiers are stackable: @LE:name
 *               See: milk-stream-help for details.
 *
 *   is_primary  1 if this parameter is a primary CLI
 *               argument (positional), 0 otherwise.
 *               Primary args are passed positionally
 *               on the command line.
 *
 *   flag        Bitwise OR of FPFLAG_* constants:
 *               FPFLAG_DEFAULT_INPUT  - standard input
 *               FPFLAG_DEFAULT_OUTPUT - standard output
 *               Other flags control visibility,
 *               writability, and behavior.
 *
 *   descr       Human-readable description string
 *               shown in help output and fps-info.
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".p_int32", &param_int32, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT, "Example INT32")            \
    X(".p_uint32", &param_uint32, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT, "Example UINT32")        \
    X(".p_int64", &param_int64, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Example INT64")            \
    X(".p_uint64", &param_uint64, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT, "Example UINT64")        \
    X(".p_float32", &param_float32, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT, "Example FLOAT32")    \
    X(".p_float64", &param_float64, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT, "Example FLOAT64")    \
    X(".p_onoff", &param_onoff, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT, "Example ONOFF")            \
    X(".p_streamname", param_streamname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM,      \
      "Example STREAMNAME (trigger)")                                                              \
    X(".p_fitsfile", param_fitsfilename, FPTYPE_FITSFILENAME, 0, FPFLAG_DEFAULT_INPUT,             \
      "Example FITSFILENAME")                                                                      \
    X(".p_string", param_string, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "Example STRING")         \
    X(".p_pid", &param_pid, FPTYPE_PID, 0, FPFLAG_DEFAULT_INPUT, "Example PID")                    \
    X(".p_timespec", &param_timespec, FPTYPE_TIMESPEC, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Example TIMESPEC")                                                                          \
    X(".p_filename", param_filename, FPTYPE_FILENAME, 0, FPFLAG_DEFAULT_INPUT, "Example FILENAME") \
    X(".p_execfile", param_execfilename, FPTYPE_EXECFILENAME, 0, FPFLAG_DEFAULT_INPUT,             \
      "Example EXECFILENAME")                                                                      \
    X(".p_dirname", param_dirname, FPTYPE_DIRNAME, 0, FPFLAG_DEFAULT_INPUT, "Example DIRNAME")     \
    X(".p_process", param_processname, FPTYPE_PROCESS, 0, FPFLAG_DEFAULT_INPUT, "Example PROCESS") \
    X(".p_fpsname", param_fpsname, FPTYPE_FPSNAME, 0, FPFLAG_DEFAULT_INPUT, "Example FPSNAME")     \
    X(".p_strnotstrm", param_string_not_stream, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT, \
      "Example STRING_NOT_STREAM")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 *
 * This is where the actual work happens.  Replace
 * this function with your algorithm.
 *
 * The local static variables (section 2) are
 * guaranteed to be synced with FPS shared memory
 * before this function is called.  Simply read them
 * directly — no FPS API calls needed here.
 *
 * Return RETURN_SUCCESS on success, or an errno_t
 * error code on failure.
 * ============================================================= */

/**
 * @brief Core computation function.
 *
 * Reads parameters from local static variables
 * (auto-synced from FPS) and performs the compute
 * unit's work.
 */
static MILK_HOT errno_t fpsexec()
{
    printf("\n[COMPUTATION] All FPS parameter types:\n");
    printf("  INT32              = %d\n", param_int32);
    printf("  UINT32             = %u\n", param_uint32);
    printf("  INT64              = %ld\n", param_int64);
    printf("  UINT64             = %lu\n", param_uint64);
    printf("  FLOAT32            = %f\n", param_float32);
    printf("  FLOAT64            = %f\n", param_float64);
    printf("  PID                = %d\n", (int) param_pid);
    printf("  TIMESPEC           = %ld.%09ld\n", param_timespec.tv_sec, param_timespec.tv_nsec);
    printf("  FILENAME           = %s\n", param_filename);
    printf("  FITSFILENAME       = %s\n", param_fitsfilename);
    printf("  EXECFILENAME       = %s\n", param_execfilename);
    printf("  DIRNAME            = %s\n", param_dirname);
    printf("  STREAMNAME         = %s\n", param_streamname);
    printf("  STRING             = %s\n", param_string);
    printf("  ONOFF              = %s\n", param_onoff ? "ON" : "OFF");
    printf("  PROCESS            = %s\n", param_processname);
    printf("  FPSNAME            = %s\n", param_fpsname);
    printf("  STRING_NOT_STREAM  = %s\n", param_string_not_stream);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 4b. CUSTOM CONFIGURATION CHECK
 *
 * Optional function called on every iteration of
 * the FPS configuration monitoring loop.  Use this
 * to validate parameter values and toggle parameter
 * visibility/flags based on the current state of
 * other parameters.
 *
 * It is set as .confcheck in the descriptor (section 7)
 * and called by the framework during confstart / confstep.
 *
 * In V2 compute units, parameter indices are NOT
 * known at compile time (no fpi_* variables).
 * Use functionparameter_GetParamIndex() to look up
 * the parray[] index by keyword string.
 * ============================================================= */

/**
 * @brief Example custom configuration check.
 *
 * Demonstrates two common patterns:
 *  1. Toggle visibility — show p_float64 only when
 *     p_onoff is ON.
 *  2. Range clamp — keep p_float32 within [0, 100].
 */
static MILK_COLD errno_t customCONFcheck()
{
    static long confcheck_cnt = 0;
    confcheck_cnt++;

    printf("[customCONFcheck] iteration %ld"
           "  fpsptr=%p\n",
           confcheck_cnt, (void *) dcfpsptr);

    if (dcfpsptr == NULL)
    {
        return RETURN_SUCCESS;
    }

    FPS *fps = dcfpsptr;

    /* --- Toggle p_float64 visibility --- */
    {
        int idx_onoff = functionparameter_GetParamIndex(fps, ".p_onoff");
        int idx_f64   = functionparameter_GetParamIndex(fps, ".p_float64");

        if (idx_onoff >= 0 && idx_f64 >= 0)
        {
            if (fps->parray[idx_onoff].fpflag & FPFLAG_ONOFF)
            {
                /* ON: show p_float64 */
                fps->parray[idx_f64].fpflag |= FPFLAG_USED;
                fps->parray[idx_f64].fpflag |= FPFLAG_VISIBLE;
            }
            else
            {
                /* OFF: hide p_float64 */
                fps->parray[idx_f64].fpflag &= ~FPFLAG_USED;
                fps->parray[idx_f64].fpflag &= ~FPFLAG_VISIBLE;
            }
        }
    }

    /* --- Range-clamp p_float32 to [0, 100] --- */
    {
        int idx_f32 = functionparameter_GetParamIndex(fps, ".p_float32");

        if (idx_f32 >= 0)
        {
            float val = fps->parray[idx_f32].val.f32[0];

            if (val < 0.0f)
            {
                fps->parray[idx_f32].val.f32[0] = 0.0f;
            }
            else if (val > 100.0f)
            {
                fps->parray[idx_f32].val.f32[0] = 100.0f;
            }
        }
    }

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  PARAMETER BINDINGS
 *
 * MILK_PARAMS_DEFINE generates fpscli_bind[] (FPS sync),
 * fpscli_farg[] (CLI arguments) and fpscli_rt (state).
 * ============================================================= */

MILK_PARAMS_DEFINE(fpscli, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER (processinfo loop support)
 *
 * This thin wrapper calls fpsexec() inside the
 * standard processinfo loop macros.  The macros
 * provide:
 *   - Process registration in shared memory
 *   - Timing and iteration counting
 *   - Signal handling (pause, stop, etc.)
 *   - Loop control for continuous-run mode
 *
 * For a one-shot "exec" command, the loop runs
 * exactly once.  For continuous processing (e.g.
 * runstart), the loop repeats until stopped.
 *
 * If your computation needs to resolve streams,
 * do so BEFORE MILK_PROCINFO_LOOPSTART(self)
 * (outside the loop).  Stream updates happen inside.
 * ============================================================= */

/**
 * @brief Processinfo-wrapped computation entry point.
 *
 * Called by both the standalone lifecycle and the
 * milk CLI.  Do NOT call fpsexec() directly from
 * outside this file — always go through
 * fpscli_compute() to get proper processinfo
 * tracking.
 */
static MILK_HOT errno_t fpscli_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    fpsexec();

    MILK_PROCINFO_END(self)

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK_RUNNABLE DESCRIPTOR AND FPSEXEC EXPORT
 *
 * The module registry (milk_module_example.c) lists
 * runnable_modex_fpscli, which makes it a milk CLI
 * command. The CLI core runs the FPS lifecycle and
 * calls .confcheck and .compute.
 *
 * EXPORT_FPSEXEC generates milk-fpsexec-clitest. Its
 * main() drives the same lifecycle (create, exec,
 * confstart, confstop, runstart, runstop, --help).
 * ============================================================= */

const MILK_RUNNABLE runnable_modex_fpscli = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "exfpscli",
                 .cmdkey      = "fpsclitest",
                 .description = "Test FPS-CLI unification" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(fpscli),
    .compute   = fpscli_compute,
    .confcheck = customCONFcheck,
};
EXPORT_FPSEXEC(milk, "clitest", runnable_modex_fpscli);
