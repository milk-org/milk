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
 * The V2 framework unifies three execution modes into
 * a single source file:
 *
 *  (A) Standalone executable (milk-fpsexec-*)
 *      Built with -DFPS_STANDALONE.  The
 *      FPS_MAIN_STANDALONE_V2 macro generates main()
 *      and handles the FPS lifecycle (create, exec,
 *      confstart, confstop, runstart, runstop).
 *
 *  (B) milk CLI command
 *      When compiled as part of a shared library
 *      module (e.g. libmilkCOREMODarith.so), the
 *      CLIADDCMD function registers the command so
 *      it can be called from the milk CLI prompt.
 *
 *  (C) Direct C function call
 *      The fpsexec() function can be called directly
 *      from other C code in the same process.
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
 * 2. Update section 1 (FPS_APP_INFO) with the new
 *    FPS name, CLI key, and description.
 * 3. Replace the local variables (section 2) with
 *    your parameters.
 * 4. Replace FPS_PARAMS (section 3) with entries
 *    matching your parameters.
 * 5. Replace fpsexec() (section 4) with your
 *    computation logic.
 * 6. Update the CLIADDCMD function name (section 7)
 *    and register it in your module's init function.
 * 7. Add the CMake targets for both the shared
 *    library (.so) and standalone executable.
 *
 * FILE STRUCTURE (8 sections)
 * ---------------------------
 *   1. FPS_APP_INFO    - identity (names, description)
 *   2. Local variables - C variables for parameters
 *   3. FPS_PARAMS      - X-macro parameter table
 *   4. fpsexec()       - computation logic
 *   5. Bindings/farg   - generated from FPS_PARAMS
 *   6. compute_function - processinfo loop wrapper
 *   7. CLI registration - CLIADDCMD + CLIfunction
 *   8. Standalone main  - FPS_MAIN_STANDALONE_V2
 */

/* ================================================================
 * INCLUDES
 *
 * Dependency structure:
 *   - fps.h: FPS types, X-macro expanders,
 *     FPS_MAIN_STANDALONE_V2 macro
 *   - CLIcore.h (full CLI build): CLICMDDATA,
 *     CLICMDARGDEF, INSERT_STD_* macros, module
 *     registration
 *   - CLIcore_standalone.h (standalone build):
 *     Provides stub types so the same source file
 *     compiles in both modes
 *
 * If your compute unit is PURE COMPUTATION (no
 * CLI registration, no INSERT_STD macros), you
 * can replace CLIcore.h with targeted includes:
 *   #include "libmilkdata/milkdata.h"
 *   #include "milkDebugTools.h"
 * ============================================================= */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 *
 * FPS_APP_INFO is the single source of truth for the
 * compute unit's identity. All other references
 * (CLIcmddata, help text, FPS shm name) are derived
 * from these fields.
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

static FPS_APP_INFO FPS_app_info = { .fps_name    = "exfpscli",
                                     .cmdkey      = "fpsclitest",
                                     .description = "Test FPS-CLI unification" };


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
 *   X_HELP_PRINT_V2 -> help text for standalone --help
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
 * This function is registered by assigning it to
 * CLIcmddata.FPS_customCONFcheck in CLIADDCMD
 * (section 7).  The framework calls it automatically
 * during confstart / confstep.
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
 * 5.  BINDINGS, FARG, AND CLI DATA
 *
 * These arrays are auto-generated from FPS_PARAMS
 * using the X-macro expansion.  In most cases you
 * do NOT need to modify this section — just change
 * FPS_PARAMS above and everything updates.
 *
 * my_bindings[] - Maps each FPS keyword to its local
 *   C variable pointer, type, and flags.  Used by
 *   the sync engine (fps_cli_sync.c) to copy values
 *   between FPS shared memory and local variables.
 *
 * farg[] - CLI argument definitions consumed by the
 *   milk CLI parser.  Determines how command-line
 *   arguments are parsed and validated.
 *
 * CLIcmddata - Command metadata (key, description).
 *   Populated from FPS_app_info at startup by the
 *   constructor below.  Used by INSERT_STD macros.
 *
 * ORDERING CONSTRAINT: This section MUST appear
 * before compute_function() because the
 * INSERT_STD_PROCINFO_COMPUTEFUNC_* macros
 * reference CLIcmddata.
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int nb_bindings = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };

/**
 * @brief Auto-initialize CLIcmddata from FPS_app_info.
 *
 * GCC constructor: runs before main().  Copies key
 * and description from FPS_app_info so that the
 * compute unit's identity is defined in ONE place.
 *
 * Also provides a valid cmdsettings object.  The
 * INSERT_STD_PROCINFO_COMPUTEFUNC_END macro
 * dereferences CLIcmddata.cmdsettings, which is NULL
 * from CLICMD_FIELDS_DEFAULTS.
 */
FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


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
 * do so BEFORE INSERT_STD_PROCINFO_COMPUTEFUNC_START
 * (outside the loop).  Stream updates happen inside.
 * ============================================================= */

/**
 * @brief Processinfo-wrapped computation entry point.
 *
 * Called by both the standalone lifecycle and the
 * milk CLI.  Do NOT call fpsexec() directly from
 * outside this file — always go through
 * compute_function() to get proper processinfo
 * tracking.
 */
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    fpsexec();

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 *
 * This section is compiled only when building as
 * part of a milk shared library module (i.e. when
 * FPS_STANDALONE is NOT defined).
 *
 * CLIfunction() is the entry point called when the
 * user types the command at the milk CLI prompt.
 * It delegates to safe_fps_generic_CLIfunction()
 * which handles the full FPS lifecycle:
 *   1. Parse CLI arguments
 *   2. Create/connect to FPS shared memory
 *   3. Sync CLI args -> FPS -> local variables
 *   4. Call compute_function()
 *
 * CLIADDCMD_* is called once at module load time
 * to register the command.  The function naming
 * convention is:
 *   CLIADDCMD_<module>__<function>
 * and must match the registration call in the
 * module's init function.
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

/**
 * @brief CLI entry point for this compute unit.
 *
 * Delegates to the generic FPS-CLI handler which
 * manages the full lifecycle.
 */
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

/**
 * @brief Register this compute unit with the milk CLI.
 *
 * Called from the module's init function.
 * safe_fps_fill_farg_examples() copies default values
 * from the bindings into farg[] so that --help shows
 * meaningful example values.
 */
errno_t CLIADDCMD_milk_module_example__fpscli()
{
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif /* !FPS_STANDALONE && !MILK_NO_CLI */


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 *
 * When compiled with -DFPS_STANDALONE, the
 * FPS_MAIN_STANDALONE_V2 macro generates a main()
 * function that provides the full standalone
 * executable lifecycle:
 *
 *   milk-fpsexec-<name> create [args...]
 *       Create a new FPS and set initial values
 *
 *   milk-fpsexec-<name> exec [args...]
 *       One-shot execution: create FPS, run once
 *
 *   milk-fpsexec-<name> confstart
 *       Start the configuration loop (watches FPS
 *       for parameter changes)
 *
 *   milk-fpsexec-<name> runstart
 *       Start the run loop (continuous processing)
 *
 *   milk-fpsexec-<name> confstop / runstop
 *       Stop the conf/run loop
 *
 *   milk-fpsexec-<name> --help
 *       Print help text with parameter descriptions
 *
 * The three arguments are:
 *   1. FPS_app_info     - compute unit identity
 *   2. FPS_PARAMS       - parameter X-macro
 *   3. compute_function - processinfo-wrapped entry
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(FPS_app_info, FPS_PARAMS, compute_function, customCONFcheck)
#endif
