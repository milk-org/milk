/**
 * @file    examplefunc_fps_cli_poc.c
 * @brief   Proof of Concept (POC) for Unifying FPS and CLI Arguments
 *
 * This implementation demonstrates the "FPS-as-Primary" architecture.
 * Traditional MILK modules define CLI arguments in a static CLICMDARGDEF
 * array, which is then manually mapped to Function Parameter Structure
 * (FPS) entries.
 *
 * This POC reverses that dependency:
 * 1. Parameters are defined ONCE in a unified structure (MY_PARAMS).
 * 2. This definition automatically populates both:
 *    - The FPS Shared Memory (for external control via milk-fps-set).
 *    - The CLI argument mapping (for order-based positional arguments).
 * 3. Local C variables are synchronized with FPS entries automatically.
 *
 * Benefits:
 * - Single source of truth for all parameters.
 * - Automatic CLI help generation derived from FPS descriptions.
 * - Support for both module-based (milk CLI) and standalone execution.
 *
 *
 * FILE ORGANIZATION
 * =================
 * This file is divided into two major sections to clarify which code
 * is specific to this particular compute unit vs. which code is
 * generic infrastructure that could be moved to libfps /
 * libprocessinfo.
 *
 * SECTION 1 — COMPUTE UNIT (module-specific)
 *   Code that each module author writes. Stays with the module.
 *
 * SECTION 2 — INFRASTRUCTURE (generic / library candidates)
 *   Boilerplate that is identical (or near-identical) across all
 *   FPS-based modules. Candidates for migration to libfps or
 *   libprocessinfo.
 */

#include "CLIcore.h"
#include "fps_add_entry.h"
#include <stdlib.h>








/* *********************************************************************
 * *********************************************************************
 * **                                                                 **
 * **  SECTION 1 — COMPUTE UNIT : MODULE-SPECIFIC CODE                **
 * **                                                                 **
 * **  Everything in this section is written by the module author     **
 * **  and stays with the module source file.                         **
 * **                                                                 **
 * *********************************************************************
 * *********************************************************************/








// =====================================================================
// 1.1  FPS COMPONENT IDENTITY
// =====================================================================

static struct {
    const char *fps_name;
    const char *cmdkey;
    const char *description;
} FPS_app_info = {
    .fps_name    = "exfpscli",
    .cmdkey      = "fpsclitest",
    .description = "Test FPS-CLI unification"
};

// =====================================================================
// 1.2  LOCAL PARAMETER VARIABLES
// =====================================================================

/**
 * Local variables that the computation logic will actually use.
 * In this architecture, they are "synced" with FPS shared memory.
 */
static int32_t  param_int32   = 123;
static uint32_t param_uint32  = 456;
static int64_t  param_int64   = 789;
static uint64_t param_uint64  = 101112;
static float    param_float32 = 3.14f;
static double   param_float64 = 2.718;
static pid_t    param_pid     = 1000;

static struct timespec param_timespec = {1709424000, 123456789};

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
static char param_string_not_stream[FUNCTION_PARAMETER_STRMAXLEN]
    = "not_a_stream";


// =====================================================================
// 1.3  UNIFIED PARAMETER TABLE (X-Macro)
// =====================================================================

/**
 * @brief Unified Parameter Macro (X-Macro Pattern)
 *
 * This macro centralizes the definition of all parameters.
 * Syntax: X(keyword, local_ptr, fptype, is_primary, fpflag, descr)
 *
 * - keyword:    Name in the FPS (e.g. ".gain" -> "exfpscli.gain")
 * - local_ptr:  Pointer to the local C variable for syncing.
 * - fptype:     MILK Function Parameter Type (FPTYPE_FLOAT64, etc.)
 * - is_primary: 1 if this is a primary CLI argument, 0 otherwise.
 * - fpflag:     Standard FPS flags (FPFLAG_DEFAULT_INPUT, etc.)
 * - descr:      Human-readable help text.
 */
#define MY_PARAMS(X) \
    X(".p_int32",       &param_int32,      FPTYPE_INT32,        1, FPFLAG_DEFAULT_INPUT, "Example INT32") \
    X(".p_uint32",      &param_uint32,     FPTYPE_UINT32,       0, FPFLAG_DEFAULT_INPUT, "Example UINT32") \
    X(".p_int64",       &param_int64,      FPTYPE_INT64,        0, FPFLAG_DEFAULT_INPUT, "Example INT64") \
    X(".p_uint64",      &param_uint64,     FPTYPE_UINT64,       0, FPFLAG_DEFAULT_INPUT, "Example UINT64") \
    X(".p_float32",     &param_float32,    FPTYPE_FLOAT32,      0, FPFLAG_DEFAULT_INPUT, "Example FLOAT32") \
    X(".p_float64",     &param_float64,    FPTYPE_FLOAT64,      0, FPFLAG_DEFAULT_INPUT, "Example FLOAT64") \
    X(".p_onoff",       &param_onoff,      FPTYPE_ONOFF,        0, FPFLAG_DEFAULT_INPUT, "Example ONOFF") \
    X(".p_streamname",  param_streamname,  FPTYPE_STREAMNAME,   1, FPFLAG_DEFAULT_INPUT, "Example STREAMNAME") \
    X(".p_fitsfile",    param_fitsfilename,FPTYPE_FITSFILENAME, 0, FPFLAG_DEFAULT_INPUT, "Example FITSFILENAME") \
    X(".p_string",      param_string,      FPTYPE_STRING,       0, FPFLAG_DEFAULT_INPUT, "Example STRING") \
    X(".p_pid",         &param_pid,        FPTYPE_PID,          0, FPFLAG_DEFAULT_INPUT, "Example PID") \
    X(".p_timespec",    &param_timespec,   FPTYPE_TIMESPEC,     0, FPFLAG_DEFAULT_INPUT, "Example TIMESPEC") \
    X(".p_filename",    param_filename,    FPTYPE_FILENAME,     0, FPFLAG_DEFAULT_INPUT, "Example FILENAME") \
    X(".p_execfile",    param_execfilename,FPTYPE_EXECFILENAME, 0, FPFLAG_DEFAULT_INPUT, "Example EXECFILENAME") \
    X(".p_dirname",     param_dirname,     FPTYPE_DIRNAME,      0, FPFLAG_DEFAULT_INPUT, "Example DIRNAME") \
    X(".p_process",     param_processname, FPTYPE_PROCESS,      0, FPFLAG_DEFAULT_INPUT, "Example PROCESS") \
    X(".p_fpsname",     param_fpsname,     FPTYPE_FPSNAME,      0, FPFLAG_DEFAULT_INPUT, "Example FPSNAME") \
    X(".p_strnotstrm",  param_string_not_stream, FPTYPE_STRING_NOT_STREAM, 0, FPFLAG_DEFAULT_INPUT, "Example STRING_NOT_STREAM")


// =====================================================================
// 1.4  COMPUTATION LOGIC
// =====================================================================

/**
 * @brief The actual "heavy lifting" function.
 *
 * This function is agnostic of how parameters were set. It simply
 * uses the local static variables, which are guaranteed to be synced.
 */
static errno_t example_fps_computation()
{
    printf("\n[COMPUTATION] Demo of all FPS parameter types:\n");
    printf("  INT32              = %d\n", param_int32);
    printf("  UINT32             = %u\n", param_uint32);
    printf("  INT64              = %ld\n", param_int64);
    printf("  UINT64             = %lu\n", param_uint64);
    printf("  FLOAT32            = %f\n", param_float32);
    printf("  FLOAT64            = %f\n", param_float64);
    printf("  PID                = %d\n", (int) param_pid);
    printf("  TIMESPEC           = %ld.%09ld\n",
           param_timespec.tv_sec, param_timespec.tv_nsec);
    printf("  FILENAME           = %s\n", param_filename);
    printf("  FITSFILENAME       = %s\n", param_fitsfilename);
    printf("  EXECFILENAME       = %s\n", param_execfilename);
    printf("  DIRNAME            = %s\n", param_dirname);
    printf("  STREAMNAME         = %s\n", param_streamname);
    printf("  STRING             = %s\n", param_string);
    printf("  ONOFF              = %s\n",
           param_onoff ? "ON" : "OFF");
    printf("  PROCESS            = %s\n", param_processname);
    printf("  FPSNAME            = %s\n", param_fpsname);
    printf("  STRING_NOT_STREAM  = %s\n",
           param_string_not_stream);

    return RETURN_SUCCESS;
}








/* *********************************************************************
 * *********************************************************************
 * **                                                                 **
 * **  SECTION 2 — INFRASTRUCTURE : GENERIC CODE                      **
 * **                                                                 **
 * **  Everything below is boilerplate that is identical (or nearly   **
 * **  identical) across all FPS-based modules. These are candidates  **
 * **  for migration into libfps or libprocessinfo.                   **
 * **                                                                 **
 * *********************************************************************
 * *********************************************************************/








// =====================================================================
// 2.1  FPS_CLI_BINDING type + X_BINDING expansion  → libfps
// =====================================================================

/**
 * @brief Binding structure between FPS keyword and local C variable.
 *
 * This structure effectively replaces CLICMDARGDEF by linking FPS
 * keywords directly to the memory locations of local variables.
 */
typedef struct
{
    const char *fpskeyword; /**< FPS keyword (e.g. "gain") */
    void       *ptr;        /**< Pointer to local variable */
    uint64_t    type;       /**< Expected type (FPTYPE_...) */
    int         is_primary; /**< 1 = primary CLI argument */
    uint64_t    fpflag;     /**< FPS flags */
    const char *descr;      /**< Description for help text */
} FPS_CLI_BINDING;

/**
 * @brief Expansion of MY_PARAMS into an array of bindings.
 */
#define X_BINDING(kw, ptr, type, is_primary, flag, desc) \
    { kw, ptr, type, is_primary, flag, desc },

static FPS_CLI_BINDING my_bindings[] = {
    MY_PARAMS(X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);


// =====================================================================
// 2.2  FPS_init_local()  → libfps
// =====================================================================

static int local_fps_initialized = 0;
static FUNCTION_PARAMETER_STRUCT local_fps_struct =
    {NULL, NULL, 0, -1, 0, 0, 0, {0}};

/**
 * @brief Allocate a process-local (non-SHM) FPS structure.
 *
 * Used when the FPS name starts with '_', meaning the FPS lives
 * only in the current process address space.
 */
static void FPS_init_local(
    const char *fps_name,
    long        NBparamMAX
)
{
    if (local_fps_initialized) {
        if (local_fps_struct.md != NULL) {
            free(local_fps_struct.md);
        }
        if (local_fps_struct.parray != NULL) {
            free(local_fps_struct.parray);
        }
        memset(&local_fps_struct, 0,
               sizeof(FUNCTION_PARAMETER_STRUCT));
    }

    local_fps_struct.md =
        malloc(sizeof(FUNCTION_PARAMETER_STRUCT_MD));
    memset(local_fps_struct.md, 0,
           sizeof(FUNCTION_PARAMETER_STRUCT_MD));

    local_fps_struct.parray =
        malloc(sizeof(FUNCTION_PARAMETER) * NBparamMAX);
    memset(local_fps_struct.parray, 0,
           sizeof(FUNCTION_PARAMETER) * NBparamMAX);

    strncpy(local_fps_struct.md->name, fps_name,
            STRINGMAXLEN_FPS_NAME - 1);
    local_fps_struct.md->NBparamMAX = NBparamMAX;
    local_fps_struct.NBparam = 0;
    local_fps_struct.SMfd = -1;
    local_fps_initialized = 1;
}


// =====================================================================
// 2.3  FPS_init_from_bindings()  → libfps
// =====================================================================

/**
 * @brief Initialize FPS entries from the bindings array.
 *
 * Iterates through the bindings and ensures the corresponding
 * entries exist in the FPS structure. If they don't exist, they
 * are created with the provided metadata and the current values
 * of the local variables.
 */
static errno_t FPS_init_from_bindings(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *cmdkey,
    const char                *description,
    FPS_CLI_BINDING           *bindings,
    int                        nb_b
)
{
    strncpy(fps->md->callprogname, cmdkey,
            FPS_CALLPROGNAME_STRMAXLEN - 1);
    strncpy(fps->md->description, description,
            FPS_DESCR_STRMAXLEN - 1);

    int current_cli_index = 0;

    for (int i = 0; i < nb_b; i++)
    {
        long pindex;
        uint64_t fpflag = bindings[i].fpflag;
        int cli_index = -1;

        if (bindings[i].is_primary) {
            fpflag |= FPFLAG_PRIMARY_CLI_INPUT;
            cli_index = current_cli_index++;
        }

        function_parameter_add_entry(
            fps,
            bindings[i].fpskeyword,
            bindings[i].descr,
            bindings[i].type,
            fpflag,
            bindings[i].ptr,
            &pindex
        );
        functionparameter_SetParamCLIindex(
            fps, pindex, cli_index);
    }
    return RETURN_SUCCESS;
}


// =====================================================================
// 2.4  CLI argument array + CLIcmddata  → libfps (CLI glue)
// =====================================================================

#define X_FARG(kw, ptr, fctype, is_primary, flag, desc) \
    { fctype, kw, desc, "", \
      flag | (is_primary ? FPFLAG_PRIMARY_CLI_INPUT : 0), \
      NULL, NULL },

static CLICMDARGDEF farg[] = {
    MY_PARAMS(X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};


// =====================================================================
// 2.5  FPS_process_CLI_and_sync()  → libfps
// =====================================================================

/* Global argc/argv for standalone argument capture */
static int   standalone_argc = 0;
static char **standalone_argv = NULL;

/**
 * @brief Sync CLI arguments to FPS and local variables.
 *
 * This is the core "Unification" function. It performs two steps:
 * 1. Updates FPS values from CLI command line tokens (if any).
 * 2. Updates local C variables from the (possibly updated) FPS.
 *
 * This ensures the computation logic always uses the most
 * up-to-date values, whether they came from the CLI call or
 * from an external milk-fps-set.
 */
static errno_t FPS_process_CLI_and_sync(
    FUNCTION_PARAMETER_STRUCT *fps,
    FPS_CLI_BINDING           *bindings,
    int                        nb_b
)
{
    /* ---- Step 1: Sync from CLI to FPS ---- */
    if (standalone_argv != NULL) {
        /* MODE B: Running as standalone executable.
         * Look for the subcommand and take arguments after it.
         */
        int cmd_pos = -1;
        for (int j = 1; j < standalone_argc; j++) {
            if (strcmp(standalone_argv[j], "runstart") == 0 ||
                strcmp(standalone_argv[j], "run") == 0 ||
                strcmp(standalone_argv[j], "exec") == 0 ||
                strcmp(standalone_argv[j], "confstart") == 0 ||
                strcmp(standalone_argv[j], "confstep") == 0 ||
                strcmp(standalone_argv[j], "fpsinit") == 0) {
                cmd_pos = j;
                break;
            }
        }

        /* If no explicit command found, try implicit 'run' */
        if (cmd_pos == -1) {
            for (int j = 1; j < standalone_argc; j++) {
                if (standalone_argv[j][0] != '-') {
                    cmd_pos = j;
                    break;
                }
            }
        }

        if (cmd_pos != -1) {
            int current_cli_index = 0;
            for (int i = 0; i < nb_b; i++) {
                if (bindings[i].is_primary) {
                    int arg_idx =
                        cmd_pos + 1 + current_cli_index;
                    current_cli_index++;

                    if (arg_idx < standalone_argc) {
                        long pindex =
                            functionparameter_GetParamIndex(
                                fps,
                                bindings[i].fpskeyword);
                        if (pindex != -1) {
                            if (bindings[i].type == FPTYPE_FLOAT64) {
                                fps->parray[pindex].val.f64[0] = atof(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_FLOAT32) {
                                fps->parray[pindex].val.f32[0] = (float) atof(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_INT64) {
                                fps->parray[pindex].val.i64[0] = atoll(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_UINT64) {
                                fps->parray[pindex].val.ui64[0] = (uint64_t) atoll(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_INT32 || bindings[i].type == FPTYPE_ONOFF) {
                                fps->parray[pindex].val.i32[0] = atoi(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_UINT32) {
                                fps->parray[pindex].val.ui32[0] = (uint32_t) atoi(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_PID) {
                                fps->parray[pindex].val.pid[0] = (pid_t) atoi(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_TIMESPEC) {
                                double val = atof(standalone_argv[arg_idx]);
                                fps->parray[pindex].val.ts[0].tv_sec = (long) val;
                                fps->parray[pindex].val.ts[0].tv_nsec = (long) ((val - (long) val) * 1e9);
                            } else if (FPTYPE_IS_STRING(bindings[i].type)) {
                                strncpy(fps->parray[pindex].val.string[0], standalone_argv[arg_idx], FUNCTION_PARAMETER_STRMAXLEN - 1);
                            }
                        }
                    }
                }
            }
        }
    } else {
        /* MODE A: Running as module in milk CLI.
         * Sync from CLI argdata (filled by CLI_checkarg_array)
         * to FPS.
         */
        CLIargs_to_FPSparams_setval(farg, nb_b, fps);
    }

    /* ---- Step 2: Sync from FPS to local C variables ---- */
    for (int i = 0; i < nb_b; i++) {
        long pindex = functionparameter_GetParamIndex(
            fps, bindings[i].fpskeyword);
        
        printf("DEBUG: Syncing %s (type %lu) -> pindex %ld\n", 
               bindings[i].fpskeyword, bindings[i].type, pindex);
        if (pindex != -1) {
            printf("DEBUG:   Found keywordfull: %s\n", fps->parray[pindex].keywordfull);
            if (bindings[i].type == FPTYPE_FLOAT64) {
                *((double *) bindings[i].ptr) = fps->parray[pindex].val.f64[0];
            } else if (bindings[i].type == FPTYPE_INT64) {
                *((int64_t *) bindings[i].ptr) = fps->parray[pindex].val.i64[0];
            } else if (bindings[i].type == FPTYPE_UINT64) {
                *((uint64_t *) bindings[i].ptr) = fps->parray[pindex].val.ui64[0];
            } else if (bindings[i].type == FPTYPE_INT32 || bindings[i].type == FPTYPE_ONOFF) {
                *((int32_t *) bindings[i].ptr) = fps->parray[pindex].val.i32[0];
            } else if (bindings[i].type == FPTYPE_UINT32) {
                *((uint32_t *) bindings[i].ptr) = fps->parray[pindex].val.ui32[0];
            } else if (bindings[i].type == FPTYPE_FLOAT32) {
                *((float *) bindings[i].ptr) = fps->parray[pindex].val.f32[0];
            } else if (bindings[i].type == FPTYPE_PID) {
                *((pid_t *) bindings[i].ptr) = fps->parray[pindex].val.pid[0];
            } else if (bindings[i].type == FPTYPE_TIMESPEC) {
                *((struct timespec *) bindings[i].ptr) = fps->parray[pindex].val.ts[0];
            } else if (FPTYPE_IS_STRING(bindings[i].type)) {
                strncpy((char *) bindings[i].ptr, fps->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN - 1);
            }
        }
    }

    return RETURN_SUCCESS;
}



// =====================================================================
// 2.6  FPSINIT  — standalone lifecycle: init  → libfps
// =====================================================================

int FPSINIT_exfpscli(
    const char *fps_name,
    const char *keywords,
    const char *description
)
{
    FUNCTION_PARAMETER_STRUCT fps;

    if (fps_name[0] == '_') {
        FPS_init_local(
            fps_name, FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        fps = local_fps_struct;
    } else {
        FPS_INIT_STD_PREAMBLE(
            fps, fps_name, keywords, description,
            "Unified FPS-CLI POC");
    }

    /* Check for -procinfo flag */
    int enable_procinfo = 0;
    if (standalone_argv != NULL) {
        for (int j = 1; j < standalone_argc; j++) {
            if (strcmp(standalone_argv[j], "-procinfo") == 0 ||
                strcmp(standalone_argv[j], "--procinfo") == 0)
            {
                enable_procinfo = 1;
                break;
            }
        }
    }

    if (data.cmd[data.cmdindex].cmdsettings.flags
        & CLICMDFLAG_PROCINFO) {
        enable_procinfo = 1;
    }

    if (enable_procinfo) {
        fps.cmdset.flags |= CLICMDFLAG_PROCINFO;
        fps_add_processinfo_entries(&fps);
    }

    /* Populate the FPS from our bindings */
    FPS_init_from_bindings(
        &fps,
        FPS_app_info.cmdkey,
        FPS_app_info.description,
        my_bindings,
        nb_bindings);

    if (fps_name[0] == '_') {
        /* Keep it initialized in local_fps_struct */
    } else {
        function_parameter_struct_disconnect(&fps);
    }
    return 0;
}


// =====================================================================
// 2.7  FPSCONF  — standalone lifecycle: conf  → libfps
// =====================================================================

int FPSCONF_exfpscli(const char *fps_name, int loop)
{
    if (fps_name[0] == '_') {
        printf("Local FPS '%s' — "
               "monitoring loop skipped.\n", fps_name);
        return 0;
    }
    FPS_CONF_STD_BODY(fps_name, loop, {}, {
        /* Optional: validation after parameter changes */
    });
    return 0;
}


// =====================================================================
// 2.8  FPSRUN  — standalone lifecycle: run  → libfps
// =====================================================================

int FPSRUN_exfpscli(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;
    PROCESSINFO *processinfo = NULL;

    if (fps_name[0] == '_') {
        if (!local_fps_initialized ||
            strcmp(local_fps_struct.md->name,
                   fps_name) != 0) {
            FPSINIT_exfpscli(
                fps_name, NULL, "Auto-initialized local");
        }
        fps = local_fps_struct;
        FPS_process_CLI_and_sync(
            &fps, my_bindings, nb_bindings);
    } else {
        FPS_RUN_STD_PREAMBLE(fps_name, fps, {
            FPS_process_CLI_and_sync(
                &fps, my_bindings, nb_bindings);
        });
    }

    if (functionparameter_GetParamIndex(
            &fps, ".procinfo.enabled") != -1) {
        FPS_RUN_PROCESSINFO_SETUP(
            processinfo, fps_name,
            FPS_app_info.cmdkey,
            FPS_app_info.description,
            NULL, fps);

        FPS_RUN_PROCESSINFO_LOOP(
            processinfo, fps, NULL, NULL, {
            example_fps_computation();
        });
    } else {
        example_fps_computation();
        if (fps_name[0] != '_') {
            function_parameter_struct_disconnect(&fps);
        }
    }

    return 0;
}


// =====================================================================
// 2.9  FPSRUNSTOP / FPSCONFSTOP  — lifecycle: stop  → libfps
// =====================================================================

int FPSRUNSTOP_exfpscli(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Stopping run process for '%s'\n", fps_name);
    if (fps_name[0] == '_') {
        printf("Local FPS '%s' — stop signal ignored "
               "(lifetime limited to process).\n",
               fps_name);
        return 0;
    }
    if (function_parameter_struct_connect(
            fps_name, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr,
                "Error: FPS '%s' not found.\n", fps_name);
        return 1;
    }
    functionparameter_RUNstop(&fps);
    function_parameter_struct_disconnect(&fps);
    functionparameter_FPS_processinfo_signal(fps_name, 3);
    return 0;
}

int FPSCONFSTOP_exfpscli(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Stopping configuration process for '%s'\n",
           fps_name);
    if (fps_name[0] == '_') {
        printf("Local FPS '%s' — stop signal ignored "
               "(lifetime limited to process).\n",
               fps_name);
        return 0;
    }
    if (function_parameter_struct_connect(
            fps_name, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr,
                "Error: FPS '%s' not found.\n", fps_name);
        return 1;
    }
    functionparameter_CONFstop(&fps);
    function_parameter_struct_disconnect(&fps);
    return 0;
}


// =====================================================================
// 2.10  compute_function() — procinfo wrapper  → libprocessinfo
// =====================================================================

/**
 * @brief Compute wrapper with processinfo loop support.
 *
 * Wraps example_fps_computation() in the standard processinfo loop
 * macros so that `..procinfo 1` and `..loopcntMax N` are respected
 * when running from the milk CLI.
 *
 * Requires:
 *  - data.FPS_name set to the FPS name string
 *  - CLIcmddata.cmdsettings pointing to a valid CMDSETTINGS
 *  - fps.cmdset zeroed so *ptr fields are NULL (not garbage)
 */
static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    example_fps_computation();

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


// =====================================================================
// 2.11  CLIfunction() — milk CLI entry point  → libfps
// =====================================================================

static errno_t CLIfunction(void)
{
    FUNCTION_PARAMETER_STRUCT fps;
    char fpsname_with_session[200];

    if (data.processname[0] != '\0') {
        snprintf(fpsname_with_session,
                 sizeof(fpsname_with_session),
                 "%s.%s",
                 FPS_app_info.fps_name,
                 data.processname);
    } else {
        strncpy(fpsname_with_session,
                FPS_app_info.fps_name,
                sizeof(fpsname_with_session) - 1);
    }

    /* Support standard FPS tags and cmdkey:fpsname:action */
    function_parameter_getFPSargs_from_CLIfunc(
        fpsname_with_session);

    if (data.FPS_CMDCODE == FPSCMDCODE_IGNORE) {
        return RETURN_SUCCESS;
    }

    /* If initialization action was requested via CLI */
    if (data.FPS_CMDCODE == FPSCMDCODE_FPSINIT ||
        data.FPS_CMDCODE == FPSCMDCODE_FPSINITCREATE) {
        FPSINIT_exfpscli(
            data.FPS_name, NULL, "Auto-initialized");
        return RETURN_SUCCESS;
    }

    if (data.FPS_CMDCODE == FPSCMDCODE_IGNORE) {
        return RETURN_SUCCESS;
    }

    /* Connect to existing FPS or use local */
    memset(&fps, 0, sizeof(FUNCTION_PARAMETER_STRUCT));
    fps.SMfd = -1;

    if (data.FPS_name[0] == '_') {
        if (!local_fps_initialized ||
            strcmp(local_fps_struct.md->name,
                   data.FPS_name) != 0) {
            FPSINIT_exfpscli(
                data.FPS_name,
                NULL, "Auto-initialized local");
        }
        fps = local_fps_struct;
    } else {
        if (function_parameter_struct_connect(
                data.FPS_name, &fps,
                FPSCONNECT_SIMPLE) == -1) {
            FPSINIT_exfpscli(
                data.FPS_name,
                NULL, "Auto-initialized");
            if (function_parameter_struct_connect(
                    data.FPS_name, &fps,
                    FPSCONNECT_SIMPLE) == -1) {
                printf("Failed to connect to FPS %s\n",
                       data.FPS_name);
                return RETURN_SUCCESS;
            }
        }
    }

    data.fpsptr = &fps;
    errno_t retval =
        CLI_checkarg_array(farg, CLIcmddata.nbarg);

    if (retval == RETURN_SUCCESS)
    {
        FPS_process_CLI_and_sync(
            &fps, my_bindings, nb_bindings);

        if (data.FPS_name[0] == '\0') {
            strncpy(data.FPS_name,
                    fpsname_with_session,
                    STRINGMAXLEN_FPS_NAME - 1);
            data.FPS_name[STRINGMAXLEN_FPS_NAME - 1] =
                '\0';
        }

        CLIcmddata.cmdsettings =
            &data.cmd[data.cmdindex].cmdsettings;

        if (CLIcmddata.cmdsettings->flags
            & CLICMDFLAG_PROCINFO)
        {
            memset(&fps.cmdset, 0, sizeof(fps.cmdset));
            fps.cmdset.procinfo_loopcntMax =
                CLIcmddata.cmdsettings->procinfo_loopcntMax;
            fps.cmdset.triggermode =
                CLIcmddata.cmdsettings->triggermode;
            strncpy(
                fps.cmdset.triggerstreamname,
                CLIcmddata.cmdsettings->triggerstreamname,
                STRINGMAXLEN_IMAGE_NAME - 1);
            fps.cmdset.triggerdelay =
                CLIcmddata.cmdsettings->triggerdelay;
            fps.cmdset.triggertimeout =
                CLIcmddata.cmdsettings->triggertimeout;
            fps.cmdset.semindexrequested =
                CLIcmddata.cmdsettings->semindexrequested;
            fps.cmdset.RT_priority =
                CLIcmddata.cmdsettings->RT_priority;
            fps.cmdset.procinfo_MeasureTiming =
                CLIcmddata.cmdsettings
                    ->procinfo_MeasureTiming;

            fps_add_processinfo_entries(&fps);
        }

        compute_function();
    }
    else if (retval == RETURN_CLICHECKARGARRAY_HELP ||
             retval == RETURN_CLICHECKARGARRAY_FUNCPARAMSET)
    {
        retval = RETURN_SUCCESS;
    }

    data.fpsptr = NULL;
    if (data.FPS_name[0] != '_') {
        function_parameter_struct_disconnect(&fps);
    }
    return retval;
}


// =====================================================================
// 2.12  CLIADDCMD — module registration  → libfps
// =====================================================================

/**
 * @brief Module registration function.
 */
errno_t CLIADDCMD_milk_module_example__fpscli()
{
    strncpy(CLIcmddata.key, FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);

    for (int i = 0; i < nb_bindings; i++) {
        switch (my_bindings[i].type) {
        case FPTYPE_INT32:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%d", *(int32_t *) my_bindings[i].ptr);
            break;
        case FPTYPE_UINT32:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%u", *(uint32_t *) my_bindings[i].ptr);
            break;
        case FPTYPE_INT64:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%ld", *(int64_t *) my_bindings[i].ptr);
            break;
        case FPTYPE_UINT64:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%lu", *(uint64_t *) my_bindings[i].ptr);
            break;
        case FPTYPE_FLOAT32:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%f", *(float *) my_bindings[i].ptr);
            break;
        case FPTYPE_FLOAT64:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%lf", *(double *) my_bindings[i].ptr);
            break;
        case FPTYPE_ONOFF:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%ld", *(uint64_t *) my_bindings[i].ptr);
            break;
        case FPTYPE_PID:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%d", *(pid_t *) my_bindings[i].ptr);
            break;
        case FPTYPE_TIMESPEC:
            snprintf(
                farg[i].example,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE,
                "%ld.%09ld",
                ((struct timespec *) my_bindings[i].ptr)->tv_sec,
                ((struct timespec *) my_bindings[i].ptr)->tv_nsec);
            break;
        case FPTYPE_STRING:
        case FPTYPE_STREAMNAME:
        case FPTYPE_FILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_FPSNAME:
        case FPTYPE_DIRNAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_PROCESS:
        case FPTYPE_STRING_NOT_STREAM:
            strncpy(
                farg[i].example,
                (char *) my_bindings[i].ptr,
                STRINGMAXLEN_FPSCLIARG_EXAMPLE - 1);
            break;
        }
    }

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}


// =====================================================================
// 2.13  Standalone main / FPS_MAIN_STANDALONE  → libfps
// =====================================================================

/**
 * Redefine X_HELP_PRINT for the FPS_MAIN_STANDALONE help message.
 * Ensures that './milk-fpsclitest --help' displays parameters.
 * We strip the leading dot for cleaner CLI help output.
 */
#undef X_HELP_PRINT
#define X_HELP_PRINT(kw, ptr, type, is_primary, flag, desc) \
    { \
        char cli_idx_str[8]; \
        char val_str[64] = ""; \
        const char *disp_kw = (kw[0] == '.') ? &kw[1] : kw; \
        if(is_primary) sprintf(cli_idx_str, "%3d", CLIargcnt); \
        else strcpy(cli_idx_str, " - "); \
        if (type == FPTYPE_INT32) sprintf(val_str, "%d", *(int32_t*)ptr); \
        else if (type == FPTYPE_UINT32) sprintf(val_str, "%u", *(uint32_t*)ptr); \
        else if (type == FPTYPE_INT64) sprintf(val_str, "%ld", *(int64_t*)ptr); \
        else if (type == FPTYPE_UINT64) sprintf(val_str, "%lu", *(uint64_t*)ptr); \
        else if (type == FPTYPE_FLOAT32) sprintf(val_str, "%f", *(float*)ptr); \
        else if (type == FPTYPE_FLOAT64) sprintf(val_str, "%f", *(double*)ptr); \
        else if (type == FPTYPE_ONOFF) sprintf(val_str, "%s", (*(int32_t*)ptr) ? "ON" : "OFF"); \
        else if (FPTYPE_IS_STRING(type)) strncpy(val_str, (char*)ptr, 63); \
        else if (type == FPTYPE_PID) sprintf(val_str, "%d", (int)*(pid_t*)ptr); \
        else if (type == FPTYPE_TIMESPEC) sprintf(val_str, "%ld.%09ld", ((struct timespec*)ptr)->tv_sec, ((struct timespec*)ptr)->tv_nsec); \
        if(show_help_color) { \
            if (is_primary) printf("  %s %s%-15s%s %-15s %s\n", cli_idx_str, COLORPRIMARY, disp_kw, COLORRESET, val_str, desc); \
            else printf("  %s %s%-15s%s %-15s %s\n", cli_idx_str, COLORARGnotCLI, disp_kw, COLORRESET, val_str, desc); \
        } else { \
            printf("  %s %-15s %-15s %s\n", cli_idx_str, disp_kw, val_str, desc); \
        } \
        if (is_primary) CLIargcnt++; \
    }

/**
 * @brief Generate the standalone main() with argument capture.
 *
 * We use a macro trick to rename the standard main generated by
 * FPS_MAIN_STANDALONE, then define our own main to capture argc/argv.
 */
#define main main_real
FPS_MAIN_STANDALONE(
    FPS_app_info.fps_name,
    exfpscli,
    FPS_app_info.description,
    MY_PARAMS)
#undef main

int main(int argc, char *argv[])
{
    standalone_argc = argc;
    standalone_argv = argv;
    return main_real(argc, argv);
}
