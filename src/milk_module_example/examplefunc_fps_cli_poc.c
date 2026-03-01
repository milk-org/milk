/**
 * @file    examplefunc_fps_cli_poc.c
 * @brief   Proof of Concept (POC) for Unifying FPS and CLI Arguments
 *
 * This implementation demonstrates the "FPS-as-Primary" architecture.
 * Traditional MILK modules define CLI arguments in a static CLICMDARGDEF array,
 * which is then manually mapped to Function Parameter Structure (FPS) entries.
 *
 * This POC reverses that dependency:
 * 1. Parameters are defined ONCE in a unified structure (MY_PARAMS macro).
 * 2. This definition automatically populates both:
 *    - The FPS Shared Memory (for external control via milk-fps-set).
 *    - The CLI argument mapping (for order-based positional arguments).
 * 3. Local C variables are synchronized with FPS entries automatically.
 *
 * Benefits:
 * - Single source of truth for all parameters.
 * - Automatic CLI help generation derived from FPS descriptions.
 * - Support for both module-based (milk CLI) and standalone execution.
 */

#include "CLIcore.h"
#include "fps_add_entry.h"
#include <stdlib.h>

// =============================================================================
// PARAMETER DEFINITIONS
// =============================================================================

/* Local variables that the computation logic will actually use.
 * In this architecture, they are "synced" with FPS shared memory.
 */
static double  param_gain = 1.0;
static long    param_iter = 10;
static int32_t param_mode = 0;
static int32_t param_verbose = 0;
static float   param_threshold = 0.5;
static char    param_input_stream[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    param_output_path[FUNCTION_PARAMETER_STRMAXLEN] = "/tmp/output.fits";

/**
 * @brief Unified Parameter Macro (X-Macro Pattern)
 *
 * This macro centralizes the definition of all parameters.
 * Syntax: X(keyword, local_ptr, fptype, cli_index, fpflag, description)
 *
 * - keyword:   The name in the FPS (e.g. "gain" -> "exfpscli.gain")
 * - local_ptr: Pointer to the local C variable for syncing.
 * - fptype:    The MILK Function Parameter Type (FPTYPE_FLOAT64, etc.)
 * - cli_index: The 0-based position in the CLI call (milk-fpsclitest <arg0> <arg1>)
 * - fpflag:    Standard FPS flags (FPFLAG_DEFAULT_INPUT, etc.)
 * - description: Human-readable help text.
 */
#define MY_PARAMS(X) \
    X(".gain",         &param_gain,      FPTYPE_FLOAT64,    0, FPFLAG_DEFAULT_INPUT, "Gain parameter") \
    X(".iterations",   &param_iter,      FPTYPE_INT64,      1, FPFLAG_DEFAULT_INPUT, "Number of iterations") \
    X(".mode",         &param_mode,      FPTYPE_INT32,      2, FPFLAG_DEFAULT_INPUT, "Execution mode") \
    X(".verbose",      &param_verbose,   FPTYPE_ONOFF,      3, FPFLAG_DEFAULT_INPUT, "Verbose output") \
    X(".threshold",    &param_threshold, FPTYPE_FLOAT32,    4, FPFLAG_DEFAULT_INPUT, "Detection threshold") \
    X(".input_stream", param_input_stream, FPTYPE_STREAMNAME, -1, FPFLAG_DEFAULT_INPUT, "Input stream name") \
    X(".output_path",  param_output_path,  FPTYPE_STRING,      5, FPFLAG_DEFAULT_INPUT, "Output file path")


// =============================================================================
// INFRASTRUCTURE HELPERS
// =============================================================================

/**
 * @brief Binding structure between FPS keyword and local C variable.
 *
 * This structure effectively replaces CLICMDARGDEF by linking FPS keywords
 * directly to the memory locations of local variables.
 */
typedef struct
{
    const char *fpskeyword; /**< FPS keyword (e.g. "gain") */
    void       *ptr;        /**< Pointer to local variable for syncing */
    uint64_t    type;       /**< Expected type (FPTYPE_...) */
    int         cli_index;  /**< 0-based index in CLI arguments (-1 if not in CLI) */
    uint64_t    fpflag;     /**< FPS flags (e.g. FPFLAG_DEFAULT_INPUT) */
    const char *descr;      /**< Description for FPS and CLI help */
} FPS_CLI_BINDING;

/**
 * @brief Expansion of MY_PARAMS into an array of bindings.
 */
#define X_BINDING(kw, ptr, type, cli_idx, flag, desc) \
    { kw, ptr, type, cli_idx, flag, desc },

static FPS_CLI_BINDING my_bindings[] = {
    MY_PARAMS(X_BINDING)
};

static const int nb_bindings = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

/**
 * @brief Initialize FPS from bindings.
 *
 * Iterates through the bindings and ensures the corresponding entries exist
 * in the FPS structure. If they don't exist, they are created with the 
 * provided metadata and the current values of the local variables.
 */
static errno_t FPS_init_from_bindings(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *cmdkey,
    const char                *description,
    FPS_CLI_BINDING           *bindings,
    int                        nb_b
)
{
    strncpy(fps->md->callprogname, cmdkey, FPS_CALLPROGNAME_STRMAXLEN - 1);
    strncpy(fps->md->description, description, FPS_DESCR_STRMAXLEN - 1);

    for(int i = 0; i < nb_b; i++)
    {
        long pindex;
        uint64_t fpflag = bindings[i].fpflag;
        if (bindings[i].cli_index >= 0) {
            fpflag |= FPFLAG_PRIMARY_CLI_INPUT;
        }

        function_parameter_add_entry(
            fps,
            bindings[i].fpskeyword,
            bindings[i].descr,
            bindings[i].type,
            fpflag,
            bindings[i].ptr, // Initialize FPS value from the local C variable
            &pindex
        );
        functionparameter_SetParamCLIindex(fps, pindex, bindings[i].cli_index);
    }
    return RETURN_SUCCESS;
}

/* Global argc/argv for standalone argument capture */
static int   standalone_argc = 0;
static char **standalone_argv = NULL;

/**
 * @brief Sync CLI arguments to FPS and local variables.
 *
 * This is the core "Unification" function. It performs two steps:
 * 1. Updates FPS Shared Memory values from CLI command line tokens (if any).
 * 2. Updates local C variables from the (possibly updated) FPS values.
 *
 * This ensures that the computation logic always uses the most up-to-date
 * values, whether they came from the CLI call or from an external milk-fps-set.
 */
static errno_t FPS_process_CLI_and_sync(
    FUNCTION_PARAMETER_STRUCT *fps,
    FPS_CLI_BINDING           *bindings,
    int                        nb_b
)
{
    /* 1. Sync from CLI to FPS if applicable */
    
    if (standalone_argv != NULL) {
        /* MODE B: Running as standalone executable */
        /* We look for the subcommand (runstart, etc.) and take arguments after it */
        int cmd_pos = -1;
        for (int j = 1; j < standalone_argc; j++) {
            if (strcmp(standalone_argv[j], "runstart") == 0 || 
                strcmp(standalone_argv[j], "confstart") == 0 ||
                strcmp(standalone_argv[j], "confstep") == 0 ||
                strcmp(standalone_argv[j], "fpsinit") == 0) {
                cmd_pos = j;
                break;
            }
        }
        
        if (cmd_pos != -1) {
            for(int i = 0; i < nb_b; i++) {
                if (bindings[i].cli_index >= 0) {
                    int arg_idx = cmd_pos + 1 + bindings[i].cli_index;
                    if (arg_idx < standalone_argc) {
                        long pindex = functionparameter_GetParamIndex(fps, bindings[i].fpskeyword);
                        if (pindex != -1) {
                            if (bindings[i].type == FPTYPE_FLOAT64) {
                                fps->parray[pindex].val.f64[0] = atof(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_INT64) {
                                fps->parray[pindex].val.i64[0] = atol(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_INT32 || bindings[i].type == FPTYPE_ONOFF) {
                                fps->parray[pindex].val.i32[0] = atoi(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_FLOAT32) {
                                fps->parray[pindex].val.f32[0] = (float)atof(standalone_argv[arg_idx]);
                            } else if (bindings[i].type == FPTYPE_STREAMNAME || bindings[i].type == FPTYPE_STRING) {
                                strncpy(fps->parray[pindex].val.string[0], standalone_argv[arg_idx], FUNCTION_PARAMETER_STRMAXLEN - 1);
                            }
                        }
                    }
                }
            }
        }
    } else if (data.cmdindex >= 0) {
        /* MODE A: Running inside MILK CLI (module mode) */
        for(int i = 0; i < nb_b; i++) {
            if (bindings[i].cli_index >= 0) {
                /* Argument indexing: arg 0 is the command name, so we add 1 */
                int arg_idx = bindings[i].cli_index + 1;
                
                if (data.cmdargtoken[arg_idx].type != CLIARG_MISSING) {
                    long pindex = functionparameter_GetParamIndex(fps, bindings[i].fpskeyword);
                    if (pindex != -1) {
                        /* Update the FPS shared memory value */
                        if (bindings[i].type == FPTYPE_FLOAT64) {
                            fps->parray[pindex].val.f64[0] = data.cmdargtoken[arg_idx].val.numf;
                        } else if (bindings[i].type == FPTYPE_INT64) {
                            fps->parray[pindex].val.i64[0] = data.cmdargtoken[arg_idx].val.numl;
                        } else if (bindings[i].type == FPTYPE_INT32 || bindings[i].type == FPTYPE_ONOFF) {
                            fps->parray[pindex].val.i32[0] = (int32_t)data.cmdargtoken[arg_idx].val.numl;
                        } else if (bindings[i].type == FPTYPE_FLOAT32) {
                            fps->parray[pindex].val.f32[0] = (float)data.cmdargtoken[arg_idx].val.numf;
                        } else if (bindings[i].type == FPTYPE_STREAMNAME || bindings[i].type == FPTYPE_STRING) {
                            strncpy(fps->parray[pindex].val.string[0], data.cmdargtoken[arg_idx].val.string, FUNCTION_PARAMETER_STRMAXLEN - 1);
                        }
                    }
                }
            }
        }
    }

    /* 2. Sync from FPS to local C variables */
    for(int i = 0; i < nb_b; i++) {
        long pindex = functionparameter_GetParamIndex(fps, bindings[i].fpskeyword);
        if (pindex != -1) {
            if (bindings[i].type == FPTYPE_FLOAT64) {
                *((double *)bindings[i].ptr) = fps->parray[pindex].val.f64[0];
            } else if (bindings[i].type == FPTYPE_INT64) {
                *((long *)bindings[i].ptr) = fps->parray[pindex].val.i64[0];
            } else if (bindings[i].type == FPTYPE_INT32 || bindings[i].type == FPTYPE_ONOFF) {
                *((int32_t *)bindings[i].ptr) = fps->parray[pindex].val.i32[0];
            } else if (bindings[i].type == FPTYPE_FLOAT32) {
                *((float *)bindings[i].ptr) = fps->parray[pindex].val.f32[0];
            } else if (bindings[i].type == FPTYPE_STREAMNAME || bindings[i].type == FPTYPE_STRING) {
                strncpy((char *)bindings[i].ptr, fps->parray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN - 1);
            }
        }
    }

    return RETURN_SUCCESS;
}


// =============================================================================
// COMPUTATION LOGIC
// =============================================================================

/**
 * @brief The actual "heavy lifting" function.
 *
 * Note that this function is agnostic of how parameters were set. It simply
 * uses the local static variables, which are guaranteed to be synced.
 */
static errno_t example_fps_computation()
{
    printf("\n[COMPUTATION] Running with:\n");
    printf("  Gain        = %f\n", param_gain);
    printf("  Iterations  = %ld\n", param_iter);
    printf("  Mode        = %d\n", param_mode);
    printf("  Verbose     = %s\n", param_verbose ? "ON" : "OFF");
    printf("  Threshold   = %f\n", param_threshold);
    printf("  InStream    = %s\n", param_input_stream);
    printf("  OutPath     = %s\n", param_output_path);
    return RETURN_SUCCESS;
}


// =============================================================================
// STANDALONE FPS LIFECYCLE (Used by milk-fpsclitest executable)
// =============================================================================

/* These functions are called by the main() generated by FPS_MAIN_STANDALONE
 * when the user runs commands like './milk-fpsclitest fpsinit' or './milk-fpsclitest runstart'.
 */

int FPSINIT_exfpscli(const char *fps_name, const char *keywords, const char *description)
{
    FUNCTION_PARAMETER_STRUCT fps;
    /* Standard preamble handles SHM segment creation/connection */
    FPS_INIT_STD_PREAMBLE(fps, fps_name, keywords, description, "Unified FPS-CLI POC");
    
    /* Check for -procinfo flag in standalone_argv */
    int enable_procinfo = 0;
    if (standalone_argv != NULL) {
        for (int j = 1; j < standalone_argc; j++) {
            if (strcmp(standalone_argv[j], "-procinfo") == 0 || strcmp(standalone_argv[j], "--procinfo") == 0) {
                enable_procinfo = 1;
                break;
            }
        }
    }
    
    if (enable_procinfo) {
        fps.cmdset.flags |= CLICMDFLAG_PROCINFO;
        fps_add_processinfo_entries(&fps);
    }
    
    /* Populate the SHM from our bindings */
    FPS_init_from_bindings(&fps, "milk-fpsclitest", "Test FPS-CLI unification", my_bindings, nb_bindings);
    
    function_parameter_struct_disconnect(&fps);
    return 0;
}

int FPSCONF_exfpscli(const char *fps_name, int loop)
{
    /* FPS_CONF_STD_BODY implements a monitoring loop that watches for 
     * external changes to the FPS (e.g. from milk-fps-set).
     */
    FPS_CONF_STD_BODY(fps_name, loop, {}, {
        /* Optional: validation of parameters after they change */
    });
    return 0;
}

int FPSRUN_exfpscli(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;
    PROCESSINFO *processinfo = NULL;

    /* Standard preamble connects to the FPS and sets up process info */
    FPS_RUN_STD_PREAMBLE(fps_name, fps, {
        /* Sync our local pointers from the current FPS SHM values */
        FPS_process_CLI_and_sync(&fps, my_bindings, nb_bindings);
    });
    
    if (functionparameter_GetParamIndex(&fps, ".procinfo.enabled") != -1) {
        /* Setup processinfo using the FPS macros */
        FPS_RUN_PROCESSINFO_SETUP(processinfo, fps_name, "fpscli POC", "Unified FPS-CLI Demo", NULL, fps);
        
        FPS_RUN_PROCESSINFO_LOOP(processinfo, fps, NULL, NULL, {
            example_fps_computation();
        });
    } else {
        /* Execute the core logic */
        example_fps_computation();
        function_parameter_struct_disconnect(&fps);
    }
    
    return 0;
}

/* Placeholders for stop signals */
int FPSRUNSTOP_exfpscli(const char *fps_name)
{
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Stopping run process for '%s'\n", fps_name);
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found.\n", fps_name);
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
    printf("Stopping configuration process for '%s'\n", fps_name);
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found.\n", fps_name);
        return 1;
    }
    functionparameter_CONFstop(&fps);
    function_parameter_struct_disconnect(&fps);
    return 0;
}


// =============================================================================
// MILK CLI WRAPPER (Used when loaded as a module)
// =============================================================================

static errno_t CLIfunction(void)
{
    FUNCTION_PARAMETER_STRUCT fps;
    
    /* Try to connect to existing shared memory FPS */
    if (function_parameter_struct_connect("exfpscli", &fps, FPSCONNECT_SIMPLE) == -1) {
        /* If it doesn't exist, auto-initialize it */
        FPSINIT_exfpscli("exfpscli", NULL, "Auto-initialized");
        function_parameter_struct_connect("exfpscli", &fps, FPSCONNECT_SIMPLE);
    }

    /* Process positional CLI arguments and sync them to our local variables */
    FPS_process_CLI_and_sync(&fps, my_bindings, nb_bindings);
    
    /* Run the computation */
    example_fps_computation();
    
    function_parameter_struct_disconnect(&fps);
    return RETURN_SUCCESS;
}

/**
 * @brief Module registration function.
 */
errno_t CLIADDCMD_milk_module_example__fpscli()
{
    FUNCTION_PARAMETER_STRUCT fps;

    /* Connect to existing or auto-initialize FPS */
    if (function_parameter_struct_connect("exfpscli", &fps, FPSCONNECT_SIMPLE) == -1) {
        FPSINIT_exfpscli("exfpscli", NULL, "Test FPS-CLI unification");
        function_parameter_struct_connect("exfpscli", &fps, FPSCONNECT_SIMPLE);
    }
    
    CLICMDDATA CLIcmddata = {
        "",
        "",
        CLICMD_FIELDS_NOPARAM
    };
    strncpy(CLIcmddata.key, fps.md->callprogname, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, fps.md->description, sizeof(CLIcmddata.description) - 1);

    INSERT_STD_CLIREGISTERFUNC

    function_parameter_struct_disconnect(&fps);
    return RETURN_SUCCESS;
}


// =============================================================================
// STANDALONE MAIN
// =============================================================================

/* Redefine X_HELP_PRINT for the FPS_MAIN_STANDALONE help message generation.
 * This ensures that './milk-fpsclitest --help' displays parameters correctly.
 * We strip the leading dot if it exists for cleaner CLI help output.
 */
#undef X_HELP_PRINT
#define X_HELP_PRINT(kw, ptr, type, cli_idx, flag, desc) \
    { \
        char cli_idx_str[8]; \
        const char *disp_kw = (kw[0] == '.') ? &kw[1] : kw; \
        if(cli_idx >= 0) sprintf(cli_idx_str, "%2d", cli_idx); \
        else strcpy(cli_idx_str, " -"); \
        if(show_help_color && (cli_idx >= 0)) { \
            printf("  %s %s%-15s%s %s\n", cli_idx_str, COLORPRIMARY, disp_kw, COLORRESET, desc); \
        } else { \
            printf("  %s %-15s %s\n", cli_idx_str, disp_kw, desc); \
        } \
    }

/**
 * @brief Generate the standalone main() function with argument capture.
 *
 * We use a macro trick to rename the standard main generated by 
 * FPS_MAIN_STANDALONE, then define our own main to capture argc/argv.
 */
#define main main_real
FPS_MAIN_STANDALONE("exfpscli", exfpscli, "Unified FPS-CLI Demo", MY_PARAMS)
#undef main

int main(int argc, char *argv[])
{
    standalone_argc = argc;
    standalone_argv = argv;
    return main_real(argc, argv);
}
