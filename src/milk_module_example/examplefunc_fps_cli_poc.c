/**
 * @file    examplefunc_fps_cli_poc.c
 * @brief   POC for unifying FPS and CLI arguments
 *
 * This example demonstrates:
 * 1. Defining parameters primarily via FPS.
 * 2. Using a binding structure to link FPS parameters to local variables.
 * 3. Parsing CLI arguments to update FPS values.
 */

#include "CLIcore.h"
#include "fps_add_entry.h" // For function_parameter_add_entry

// =============================================================================
// PROPOSED NEW INFRASTRUCTURE (To be moved to CLIcore / libfps later)
// =============================================================================

/**
 * @brief Binding structure between FPS keyword and local C variable
 */
typedef struct
{
    const char *fpskeyword; // FPS keyword (e.g. "params.gain")
    void       *ptr;        // Pointer to local variable
    uint64_t    type;       // Expected type (FPTYPE_...)
    int         cli_index;  // 0-based index in CLI arguments (-1 if not in CLI)
    uint64_t    fpflag;     // FPS flags (e.g. FPFLAG_DEFAULT_INPUT)
    const char *descr;      // Description for FPS
} FPS_CLI_BINDING;

/**
 * @brief Initialize FPS from bindings
 * 
 * Ensures all parameters in bindings exist in FPS.
 */
errno_t FPS_init_from_bindings(
    FUNCTION_PARAMETER_STRUCT *fps,
    FPS_CLI_BINDING           *bindings,
    int                        nb_bindings
)
{
    for(int i=0; i<nb_bindings; i++)
    {
        // Add entry if not exists (or update if exists)
        // Note: function_parameter_add_entry handles check internally
        function_parameter_add_entry(
            fps,
            bindings[i].fpskeyword,
            bindings[i].descr,
            bindings[i].type,
            bindings[i].fpflag,
            NULL, // Don't set value pointer yet, we do it via binding sync
            NULL
        );
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Process CLI arguments and update FPS
 * 
 * Matches CLI arguments (by index) to bindings, then updates FPS.
 * Also syncs FPS values to local pointers.
 * 
 * @param fps Connected FPS structure
 * @param bindings Array of bindings
 * @param nb_bindings Number of bindings
 * @param cli_args_start Index in CLIcmddata.cmdargtoken where args start (usually 1)
 */
errno_t FPS_process_CLI_and_sync(
    FUNCTION_PARAMETER_STRUCT *fps,
    FPS_CLI_BINDING           *bindings,
    int                        nb_bindings
)
{
    // 1. Update FPS from CLI arguments
    for(int i=0; i<nb_bindings; i++)
    {
        if(bindings[i].cli_index >= 0)
        {
            int arg_idx = bindings[i].cli_index + 1; // +1 because arg 0 is command itself
            
            // Check if argument exists provided by user
            if(data.cmdargtoken[arg_idx].type == CLIARG_MISSING) {
                // Handle optional/missing args if needed, or error
                continue; 
            }

            // Update FPS value from CLI token
            // We use functionparameter_SetParamValue_... functions usually, 
            // but here we might need to parse the token string/value.
            
            // Simplified logic: get string from token and use fps_set_param_from_string
            // or write directly to FPS memory if we are in process.
            
            // For this POC, let's assume we use the FPS API to set values
            // We need to resolve the parameter index first
            long pindex = -1;
            pindex = functionparameter_GetParamIndex(fps, bindings[i].fpskeyword);
            
            if(pindex != -1) {
                 // Convert CLI token to appropriate value and set in FPS
                 // accessing fps->parray[pindex].val...
                 
                 // Note: Implementation specific details of copying CLI token 
                 // to FPS value would go here.
                 // For now, let's demo FLOAT and INT
                 
                 if(bindings[i].type == FPTYPE_FLOAT64) {
                     double val = data.cmdargtoken[arg_idx].val.numf;
                     fps->parray[pindex].val.f64[0] = val;
                 }
                 else if(bindings[i].type == FPTYPE_INT64) {
                     long val = data.cmdargtoken[arg_idx].val.numl;
                     fps->parray[pindex].val.i64[0] = val;
                 }
                 // ... handle other types
            }
        }
    }

    // 2. Sync FPS to local pointers available in bindings
    for(int i=0; i<nb_bindings; i++)
    {
        if(bindings[i].ptr != NULL)
        {
            long pindex = -1;
            pindex = functionparameter_GetParamIndex(fps, bindings[i].fpskeyword);
            
            if(pindex != -1)
            {
                if(bindings[i].type == FPTYPE_FLOAT64) {
                    *((double*)bindings[i].ptr) = fps->parray[pindex].val.f64[0];
                }
                else if(bindings[i].type == FPTYPE_INT64) {
                    *((long*)bindings[i].ptr) = fps->parray[pindex].val.i64[0];
                }
                 // ... handle other types
            }
        }
    }
    
    return RETURN_SUCCESS;
}


// =============================================================================
// EXAMPLE FUNCTION USING NEW ARCHITECTURE
// =============================================================================

static double param_gain = 1.0;
static long   param_iter = 10;

// X-Macro definition of parameters for Standalone FPS creation
#define EXAMPLE_CLI_PARAMS(X) \
    X(FPTYPE_FLOAT64, double, "gain", "Gain parameter", "1.0", 1.0, &param_gain, FPFLAG_DEFAULT_INPUT) \
    X(FPTYPE_INT64, long, "iterations", "Number of iterations", "10", 10, &param_iter, FPFLAG_DEFAULT_INPUT)

#define EXAMPLE_CLI_HELPTEXT "Usage: fpsclitest <gain> <iterations>\nExample: fpsclitest 2.5 100\n"


// Define bindings for CLI (we still need this array to map indices to keywords)
static FPS_CLI_BINDING my_bindings[] = {
    {
        .fpskeyword = "gain",
        .ptr        = &param_gain,
        .type       = FPTYPE_FLOAT64,
        .cli_index  = 0, // First argument
        .fpflag     = FPFLAG_DEFAULT_INPUT,
        .descr      = "Gain parameter"
    },
    {
        .fpskeyword = "iterations",
        .ptr        = &param_iter,
        .type       = FPTYPE_INT64,
        .cli_index  = 1, // Second argument
        .fpflag     = FPFLAG_DEFAULT_INPUT,
        .descr      = "Number of iterations"
    }
};


static errno_t example_fps_computation()
{
    printf("Computing with:\n");
    printf("  Gain: %f\n", param_gain);
    printf("  Iter: %ld\n", param_iter);
    return RETURN_SUCCESS;
}

// -----------------------------------------------------------------------------
// CLI function implementation (Runs entirely in LOCAL memory)
// -----------------------------------------------------------------------------
static errno_t example_fps_cli_wrapper(void)
{
    // 1. Manually initialize a completely local FPS structure to avoid shared memory
    FUNCTION_PARAMETER_STRUCT fps = {0};
    FUNCTION_PARAMETER_STRUCT_MD fps_md = {0};
    // Allocate max 10 parameters (enough for our bindings)
    FUNCTION_PARAMETER fps_parray[10];
    memset(fps_parray, 0, sizeof(fps_parray));
    
    // Link the components together
    fps.md = &fps_md;
    fps.parray = fps_parray;
    fps.md->NBparamMAX = 10;
    fps.SMfd = -1; // Explicitly flag as NOT using shared memory
    fps.CMDmode = 0;
    
    strncpy(fps.md->name, "example_fps_cli", STRINGMAXLEN_FPS_NAME - 1);
    
    // 2. Set FPS core metadata representing CLI help, description, and syntax natively
    strncpy(fps.md->description, "POC: Test unifying CLI arguments with FPS", FPS_DESCR_STRMAXLEN - 1);
    strncpy(fps.md->helptext, EXAMPLE_CLI_HELPTEXT, FPS_HELPTEXT_STRMAXLEN - 1);
    
    // 3. Ensure parameters exist (Auto-definition)
    FPS_init_from_bindings(&fps, my_bindings, 2);
    
    // 4. Process CLI args -> FPS -> Local Vars
    // Here we can use CLI_checkarg logic or our custom parsing depending on how deep we want to integrate.
    FPS_process_CLI_and_sync(&fps, my_bindings, 2);
    
    // 5. Run computation
    example_fps_computation();
    
    // 6. Cleanup bypasses shared memory disconnect because SMfd == -1
    // function_parameter_struct_disconnect(&fps);
    
    return RETURN_SUCCESS;
}

// CLI Registration (New style registration)
static CLICMDDATA CLIcmddata =
{
    "fpsclitest",
    "Test FPS-CLI unification (New API)",
    __FILE__,
    0,     // nbarg (we handle args internally)
    NULL,  // funcfpscliarg pointer
    0,     // flags
    NULL,  // cmdsettings
    NULL,  // FPS_customCONFsetup
    NULL   // FPS_customCONFcheck
};

errno_t CLIADDCMD_milk_module_example__fpscli()
{
    // Use the new registration paradigm
    int cmdindex = RegisterCLIcmd(CLIcmddata, example_fps_cli_wrapper);
    if (cmdindex < 0) {
        return RETURN_FAILURE;
    }
    
    // Attach cmdsettings for full compatibility if needed
    CLIcmddata.cmdsettings = &data.cmd[cmdindex].cmdsettings;

    return RETURN_SUCCESS;
}


// -----------------------------------------------------------------------------
// STANDALONE IMPLEMENTATION (Uses Shared Memory ecosystem)
// -----------------------------------------------------------------------------

int FPSINIT_fpsclitest(
    const char *fps_name,
    const char *keywords,
    const char *description)
{
    FUNCTION_PARAMETER_STRUCT fps;
    FPS_INIT_STD_PREAMBLE(fps, fps_name, keywords, description, EXAMPLE_CLI_HELPTEXT);
    FPS_INIT_PROCINFO_DEFAULTS(fps, "", 10);

    // Use X-Macro to add all parameters to the FPS
#define X_FPS_INIT(fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) \
{ \
    c_type val = def_val; \
    void *vptr = &val; \
    if (FPTYPE_IS_STRING(fps_type)) { \
        vptr = *(void**)&val; \
    } \
    function_parameter_add_entry(&fps, key, descr, fps_type, cli_flags, vptr, NULL); \
}
    EXAMPLE_CLI_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT

    function_parameter_FPCONFexit(&fps);
    return 0;
}


int FPSCONF_fpsclitest(
    const char *fps_name,
    int loop)
{
    double *param_gain_ptr = NULL;
    long *param_iter_ptr = NULL;

    FPS_CONF_STD_BODY(fps_name, loop, 
        {
            // Map local pointers to FPS shared memory entries
            param_gain_ptr = functionparameter_GetParamPtr_FLOAT64(&fps, "gain");
            param_iter_ptr = functionparameter_GetParamPtr_INT64(&fps, "iterations");

            if (!param_gain_ptr || !param_iter_ptr) {
                fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
                function_parameter_FPCONFexit(&fps);
                return 1;
            }
        },
        {
            // Update local copies from the mapped pointers if necessary,
            // or perform validation
            param_gain = *param_gain_ptr;
            param_iter = *param_iter_ptr;
        }
    );
    return 0;
}

// Generate standard stop commands
FPS_MAKE_STANDALONE_CONFSTOP(fpsclitest)
FPS_MAKE_STANDALONE_RUNSTOP(fpsclitest)

int FPSRUN_fpsclitest(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    double *param_gain_ptr = NULL;
    long *param_iter_ptr = NULL;

    FPS_RUN_STD_PREAMBLE(fps_name, fps, {
        // Map local pointers
        param_gain_ptr = functionparameter_GetParamPtr_FLOAT64(&fps, "gain");
        param_iter_ptr = functionparameter_GetParamPtr_INT64(&fps, "iterations");

        if (!param_gain_ptr || !param_iter_ptr) {
            fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
            function_parameter_struct_disconnect(&fps);
            return 1;
        }
    });

    // Update global static vars so `example_fps_computation` uses correct values
    param_gain = *param_gain_ptr;
    param_iter = *param_iter_ptr;
    
    // Simulate process looping if needed or just one pass
    example_fps_computation();
    
    function_parameter_struct_disconnect(&fps);
    return 0;
}

#ifndef MILK_MODULE

int main(int argc, char *argv[]) {
    char fps_name[STRINGMAXLEN_FPS_NAME] = "fpsclitest";
    char arg_fps_name[STRINGMAXLEN_FPS_NAME] = "";
    int use_tmux = 0;
    int show_help = 0;
    int show_help_color = 0;
    char *command = NULL;
    char *keywords = NULL;
    char *description = NULL;
    char *colon_pos = NULL;

    // 1. First Pass: parse specific FPS standalone flags
    // Let's copy basic parsing logic from FPS_MAIN_STANDALONE
    int custom_args_start = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help = 1;
        } else if (strcmp(argv[i], "-hc") == 0 || strcmp(argv[i], "--help-color") == 0) {
            show_help = 1;
            show_help_color = 1;
        } else if (strcmp(argv[i], "-tmux") == 0) {
            use_tmux = 1;
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--keywords") == 0) && i + 1 < argc) {
            keywords = argv[++i];
        } else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--description") == 0) && i + 1 < argc) {
            description = argv[++i];
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--name") == 0) && i + 1 < argc) {
            strncpy(arg_fps_name, argv[++i], STRINGMAXLEN_FPS_NAME - 1);
        } else if (command == NULL) {
            command = argv[i];
            
            // Check for fpsname:command format
            if ((colon_pos = strchr(command, ':')) != NULL) {
                *colon_pos = '\0';
                strncpy(arg_fps_name, command, STRINGMAXLEN_FPS_NAME - 1);
                command = colon_pos + 1;
            }
            custom_args_start = i; // This or onwards are execution args
            break; // Stop parsing standard standalone flags once a positional arg is hit
        }
    }

    if (strlen(arg_fps_name) > 0) {
        strncpy(fps_name, arg_fps_name, STRINGMAXLEN_FPS_NAME - 1);
    }

    if (show_help || (argc < 2)) {
        printf("\nUsage: %s [fpsname:]<Command> [Options]\n", argv[0]);
        printf("       %s [fpsname:]<gain> <iterations>\n\n", argv[0]);
        printf("Commands:\n");
        printf("  fpsinit    One-time setup: creates the FPS shared memory segment.\n");
        printf("  fps        Print content of the FPS.\n");
        printf("  fpslist    List all FPS instances matching this executable.\n");
        printf("  confstart  Run the configuration monitoring loop.\n");
        printf("  confstep   Run a single configuration monitoring step.\n");
        printf("  confstop   Stop the configuration monitoring loop.\n");
        printf("  runstart   Run the main processing loop.\n");
        printf("  runstop    Stop the main processing loop.\n\n");
        printf("Detailed Help:\n");
        printf("--------------\n");
        printf("%s\n\n", EXAMPLE_CLI_HELPTEXT);
        return 0;
    }

    if (command == NULL) {
        fprintf(stderr, "Error: Missing command argument.\n");
        return 1;
    }

    // 2. Try handling standard standalone commands
    if (strcmp(command, "fps") == 0) {
        char cmd[STRINGMAXLEN_FPS_NAME + 20];
        snprintf(cmd, sizeof(cmd), "milk-fps-info %s", fps_name);
        return system(cmd);
    } else if (strcmp(command, "fpsinit") == 0) {
        return FPSINIT_fpsclitest(fps_name, keywords, description);
    } else if (strcmp(command, "confstart") == 0) {
        return FPSCONF_fpsclitest(fps_name, 1);
    } else if (strcmp(command, "confstep") == 0) {
        return FPSCONF_fpsclitest(fps_name, 0);
    } else if (strcmp(command, "confstop") == 0) {
        return FPSCONFSTOP_fpsclitest(fps_name);
    } else if (strcmp(command, "runstart") == 0) {
        return FPSRUN_fpsclitest(fps_name);
    } else if (strcmp(command, "runstop") == 0) {
        return FPSRUNSTOP_fpsclitest(fps_name);
    }

    // 3. Fallback: Parse positional CLI execution arguments!
    // At this point, `command` is considered the first positional parameter (e.g. gain)
    
    // We connect to the shared memory FPS (or local if it doesn't exist)
    FUNCTION_PARAMETER_STRUCT fps;
    int is_connected = 0;
    
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr, "Warning: FPS '%s' not found. Using local memory execution.\n", fps_name);
        
        // Setup local FPS memory
        FUNCTION_PARAMETER_STRUCT_MD *fps_md_mem = calloc(1, sizeof(FUNCTION_PARAMETER_STRUCT_MD));
        FUNCTION_PARAMETER *fps_parray_mem = calloc(10, sizeof(FUNCTION_PARAMETER));
        
        fps.md = fps_md_mem;
        fps.parray = fps_parray_mem;
        fps.md->NBparamMAX = 10;
        fps.SMfd = -1; 
        fps.CMDmode = 0;
        strncpy(fps.md->name, fps_name, STRINGMAXLEN_FPS_NAME - 1);
        FPS_init_from_bindings(&fps, my_bindings, 2);
    } else {
        is_connected = 1;
    }

    // Parse the positional args using the bindings
    int arg_idx = custom_args_start;
    for(int i = 0; i < 2; i++) {
        if (arg_idx < argc) {
            long pindex = functionparameter_GetParamIndex(&fps, my_bindings[i].fpskeyword);
            if (pindex != -1) {
                if(my_bindings[i].type == FPTYPE_FLOAT64) {
                    double val = atof(argv[arg_idx]);
                    fps.parray[pindex].val.f64[0] = val;
                } else if(my_bindings[i].type == FPTYPE_INT64) {
                    long val = atol(argv[arg_idx]);
                    fps.parray[pindex].val.i64[0] = val;
                }
            }
            arg_idx++;
        }
    }
    
    // Sync FPS to local pointers
    for(int i = 0; i < 2; i++) {
        long pindex = functionparameter_GetParamIndex(&fps, my_bindings[i].fpskeyword);
        if (pindex != -1) {
            if(my_bindings[i].type == FPTYPE_FLOAT64) {
                *((double*)my_bindings[i].ptr) = fps.parray[pindex].val.f64[0];
            } else if(my_bindings[i].type == FPTYPE_INT64) {
                *((long*)my_bindings[i].ptr) = fps.parray[pindex].val.i64[0];
            }
        }
    }

    // Run the actual computation
    printf("\n[Standalone FPS Execution]\n");
    example_fps_computation();
    
    // Cleanup
    if (is_connected) {
        function_parameter_struct_disconnect(&fps);
    } else {
        free(fps.md);
        free(fps.parray);
    }

    return 0;
}
#endif

