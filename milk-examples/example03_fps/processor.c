/**
 * @file milk-example-03-processor.c
 * @brief Integration of ImageStreamIO, libprocessinfo, and libfps.
 *
 * This program demonstrates the full suite of milk standalone libraries:
 * 1. libfps: Manages configurable parameters in shared memory.
 * 2. libprocessinfo: Provides process monitoring and lifecycle control.
 * 3. ImageStreamIO: High-performance data streaming.
 *
 * DUAL COMPILATION MODE:
 * This source file is designed to be compiled in two ways:
 *
 * 1. STANDALONE EXECUTABLE:
 *    Compiled as a standard executable. It manages its own main() loop,
 *    argument parsing, and process lifecycle.
 *    Usage: ./milk-example-03-processor [options] <command>
 *
 * 2. MILK CLI MODULE (Shared Object):
 *    Compiled as a dynamic library (.so) with -DMILK_MODULE.
 *    It binds to the milk CLI framework, registering a new command 'processor03'.
 *    FPS and ProcessInfo are managed by the CLI core.
 *    Usage (inside milk): mload processor03.so; processor03
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <sys/mman.h>

// Main FPS headers
#include "fps.h"
#include "fps_add_entry.h"
#include "fps_paramvalue.h"
#include "fps_FPCONFsetup.h"
#include "fps_FPCONFloopstep.h"
#include "fps_FPCONFexit.h"
#include "fps_CONFstop.h"
#include "fps_RUNstop.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_RUNexit.h"

// ProcessInfo headers
#include "processinfo.h"
#include "processinfo_shm_link.h"
#include "processinfo_procdirname.h"
#include "processtools_trigger.h"
#include "processinfo_update_output_stream.h"
#include "processinfo_setup.h"
#include "processinfo_loopstep.h"
#include "processinfo_exec_start.h"
#include "processinfo_exec_end.h"
#include "processinfo_signals.h"
#include "fps_processinfo_entries.h"

#include "ImageStreamIO.h"

#ifdef MILK_MODULE
#include "CLIcore.h"
#endif

// ===============================================================================================
// GLOBAL PARAMETERS (SHARED)
// ===============================================================================================
// These pointers hold the addresses of the parameter values in the FPS shared memory.
// They are initialized differently depending on the build mode:
// - Standalone: Initialized in FPSRUN_processor/FPSCONF_processor using functionparameter_GetParamPtr_...
// - Module: Initialized via the CLI argument binding mechanism (farg list).
static char *in_name_ptr = NULL;
static char *out_name_ptr = NULL;
static uint32_t *roi_size_ptr = NULL;
static uint32_t *off_x_ptr = NULL;


/* =============================================================================================== */
/* =============================================================================================== */
/* SHARED LOGIC                                                                                    */
/* =============================================================================================== */
/* =============================================================================================== */

/**
 * @brief Shared processing logic for one loop iteration.
 *
 * This function performs the actual data processing (copying ROI).
 * It is called by both the standalone run loop and the CLI compute function.
 *
 * @param fps          Pointer to FPS structure (can be NULL if not available/needed)
 * @param processinfo  Pointer to ProcessInfo structure (for timing/status)
 * @param input_image  Pointer to input IMAGE structure
 * @param output_image Pointer to output IMAGE structure
 */
static void processor03_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *input_image, IMAGE *output_image) {
    // RE-READ PARAMETERS DYNAMICALLY
    // Update local ProcessInfo settings (like triggers, priority) from FPS if changed
    if (fps) {
        fps_to_processinfo(fps, processinfo);
    }
    
    // Safety check
    if (!off_x_ptr || !roi_size_ptr) return;

    // Dereference pointers to get current parameter values from shared memory
    uint32_t off_x = *off_x_ptr;
    uint32_t roi_size = *roi_size_ptr;
    
    uint32_t in_w = input_image->md[0].size[0];
    float *in_data = (float*)input_image->array.raw;
    float *out_data = (float*)output_image->array.raw;

    // Process: Copy ROI from input to output
    for(uint32_t y=0; y<roi_size; y++) {
        for(uint32_t x=0; x<roi_size; x++) {
            if (x + off_x < in_w)
                out_data[y*roi_size + x] = in_data[y*in_w + (x + off_x)];
            else
                out_data[y*roi_size + x] = 0;
        }
    }
    printf(".");
    fflush(stdout);
}

/**
 * @brief Shared validation logic for configuration loop.
 *
 * Checks if parameters are valid (e.g., ROI fits in image) and clamps them if necessary.
 * Called by FPSCONF_processor (Standalone) and customCONFcheck (Module).
 */
static void processor03_validate() {
    if (!in_name_ptr || !roi_size_ptr || !off_x_ptr) return;

    // We need to access the input image metadata to check dimensions.
    // We open a temporary connection to the stream.
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name_ptr, &input_image) == 0) {
        uint32_t width = input_image.md[0].size[0];
        
        // Check if ROI + Offset exceeds image width
        if (*off_x_ptr + *roi_size_ptr > width) {
            // Logic: prioritize keeping ROI size, shift offset if possible
            
            // 1. Clamp off_x if it starts outside image
            if (*off_x_ptr > width) {
                *off_x_ptr = 0;
            }
            // 2. If still out of bounds
            if (*off_x_ptr + *roi_size_ptr > width) {
                if (*roi_size_ptr > width) {
                    // ROI is bigger than image -> clamp ROI to image size, reset offset
                    *roi_size_ptr = width;
                    *off_x_ptr = 0;
                } else {
                    // ROI fits, but offset is too large -> shift offset back
                    *off_x_ptr = width - *roi_size_ptr;
                }
            }
        }
        ImageStreamIO_closeIm(&input_image);
    }
}


/* =============================================================================================== */
/* =============================================================================================== */
/* STANDALONE IMPLEMENTATION                                                                       */
/* =============================================================================================== */
/* =============================================================================================== */

/**
 * @brief Performs one-time setup of the Function Parameter Structure (FPS).
 * This creates the shared memory segment and initializes default values.
 */
int FPSINIT_processor(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Initializing FPS '%s'...\n", fps_name);

    // Create the FPS entry
    fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
    strncpy(fps.md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX - 1);
    fps.md->sourceline = __LINE__;

    if (keywords != NULL) {
        strncpy(fps.md->keywordarray, keywords, FPS_KEYWORDARRAY_STRMAXLEN - 1);
    }
    if (description != NULL) {
        strncpy(fps.md->description, description, FPS_DESCR_STRMAXLEN - 1);
    }

    // ------------------------------------------------------------------------
    // INITIALIZE DEFAULTS IN cmdset
    // ------------------------------------------------------------------------
    strncpy(fps.cmdset.triggerstreamname, "stream03", STRINGMAXLEN_IMAGE_NAME - 1);
    fps.cmdset.procinfo_loopcntMax = -1; // Infinite
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
    fps.cmdset.triggertimeout.tv_sec = 10;
    fps.cmdset.triggertimeout.tv_nsec = 0;

    // ------------------------------------------------------------------------
    // REGISTER CUSTOM PARAMETERS
    // ------------------------------------------------------------------------
    char *in_name = "stream03";
    function_parameter_add_entry(&fps, ".in_name", "Input Stream Name", FPTYPE_STRING, FPFLAG_DEFAULT_INPUT, (void*)in_name, NULL);

    char *out_name = "stream03_proc";
    function_parameter_add_entry(&fps, ".out_name", "Output Stream Name", FPTYPE_STRING, FPFLAG_DEFAULT_INPUT, (void*)out_name, NULL);

    uint32_t roi_size = 50;
    function_parameter_add_entry(&fps, ".roi_size", "ROI Size", FPTYPE_UINT32, FPFLAG_DEFAULT_INPUT, (void*)&roi_size, NULL);

    uint32_t off_x = 0;
    function_parameter_add_entry(&fps, ".off_x", "Offset X", FPTYPE_UINT32, FPFLAG_DEFAULT_INPUT, (void*)&off_x, NULL);

    // Add standard processinfo parameters (RTprio, cset, triggermode, etc.)
    fps_add_processinfo_entries(&fps);
    functionparameter_SetParamValue_ONOFF(&fps, ".procinfo.MeasureTiming", 1);

    // Finalize creation
    function_parameter_FPCONFexit(&fps);
    return 0;
}

/**
 * @brief Helper to handle tmux logic for automated multi-window setup.
 */
void handle_tmux(const char *fps_name, const char *command, int argc, char *argv[], const char *keywords, const char *description) {
    char cmd[2048];

    // Check if tmux is installed
    if (system("command -v tmux > /dev/null 2>&1") != 0) {
        fprintf(stderr, "\nError: 'tmux' is not installed or not in PATH.\n");
        fprintf(stderr, "The -tmux option requires tmux to be installed on your system.\n\n");
        exit(EXIT_FAILURE);
    }

    // Check if session exists
    snprintf(cmd, sizeof(cmd), "tmux has-session -t %s 2>/dev/null", fps_name);
    int ret = system(cmd);
    if (ret != 0) {
        printf("Creating tmux session '%s'\n", fps_name);
        snprintf(cmd, sizeof(cmd), "tmux new-session -d -s %s -n ctrl", fps_name);
        system(cmd);
        snprintf(cmd, sizeof(cmd), "tmux new-window -t %s -n conf", fps_name);
        system(cmd);
        snprintf(cmd, sizeof(cmd), "tmux new-window -t %s -n run", fps_name);
        system(cmd);
        sleep(1); // Wait for shells
    }

    char path[1024];
    ssize_t path_len = readlink("/proc/self/exe", path, sizeof(path)-1);
    if (path_len != -1) {
        path[path_len] = '\0';
    } else {
        if (realpath(argv[0], path) == NULL) strncpy(path, argv[0], 1023);
    }

    // Reconstruct arguments to pass custom name if needed
    char name_arg[256] = "";
    if (strcmp(fps_name, "processor03") != 0) {
        snprintf(name_arg, sizeof(name_arg), " -n %s", fps_name);
    }

    if (strcmp(command, "confstart") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t %s:conf \"%s confstart%s\" C-m", fps_name, path, name_arg);
        system(cmd);
        printf("Dispatched 'confstart' to tmux window %s:conf\n", fps_name);
    } else if (strcmp(command, "confstep") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t %s:conf \"%s confstep%s\" C-m", fps_name, path, name_arg);
        system(cmd);
        printf("Dispatched 'confstep' to tmux window %s:conf\n", fps_name);
    } else if (strcmp(command, "runstart") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t %s:run \"%s runstart%s\" C-m", fps_name, path, name_arg);
        system(cmd);
        printf("Dispatched 'runstart' to tmux window %s:run\n", fps_name);
    } else if (strcmp(command, "fpsinit") == 0) {
        FPSINIT_processor(fps_name, keywords, description);
    }
}

/**
 * @brief The Configuration process (Standalone).
 * This process typically runs in the background and validates parameter changes.
 */
int FPSCONF_processor(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;

    if (loop) {
        printf("Starting configuration process loop for '%s'\n", fps_name);
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);

        // Map pointers to shared memory parameters
        in_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
        roi_size_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
        off_x_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

        if (!in_name_ptr || !roi_size_ptr || !off_x_ptr) {
            fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
            function_parameter_FPCONFexit(&fps);
            return 1;
        }

        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) {
            if (function_parameter_FPCONFloopstep(&fps)) {
                processor03_validate();
            }
            usleep(10000);
        }
    } else {
        printf("Running single configuration step for '%s'\n", fps_name);
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
        function_parameter_FPCONFloopstep(&fps);
    }

    function_parameter_FPCONFexit(&fps);
    return 0;
}

int FPSCONFSTOP_processor(const char *fps_name) {
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

int FPSRUNSTOP_processor(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Stopping run process for '%s'\n", fps_name);
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found.\n", fps_name);
        return 1;
    }
    functionparameter_RUNstop(&fps);
    function_parameter_struct_disconnect(&fps);
    
    // Also signal via ProcessInfo shared memory
    char procdname[STRINGMAXLEN_DIR_NAME];
    processinfo_procdirname(procdname);
    DIR *d = opendir(procdname);
    if (d) {
        struct dirent *dir;
        while ((dir = readdir(d)) != NULL) {
            char prefix[256];
            snprintf(prefix, sizeof(prefix), "proc.%s.", fps_name);
            if (strncmp(dir->d_name, prefix, strlen(prefix)) == 0) {
                char fullpath[2048];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, dir->d_name);
                int fd;
                PROCESSINFO *pinfo = processinfo_shm_link(fullpath, &fd);
                if (pinfo != (PROCESSINFO *)MAP_FAILED) {
                    printf("Signaling process PID %d to exit...\n", (int)pinfo->PID);
                    pinfo->CTRLval = 3; 
                    munmap(pinfo, sizeof(PROCESSINFO));
                    close(fd);
                }
            }
        }
        closedir(d);
    }
    return 0;
}

/**
 * @brief The main Processing loop (Standalone).
 */
int FPSRUN_processor(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;

    // 1. CONNECT TO FPS
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found. Run 'fpsinit' first.\n", fps_name);
        return 1;
    }

    // 2. RETRIEVE CURRENT PARAMETERS (Get Pointers)
    in_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
    out_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".out_name");
    roi_size_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
    off_x_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

    if (!in_name_ptr || !out_name_ptr || !roi_size_ptr || !off_x_ptr) {
        fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
        function_parameter_struct_disconnect(&fps);
        return 1;
    }

    // 3. INITIALIZE STREAMS
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name_ptr, &input_image) != 0) {
        fprintf(stderr, "Error connecting to input %s\n", in_name_ptr);
        return 1;
    }

    IMAGE output_image;
    uint32_t dims[2] = {*roi_size_ptr, *roi_size_ptr};
    if (ImageStreamIO_createIm_gpu(&output_image, out_name_ptr, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != 0) {
        return 1;
    }

    // 4. SETUP PROCESS MONITORING
    PROCESSINFO *processinfo = processinfo_setup((char*)fps_name, "Ex03 Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    if (!processinfo) return 1;

    processinfo_CatchSignals();
    processinfo_waitoninputstream_init(processinfo, &input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, processinfo);
    processinfo_loopstart(processinfo);

    // 5. MAIN LOOP
    int loopOK = 1;
    while(loopOK) {
        loopOK = processinfo_loopstep(processinfo);
        if(!loopOK) break;

        processinfo_waitoninputstream(processinfo);
        if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;

        processinfo_exec_start(processinfo);
        
        processor03_compute(&fps, processinfo, &input_image, &output_image);

        processinfo_exec_end(processinfo);
        processinfo_update_output_stream(processinfo, &output_image, &input_image);
    }

    processinfo_cleanExit(processinfo);
    function_parameter_struct_disconnect(&fps);
    return 0;
}

#ifndef MILK_MODULE
int main(int argc, char *argv[]) {
    // Basic argument check
    if (argc < 2) {
        printf("Usage: %s <fpsinit|confstart|confstep|confstop|runstart|runstop> [Options]\n", "milk-example-03-processor");
        printf("Run '%s -h' for detailed help.\n", "milk-example-03-processor");
        return 1;
    }

    // Help display
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        printf("\nUsage: %s <Command> [Options]\n\n", "milk-example-03-processor");
        printf("Description:\n");
        printf("  Full-featured example demonstrating FPS (parameters), ProcessInfo (monitoring),\n");
        printf("  and ImageStreamIO (data) working standalone from CLIcore.\n\n");
        printf("Commands:\n");
        printf("  fpsinit    One-time setup: creates the FPS shared memory segment.\n");
        printf("  confstart  Run the configuration monitoring loop (infinite). Validates parameter changes.\n");
        printf("  confstep   Run a single configuration monitoring step.\n");
        printf("  confstop   Stop the configuration monitoring loop.\n");
        printf("  runstart   Run the main ROI processing loop.\n");
        printf("  runstop    Stop the main ROI processing loop.\n\n");
        printf("Options:\n");
        printf("  -n, --name NAME          Specify FPS name (default: processor03).\n");
        printf("  -k, --keywords KEYWORDS  Specify FPS keywords (default: NULL).\n");
        printf("  -d, --description DESC   Specify FPS description (default: NULL).\n");
        printf("  -tmux                    Auto-create a tmux session and dispatch commands.\n");
        printf("                           - 'confstart' command goes to window 1.\n");
        printf("                           - 'runstart' command goes to window 2.\n");
        printf("                           - 'ctrl' window (index 0) remains for user interaction.\n\n");
        printf("Typical Workflow:\n");
        printf("  1. Terminal A: ./milk-example-03-writer\n");
        printf("  2. Terminal B: ./milk-example-03-processor fpsinit\n");
        printf("  3. Terminal B: ./milk-example-03-processor runstart -tmux\n");
        printf("  4. Terminal B: ./milk-example-03-processor confstart -tmux\n");
        printf("  5. Connect:    tmux a -t processor03\n\n");
        return 0;
    }

    char fps_name[STRINGMAXLEN_FPS_NAME] = "processor03";
    int use_tmux = 0;
    char *command = NULL;
    char *keywords = NULL;
    char *description = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-tmux") == 0) {
            use_tmux = 1;
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--name") == 0) && i + 1 < argc) {
            strncpy(fps_name, argv[++i], STRINGMAXLEN_FPS_NAME - 1);
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--keywords") == 0) && i + 1 < argc) {
            keywords = argv[++i];
        } else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--description") == 0) && i + 1 < argc) {
            description = argv[++i];
        } else if (command == NULL) {
            command = argv[i];
        }
    }

    if (command == NULL) {
        fprintf(stderr, "Error: Missing command argument.\n");
        return 1;
    }

    if (use_tmux) {
        handle_tmux(fps_name, command, argc, argv, keywords, description);
        return 0;
    }

    if (strcmp(command, "fpsinit") == 0) {
        return FPSINIT_processor(fps_name, keywords, description);
    } else if (strcmp(command, "confstart") == 0) {
        return FPSCONF_processor(fps_name, 1);
    } else if (strcmp(command, "confstep") == 0) {
        return FPSCONF_processor(fps_name, 0);
    } else if (strcmp(command, "confstop") == 0) {
        return FPSCONFSTOP_processor(fps_name);
    } else if (strcmp(command, "runstart") == 0) {
        return FPSRUN_processor(fps_name);
    } else if (strcmp(command, "runstop") == 0) {
        return FPSRUNSTOP_processor(fps_name);
    }

    fprintf(stderr, "Invalid command: %s\n", command);
    return 1;
}
#endif


/* =============================================================================================== */
/* =============================================================================================== */
/* MODULE IMPLEMENTATION (milk-CLI)                                                                */
/* =============================================================================================== */
/* =============================================================================================== */

#ifdef MILK_MODULE

#define MODULE_SHORTNAME_DEFAULT "proc03"
#define MODULE_DESCRIPTION "Example processor 03 module"

// CLI argument definition
// This array binds CLI arguments to the global parameter pointers.
// When the function is called from CLI, the CLIcore logic will set these pointers
// to point to the values within the FPS structure managed by the CLI.
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR, ".in_name", "input stream name", "stream03",
        CLIARG_VISIBLE_DEFAULT, (void **) &in_name_ptr, NULL
    },
    {
        CLIARG_STR, ".out_name", "output stream name", "stream03_proc",
        CLIARG_VISIBLE_DEFAULT, (void **) &out_name_ptr, NULL
    },
    {
        CLIARG_UINT32, ".roi_size", "ROI size", "50",
        CLIARG_VISIBLE_DEFAULT, (void **) &roi_size_ptr, NULL
    },
    {
        CLIARG_UINT32, ".off_x", "offset X", "0",
        CLIARG_VISIBLE_DEFAULT, (void **) &off_x_ptr, NULL
    }
};

/**
 * @brief Custom configuration check for module.
 * Called by FPS framework during configuration loop.
 */
static errno_t customCONFcheck() {
    // In CLI mode, STD_FARG_LINKfunction (called by FPSCONFfunction macro)
    // links the global variables (in_name_ptr etc) to the FPS entries.
    // So we can directly call the shared validation logic.
    processor03_validate();
    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata =
{
    "processor03",
    "processor03 example with FPS",
    CLICMD_FIELDS_DEFAULTS
};

static errno_t help_function() {
    return RETURN_SUCCESS;
}

/**
 * @brief Compute function called by FPSRUNfunction macro.
 */
static errno_t compute_function() {
    DEBUG_TRACE_FSTART();
    
    // Resolve input image (pointer set by CLI arg binding)
    IMGID inimg = mkIMGID_from_name(in_name_ptr);
    resolveIMGID(&inimg, ERRMODE_ABORT);
    
    // Create output image
    // Note: standalone uses createIm_gpu with explicit params.
    // Here we use standard CLIcore creation to register in image list
    IMGID outimg = mkIMGID_from_name(out_name_ptr);
    outimg.naxis = 2;
    outimg.size[0] = *roi_size_ptr;
    outimg.size[1] = *roi_size_ptr;
    outimg.datatype = _DATATYPE_FLOAT;
    imcreateIMGID(&outimg);
    
    // Standard ProcessInfo Loop Start
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    
    // Execute Shared Computation Logic
    processor03_compute(data.fpsptr, processinfo, inimg.im, outimg.im);
    
    // Standard Output Update
    processinfo_update_output_stream(processinfo, outimg.im, inimg.im);
    
    // Standard ProcessInfo Loop End
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// Macro to generate FPSCLI functions (FPSCONF, FPSRUN, FPSCLI)
INSERT_STD_FPSCLIfunctions

// Register the command with CLI
errno_t CLIADDCMD_processor03() {
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

// Module Initialization
INIT_MODULE_LIB(processor03)

static errno_t init_module_CLI() {
    CLIADDCMD_processor03();
    return RETURN_SUCCESS;
}

#endif
