/**
 * @file milk-example-03-processor.c
 * @brief Integration of ImageStreamIO, libprocessinfo, and libfps.
 *
 * This program demonstrates the full suite of milk standalone libraries:
 * 1. libfps: Manages configurable parameters in shared memory.
 * 2. libprocessinfo: Provides process monitoring and lifecycle control.
 * 3. ImageStreamIO: High-performance data streaming.
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

#define FPS_NAME "processor03"

/**
 * @brief Performs one-time setup of the Function Parameter Structure (FPS).
 * This creates the shared memory segment and initializes default values.
 */
int FPSINIT_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Initializing FPS '%s'...\n", FPS_NAME);

    // Create the FPS entry
    fps = function_parameter_FPCONFsetup(FPS_NAME, FPSCMDCODE_FPSINIT);

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
    // These appear in the FPS tools (e.g., milk-fpsCTRL)
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
void handle_tmux(const char *command, int argc, char *argv[]) {
    char cmd[2048];

    // Check if tmux is installed
    if (system("command -v tmux > /dev/null 2>&1") != 0) {
        fprintf(stderr, "\nError: 'tmux' is not installed or not in PATH.\n");
        fprintf(stderr, "The -tmux option requires tmux to be installed on your system.\n\n");
        exit(EXIT_FAILURE);
    }

    // Check if session exists
    int ret = system("tmux has-session -t " FPS_NAME " 2>/dev/null");
    if (ret != 0) {
        printf("Creating tmux session '%s'\n", FPS_NAME);
        system("tmux new-session -d -s " FPS_NAME " -n ctrl");
        system("tmux new-window -t " FPS_NAME " -n conf");
        system("tmux new-window -t " FPS_NAME " -n run");
        sleep(1); // Wait for shells
    }

    char path[1024];
    ssize_t path_len = readlink("/proc/self/exe", path, sizeof(path)-1);
    if (path_len != -1) {
        path[path_len] = '\0';
    } else {
        if (realpath(argv[0], path) == NULL) strncpy(path, argv[0], 1023);
    }

    if (strcmp(command, "confstart") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t " FPS_NAME ":conf \"%s confstart\" C-m", path);
        system(cmd);
        printf("Dispatched 'confstart' to tmux window " FPS_NAME ":conf\n");
    } else if (strcmp(command, "confstep") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t " FPS_NAME ":conf \"%s confstep\" C-m", path);
        system(cmd);
        printf("Dispatched 'confstep' to tmux window " FPS_NAME ":conf\n");
    } else if (strcmp(command, "runstart") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t " FPS_NAME ":run \"%s runstart\" C-m", path);
        system(cmd);
        printf("Dispatched 'runstart' to tmux window " FPS_NAME ":run\n");
    } else if (strcmp(command, "fpsinit") == 0) {
        FPSINIT_processor();
    }
}



/**
 * @brief The Configuration process.
 * This process typically runs in the background and validates parameter changes.
 * @param loop If non-zero, run an infinite monitoring loop.
 */
int FPSCONF_processor(int loop) {
    FUNCTION_PARAMETER_STRUCT fps;

    if (loop) {
        printf("Starting configuration process loop for '%s'\n", FPS_NAME);
        // Connect as configuration owner and set the loop bit
        fps = function_parameter_FPCONFsetup(FPS_NAME, FPSCMDCODE_CONFSTART);

        // Monitoring loop for validation
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) {
            if (function_parameter_FPCONFloopstep(&fps)) {
                // Logic to validate dependencies between parameters would go here
                // For example, ensuring off_x + roi_size < stream_width
            }
            usleep(10000);
        }
    } else {
        printf("Running single configuration step for '%s'\n", FPS_NAME);
        // Connect without setting the loop bit
        fps = function_parameter_FPCONFsetup(FPS_NAME, FPSCMDCODE_FPSINIT);
        function_parameter_FPCONFloopstep(&fps);
    }

    function_parameter_FPCONFexit(&fps);
    return 0;
}


/**
 * @brief Stop the configuration process.
 */
int FPSCONFSTOP_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Stopping configuration process for '%s'\n", FPS_NAME);

    if (function_parameter_struct_connect(FPS_NAME, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found.\n", FPS_NAME);
        return 1;
    }

    functionparameter_CONFstop(&fps);
    function_parameter_struct_disconnect(&fps);

    return 0;
}


/**
 * @brief Stop the run process.
 */
int FPSRUNSTOP_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Stopping run process for '%s'\n", FPS_NAME);

    if (function_parameter_struct_connect(FPS_NAME, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found.\n", FPS_NAME);
        return 1;
    }

    // 1. Signal through FPS
    functionparameter_RUNstop(&fps);
    function_parameter_struct_disconnect(&fps);

    // 2. Signal through processinfo shared memory
    char procdname[STRINGMAXLEN_DIR_NAME];
    processinfo_procdirname(procdname);

    DIR *d = opendir(procdname);
    if (d) {
        struct dirent *dir;
        while ((dir = readdir(d)) != NULL) {
            // Looking for proc.processor03.<PID>.shm
            if (strncmp(dir->d_name, "proc." FPS_NAME ".", 5 + strlen(FPS_NAME) + 1) == 0) {
                char fullpath[1024];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, dir->d_name);
                int fd;
                PROCESSINFO *pinfo = processinfo_shm_link(fullpath, &fd);
                if (pinfo != (PROCESSINFO *)MAP_FAILED) {
                    printf("Signaling process PID %d to exit...\n", (int)pinfo->PID);
                    pinfo->CTRLval = 3; // Request exit
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
 * @brief The main Processing loop.
 * Reads data, processes it according to FPS parameters, and reports status.
 */
int FPSRUN_processor() {
    FUNCTION_PARAMETER_STRUCT fps;

    // 1. CONNECT TO FPS
    if (function_parameter_struct_connect(FPS_NAME, &fps, FPSCONNECT_RUN) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found. Run 'fpsinit' first.\n", FPS_NAME);
        return 1;
    }

    // 2. RETRIEVE CURRENT PARAMETERS
    char in_name[200];
    char out_name[200];
    strncpy(in_name, functionparameter_GetParamPtr_STRING(&fps, ".in_name"), 199);
    strncpy(out_name, functionparameter_GetParamPtr_STRING(&fps, ".out_name"), 199);
    uint32_t roi_size = functionparameter_GetParamValue_UINT32(&fps, ".roi_size");
    uint32_t off_x = functionparameter_GetParamValue_UINT32(&fps, ".off_x");

    // 3. INITIALIZE STREAMS
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &input_image) != 0) {
        fprintf(stderr, "Error connecting to input %s\n", in_name);
        return 1;
    }

    IMAGE output_image;
    uint32_t dims[2] = {roi_size, roi_size};
    if (ImageStreamIO_createIm_gpu(&output_image, out_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != 0) {
        return 1;
    }

    // 4. SETUP PROCESS MONITORING
    PROCESSINFO *processinfo = processinfo_setup(FPS_NAME, "Ex03 Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    if (!processinfo) return 1;

    // Capture SIGINT and other signals
    processinfo_CatchSignals();

    // Use current FPS settings to configure ProcessInfo triggers/priority
    processinfo_waitoninputstream_init(processinfo, &input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, processinfo);

    processinfo_loopstart(processinfo);

    float *in_data = (float*)input_image.array.raw;
    float *out_data = (float*)output_image.array.raw;
    uint32_t in_w = input_image.md[0].size[0];

    // 5. MAIN LOOP
    int loopOK = 1;
    while(loopOK) {
        loopOK = processinfo_loopstep(processinfo);
        if(!loopOK) break;

        processinfo_waitoninputstream(processinfo);
        if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;

        processinfo_exec_start(processinfo);

        // RE-READ PARAMETERS DYNAMICALLY
        // This allows real-time adjustment of processing logic (e.g., changing off_x via TUI)
        fps_to_processinfo(&fps, processinfo);
        off_x = functionparameter_GetParamValue_UINT32(&fps, ".off_x");

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


        processinfo_exec_end(processinfo);
        processinfo_update_output_stream(processinfo, &output_image, &input_image);
    }

    // Cleanup
    processinfo_cleanExit(processinfo);
    function_parameter_struct_disconnect(&fps);
    return 0;
}



/**
 * @brief Main dispatcher for Example 03.
 */
int main(int argc, char *argv[]) {
    // Basic argument check
    if (argc < 2) {
        printf("Usage: %s <fpsinit|confstart|confstep|confstop|runstart|runstop> [-tmux]\n", "milk-example-03-processor");
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
        printf("  -tmux    Auto-create a tmux session named '%s' and dispatch commands.\n", FPS_NAME);
        printf("           - 'confstart' command goes to window 1.\n");
        printf("           - 'runstart' command goes to window 2.\n");
        printf("           - 'ctrl' window (index 0) remains for user interaction.\n\n");
        printf("Typical Workflow:\n");
        printf("  1. Terminal A: ./milk-example-03-writer\n");
        printf("  2. Terminal B: ./milk-example-03-processor fpsinit\n");
        printf("  3. Terminal B: ./milk-example-03-processor runstart -tmux\n");
        printf("  4. Terminal B: ./milk-example-03-processor confstart -tmux\n");
        printf("  5. Connect:    tmux a -t %s\n\n", FPS_NAME);
        return 0;
    }

    // Check for -tmux flag
    int use_tmux = 0;
    char *command = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-tmux") == 0) {
            use_tmux = 1;
        } else if (command == NULL) {
            command = argv[i];
        }
    }

    if (command == NULL) {
        fprintf(stderr, "Error: Missing command argument.\n");
        return 1;
    }

    // Dispatching
    if (use_tmux) {
        handle_tmux(command, argc, argv);
        return 0;
    }

    if (strcmp(command, "fpsinit") == 0) {
        return FPSINIT_processor();
    } else if (strcmp(command, "confstart") == 0) {
        return FPSCONF_processor(1);
    } else if (strcmp(command, "confstep") == 0) {
        return FPSCONF_processor(0);
    } else if (strcmp(command, "confstop") == 0) {
        return FPSCONFSTOP_processor();
    } else if (strcmp(command, "runstart") == 0) {
        return FPSRUN_processor();
    } else if (strcmp(command, "runstop") == 0) {
        return FPSRUNSTOP_processor();
    }

    fprintf(stderr, "Invalid command: %s\n", command);
    return 1;
}