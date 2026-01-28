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
 * @brief The Configuration process.
 * This process typically runs in the background and validates parameter changes.
 * @param loop If non-zero, run an infinite monitoring loop.
 */
int FPSCONF_processor(const char *fps_name, int loop) {
    FUNCTION_PARAMETER_STRUCT fps;

    if (loop) {
        printf("Starting configuration process loop for '%s'\n", fps_name);
        // Connect as configuration owner and set the loop bit
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_CONFSTART);

        char *in_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
        uint32_t *roi_size_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
        uint32_t *off_x_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

        if (!in_name_ptr || !roi_size_ptr || !off_x_ptr) {
            fprintf(stderr, "Error: Could not retrieve parameter pointers.\n");
            function_parameter_FPCONFexit(&fps);
            return 1;
        }

        // Monitoring loop for validation
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) {
            if (function_parameter_FPCONFloopstep(&fps)) {
                // Logic to validate dependencies between parameters
                // For example, ensuring off_x + roi_size < stream_width

                IMAGE input_image;
                if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name_ptr, &input_image) == 0) {
                    uint32_t width = input_image.md[0].size[0];
                    if (*off_x_ptr + *roi_size_ptr > width) {
                        // Clamp off_x first
                        if (*off_x_ptr > width) {
                            *off_x_ptr = 0;
                        }
                        // Then adjust if still too large (meaning roi_size is big or off_x is pushing it)
                        if (*off_x_ptr + *roi_size_ptr > width) {
                            if (*roi_size_ptr > width) {
                                *roi_size_ptr = width;
                                *off_x_ptr = 0;
                            } else {
                                *off_x_ptr = width - *roi_size_ptr;
                            }
                        }
                    }
                    ImageStreamIO_closeIm(&input_image);
                }
            }
            usleep(10000);
        }
    } else {
        printf("Running single configuration step for '%s'\n", fps_name);
        // Connect without setting the loop bit
        fps = function_parameter_FPCONFsetup(fps_name, FPSCMDCODE_FPSINIT);
        function_parameter_FPCONFloopstep(&fps);
    }

    function_parameter_FPCONFexit(&fps);
    return 0;
}

/**
 * @brief Stop the configuration process.
 */
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


/**
 * @brief Stop the run process.
 */
int FPSRUNSTOP_processor(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Stopping run process for '%s'\n", fps_name);

    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found.\n", fps_name);
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
            // Looking for proc.<fps_name>.<PID>.shm
            char prefix[256];
            snprintf(prefix, sizeof(prefix), "proc.%s.", fps_name);
            if (strncmp(dir->d_name, prefix, strlen(prefix)) == 0) {
                char fullpath[2048];
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
int FPSRUN_processor(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps;

    // 1. CONNECT TO FPS
    if (function_parameter_struct_connect(fps_name, &fps, FPSCONNECT_RUN) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found. Run 'fpsinit' first.\n", fps_name);
        return 1;
    }

    // 2. RETRIEVE CURRENT PARAMETERS
    char *in_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
    char *out_name_ptr = functionparameter_GetParamPtr_STRING(&fps, ".out_name");
    uint32_t *roi_size_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".roi_size");
    uint32_t *off_x_ptr = functionparameter_GetParamPtr_UINT32(&fps, ".off_x");

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
        uint32_t off_x = *off_x_ptr;
        uint32_t roi_size = *roi_size_ptr;

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

    // Default settings
    char fps_name[STRINGMAXLEN_FPS_NAME] = "processor03";
    int use_tmux = 0;
    char *command = NULL;
    char *keywords = NULL;
    char *description = NULL;

    // Argument parsing
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

    // Dispatching
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