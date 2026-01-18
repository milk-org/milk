#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "fps.h"
#include "fps_add_entry.h"
#include "fps_paramvalue.h"
#include "fps_FPCONFsetup.h"
#include "fps_FPCONFloopstep.h"
#include "fps_FPCONFexit.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_RUNexit.h"

// For processinfo integration
#include "processinfo.h"
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

// Helper to handle tmux logic
void handle_tmux(const char *arg, int argc, char *argv[]) {
    char cmd[1024];
    // Check if session exists
    int ret = system("tmux has-session -t " FPS_NAME " 2>/dev/null");
    if (ret != 0) {
        printf("Creating tmux session '%s'...\n", FPS_NAME);
        system("tmux new-session -d -s " FPS_NAME " -n ctrl");
        system("tmux new-window -t " FPS_NAME ":1 -n conf");
        system("tmux new-window -t " FPS_NAME ":2 -n run");
    }

    // Get absolute path of this executable
    char path[1024];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path)-1);
    if (len != -1) {
        path[len] = '\0';
    } else {
        strncpy(path, argv[0], 1023);
    }

    if (strcmp(arg, "conf") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t " FPS_NAME ":conf \"%s conf\" C-m", path);
        system(cmd);
        printf("Launched 'conf' in tmux window " FPS_NAME ":conf\n");
    } else if (strcmp(arg, "run") == 0) {
        snprintf(cmd, sizeof(cmd), "tmux send-keys -t " FPS_NAME ":run \"%s run\" C-m", path);
        system(cmd);
        printf("Launched 'run' in tmux window " FPS_NAME ":run\n");
    }
}

// FPS Initialization
int FPSINIT_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Initializing FPS '%s'...\n", FPS_NAME);
    fps = function_parameter_FPCONFsetup(FPS_NAME, FPSCMDCODE_FPSINIT);
    
    // Default values
    strncpy(fps.cmdset.triggerstreamname, "stream03", STRINGMAXLEN_IMAGE_NAME - 1);
    fps.cmdset.procinfo_loopcntMax = -1;
    fps.cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
    fps.cmdset.triggertimeout.tv_sec = 10;
    fps.cmdset.triggertimeout.tv_nsec = 0;

    // Add entries
    char *in_name = "stream03";
    function_parameter_add_entry(&fps, ".in_name", "Input Stream Name", FPTYPE_STRING, FPFLAG_DEFAULT_INPUT, (void*)in_name, NULL);
    char *out_name = "stream03_proc";
    function_parameter_add_entry(&fps, ".out_name", "Output Stream Name", FPTYPE_STRING, FPFLAG_DEFAULT_INPUT, (void*)out_name, NULL);
    uint32_t roi_size = 50;
    function_parameter_add_entry(&fps, ".roi_size", "ROI Size", FPTYPE_UINT32, FPFLAG_DEFAULT_INPUT, (void*)&roi_size, NULL);
    uint32_t off_x = 0;
    function_parameter_add_entry(&fps, ".off_x", "Offset X", FPTYPE_UINT32, FPFLAG_DEFAULT_INPUT, (void*)&off_x, NULL);
    
    fps_add_processinfo_entries(&fps);
    function_parameter_FPCONFexit(&fps);
    return 0;
}

// FPS Configuration Process
int FPSCONF_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    printf("Starting configuration process for '%s'...\n", FPS_NAME);
    fps = function_parameter_FPCONFsetup(FPS_NAME, FPSCMDCODE_CONFSTART);

    while (function_parameter_FPCONFloopstep(&fps)) {
        usleep(100000); // 10Hz check
    }

    function_parameter_FPCONFexit(&fps);
    return 0;
}

// FPS Run Process
int FPSRUN_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    if (function_parameter_struct_connect(FPS_NAME, &fps, FPSCONNECT_RUN) == -1) {
        fprintf(stderr, "Error: FPS '%s' not found. Run 'fpsinit' first.\n", FPS_NAME);
        return 1;
    }

    char in_name[200];
    char out_name[200];
    strncpy(in_name, functionparameter_GetParamPtr_STRING(&fps, ".in_name"), 199);
    strncpy(out_name, functionparameter_GetParamPtr_STRING(&fps, ".out_name"), 199);
    uint32_t roi_size = functionparameter_GetParamValue_UINT32(&fps, ".roi_size");
    uint32_t off_x = functionparameter_GetParamValue_UINT32(&fps, ".off_x");

    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &input_image) != 0) {
        fprintf(stderr, "Error connecting to %s\n", in_name);
        return 1;
    }

    IMAGE output_image;
    uint32_t dims[2] = {roi_size, roi_size};
    ImageStreamIO_createIm_gpu(&output_image, out_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);

    PROCESSINFO *processinfo = processinfo_setup(FPS_NAME, "Ex03 Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    if (!processinfo) return 1;

    processinfo_waitoninputstream_init(processinfo, &input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, processinfo);
    processinfo_loopstart(processinfo);

    float *in_data = (float*)input_image.array.raw;
    float *out_data = (float*)output_image.array.raw;
    uint32_t in_w = input_image.md[0].size[0];

    int loopOK = 1;
    while(loopOK) {
        loopOK = processinfo_loopstep(processinfo);
        if(!loopOK) break;
        processinfo_waitoninputstream(processinfo);
        if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(processinfo);

        output_image.md[0].write = 1;
        // Dynamically re-read offset from FPS
        off_x = functionparameter_GetParamValue_UINT32(&fps, ".off_x");
        
        for(uint32_t y=0; y<roi_size; y++) {
            for(uint32_t x=0; x<roi_size; x++) {
                if (x + off_x < in_w) 
                    out_data[y*roi_size + x] = in_data[y*in_w + (x + off_x)];
                else 
                    out_data[y*roi_size + x] = 0;
            }
        }
        processinfo_exec_end(processinfo);
        processinfo_update_output_stream(processinfo, &output_image, &input_image);
    }

    processinfo_cleanExit(processinfo);
    function_parameter_struct_disconnect(&fps);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <fpsinit|conf|run> [-tmux]\n", argv[0]);
        return 1;
    }

    int use_tmux = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-tmux") == 0) use_tmux = 1;
    }

    if (strcmp(argv[1], "fpsinit") == 0) {
        return FPSINIT_processor();
    } 
    
    if (use_tmux) {
        handle_tmux(argv[1], argc, argv);
        return 0;
    }

    if (strcmp(argv[1], "conf") == 0) {
        return FPSCONF_processor();
    } else if (strcmp(argv[1], "run") == 0) {
        return FPSRUN_processor();
    }

    printf("Invalid argument: %s\n", argv[1]);
    return 1;
}