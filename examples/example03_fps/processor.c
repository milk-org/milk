#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "fps.h" // Main FPS header
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

// Define Global Data struct required by FPS/CLIcore macros if using them, 
// or manually handle it.
// The standalone libfps does NOT require the big DATA struct from CLIcore.
// But we need a local fps variable.

// FPS Configuration Function
int FPSCONF_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    
    // Create FPS
    // Use low-level function or macro if available. 
    // Here we use function_parameter_FPCONFsetup directly.
    fps = function_parameter_FPCONFsetup("processor03", FPSCMDCODE_FPSINIT);

    // Add Parameters
    
    // Input Stream Name
    char *in_name = "stream03";
    function_parameter_add_entry(&fps, ".in_name", "Input Stream Name", FPTYPE_STRING, FPFLAG_DEFAULT_INPUT, (void*)in_name, NULL);

    // Output Stream Name
    char *out_name = "stream03_proc";
    function_parameter_add_entry(&fps, ".out_name", "Output Stream Name", FPTYPE_STRING, FPFLAG_DEFAULT_INPUT, (void*)out_name, NULL);

    // ROI Size
    uint32_t roi_size = 50;
    function_parameter_add_entry(&fps, ".roi_size", "ROI Size", FPTYPE_UINT32, FPFLAG_DEFAULT_INPUT, (void*)&roi_size, NULL);

    // ROI Offsets
    uint32_t off_x = 0;
    function_parameter_add_entry(&fps, ".off_x", "Offset X", FPTYPE_UINT32, FPFLAG_DEFAULT_INPUT, (void*)&off_x, NULL);
    
    // Add ProcessInfo standard parameters (triggers, etc.)
    // Note: fps_add_processinfo_entries is in libfps if compiled with it, or we add manually.
    // Assuming we linked with libfps which has this utility or we do it manually.
    // For standalone, let's add minimal trigger params manually to show how it's done.
    
    // Configuration Loop
    while (function_parameter_FPCONFloopstep(&fps)) {
        // Logic to validate parameters can go here
        usleep(10000);
    }

    function_parameter_FPCONFexit(&fps);
    return 0;
}

// FPS Run Function
int FPSRUN_processor() {
    FUNCTION_PARAMETER_STRUCT fps;
    
    // Connect to FPS
    if (function_parameter_struct_connect("processor03", &fps, FPSCONNECT_RUN) == -1) {
        fprintf(stderr, "Error connecting to FPS\n");
        return 1;
    }

    // Get Parameter Values
    char in_name[200];
    char out_name[200];
    uint32_t roi_size;
    uint32_t off_x;

    strncpy(in_name, functionparameter_GetParamPtr_STRING(&fps, ".in_name"), 199);
    strncpy(out_name, functionparameter_GetParamPtr_STRING(&fps, ".out_name"), 199);
    roi_size = functionparameter_GetParamValue_UINT32(&fps, ".roi_size");
    off_x = functionparameter_GetParamValue_UINT32(&fps, ".off_x");

    // Initialize ImageStreamIO
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &input_image) != 0) {
        fprintf(stderr, "Error connecting to %s\n", in_name);
        return 1;
    }

    IMAGE output_image;
    uint32_t dims[2] = {roi_size, roi_size};
    ImageStreamIO_createIm_gpu(&output_image, out_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);

    // Initialize ProcessInfo
    PROCESSINFO *processinfo = processinfo_setup(
        "proc_ex03", 
        "Example 03 Processor", 
        "Starting", 
        __FUNCTION__, __FILE__, __LINE__
    );

    // Setup Trigger
    processinfo_waitoninputstream_init(processinfo, &input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    processinfo_loopstart(processinfo);

    float *in_data = (float*)input_image.array.raw;
    float *out_data = (float*)output_image.array.raw;
    uint32_t in_w = input_image.md[0].size[0];

    // Processing Loop
    int loopOK = 1;
    while(loopOK) {
        loopOK = processinfo_loopstep(processinfo);
        if(!loopOK) break;

        processinfo_waitoninputstream(processinfo);
        processinfo_exec_start(processinfo);

        output_image.md[0].write = 1;
        // Simple Crop & Sum (simplified for example)
        // Using parameter off_x for dynamic control
        // Re-read parameters if needed (usually done at check/update signal)
        
        // Note: In a real FPS loop, you check for updates.
        // Here we just run.
        
        for(uint32_t y=0; y<roi_size; y++) {
            for(uint32_t x=0; x<roi_size; x++) {
                // Just one ROI for this simple fps demo, shifted by off_x
                if (x + off_x < in_w) {
                     out_data[y*roi_size + x] = in_data[y*in_w + (x + off_x)];
                } else {
                     out_data[y*roi_size + x] = 0;
                }
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
    // Simple command line dispatcher
    if(argc > 1) {
        if(strcmp(argv[1], "conf") == 0) {
            return FPSCONF_processor();
        }
        if(strcmp(argv[1], "run") == 0) {
            return FPSRUN_processor();
        }
    }
    
    printf("Usage: %s [conf|run]\n", argv[0]);
    return 0;
}
