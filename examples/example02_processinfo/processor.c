#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "processinfo.h"
#include "processtools_trigger.h"
#include "processinfo_update_output_stream.h"
#include "processinfo_setup.h"
#include "processinfo_loopstep.h"
#include "processinfo_exec_start.h"
#include "processinfo_exec_end.h"
#include "processinfo_signals.h"
#include "ImageStreamIO.h"

int main() {
    const char *in_name = "stream02";
    const char *out_name = "stream02_proc";

    // 1. Connect Input
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &input_image) != 0) {
        fprintf(stderr, "Error: Could not connect to '%s'.\n", in_name);
        return 1;
    }

    // 2. Create Output
    uint32_t roi_w = 50;
    uint32_t roi_h = 50;
    uint32_t dims[2] = {roi_w, roi_h};
    IMAGE output_image;
    ImageStreamIO_createIm_gpu(&output_image, out_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);

    // 3. ProcessInfo Setup
    PROCESSINFO *processinfo = processinfo_setup(
        "proc_ex02",            // Process name
        "Example 02 Processor", // Description
        "Starting...",          // Msg
        __FUNCTION__, __FILE__, __LINE__
    );

    // Trigger configuration
    processinfo_waitoninputstream_init(processinfo, &input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    
    // Loop start
    processinfo_loopstart(processinfo);

    float *in_data = (float*)input_image.array.raw;
    float *out_data = (float*)output_image.array.raw;
    uint32_t in_w = input_image.md[0].size[0];
    int off_x[4] = {0, 150, 0, 150};
    int off_y[4] = {0, 0, 150, 150};

    int loopOK = 1;
    while(loopOK) {
        // Handle Loop Control
        loopOK = processinfo_loopstep(processinfo);
        if(!loopOK) break;

        // Wait for Trigger
        processinfo_waitoninputstream(processinfo);

        output_image.md[0].write = 1;
        // Exec Start (Timing)
        processinfo_exec_start(processinfo);

        // Computation
        for(uint32_t y=0; y<roi_h; y++) {
            for(uint32_t x=0; x<roi_w; x++) {
                float sum = 0.0;
                for(int k=0; k<4; k++) {
                    sum += in_data[(off_y[k] + y)*in_w + (off_x[k] + x)];
                }
                out_data[y*roi_w + x] = sum;
            }
        }

        // Exec End (Timing, Signals) & Update Output
        processinfo_exec_end(processinfo);
        processinfo_update_output_stream(processinfo, &output_image, &input_image);
    }

    // Cleanup
    processinfo_cleanExit(processinfo);
    
    return 0;
}
