/**
 * @file milk-example-02-processor.c
 * @brief Integration of ImageStreamIO with libprocessinfo.
 *
 * This program demonstrates how to wrap a processing loop with the
 * milk process monitoring system. This allows the process to be:
 * 1. Listed in the 'milk-procCTRL' or TUI tools.
 * 2. Externally paused, resumed, or stepped.
 * 3. Profiled for execution and iteration timing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

// Standard processinfo headers from libprocessinfo
#include "processinfo.h"
#include "processtools_trigger.h"
#include "processinfo_update_output_stream.h"
#include "processinfo_setup.h"
#include "processinfo_loopstep.h"
#include "processinfo_exec_start.h"
#include "processinfo_exec_end.h"
#include "processinfo_signals.h"

#include "ImageStreamIO.h"

/**
 * @brief Main entry point for the processinfo integration example.
 */
int main(int argc, char *argv[])
{
    // ------------------------------------------------------------------------
    // 1. HELP MESSAGE
    // ------------------------------------------------------------------------
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))
    {
        printf("\nUsage: %s\n\n", "milk-example-02-processor");
        printf("Description:\n");
        printf(
            "  This program performs ROI processing and registers itself with libprocessinfo.\n");
        printf("  It demonstrates how to use the 'processinfo' structure for lifecycle "
               "management.\n\n");
        printf("ProcessInfo Features demonstrated:\n");
        printf("  - Registration: Appears in milk process lists ('proc_ex02').\n");
        printf("  - Signals: Handles SIGINT/SIGTERM gracefully via processinfo_ProcessSignals.\n");
        printf(
            "  - Triggering: Waits on input stream semaphores with built-in timeout handling.\n");
        printf("  - Monitoring: Reports iteration timing and status messages.\n\n");
        printf("Requirements:\n");
        printf("  The 'milk-example-02-writer' must be running to provide 'stream02'.\n\n");
        return 0;
    }

    const char *in_name  = "stream02";
    const char *out_name = "stream02_proc";

    // ------------------------------------------------------------------------
    // 2. CONNECT TO IMAGE STREAMS
    // ------------------------------------------------------------------------
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &input_image) != 0)
    {
        fprintf(stderr, "Error: Could not connect to '%s'.\n", in_name);
        return 1;
    }

    uint32_t roi_w   = 50;
    uint32_t roi_h   = 50;
    uint32_t dims[2] = { roi_w, roi_h };
    IMAGE    output_image;
    if (ImageStreamIO_createIm_gpu(&output_image, out_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0,
                                   0, 0) != 0)
    {
        fprintf(stderr, "Error: Could not create output stream.\n");
        return 1;
    }

    // ------------------------------------------------------------------------
    // 3. PROCESSINFO SETUP
    // ------------------------------------------------------------------------
    // Registers the process in shared memory (/tmp/proc.proc_ex02.PID.shm)
    PROCESSINFO *processinfo = processinfo_setup("proc_ex02", // Process name (for monitoring)
                                                 "Example 02 Processor", // Description
                                                 "Starting...",          // Initial status message
                                                 __FUNCTION__, __FILE__, __LINE__);
    if (processinfo == NULL)
    {
        fprintf(stderr, "Error: processinfo_setup failed\n");
        return 1;
    }

    // Capture SIGINT and other signals
    processinfo_CatchSignals();

    // Trigger configuration

    // Initialize the trigger mechanism to wait on the input image's semaphore.
    // Default mode: PROCESSINFO_TRIGGERMODE_SEMAPHORE.
    processinfo_waitoninputstream_init(processinfo, &input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE,
                                       -1);

    // Notify the system that the initialization is complete and the loop is starting.
    processinfo_loopstart(processinfo);

    // ------------------------------------------------------------------------
    // 4. DATA PREPARATION
    // ------------------------------------------------------------------------
    float   *in_data  = (float *) input_image.array.raw;
    float   *out_data = (float *) output_image.array.raw;
    uint32_t in_w     = input_image.md[0].size[0];
    int      off_x[4] = { 0, 150, 0, 150 };
    int      off_y[4] = { 0, 0, 150, 150 };

    // ------------------------------------------------------------------------
    // 5. MAIN CONTROLLED LOOP
    // ------------------------------------------------------------------------
    int loopOK = 1;
    while (loopOK)
    {
        // [A] Lifecycle Step: Check if external tools requested a pause or exit.
        loopOK = processinfo_loopstep(processinfo);
        if (!loopOK)
        {
            break;
        }

        // [B] Synchronization Step: Wait for the next input frame.
        processinfo_waitoninputstream(processinfo);

        // If the wait timed out (default 2s), skip processing and try again.
        if (processinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)
        {
            continue;
        }

        // [C] Execution Start: Record timestamps for performance profiling.
        processinfo_exec_start(processinfo);

        // [D] Work Step
        output_image.md[0].write = 1;
        for (uint32_t y = 0; y < roi_h; y++)
        {
            for (uint32_t x = 0; x < roi_w; x++)
            {
                float sum = 0.0;
                for (int k = 0; k < 4; k++)
                {
                    sum += in_data[(off_y[k] + y) * in_w + (off_x[k] + x)];
                }
                out_data[y * roi_w + x] = sum;
            }
        }

        // [E] Execution End: Calculate compute time, check for OS signals, and increment loop counter.
        processinfo_exec_end(processinfo);

        // [F] Update Output: Post results and update stream metadata (PID, timestamps, dependency trace).
        processinfo_update_output_stream(processinfo, &output_image, &input_image);
    }

    // ------------------------------------------------------------------------
    // 6. CLEAN EXIT
    // ------------------------------------------------------------------------
    // Unregisters the process and removes its shared memory entry.
    processinfo_cleanExit(processinfo);

    return 0;
}
