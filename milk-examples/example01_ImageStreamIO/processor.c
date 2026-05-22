/**
 * @file milk-example-01-processor.c
 * @brief Basic ImageStreamIO consumer/processor example.
 *
 * This program demonstrates how to:
 * 1. Connect to an existing shared memory stream.
 * 2. Wait for updates using semaphores.
 * 3. Perform a simple ROI crop and sum operation.
 * 4. Write results to a new output stream.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include "ImageStruct.h"
#include "ImageStreamIO.h"

// Static flag for graceful exit
static int keep_running = 1;

/**
 * @brief Signal handler for SIGINT (CTRL+C)
 */
void signal_handler(int sig)
{
    keep_running = 0;
}

/**
 * @brief Main entry point for the processor example.
 */
int main(int argc, char *argv[])
{
    // ------------------------------------------------------------------------
    // 1. HELP MESSAGE
    // ------------------------------------------------------------------------
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))
    {
        printf("\nUsage: %s\n\n", "milk-example-01-processor");
        printf("Description:\n");
        printf("  This program consumes 'stream01' and produces 'stream01_proc'.\n");
        printf("  It demonstrates high-performance stream synchronization using semaphores.\n\n");
        printf("Operation:\n");
        printf("  1. Connects to the existing input stream 'stream01'.\n");
        printf("  2. Creates a 50x50 output stream named 'stream01_proc'.\n");
        printf("  3. Waits on semaphore #0 of the input stream for new data.\n");
        printf("  4. Crops four 50x50 corner regions from the 200x200 input and sums them.\n");
        printf("  5. Writes the resulting summed image to the output stream.\n\n");
        printf("Requirements:\n");
        printf("  The 'milk-example-01-writer' must be running to provide the input stream.\n\n");
        return 0;
    }

    signal(SIGINT, signal_handler);

    const char *in_name  = "stream01";
    const char *out_name = "stream01_proc";

    // ------------------------------------------------------------------------
    // 2. CONNECT TO INPUT STREAM
    // ------------------------------------------------------------------------
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &input_image) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "Error: Could not connect to stream '%s'. Start writer first.\n", in_name);
        return 1;
    }

    // ------------------------------------------------------------------------
    // 3. CREATE OUTPUT STREAM
    // ------------------------------------------------------------------------
    uint32_t roi_w   = 50;
    uint32_t roi_h   = 50;
    uint32_t dims[2] = { roi_w, roi_h };
    IMAGE    output_image;
    if (ImageStreamIO_createIm_gpu(&output_image, out_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0,
                                   0, 0) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "Error: Could not create output stream '%s'\n", out_name);
        return 1;
    }

    printf("Processor: Monitoring '%s' (%dx%d) -> Producing '%s' (%dx%d)\n", in_name,
           (int) input_image.md[0].size[0], (int) input_image.md[0].size[1], out_name, roi_w,
           roi_h);
    printf("Press CTRL+C to stop.\n");

    // Local pointers to the shared memory data
    float   *in_data  = (float *) input_image.array.raw;
    float   *out_data = (float *) output_image.array.raw;
    uint32_t in_w     = input_image.md[0].size[0];

    // Offsets for 4 corner ROIs (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    int off_x[4] = { 0, 150, 0, 150 };
    int off_y[4] = { 0, 0, 150, 150 };

    // ------------------------------------------------------------------------
    // 4. MAIN PROCESSING LOOP
    // ------------------------------------------------------------------------
    while (keep_running)
    {
        // Blocks until the writer calls ImageStreamIO_UpdateIm on the input stream.
        // We use semaphore #0.
        ImageStreamIO_semwait(&input_image, 0);

        // Notify that we are writing to the output
        output_image.md[0].write = 1;

        // ROI SUM OPERATION
        // For each pixel in the 50x50 output, sum pixels from 4 corners of input
        for (uint32_t y = 0; y < roi_h; y++)
        {
            for (uint32_t x = 0; x < roi_w; x++)
            {
                float sum = 0.0;
                for (int k = 0; k < 4; k++)
                {
                    uint32_t ix = off_x[k] + x;
                    uint32_t iy = off_y[k] + y;
                    sum += in_data[iy * in_w + ix];
                }
                out_data[y * roi_w + x] = sum;
            }
        }

        // Post results to subscribers of the output stream
        ImageStreamIO_UpdateIm(&output_image);
    }

    printf("\nProcessor: Cleaning up...\n");
    return 0;
}
