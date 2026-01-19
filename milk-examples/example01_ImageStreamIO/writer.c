/**
 * @file milk-example-01-writer.c
 * @brief Basic ImageStreamIO writer example.
 *
 * This program demonstrates how to:
 * 1. Create a shared memory stream using ImageStreamIO.
 * 2. Set the 'write' flag to indicate data is being modified.
 * 3. Update the stream and post semaphores to notify consumers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include "ImageStruct.h"
#include "ImageStreamIO.h"

// Static flag for graceful exit on CTRL+C
static int keep_running = 1;

/**
 * @brief Signal handler for SIGINT (CTRL+C)
 */
void signal_handler(int sig) {
    printf("\nWriter: Exiting on CTRL+C.\n");
    exit(0);
}

/**
 * @brief Main entry point for the writer example.
 */
int main(int argc, char *argv[]) {
    // ------------------------------------------------------------------------
    // 1. HELP MESSAGE
    // ------------------------------------------------------------------------
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("\nUsage: %s\n\n", "milk-example-01-writer");
        printf("Description:\n");
        printf("  This program demonstrates basic shared memory stream creation and writing.\n");
        printf("  It uses the ImageStreamIO library to manage high-speed data transfer.\n\n");
        printf("Operation:\n");
        printf("  1. Creates a 200x200 FLOAT shared memory stream named 'stream01'.\n");
        printf("  2. Enters a loop (100Hz) generating a moving sine/cosine gradient.\n");
        printf("  3. Sets the metadata 'write' flag to 1 while writing data.\n");
        printf("  4. Calls ImageStreamIO_UpdateIm() to increment counters and post semaphores.\n\n");
        printf("Environment Variables:\n");
        printf("  MILK_SHM_DIR  Location of shared memory files (defaults to /milk/shm).\n\n");
        return 0;
    }

    // Register signal handler for clean shutdown
    signal(SIGINT, signal_handler);

    // ------------------------------------------------------------------------
    // 2. STREAM INITIALIZATION
    // ------------------------------------------------------------------------
    const char *stream_name = "stream01";
    uint32_t width = 200;
    uint32_t height = 200;
    uint32_t dims[2] = {width, height};
    
    // Create the stream in shared memory.
    // Parameters: structure, name, naxis, dims, type, device_id (-1 for CPU), 
    //             shared (1), NB_sem (10), NB_kw (0), imagetype (0), CBsize (0)
    IMAGE image;
    if (ImageStreamIO_createIm_gpu(&image, stream_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != IMAGESTREAMIO_SUCCESS) {
        fprintf(stderr, "Error: Could not create stream '%s'\n", stream_name);
        return 1;
    }

    printf("Writer: Stream '%s' created at 200x200. Data address: %p\n", stream_name, image.array.raw);
    printf("Press CTRL+C to stop.\n");

    // Local pointer to the shared memory data array
    float *data = (float*)image.array.raw;
    uint64_t counter = 0;

    // ------------------------------------------------------------------------
    // 3. MAIN LOOP
    // ------------------------------------------------------------------------
    while(keep_running) {
        // Set 'write' flag to 1 to notify consumers that data is currently unstable
        image.md[0].write = 1;

        // Generate a simple moving pattern
        for(uint32_t y=0; y<height; y++) {
            for(uint32_t x=0; x<width; x++) {
                data[y*width + x] = 0.5 * sin((x + counter)*0.1) + 0.5 * cos((y + counter)*0.05);
            }
        }

        // Post updates:
        // - Sets write flag back to 0
        // - Increments internal update counter (cnt0)
        // - Updates timestamps
        // - Posts all semaphores to wake up consumers
        ImageStreamIO_UpdateIm(&image);

        counter++;
        usleep(10000); // Wait 10ms (approx 100Hz)
    }

    printf("\nWriter: Cleaning up...\n");
    // In a production environment, you might use ImageStreamIO_destroyIm()
    // if this process is the exclusive owner.
    
    return 0;
}