#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include "ImageStruct.h"
#include "ImageStreamIO.h"

static int keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
}

int main(int argc, char *argv[]) {
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("\nUsage: %s\n\n", argv[0]);
        printf("Description:\n");
        printf("  This program demonstrates the basic usage of ImageStreamIO for writing.\n");
        printf("  It creates a shared memory stream named 'stream01' with dimensions 200x200\n");
        printf("  and data type FLOAT. It then enters a loop writing a moving gradient\n");
        printf("  pattern at approximately 100Hz.\n\n");
        printf("Environment Variables:\n");
        printf("  MILK_SHM_DIR  Directory for shared memory files (default: /milk/shm)\n\n");
        return 0;
    }
    signal(SIGINT, signal_handler);

    const char *stream_name = "stream01";
    uint32_t width = 200;
    uint32_t height = 200;
    uint32_t dims[2] = {width, height};

    // 1. Create Stream
    IMAGE image;
    ImageStreamIO_createIm_gpu(&image, stream_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);

    printf("Writer: Stream '%s' created. Press CTRL+C to stop.\n", stream_name);

    float *data = (float*)image.array.raw;
    uint64_t counter = 0;

    // 2. Loop update
    while(keep_running) {
        image.md[0].write = 1;
        // Generate dummy data (moving gradient)
        for(uint32_t y=0; y<height; y++) {
            for(uint32_t x=0; x<width; x++) {
                data[y*width + x] = 0.5 * sin((x + counter)*0.1) + 0.5 * cos((y + counter)*0.05);
            }
        }

        // Post semaphore
        ImageStreamIO_UpdateIm(&image);
        counter++;
        usleep(10000); // 100 Hz
    }

    printf("Writer: Cleaning up...\n");
    // Not destroying the stream here so the processor doesn't crash if it's running
    // In a real app, you might want to clean up.
    return 0;
}
