#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include "ImageStruct.h"
#include "ImageStreamIO.h"

static int keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
}

int main() {
    signal(SIGINT, signal_handler);

    const char *in_name = "stream01";
    const char *out_name = "stream01_proc";

    // 1. Connect to Input Stream
    IMAGE input_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(in_name, &input_image) != 0) {
        fprintf(stderr, "Error: Could not connect to stream '%s'. Start writer first.\n", in_name);
        return 1;
    }

    // 2. Create Output Stream (Sum of 4 ROIs)
    // ROIs are 50x50
    uint32_t roi_w = 50;
    uint32_t roi_h = 50;
    uint32_t dims[2] = {roi_w, roi_h};
    IMAGE output_image;
    ImageStreamIO_createIm_gpu(&output_image, out_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);

    printf("Processor: Processing '%s' -> '%s'. Press CTRL+C to stop.\n", in_name, out_name);

    float *in_data = (float*)input_image.array.raw;
    float *out_data = (float*)output_image.array.raw;
    uint32_t in_w = input_image.md[0].size[0];

    // Offsets for 4 ROIs (Top-Left, Top-Right, Bottom-Left, Bottom-Right of 200x200 image)
    int off_x[4] = {0, 150, 0, 150};
    int off_y[4] = {0, 0, 150, 150};

    // 3. Processing Loop
    while(keep_running) {
        // Wait for new frame on input
        // Using semaphore 0
        ImageStreamIO_semwait(&input_image, 0);

        output_image.md[0].write = 1;
        // Process
        for(uint32_t y=0; y<roi_h; y++) {
            for(uint32_t x=0; x<roi_w; x++) {
                float sum = 0.0;
                for(int k=0; k<4; k++) {
                    uint32_t ix = off_x[k] + x;
                    uint32_t iy = off_y[k] + y;
                    sum += in_data[iy * in_w + ix];
                }
                out_data[y*roi_w + x] = sum;
            }
        }

        // Post output
        ImageStreamIO_UpdateIm(&output_image);
    }

    return 0;
}
