/**
 * @file milk-example-03-writer.c
 * @brief Writer for the FPS example.
 *
 * This is similar to previous examples but uses 'stream03'.
 */

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
    printf("\nWriter: Exiting on CTRL+C.\n");
    exit(0);
}

int main(int argc, char *argv[]) {
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("\nUsage: %s\n\n", "milk-example-03-writer");
        printf("Description:\n");
        printf("  Simple writer creating 'stream03' for the full FPS/ProcessInfo example.\n\n");
        return 0;
    }

    signal(SIGINT, signal_handler);

    const char *stream_name = "stream03";
    uint32_t width = 200;
    uint32_t height = 200;
    uint32_t dims[2] = {width, height};
    
    IMAGE image;
    if (ImageStreamIO_createIm_gpu(&image, stream_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != 0) {
        return 1;
    }

    printf("Writer: Stream '%s' created.\n", stream_name);

    float *data = (float*)image.array.raw;
    uint64_t counter = 0;

    while(keep_running) {
        image.md[0].write = 1;
        for(uint32_t y=0; y<height; y++) {
            for(uint32_t x=0; x<width; x++) {
                data[y*width + x] = 0.5 * sin((x + counter)*0.1) + 0.5 * cos((y + counter)*0.05);
            }
        }
        ImageStreamIO_UpdateIm(&image);
        counter++;
        usleep(10000); 
    }
    return 0;
}