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
        printf("  Example 02 Writer: Creates 'stream02' (200x200 FLOAT) and writes a\n");
        printf("  moving gradient pattern. This stream is intended for the Example 02\n");
        printf("  processor which includes processinfo monitoring.\n\n");
        return 0;
    }
    signal(SIGINT, signal_handler);

    const char *stream_name = "stream02";
    uint32_t width = 200;
    uint32_t height = 200;
    uint32_t dims[2] = {width, height};
    
    IMAGE image;
    ImageStreamIO_createIm_gpu(&image, stream_name, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);

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
