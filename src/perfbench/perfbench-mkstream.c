/**
 * @file    perfbench-mkstream.c
 * @brief   Create a dummy SHM stream for benchmarking
 *
 * Usage: milk-perfbench-mkstream <name> <xsize> <ysize>
 *
 * Creates a 2D float shared-memory stream.
 * Engine-tier only (no CLI dependency).
 */

#include <stdio.h>
#include <stdlib.h>

#include "ImageStruct.h"
#include "ImageStreamIO.h"

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <name>"
                " <xsize> <ysize>\n",
                argv[0]);
        return 1;
    }

    const char *name = argv[1];
    uint32_t xsize = (uint32_t) atoi(argv[2]);
    uint32_t ysize = (uint32_t) atoi(argv[3]);

    IMAGE img;
    uint32_t dims[2] = {xsize, ysize};

    if (ImageStreamIO_createIm(
            &img, name, 2, dims,
            _DATATYPE_FLOAT, 1, 10, 0)
        != EXIT_SUCCESS)
    {
        fprintf(stderr,
                "Error creating stream '%s'\n",
                name);
        return 1;
    }

    printf("Created stream '%s' %ux%u float\n",
           name, xsize, ysize);

    return 0;
}
