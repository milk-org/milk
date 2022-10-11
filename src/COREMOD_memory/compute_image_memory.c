/**
 * @file    compute_image_memory.c
 */

#include "CommandLineInterface/CLIcore.h"

uint64_t compute_image_memory()
{
    uint64_t totalmem = 0;

    for(imageID i = 0; i < data.NB_MAX_IMAGE; i++)
    {
        //printf("%5ld / %5ld  %d\n", i, data.NB_MAX_IMAGE, data.image[i].used);
        //	fflush(stdout);

        if(data.image[i].used == 1)
        {
            totalmem += data.image[i].md[0].nelement *
                        ImageStreamIO_typesize(data.image[i].md[0].datatype);
        }
    }

    return totalmem;
}
