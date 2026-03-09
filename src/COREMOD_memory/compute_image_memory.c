/**
 * @file    compute_image_memory.c
 */

#include "CLIcore.h"


uint64_t compute_image_memory()
{
    uint64_t totalmem = 0;

    for(imageID i = 0; i < dcnimg; i++)
    {
        //printf("%5ld / %5ld  %d\n", i, dcnimg, dcimg[i].used);
        //	fflush(stdout);

        if(dcimg[i].used == 1)
        {
            totalmem += dcimg[i].md[0].nelement *
                        ImageStreamIO_typesize(dcimg[i].md[0].datatype);
        }
    }

    return totalmem;
}
