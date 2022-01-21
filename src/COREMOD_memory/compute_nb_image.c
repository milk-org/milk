/**
 * @file    compute_nb_image.c
 */

#include "CommandLineInterface/CLIcore.h"

long compute_nb_image()
{
    long NBimage = 0;

    //printf("NB_MAX_IMAGE = %d\n", data.NB_MAX_IMAGE);
    //fflush(stdout);

    for (imageID i = 0; i < data.NB_MAX_IMAGE; i++)
    {
        if (data.image[i].used == 1)
        {
            NBimage += 1;
        }
    }

    return NBimage;
}
