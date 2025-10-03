/**
 * @file    compute_nb_image.c
 */

#include "CommandLineInterface/CLIcore.h"

long compute_nb_image()
{
    long image_count = 0;

    for(imageID i = 0; i < data.NB_MAX_IMAGE; i++)
    {
        if(data.image[i].used == 1)
        {
            image_count++;
        }
    }

    return image_count;
}
