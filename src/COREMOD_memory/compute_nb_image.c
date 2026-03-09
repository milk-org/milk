/**
 * @file    compute_nb_image.c
 */

#include "CLIcore.h"

long compute_nb_image()
{
    long image_count = 0;

    for(imageID i = 0; i < dcnimg; i++)
    {
        if(dcimg[i].used == 1)
        {
            image_count++;
        }
    }

    return image_count;
}
