/**
 * @file compute_nb_image.c
 * @brief Compute nb image module
 */

/**
 * @file    compute_nb_image.c
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#endif

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
