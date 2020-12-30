/**
 * @file    CLIcore_utils.c
 * @brief   Util functions for coding convenience
 * 
 */


#include <string.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

//#include "ImageStreamIO/ImageStreamIO.h"
//#include "CommandLineInterface/IMGID.h"



inline IMGID makeIMGID(
    const char *restrict name
)
{
    IMGID img;

    img.ID = -1;
    strcpy(img.name, name);

    return img;
}

inline imageID resolveIMGID(
    IMGID *img
)
{
    if(img->ID == -1)
    {
        img->ID = image_ID(img->name);
        if(img->ID > -1)
        {
            img->im = &data.image[img->ID];
            img->md = &data.image[img->ID].md[0];
        }
    }

    return img->ID;
}

