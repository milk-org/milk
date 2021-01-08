/**
 * @file    CLIcore_utils.h
 * @brief   Util functions for coding convenience
 *
 */

#ifndef CLICORE_UTILS_H
#define CLICORE_UTILS_H

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "CommandLineInterface/IMGID.h"


static inline IMGID makeIMGID(
    const char *restrict name
)
{
    IMGID img;

    img.ID = -1;
    strcpy(img.name, name);

    img.im = NULL;
    img.md = NULL;
    img.createcnt = -1;

    return img;
}


static inline imageID resolveIMGID(
    IMGID *img,
    int ERRMODE
)
{
    if(img->ID == -1)
    {
        // has not been previously resolved -> resolve
        img->ID = image_ID(img->name);
        if(img->ID > -1)
        {
            img->im = &data.image[img->ID];
            img->md = &data.image[img->ID].md[0];
            img->createcnt = data.image[img->ID].createcnt;
        }
    }
    else
    {
        // check that create counter matches and image is in use
        if((img->createcnt != data.image[img->ID].createcnt)
                || (data.image[img->ID].used != 1))
        {
            // create counter mismatch -> need to re-resolve
            img->ID = image_ID(img->name);
            if(img->ID > -1)
            {
                img->im = &data.image[img->ID];
                img->md = &data.image[img->ID].md[0];
                img->createcnt = data.image[img->ID].createcnt;
            }
        }

    }

    if(img->ID == -1)
    {
        if( (ERRMODE == ERRMODE_FAIL) || (ERRMODE == ERRMODE_ABORT))
        {
            printf("ERROR: %c[%d;%dm Cannot resolve image %s %c[%d;m\n", (char) 27, 1, 31, img->name, (char) 27, 0);
            abort();
        }
        else if(ERRMODE == ERRMODE_WARN)
        {
            printf("WARNING: %c[%d;%dm Cannot resolve image %s %c[%d;m\n", (char) 27, 1, 35, img->name, (char) 27, 0);
        }
    }

    return img->ID;
}


#endif
