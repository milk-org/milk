#ifndef IMAGEID_H
#define IMAGEID_H

#include "CLIcore.h"
#include "image_ID.h"
#include <string.h>
#include <stdlib.h>

static inline imageID RegisterIMGID(
    IMGID *img,
    IMAGE *imagearray,
    long NB_images
)
{
    imageID ID = -1;

    if (imagearray == NULL)
    {
        // If no array provided, we close the image and return 0 (success) or -1 (fail)
        // This corresponds to non-CLI mode check
        if (img->ID != -1)
        {
            ImageStreamIO_closeIm(img->im);
            free(img->im);
            img->ID = 0;
            return 0;
        }
        return -1;
    }

    // Check if already loaded
    ID = image_ID(img->name, imagearray, NB_images);
    if(ID != -1)
    {
        // Already loaded: close the one we just opened and point to the existing one
        if (img->im != NULL)
        {
            ImageStreamIO_closeIm(img->im);
            free(img->im);
        }
        img->ID = ID;
        img->im = &imagearray[ID];
        img->md = &imagearray[ID].md[0];
        img->createcnt = imagearray[ID].createcnt;
        updateIMGIDcreationparams(img);
    }
    else
    {
        // Not loaded: find slot and move it
        ID = next_avail_image_ID(-1);
        if (ID != -1)
        {
            // We assume imagearray has enough space and ID is valid index
            // Move content
            memcpy(&imagearray[ID], img->im, sizeof(IMAGE));
            // Free temporary structure
            free(img->im);

            img->ID = ID;
            img->im = &imagearray[ID];
            img->md = &imagearray[ID].md[0];
            // img.createcnt = data.image[img.ID].createcnt; // Should be set? ImageStreamIO doesn't set createcnt?
            // Actually createcnt is in IMAGE struct, so it was copied.

            imagearray[ID].used = 1; // next_avail_image_ID sets this, but just to be sure if we used different logic

            updateIMGIDcreationparams(img);
        }
        else
        {
            // No space available
            if (img->im != NULL)
            {
                ImageStreamIO_closeIm(img->im);
                free(img->im);
            }
            img->ID = -1;
        }
    }

    return ID;
}

#endif
