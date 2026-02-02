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




/** @brief Resolve image already in memory
 *
 *
 *
 * ERRMODE values
 * ERRMODE_WARN : print warning
 * ERRMODE_FAIL : error
 * ERRMODE_ABORT : abort
 */
static inline imageID resolveIMGID(
    IMGID *img,
    int ERRMODE,
    IMAGE *imagearray,
    long NB_images
)
{
    // IF:
    // Not resolved before OR create counter mismatch OR not used.
    // Note: we are comparing img->createcnt to data.image[img->ID].createcnt to check if the
    // image has been re-created, indicating that our pointers are stale.
    if((img->ID == -1)
            || (img->createcnt != imagearray[img->ID].createcnt)
            || (imagearray[img->ID].used != 1))
    {
        img->ID = image_ID(img->name, data.image, data.NB_MAX_IMAGE);
        if(img->ID > -1)  // Resolve success !
        {
            img->im        = &imagearray[img->ID];
            img->md        = &imagearray[img->ID].md[0];
            img->createcnt = imagearray[img->ID].createcnt;

            // Populate the IMGID from the imageID metadata
            updateIMGIDcreationparams(img);
        }
    }

    // if still unresolved
    //
    if(img->ID == -1)
    {
        if((ERRMODE == ERRMODE_FAIL) || (ERRMODE == ERRMODE_ABORT))
        {
            PRINT_ERROR("Cannot resolve image %s\n", img->name);
            abort();
        }
        else if(ERRMODE == ERRMODE_WARN)
        {
            PRINT_WARNING("Cannot resolve image %s\n", img->name);
        }
    }

    return img->ID;
}






static inline IMGID makesetIMGID(CONST_WORD name, imageID ID)
{
    IMGID img;

    img.ID = ID;
    strncpy(img.name, name, STRINGMAXLEN_IMAGE_NAME - 1);

    img.im        = &data.image[ID];
    img.md        = &data.image[ID].md[0];
    img.createcnt = data.image[ID].createcnt;

    return img;
}





/**
 * @brief Connnect to stream
 *
 * @param imname  stream name
 * @return IMGID
 */
static inline IMGID
stream_connect(
    const char *__restrict imname
)
{
    IMGID img = mkIMGID_from_name(imname);
    resolveIMGID(&img, ERRMODE_WARN, data.image, data.NB_MAX_IMAGE);

    if(img.ID == -1)
    {
        // try to connect to shared memory if not in local memory already
        read_sharedmem_image(imname, data.image, data.NB_MAX_IMAGE);
        resolveIMGID(&img, ERRMODE_WARN, data.image, data.NB_MAX_IMAGE);
    }

    return img;
}



static inline imageID createimagefromIMGID(IMGID *img)
{
    create_image_ID(img->name,
                    img->naxis,
                    img->size,
                    img->datatype,
                    img->shared,
                    img->NBkw,
                    img->CBsize,
                    &img->ID);

    img->im        = &data.image[img->ID];
    img->md        = &data.image[img->ID].md[0];
    img->createcnt = data.image[img->ID].createcnt;

    return img->ID;
}




/** Create image according to IMGID entries of existing image
 */
static inline imageID imcreatelikewiseIMGID(
    IMGID *target_img,
    IMGID *source_img
)
{
    if(target_img->ID == -1)
    {
        if(target_img != source_img)
        {
            printf("Creating image %s from %s, shared = %d, kw = %d\n",
                   target_img->name,
                   source_img->name,
                   source_img->shared,
                   source_img->NBkw);
        }
        else
        {
            printf("Creating image %s, shared = %d, kw = %d\n",
                   source_img->name,
                   source_img->shared,
                   source_img->NBkw);
        }

        DEBUG_TRACEPOINT("Creating 2D image");
        create_image_ID(target_img->name,
                        source_img->naxis,
                        source_img->size,
                        source_img->datatype,
                        source_img->shared,
                        source_img->NBkw,
                        source_img->CBsize,
                        &target_img->ID);
        DEBUG_TRACEPOINT(" ");
        target_img->im        = &data.image[target_img->ID];
        target_img->md        = &data.image[target_img->ID].md[0];
        target_img->createcnt = data.image[target_img->ID].createcnt;


        target_img->size[0] = source_img->size[0];
        if(source_img->naxis > 1)
        {
            target_img->size[1] = source_img->size[1];
        }
        if(source_img->naxis > 2)
        {
            target_img->size[2] = source_img->size[2];
        }
    }
    return target_img->ID;
}



/** Create image according to IMGID entries
 *  See cloning creation function imcreatelikewiseIMGID()
 */
static inline imageID imcreateIMGID(IMGID *img)
{
    return imcreatelikewiseIMGID(img, img);
}



static inline IMGID stream_connect_create_2D(
    const char *__restrict imname,
    uint32_t xsize,
    uint32_t ysize,
    uint8_t  datatype
)
{
    IMGID img = mkIMGID_from_name(imname);
    resolveIMGID(&img, ERRMODE_WARN, data.image, data.NB_MAX_IMAGE);


    if(img.ID == -1)
    {
        // try to connect to shared memory if not in local memory already
        read_sharedmem_image(imname, data.image, data.NB_MAX_IMAGE);
        resolveIMGID(&img, ERRMODE_WARN, data.image, data.NB_MAX_IMAGE);
    }

    if(img.ID != -1)
    {
        // if in local memory,
        // create blank img for comparison
        IMGID imgc      = makeIMGID_blank();
        imgc.datatype   = datatype;
        imgc.naxis      = 2;
        imgc.size[0]    = xsize;
        imgc.size[1]    = ysize;
        imgc.NBkw       = NB_KEYWNODE_MAX;
        uint64_t imgerr = IMGIDcompare(img, imgc);
        printf("%lu errors\n", imgerr);

        // if doesn't pass test, erase from local memory
        if(imgerr != 0)
        {
            delete_image_ID(imname, DELETE_IMAGE_ERRMODE_WARNING);
            img.ID = -1;
        }
    }

    // if not in local memory, (re)-create
    if(img.ID == -1)
    {
        uint32_t arraytmp[2];

        arraytmp[0] = xsize;
        arraytmp[1] = ysize;

        create_image_ID(imname, 2, arraytmp, datatype, 1, NB_KEYWNODE_MAX, 0, &img.ID);
    }


    if(img.ID != -1)
    {
        imageID ID    = img.ID;
        img.im        = &data.image[ID];
        img.md        = data.image[ID].md;
        img.createcnt = data.image[ID].createcnt;
        updateIMGIDcreationparams(&img);
    }

    return img;
}

/**
 * @brief Connnect to stream or create if doesn't exist
 *
 * If stream exists but has wrong size type, recreate
 *
 * @param imname  stream name
 * @param xsize   x size
 * @param ysize   y size
 * @return IMGID
 */
static inline IMGID
stream_connect_create_2Df32(
    const char *__restrict imname,
    uint32_t xsize,
    uint32_t ysize
)
{
    return stream_connect_create_2D(imname, xsize, ysize, _DATATYPE_FLOAT);
}

static inline IMGID stream_connect_create_3D(
    const char *__restrict imname,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize,
    uint8_t  datatype
)
{
    IMGID img = mkIMGID_from_name(imname);
    resolveIMGID(&img, ERRMODE_WARN, data.image, data.NB_MAX_IMAGE);


    if(img.ID == -1)
    {
        // try to connect to shared memory if not in local memory already
        read_sharedmem_image(imname, data.image, data.NB_MAX_IMAGE);
        resolveIMGID(&img, ERRMODE_WARN, data.image, data.NB_MAX_IMAGE);
    }

    if(img.ID != -1)
    {
        // if in local memory,
        // create blank img for comparison
        IMGID imgc      = makeIMGID_blank();
        imgc.datatype   = datatype;
        imgc.naxis      = 3;
        imgc.size[0]    = xsize;
        imgc.size[1]    = ysize;
        imgc.size[2]    = zsize;
        imgc.NBkw       = NB_KEYWNODE_MAX;
        uint64_t imgerr = IMGIDcompare(img, imgc);
        printf("%lu errors\n", imgerr);

        // if doesn't pass test, erase from local memory
        if(imgerr != 0)
        {
            delete_image_ID(imname, DELETE_IMAGE_ERRMODE_WARNING);
            img.ID = -1;
        }
    }

    // if not in local memory, (re)-create
    if(img.ID == -1)
    {
        uint32_t arraytmp[3];

        arraytmp[0] = xsize;
        arraytmp[1] = ysize;
        arraytmp[2] = zsize;

        printf("CREATING image size %u %u %u\n", xsize, ysize, zsize);

        create_image_ID(imname, 3, arraytmp, datatype, 1, NB_KEYWNODE_MAX, 0, &img.ID);
    }


    if(img.ID != -1)
    {
        imageID ID    = img.ID;
        img.im        = &data.image[ID];
        img.md        = data.image[ID].md;
        img.createcnt = data.image[ID].createcnt;
        updateIMGIDcreationparams(&img);
    }

    return img;
}

/**
 * @brief Connnect to stream or create if doesn't exist
 *
 * If stream exists but has wrong size type, recreate
 *
 * @param imname  stream name
 * @param xsize   x size
 * @param ysize   y size
 * @param zsize   z size
 * @return IMGID
 */
static inline IMGID stream_connect_create_3Df32(
    const char *__restrict imname,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize)
{
    return stream_connect_create_3D(imname, xsize, ysize, zsize, _DATATYPE_FLOAT);
}





#endif
