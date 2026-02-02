/**
 * @file    IMGID.h
 * @brief   Image identifying structure
 *
 */

#ifndef IMGID_H
#define IMGID_H

#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"




#ifdef __cplusplus
typedef const char *CONST_WORD;
#else
typedef const char *__restrict CONST_WORD;
#endif

#define IMGID_NB_KEYWO_MAX 10





// Image identifier for internal use by milk
//
// This is the preferred format for handling images in milk as function arguments,
// providing additional information and context for the current process/application.
//
// When used with milk-CLI, the type avoids name-resolving imageID multiple times
// provides quick and convenient access to data and metadata pointers
// Pass this as argument to functions to have both input-by-ID (ID>-1)
// and input-by-name (ID=-1).
typedef struct
{
    imageID ID; // -1 if not resolved
    // If using an array of images (as done by milk-CLI) this is the index
    // of the image in the array.
    // If not using an array of images, ID=0 if the image is loaded in memory.

    // increments when image created, used to check if re-resolving needed
    int64_t createcnt;

    // used to resolve if needed
    char            name[STRINGMAXLEN_IMAGE_NAME];

    // image content, data and metadata
    IMAGE          *im;
    // md points at im.md
    IMAGE_METADATA *md;

    // TEMPLATE
    // Requested image params.
    // Used to create image or test if existing image matches.
    // These fields do not always match the image content.
    //
    uint8_t  datatype;
    int      naxis;
    uint32_t size[3];
    int      shared;
    // number of keywords
    int      NBkw;

    // fast circular buffer size
    int CBsize;

} IMGID;



/** make blank IMGID
 *
 * All fields are uninitialized
 * Can be used for comparison
*/
static inline IMGID makeIMGID_blank()
{
    IMGID img;

    // default values for image creation
    img.datatype = _DATATYPE_UNINITIALIZED;
    img.naxis    = -1;
    img.size[0]  = 0;
    img.size[1]  = 0;
    img.size[2]  = 0;
    img.shared   = -1;
    img.NBkw     = -1;
    img.CBsize   = -1;

    img.ID        = -1;
    img.createcnt = -1;
    strncpy(img.name, "", STRINGMAXLEN_IMAGE_NAME - 1);
    img.im = NULL;
    img.md = NULL;

    return img;
}



/** make IMGID from name
 *
 * Some settings can be embedded in the image name string for convenience :
 *
 * Examples:
 * "im1" no optional setting, image name = im1
 * "s>im1" : set shared memory flag
 * "k10>im1" : number of keyword = 10
 * "c20>im1" : 20-sized circular buffer
 * "tf64>im1" : datatype is double (64 bit floating point)
*/
static inline IMGID mkIMGID_from_name(CONST_WORD name)
{
    IMGID img = {0};

    // default values for image creation
    img.datatype = _DATATYPE_FLOAT;
    img.naxis    = 2;
    img.size[0]  = 1;
    img.size[1]  = 1;
    img.shared   = 0;
    img.NBkw     = IMGID_NB_KEYWO_MAX;
    img.CBsize   = 0;

    char *pch;
    char *pch1;
    int   nbword = 0;

    char  namestring[200];
    strncpy(namestring, name, 199);

    pch1 = namestring;
    if(strlen(namestring) != 0)
    {
        pch = strtok(namestring, ">");
        while(pch != NULL)
        {
            pch1 = pch;
            //printf("[%2d] %s\n", nbword, pch);

            if(strcmp(pch, "s") == 0)
            {
                printf("    shared memory\n");
                img.shared = 1;
            }

            if(strcmp(pch, "tui8") == 0)
            {
                printf("    data type unsigned 8-bit int\n");
                img.datatype = _DATATYPE_UINT8;
            }
            if(strcmp(pch, "tsi8") == 0)
            {
                printf("    data type signed 8-bit int\n");
                img.datatype = _DATATYPE_INT8;
            }
            if(strcmp(pch, "tui16") == 0)
            {
                printf("    data type unsigned 16-bit int\n");
                img.datatype = _DATATYPE_UINT16;
            }
            if(strcmp(pch, "tsi16") == 0)
            {
                printf("    data type signed 16-bit int\n");
                img.datatype = _DATATYPE_INT16;
            }
            if(strcmp(pch, "tui32") == 0)
            {
                printf("    data type unsigned 32-bit int\n");
                img.datatype = _DATATYPE_UINT32;
            }
            if(strcmp(pch, "tsi32") == 0)
            {
                printf("    data type signed 32-bit int\n");
                img.datatype = _DATATYPE_INT32;
            }
            if(strcmp(pch, "tui64") == 0)
            {
                printf("    data type unsigned 64-bit int\n");
                img.datatype = _DATATYPE_UINT64;
            }
            if(strcmp(pch, "tsi64") == 0)
            {
                printf("    data type signed 64-bit int\n");
                img.datatype = _DATATYPE_INT64;
            }

            if(strcmp(pch, "tf32") == 0)
            {
                printf("    data type double (32)\n");
                img.datatype = _DATATYPE_FLOAT;
            }
            if(strcmp(pch, "tf64") == 0)
            {
                printf("    data type float (64)\n");
                img.datatype = _DATATYPE_DOUBLE;
            }

            /*            if(pch[0] == 'k')
                        {
                            int nbkw;
                            sscanf(pch, "k%d", &nbkw);
                            printf("    %d keywords\n", nbkw);
                            img.NBkw = nbkw;
                        }

                        if(pch[0] == 'c')
                        {
                            int cbsize;
                            sscanf(pch, "c%d", &cbsize);
                            printf("    %d circular buffer size\n", cbsize);
                            img.CBsize = cbsize;
                        }
            */
            pch = strtok(NULL, ">");
            nbword++;
        }
    }

    img.ID        = -1;
    img.createcnt = -1;
    strncpy(img.name, pch1, STRINGMAXLEN_IMAGE_NAME - 1);
    img.im = NULL;
    img.md = NULL;

    return img;
}





static inline IMGID makeIMGID_2D(
    CONST_WORD name,
    uint32_t xsize,
    uint32_t ysize
)
{
    IMGID img   = mkIMGID_from_name(name);
    img.naxis   = 2;
    img.size[0] = xsize;
    img.size[1] = ysize;

    return img;
}

static inline IMGID makeIMGID_3D(
    CONST_WORD name,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
{
    IMGID img   = mkIMGID_from_name(name);
    img.naxis   = 3;
    img.size[0] = xsize;
    img.size[1] = ysize;
    img.size[2] = zsize;

    return img;
}



static inline void copyIMGID(
    IMGID *imgin,
    IMGID *imgout
)
{
    imgout->datatype = imgin->datatype;
    imgout->shared   = imgin->shared;

    imgout->naxis = imgin->naxis;

    imgout->size[0] = imgin->size[0];
    imgout->size[1] = imgin->size[1];
    imgout->size[2] = imgin->size[2];

    imgout->NBkw   = imgin->NBkw;
    imgout->CBsize = imgin->CBsize;
}



static inline void updateIMGIDcreationparams(IMGID *img)
{
    img->datatype = img->md->datatype;
    img->naxis    = img->md->naxis;
    for(int ii = 0; ii < 3; ++ii)
    {
        img->size[ii] = img->md->size[ii];
    }
    img->shared = img->md->shared;
    img->NBkw   = img->md->NBkw;
    img->CBsize = img->md->CBsize;
}





/**
 * @brief Check if img complies to imgtemplate
 *
 */
static inline uint64_t IMGIDcompare(
    IMGID img,
    IMGID imgtemplate
)
{
    int compErr = 0;

    if(imgtemplate.datatype != _DATATYPE_UNINITIALIZED)
    {
        printf("Checking datatype       ");
        if(imgtemplate.datatype != img.datatype)
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.naxis != -1)
    {
        printf("Checking naxis  %d %d    ", imgtemplate.naxis, img.naxis);
        if(imgtemplate.naxis != img.naxis)
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.size[0] != 0)
    {
        printf("Checking size[0]        ");
        if(imgtemplate.size[0] != img.size[0])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.size[1] != 0)
    {
        printf("Checking size[1]        ");
        if(imgtemplate.size[1] != img.size[1])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.size[2] != 0)
    {
        printf("Checking size[2]        ");
        if(imgtemplate.size[2] != img.size[2])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    printf("Checking NBkw           ");
    if(imgtemplate.NBkw != img.NBkw)
    {
        printf("FAIL\n");
        printf("   %4u  %s\n", imgtemplate.NBkw, imgtemplate.name);
        printf("   %4u  %s\n", img.NBkw, img.name);
        compErr++;
    }
    else
    {
        printf("PASS\n");
    }


    return compErr;
}




/**
 * @brief Check if img complies to imgtemplate
 *
 */
static inline uint64_t IMGIDmdcompare(
    IMGID img,
    IMGID imgtemplate
)
{
    int compErr = 0;

    printf("COMPARING %s %s\n", img.name, imgtemplate.name);

    if(imgtemplate.md->datatype != _DATATYPE_UNINITIALIZED)
    {
        printf("Checking md->datatype       ");
        if(imgtemplate.md->datatype != img.md->datatype)
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->naxis != 0)
    {
        printf("Checking md->naxis  %d %d    ", imgtemplate.md->naxis, img.md->naxis);
        if(imgtemplate.md->naxis != img.md->naxis)
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->size[0] != 0)
    {
        printf("Checking md->size[0]        ");
        if(imgtemplate.md->size[0] != img.md->size[0])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->size[1] != 0)
    {
        printf("Checking md->size[1]        ");
        if(imgtemplate.md->size[1] != img.md->size[1])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    if(imgtemplate.md->size[2] != 0)
    {
        printf("Checking md->size[2]        ");
        if(imgtemplate.md->size[2] != img.md->size[2])
        {
            printf("FAIL\n");
            compErr++;
        }
        else
        {
            printf("PASS\n");
        }
    }

    printf("Checking NBkw           ");
    if(imgtemplate.md->NBkw != img.md->NBkw)
    {
        printf("FAIL\n");
        printf("   %4u  %s\n", imgtemplate.md->NBkw, imgtemplate.md->name);
        printf("   %4u  %s\n", img.md->NBkw, img.md->name);
        printf("    template : %u\n", imgtemplate.md->NBkw);
        printf("    dest     : %u\n", img.md->NBkw);
        compErr++;
    }
    else
    {
        printf("PASS\n");
    }

    return compErr;
}




// Read ImageStreamIO from shared memory
//
static inline IMGID read_sharedmem_img(
    CONST_WORD sname
)
{
    IMGID img;
    img.ID = -1;

    if(strlen(sname) != 0)
    {
        // Allocating IMAGE structure
        IMAGE *image = (IMAGE*) malloc(sizeof(IMAGE));
        if (image == NULL)
        {
            return img;
        }

        if(ImageStreamIO_read_sharedmem_image_toIMAGE(sname, image) !=
                IMAGESTREAMIO_SUCCESS)
        {
            printf("read shared mem image failed -> ID = -1\n");
            fflush(stdout);
            free(image);
            img.ID = -1;
        }
        else
        {
            img.im = image;
            img.md = image->md;
            strcpy(img.name, sname);
            img.ID = 0; // Temporary ID indicating success
        }
    }

    return(img);
}

// Create image from IMGID
static inline void mkimage(IMGID * img)
{
    ImageStreamIO_createIm(img->im, img->name, img->naxis, img->size, img->datatype, img->shared, img->NBkw, img->CBsize);
    img->createcnt++;
}




#endif
