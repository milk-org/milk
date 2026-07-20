// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    create_image.c
 * @brief   create images and streams
 */
#include "CommandLineInterface/CLIcore.h"
#include "create_image.h"
#include "image_ID.h"
#include "list_image.h"
#include "stream_sem.h"

/* creates an image ID */

/* all images should be created by this function */
errno_t create_image_ID_IMGID(IMGID *img)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s %ld %d %d %d %d", img->name, (long) img->naxis, (int) img->datatype,
                     img->shared, img->NBkw, img->CBsize);
    if (image_ID(img->name) == -1)
    {
        img->ID = next_avail_image_ID(img->ID);
        ImageStreamIO_createIm(&data.image[img->ID], img->name, img->naxis, img->size,
                               img->datatype, img->shared, img->NBkw, img->CBsize);
    }
    else
    {
        // Cannot create image : name already in use
        img->ID = image_ID(img->name);
        if (data.image[img->ID].md->datatype != img->datatype)
        {
            FUNC_RETURN_FAILURE("Pre-existing image \"%s\" has wrong type", img->name);
        }
        if (data.image[img->ID].md->naxis != img->naxis)
        {
            FUNC_RETURN_FAILURE("Pre-existing image \"%s\" has wrong naxis", img->name);
        }
        for (int i = 0; i < img->naxis; i++)
        {
            if (data.image[img->ID].md->size[i] != img->size[i])
            {
                FUNC_RETURN_FAILURE("Pre-existing image \"%s\" has wrong size: axis %d "
                                    ":  %ld  %ld",
                                    img->name, i, (long) data.image[img->ID].md->size[i],
                                    (long) img->size[i]);
            }
        }
    }
    if (data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t create_image_ID(const char *__restrict name,
                        long      naxis,
                        uint32_t *size,
                        uint8_t   datatype,
                        int       shared,
                        int       NBkw,
                        int       CBsize,
                        imageID  *outID)
{
    IMGID img;
    strncpy(img.name, name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.naxis    = naxis;
    img.datatype = datatype;
    img.shared   = shared;
    img.NBkw     = NBkw;
    img.CBsize   = CBsize;
    for (int i = 0; i < naxis; i++)
    {
        img.size[i] = size[i];
    }
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_image_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_1Dimage_ID_IMGID(IMGID *img)
{
    img->naxis  = 1;
    img->shared = data.SHARED_DFT;
    img->NBkw   = NB_KEYWNODE_MAX;
    img->CBsize = 0;
    if (data.precision == 0)
    {
        img->datatype = _DATATYPE_FLOAT;
    }
    if (data.precision == 1)
    {
        img->datatype = _DATATYPE_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_1Dimage_ID(const char *restrict ID_name, uint32_t xsize, imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_1Dimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_1DCimage_ID_IMGID(IMGID *img)
{
    img->naxis  = 1;
    img->shared = data.SHARED_DFT;
    img->NBkw   = NB_KEYWNODE_MAX;
    img->CBsize = 0;
    if (data.precision == 0)
    {
        img->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if (data.precision == 1)
    {
        img->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_1DCimage_ID(const char *__restrict ID_name, uint32_t xsize, imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_1DCimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_2Dimage_ID_IMGID(IMGID *img)
{
    img->naxis  = 2;
    img->shared = data.SHARED_DFT;
    img->NBkw   = NB_KEYWNODE_MAX;
    img->CBsize = 0;
    if (data.precision == 0)
    {
        img->datatype = _DATATYPE_FLOAT;
    }
    else if (data.precision == 1)
    {
        img->datatype = _DATATYPE_DOUBLE;
    }
    else
    {
        img->datatype = _DATATYPE_FLOAT;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_2Dimage_ID(const char *__restrict ID_name,
                          uint32_t xsize,
                          uint32_t ysize,
                          imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2Dimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_2Dimage_ID_double_IMGID(IMGID *img)
{
    img->naxis    = 2;
    img->datatype = _DATATYPE_DOUBLE;
    img->shared   = data.SHARED_DFT;
    img->NBkw     = NB_KEYWNODE_MAX;
    img->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

errno_t create_2Dimage_ID_double(const char *__restrict ID_name,
                                 uint32_t xsize,
                                 uint32_t ysize,
                                 imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2Dimage_ID_double_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_2DCimage_ID_IMGID(IMGID *img)
{
    img->naxis  = 2;
    img->shared = data.SHARED_DFT;
    img->NBkw   = NB_KEYWNODE_MAX;
    img->CBsize = 0;
    if (data.precision == 0)
    {
        img->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if (data.precision == 1)
    {
        img->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

/* 2D complex image */
errno_t create_2DCimage_ID(const char *__restrict ID_name,
                           uint32_t xsize,
                           uint32_t ysize,
                           imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2DCimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_2DCimage_ID_double_IMGID(IMGID *img)
{
    img->naxis    = 2;
    img->datatype = _DATATYPE_COMPLEX_DOUBLE;
    img->shared   = data.SHARED_DFT;
    img->NBkw     = NB_KEYWNODE_MAX;
    img->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 2D complex image */
errno_t create_2DCimage_ID_double(const char *__restrict ID_name,
                                  uint32_t xsize,
                                  uint32_t ysize,
                                  imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2DCimage_ID_double_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_3Dimage_ID_float_IMGID(IMGID *img)
{
    img->naxis    = 3;
    img->datatype = _DATATYPE_FLOAT;
    img->shared   = data.SHARED_DFT;
    img->NBkw     = NB_KEYWNODE_MAX;
    img->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 3D image, single precision */
errno_t create_3Dimage_ID_float(const char *__restrict ID_name,
                                uint32_t xsize,
                                uint32_t ysize,
                                uint32_t zsize,
                                imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.size[2]    = zsize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3Dimage_ID_float_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_3Dimage_ID_double_IMGID(IMGID *img)
{
    img->naxis    = 3;
    img->datatype = _DATATYPE_DOUBLE;
    img->shared   = data.SHARED_DFT;
    img->NBkw     = NB_KEYWNODE_MAX;
    img->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 3D image, double precision */
errno_t create_3Dimage_ID_double(const char *__restrict ID_name,
                                 uint32_t xsize,
                                 uint32_t ysize,
                                 uint32_t zsize,
                                 imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.size[2]    = zsize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3Dimage_ID_double_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_3Dimage_ID_IMGID(IMGID *img)
{
    img->naxis  = 3;
    img->shared = data.SHARED_DFT;
    img->NBkw   = NB_KEYWNODE_MAX;
    img->CBsize = 0;
    if (data.precision == 0)
    {
        img->datatype = _DATATYPE_FLOAT;
    }
    if (data.precision == 1)
    {
        img->datatype = _DATATYPE_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

/* 3D image, default precision */
errno_t create_3Dimage_ID(const char *__restrict ID_name,
                          uint32_t xsize,
                          uint32_t ysize,
                          uint32_t zsize,
                          imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.size[2]    = zsize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3Dimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}

errno_t create_3DCimage_ID_IMGID(IMGID *img)
{
    img->naxis  = 3;
    img->shared = data.SHARED_DFT;
    img->NBkw   = NB_KEYWNODE_MAX;
    img->CBsize = 0;
    if (data.precision == 0)
    {
        img->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if (data.precision == 1)
    {
        img->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

/* 3D complex image */
errno_t create_3DCimage_ID(const char *__restrict ID_name,
                           uint32_t xsize,
                           uint32_t ysize,
                           uint32_t zsize,
                           imageID *outID)
{
    IMGID img;
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.size[0]    = xsize;
    img.size[1]    = ysize;
    img.size[2]    = zsize;
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3DCimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    return retval;
}
