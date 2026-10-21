// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    create_image.c
 * @brief   Image and stream creation API
 *
 * Central factory for creating images/streams in the
 * milk framework. All images should be created through
 * create_image_ID_IMGID() (or its wrappers), which:
 *
 *  1. Checks if an image with the same name exists.
 *  2. If not, allocates a new slot and calls
 *     ImageStreamIO_createIm().
 *  3. If it exists, checks for type/size mismatch
 *     and re-creates if needed.
 *
 * Convenience wrappers set naxis + default precision
 * before delegating:
 *  - create_{1,2,3}Dimage_ID[_double|_float]()
 *  - create_{1,2,3}DCimage_ID[_double]()  (complex)
 *
 * Each wrapper has both a string API (name-based)
 * and an IMGID API.
 */
#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif
#include <string.h>
#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "COREMOD_memory/imageID.h"
#include "list_image.h"
#include "stream_sem.h"
#include <computeunit.h>

/**
 * @brief Create or reuse an image (IMGID API)
 *
 * Master creation function. Checks for existing
 * image with same name; if found and compatible,
 * reuses it. If mismatched (type, naxis, size),
 * deletes and re-creates.
 *
 * @param img  IMGID with name, mdt fields set
 * @return RETURN_SUCCESS
 */
errno_t create_image_ID_IMGID(IMGID *img)
{
    DEBUG_TRACE_FSTART();
    if (img->stream_parse_error)
    {
        img->ID = -1;
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }

    if (img->name[0] == '@' || strncmp(img->name, "s.", 2) == 0 ||
        strncmp(img->name, "stream.", 7) == 0 || strncmp(img->name, "l.", 2) == 0 ||
        strncmp(img->name, "local.", 6) == 0)
    {
        FPS_STREAMNAME_PARSED sp = fps_streamname_parse(img->name);
        if (sp.error)
        {
            fprintf(stderr, "ERROR: invalid stream modifier prefix in \"%s\"\n", img->name);
            img->stream_parse_error = 1;
            img->ID                 = -1;
            DEBUG_TRACE_FEXIT();
            return RETURN_FAILURE;
        }
        if (sp.loc != '\0')
        {
            img->stream_loc = sp.loc;
        }
        img->stream_must_exist = sp.must_exist;
        img->stream_must_new   = sp.must_new;
        if (sp.datatype != 0)
        {
            img->stream_datatype = sp.datatype;
        }
        memmove(img->name, sp.name, strlen(sp.name) + 1);
    }

    if (img->name[0] == '\0' || strchr(img->name, '@') != NULL || strchr(img->name, ':') != NULL)
    {
        fprintf(stderr, "ERROR: invalid image name \"%s\"\n", img->name);
        img->ID = -1;
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }

    if (img->stream_loc == 'L')
    {
        img->mdt->shared = 0;
    }
    else if (img->stream_loc == 'S')
    {
        img->mdt->shared = 1;
    }
    if (img->stream_datatype != 0)
    {
        img->mdt->datatype = img->stream_datatype;
    }

    DEBUG_TRACEPOINT("FARG %s %ld %d %d %d %d", img->name, (long) img->mdt->naxis,
                     (int) img->mdt->datatype, img->mdt->shared, img->mdt->NBkw, img->mdt->CBsize);
    IMGID exist_img = imgid_make_from_name(img->name);
    resolveIMGID(&exist_img, ERRMODE_NULL, dcimg, dcnimg);

    if (img->stream_must_new && exist_img.ID != -1)
    {
        fprintf(stderr, "ERROR: @N modifier -- image \"%s\" already exists\n", img->name);
        imgid_free(&exist_img);
        img->ID = -1;
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }

    if (exist_img.ID == -1)
    {
        imgid_free(&exist_img);
        img->ID     = next_avail_image_ID(img->ID);
        errno_t ret = ImageStreamIO_createIm(&dcimg[img->ID], img->name, img->mdt->naxis,
                                             img->mdt->size, img->mdt->datatype, img->mdt->shared,
                                             img->mdt->NBkw, img->mdt->CBsize);
        if (ret != IMAGESTREAMIO_SUCCESS)
        {
            dcimg[img->ID].used = 0;
            img->ID             = -1;
            DEBUG_TRACE_FEXIT();
            return RETURN_FAILURE;
        }
    }
    else
    {
        // Image name already in use — check compatibility
        img->ID = exist_img.ID;
        imgid_free(&exist_img);

        int mismatch = 0;

        if (dcimg[img->ID].md->datatype != img->mdt->datatype)
        {
            printf("\033[33mWARNING:\033[0m"
                   " image \"%s\" type mismatch"
                   " -> re-creating\n",
                   img->name);
            mismatch = 1;
        }

        if (!mismatch && dcimg[img->ID].md->naxis != img->mdt->naxis)
        {
            printf("\033[33mWARNING:\033[0m"
                   " image \"%s\" naxis mismatch"
                   " (%ld vs %ld)"
                   " -> re-creating\n",
                   img->name, (long) dcimg[img->ID].md->naxis, (long) img->mdt->naxis);
            mismatch = 1;
        }

        if (!mismatch)
        {
            for (int i = 0; i < img->mdt->naxis; i++)
            {
                if (dcimg[img->ID].md->size[i] != img->mdt->size[i])
                {
                    printf("\033[33mWARNING:"
                           "\033[0m"
                           " image \"%s\" size"
                           " mismatch axis %d"
                           " (%ld vs %ld)"
                           " -> re-creating\n",
                           img->name, i, (long) dcimg[img->ID].md->size[i],
                           (long) img->mdt->size[i]);
                    mismatch = 1;
                    break;
                }
            }
        }

        if (mismatch)
        {
            delete_image_ID(img->name, DELETE_IMAGE_ERRMODE_WARNING);
            img->ID     = next_avail_image_ID(img->ID);
            errno_t ret = ImageStreamIO_createIm(
                &dcimg[img->ID], img->name, img->mdt->naxis, img->mdt->size, img->mdt->datatype,
                img->mdt->shared, img->mdt->NBkw, img->mdt->CBsize);
            if (ret != IMAGESTREAMIO_SUCCESS)
            {
                dcimg[img->ID].used = 0;
                img->ID             = -1;
                DEBUG_TRACE_FEXIT();
                return RETURN_FAILURE;
            }
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * @brief Create image from explicit parameters
 *
 * Builds an IMGID from arguments and delegates to
 * create_image_ID_IMGID(). This is the legacy
 * string API used by most of the codebase.
 *
 * @param name     Image name
 * @param naxis    Number of dimensions
 * @param size     Array of axis sizes
 * @param datatype Pixel data type token
 * @param shared   1 for shared memory, 0 for local
 * @param NBkw     Number of keyword slots
 * @param CBsize   Circular buffer size
 * @param outID    If non-NULL, receives the slot ID
 * @return RETURN_SUCCESS
 */
errno_t create_image_ID(const char *__restrict name,
                        long      naxis,
                        uint32_t *size,
                        uint8_t   datatype,
                        int       shared,
                        int       NBkw,
                        int       CBsize,
                        imageID  *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->naxis    = naxis;
    img.mdt->datatype = datatype;
    img.mdt->shared   = shared;
    img.mdt->NBkw     = NBkw;
    img.mdt->CBsize   = CBsize;
    for (int i = 0; i < naxis; i++)
    {
        img.mdt->size[i] = size[i];
    }
    img.ID         = (outID != NULL) ? *outID : -1;
    errno_t retval = create_image_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

/**
 * @brief Create 1D image with default precision
 *
 * Uses global dcprecision (0=float, 1=double).
 *
 * @param img  IMGID with name and size[0] set
 * @return RETURN_SUCCESS
 */
errno_t create_1Dimage_ID_IMGID(IMGID *img)
{
    img->mdt->naxis  = 1;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if (dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    if (dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_1Dimage_ID(const char *restrict ID_name, uint32_t xsize, imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_1Dimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_1DCimage_ID_IMGID(IMGID *img)
{
    img->mdt->naxis  = 1;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if (dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if (dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_1DCimage_ID(const char *__restrict ID_name, uint32_t xsize, imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_1DCimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

/**
 * @brief Create 2D image with default precision
 *
 * @param img  IMGID with name, size[0..1] set
 * @return RETURN_SUCCESS
 */
errno_t create_2Dimage_ID_IMGID(IMGID *img)
{
    img->mdt->naxis  = 2;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if (dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    else if (dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_DOUBLE;
    }
    else
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_2Dimage_ID(const char *__restrict ID_name,
                          uint32_t xsize,
                          uint32_t ysize,
                          imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_2Dimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_2Dimage_ID_double_IMGID(IMGID *img)
{
    img->mdt->naxis    = 2;
    img->mdt->datatype = _DATATYPE_DOUBLE;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

errno_t create_2Dimage_ID_double(const char *__restrict ID_name,
                                 uint32_t xsize,
                                 uint32_t ysize,
                                 imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_2Dimage_ID_double_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_2DCimage_ID_IMGID(IMGID *img)
{
    img->mdt->naxis  = 2;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if (dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if (dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

/* 2D complex image */
errno_t create_2DCimage_ID(const char *__restrict ID_name,
                           uint32_t xsize,
                           uint32_t ysize,
                           imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_2DCimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_2DCimage_ID_double_IMGID(IMGID *img)
{
    img->mdt->naxis    = 2;
    img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 2D complex image */
errno_t create_2DCimage_ID_double(const char *__restrict ID_name,
                                  uint32_t xsize,
                                  uint32_t ysize,
                                  imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_2DCimage_ID_double_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3Dimage_ID_float_IMGID(IMGID *img)
{
    img->mdt->naxis    = 3;
    img->mdt->datatype = _DATATYPE_FLOAT;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 3D image, single precision */
errno_t create_3Dimage_ID_float(const char *__restrict ID_name,
                                uint32_t xsize,
                                uint32_t ysize,
                                uint32_t zsize,
                                imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_3Dimage_ID_float_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3Dimage_ID_double_IMGID(IMGID *img)
{
    img->mdt->naxis    = 3;
    img->mdt->datatype = _DATATYPE_DOUBLE;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 3D image, double precision */
errno_t create_3Dimage_ID_double(const char *__restrict ID_name,
                                 uint32_t xsize,
                                 uint32_t ysize,
                                 uint32_t zsize,
                                 imageID *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_3Dimage_ID_double_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3Dimage_ID_IMGID(IMGID *img)
{
    img->mdt->naxis  = 3;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if (dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    if (dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_DOUBLE;
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
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_3Dimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3DCimage_ID_IMGID(IMGID *img)
{
    img->mdt->naxis  = 3;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if (dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if (dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
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
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID           = (outID != NULL) ? *outID : -1;
    errno_t retval   = create_3DCimage_ID_IMGID(&img);
    if (outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t imgid_create_registered(IMGID *img)
{
    if (create_image_ID_IMGID(img) != RETURN_SUCCESS || img->ID == -1)
    {
        img->im = NULL;
        img->md = NULL;
        return RETURN_FAILURE;
    }
    img->im        = &dcimg[img->ID];
    img->md        = img->im->md;
    img->createcnt = img->im->createcnt;
    /* a compatible image of the same name is reused: clear it like a new one */
    memset(img->im->array.raw, 0,
           (size_t) ImageStreamIO_typesize(img->md->datatype) * img->md->nelement);
    return RETURN_SUCCESS;
}
