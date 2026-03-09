/**
 * @file    create_image.c
 * @brief   create images and streams
 */
#include "CLIcore.h"
#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "list_image.h"
#include "stream_sem.h"

/* creates an image ID */

/* all images should be created by this function */
errno_t create_image_ID_IMGID(
    IMGID *img
)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s %ld %d %d %d %d",
                     img->name,
                     (long) img->mdt->naxis,
                     (int) img->mdt->datatype,
                     img->mdt->shared,
                     img->mdt->NBkw,
                     img->mdt->CBsize);
    if(image_ID(img->name, dcimg, dcnimg) == -1)
    {
        img->ID = next_avail_image_ID(img->ID);
        ImageStreamIO_createIm(&dcimg[img->ID],
                               img->name,
                               img->mdt->naxis,
                               img->mdt->size,
                               img->mdt->datatype,
                               img->mdt->shared,
                               img->mdt->NBkw,
                               img->mdt->CBsize);
    }
    else
    {
        // Image name already in use — check compatibility
        img->ID = image_ID(img->name,
                           dcimg,
                           dcnimg);

        int mismatch = 0;

        if (dcimg[img->ID].md->datatype
            != img->mdt->datatype)
        {
            printf("\033[33mWARNING:\033[0m"
                   " image \"%s\" type mismatch"
                   " -> re-creating\n",
                   img->name);
            mismatch = 1;
        }

        if (!mismatch &&
            dcimg[img->ID].md->naxis
            != img->mdt->naxis)
        {
            printf("\033[33mWARNING:\033[0m"
                   " image \"%s\" naxis mismatch"
                   " (%ld vs %ld)"
                   " -> re-creating\n",
                   img->name,
                   (long) dcimg[img->ID]
                       .md->naxis,
                   (long) img->mdt->naxis);
            mismatch = 1;
        }

        if (!mismatch)
        {
            for (int i = 0;
                 i < img->mdt->naxis; i++)
            {
                if (dcimg[img->ID].md
                        ->size[i]
                    != img->mdt->size[i])
                {
                    printf(
                        "\033[33mWARNING:"
                        "\033[0m"
                        " image \"%s\" size"
                        " mismatch axis %d"
                        " (%ld vs %ld)"
                        " -> re-creating\n",
                        img->name, i,
                        (long) dcimg[
                            img->ID]
                            .md->size[i],
                        (long) img->mdt
                            ->size[i]);
                    mismatch = 1;
                    break;
                }
            }
        }

        if (mismatch)
        {
            delete_image_ID(
                img->name,
                DELETE_IMAGE_ERRMODE_WARNING);
            img->ID = next_avail_image_ID(
                img->ID);
            ImageStreamIO_createIm(
                &dcimg[img->ID],
                img->name,
                img->mdt->naxis,
                img->mdt->size,
                img->mdt->datatype,
                img->mdt->shared,
                img->mdt->NBkw,
                img->mdt->CBsize);
        }
    }
    if(dcmemmon == 1)
    {
        list_image_ID_ncurses();
    }
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t create_image_ID(
    const char *__restrict name,
    long        naxis,
    uint32_t   *size,
    uint8_t     datatype,
    int         shared,
    int         NBkw,
    int         CBsize,
    imageID    *outID
)
{
    IMGID img = imgid_make();
    strncpy(img.name, name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->naxis    = naxis;
    img.mdt->datatype = datatype;
    img.mdt->shared   = shared;
    img.mdt->NBkw     = NBkw;
    img.mdt->CBsize   = CBsize;
    for(int i = 0; i < naxis; i++)
    {
        img.mdt->size[i] = size[i];
    }
    img.ID = (outID != NULL) ? *outID : -1;
    errno_t retval = create_image_ID_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_1Dimage_ID_IMGID(
    IMGID *img
)
{
    img->mdt->naxis  = 1;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if(dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    if(dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_1Dimage_ID(
    const char *restrict ID_name,
    uint32_t xsize,
    imageID *outID
)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_1Dimage_ID_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_1DCimage_ID_IMGID(
    IMGID *img
)
{
    img->mdt->naxis  = 1;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if(dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if(dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_1DCimage_ID(
    const char *__restrict ID_name,
    uint32_t xsize,
    imageID *outID
)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_1DCimage_ID_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_2Dimage_ID_IMGID(
    IMGID *img
)
{
    img->mdt->naxis = 2;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if(dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    else if(dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_DOUBLE;
    }
    else
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    return create_image_ID_IMGID(img);
}

errno_t create_2Dimage_ID(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2Dimage_ID_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_2Dimage_ID_double_IMGID(
    IMGID *img
)
{
    img->mdt->naxis    = 2;
    img->mdt->datatype = _DATATYPE_DOUBLE;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

errno_t create_2Dimage_ID_double(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2Dimage_ID_double_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_2DCimage_ID_IMGID(
    IMGID *img
)
{
    img->mdt->naxis  = 2;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if(dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if(dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

/* 2D complex image */
errno_t create_2DCimage_ID(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2DCimage_ID_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_2DCimage_ID_double_IMGID(
    IMGID *img
)
{
    img->mdt->naxis    = 2;
    img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 2D complex image */
errno_t create_2DCimage_ID_double(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_2DCimage_ID_double_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3Dimage_ID_float_IMGID(
    IMGID *img
)
{
    img->mdt->naxis    = 3;
    img->mdt->datatype = _DATATYPE_FLOAT;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 3D image, single precision */
errno_t create_3Dimage_ID_float(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3Dimage_ID_float_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3Dimage_ID_double_IMGID(
    IMGID *img
)
{
    img->mdt->naxis    = 3;
    img->mdt->datatype = _DATATYPE_DOUBLE;
    img->mdt->shared   = dcshareddft;
    img->mdt->NBkw     = NB_KEYWNODE_MAX;
    img->mdt->CBsize   = 0;
    return create_image_ID_IMGID(img);
}

/* 3D image, double precision */
errno_t create_3Dimage_ID_double(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3Dimage_ID_double_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3Dimage_ID_IMGID(
    IMGID *img
)
{
    img->mdt->naxis  = 3;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if(dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_FLOAT;
    }
    if(dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

/* 3D image, default precision */
errno_t create_3Dimage_ID(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3Dimage_ID_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}

errno_t create_3DCimage_ID_IMGID(
    IMGID *img
)
{
    img->mdt->naxis  = 3;
    img->mdt->shared = dcshareddft;
    img->mdt->NBkw   = NB_KEYWNODE_MAX;
    img->mdt->CBsize = 0;
    if(dcprecision == 0)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_FLOAT;
    }
    if(dcprecision == 1)
    {
        img->mdt->datatype = _DATATYPE_COMPLEX_DOUBLE;
    }
    return create_image_ID_IMGID(img);
}

/* 3D complex image */
errno_t create_3DCimage_ID(
    const char *__restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    IMGID img = imgid_make();
    strncpy(img.name, ID_name, STRINGMAXLEN_IMAGE_NAME - 1);
    img.mdt->size[0] = xsize;
    img.mdt->size[1] = ysize;
    img.mdt->size[2] = zsize;
    img.ID      = (outID != NULL) ? *outID : -1;
    errno_t retval = create_3DCimage_ID_IMGID(&img);
    if(outID != NULL)
    {
        *outID = img.ID;
    }
    imgid_free(&img);
    return retval;
}
