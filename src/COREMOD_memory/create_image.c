/**
 * @file    create_image.c
 * @brief   create images and streams
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"
#include "stream_sem.h"








/* creates an image ID */
/* all images should be created by this function */
errno_t create_image_ID(
    const char * __restrict name,
    long        naxis,
    uint32_t   *size,
    uint8_t     datatype,
    int         shared,
    int         NBkw,
    int         CBsize,
    imageID    *outID
)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s %ld %d %d %d %d",
                     name,
                     naxis,
                     (int) datatype,
                     shared,
                     NBkw,
                     CBsize);

    imageID ID;
    if(image_ID(name) == -1)
    {
        ID = next_avail_image_ID(*outID);

        ImageStreamIO_createIm(&data.image[ID],
                               name,
                               naxis,
                               size,
                               datatype,
                               shared,
                               NBkw,
                               CBsize);
    }
    else
    {
        // Cannot create image : name already in use
        ID = image_ID(name);

        if(data.image[ID].md->datatype != datatype)
        {
            FUNC_RETURN_FAILURE("Pre-existing image \"%s\" has wrong type",
                                name);
        }
        if(data.image[ID].md->naxis != naxis)
        {
            FUNC_RETURN_FAILURE("Pre-existing image \"%s\" has wrong naxis",
                                name);
        }

        for(int i = 0; i < naxis; i++)
            if(data.image[ID].md->size[i] != size[i])
            {
                FUNC_RETURN_FAILURE(
                    "Pre-existing image \"%s\" has wrong size: axis %d "
                    ":  %ld  %ld",
                    name,
                    i,
                    (long) data.image[ID].md->size[i],
                    (long) size[i]);
            }
    }

    if(data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



errno_t create_1Dimage_ID(
    const char * restrict ID_name,
    uint32_t xsize,
    imageID *outID
)
{
    DEBUG_TRACE_FSTART();

    imageID  ID    = -1;
    long     naxis = 1;
    uint32_t naxes[1];

    naxes[0] = xsize;

    if(data.precision == 0)
    {
        // single precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_FLOAT,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }
    if(data.precision == 1)
    {
        // double precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_DOUBLE,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



errno_t create_1DCimage_ID(
    const char * __restrict ID_name,
    uint32_t xsize,
    imageID *outID
)
{
    DEBUG_TRACE_FSTART();

    imageID  ID    = -1;
    long     naxis = 1;
    uint32_t naxes[1];

    naxes[0] = xsize;

    if(data.precision == 0)
    {
        // single precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_COMPLEX_FLOAT,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }
    if(data.precision == 1)
    {
        // double precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_COMPLEX_DOUBLE,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



errno_t create_2Dimage_ID(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID  ID    = -1;
    long     naxis = 2;
    uint32_t naxes[2] = { xsize, ysize };

    if(data.precision == 0)
    {
        // single precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_FLOAT,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }
    else if(data.precision == 1)
    {
        // double precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_DOUBLE,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }
    else
    {
        // single precision
        printf(
            "Default precision (%d) invalid value: assuming single "
            "precision\n",
            data.precision);
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_FLOAT,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


errno_t create_2Dimage_ID_double(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID  ID    = -1;
    long     naxis = 2;
    uint32_t naxes[2] = { xsize, ysize };


    FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                      naxis,
                                      naxes,
                                      _DATATYPE_DOUBLE,
                                      data.SHARED_DFT,
                                      NB_KEYWNODE_MAX,
                                      0,
                                      &ID));

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



/* 2D complex image */
errno_t create_2DCimage_ID(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID  ID    = -1;
    long     naxis = 2;
    uint32_t naxes[2] = { xsize, ysize };

    if(data.precision == 0)
    {
        // single precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_COMPLEX_FLOAT,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }
    if(data.precision == 1)
    {
        // double precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_COMPLEX_DOUBLE,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



/* 2D complex image */
errno_t create_2DCimage_ID_double(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID  ID    = -1;
    long     naxis = 2;
    uint32_t naxes[2]  = { xsize, ysize };


    FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                      naxis,
                                      naxes,
                                      _DATATYPE_COMPLEX_DOUBLE,
                                      data.SHARED_DFT,
                                      NB_KEYWNODE_MAX,
                                      0,
                                      &ID));

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* 3D image, single precision */
errno_t create_3Dimage_ID_float(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID  ID    = -1;
    long     naxis = 3;
    uint32_t naxes[3] = { xsize, ysize, zsize };

    //  printf("CREATING 3D IMAGE: %s %ld %ld %ld\n", ID_name, xsize, ysize, zsize);
    //  fflush(stdout);

    FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                      naxis,
                                      naxes,
                                      _DATATYPE_FLOAT,
                                      data.SHARED_DFT,
                                      NB_KEYWNODE_MAX,
                                      0,
                                      &ID));

    //  printf("IMAGE CREATED WITH ID = %ld\n",ID);
    //  fflush(stdout);

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



/* 3D image, double precision */
errno_t create_3Dimage_ID_double(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID  ID;
    long     naxis = 3;
    uint32_t naxes[3] = { xsize, ysize, zsize };

    FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                      naxis,
                                      naxes,
                                      _DATATYPE_DOUBLE,
                                      data.SHARED_DFT,
                                      NB_KEYWNODE_MAX,
                                      0,
                                      &ID));

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




/* 3D image, default precision */
errno_t create_3Dimage_ID(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID   ID    = -1;
    long      naxis = 3;

    uint32_t naxes[3] = { xsize, ysize, zsize };


    printf("CREATE 3D IMAGE SIZE %u %u %u\n", xsize, ysize, zsize);

    if(data.precision == 0)
    {
        // single precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_FLOAT,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }

    if(data.precision == 1)
    {
        // double precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_DOUBLE,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* 3D complex image */
errno_t create_3DCimage_ID(
    const char * __restrict ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID   ID    = -1;
    long      naxis = 3;
    uint32_t naxes[3] = { xsize, ysize, zsize };

    if(data.precision == 0)
    {
        // single precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_COMPLEX_FLOAT,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }

    if(data.precision == 1)
    {
        // double precision
        FUNC_CHECK_RETURN(create_image_ID(ID_name,
                                          naxis,
                                          naxes,
                                          _DATATYPE_COMPLEX_DOUBLE,
                                          data.SHARED_DFT,
                                          NB_KEYWNODE_MAX,
                                          0,
                                          &ID));
    }


    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
