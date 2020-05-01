/**
 * @file    create_image.c
 * @brief   create images and streams
 */


#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"


/* creates an image ID */
/* all images should be created by this function */
imageID create_image_ID(
    const char *name,
    long        naxis,
    uint32_t   *size,
    uint8_t     datatype,
    int         shared,
    int         NBkw
)
{
    imageID ID;
    long    i;


    ID = -1;
    if(image_ID(name) == -1)
    {
        ID = next_avail_image_ID();
        ImageStreamIO_createIm(&data.image[ID], name, naxis, size, datatype, shared,
                               NBkw);
    }
    else
    {
        // Cannot create image : name already in use
        ID = image_ID(name);

        if(data.image[ID].md[0].datatype != datatype)
        {
            fprintf(stderr, "%c[%d;%dm ERROR: [ %s %s %d ] %c[%d;m\n", (char) 27, 1, 31,
                    __FILE__, __func__, __LINE__, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm Pre-existing image \"%s\" has wrong type %c[%d;m\n",
                    (char) 27, 1, 31, name, (char) 27, 0);
            exit(0);
        }
        if(data.image[ID].md[0].naxis != naxis)
        {
            fprintf(stderr, "%c[%d;%dm ERROR: [ %s %s %d ] %c[%d;m\n", (char) 27, 1, 31,
                    __FILE__, __func__, __LINE__, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm Pre-existing image \"%s\" has wrong naxis %c[%d;m\n",
                    (char) 27, 1, 31, name, (char) 27, 0);
            exit(0);
        }

        for(i = 0; i < naxis; i++)
            if(data.image[ID].md[0].size[i] != size[i])
            {
                fprintf(stderr, "%c[%d;%dm ERROR: [ %s %s %d ] %c[%d;m\n", (char) 27, 1, 31,
                        __FILE__, __func__, __LINE__, (char) 27, 0);
                fprintf(stderr, "%c[%d;%dm Pre-existing image \"%s\" has wrong size %c[%d;m\n",
                        (char) 27, 1, 31, name, (char) 27, 0);
                fprintf(stderr, "Axis %ld :  %ld  %ld\n", i,
                        (long) data.image[ID].md[0].size[i], (long) size[i]);
                exit(0);
            }
    }

    if(data.MEM_MONITOR == 1)
    {
        list_image_ID_ncurses();
    }

    return ID;
}




imageID create_1Dimage_ID(
    const char *ID_name,
    uint32_t    xsize
)
{
    imageID ID = -1;
    long naxis = 1;
    uint32_t naxes[1];

    naxes[0] = xsize;

    if(data.precision == 0)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_FLOAT, data.SHARED_DFT,
                             data.NBKEWORD_DFT);    // single precision
    }
    if(data.precision == 1)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_DOUBLE, data.SHARED_DFT,
                             data.NBKEWORD_DFT);    // double precision
    }

    return ID;
}



imageID create_1DCimage_ID(
    const char *ID_name,
    uint32_t    xsize
)
{
    imageID ID = -1;
    long naxis = 1;
    uint32_t naxes[1];

    naxes[0] = xsize;

    if(data.precision == 0)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_COMPLEX_FLOAT,
                             data.SHARED_DFT, data.NBKEWORD_DFT);    // single precision
    }
    if(data.precision == 1)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_COMPLEX_DOUBLE,
                             data.SHARED_DFT, data.NBKEWORD_DFT);    // double precision
    }

    return ID;
}



imageID create_2Dimage_ID(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize
)
{
    imageID ID = -1;
    long naxis = 2;
    uint32_t naxes[2];

    naxes[0] = xsize;
    naxes[1] = ysize;

    if(data.precision == 0)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_FLOAT, data.SHARED_DFT,
                             data.NBKEWORD_DFT);    // single precision
    }
    else if(data.precision == 1)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_DOUBLE, data.SHARED_DFT,
                             data.NBKEWORD_DFT);    // double precision
    }
    else
    {
        printf("Default precision (%d) invalid value: assuming single precision\n",
               data.precision);
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_FLOAT, data.SHARED_DFT,
                             data.NBKEWORD_DFT); // single precision
    }

    return ID;
}




imageID create_2Dimage_ID_double(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize
)
{
    imageID ID = -1;
    long naxis = 2;
    uint32_t naxes[2];

    naxes[0] = xsize;
    naxes[1] = ysize;

    ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_DOUBLE, data.SHARED_DFT,
                         data.NBKEWORD_DFT);

    return ID;
}


/* 2D complex image */
imageID create_2DCimage_ID(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize
)
{
    imageID ID = -1;
    long naxis = 2;
    uint32_t naxes[2];

    naxes[0] = xsize;
    naxes[1] = ysize;

    if(data.precision == 0)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_COMPLEX_FLOAT,
                             data.SHARED_DFT, data.NBKEWORD_DFT);    // single precision
    }
    if(data.precision == 1)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_COMPLEX_DOUBLE,
                             data.SHARED_DFT, data.NBKEWORD_DFT);    // double precision
    }

    return ID;
}



/* 2D complex image */
imageID create_2DCimage_ID_double(
    const char    *ID_name,
    uint32_t       xsize,
    uint32_t       ysize
)
{
    imageID ID = -1;
    long naxis = 2;
    uint32_t naxes[2];

    naxes[0] = xsize;
    naxes[1] = ysize;

    ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_COMPLEX_DOUBLE,
                         data.SHARED_DFT, data.NBKEWORD_DFT); // double precision

    return ID;
}



/* 3D image, single precision */
imageID create_3Dimage_ID_float(
    const char *ID_name,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
{
    imageID ID = -1;
    long naxis = 3;
    uint32_t naxes[3];

    naxes[0] = xsize;
    naxes[1] = ysize;
    naxes[2] = zsize;

    //  printf("CREATING 3D IMAGE: %s %ld %ld %ld\n", ID_name, xsize, ysize, zsize);
    //  fflush(stdout);

    ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_FLOAT, data.SHARED_DFT,
                         data.NBKEWORD_DFT); // single precision

    //  printf("IMAGE CREATED WITH ID = %ld\n",ID);
    //  fflush(stdout);

    return ID;
}


/* 3D image, double precision */
imageID create_3Dimage_ID_double(
    const char *ID_name,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
{
    imageID ID;
    long naxis = 3;
    uint32_t naxes[3];

    naxes[0] = xsize;
    naxes[1] = ysize;
    naxes[2] = zsize;

    ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_DOUBLE, data.SHARED_DFT,
                         data.NBKEWORD_DFT); // double precision

    return ID;
}



/* 3D image, default precision */
imageID create_3Dimage_ID(
    const char *ID_name,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
{
    imageID ID = -1;
    long naxis = 3;
    uint32_t *naxes;


    naxes = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    naxes[0] = xsize;
    naxes[1] = ysize;
    naxes[2] = zsize;

    if(data.precision == 0)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_FLOAT, data.SHARED_DFT,
                             data.NBKEWORD_DFT); // single precision
    }

    if(data.precision == 1)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_DOUBLE, data.SHARED_DFT,
                             data.NBKEWORD_DFT); // double precision
    }

    free(naxes);

    return ID;
}


/* 3D complex image */
imageID create_3DCimage_ID(
    const char *ID_name,
    uint32_t xsize,
    uint32_t ysize,
    uint32_t zsize
)
{
    imageID ID = -1;
    long naxis = 3;
    uint32_t *naxes;


    naxes = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    naxes[0] = xsize;
    naxes[1] = ysize;
    naxes[2] = zsize;

    if(data.precision == 0)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_COMPLEX_FLOAT,
                             data.SHARED_DFT, data.NBKEWORD_DFT); // single precision
    }

    if(data.precision == 1)
    {
        ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_COMPLEX_DOUBLE,
                             data.SHARED_DFT, data.NBKEWORD_DFT); // double precision
    }

    free(naxes);

    return ID;
}






