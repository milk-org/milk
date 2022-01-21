/**
 * @file    image_checksize.c
 * @brief   check image size
 */

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"

//  check only is size > 0
int check_2Dsize(const char *ID_name, uint32_t xsize, uint32_t ysize)
{
    int     retval;
    imageID ID;

    retval = 1;
    ID     = image_ID(ID_name);
    if (data.image[ID].md[0].naxis != 2)
    {
        retval = 0;
    }
    if (retval == 1)
    {
        if ((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
        {
            retval = 0;
        }
        if ((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
        {
            retval = 0;
        }
    }

    return retval;
}

int check_3Dsize(const char *ID_name,
                 uint32_t    xsize,
                 uint32_t    ysize,
                 uint32_t    zsize)
{
    int     retval;
    imageID ID;

    retval = 1;
    ID     = image_ID(ID_name);
    if (data.image[ID].md[0].naxis != 3)
    {
        /*      printf("Wrong naxis : %ld - should be 3\n",data.image[ID].md[0].naxis);*/
        retval = 0;
    }
    if (retval == 1)
    {
        if ((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
        {
            /*	  printf("Wrong xsize : %ld - should be %ld\n",data.image[ID].md[0].size[0],xsize);*/
            retval = 0;
        }
        if ((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
        {
            /*	  printf("Wrong ysize : %ld - should be %ld\n",data.image[ID].md[0].size[1],ysize);*/
            retval = 0;
        }
        if ((zsize > 0) && (data.image[ID].md[0].size[2] != zsize))
        {
            /*	  printf("Wrong zsize : %ld - should be %ld\n",data.image[ID].md[0].size[2],zsize);*/
            retval = 0;
        }
    }
    /*  printf("CHECK = %d\n",value);*/

    return retval;
}

int COREMOD_MEMORY_check_2Dsize(const char *IDname,
                                uint32_t    xsize,
                                uint32_t    ysize)
{
    int     sizeOK = 1; // 1 if size matches
    imageID ID;

    ID = image_ID(IDname);
    if (data.image[ID].md[0].naxis != 2)
    {
        printf(
            "WARNING : image %s naxis = %d does not match expected value "
            "2\n",
            IDname,
            (int) data.image[ID].md[0].naxis);
        sizeOK = 0;
    }
    if ((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
    {
        printf(
            "WARNING : image %s xsize = %d does not match expected value "
            "%d\n",
            IDname,
            (int) data.image[ID].md[0].size[0],
            (int) xsize);
        sizeOK = 0;
    }
    if ((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
    {
        printf(
            "WARNING : image %s ysize = %d does not match expected value "
            "%d\n",
            IDname,
            (int) data.image[ID].md[0].size[1],
            (int) ysize);
        sizeOK = 0;
    }

    return sizeOK;
}

int COREMOD_MEMORY_check_3Dsize(const char *IDname,
                                uint32_t    xsize,
                                uint32_t    ysize,
                                uint32_t    zsize)
{
    int     sizeOK = 1; // 1 if size matches
    imageID ID;

    ID = image_ID(IDname);
    if (data.image[ID].md[0].naxis != 3)
    {
        printf(
            "WARNING : image %s naxis = %d does not match expected value "
            "3\n",
            IDname,
            (int) data.image[ID].md[0].naxis);
        sizeOK = 0;
    }
    if ((xsize > 0) && (data.image[ID].md[0].size[0] != xsize))
    {
        printf(
            "WARNING : image %s xsize = %d does not match expected value "
            "%d\n",
            IDname,
            (int) data.image[ID].md[0].size[0],
            (int) xsize);
        sizeOK = 0;
    }
    if ((ysize > 0) && (data.image[ID].md[0].size[1] != ysize))
    {
        printf(
            "WARNING : image %s ysize = %d does not match expected value "
            "%d\n",
            IDname,
            (int) data.image[ID].md[0].size[1],
            (int) ysize);
        sizeOK = 0;
    }
    if ((zsize > 0) && (data.image[ID].md[0].size[2] != zsize))
    {
        printf(
            "WARNING : image %s zsize = %d does not match expected value "
            "%d\n",
            IDname,
            (int) data.image[ID].md[0].size[2],
            (int) zsize);
        sizeOK = 0;
    }

    return sizeOK;
}
