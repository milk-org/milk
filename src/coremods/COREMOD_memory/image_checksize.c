/**
 * @file    image_checksize.c
 * @brief   check image size
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#endif
#include "image_ID.h"
#include "COREMOD_memory/imageID.h"

//  check only is size > 0
/**
 * @brief Verify that an image has the expected 2D size.
 */
int check_2Dsize(const char *ID_name, uint32_t xsize, uint32_t ysize)
{
    int   retval;
    IMGID img = imgid_make_from_name(ID_name);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1)
    {
        return 0;
    }

    retval = 1;
    if (img.im->md[0].naxis != 2)
    {
        retval = 0;
    }
    if (retval == 1)
    {
        if ((xsize > 0) && (img.im->md[0].size[0] != xsize))
        {
            retval = 0;
        }
        if ((ysize > 0) && (img.im->md[0].size[1] != ysize))
        {
            retval = 0;
        }
    }

    return retval;
}

/**
 * @brief Verify that an image has the expected 3D size.
 */
int check_3Dsize(const char *ID_name, uint32_t xsize, uint32_t ysize, uint32_t zsize)
{
    int   retval;
    IMGID img = imgid_make_from_name(ID_name);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1)
    {
        return 0;
    }

    retval = 1;
    if (img.im->md[0].naxis != 3)
    {
        /*      printf("Wrong naxis : %ld - should be 3\n",img.im->md[0].naxis);*/
        retval = 0;
    }
    if (retval == 1)
    {
        if ((xsize > 0) && (img.im->md[0].size[0] != xsize))
        {
            /*	  printf("Wrong xsize : %ld - should be %ld\n",img.im->md[0].size[0],xsize);*/
            retval = 0;
        }
        if ((ysize > 0) && (img.im->md[0].size[1] != ysize))
        {
            /*	  printf("Wrong ysize : %ld - should be %ld\n",img.im->md[0].size[1],ysize);*/
            retval = 0;
        }
        if ((zsize > 0) && (img.im->md[0].size[2] != zsize))
        {
            /*	  printf("Wrong zsize : %ld - should be %ld\n",img.im->md[0].size[2],zsize);*/
            retval = 0;
        }
    }
    /*  printf("CHECK = %d\n",value);*/

    return retval;
}

int COREMOD_MEMORY_check_2Dsize(const char *IDname, uint32_t xsize, uint32_t ysize)
{
    int   sizeOK = 1; // 1 if size matches
    IMGID img    = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1)
    {
        return 0;
    }
    if (img.im->md[0].naxis != 2)
    {
        printf("WARNING : image %s naxis = %d does not match expected value "
               "2\n",
               IDname, (int) img.im->md[0].naxis);
        sizeOK = 0;
    }
    if ((xsize > 0) && (img.im->md[0].size[0] != xsize))
    {
        printf("WARNING : image %s xsize = %d does not match expected value "
               "%d\n",
               IDname, (int) img.im->md[0].size[0], (int) xsize);
        sizeOK = 0;
    }
    if ((ysize > 0) && (img.im->md[0].size[1] != ysize))
    {
        printf("WARNING : image %s ysize = %d does not match expected value "
               "%d\n",
               IDname, (int) img.im->md[0].size[1], (int) ysize);
        sizeOK = 0;
    }

    return sizeOK;
}

int COREMOD_MEMORY_check_3Dsize(const char *IDname, uint32_t xsize, uint32_t ysize, uint32_t zsize)
{
    int   sizeOK = 1; // 1 if size matches
    IMGID img    = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1)
    {
        return 0;
    }
    if (img.im->md[0].naxis != 3)
    {
        printf("WARNING : image %s naxis = %d does not match expected value "
               "3\n",
               IDname, (int) img.im->md[0].naxis);
        sizeOK = 0;
    }
    if ((xsize > 0) && (img.im->md[0].size[0] != xsize))
    {
        printf("WARNING : image %s xsize = %d does not match expected value "
               "%d\n",
               IDname, (int) img.im->md[0].size[0], (int) xsize);
        sizeOK = 0;
    }
    if ((ysize > 0) && (img.im->md[0].size[1] != ysize))
    {
        printf("WARNING : image %s ysize = %d does not match expected value "
               "%d\n",
               IDname, (int) img.im->md[0].size[1], (int) ysize);
        sizeOK = 0;
    }
    if ((zsize > 0) && (img.im->md[0].size[2] != zsize))
    {
        printf("WARNING : image %s zsize = %d does not match expected value "
               "%d\n",
               IDname, (int) img.im->md[0].size[2], (int) zsize);
        sizeOK = 0;
    }

    return sizeOK;
}
