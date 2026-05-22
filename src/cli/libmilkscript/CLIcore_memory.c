/**
 * @file CLIcore_memory.c
 * @brief Clicore memory module
 */

#include "CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

/**
 * @brief Grow image and variable arrays when nearly
 *        full.
 *
 * Maintains a buffer of NB_IMAGES_BUFFER free slots
 * above the current count. When the buffer is
 * exhausted, realloc() extends the array by
 * NB_IMAGES_BUFFER_REALLOC entries.
 *
 * Same logic applies to the VARIABLE array.
 * No-op when DATA_STATIC_ALLOC is defined.
 */
errno_t memory_re_alloc()
{
    /* keeps the number of images addresses available
     *  NB_IMAGES_BUFFER above the number of used images
     */

#ifdef DATA_STATIC_ALLOC
    //printf("image static allocation mode\n");
    //fflush(stdout);
#else
    //printf("image dynamic allocation mode\n");
    //fflush(stdout);

    int current_NBimage = compute_nb_image();

    //printf("DYNAMIC ALLOC. Current = %d, buffer = %d, max = %ld\n", current_NBimage, NB_IMAGES_BUFFER, dcnimg);
    //fflush(stdout);

    if ((current_NBimage + NB_IMAGES_BUFFER) > dcnimg)
    {
        long   tmplong;
        IMAGE *ptrtmp;

        //   if(dcdebug>0)
        //    {
        printf("%p IMAGE STRUCT SIZE = %ld\n", dcimg, (long) sizeof(IMAGE));
        printf("REALLOCATING IMAGE DATA BUFFER: %ld -> %ld\n", dcnimg,
               dcnimg + NB_IMAGES_BUFFER_REALLOC);
        fflush(stdout);
        //    }
        long new_dcnimg = dcnimg + NB_IMAGES_BUFFER_REALLOC;
        ptrtmp          = (IMAGE *) realloc(dcimg, sizeof(IMAGE) * new_dcnimg);
        if (ptrtmp == NULL)
        {
            PRINT_ERROR("Reallocation of dcimg has failed - exiting program");
            return -1;
        }
        if (dcdebug > 0)
        {
            printf("NEW POINTER = %p\n", ptrtmp);
            fflush(stdout);
        }
        tmplong = dcnimg;
        dcnimg  = new_dcnimg;
        dcimg   = ptrtmp;
        if (dcdebug > 0)
        {
            printf("REALLOCATION DONE\n");
            fflush(stdout);
        }

        imageID i;
        for (i = tmplong; i < dcnimg; i++)
        {
            dcimg[i].used      = 0;
            dcimg[i].createcnt = 0;
            dcimg[i].shmfd     = -1;
            dcimg[i].memsize   = 0;
            dcimg[i].semptr    = NULL;
            dcimg[i].semlog    = NULL;
        }
    }
#endif

    /* keeps the number of variables addresses available
     *  NB_VARIABLES_BUFFER above the number of used variables
     */

#ifdef DATA_STATIC_ALLOC
    // variable static allocation mode
#else
    if ((compute_nb_variable() + NB_VARIABLES_BUFFER) > dcnvar)
    {
        long tmplong;

        if (dcdebug > 0)
        {
            printf("REALLOCATING VARIABLE DATA BUFFER\n");
            fflush(stdout);
        }
        long      new_dcnvar = dcnvar + NB_VARIABLES_BUFFER_REALLOC;
        VARIABLE *ptrtmp     = (VARIABLE *) realloc(dcvar, sizeof(VARIABLE) * new_dcnvar);
        if (ptrtmp == NULL)
        {
            PRINT_ERROR("Reallocation of dcvar has failed - exiting program");
            return -1;
        }
        tmplong = dcnvar;
        dcnvar  = new_dcnvar;
        dcvar   = ptrtmp;

        int i;
        for (i = tmplong; i < dcnvar; i++)
        {
            dcvar[i].used = 0;
            dcvar[i].type = -1;
        }
    }
#endif

    return RETURN_SUCCESS;
}
