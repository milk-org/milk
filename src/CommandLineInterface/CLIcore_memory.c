#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"



errno_t memory_re_alloc()
{
    /* keeps the number of images addresses available
     *  NB_IMAGES_BUFFER above the number of used images
     */


#ifdef DATA_STATIC_ALLOC
    // image static allocation mode
#else
    int current_NBimage = compute_nb_image(data);

    //	printf("DYNAMIC ALLOC. Current = %d, buffer = %d, max = %d\n", current_NBimage, NB_IMAGES_BUFFER, data.NB_MAX_IMAGE);///

    if((current_NBimage + NB_IMAGES_BUFFER) > data.NB_MAX_IMAGE)
    {
        long tmplong;
        IMAGE *ptrtmp;

        //   if(data.Debug>0)
        //    {
        printf("%p IMAGE STRUCT SIZE = %ld\n", data.image, (long) sizeof(IMAGE));
        printf("REALLOCATING IMAGE DATA BUFFER: %ld -> %ld\n", data.NB_MAX_IMAGE,
               data.NB_MAX_IMAGE + NB_IMAGES_BUFFER_REALLOC);
        fflush(stdout);
        //    }
        tmplong = data.NB_MAX_IMAGE;
        data.NB_MAX_IMAGE = data.NB_MAX_IMAGE + NB_IMAGES_BUFFER_REALLOC;
        ptrtmp = (IMAGE *) realloc(data.image, sizeof(IMAGE) * data.NB_MAX_IMAGE);
        if(data.Debug > 0)
        {
            printf("NEW POINTER = %p\n", ptrtmp);
            fflush(stdout);
        }
        data.image = ptrtmp;
        if(data.image == NULL)
        {
            PRINT_ERROR("Reallocation of data.image has failed - exiting program");
            return -1;      //  exit(0);
        }
        if(data.Debug > 0)
        {
            printf("REALLOCATION DONE\n");
            fflush(stdout);
        }

        imageID i;
        for(i = tmplong; i < data.NB_MAX_IMAGE; i++)
        {
            data.image[i].used    = 0;
            data.image[i].shmfd   = -1;
            data.image[i].memsize = 0;
            data.image[i].semptr  = NULL;
            data.image[i].semlog  = NULL;
        }
    }
#endif


    /* keeps the number of variables addresses available
     *  NB_VARIABLES_BUFFER above the number of used variables
     */


#ifdef DATA_STATIC_ALLOC
    // variable static allocation mode
#else
    if((compute_nb_variable(data) + NB_VARIABLES_BUFFER) > data.NB_MAX_VARIABLE)
    {
        long tmplong;

        if(data.Debug > 0)
        {
            printf("REALLOCATING VARIABLE DATA BUFFER\n");
            fflush(stdout);
        }
        tmplong = data.NB_MAX_VARIABLE;
        data.NB_MAX_VARIABLE = data.NB_MAX_VARIABLE + NB_VARIABLES_BUFFER_REALLOC;
        data.variable = (VARIABLE *) realloc(data.variable,
                                             sizeof(VARIABLE) * data.NB_MAX_VARIABLE);
        if(data.variable == NULL)
        {
            PRINT_ERROR("Reallocation of data.variable has failed - exiting program");
            return -1;   // exit(0);
        }

        int i;
        for(i = tmplong; i < data.NB_MAX_VARIABLE; i++)
        {
            data.variable[i].used = 0;
            data.variable[i].type = -1;
        }
    }
#endif


    return RETURN_SUCCESS;
}





