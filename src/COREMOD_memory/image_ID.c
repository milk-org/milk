/**
 * @file    image_ID.c
 * @brief   find image ID(s) from name
 */


#include "CommandLineInterface/CLIcore.h"



/* ID number corresponding to a name */
imageID image_ID(
    const char *name
)
{
    DEBUG_TRACE_FSTART();

    imageID    i;
    int        loopOK;
    imageID    tmpID = 0;

    i = 0;
    loopOK = 1;
    while(loopOK == 1)
    {
        if(data.image[i].used == 1)
        {
            if((strncmp(name, data.image[i].name, strlen(name)) == 0)
                    && (data.image[i].name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID = i;
                clock_gettime(CLOCK_REALTIME, &data.image[i].md[0].lastaccesstime);
            }
        }
        i++;

        if(i == data.NB_MAX_IMAGE)
        {
            loopOK = 0;
            tmpID = -1;
        }
    }

    DEBUG_TRACE_FEXIT();
    return tmpID;
}


/* ID number corresponding to a name */
imageID image_ID_noaccessupdate(
    const char *name
)
{
    DEBUG_TRACE_FSTART();

    imageID   i;
    imageID   tmpID = 0;
    int       loopOK;

    i = 0;
    loopOK = 1;
    while(loopOK == 1)
    {
        if(data.image[i].used == 1)
        {
            if((strncmp(name, data.image[i].name, strlen(name)) == 0)
                    && (data.image[i].name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID = i;
            }
        }
        i++;

        if(i == data.NB_MAX_IMAGE)
        {
            loopOK = 0;
            tmpID = -1;
        }
    }

    DEBUG_TRACE_FEXIT();
    return tmpID;
}




/* next available ID number */
imageID next_avail_image_ID()
{
    DEBUG_TRACE_FSTART();

    imageID i;
    imageID ID = -1;

# ifdef _OPENMP
    #pragma omp critical
    {
#endif
        for(i = 0; i < data.NB_MAX_IMAGE; i++)
        {
            if(data.image[i].used == 0)
            {
                ID = i;
                data.image[ID].used = 1;
                break;
            }
        }
# ifdef _OPENMP
    }
# endif

    if(ID == -1)
    {
        printf("ERROR: ran out of image IDs - cannot allocate new ID\n");
        printf("NB_MAX_IMAGE should be increased above current value (%ld)\n",
               data.NB_MAX_IMAGE);
        exit(0);
    }

    DEBUG_TRACEPOINT("FOUT ID : %ld", ID);

    DEBUG_TRACE_FEXIT();
    return ID;
}
