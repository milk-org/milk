/**
 * @file    image_ID.c
 * @brief   find image ID(s) from name
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#endif

/* ID number corresponding to a name */
imageID image_ID(const char *name, IMAGE *imagearray, long NB_images)
{
    DEBUG_TRACE_FSTART();

    imageID i;
    int     loopOK;
    imageID tmpID = 0;

    if(imagearray == NULL)
    {
        return -1;
    }

    i      = 0;
    loopOK = 1;
    while(loopOK == 1)
    {
        if(imagearray[i].used == 1)
        {
            if((strncmp(name, imagearray[i].name, strlen(name)) == 0) &&
                    (imagearray[i].name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID  = i;
                clock_gettime(CLOCK_MILK,
                              &imagearray[i].md[0].lastaccesstime);
            }
        }
        i++;

        if(i == NB_images)
        {
            loopOK = 0;
            tmpID  = -1;
        }
    }

    DEBUG_TRACEPOINT("FOUT %s -> %ld", name, tmpID);
    DEBUG_TRACE_FEXIT();
    return tmpID;
}

/* ID number corresponding to a name */
imageID image_ID_noaccessupdate(const char *name, IMAGE *imagearray, long NB_images)
{
    DEBUG_TRACE_FSTART();

    imageID i;
    imageID tmpID = 0;
    int     loopOK;

    if(imagearray == NULL)
    {
        return -1;
    }

    i      = 0;
    loopOK = 1;
    while(loopOK == 1)
    {
        if(imagearray[i].used == 1)
        {
            if((strncmp(name, imagearray[i].name, strlen(name)) == 0) &&
                    (imagearray[i].name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID  = i;
            }
        }
        i++;

        if(i == NB_images)
        {
            loopOK = 0;
            tmpID  = -1;
        }
    }

    DEBUG_TRACE_FEXIT();
    return tmpID;
}

/* next available ID number */
imageID next_avail_image_ID(
    imageID preferredID
)
{
    DEBUG_TRACE_FSTART();

    imageID i;
    imageID ID = -1;

    if ( (preferredID > -1)
            && (preferredID<dcnimg)
            && (dcimg[preferredID].used == 0) )
    {
        ID = preferredID;
        dcimg[ID].used = 1;
    }
    else
    {


#ifdef _OPENMP
        #pragma omp critical
        {
#endif
            for(i = 0; i < dcnimg; i++)
            {
                if(dcimg[i].used == 0)
                {
                    ID                  = i;
                    dcimg[ID].used = 1;
                    break;
                }
            }
#ifdef _OPENMP
        }
#endif

    }
    if(ID == -1)
    {
        printf("ERROR: ran out of image IDs - cannot allocate new ID\n");
        printf("NB_MAX_IMAGE should be increased above current value (%ld)\n",
               dcnimg);
        exit(0);
    }

    DEBUG_TRACEPOINT("FOUT ID : %ld", ID);

    DEBUG_TRACE_FEXIT();
    return ID;
}
