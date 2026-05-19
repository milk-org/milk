/**
 * @file    fps_ID.c
 * @brief   find fps ID(s) from name
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#include "COREMOD_memory/COREMOD_memory.h"
#else
#include "libmilkdata/milkdata.h"
#include <fps.h>
#endif

/* ID number corresponding to a name */
long fps_ID(const char *name)
{
    long i;
    int  loopOK;
    long tmpID = 0;

    i      = 0;
    loopOK = 1;
    while(loopOK == 1)
    {

        if(dcfpsarr[i].SMfd >= 0)
        {
            // fps in use

            if((strncmp(name, dcfpsarr[i].md->name, strlen(name)) == 0) &&
                    (dcfpsarr[i].md->name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID  = i;
            }
        }

        i++;

        if(i == dcnfps)
        {
            loopOK = 0;
            tmpID  = -1;
        }
    }

    return tmpID;
}

/* next available ID number */
long next_avail_fps_ID()
{

    long ID = -1;

#ifdef _OPENMP
    #pragma omp critical
    {
#endif
        for(long i = 0; i < dcnfps; i++)
        {
            if(dcfpsarr[i].SMfd < 0)
            {
                // fps is unused, lets grab it
                ID = i;
                break;
            }
        }
#ifdef _OPENMP
    }
#endif

    if(ID == -1)
    {
        PRINT_ERROR("ran out of FPS IDs" " (NB_MAX_FPS=%ld)", dcnfps);
    }

    return ID;
}
