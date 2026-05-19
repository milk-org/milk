/**
 * @file    fps_load.c
 * @brief   Load FPS
 */

#include "fps.h"
#include "fps_globals.h"


// Forward declaration or include for fps_ID
/**
 * @brief Look up an FPS index by name.
 *
 * Searches the connected FPS array for the
 * given name. Returns -1 if not found.
 */
long fps_ID(const char *fpsname);


/**
 * @brief Loads an FPS structure from shared memory by name.
 */
long function_parameter_structure_load(char *fpsname)
{
    long fpsID;

    DEBUG_TRACEPOINT("loading FPS %s", fpsname);

    fpsID = fps_ID(fpsname);

    if(fpsID == -1)
    {
        // not found, searching

        // next fpsID available
        fpsID = 0;

        int foundflag = 0;

        while((foundflag == 0) && (fpsID < NB_FPS_MAX))
        {
            if(fpsarray[fpsID].SMfd < 0)
            {
                foundflag = 1;
            }
            else
            {
                fpsID++;
            }
        }

        if(foundflag == 1)
        {
            fpsarray[fpsID].NBparam =
                fps_connect(fpsname,
                                                  &fpsarray[fpsID],
                                                  FPSCONNECT_SIMPLE);
            if(fpsarray[fpsID].NBparam < 1)
            {
                printf("--- cannot load FPS %s\n", fpsname);
                fpsID = -1;
            }
            else
            {
                DEBUG_TRACEPOINT("loaded FPS %s to ID %ld\n", fpsname, fpsID);
            }
        }
        else
        {
            fpsID = -1;
        }
    }
    else
    {
        printf("FPS already loaded at index %ld\n", fpsID);
    }

    return fpsID;
}
