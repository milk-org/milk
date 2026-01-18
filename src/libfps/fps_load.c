/**
 * @file    fps_load.c
 * @brief   Load FPS
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"

#include "fps_connect.h"

// Forward declaration or include for fps_ID
long fps_ID(const char *fpsname);


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
                function_parameter_struct_connect(fpsname,
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
