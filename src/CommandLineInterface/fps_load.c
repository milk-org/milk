/**
 * @file    fps_load.c
 * @brief   Load FPS
 */

#include "COREMOD_memory/COREMOD_memory.h"
#include "CommandLineInterface/CLIcore.h"

#include "fps_connect.h"

long function_parameter_structure_load(char *fpsname)
{
    long fpsID;

    DEBUG_TRACEPOINT("loading FPS %s", fpsname);

    fpsID = fps_ID(fpsname);

    if (fpsID == -1)
    {
        // not found, searching

        // next fpsID available
        fpsID = 0;

        int foundflag = 0;

        while ((foundflag == 0) && (fpsID < data.NB_MAX_FPS))
        {
            if (data.fpsarray[fpsID].SMfd < 0)
            {
                foundflag = 1;
            }
            else
            {
                fpsID++;
            }
        }

        if (foundflag == 1)
        {
            data.fpsarray[fpsID].NBparam =
                function_parameter_struct_connect(fpsname, &data.fpsarray[fpsID], FPSCONNECT_SIMPLE);
            if (data.fpsarray[fpsID].NBparam < 1)
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
