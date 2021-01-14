/**
 * @file    fps_loadstream.c
 * @brief   Load image stream
 */


#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"


imageID functionparameter_LoadStream(
    FUNCTION_PARAMETER_STRUCT *fps,
    int                        pindex,
    int                        fpsconnectmode
)
{
    imageID ID = -1;
    uint32_t     imLOC;


    printf("====================== Loading stream \"%s\" = %s\n",
           fps->parray[pindex].keywordfull, fps->parray[pindex].val.string[0]);
    ID = COREMOD_IOFITS_LoadMemStream(fps->parray[pindex].val.string[0],
                                      &(fps->parray[pindex].fpflag), &imLOC);


    if(fpsconnectmode == FPSCONNECT_CONF)
    {
        if(fps->parray[pindex].fpflag & FPFLAG_STREAM_CONF_REQUIRED)
        {
            printf("    FPFLAG_STREAM_CONF_REQUIRED\n");
            if(ID == -1)
            {
                printf("FAILURE: Required stream %s could not be loaded\n",
                       fps->parray[pindex].val.string[0]);
                exit(EXIT_FAILURE);
            }
        }
    }

    if(fpsconnectmode == FPSCONNECT_RUN)
    {
        if(fps->parray[pindex].fpflag & FPFLAG_STREAM_RUN_REQUIRED)
        {
            printf("    FPFLAG_STREAM_RUN_REQUIRED\n");
            if(ID == -1)
            {
                printf("FAILURE: Required stream %s could not be loaded\n",
                       fps->parray[pindex].val.string[0]);
                exit(EXIT_FAILURE);
            }
        }
    }


    // TODO: Add testing for fps



    return ID;
}
