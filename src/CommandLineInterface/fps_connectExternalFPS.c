/**
 * @file    fps_connectExternalFPS.c
 * @brief   connect to external FPS
 */

#include "CommandLineInterface/CLIcore.h"

#include "fps_connect.h"

int functionparameter_ConnectExternalFPS(FUNCTION_PARAMETER_STRUCT *FPS,
                                         int                        pindex,
                                         FUNCTION_PARAMETER_STRUCT *FPSext)
{
    FPS->parray[pindex].info.fps.FPSNBparamMAX =
        function_parameter_struct_connect(FPS->parray[pindex].val.string[0],
                                          FPSext,
                                          FPSCONNECT_SIMPLE);

    FPS->parray[pindex].info.fps.FPSNBparamActive = 0;
    FPS->parray[pindex].info.fps.FPSNBparamUsed   = 0;
    int pindexext;
    for (pindexext = 0; pindexext < FPS->parray[pindex].info.fps.FPSNBparamMAX;
         pindexext++)
        {
            if (FPSext->parray[pindexext].fpflag & FPFLAG_ACTIVE)
                {
                    FPS->parray[pindex].info.fps.FPSNBparamActive++;
                }
            if (FPSext->parray[pindexext].fpflag & FPFLAG_USED)
                {
                    FPS->parray[pindex].info.fps.FPSNBparamUsed++;
                }
        }

    return 0;
}
