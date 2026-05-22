/**
 * @file    fps_connectExternalFPS.c
 * @brief   connect to external FPS
 */

#include "fps.h"


/**
 * @brief Connect an FPS to an external (foreign) FPS.
 *
 * Opens the target FPS shared memory and maps its
 * parameter array, allowing cross-process parameter
 * access.
 */
int functionparameter_ConnectExternalFPS(FPS *fps, int pindex, FPS *FPSext)
{
    fps->parray[pindex].info.fps.FPSNBparamMAX =
        fps_connect(fps->parray[pindex].val.string[0], FPSext, FPSCONNECT_SIMPLE);

    fps->parray[pindex].info.fps.FPSNBparamActive = 0;
    fps->parray[pindex].info.fps.FPSNBparamUsed   = 0;

    for (int pindexext = 0; pindexext < fps->parray[pindex].info.fps.FPSNBparamMAX; pindexext++)
    {
        if (FPSext->parray[pindexext].fpflag & FPFLAG_ACTIVE)
        {
            fps->parray[pindex].info.fps.FPSNBparamActive++;
        }
        if (FPSext->parray[pindexext].fpflag & FPFLAG_USED)
        {
            fps->parray[pindex].info.fps.FPSNBparamUsed++;
        }
    }

    return 0;
}
