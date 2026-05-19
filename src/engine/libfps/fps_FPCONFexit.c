/**
 * @file    fps_FPCONFexit.c
 * @brief   Exit FPS conf process
 */

#include "fps.h"


/**
 * @brief Exit the FPS configuration loop.
 *
 * Sets conf loop status to exit and flushes any
 * pending parameter writes.
 */
uint16_t function_parameter_FPCONFexit(FPS *fps)
{
    //fps->md->confpid = 0;

    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    fps_disconnect(fps);

    return 0;
}
