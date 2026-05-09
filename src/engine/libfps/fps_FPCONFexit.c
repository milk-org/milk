/**
 * @file    fps_FPCONFexit.c
 * @brief   Exit FPS conf process
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"

#include "fps_disconnect.h"

uint16_t function_parameter_FPCONFexit(FPS *fps)
{
    //fps->md->confpid = 0;

    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    fps_disconnect(fps);

    return 0;
}
