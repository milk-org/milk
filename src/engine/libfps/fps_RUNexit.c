/**
 * @file    fps_RUNexit.c
 * @brief   Exit FPS run process
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"

uint16_t function_parameter_RUNexit(FPS *fps)
{
    //fps->md->confpid = 0;

    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps_disconnect(fps);

    return 0;
}
