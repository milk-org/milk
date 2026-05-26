/**
 * @file    fps_RUNexit.c
 * @brief   Exit FPS run process
 */

#include "fps.h"

/**
 * @brief Exit the FPS run loop.
 *
 * Sets the run loop status to exit and updates
 * processinfo state.
 */
uint16_t function_parameter_RUNexit(FPS *fps)
{
    //fps->md->confpid = 0;

    fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    fps_disconnect(fps);

    return 0;
}
