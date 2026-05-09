/**
 * @file    fps_disconnect.h
 * @brief   Disconnect from FPS
 */

#ifndef FPS_DISCONNECT_H
#define FPS_DISCONNECT_H

#include "fps.h"

int function_parameter_struct_disconnect(
    FUNCTION_PARAMETER_STRUCT *funcparamstruct);

/**
 * @brief Short alias for function_parameter_struct_disconnect
 */
static inline int fps_disconnect(FPS *fps)
{
    return function_parameter_struct_disconnect(fps);
}

#endif

