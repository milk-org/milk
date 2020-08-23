
/**
 * @file    fps_checkparameters.h
 * @brief   check FPS entries
 */

#ifndef FPS_CHECKPARAMETERS_H
#define FPS_CHECKPARAMETERS_H


int functionparameter_CheckParameter(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int pindex
);




int functionparameter_CheckParametersAll(
    FUNCTION_PARAMETER_STRUCT *fpsentry
);

#endif