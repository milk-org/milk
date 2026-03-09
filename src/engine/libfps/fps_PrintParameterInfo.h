/**
 * @file    fps_PrintParameterInfo.h
 * @brief   print FPS parameter status/values
 */

#ifndef FPS_PRINTPARAMETERINFO_H
#define FPS_PRINTPARAMETERINFO_H

#include "fps.h"

errno_t
functionparameter_PrintParameterInfo(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                     int                        pindex);

#endif
