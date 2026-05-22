/**
 * @file    fps_printparameter_valuestring.h
 * @brief   print parameter value string
 */

#ifndef FPS_PRINTPARAMETER_VALUESTRING_H
#define FPS_PRINTPARAMETER_VALUESTRING_H

#include "fps.h"

errno_t functionparameter_PrintParameter_ValueString(FPS_PARAM *fpsentry,
                                                     char      *outstring,
                                                     int        stringmaxlen);


errno_t functionparameter_GetParamValueString(FPS_PARAM *fpsentry,
                                              char      *outstring,
                                              int        stringmaxlen);

#endif
