/**
 * @file fps_WriteParameterToDisk.h
 * @brief Fps writeparametertodisk module
 */

/**
 * @file fps_WriteParameterToDisk.h
 *
 */

#ifndef FPS_WRITEPARAMETERTODISK_H
#define FPS_WRITEPARAMETERTODISK_H

#include "fps.h"

int functionparameter_WriteParameterToDisk(FUNCTION_PARAMETER_STRUCT *fpsentry,
        int                        pindex,
        char                      *tagname,
        char *commentstr);

#endif
