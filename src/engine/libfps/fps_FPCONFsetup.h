/**
 * @file    fps_FPCONFsetup.h
 * @brief   FPS config setup
 */

#ifndef FPS_FPCONFSETUP_H
#define FPS_FPCONFSETUP_H

#include "fps.h"

FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup_sized(
        const char *fpsname,
        uint32_t    CMDmode,
        long        NBparamMAX
        );

FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(
        const char *fpsname,
        uint32_t    CMDmode
        );

#endif
