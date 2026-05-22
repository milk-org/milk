/**
 * @file    fps_FPCONFsetup.h
 * @brief   FPS config setup
 */

#ifndef FPS_FPCONFSETUP_H
#define FPS_FPCONFSETUP_H

#include "fps.h"

FPS function_parameter_FPCONFsetup_sized(const char *fpsname, uint32_t CMDmode, long NBparamMAX);

FPS function_parameter_FPCONFsetup(const char *fpsname, uint32_t CMDmode);

#endif
