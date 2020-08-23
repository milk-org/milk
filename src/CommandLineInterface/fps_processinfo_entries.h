/**
 * @file    fps_add_RTsetting_entries.h
 * @brief   Add parameters to FPS for real-time process settings
 */

#ifndef FPS_ADD_RTSETTING_ENTRIES_H
#define FPS_ADD_RTSETTING_ENTRIES_H

#include "function_parameters.h"
#include "processtools.h"

errno_t fps_add_processinfo_entries(FUNCTION_PARAMETER_STRUCT *fps);

errno_t fps_to_processinfo(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *procinfo);

#endif