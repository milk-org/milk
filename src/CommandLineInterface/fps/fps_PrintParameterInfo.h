// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_PrintParameterInfo.h
 * @brief   print FPS parameter status/values
 */

#ifndef FPS_PRINTPARAMETERINFO_H
#define FPS_PRINTPARAMETERINFO_H

#include "function_parameters.h"

errno_t
functionparameter_PrintParameterInfo(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                     int                        pindex);

#endif
