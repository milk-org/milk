// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_printparameter_valuestring.h
 * @brief   print parameter value string
 */

#ifndef FPS_PRINTPARAMETER_VALUESTRING_H
#define FPS_PRINTPARAMETER_VALUESTRING_H

#include "function_parameters.h"

errno_t functionparameter_PrintParameter_ValueString(
    FUNCTION_PARAMETER *fpsentry, char *outstring, int stringmaxlen);

#endif
