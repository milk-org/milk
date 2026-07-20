// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_GetFileName.h
 * @brief   get FPS filename for entry
 */

#ifndef FPS_GETFILENAME_H
#define FPS_GETFILENAME_H

#include "function_parameters.h"

int functionparameter_GetFileName(FUNCTION_PARAMETER_STRUCT *fps,
                                  FUNCTION_PARAMETER        *fparam,
                                  char                      *outfname,
                                  char                      *tagname);

#endif
