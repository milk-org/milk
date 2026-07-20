// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_loadstream.h
 * @brief   Load image stream
 */

#ifndef FPS_LOADSTREAM_H
#define FPS_LOADSTREAM_H

#include "function_parameters.h"

imageID functionparameter_LoadStream(FUNCTION_PARAMETER_STRUCT *fps,
                                     int                        pindex,
                                     int                        fpsconnectmode);

#endif
