// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

int functionparameter_WriteParameterToDisk(FPS  *fpsentry,
                                           int   pindex,
                                           char *tagname,
                                           char *commentstr);

#endif
