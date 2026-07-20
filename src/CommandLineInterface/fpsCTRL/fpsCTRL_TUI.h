// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_CTRLscreen.h
 * @brief   FPS control TUI
 */

#ifndef FPS_CTRLSCREEN_H
#define FPS_CTRLSCREEN_H

#include "../processinfo.h"

errno_t functionparameter_CTRLscreen(uint32_t mode,
                                     char    *fpsnamemask,
                                     char    *fpsCTRLfifoname);

#endif
