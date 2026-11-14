// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_CONFstop.c
 * @brief   FPS conf process stop
 */

#include "computeunit.h"

/** @brief Stop the FPS conf process (fps_unit_stop()), quietly; returns once it is gone */
errno_t functionparameter_CONFstop(FPS *fps)
{
    return (fps_unit_stop(fps->md->name, FPS_UNIT_CONF, FPS_UNIT_QUIET) == 0) ? RETURN_SUCCESS
                                                                              : RETURN_FAILURE;
}
