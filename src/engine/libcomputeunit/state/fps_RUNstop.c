// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_RUNstop.c
 * @brief   FPS run process stop
 */

#include "computeunit.h"

/** @brief Stop the FPS run process (fps_unit_stop()), quietly; returns once it is gone */
errno_t functionparameter_RUNstop(FPS *fps)
{
    return (fps_unit_stop(fps->md->name, FPS_UNIT_RUN, FPS_UNIT_QUIET) == 0) ? RETURN_SUCCESS
                                                                             : RETURN_FAILURE;
}
