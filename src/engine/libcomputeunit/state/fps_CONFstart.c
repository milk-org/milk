// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_CONFstart.c
 * @brief   FPS conf process start
 */

#include "computeunit.h"

/** @brief Start the FPS conf process as a unit (fps_unit_start()), quietly */
errno_t functionparameter_CONFstart(FPS *fps)
{
    int rc = fps_unit_start(fps->md->name, FPS_UNIT_CONF, 0.0, FPS_UNIT_QUIET);
    return (rc == 0 || rc == FPS_UNIT_START_TIMEOUT) ? RETURN_SUCCESS : RETURN_FAILURE;
}
