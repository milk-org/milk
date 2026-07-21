// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_CONFstop.c
 * @brief   FPS conf process stop
 */

#include "fps.h"

/** @brief FPS stop CONF process
 *
 */
errno_t functionparameter_CONFstop(FPS *fps)
{
    // send conf stop signal
    fps->md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;

    return RETURN_SUCCESS;
}
