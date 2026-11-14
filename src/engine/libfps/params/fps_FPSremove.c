// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_FPSremove.c
 * @brief   remove FPS
 */

#include "fps.h"
#include "fps_unit_state.h"

/** @brief remove the FPS file and its unit directory
 *
 * Requires CONF and RUN to be off (milk-fps-rm stops them first)
 */
errno_t functionparameter_FPSremove(FPS *fps)
{
    char shmdname[STRINGMAXLEN_DIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    char fpsfname[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(fpsfname, "%s/%s.fps.shm", shmdname, fps->md->name);

    remove(fpsfname);
    fps_unit_dir_remove(fps->md->name);

    return RETURN_SUCCESS;
}
