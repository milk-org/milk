// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_RUNstart.c
 * @brief   FPS run process start
 */

#include "computeunit.h"
#include "milk_path.h"


/** @brief Start the FPS run process as a unit (fps_unit_start()), quietly
 *
 * The worker checks the parameters itself and returns its loop once ready; a start
 * still in progress after the timeout counts as started.
 */
errno_t functionparameter_RUNstart(FPS *fps)
{
    // override output directory if applicable
    long pindex = functionparameter_GetParamIndex(fps, ".conf.datadir");
    if (pindex > -1)
    {
        snprintf(fps->md->datadir, FPS_DIR_STRLENMAX, "%s", fps->parray[pindex].val.string[0]);
    }
    if (fps->md->datadir[0] != '\0')
    {
        char dir[FPS_CWD_STRLENMAX + FPS_DIR_STRLENMAX + 2];
        if (fps->md->datadir[0] == '/')
        {
            snprintf(dir, sizeof(dir), "%s", fps->md->datadir);
        }
        else
        {
            snprintf(dir, sizeof(dir), "%s/%s", fps->md->workdir, fps->md->datadir);
        }
        milkpath_mkdir_p(dir, 0755);
    }

    int rc = fps_unit_start(fps->md->name, FPS_UNIT_RUN, 0.0, FPS_UNIT_QUIET);
    return (rc == 0 || rc == FPS_UNIT_START_TIMEOUT) ? RETURN_SUCCESS : RETURN_FAILURE;
}
