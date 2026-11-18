// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_GetParamIndex.h
 * @brief   Get index of parameter
 */

#include "fps_types.h"
#include "fps_internal.h"

/**
 * @brief Look up a parameter index by its dot-separated
 * keyword name.
 *
 * Exact match on the keyword chain. @paramname may be given
 * as the full keyword ("fpsname.a.b"), relative to the FPS
 * name with a leading dot (".a.b"), or without it ("a.b").
 * Returns -1 if not found.
 */
int functionparameter_GetParamIndex(FPS *fps, const char *paramname)
{
    // Relative forms: skip a leading dot
    const char *relname = (paramname[0] == '.') ? paramname + 1 : paramname;

    long NBparamMAX = fps->md->NBparamMAX;

    for (long pindex = 0; pindex < NBparamMAX; pindex++)
    {
        if (!(fps->parray[pindex].fpflag & FPFLAG_ACTIVE))
        {
            continue;
        }

        const char *kwfull = fps->parray[pindex].keywordfull;
        if (strcmp(kwfull, paramname) == 0)
        {
            return pindex;
        }

        // keywordfull = keyword[0] "." rest
        size_t len0 = strlen(fps->parray[pindex].keyword[0]);
        if (strncmp(kwfull, fps->parray[pindex].keyword[0], len0) == 0 && kwfull[len0] == '.' &&
            strcmp(kwfull + len0 + 1, relname) == 0)
        {
            return pindex;
        }
    }

    return -1;
}
