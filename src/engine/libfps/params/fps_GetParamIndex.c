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
 * functionparameter_GetParamIndex() - index of parameter @paramname: its full keyword, or what follows the
 * FPS name (".gain" or "gain"). A keyword that only contains @paramname is taken when none matches exactly
 * (before, the first one containing it won: "n" found ".conf.timestring"). -1 if not found.
 */
int functionparameter_GetParamIndex(FPS *fps, const char *paramname)
{
    const char *fname = fps->md->name;
    size_t      flen  = strlen(fname);
    const char *rel   = paramname + (paramname[0] == '.');
    long        sub   = -1;

    for (long pindex = 0; pindex < fps->md->NBparamMAX; pindex++)
    {
        const char *kw = fps->parray[pindex].keywordfull;
        if (!(fps->parray[pindex].fpflag & FPFLAG_ACTIVE))
        {
            continue;
        }
        if (strcmp(kw, paramname) == 0 ||
            (strncmp(kw, fname, flen) == 0 && kw[flen] == '.' && strcmp(kw + flen + 1, rel) == 0))
        {
            return (int) pindex;
        }
        if (sub == -1 && strstr(kw, paramname) != NULL)
        {
            sub = pindex;
        }
    }
    return (int) sub;
}
