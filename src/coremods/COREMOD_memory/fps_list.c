// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_list.c
 * @brief   list function parameter structures
 *
 * Uses FPS V2 framework.
 */

#include <dirent.h>
#include <stdio.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t fps_list()
{
    long fpscnt = 0;

    int NBchar_fpsID   = 5;
    int NBchar_fpsname = 12;
    int NBchar_NBparam = 4;

    for (long fpsID = 0; fpsID < dcnfps; fpsID++)
    {
        if (dcfpsarr[fpsID].SMfd > -1)
        {
            if (fpscnt == 0)
            {
                printf("FPSs currently connected :\n");
            }
            printf("%*ld  %*s  %*ld/%*ld entries\n", NBchar_fpsID, fpsID, NBchar_fpsname,
                   dcfpsarr[fpsID].md[0].name, NBchar_NBparam, dcfpsarr[fpsID].NBparamActive,
                   NBchar_NBparam, dcfpsarr[fpsID].NBparam);

            fpscnt++;
        }
    }
    if (fpscnt == 0)
    {
        printf("No FPS currently connected\n");
    }

    printf("FPSs in system shared memory (%s):\n", dcshmdir);

    /* Scan SHM directory for *.fps.shm files */
    {
        struct dirent *de;
        DIR           *dr = opendir(dcshmdir);
        if (dr == NULL)
        {
            printf("Could not open current directory");
            return RETURN_FAILURE;
        }

        fpscnt = 0;
        while ((de = readdir(dr)) != NULL)
        {
            if (strstr(de->d_name, ".fps.shm") != NULL)
            {
                char fpsname[100];
                int  slen1 = 100 - strlen(".fps.shm");

                strncpy(fpsname, de->d_name, slen1);
                fpsname[slen1] = '\0';
                printf("%*ld  %*s\n", NBchar_fpsID, fpscnt, NBchar_fpsname, fpsname);
                fpscnt++;
            }
        }
        closedir(dr);
    } // Scan SHM directory

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  RUNNABLE (no FPS parameter)
 * ============================================================= */

MILK_RUNNABLE_RT_DEFINE(fps_list);

static MILK_HOT errno_t fps_list_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(fps_list());

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_fps_list = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "fpslist",
                 .cmdkey           = "fpslist",
                 .description      = "list function parameter structures",
                 .description_long = "List all FPS (Function Parameter Structure) instances "
                                     "currently active in shared memory. Shows FPS name, status, "
                                     "and associated process." },
    .srcfile = __FILE__,
    .compute = fps_list_compute,
    .rt      = &fps_list_rt,
};
