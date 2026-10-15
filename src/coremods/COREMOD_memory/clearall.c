// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    clearall.c
 * @brief   remove all images, variables, and FPS
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "delete_image.h"
#include "delete_variable.h"
#include "image_ID.h"


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t clearall()
{
    imageID ID;

    // clear images
    for (ID = 0; ID < dcnimg; ID++)
    {
        if (dcimg[ID].used == 1)
        {
            delete_image_ID(dcimg[ID].name, DELETE_IMAGE_ERRMODE_WARNING);
        }
    }

    // clear variables
    for (ID = 0; ID < dcnvar; ID++)
    {
        if (dcvar[ID].used == 1)
        {
            delete_variable_ID(dcvar[ID].name);
        }
    }

    // clear FPS
    for (int fpsindex = 0; fpsindex < dcnfps; fpsindex++)
    {
        DEBUG_TRACEPOINT("clear FPS %d", fpsindex);
        dcfpsarr[fpsindex].SMfd = -1;
        if (dcfpsarr[fpsindex].parray != NULL)
        {
            dcfpsarr[fpsindex].parray = NULL;
        }
        if (dcfpsarr[fpsindex].md != NULL)
        {
            dcfpsarr[fpsindex].md = NULL;
        }
    }

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  RUNNABLE (no FPS parameter)
 * ============================================================= */

MILK_RUNNABLE_RT_DEFINE(clearall);

static MILK_HOT errno_t clearall_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(clearall());

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_clearall = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "rmall",
                 .cmdkey      = "rmall",
                 .description = "remove all images",
                 .description_long =
                     "Remove all images and variables from the current process memory space. Frees "
                     "all locally allocated image buffers but does not destroy shared memory "
                     "segments." },
    .srcfile = __FILE__,
    .compute = clearall_compute,
    .rt      = &clearall_rt,
};
