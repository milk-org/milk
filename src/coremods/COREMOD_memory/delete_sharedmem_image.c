// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    delete_sharedmem_image.c
 * @brief   delete shared memory image and files
 *
 * Uses FPS V2 framework.
 */

#include <malloc.h>
#include <sys/mman.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char imname[FUNCTION_PARAMETER_STRMAXLEN] = "im";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) X(".imname", imname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "image name")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t destroy_shared_image_ID(const char *__restrict imname)
{
    IMGID img = imgid_make_from_name(imname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);

    if ((img.ID != -1) && (img.im->md[0].shared == 1))
    {
        ImageStreamIO_destroyIm(img.im);
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm WARNING: shared image"
                " %s does not exist [ %s  "
                "%s  %d ] %c[%d;m\n",
                (char) 27, 1, 31, imname, __FILE__, __func__, __LINE__, (char) 27, 0);
    }

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(delete_sharedmem_image, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t delete_sharedmem_image_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    FUNC_CHECK_RETURN(destroy_shared_image_ID(imname));

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_delete_sharedmem_image = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "rmshmim",
             .cmdkey           = "rmshmim",
             .description      = "remove shared image and files",
             .description_long = "Destroy a shared memory image stream and its associated files in "
                                 "/dev/shm. Removes the stream from all connected processes." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(delete_sharedmem_image),
    .compute = delete_sharedmem_image_compute,
};
