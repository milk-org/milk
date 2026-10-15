// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    read_shmim.c
 * @brief   read shared memory stream
 *
 * Uses FPS V2 framework.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "image_keyword_list.h"
#include "list_image.h"
#include "imageID.h"
#include "read_shmim.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char insname[FUNCTION_PARAMETER_STRMAXLEN] = "stream";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_sname", insname, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, "input stream")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

imageID read_sharedmem_image(const char *restrict sname, IMAGE *imagearray, long NB_images)
{
    IMGID img = imgid_make_from_name(sname);
    resolveIMGID(&img, ERRMODE_NULL, imagearray, NB_images);
    imgid_connect(&img, IMGID_CONNECT_NOCHECK);
    if (img.ID == -1)
    {
        return -1;
    }

    return RegisterIMGID(&img, imagearray, NB_images);
}


/* ================================================================
 * 5.  BINDINGS AND FARG
 * ============================================================= */

MILK_PARAMS_DEFINE(read_sharedmem_image, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t read_sharedmem_image_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    read_sharedmem_image(insname, dcimg, dcnimg);

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_readshmim = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "readshmim",
                 .cmdkey      = "readshmim",
                 .description = "read shared memory image",
                 .description_long =
                     "Connect to an existing shared memory image stream by name. Maps the stream "
                     "into the current process address space for reading. Returns failure if the "
                     "stream does not exist." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(read_sharedmem_image),
    .compute = read_sharedmem_image_compute,
};
