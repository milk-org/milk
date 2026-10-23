// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    read_shmimall.c
 * @brief   read all shared memory streams
 *
 * Uses FPS V2 framework.
 */

#include <fcntl.h> // open
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h> // mmap
#include <sys/stat.h>
#include <unistd.h> // close

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"
#include "read_shmimall.h"

#include "streamCTRL_find_streams.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char strfilter[FUNCTION_PARAMETER_STRMAXLEN] = "aol_";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".strfilter", strfilter, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "string filter")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t read_sharedmem_image_all(const char *strfilter)
{
    int         NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    streaminfo = (STREAMINFO *) calloc(NBstreamMAX, sizeof(STREAMINFO));

    int NBstream = find_streams(streaminfo, 1, strfilter);

    for (int sindex = 0; sindex < NBstream; sindex++)
    {
        if (!imgid_exists(streaminfo[sindex].sname))
        {
            read_sharedmem_image(streaminfo[sindex].sname, dcimg, dcnimg);
        }
    }

    free(streaminfo);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS AND FARG
 * ============================================================= */

MILK_PARAMS_DEFINE(readshmimall, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t readshmimall_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    FUNC_CHECK_RETURN(read_sharedmem_image_all(strfilter));

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_readshmimall = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "readshmimall",
                 .cmdkey      = "readshmimall",
                 .description = "read all shared memory images",
                 .description_long =
                     "Connect to all shared memory image streams currently present in /dev/shm. "
                     "Maps every stream into the process address space." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(readshmimall),
    .compute = readshmimall_compute,
};
