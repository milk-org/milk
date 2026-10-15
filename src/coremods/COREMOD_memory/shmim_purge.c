// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    shmim_purge.c
 * @brief   purge shared memory stream
 *
 * Uses FPS V2 framework.
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "streamCTRL_find_streams.h"

#include "image_ID.h"
#include "read_shmim.h"
#include "shmim_purge.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char stringfilter[FUNCTION_PARAMETER_STRMAXLEN] = "";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".strfilter", stringfilter, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "string filter")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/** @brief purge orphan shared memory streams */
errno_t shmim_purge(const char *strfilter)
{
    int         NBstreamMAX = 10000;
    STREAMINFO *streaminfo;

    DEBUG_TRACEPOINT("Searching for streams");
    streaminfo   = (STREAMINFO *) calloc(NBstreamMAX, sizeof(STREAMINFO));
    int NBstream = find_streams(streaminfo, 1, strfilter);
    printf("%d stream(s) found\n", NBstream);

    DEBUG_TRACEPOINT("scanning %d streams for purging", NBstream);
    for (int sindex = 0; sindex < NBstream; sindex++)
    {
        printf(" STREAM %3d   %s\n", sindex, streaminfo[sindex].sname);
        IMGID img = imgid_make_from_name(streaminfo[sindex].sname);
        resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
        if (img.ID == -1)
        {
            imageID fid = read_sharedmem_image(streaminfo[sindex].sname, dcimg, dcnimg);
            if (fid == -1)
            {
                printf("Failed to load stream %s\n", streaminfo[sindex].sname);
                continue;
            }

            resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
            if (img.ID == -1 || img.im == NULL)
            {
                printf("Failed to resolve stream %s after load\n", streaminfo[sindex].sname);
                continue;
            }
        }
        DEBUG_TRACEPOINT("stream %s loaded ID %ld", streaminfo[sindex].sname, (long) img.ID);

        pid_t opid;
        opid = img.im->md[0].ownerPID;
        DEBUG_TRACEPOINT("owner PID : %ld", (long) opid);
        printf("owner PID : %ld\n", (long) opid);

        if (opid != 0)
        {
            if (getpgid(opid) >= 0)
            {
                printf("Keeping stream %s\n", streaminfo[sindex].sname);
            }
            else
            {
                printf("Purging stream %s\n", streaminfo[sindex].sname);
                ImageStreamIO_destroyIm(img.im);
            }
        }
        else
        {
            printf("Purging stream %s\n", streaminfo[sindex].sname);
            ImageStreamIO_destroyIm(img.im);
        }
    }

    free(streaminfo);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS AND FARG
 * ============================================================= */

MILK_PARAMS_DEFINE(shmim_purge, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t shmim_purge_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    shmim_purge(stringfilter);

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_shmimpurge = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "shmimpurge",
                 .cmdkey      = "shmimpurge",
                 .description = "purge orphan streams",
                 .description_long =
                     "Scan /dev/shm for stale (orphaned) image streams and remove them. A stream "
                     "is considered stale if no process has it open or if its creator PID no "
                     "longer exists." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(shmim_purge),
    .compute = shmim_purge_compute,
};
