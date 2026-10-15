// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_monitorlimits.c
 * @brief   Monitor stream values for safety limits
 *
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "create_image.h"
#include "image_ID.h"
#include "processinfo.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    inimname[FUNCTION_PARAMETER_STRMAXLEN] = "stream";
static int64_t dtus                                   = 100000;
static int32_t minON                                  = 0;
static float   minVal                                 = 0.0;
static int32_t maxON                                  = 0;
static float   maxVal                                 = 1000.0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                     \
    X(".in_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream")   \
    X(".dtus", &dtus, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "loop period [us]")          \
    X(".minON", &minON, FPTYPE_ONOFF, 1, FPFLAG_DEFAULT_INPUT, "minimum limit toggle")    \
    X(".minVal", &minVal, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "minimum limit value") \
    X(".maxON", &maxON, FPTYPE_ONOFF, 1, FPFLAG_DEFAULT_INPUT, "maximum limit toggle")    \
    X(".maxVal", &maxVal, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "maximum limit value")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Actual monitoring logic
 */
static errno_t monitor_logic(IMGID *imgptr)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(imgptr, ERRMODE_ABORT, dcimg, dcnimg);

    uint32_t xsize  = imgptr->md->size[0];
    uint32_t ysize  = imgptr->md->size[1];
    uint64_t xysize = (uint64_t) xsize * ysize;

    float minv = imgptr->im->array.F[0];
    float maxv = imgptr->im->array.F[0];

    for (uint64_t ii = 1; ii < xysize; ii++)
    {
        if (imgptr->im->array.F[ii] < minv)
        {
            minv = imgptr->im->array.F[ii];
        }
        if (imgptr->im->array.F[ii] > maxv)
        {
            maxv = imgptr->im->array.F[ii];
        }
    }

    int  limit_exceeded = 0;
    char msg[STRINGMAXLEN_PROCESSINFO_STATUSMSG];
    msg[0] = '\0';

    if (minON && (minv < minVal))
    {
        limit_exceeded = 1;
        snprintf(msg, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "MIN LIMIT EXCEEDED: %f < %f", minv,
                 minVal);
    }

    if (maxON && (maxv > maxVal))
    {
        limit_exceeded = 1;
        if (msg[0] != '\0')
        {
            strncat(msg, " | ", STRINGMAXLEN_PROCESSINFO_STATUSMSG - strlen(msg) - 1);
        }
        char tmpmsg[100];
        snprintf(tmpmsg, 100, "MAX LIMIT EXCEEDED: %f > %f", maxv, maxVal);
        strncat(msg, tmpmsg, STRINGMAXLEN_PROCESSINFO_STATUSMSG - strlen(msg) - 1);
    }

    if (limit_exceeded)
    {
        PRINT_WARNING("%s", msg);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  RUNNABLE
 * ============================================================= */

MILK_PARAMS_DEFINE(streammlim, FPS_PARAMS);

static MILK_HOT errno_t streammlim_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg, dcnimg);

    MILK_PROCINFO_INIT(self)

    if (dcprocinfo == 1)
    {
        processinfo_waitoninputstream_init(processinfo, inimg.im, PROCESSINFO_TRIGGERMODE_DELAY,
                                           -1);
        processinfo->triggerdelay.tv_sec  = 0;
        processinfo->triggerdelay.tv_nsec = dtus * 1000;
        while (processinfo->triggerdelay.tv_nsec >= 1000000000)
        {
            processinfo->triggerdelay.tv_nsec -= 1000000000;
            processinfo->triggerdelay.tv_sec += 1;
        }
    }

    MILK_PROCINFO_LOOPSTART(self)
    {
        monitor_logic(&inimg);
    }
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_streammlim = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "streammlim",
                 .cmdkey      = "streammlim",
                 .description = "monitor stream values for safety",
                 .description_long =
                     "Monitor pixel values in a stream and flag frames where values exceed "
                     "configurable min/max thresholds. Reports out-of-range statistics." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(streammlim),
    .compute = streammlim_compute,
};


/* ================================================================
 * BACKWARD COMPAT WRAPPER
 * ============================================================= */

errno_t stream_monitorlimits(const char *instreamname)
{
    strncpy(inimname, instreamname, FUNCTION_PARAMETER_STRMAXLEN - 1);
    dtus   = 100000;
    minON  = 0;
    minVal = 0.0;
    maxON  = 0;
    maxVal = 1000.0;

    milk_runnable_prepare(&runnable_COREMOD_memory_streammlim);
    return streammlim_compute(&runnable_COREMOD_memory_streammlim);
}
