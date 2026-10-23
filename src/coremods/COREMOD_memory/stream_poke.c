// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_poke.c
 * @brief   poke image stream
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN] = "stream";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_sname", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/* (no separate fpsexec — logic is trivial) */


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(shmimpoke, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t shmimpoke_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID img = imgid_make_from_name(inimname);
    resolveIMGID(&img, ERRMODE_ABORT, dcimg, dcnimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    processinfo_update_output_stream(processinfo, img.im, NULL);
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_shmimpoke = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "shmimpoke",
                 .cmdkey      = "shmimpoke",
                 .description = "update stream without changing content",
                 .description_long =
                     "Write a test pattern or constant value into a shared memory stream at a "
                     "configurable rate. Useful for testing downstream consumers and verifying "
                     "semaphore triggering." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(shmimpoke),
    .compute = shmimpoke_compute,
};
EXPORT_FPSEXEC(milk, "mem-streampoke", runnable_COREMOD_memory_shmimpoke);
