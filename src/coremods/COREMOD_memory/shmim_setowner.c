// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    shmim_setowner.c
 * @brief   set stream owner PID
 *
 * Uses FPS V2 framework.
 */

#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "shmim_setowner.h"


/* ================================================================
 *  COMPUTATION LOGIC
 * ============================================================= */

/** @brief set owner to creator */
imageID shmim_setowner_creator(const char *name)
{
    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].ownerPID = img.im->md[0].creatorPID;
    }

    return img.ID;
}

/** @brief set owner to current PID */
imageID shmim_setowner_current(const char *name)
{
    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].ownerPID = getpid();
    }

    return img.ID;
}

/**
 * @brief set owner to init process
 *
 * Makes the stream immune to orphan purging
 */
imageID shmim_setowner_init(const char *name)
{
    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].ownerPID = 1;
    }

    return img.ID;
}


/* ================================================================
 *  COMMON PARAMETER (1 stream arg)
 * ============================================================= */

static char p_sname[FUNCTION_PARAMETER_STRMAXLEN] = "stream0";

#define FPS_PARAMS_1STREAM(X) \
    X(".sname", p_sname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "stream name")


/* ================================================================
 *  CMD 1: shmimsetowncreator
 * ============================================================= */

MILK_PARAMS_DEFINE(owncreator, FPS_PARAMS_1STREAM);

static MILK_HOT errno_t owncreator_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    shmim_setowner_creator(p_sname);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_shmimsetowncreator = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "shmimsetowncreator",
             .cmdkey           = "shmimsetowncreator",
             .description      = "set owner to creator PID",
             .description_long = "Change the owner PID of a shared memory image stream. Useful for "
                                 "transferring ownership when a process is restarted." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(owncreator),
    .compute = owncreator_compute,
};


/* ================================================================
 *  CMD 2: shmimsetowncurrent
 * ============================================================= */

MILK_PARAMS_DEFINE(owncurrent, FPS_PARAMS_1STREAM);

static MILK_HOT errno_t owncurrent_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    shmim_setowner_current(p_sname);
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_shmimsetowncurrent = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "shmimsetowncurrent",
             .cmdkey           = "shmimsetowncurrent",
             .description      = "set owner to current PID",
             .description_long = "Change the owner PID of a shared memory image stream. Useful for "
                                 "transferring ownership when a process is restarted." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(owncurrent),
    .compute = owncurrent_compute,
};


/* ================================================================
 *  CMD 3: shmimsetowninit
 * ============================================================= */

MILK_PARAMS_DEFINE(owninit, FPS_PARAMS_1STREAM);

static MILK_HOT errno_t owninit_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    shmim_setowner_init(p_sname);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_shmimsetowninit = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name         = "shmimsetowninit",
             .cmdkey           = "shmimsetowninit",
             .description      = "set owner to init PID",
             .description_long = "Change the owner PID of a shared memory image stream. Useful for "
                                 "transferring ownership when a process is restarted." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(owninit),
    .compute = owninit_compute,
};
