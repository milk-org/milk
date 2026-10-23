// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_set_counters.c
 * @brief   set image flags / counters
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"


/* ================================================================
 *  COMPUTATION LOGIC
 * ============================================================= */

errno_t COREMOD_MEMORY_image_set_status(const char *IDname, int status)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].status = status;
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Set the write counter (cnt0) of a shared stream.
 *
 * Directly overwrites the counter in SHM metadata.
 */
errno_t COREMOD_MEMORY_image_set_cnt0(const char *IDname, int cnt0)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].cnt0 = cnt0;
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Set the auxiliary counter (cnt1) of a shared stream.
 */
errno_t COREMOD_MEMORY_image_set_cnt1(const char *IDname, int cnt1)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].cnt1 = cnt1;
    }

    return RETURN_SUCCESS;
}


/* ================================================================
 *  COMMON PARAMETERS (image + int64)
 * ============================================================= */

static char      p_imname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static long long p_value                                = 2;

#define FPS_PARAMS_IMGINT(X)                                                         \
    X(".imname", p_imname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "image name") \
    X(".value", &p_value, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "value")


/* ================================================================
 *  CMD 1: imsetstatus
 * ============================================================= */

MILK_PARAMS_DEFINE(status, FPS_PARAMS_IMGINT);

static MILK_HOT errno_t status_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    COREMOD_MEMORY_image_set_status(p_imname, p_value);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_imsetstatus = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imsetstatus",
                 .cmdkey           = "imsetstatus",
                 .description      = "set image status variable",
                 .description_long = "Reset or set the frame counter and write count of a shared "
                                     "memory image stream. Useful for synchronization and "
                                     "testing." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(status),
    .compute = status_compute,
};


/* ================================================================
 *  CMD 2: imsetcnt0
 * ============================================================= */

MILK_PARAMS_DEFINE(cnt0, FPS_PARAMS_IMGINT);

static MILK_HOT errno_t cnt0_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    COREMOD_MEMORY_image_set_cnt0(p_imname, p_value);
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_imsetcnt0 = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imsetcnt0",
                 .cmdkey           = "imsetcnt0",
                 .description      = "set image cnt0 variable",
                 .description_long = "Reset or set the frame counter and write count of a shared "
                                     "memory image stream. Useful for synchronization and "
                                     "testing." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cnt0),
    .compute = cnt0_compute,
};


/* ================================================================
 *  CMD 3: imsetcnt1
 * ============================================================= */

MILK_PARAMS_DEFINE(cnt1, FPS_PARAMS_IMGINT);

static MILK_HOT errno_t cnt1_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    COREMOD_MEMORY_image_set_cnt1(p_imname, p_value);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_imsetcnt1 = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imsetcnt1",
                 .cmdkey           = "imsetcnt1",
                 .description      = "set image cnt1 variable",
                 .description_long = "Reset or set the frame counter and write count of a shared "
                                     "memory image stream. Useful for synchronization and "
                                     "testing." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cnt1),
    .compute = cnt1_compute,
};
