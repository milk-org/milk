// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mvprocCPUset.c
 * @brief CPU set and RT priority utilities
 *
 * Uses FPS V2 framework.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "milk_rt.h"

/* ================================================================
 *  COMMON PARAMS
 * ============================================================= */

static long long p_pid                                = 0;
static char      p_name[FUNCTION_PARAMETER_STRMAXLEN] = "realtime";
static long long p_rtprio                             = 80;

/* ================================================================
 *  CMD 1: rtprio (no FPS parameter)
 * ============================================================= */

MILK_RUNNABLE_RT_DEFINE(rtp);

static errno_t rtp_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    milkrt_RTPrio(p_rtprio);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_tools_rtprio = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "rtprio",
                 .cmdkey      = "rtprio",
                 .description = "set SCHED_FIFO priority",
                 .description_long =
                     "Move processes to specific CPU sets for core pinning and isolation." },
    .srcfile = __FILE__,
    .compute = rtp_compute,
    .rt      = &rtp_rt,
};

/* ================================================================
 *  CMD 2: tsetpmove (1 arg)
 * ============================================================= */

#define FPS_PARAMS_TSET(X) \
    X(".name", p_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "taskset spec list")

MILK_PARAMS_DEFINE(tset, FPS_PARAMS_TSET);

static errno_t tset_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    milkrt_Tset(p_name);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_tools_tsetpmove = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "tsetpmove",
                 .cmdkey      = "tsetpmove",
                 .description = "assign taskset to current process",
                 .description_long =
                     "Move processes to specific CPU sets for core pinning and isolation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(tset),
    .compute = tset_compute,
};

/* ================================================================
 *  CMD 3: tsetpmoveext (2 args)
 * ============================================================= */

#define FPS_PARAMS_TSETE(X)                                         \
    X(".pid", &p_pid, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "PID") \
    X(".name", p_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "taskset spec list")

MILK_PARAMS_DEFINE(tsete, FPS_PARAMS_TSETE);

static errno_t tsete_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    milkrt_TsetExt(p_pid, p_name);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_tools_tsetpmoveext = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "tsetpmoveext",
                 .cmdkey      = "tsetpmoveext",
                 .description = "assign taskset for any process",
                 .description_long =
                     "Move processes to specific CPU sets for core pinning and isolation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(tsete),
    .compute = tsete_compute,
};

/* ================================================================
 *  CMD 4: csetpmove (1 arg)
 * ============================================================= */

MILK_PARAMS_DEFINE(cset, FPS_PARAMS_TSET);

static errno_t cset_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    milkrt_CPUset(p_name);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_tools_csetpmove = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "csetpmove",
                 .cmdkey      = "csetpmove",
                 .description = "move current process to CPU set",
                 .description_long =
                     "Move processes to specific CPU sets for core pinning and isolation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cset),
    .compute = cset_compute,
};

/* ================================================================
 *  CMD 5: csetandprioext (3 args, primary)
 * ============================================================= */

#define FPS_PARAMS(X)                                                          \
    X(".pid", &p_pid, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "PID")            \
    X(".name", p_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "CPU set name") \
    X(".rtprio", &p_rtprio, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "RT priority (0=ignore)")

MILK_PARAMS_DEFINE(csetandprioext, FPS_PARAMS);

static MILK_HOT errno_t csetandprioext_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    milkrt_CPUsetExt(p_pid, p_name, p_rtprio);
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_tools_csetandprioext = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "csetandprioext",
                 .cmdkey      = "csetandprioext",
                 .description = "move PID to CPU set and assign RT priority",
                 .description_long =
                     "Move processes to specific CPU sets for core pinning and isolation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(csetandprioext),
    .compute = csetandprioext_compute,
};
