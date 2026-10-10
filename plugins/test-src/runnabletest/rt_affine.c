// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    rt_affine.c
 * @brief   Four MILK_RUNNABLE units in one file: out = gain * in + offset
 *
 *  - scale    : CLI command and fpsexec
 *  - offset   : CLI command and fpsexec
 *  - negate   : CLI command only (listed in the module registry, no EXPORT_FPSEXEC)
 *  - passthru : fpsexec only (not in the module registry)
 *
 * Each unit has its own parameter table, statics and compute function; nothing is named implicitly.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "runnabletest/rt_kernels.h"
#include "runnabletest.h"

static const float rt_zero = 0.0f;
static const float rt_one  = 1.0f;
static const float rt_neg  = -1.0f;


/**
 * @brief Processinfo loop shared by all units of this file.
 *
 * Gain and offset are passed by address so that parameters updated through the FPS are seen.
 */
static errno_t rt_affine_run(const MILK_RUNNABLE *self,
                             const char          *inname,
                             const char          *outname,
                             const float         *gain,
                             const float         *offset)
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inname);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID outimg = imgid_make_from_name(outname);
    imgid_copy(&inimg, &outimg);
    imcreateIMGID(&outimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        rt_stream_affine(&inimg, &outimg, *gain, *offset);
        processinfo_update_output_stream(processinfo, outimg.im, inimg.im);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * scale: out = gain * in
 * ============================================================= */

static char  scale_in[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char  scale_out[FUNCTION_PARAMETER_STRMAXLEN] = "";
static float scale_gain                              = 1.0f;

#define RT_SCALE_PARAMS(X)                                                                \
    X(".in_name", scale_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM, "input") \
    X(".out_name", scale_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")           \
    X(".gain", &scale_gain, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "multiplicative gain")

MILK_PARAMS_DEFINE(scale, RT_SCALE_PARAMS);

static errno_t scale_compute(const MILK_RUNNABLE *self)
{
    return rt_affine_run(self, scale_in, scale_out, &scale_gain, &rt_zero);
}

const MILK_RUNNABLE runnable_rt_scale = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "rtscale",
                 .cmdkey      = "scale",
                 .description = "runnabletest: out = gain * in" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(scale),
    .compute = scale_compute,
};
EXPORT_FPSEXEC(milk, "runnabletest-scale", runnable_rt_scale);


/* ================================================================
 * offset: out = in + offset
 * ============================================================= */

static char  offset_in[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char  offset_out[FUNCTION_PARAMETER_STRMAXLEN] = "";
static float offset_val                               = 0.0f;

#define RT_OFFSET_PARAMS(X)                                                                \
    X(".in_name", offset_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM, "input") \
    X(".out_name", offset_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")           \
    X(".offset", &offset_val, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "additive offset")

MILK_PARAMS_DEFINE(offset, RT_OFFSET_PARAMS);

static errno_t offset_compute(const MILK_RUNNABLE *self)
{
    return rt_affine_run(self, offset_in, offset_out, &rt_one, &offset_val);
}

const MILK_RUNNABLE runnable_rt_offset = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "rtoffset",
                 .cmdkey      = "offset",
                 .description = "runnabletest: out = in + offset" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(offset),
    .compute = offset_compute,
};
EXPORT_FPSEXEC(milk, "runnabletest-offset", runnable_rt_offset);


/* ================================================================
 * negate: out = -in (CLI only)
 * ============================================================= */

static char neg_in[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char neg_out[FUNCTION_PARAMETER_STRMAXLEN] = "";

#define RT_NEGATE_PARAMS(X)                                                             \
    X(".in_name", neg_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM, "input") \
    X(".out_name", neg_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

MILK_PARAMS_DEFINE(negate, RT_NEGATE_PARAMS);

static errno_t negate_compute(const MILK_RUNNABLE *self)
{
    return rt_affine_run(self, neg_in, neg_out, &rt_neg, &rt_zero);
}

const MILK_RUNNABLE runnable_rt_negate = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "rtnegate",
                 .cmdkey      = "negate",
                 .description = "runnabletest: out = -in (CLI only)" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(negate),
    .compute = negate_compute,
};


/* ================================================================
 * passthru: out = in (fpsexec only)
 * ============================================================= */

static char pass_in[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char pass_out[FUNCTION_PARAMETER_STRMAXLEN] = "";

#define RT_PASSTHRU_PARAMS(X)                                                            \
    X(".in_name", pass_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM, "input") \
    X(".out_name", pass_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

MILK_PARAMS_DEFINE(passthru, RT_PASSTHRU_PARAMS);

static errno_t passthru_compute(const MILK_RUNNABLE *self)
{
    return rt_affine_run(self, pass_in, pass_out, &rt_one, &rt_zero);
}

const MILK_RUNNABLE runnable_rt_passthru = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "rtpassthru",
                 .cmdkey      = "passthru",
                 .description = "runnabletest: out = in (fpsexec only)" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(passthru),
    .compute = passthru_compute,
};
EXPORT_FPSEXEC(milk, "runnabletest-passthru", runnable_rt_passthru);
