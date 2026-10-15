// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_copy.c
 * @brief   copy image stream
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"
#include "libmilkcommon/milk_compiler.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static char outimname[FUNCTION_PARAMETER_STRMAXLEN] = "imout";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                    \
    X(".in_sname", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream") \
    X(".out_sname", outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output stream")


/* ================================================================
 * 5.  RUNNABLE
 * ============================================================= */

MILK_PARAMS_DEFINE(shmimcopy, FPS_PARAMS);

static MILK_HOT errno_t shmimcopy_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(inimname);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID imgout = imgid_make_from_name(outimname);
    resolveIMGID(&imgout, ERRMODE_ABORT, dcimg, dcnimg);

    uint64_t im_in_datasize =
        ImageStreamIO_typesize(imgin.im->md->datatype) * imgin.im->md->nelement;
    uint64_t im_out_datasize =
        ImageStreamIO_typesize(imgin.im->md->datatype) * imgout.im->md->nelement;
    uint64_t byte_copy_size = im_in_datasize < im_out_datasize ? im_in_datasize : im_out_datasize;

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        ImageStreamIO_semflush(imgin.im, processinfo->triggersem);

        if (imgin.im->md->datatype == _DATATYPE_FLOAT)
        {
            float *MILK_RESTRICT       out = MILK_ASSUME_ALIGNED(imgout.im->array.F);
            const float *MILK_RESTRICT in  = MILK_ASSUME_ALIGNED(imgin.im->array.F);

            __builtin_memcpy(out, in, byte_copy_size);
        }
        else if (imgin.im->md->datatype == _DATATYPE_UINT16)
        {
            uint16_t *MILK_RESTRICT       out = MILK_ASSUME_ALIGNED(imgout.im->array.UI16);
            const uint16_t *MILK_RESTRICT in  = MILK_ASSUME_ALIGNED(imgin.im->array.UI16);

            __builtin_memcpy(out, in, byte_copy_size);
        }
        else
        {
            __builtin_memcpy(imgout.im->array.raw, imgin.im->array.raw, byte_copy_size);
        }

        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_shmimcopy = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "shmimcopy",
              .cmdkey      = "shmimcopy",
              .description = "copy in stream to existing out stream",
              .description_long =
                  "Copy frames from one shared memory stream to another in real-time. Triggered "
                  "by semaphore on the input stream. Supports type conversion between streams of "
                  "different data types." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(shmimcopy),
    .compute = shmimcopy_compute,
};
EXPORT_FPSEXEC(milk, "mem-streamcopy", runnable_COREMOD_memory_shmimcopy);
