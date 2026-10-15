// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_paste.c
 * @brief   paste two 2D streams into output
 *
 * Uses FPS V2 framework.
 */

#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "stream_sem.h"
#include "libmilkcommon/pixel_dispatch.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char p_stream0[FUNCTION_PARAMETER_STRMAXLEN] = "stream0";

static char p_stream1[FUNCTION_PARAMETER_STRMAXLEN] = "stream1";

static char p_outstream[FUNCTION_PARAMETER_STRMAXLEN] = "outstream";

static long long p_semtrig0 = 3;
static long long p_semtrig1 = 3;
static long long p_master   = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".in_stream0", p_stream0, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream 0")   \
    X(".in_stream1", p_stream1, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream 1")   \
    X(".out_stream", p_outstream, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_OUTPUT, "output stream") \
    X(".semtrig0", &p_semtrig0, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "sem trigger 0")         \
    X(".semtrig1", &p_semtrig1, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "sem trigger 1")         \
    X(".master", &p_master, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "master frame index")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * Paste two 2D streams side-by-side into one
 * output stream. Triggers alternately on stream0
 * and stream1.
 */
imageID COREMOD_MEMORY_streamPaste(const char *IDstream0_name,
                                   const char *IDstream1_name,
                                   const char *IDstreamout_name,
                                   long        semtrig0,
                                   long        semtrig1,
                                   int         master)
{
    IMGID img0 = imgid_make_from_name(IDstream0_name);
    resolveIMGID(&img0, ERRMODE_WARN, dcimg, dcnimg);
    if (img0.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID img1 = imgid_make_from_name(IDstream1_name);
    resolveIMGID(&img1, ERRMODE_WARN, dcimg, dcnimg);
    if (img1.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t xsize    = img0.md->size[0];
    uint32_t ysize    = img0.md->size[1];
    uint8_t  datatype = img0.md->datatype;

    IMGID imgout = imgid_make_from_name(IDstreamout_name);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if (imgout.ID == -1)
    {
        imgout = stream_connect_create_2D(IDstreamout_name, 2 * xsize, ysize, datatype);
    }

    IMGID              imgin[2]    = { img0, img1 };
    long               semtrigs[2] = { semtrig0, semtrig1 };
    int                FrameIndex  = 0;
    unsigned long long cnt         = 0;

    while (1)
    {
        IMGID *cur = &imgin[FrameIndex];
        if (cur->md->sem == 0)
        {
            while (cnt == cur->md->cnt0)
            {
                usleep(5);
            }
            cnt = cur->md->cnt0;
        }
        else
        {
            ImageStreamIO_semwait(cur->im, semtrigs[FrameIndex]);
        }

        long Xoffset = FrameIndex * xsize;

        imgout.md->write = 1;

#define PASTE_CASE_(DT, ACC, CT)                                      \
    case DT:                                                          \
        for (uint32_t ii = 0; ii < xsize; ii++)                       \
            for (uint32_t jj = 0; jj < ysize; jj++)                   \
            {                                                         \
                imgout.im->array.ACC[jj * 2 * xsize + ii + Xoffset] = \
                    cur->im->array.ACC[jj * xsize + ii];              \
            }                                                         \
        break;

        switch (datatype)
        {
            FOREACH_REAL_DATATYPE(PASTE_CASE_)
        case _DATATYPE_COMPLEX_FLOAT:
            for (uint32_t ii = 0; ii < xsize; ii++)
            {
                for (uint32_t jj = 0; jj < ysize; jj++)
                {
                    imgout.im->array.CF[jj * 2 * xsize + ii + Xoffset] =
                        cur->im->array.CF[jj * xsize + ii];
                }
            }
            break;
        case _DATATYPE_COMPLEX_DOUBLE:
            for (uint32_t ii = 0; ii < xsize; ii++)
            {
                for (uint32_t jj = 0; jj < ysize; jj++)
                {
                    imgout.im->array.CD[jj * 2 * xsize + ii + Xoffset] =
                        cur->im->array.CD[jj * xsize + ii];
                }
            }
            break;
        default:
            PRINT_ERROR("Unknown data type");
            break;
        }
#undef PASTE_CASE_

        if (FrameIndex == master)
        {
            COREMOD_MEMORY_image_set_sempost_byID(imgout.ID, -1);
            imgout.md->cnt0++;
        }
        imgout.md->cnt1  = FrameIndex;
        imgout.md->write = 0;

        FrameIndex = (FrameIndex == 0) ? 1 : 0;
    }

    return imgout.ID;
}


/* ================================================================
 * 5.  RUNNABLE
 * ============================================================= */

MILK_PARAMS_DEFINE(streampaste, FPS_PARAMS);

static MILK_HOT errno_t streampaste_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    COREMOD_MEMORY_streamPaste(p_stream0, p_stream1, p_outstream, p_semtrig0, p_semtrig1, p_master);

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_streampaste = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "streampaste",
                 .cmdkey      = "streampaste",
                 .description = "paste two 2D streams",
                 .description_long =
                     "Paste (overlay) one image stream onto a region of another. The source "
                     "stream is copied into the destination at a specified offset position." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(streampaste),
    .compute = streampaste_compute,
};
