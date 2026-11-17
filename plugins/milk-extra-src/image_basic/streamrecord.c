// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file streamrecord.c
 * @brief Record stream of images
 */

#include <sched.h>

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_BASIC_streamrecord(const char *__restrict streamname,
                                 long NBframes,
                                 const char *__restrict IDname);

static char      p_stream[FUNCTION_PARAMETER_STRMAXLEN] = "imstream";
static long long p_nframes                              = 100;
static char      p_out[FUNCTION_PARAMETER_STRMAXLEN]    = "imrec";

#define FPS_PARAMS(X)                                                                    \
    X(".stream", p_stream, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream")   \
    X(".nframes", &p_nframes, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "number of frames") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")

MILK_PARAMS_DEFINE(streamrecord, FPS_PARAMS);

static MILK_HOT errno_t streamrecord_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMAGE_BASIC_streamrecord(p_stream, (long) p_nframes, p_out);
    return RETURN_SUCCESS;
}

/**
 * Record NBframes from a float stream
 * into a 3D cube.
 */
imageID IMAGE_BASIC_streamrecord(const char *__restrict streamname,
                                 long NBframes,
                                 const char *__restrict IDname)
{
    IMGID imgin = imgid_make_from_name(streamname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long xsize  = imgin.md->size[0];
    long ysize  = imgin.md->size[1];
    long xysize = xsize * ysize;

    IMGID imgout       = imgid_make_from_name_3D(IDname, xsize, ysize, NBframes);
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    unsigned long cnt         = imgin.md->cnt0;
    long          kk          = 0;
    long          waitdelayus = 50;

    char *ptr = (char *) imgout.im->array.F;
    while (kk != NBframes)
    {
        while (cnt > imgin.md->cnt0)
        {
            usleep(waitdelayus);
        }

        cnt++;

        printf("\r%ld / %ld  [%lu %lu]"
               "      ",
               kk, NBframes, cnt, imgout.md->cnt0);
        fflush(stdout);

        memcpy(ptr, imgin.im->array.F, sizeof(float) * xysize);
        ptr += sizeof(float) * xysize;
        kk++;
    }
    printf("\n\n");

    return imgout.ID;
}

const MILK_RUNNABLE runnable_image_basic_streamrecord = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imgstreamrec",
                 .cmdkey           = "imgstreamrec",
                 .description      = "record stream of images",
                 .description_long = "Record frames from a 2D shared memory stream into a 3D cube "
                                     "or sequence of FITS files on disk." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(streamrecord),
    .compute = streamrecord_compute,
};
