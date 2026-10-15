// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    im3D_to_stream2D.c
 * @brief   convert 3D image to 2D stream
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    inimname[FUNCTION_PARAMETER_STRMAXLEN] = "im3d";
static char    outname[FUNCTION_PARAMETER_STRMAXLEN]  = "im2d";
static int64_t slice_index                            = 0;
static int32_t loop_mode                              = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                             \
    X(".in_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input 3D image")         \
    X(".outname", outname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output stream name")          \
    X(".slice_index", &slice_index, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "initial slice index") \
    X(".loop_mode", &loop_mode, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT, "loop through slices")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Extract 2D slice from 3D image.
 */
static errno_t extract_slice_to_2D(IMGID *inimg, IMGID *outimg, long slice_idx)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(inimg, ERRMODE_ABORT, dcimg, dcnimg);

    if (inimg->md->naxis != 3)
    {
        FUNC_RETURN_FAILURE("Input image is not 3D");
    }

    uint32_t xsize = inimg->mdt->size[0];
    uint32_t ysize = inimg->mdt->size[1];
    uint32_t zsize = inimg->mdt->size[2];

    if (slice_idx < 0 || slice_idx >= zsize)
    {
        FUNC_RETURN_FAILURE("Slice index out of bounds");
    }

    // Image is created once before the loop;
    // do not allocate inside the hot path.

    outimg->md->write = 1;

    long framesize = xsize * ysize * ImageStreamIO_typesize(inimg->md->datatype);

    __builtin_memcpy(outimg->im->array.raw, inimg->im->array.raw + slice_idx * framesize,
                     framesize);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(im3D_to_stream2D, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t im3D_to_stream2D_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT, dcimg, dcnimg);

    IMGID outimg;
    outimg               = imgid_make_from_name_2D(outname, inimg.mdt->size[0], inimg.mdt->size[1]);
    outimg.mdt->shared   = 1;
    outimg.mdt->datatype = inimg.md->datatype;

    // Allocate output image once before the loop.
    imcreateIMGID(&outimg);

    MILK_PROCINFO_INIT(self)

    MILK_PROCINFO_LOOPSTART(self)
    {
        if (loop_mode == 0)
        {
            extract_slice_to_2D(&inimg, &outimg, slice_index);
        }
        else
        {
            slice_index = (slice_index + 1) % inimg.mdt->size[2];
            extract_slice_to_2D(&inimg, &outimg, slice_index);
        }

        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }
    MILK_PROCINFO_END(self)
    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_im3D_to_stream2D = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "im3Dto2D",
              .cmdkey      = "im3D_to_stream2D",
              .description = "convert 3D image to 2D stream",
              .description_long =
                  "Convert a static 3D image cube into a 2D shared memory stream by sequentially "
                  "playing back each slice as a frame. Useful for testing stream consumers with "
                  "pre-recorded data." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(im3D_to_stream2D),
    .compute = im3D_to_stream2D_compute,
};
