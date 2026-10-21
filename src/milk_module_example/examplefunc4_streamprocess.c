// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_process_loop_simple.c
 * @brief   template for simple stream processing loop
 *
 * Example 4
 * Function has input stream and output stream.
 */

#include <math.h> // for sqrt()

#include "milk_runnable.h"
#include "examplefunc4_streamprocess.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Local variables pointers

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char outimname[FUNCTION_PARAMETER_STRMAXLEN] = "";

static uint32_t *cntindex;
static uint32_t *cntindexmax;

static int64_t *ex0mode;
static int64_t *ex1mode;

#define STREAMPROCESS_PARAMS(X)                                                        \
    X(".in_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".cntindex", &cntindex, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "counter index") \
    X(".cntindexmax", &cntindexmax, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT,            \
      "counter index max value")                                                       \
    X(".option.ex0mode", &ex0mode, FPTYPE_ONOFF, 1, FPFLAG_DEFAULT_INPUT, "toggle0")   \
    X(".option.ex1mode", &ex1mode, FPTYPE_ONOFF, 1, FPFLAG_DEFAULT_INPUT,              \
      "toggle1 conditional on toggle0")

MILK_PARAMS_DEFINE(streamprocess, STREAMPROCESS_PARAMS);


// Optional custom configuration checks
static MILK_COLD errno_t streamprocess_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        long fpi_ex0mode = functionparameter_GetParamIndex(dcfpsptr, ".option.ex0mode");
        long fpi_ex1mode = functionparameter_GetParamIndex(dcfpsptr, ".option.ex1mode");

        if (fpi_ex0mode >= 0 && fpi_ex1mode >= 0)
        {
            if (dcfpsptr->parray[fpi_ex0mode].fpflag & FPFLAG_ONOFF) // if ex0mode is in ON state
            {
                // Then activate ex1mode argument
                dcfpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_USED;
                dcfpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_VISIBLE;
            }
            else // OFF state
            {
                dcfpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_USED;
                dcfpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_VISIBLE;
            }
        }

        // increment counter at every configuration check
        *cntindex = *cntindex + 1;

        if (*cntindex >= *cntindexmax)
        {
            *cntindex = 0;
        }
    }

    return RETURN_SUCCESS;
}


/**
 * @brief process image to output image
 *
 * Function arguments:
 * Input and output are passed by reference as
 * they may be changed by resolveIMGID and imcreateIMGID.
 *
 * Make sure to pass by reference if the function may change
 * IMGID
 */
static errno_t streamprocess(IMGID *inimg, IMGID *outimg)
{
    DEBUG_TRACE_FSTART();
    // custom stream process function code

    // resolve image
    // This function call has low overhead, as it will acknowledge existing image
    resolveIMGID(inimg, ERRMODE_WARN, dcimg, dcnimg);

    uint32_t xsize = inimg->mdt->size[0];
    if (inimg->ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint32_t ysize  = inimg->mdt->size[1];
    uint64_t xysize = xsize * ysize;

    // outimg is pre-created before the loop.
    // No allocation here.

    outimg->md->write = 1;

    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        outimg->im->array.F[ii] = sqrtf(inimg->im->array.F[ii]);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t streamprocess_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();


    // Check if image is in memory
    // First, create an IMGIG with the image name
    IMGID inimg = imgid_make_from_name(inimname);
    // Then resolve it (connect it to an image in memory if possible)
    // Once the image is resolved, this function will execute very quickly, only checking if resolved
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);

    // Create output image/stream.
    // Here we only fill in the name.
    // The image itself will be created in the compute function.
    IMGID outimg = imgid_make_from_name(outimname);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }

    // If we are sure we want outimg to be the same format (size, type etc) as inimg, we can use:
    imgid_copy(&inimg, &outimg);

    // Allocate output image once, before the loop.
    imcreateIMGID(&outimg);

    printf(" COMPUTE Flags = %ld\n", self->rt->cmddata.cmdsettings->flags);
    MILK_PROCINFO_INIT(self)

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", self->rt->cmddata.cmdsettings->flags);
    if (self->rt->cmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    MILK_PROCINFO_LOOPSTART(self)
    {
        streamprocess(&inimg, &outimg);

        processinfo_update_output_stream(processinfo, outimg.im, inimg.im);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


const MILK_RUNNABLE runnable_modex_streamprocess = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "streamprocess",
                 .cmdkey      = "streamprocess",
                 .description = "process input stream to output stream" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(streamprocess),
    .compute   = streamprocess_compute,
    .confcheck = streamprocess_confcheck,
};
EXPORT_FPSEXEC(milk, "streamprocess", runnable_modex_streamprocess);
