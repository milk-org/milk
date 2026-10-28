// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imcube_product.c
 * @brief   Compute product between two image cubes
 *
 *
 *
 */

#include <math.h>

#include "milk_runnable.h"
#include "linopt_imtools.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimc0[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char inimc1[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char inimmask[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char imout[FUNCTION_PARAMETER_STRMAXLEN]    = "";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                       \
    X(".imcube0", inimc0, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image cube 0") \
    X(".imcube1", inimc1, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image cube 1") \
    X(".immask", inimmask, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "pixel mask")        \
    X(".outim", imout, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output matrix")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(imcube_crossproduct, FPS_PARAMS);

static errno_t imcube_crossproduct(IMGID imgcube0, IMGID imgcube1, IMGID imgmask, char *imoutname)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&imgcube0, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(&imgcube1, ERRMODE_WARN, dcimg, dcnimg);
    if (imgcube0.ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (imgcube1.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imgmask, ERRMODE_WARN, dcimg, dcnimg);

    uint32_t xsize = imgcube0.md->size[0];
    if (imgmask.ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint32_t ysize  = imgcube0.md->size[1];
    uint32_t zsize0 = imgcube0.md->size[2];
    uint32_t zsize1 = imgcube1.md->size[2];

    uint64_t xysize = xsize * ysize;

    IMGID imgout = imgid_make_from_name_2D(imoutname, zsize0, zsize1);
    createimagefromIMGID(&imgout);

    const float *restrict mptr = imgmask.im->array.F;
    const float *restrict c0   = imgcube0.im->array.F;
    const float *restrict c1   = imgcube1.im->array.F;
    float *restrict out        = imgout.im->array.F;

    // compute mask sum
    double masksum = 0.0;
#pragma omp simd reduction(+ : masksum)
    for (uint64_t pixi = 0; pixi < xysize; pixi++)
    {
        masksum += mptr[pixi];
    }

    int nth = milk_omp_team_size((uint64_t) zsize0 * (uint64_t) zsize1 * xysize, 262144);
#pragma omp      parallel for schedule(dynamic, 1) num_threads(nth) if (nth > 1)
    for (uint32_t kk0 = 0; kk0 < zsize0; kk0++)
    {
        const float *restrict s0 = c0 + xysize * kk0;
        for (uint32_t kk1 = kk0; kk1 < zsize1; kk1++)
        {
            const float *restrict s1 = c1 + xysize * kk1;
            double tmpv              = 0.0;
#pragma omp simd reduction(+ : tmpv)
            for (uint64_t pixi = 0; pixi < xysize; pixi++)
            {
                tmpv += mptr[pixi] * (s0[pixi] * s1[pixi]);
            }
            out[kk1 * zsize0 + kk0] = (float) (tmpv / masksum);
        }
    }
    imgid_free(&imgout);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Cross product of 2 image cubes
 *
 *
 * @return errno_t
 */

static MILK_HOT errno_t imcube_crossproduct_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    // connect to input mode values array and get number of modes
    //
    IMGID imginc0 = imgid_make_from_name(inimc0);
    IMGID imginc1 = imgid_make_from_name(inimc1);
    IMGID imgmask = imgid_make_from_name(inimmask);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    imcube_crossproduct(imginc0, imginc1, imgmask, imout);

    MILK_PROCINFO_END(self)

    imgid_free(&imginc0);
    imgid_free(&imginc1);
    imgid_free(&imgmask);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_linopt_imtools_imcube_crossproduct = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "imcubeXprod",
                 .cmdkey      = "imcubeXprod",
                 .description = "cross product of two image cubes",
                 .description_long =
                     "Compute the cross-product matrix of two image cubes. Each element (i,j) "
                     "is the dot product of slice i from cube A with slice j from cube B." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imcube_crossproduct),
    .compute = imcube_crossproduct_compute,
};
EXPORT_FPSEXEC(milk, "linopt-imcXprod", runnable_linopt_imtools_imcube_crossproduct);
