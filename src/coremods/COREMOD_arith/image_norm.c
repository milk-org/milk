// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_norm.c
 * @brief   Compute per-slice norm of an image
 *
 * Uses FPS V2 framework.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_arith/image_stats.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *norm_inimname  = NULL;
static char     *norm_outimname = NULL;
static uint32_t *norm_sliceaxis = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                          \
    X(".in0name", &norm_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 0") \
    X(".outname", &norm_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image") \
    X(".axis", &norm_sliceaxis, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "norm axis")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Compute L2 norm per slice along axis.
 *
 * Pure computation — takes resolved IMGIDs.
 */
static errno_t image_slicenorm_IMGID(IMGID *inimg, IMGID *outimg, uint8_t sliceaxis)
{
    if (outimg->ID == -1)
    {
        imgid_copy(inimg, outimg);
        for (uint8_t ax = 0; ax < inimg->md->naxis; ax++)
        {
            if (ax != sliceaxis)
            {
                outimg->mdt->size[ax] = 1;
            }
        }
        outimg->mdt->datatype = _DATATYPE_FLOAT;

        /* Create output stream */
        outimg->im = (IMAGE *) calloc(1, sizeof(IMAGE));
        strncpy(outimg->name, norm_outimname, 79);
        ImageStreamIO_createIm_gpu(outimg->im, outimg->name, outimg->mdt->naxis, outimg->mdt->size,
                                   outimg->mdt->datatype, -1, 1, 10, 0, 0, 0);
        outimg->md = outimg->im->md;
    }

    uint32_t sizes[3] = { inimg->md->size[0], inimg->md->size[1], inimg->md->size[2] };
    if (inimg->md->naxis < 3)
    {
        sizes[2] = 1;
    }
    if (inimg->md->naxis < 2)
    {
        sizes[1] = 1;
    }
    uint8_t dt        = inimg->md->datatype;
    double *normarray = (double *) calloc(sizes[sliceaxis], sizeof(double));
    for (uint32_t k = 0; k < sizes[2]; k++)
    {
        for (uint32_t j = 0; j < sizes[1]; j++)
        {
            for (uint32_t i = 0; i < sizes[0]; i++)
            {
                uint64_t idx = (uint64_t) k * sizes[1] * sizes[0] + (uint64_t) j * sizes[0] + i;
                double   v   = arith_read_im_pix(inimg->im, dt, idx);
                uint32_t coords[3] = { i, j, k };
                normarray[coords[sliceaxis]] += v * v;
            }
        }
    }
    for (uint32_t i = 0; i < sizes[sliceaxis]; i++)
    {
        outimg->im->array.F[i] = sqrt(normarray[i]);
    }
    free(normarray);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_norm, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_norm_compute(const MILK_RUNNABLE *self)
{
    if (!norm_sliceaxis)
    {
        return RETURN_FAILURE;
    }
    IMGID idin = imgid_make_from_name(norm_inimname);
    resolveIMGID(&idin, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID idout = imgid_make_from_name(norm_outimname);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    image_slicenorm_IMGID(&idin, &idout, *norm_sliceaxis);
    processinfo_update_output_stream(processinfo, idout.im, idin.im);

    MILK_PROCINFO_END(self)
    imgid_free(&idin);
    imgid_free(&idout);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_arith_image_norm = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "normslice",
              .cmdkey           = "normslice",
              .description      = "image norm by slice",
              .description_long = "Compute the norm (sum of absolute values or RMS) of each slice "
                                  "in a 3D image cube. Returns a 1D array of per-slice norms. "
                                  "Useful for monitoring signal levels across time series or "
                                  "modal decompositions." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_norm),
    .compute = image_norm_compute,
};
EXPORT_FPSEXEC(milk, "arith-normslice", runnable_COREMOD_arith_image_norm);
