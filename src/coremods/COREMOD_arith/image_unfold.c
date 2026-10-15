// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_unfold.c
 * @brief Image unfold module
 */

#include <stdint.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

// input image names
static char     inimname[FUNCTION_PARAMETER_STRMAXLEN];
static char     outimname[FUNCTION_PARAMETER_STRMAXLEN];
static uint32_t axisA   = 0;
static uint32_t axisB   = 0;
static uint32_t colsize = 1;


#define FPS_PARAMS(X)                                                                   \
    X(".inim", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")     \
    X(".outim", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")      \
    X(".axisA", &axisA, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "axis to merged")       \
    X(".axisB", &axisB, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "merge into this axis") \
    X(".colsize", &colsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "column size")

errno_t image_unfold(IMGID inimg, IMGID *outimg, uint8_t axisA, uint8_t axisB, int colsize)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);

    resolveIMGID(outimg, ERRMODE_NULL, dcimg, dcnimg);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (outimg->ID == -1)
    {
        imgid_copy(&inimg, outimg);
    }

    // output image size
    outimg->mdt->naxis = inimg.md->naxis - 1;

    // remove missing axis
    {
        uint8_t axout = 0;
        for (uint8_t axin = 0; axin < inimg.md->naxis; axin++)
        {
            if (axin != axisA)
            {
                outimg->mdt->size[axout] = inimg.md->size[axin];
                axout++;
            }
        }
    }

    // destination axis to grow
    uint8_t axis0 = 0;
    if (axisA > axisB)
    {
        axis0 = axisB;
    }
    else
    {
        axis0 = axisB - 1;
    }

    // overflow destination axis to grow
    uint8_t axis1 = 0;
    uint8_t axisC = 0; // in input image
    if ((axis0 == 0) && (outimg->mdt->naxis > 1))
    {
        axis1 = 1;
        axisC = 1;
    }

    int mdimsize = 0;
    if (axis0 == axis1)
    {
        outimg->mdt->size[axis0] *= inimg.md->size[axisA];
    }
    else
    {
        int mdim0 = 0; // multiplicative on axis0
        int mdim1 = 1; // multiplicative on axis1

        for (uint32_t ii = 0; ii < inimg.md->size[axisA]; ii++)
        {
            mdim0++;
            if (mdim0 == colsize)
            {
                mdim0 = 0;
                mdim1++;
            }
        }
        if (mdim1 > 0)
        {
            mdim0 = colsize;
        }

        outimg->mdt->size[axis0] *= mdim0;
        outimg->mdt->size[axis1] *= mdim1;

        mdimsize = inimg.md->size[axisC] * outimg->mdt->size[axis0];
    }

    createimagefromIMGID(outimg);

    // copy data to ouput

    // destination pix coord
    uint32_t ii = 0;
    uint32_t jj = 0;

    uint64_t pixi = 0;
    uint64_t pixo = 0;
    for (uint32_t pixi2 = 0; pixi2 < inimg.md->size[2]; pixi2++)
    {
        for (uint32_t pixi1 = 0; pixi1 < inimg.md->size[1]; pixi1++)
        {
            for (uint32_t pixi0 = 0; pixi0 < inimg.md->size[0]; pixi0++)
            {
                pixo = jj;
                pixo *= outimg->md->size[0];

                pixo += ii % outimg->md->size[0];
                pixo += mdimsize * (ii / outimg->md->size[0]);

                outimg->im->array.F[pixo] = inimg.im->array.F[pixi];
                pixi++;

                ii++;
            }
            if ((axisA == 1) && (axisB == 0))
            {
                // do nothing
            }
            else
            {
                jj++;
                ii -= inimg.md->size[0];
            }
        }

        if ((axisA == 2) && (axisB == 1))
        {
            // do nothing
        }
        else
        {
            ii += inimg.md->size[0];
            jj -= inimg.md->size[1];
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

MILK_PARAMS_DEFINE(image_unfold, FPS_PARAMS);


static MILK_HOT errno_t image_unfold_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);

    IMGID outimg = imgid_make_from_name(outimname);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }

    MILK_PROCINFO_INIT(self)

    MILK_PROCINFO_LOOPSTART(self)
    {
        image_unfold(inimg, &outimg, axisA, axisB, colsize);

        processinfo_update_output_stream(processinfo, outimg.im, NULL);
    }
    MILK_PROCINFO_END(self)
    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_arith_image_unfold = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "unfold",
              .cmdkey           = "unfold",
              .description      = "image unfold, merge axis A into axis B",
              .description_long = "Reshape an image by merging one axis into another. For example, "
                                  "unfold a 3D cube (x, y, z) into a 2D image by merging z into "
                                  "y, producing dimensions (x, y*z). The total pixel count is "
                                  "preserved." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_unfold),
    .compute = image_unfold_compute,
};
EXPORT_FPSEXEC(milk, "arith-unfold", runnable_COREMOD_arith_image_unfold);
