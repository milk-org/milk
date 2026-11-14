// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file im3Dto2D.c
 * @brief Collapse first 2 axis of 3D image
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID image_basic_3Dto2D(const char *__restrict IDname);

static char p_in[FUNCTION_PARAMETER_STRMAXLEN] = "im1";

#define FPS_PARAMS(X) X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")

MILK_PARAMS_DEFINE(im3Dto2D, FPS_PARAMS);

static MILK_HOT errno_t im3Dto2D_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    image_basic_3Dto2D(p_in);
    return RETURN_SUCCESS;
}

/* ----------------------------------------------------------------------
 *
 * turns a 3D image into a 2D image by collapsing first 2 axis
 *
 *
 * ---------------------------------------------------------------------- */

imageID image_basic_3Dto2D_byID(imageID ID)
{
    if (dcimg[ID].md[0].naxis != 3)
    {
        printf("ERROR: image needs to have 3 axis\n");
    }
    else
    {
        uint64_t d0    = 0;
        uint64_t d1    = 0;
        int      is_3d = 0;
        arith_image_flatten2d(&dcimg[ID], &d0, &d1, &is_3d);
        dcimg[ID].md[0].size[0] = (uint32_t) d0;
        dcimg[ID].md[0].size[1] = (uint32_t) d1;
        dcimg[ID].md[0].naxis   = 2;
    }

    return ID;
}

imageID image_basic_3Dto2D(const char *__restrict IDname)
{
    imageID ID;

    ID = image_ID(IDname, dcimg, dcnimg);
    image_basic_3Dto2D_byID(ID);

    return ID;
}

const MILK_RUNNABLE runnable_image_basic_im3Dto2D = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "im3Dto2D",
              .cmdkey           = "im3Dto2D",
              .description      = "collapse first 2 axis of 3D image",
              .description_long = "Collapse the first two axes of a 3D image into a single axis. "
                                  "Reshapes (x, y, z) into (x*y, z) for vectorized processing." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(im3Dto2D),
    .compute = im3Dto2D_compute,
};
