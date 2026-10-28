// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imgetcircasym.c
 * @brief Extract non-circular symmetric part
 */

#include <math.h>

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_BASIC_get_circasym_component(const char *__restrict ID_name,
                                           const char *__restrict ID_out_name,
                                           float       xcenter,
                                           float       ycenter,
                                           const char *options);

static char   p_in[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static char   p_out[FUNCTION_PARAMETER_STRMAXLEN] = "imout";
static double p_xcenter                           = 256.0;
static double p_ycenter                           = 230.5;

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".xcenter", &p_xcenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x center") \
    X(".ycenter", &p_ycenter, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y center")

MILK_PARAMS_DEFINE(imgetcircasym, FPS_PARAMS);

static MILK_HOT errno_t imgetcircasym_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMAGE_BASIC_get_circasym_component(p_in, p_out, (float) p_xcenter, (float) p_ycenter, "");
    return RETURN_SUCCESS;
}

/**
 * Extract the non-circularly symmetric
 * component of a 2D image (by imageID).
 */
imageID IMAGE_BASIC_get_circasym_component_byID(imageID ID,
                                                const char *__restrict ID_out_name,
                                                float       xcenter,
                                                float       ycenter,
                                                const char *options)
{
    (void) options;
    if (ID < 0)
    {
        return RETURN_FAILURE;
    }

    uint32_t naxes[2];
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    IMGID imgout       = imgid_make_from_name_2D(ID_out_name, naxes[0], naxes[1]);
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    (void) arith_image_circsym_im(&dcimg[ID], imgout.im, (double) xcenter, (double) ycenter, 1);

    return imgout.ID;
}

/**
 * Extract the non-circularly symmetric
 * component of a 2D image (by name).
 */
imageID IMAGE_BASIC_get_circasym_component(const char *__restrict ID_name,
                                           const char *__restrict ID_out_name,
                                           float       xcenter,
                                           float       ycenter,
                                           const char *options)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    return IMAGE_BASIC_get_circasym_component_byID(imgin.ID, ID_out_name, xcenter, ycenter,
                                                   options);
}

const MILK_RUNNABLE runnable_image_basic_imgetcircasym = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "imgetcircasym",
              .cmdkey           = "imgetcircasym",
              .description      = "extract non-circ symmetric part",
              .description_long = "Extract the non-circularly-symmetric component of a 2D image. "
                                  "Subtracts the azimuthal average to isolate asymmetric "
                                  "structures." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgetcircasym),
    .compute = imgetcircasym_compute,
};
