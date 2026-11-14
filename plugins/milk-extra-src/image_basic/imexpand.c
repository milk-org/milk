// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imexpand.c
 * @brief Expand/upsample images
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Forward declarations
imageID basic_expand(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_expand3D(const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);

/* ---- Command 1: imexpand ---- */

static char      pe1_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char      pe1_out[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static long long pe1_fx                                = 2;
static long long pe1_fy                                = 2;

#define FPS_PARAMS_1(X)                                                              \
    X(".in_name", pe1_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", pe1_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".factx", &pe1_fx, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x expand factor")   \
    X(".facty", &pe1_fy, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y expand factor")

MILK_PARAMS_DEFINE(imexpand, FPS_PARAMS_1);


static MILK_HOT errno_t imexpand_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    basic_expand(pe1_in, pe1_out, (int) pe1_fx, (int) pe1_fy);
    return RETURN_SUCCESS;
}

/* ---- Command 2: imexpand3D ---- */

static char      pe2_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char      pe2_out[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static long long pe2_fx                                = 2;
static long long pe2_fy                                = 2;
static long long pe2_fz                                = 2;

#define FPS_PARAMS_2(X)                                                              \
    X(".in_name", pe2_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", pe2_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".factx", &pe2_fx, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x expand factor")   \
    X(".facty", &pe2_fy, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y expand factor")   \
    X(".factz", &pe2_fz, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "z expand factor")

MILK_PARAMS_DEFINE(imexpand3D, FPS_PARAMS_2);


static MILK_HOT errno_t imexpand3D_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    basic_expand3D(pe2_in, pe2_out, (int) pe2_fx, (int) pe2_fy, (int) pe2_fz);
    return RETURN_SUCCESS;
}

/**
 * Expand image by factor n1 along x and
 * n2 along y (nearest-neighbour).
 */
imageID basic_expand(const char *ID_name, const char *ID_name_out, int n1, int n2)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long naxes[2];
    naxes[0] = imgin.md->size[0];
    naxes[1] = imgin.md->size[1];
    long naxes_out[2];
    naxes_out[0] = naxes[0] * n1;
    naxes_out[1] = naxes[1] * n2;

    IMGID imgout       = imgid_make_from_name_2D(ID_name_out, naxes_out[0], naxes_out[1]);
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    (void) arith_image_expand3d_im(imgin.im, imgout.im, n1, n2, 1);

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}

/**
 * Expand 3D image by factors n1, n2, n3
 * along x, y, z (nearest-neighbour).
 */
imageID basic_expand3D(const char *ID_name, const char *ID_name_out, int n1, int n2, int n3)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long naxes[3];
    naxes[0] = imgin.md->size[0];
    naxes[1] = (imgin.md->naxis > 1) ? imgin.md->size[1] : 1;
    naxes[2] = (imgin.md->naxis == 3) ? imgin.md->size[2] : 1;

    long naxes_out[3];
    naxes_out[0] = naxes[0] * n1;
    naxes_out[1] = naxes[1] * n2;
    naxes_out[2] = naxes[2] * n3;

    IMGID imgout = imgid_make_from_name_3D(ID_name_out, naxes_out[0], naxes_out[1], naxes_out[2]);
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    (void) arith_image_expand3d_im(imgin.im, imgout.im, n1, n2, n3);

    return imgout.ID;
}

const MILK_RUNNABLE runnable_image_basic_imexpand = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imexpand",
                 .cmdkey           = "imexpand",
                 .description      = "expand 2D image",
                 .description_long = "Expand (upsample) a 2D or 3D image by integer factors. Each "
                                     "input pixel is replicated into a block of output pixels." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imexpand),
    .compute = imexpand_compute,
};

const MILK_RUNNABLE runnable_image_basic_imexpand3D = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imexpand3D",
                 .cmdkey           = "imexpand3D",
                 .description      = "expand 3D image",
                 .description_long = "Expand (upsample) a 2D or 3D image by integer factors. Each "
                                     "input pixel is replicated into a block of output pixels." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imexpand3D),
    .compute = imexpand3D_compute,
};
