// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imcontract.c
 * @brief Image binning
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_arith/image_spatial.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Forward declarations
imageID basic_contract(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_contract3D(const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);

/* ---- Command 1: imcontract ---- */

static char      pc1_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char      pc1_out[FUNCTION_PARAMETER_STRMAXLEN] = "outim";
static long long pc1_bx                                = 4;
static long long pc1_by                                = 4;

#define FPS_PARAMS_1(X)                                                              \
    X(".in_name", pc1_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", pc1_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".binx", &pc1_bx, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x bin factor")       \
    X(".biny", &pc1_by, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y bin factor")

MILK_PARAMS_DEFINE(imcontract, FPS_PARAMS_1);


static MILK_HOT errno_t imcontract_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    basic_contract(pc1_in, pc1_out, (int) pc1_bx, (int) pc1_by);
    return RETURN_SUCCESS;
}

/* ---- Command 2: imcontract3D ---- */

static char      pc2_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char      pc2_out[FUNCTION_PARAMETER_STRMAXLEN] = "outim";
static long long pc2_bx                                = 4;
static long long pc2_by                                = 4;
static long long pc2_bz                                = 1;

#define FPS_PARAMS_2(X)                                                              \
    X(".in_name", pc2_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", pc2_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".binx", &pc2_bx, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x bin factor")       \
    X(".biny", &pc2_by, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y bin factor")       \
    X(".binz", &pc2_bz, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "z bin factor")

MILK_PARAMS_DEFINE(imcontract3D, FPS_PARAMS_2);


static MILK_HOT errno_t imcontract3D_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    basic_contract3D(pc2_in, pc2_out, (int) pc2_bx, (int) pc2_by, (int) pc2_bz);
    return RETURN_SUCCESS;
}

imageID basic_contract(const char *ID_name, const char *ID_name_out, int n1, int n2)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

    uint32_t naxes_out[2];
    naxes_out[0] = imgin.md->size[0] / n1;
    naxes_out[1] = imgin.md->size[1] / n2;

    IMGID imgout       = imgid_make_from_name_2D(ID_name_out, naxes_out[0], naxes_out[1]);
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    arith_image_bin3d_im(imgin.im, imgout.im, n1, n2, 1);

    return imgout.ID;
}

imageID basic_contract3D(const char *ID_name, const char *ID_name_out, int n1, int n2, int n3)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

    uint8_t datatype = imgin.md->datatype;

    uint32_t naxes_out[3];
    naxes_out[0] = imgin.md->size[0] / n1;
    naxes_out[1] = imgin.md->size[1] / n2;
    naxes_out[2] = imgin.md->size[2] / n3;

    IMGID imgout = imgid_make_from_name(ID_name_out);
    if (naxes_out[2] == 1)
    {
        imgout.mdt->naxis   = 2;
        imgout.mdt->size[0] = naxes_out[0];
        imgout.mdt->size[1] = naxes_out[1];
    }
    else
    {
        imgout.mdt->naxis    = 3;
        imgout.mdt->size[0]  = naxes_out[0];
        imgout.mdt->size[1]  = naxes_out[1];
        imgout.mdt->size[2]  = naxes_out[2];
        imgout.mdt->datatype = datatype;
    }
    imgout.mdt->shared = 0;
    imgid_create_registered(&imgout);

    arith_image_bin3d_im(imgin.im, imgout.im, n1, n2, n3);

    DEBUG_TRACE_FEXIT();

    return imgout.ID;
}

const MILK_RUNNABLE runnable_image_basic_imcontract = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imcontract",
                 .cmdkey           = "imcontract",
                 .description      = "image binning",
                 .description_long = "Bin (downsample) a 2D or 3D image by integer factors. Each "
                                     "output pixel is the sum or average of a block of input "
                                     "pixels." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imcontract),
    .compute = imcontract_compute,
};

const MILK_RUNNABLE runnable_image_basic_imcontract3D = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imcontract3D",
                 .cmdkey           = "imcontract3D",
                 .description      = "image binning (3D)",
                 .description_long = "Bin (downsample) a 2D or 3D image by integer factors. Each "
                                     "output pixel is the sum or average of a block of input "
                                     "pixels." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imcontract3D),
    .compute = imcontract3D_compute,
};
