// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imcontract.c
 * @brief Image binning
 */

#include "milk_runnable.h"
#include "computeunit.h"

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
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (uint32_t jj = 0; jj < naxes_out[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes_out[0]; ii++)
        {
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    imgout.im->array.F[jj * naxes_out[0] + ii] +=
                        imgin.im->array.F[(jj * n2 + j) * imgin.md->size[0] + ii * n1 + i];
                }
            }
        }
    }

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
        printf("(%ld x %ld x %ld)"
               "  ->  "
               "(%ld x %ld x %ld)\n",
               (long) imgin.md->size[0], (long) imgin.md->size[1], (long) imgin.md->size[2],
               (long) naxes_out[0], (long) naxes_out[1], (long) naxes_out[2]);
        imgout.mdt->naxis    = 3;
        imgout.mdt->size[0]  = naxes_out[0];
        imgout.mdt->size[1]  = naxes_out[1];
        imgout.mdt->size[2]  = naxes_out[2];
        imgout.mdt->datatype = datatype;
    }
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    switch (datatype)
    {
    case _DATATYPE_FLOAT:
        for (uint32_t jj = 0; jj < naxes_out[1]; jj++)
        {
            for (uint32_t ii = 0; ii < naxes_out[0]; ii++)
            {
                for (uint32_t kk = 0; kk < naxes_out[2]; kk++)
                {
                    for (int i = 0; i < n1; i++)
                    {
                        for (int j = 0; j < n2; j++)
                        {
                            for (int k = 0; k < n3; k++)
                            {
                                imgout.im->array
                                    .F[kk * naxes_out[0] * naxes_out[1] + jj * naxes_out[0] + ii] +=
                                    imgin.im->array
                                        .F[(kk * n3 + k) * imgin.md->size[0] * imgin.md->size[1] +
                                           (jj * n2 + j) * imgin.md->size[0] + ii * n1 + i];
                            }
                        }
                    }
                }
            }
        }
        break;
    case _DATATYPE_DOUBLE:
        for (uint32_t jj = 0; jj < naxes_out[1]; jj++)
        {
            for (uint32_t ii = 0; ii < naxes_out[0]; ii++)
            {
                for (uint32_t kk = 0; kk < naxes_out[2]; kk++)
                {
                    for (int i = 0; i < n1; i++)
                    {
                        for (int j = 0; j < n2; j++)
                        {
                            for (int k = 0; k < n3; k++)
                            {
                                imgout.im->array
                                    .D[kk * naxes_out[0] * naxes_out[1] + jj * naxes_out[0] + ii] +=
                                    imgin.im->array
                                        .D[(kk * n3 + k) * imgin.md->size[0] * imgin.md->size[1] +
                                           (jj * n2 + j) * imgin.md->size[0] + ii * n1 + i];
                            }
                        }
                    }
                }
            }
        }
        break;
    case _DATATYPE_COMPLEX_FLOAT:
        for (uint32_t jj = 0; jj < naxes_out[1]; jj++)
        {
            for (uint32_t ii = 0; ii < naxes_out[0]; ii++)
            {
                for (uint32_t kk = 0; kk < naxes_out[2]; kk++)
                {
                    long ooff = kk * naxes_out[0] * naxes_out[1] + jj * naxes_out[0] + ii;
                    for (int i = 0; i < n1; i++)
                    {
                        for (int j = 0; j < n2; j++)
                        {
                            for (int k = 0; k < n3; k++)
                            {
                                long ioff = (kk * n3 + k) * imgin.md->size[0] * imgin.md->size[1] +
                                            (jj * n2 + j) * imgin.md->size[0] + ii * n1 + i;
                                imgout.im->array.CF[ooff].re += imgin.im->array.CF[ioff].re;
                                imgout.im->array.CF[ooff].im += imgin.im->array.CF[ioff].im;
                            }
                        }
                    }
                }
            }
        }
        break;
    case _DATATYPE_COMPLEX_DOUBLE:
        for (uint32_t jj = 0; jj < naxes_out[1]; jj++)
        {
            for (uint32_t ii = 0; ii < naxes_out[0]; ii++)
            {
                for (uint32_t kk = 0; kk < naxes_out[2]; kk++)
                {
                    long ooff = kk * naxes_out[0] * naxes_out[1] + jj * naxes_out[0] + ii;
                    for (int i = 0; i < n1; i++)
                    {
                        for (int j = 0; j < n2; j++)
                        {
                            for (int k = 0; k < n3; k++)
                            {
                                long ioff = (kk * n3 + k) * imgin.md->size[0] * imgin.md->size[1] +
                                            (jj * n2 + j) * imgin.md->size[0] + ii * n1 + i;
                                imgout.im->array.CD[ooff].re += imgin.im->array.CD[ioff].re;
                                imgout.im->array.CD[ooff].im += imgin.im->array.CD[ioff].im;
                            }
                        }
                    }
                }
            }
        }
        break;
    }

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
