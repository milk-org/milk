// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_crop.c
 * @brief   crop functions
 *
 * Uses FPS V2 framework.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "milk_runnable.h"

#include "COREMOD_memory/COREMOD_memory.h"

/* forward decls */
imageID arith_image_crop(const char *ID_name,
                         const char *ID_out,
                         int64_t    *start,
                         int64_t    *end,
                         int64_t     cropdim);

/**
 * @brief Extract a 2D sub-image from an image.
 *
 * Centers the extraction at (xc, yc) with the
 * specified output size.
 */
imageID arith_image_extract2D(const char *in_name,
                              const char *out_name,
                              int64_t     size_x,
                              int64_t     size_y,
                              int64_t     xstart,
                              int64_t     ystart);

imageID arith_image_extract3D(const char *in_name,
                              const char *out_name,
                              int64_t     size_x,
                              int64_t     size_y,
                              int64_t     size_z,
                              int64_t     xstart,
                              int64_t     ystart,
                              int64_t     zstart);


/* ================================================================
 *  PARAMS
 * ============================================================= */

static char    p_inname[FUNCTION_PARAMETER_STRMAXLEN]  = "im";
static char    p_outname[FUNCTION_PARAMETER_STRMAXLEN] = "ime";
static int64_t p_sizex                                 = 256;
static int64_t p_sizey                                 = 256;
static int64_t p_sizez                                 = 5;
static int64_t p_xstart                                = 100;
static int64_t p_ystart                                = 100;
static int64_t p_zstart                                = 0;


/* ================================================================
 *  CMD 1: extractim (6 args)
 * ============================================================= */

#define FPS_PARAMS_2D(X)                                                              \
    X(".inname", p_inname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".outname", p_outname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".sizex", &p_sizex, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size X")            \
    X(".sizey", &p_sizey, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size Y")            \
    X(".xstart", &p_xstart, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "X start")         \
    X(".ystart", &p_ystart, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "Y start")

MILK_PARAMS_DEFINE(image_crop_2d, FPS_PARAMS_2D);

static errno_t image_crop_2d_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    arith_image_extract2D(p_inname, p_outname, p_sizex, p_sizey, p_xstart, p_ystart);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_arith_extractim = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "extractim",
              .cmdkey      = "extractim",
              .description = "crop 2D image",
              .description_long =
                  "Extract a rectangular sub-region from a 2D or 3D image. Legacy implementation "
                  "supporting both 2D crop (extractim) and 3D crop (extract3Dim) in a single "
                  "file." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_crop_2d),
    .compute = image_crop_2d_compute,
};


/* ================================================================
 *  CMD 2: extract3Dim (8 args, primary)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                 \
    X(".inname", p_inname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".outname", p_outname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".sizex", &p_sizex, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size X")            \
    X(".sizey", &p_sizey, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size Y")            \
    X(".sizez", &p_sizez, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size Z")            \
    X(".xstart", &p_xstart, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "X start")         \
    X(".ystart", &p_ystart, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "Y start")         \
    X(".zstart", &p_zstart, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "Z start")

MILK_PARAMS_DEFINE(image_crop_3d, FPS_PARAMS);

static MILK_HOT errno_t image_crop_3d_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    arith_image_extract3D(p_inname, p_outname, p_sizex, p_sizey, p_sizez, p_xstart, p_ystart,
                          p_zstart);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_arith_extract3Dim = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "extract3Dim",
              .cmdkey      = "extract3Dim",
              .description = "crop 3D image",
              .description_long =
                  "Extract a rectangular sub-region from a 2D or 3D image. Legacy implementation "
                  "supporting both 2D crop (extractim) and 3D crop (extract3Dim) in a single "
                  "file." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_crop_3d),
    .compute = image_crop_3d_compute,
};

/**
 * @brief Crop an image to a rectangular sub-region.
 *
 * Extracts pixels within the specified coordinate
 * bounds and creates a new image.
 */
imageID arith_image_crop(const char *ID_name,
                         const char *ID_out,
                         int64_t    *start,
                         int64_t    *end,
                         int64_t     cropdim)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);

    int64_t naxis = imgin.md->naxis;
    if (naxis < 1 || naxis > 3)
    {
        PRINT_ERROR("unsupported naxis = %ld", naxis);
        return -1;
    }
    if (strcmp(ID_out, ID_name) == 0)
    {
        PRINT_ERROR("output image must differ from the input image %s", ID_name);
        return -1;
    }

    uint8_t  datatype    = imgin.md->datatype;
    int      typesize    = ImageStreamIO_typesize(datatype);
    uint32_t naxes[3]    = { 1, 1, 1 };
    uint32_t naxesout[3] = { 1, 1, 1 };
    int64_t  start_c[3]  = { 0, 0, 0 };
    int64_t  end_c[3]    = { 1, 1, 1 };
    int64_t  start_o[3]  = { 0, 0, 0 };

    if (typesize <= 0)
    {
        PRINT_ERROR("invalid datatype %d", (int) datatype);
        return -1;
    }

    for (int d = 0; d < naxis; d++)
    {
        naxes[d]    = imgin.md->size[d];
        naxesout[d] = (uint32_t) (end[d] - start[d]);
        start_o[d]  = start[d];
        start_c[d]  = (start[d] < 0) ? 0 : start[d];
        end_c[d]    = (end[d] > (int64_t) naxes[d]) ? (int64_t) naxes[d] : end[d];
    }

    delete_image_ID(ID_out, DELETE_IMAGE_ERRMODE_IGNORE);
    IMGID imgout      = imgid_make_from_name(ID_out);
    imgout.mdt->naxis = (uint8_t) naxis;
    for (int d = 0; d < naxis; d++)
    {
        imgout.mdt->size[d] = naxesout[d];
    }
    imgout.mdt->datatype = datatype;
    imgout.mdt->shared   = dcshareddft;
    imgout.mdt->NBkw     = NB_KEYWNODE_MAX;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);
    RegisterIMGID(&imgout, dcimg, dcnimg);
    imageID IDout = imgout.ID;

    if (cropdim != naxis)
    {
        printf("Error (arith_image_crop): cropdim [%ld] and naxis [%ld] are different\n", cropdim,
               naxis);
    }

    size_t  elemsize  = (size_t) typesize;
    int64_t row_elems = end_c[0] - start_c[0];
    if (row_elems > 0)
    {
        int64_t in_slice  = (int64_t) naxes[0] * (int64_t) naxes[1];
        int64_t out_slice = (int64_t) naxesout[0] * (int64_t) naxesout[1];
        for (int64_t kk = start_c[2]; kk < end_c[2]; kk++)
        {
            for (int64_t jj = start_c[1]; jj < end_c[1]; jj++)
            {
                int64_t dst_off =
                    ((kk - start_o[2]) * out_slice + (jj - start_o[1]) * (int64_t) naxesout[0] +
                     (start_c[0] - start_o[0])) *
                    (int64_t) elemsize;
                int64_t src_off =
                    (kk * in_slice + jj * (int64_t) naxes[0] + start_c[0]) * (int64_t) elemsize;
                __builtin_memcpy((char *) imgout.im->array.raw + dst_off,
                                 (char *) imgin.im->array.raw + src_off,
                                 (size_t) row_elems * elemsize);
            }
        }
    }

    return IDout;
}


/**
 * @brief Extract a 2D sub-image from an image.
 *
 * Centers the extraction at (xc, yc) with the
 * specified output size.
 */
imageID arith_image_extract2D(const char *in_name,
                              const char *out_name,
                              int64_t     size_x,
                              int64_t     size_y,
                              int64_t     xstart,
                              int64_t     ystart)
{
    IMGID img = imgid_make_from_name(in_name);
    resolveIMGID(&img, ERRMODE_ABORT, dcimg, dcnimg);
    int naxis = img.md->naxis;
    if (naxis < 2 || naxis > 3)
    {
        return -1;
    }

    int64_t start[3] = { xstart, ystart, 0 };
    int64_t end[3]   = { xstart + size_x, ystart + size_y,
                         (naxis > 2) ? (int64_t) img.md->size[2] : 1 };
    return arith_image_crop(in_name, out_name, start, end, naxis);
}

imageID arith_image_extract3D(const char *in_name,
                              const char *out_name,
                              int64_t     size_x,
                              int64_t     size_y,
                              int64_t     size_z,
                              int64_t     xstart,
                              int64_t     ystart,
                              int64_t     zstart)
{
    int64_t start[3] = { xstart, ystart, zstart };
    int64_t end[3]   = { xstart + size_x, ystart + size_y, zstart + size_z };
    return arith_image_crop(in_name, out_name, start, end, 3);
}
