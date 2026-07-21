// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imswapaxis2D.c
 * @brief Swap axis of a 2D image
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID image_basic_SwapAxis2D(const char *__restrict IDin_name, const char *__restrict IDout_name);

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN] = "im2";

static FPS_APP_INFO FPS_app_info = { .fps_name    = "imswapaxis2D",
                                     .cmdkey      = "imswapaxis2D",
                                     .description = "swap axis of a 2D image",
                                     .description_long =
                                         "Transpose a 2D image by swapping its x and y axes. "
                                         "Equivalent to a matrix transpose operation." };

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    image_basic_SwapAxis2D(p_in, p_out);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_basic__imswapaxis2D()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

/**
 * Swap axes of a 2D image (transpose).
 */
imageID image_basic_SwapAxis2D_byID(imageID IDin, const char *__restrict IDout_name)
{
    if (dcimg[IDin].md[0].naxis != 2)
    {
        printf("ERROR: image needs "
               "to have 2 axis\n");
        return -1;
    }

    uint32_t xsize = dcimg[IDin].md[0].size[0];
    uint32_t ysize = dcimg[IDin].md[0].size[1];

    IMGID imgout       = imgid_make_from_name_2D(IDout_name, ysize, xsize);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (uint32_t ii = 0; ii < xsize; ii++)
    {
        for (uint32_t jj = 0; jj < ysize; jj++)
        {
            imgout.im->array.F[ii * ysize + jj] = dcimg[IDin].array.F[jj * xsize + ii];
        }
    }

    return imgout.ID;
}

imageID image_basic_SwapAxis2D(const char *__restrict IDin_name, const char *__restrict IDout_name)
{
    IMGID imgin = imgid_make_from_name(IDin_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    return image_basic_SwapAxis2D_byID(imgin.ID, IDout_name);
}
