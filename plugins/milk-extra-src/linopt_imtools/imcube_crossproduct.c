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

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imcubeXprod",
    .cmdkey           = "imcubeXprod",
    .description      = "cross product of two image cubes",
    .description_long = "Compute the cross-product matrix of two image cubes. Each element (i,j) "
                        "is the dot product of slice i from cube A with slice j from cube B."
};


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

FPS_V2_SECTION5(FPS_PARAMS)

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

    // compute mask sum
    double masksum = 0.0;
    for (uint64_t pixi = 0; pixi < xysize; pixi++)
    {
        masksum += imgmask.im->array.F[pixi];
    }

    for (uint32_t kk0 = 0; kk0 < zsize0; kk0++)
    {
        for (uint32_t kk1 = kk0; kk1 < zsize1; kk1++)
        {
            double   tmpv     = 0.0;
            uint64_t z0offset = xysize * kk0;
            uint64_t z1offset = xysize * kk1;
            for (uint64_t pixi = 0; pixi < xysize; pixi++)
            {
                tmpv += imgmask.im->array.F[pixi] * (imgcube0.im->array.F[z0offset + pixi] *
                                                     imgcube1.im->array.F[z1offset + pixi]);
            }
            imgout.im->array.F[kk1 * zsize0 + kk0] = tmpv / masksum;
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

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    // connect to input mode values array and get number of modes
    //
    IMGID imginc0 = imgid_make_from_name(inimc0);
    IMGID imginc1 = imgid_make_from_name(inimc1);
    IMGID imgmask = imgid_make_from_name(inimmask);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    imcube_crossproduct(imginc0, imginc1, imgmask, imout);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imginc0);
    imgid_free(&imginc1);
    imgid_free(&imgmask);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linopt_imtools__imcube_crossproduct()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
