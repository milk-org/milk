// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_cropmask.c
 * @brief Image cropmask module
 */

#include "ImageStreamIO/ImageStruct.h"
#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "cropmask",
    .cmdkey           = "cropmask",
    .description      = "crop and mask image",
    .description_long = "Extract a sub-region from an image and apply a binary mask. Pixels "
                        "outside the mask are set to zero. Combines cropping and masking in a "
                        "single operation for efficient region-of-interest extraction."
};

static char     cminsname[FUNCTION_PARAMETER_STRMAXLEN];
static char     masksname[FUNCTION_PARAMETER_STRMAXLEN];
static char     outsname[FUNCTION_PARAMETER_STRMAXLEN];
static uint32_t cropxstart = 0;
static uint32_t cropxsize  = 64;
static uint32_t cropystart = 0;
static uint32_t cropysize  = 64;

#define FPS_PARAMS(X)                                                                           \
    X(".insname", cminsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream name")   \
    X(".masksname", masksname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "mask stream name")  \
    X(".outsname", outsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output stream name")  \
    X(".cropxstart", &cropxstart, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop x coord start") \
    X(".cropxsize", &cropxsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop x coord size")    \
    X(".cropystart", &cropystart, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop y coord start") \
    X(".cropysize", &cropysize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "crop y coord size")

static MILK_COLD errno_t __attribute__((unused)) customCONFsetup()
{
    if (dcfpsptr != NULL)
    {
        long fpi = functionparameter_GetParamIndex(dcfpsptr, ".insname");
        if (fpi >= 0)
        {
            dcfpsptr->parray[fpi].fpflag |= FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
        }
    }
    return RETURN_SUCCESS;
}

static MILK_COLD errno_t __attribute__((unused)) customCONFcheck()
{
    return RETURN_SUCCESS;
}

FPS_V2_SECTION5(FPS_PARAMS)


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    // CONNECT TO INPUT STREAM
    IMGID imgin = imgid_make_from_name(cminsname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    printf("Input stream size : %u %u\n", imgin.md->size[0], imgin.md->size[1]);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }
    //long m = imgin.md->size[0] * imgin.md->size[1];

    // CONNNECT TO OR CREATE MASK STREAM
    IMGID imgmask = stream_connect_create_2Df32(masksname, cropxsize, cropysize);

    // CONNNECT TO OR CREATE OUTPUT STREAM
    IMGID imgout = stream_connect_create_2Df32(outsname, cropxsize, cropysize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        const uint32_t in_xsize    = imgin.md->size[0];
        const uint32_t crop_xsize  = cropxsize;
        const uint32_t crop_ysize  = cropysize;
        const uint32_t crop_ystart = cropystart;
        const uint32_t crop_xstart = cropxstart;

        for (uint32_t jj = 0; jj < crop_ysize; jj++)
        {
            const float *__restrict imgin_row =
                &imgin.im->array.F[(jj + crop_ystart) * in_xsize + crop_xstart];
            const float *__restrict imgmask_row = &imgmask.im->array.F[jj * crop_xsize];
            float *__restrict imgout_row        = &imgout.im->array.F[jj * crop_xsize];

            MILK_IVDEP
            for (uint32_t ii = 0; ii < crop_xsize; ii++)
            {
                imgout_row[ii] = imgmask_row[ii] * imgin_row[ii];
            }
        }
        processinfo_update_output_stream(processinfo, imgout.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

// Register function in CLI
errno_t CLIADDCMD_COREMODE_arith__cropmask()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(FPS_app_info, FPS_PARAMS, compute_function, customCONFcheck)
#endif
