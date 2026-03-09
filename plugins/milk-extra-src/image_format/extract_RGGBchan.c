/** @file extract_RGGBchan.c
 */

#include "CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "extractRGGBchan",
    .cmdkey      = "extractRGGBchan",
    .description = "extract RGGB channels from color image"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * inim = NULL;
static char * outimR = NULL;
static char * outimG1 = NULL;
static char * outimG2 = NULL;
static char * outimB = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".inim", &inim, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input RGGB image") \
    X(".outimR", &outimR, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output R image") \
    X(".outimG1", &outimG1, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output G1 image") \
    X(".outimG2", &outimG2, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output G2 image") \
    X(".outimB", &outimB, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output B image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
/*
    IMGID imgoutR,
    IMGID imgoutG1,
    IMGID imgoutG2,
    IMGID imgoutB
*/

//
// separates a single RGB image into its 4 channels
// output written in im_r, im_g1, im_g2 and im_b
//
errno_t image_format_extract_RGGBchan(
    IMGID imgin, IMGID imgoutR, IMGID imgoutG1, IMGID imgoutG2, IMGID imgoutB)
{
    DEBUG_TRACE_FSTART();

    // input image is required
    resolveIMGID(&imgin, ERRMODE_ABORT, dcimg, dcnimg);


    imgid_copy(&imgin, &imgoutR);
    imgoutR.mdt->size[0] = imgin.md->size[0] / 2;
    imgoutR.mdt->size[1] = imgin.md->size[1] / 2;

    imgid_copy(&imgoutR, &imgoutG1);
    imgid_copy(&imgoutR, &imgoutG2);
    imgid_copy(&imgoutR, &imgoutB);

    createimagefromIMGID(&imgoutR);
    createimagefromIMGID(&imgoutG1);
    createimagefromIMGID(&imgoutG2);
    createimagefromIMGID(&imgoutB);

    uint32_t xsize = imgin.md->size[0];

    list_image_ID();


    switch(imgin.md->datatype)
    {

        case _DATATYPE_FLOAT:
            for(uint32_t ii = 0; ii < imgoutR.mdt->size[0]; ii++)
                for(uint32_t jj = 0; jj < imgoutR.mdt->size[1]; jj++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * imgoutR.mdt->size[0] + ii;

                    imgoutR.im->array.F[pixi] =
                        imgin.im->array.F[(jj1 + 1) * xsize + ii1];
                    imgoutG1.im->array.F[pixi] =
                        imgin.im->array.F[jj1 * xsize + ii1];
                    imgoutG2.im->array.F[pixi] =
                        imgin.im->array.F[(jj1 + 1) * xsize + (ii1 + 1)];
                    imgoutB.im->array.F[pixi] =
                        imgin.im->array.F[jj1 * xsize + (ii1 + 1)];
                }
            break;

        case _DATATYPE_DOUBLE:
            for(uint32_t ii = 0; ii < imgoutR.mdt->size[0]; ii++)
                for(uint32_t jj = 0; jj < imgoutR.mdt->size[1]; jj++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * imgoutR.mdt->size[0] + ii;

                    imgoutR.im->array.D[pixi] =
                        imgin.im->array.D[(jj1 + 1) * xsize + ii1];
                    imgoutG1.im->array.D[pixi] =
                        imgin.im->array.D[jj1 * xsize + ii1];
                    imgoutG2.im->array.D[pixi] =
                        imgin.im->array.D[(jj1 + 1) * xsize + (ii1 + 1)];
                    imgoutB.im->array.D[pixi] =
                        imgin.im->array.D[jj1 * xsize + (ii1 + 1)];
                }
            break;


        case _DATATYPE_UINT16:
            for(uint32_t ii = 0; ii < imgoutR.mdt->size[0]; ii++)
                for(uint32_t jj = 0; jj < imgoutR.mdt->size[1]; jj++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * imgoutR.mdt->size[0] + ii;

                    imgoutR.im->array.UI16[pixi] =
                        imgin.im->array.UI16[(jj1 + 1) * xsize + ii1];
                    imgoutG1.im->array.UI16[pixi] =
                        imgin.im->array.UI16[jj1 * xsize + ii1];
                    imgoutG2.im->array.UI16[pixi] =
                        imgin.im->array.UI16[(jj1 + 1) * xsize + (ii1 + 1)];
                    imgoutB.im->array.UI16[pixi] =
                        imgin.im->array.UI16[jj1 * xsize + (ii1 + 1)];
                }
            break;
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Wrapper function, used by all CLI calls
 *
 * INSERT_STD_PROCINFO statements enable processinfo support
 */
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    IMGID img_in = imgid_make_from_name(inim);
    IMGID img_outR = imgid_make_from_name(outimR);
    IMGID img_outG1 = imgid_make_from_name(outimG1);
    IMGID img_outG2 = imgid_make_from_name(outimG2);
    IMGID img_outB = imgid_make_from_name(outimB);

    image_format_extract_RGGBchan(img_in, img_outR, img_outG1, img_outG2, img_outB);

    imgid_free(&img_in);
    imgid_free(&img_outR);
    imgid_free(&img_outG1);
    imgid_free(&img_outG2);
    imgid_free(&img_outB);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_image_format__extractRGGBchan()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif

