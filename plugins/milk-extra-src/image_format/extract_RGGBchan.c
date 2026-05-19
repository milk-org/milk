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
    .description = "extract RGGB channels from color image",
    .description_long =
        "Extract individual RGGB Bayer channels from a raw color camera image into separate monochrome images."
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

FPS_V2_SECTION5(FPS_PARAMS)

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
    IMGID *imgin,
    IMGID *imgoutR,
    IMGID *imgoutG1,
    IMGID *imgoutG2,
    IMGID *imgoutB)
{
    DEBUG_TRACE_FSTART();

    // input image is required
    resolveIMGID(imgin, ERRMODE_WARN, dcimg, dcnimg);

    // Create output images if not yet allocated.
    // Guards prevent reallocation on every frame.
    if(imgoutR->ID == -1)
    {
        imgid_copy(imgin, imgoutR);
    if (imgin->ID == -1) {
        return RETURN_FAILURE;
    }
        imgoutR->mdt->size[0] = imgin->md->size[0] / 2;
        imgoutR->mdt->size[1] = imgin->md->size[1] / 2;
        createimagefromIMGID(imgoutR);
    }
    if(imgoutG1->ID == -1)
    {
        imgid_copy(imgoutR, imgoutG1);
        createimagefromIMGID(imgoutG1);
    }
    if(imgoutG2->ID == -1)
    {
        imgid_copy(imgoutR, imgoutG2);
        createimagefromIMGID(imgoutG2);
    }
    if(imgoutB->ID == -1)
    {
        imgid_copy(imgoutR, imgoutB);
        createimagefromIMGID(imgoutB);
    }

    uint32_t xsize = imgin->md->size[0];

    list_image_ID();


    switch(imgin->md->datatype)
    {

        case _DATATYPE_FLOAT:
        {
            float * MILK_RESTRICT outR = MILK_ASSUME_ALIGNED(imgoutR->im->array.F);
            float * MILK_RESTRICT outG1 = MILK_ASSUME_ALIGNED(imgoutG1->im->array.F);
            float * MILK_RESTRICT outG2 = MILK_ASSUME_ALIGNED(imgoutG2->im->array.F);
            float * MILK_RESTRICT outB = MILK_ASSUME_ALIGNED(imgoutB->im->array.F);
            const float * MILK_RESTRICT in = MILK_ASSUME_ALIGNED(imgin->im->array.F);
            uint32_t size_x = imgoutR->mdt->size[0];
            uint32_t size_y = imgoutR->mdt->size[1];

            _Pragma("omp parallel for")
            for(uint32_t jj = 0; jj < size_y; jj++)
            {
                _Pragma("omp simd")
                for(uint32_t ii = 0; ii < size_x; ii++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * (uint64_t)size_x + ii;

                    outR[pixi]  = in[(jj1 + 1) * xsize + ii1];
                    outG1[pixi] = in[jj1 * xsize + ii1];
                    outG2[pixi] = in[(jj1 + 1) * xsize + ii1 + 1];
                    outB[pixi]  = in[jj1 * xsize + ii1 + 1];
                }
            }
            break;
        }

        case _DATATYPE_DOUBLE:
        {
            double * MILK_RESTRICT outR = MILK_ASSUME_ALIGNED(imgoutR->im->array.D);
            double * MILK_RESTRICT outG1 = MILK_ASSUME_ALIGNED(imgoutG1->im->array.D);
            double * MILK_RESTRICT outG2 = MILK_ASSUME_ALIGNED(imgoutG2->im->array.D);
            double * MILK_RESTRICT outB = MILK_ASSUME_ALIGNED(imgoutB->im->array.D);
            const double * MILK_RESTRICT in = MILK_ASSUME_ALIGNED(imgin->im->array.D);
            uint32_t size_x = imgoutR->mdt->size[0];
            uint32_t size_y = imgoutR->mdt->size[1];

            _Pragma("omp parallel for")
            for(uint32_t jj = 0; jj < size_y; jj++)
            {
                _Pragma("omp simd")
                for(uint32_t ii = 0; ii < size_x; ii++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * (uint64_t)size_x + ii;

                    outR[pixi]  = in[(jj1 + 1) * xsize + ii1];
                    outG1[pixi] = in[jj1 * xsize + ii1];
                    outG2[pixi] = in[(jj1 + 1) * xsize + ii1 + 1];
                    outB[pixi]  = in[jj1 * xsize + ii1 + 1];
                }
            }
            break;
        }

        case _DATATYPE_UINT16:
        {
            uint16_t * MILK_RESTRICT outR = MILK_ASSUME_ALIGNED(imgoutR->im->array.UI16);
            uint16_t * MILK_RESTRICT outG1 = MILK_ASSUME_ALIGNED(imgoutG1->im->array.UI16);
            uint16_t * MILK_RESTRICT outG2 = MILK_ASSUME_ALIGNED(imgoutG2->im->array.UI16);
            uint16_t * MILK_RESTRICT outB = MILK_ASSUME_ALIGNED(imgoutB->im->array.UI16);
            const uint16_t * MILK_RESTRICT in = MILK_ASSUME_ALIGNED(imgin->im->array.UI16);
            uint32_t size_x = imgoutR->mdt->size[0];
            uint32_t size_y = imgoutR->mdt->size[1];

            _Pragma("omp parallel for")
            for(uint32_t jj = 0; jj < size_y; jj++)
            {
                _Pragma("omp simd")
                for(uint32_t ii = 0; ii < size_x; ii++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * (uint64_t)size_x + ii;

                    outR[pixi]  = in[(jj1 + 1) * xsize + ii1];
                    outG1[pixi] = in[jj1 * xsize + ii1];
                    outG2[pixi] = in[(jj1 + 1) * xsize + ii1 + 1];
                    outB[pixi]  = in[jj1 * xsize + ii1 + 1];
                }
            }
            break;
        }
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Wrapper function, used by all CLI calls
 *
 * INSERT_STD_PROCINFO statements enable processinfo support
 */
static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // Declare IMGIDs before the loop so allocations
    // are guarded and only happen on the first frame.
    IMGID img_in   = imgid_make_from_name(inim);
    IMGID img_outR  = imgid_make_from_name(outimR);
    IMGID img_outG1 = imgid_make_from_name(outimG1);
    IMGID img_outG2 = imgid_make_from_name(outimG2);
    IMGID img_outB  = imgid_make_from_name(outimB);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        image_format_extract_RGGBchan(
            &img_in,
            &img_outR,
            &img_outG1,
            &img_outG2,
            &img_outB);

        processinfo_update_output_stream(
            processinfo, img_outR.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&img_in);
    imgid_free(&img_outR);
    imgid_free(&img_outG1);
    imgid_free(&img_outG2);
    imgid_free(&img_outB);

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

