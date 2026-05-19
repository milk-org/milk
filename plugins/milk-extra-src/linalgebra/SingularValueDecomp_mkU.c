#include "ImageStreamIO/ImageStruct.h"
/**
 * @file SingularValueDecomp_mkU.c
 *
 * @brief make U from M, V and S
 *
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "SGEMM.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "SVDmkU",
    .cmdkey      = "SVDmkU",
    .description = "compute SVD U",
    .description_long =
        "Compute only the U matrix of the SVD decomposition for large matrices where the full SVD is memory-prohibitive."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * inmatM = NULL;
static char * inmatV = NULL;
static char * invecS = NULL;
static char * outmatU = NULL;
static char * outmatUS = NULL;
static int32_t * GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".inM", &inmatM, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input matrix M") \
    X(".inV", &inmatV, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input matrix V") \
    X(".inS", &invecS, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input singular values vec")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

errno_t compute_SVDU(
    IMGID    imgM,
    IMGID    imgV,
    IMGID    imgS,
    IMGID    *imgU,
    IMGID    *imgUS,
    int      GPUdev
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&imgM, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(&imgV, ERRMODE_WARN, dcimg, dcnimg);
    if (imgM.ID == -1) {
        return RETURN_FAILURE;
    }
    if (imgV.ID == -1) {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imgS, ERRMODE_WARN, dcimg, dcnimg);

    computeSGEMM(
        imgM,
        imgV,
        imgUS,
        0,
        0,
        GPUdev
    );
    if (imgS.ID == -1) {
        return RETURN_FAILURE;
    }

    printf("SGEMM DONE\n");
    fflush(stdout);
    list_image_ID();

    //uint32_t Ndim = imgV.md->size[imgV.md->naxis-1];

    uint64_t framesize;
    uint32_t nbframe;
    imgU->mdt->naxis = imgUS->md->naxis;
    imgU->mdt->datatype = imgUS->md->datatype;
    switch(imgUS->md->naxis)
    {
    case 2 :
        imgU->mdt->size[0] = imgUS->md->size[0];
        imgU->mdt->size[1] = imgUS->md->size[1];
        framesize = imgUS->md->size[0];
        nbframe = imgUS->md->size[1];
        break;

    case 3 :
        imgU->mdt->size[0] = imgUS->md->size[0];
        imgU->mdt->size[1] = imgUS->md->size[1];
        imgU->mdt->size[2] = imgUS->md->size[2];
        framesize = imgUS->md->size[0] * imgUS->md->size[1];
        nbframe = imgUS->md->size[2];
        break;

    default :
        PRINT_ERROR("Invalid dimension");
        abort();
    }
    printf("CREATING imgU\n");
    fflush(stdout);
    createimagefromIMGID(imgU);

    list_image_ID();

    for(uint32_t frame = 0; frame < nbframe; frame++)
    {
        for(uint64_t ii = 0; ii < framesize; ii++)
        {
            imgU->im->array.F[frame * framesize + ii] =  imgUS->im->array.F[frame *
                    framesize + ii] / imgS.im->array.F[frame];
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM = imgid_make_from_name(inmatM);
    resolveIMGID(&imginM, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginV = imgid_make_from_name(inmatV);
    if (imginM.ID == -1) {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginV, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginS = imgid_make_from_name(invecS);
    if (imginV.ID == -1) {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginS, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgoutU  = imgid_make_from_name(outmatU);
    if (imginS.ID == -1) {
        return RETURN_FAILURE;
    }
    IMGID imgoutUS  = imgid_make_from_name(outmatUS);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        compute_SVDU(imginM, imginV, imginS, &imgoutU, &imgoutUS, *GPUdevice);
        processinfo_update_output_stream(processinfo, imgoutU.im, NULL);
        processinfo_update_output_stream(processinfo, imgoutUS.im, NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imginM);
    imgid_free(&imginV);
    imgid_free(&imginS);
    imgid_free(&imgoutU);
    imgid_free(&imgoutUS);

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
CLIADDCMD_linalgebra__compSVDU()
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

