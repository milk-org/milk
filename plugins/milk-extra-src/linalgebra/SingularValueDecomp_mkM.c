#include "ImageStreamIO/ImageStruct.h"
/**
 * @file SingularValueDecomp_mkU.c
 *
 * @brief make M from U, S, and V
 *
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "linalgebra.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = { .fps_name    = "SVDmkM",
                                     .cmdkey      = "SVDmkM",
                                     .description = "reconstruct SVD M",
                                     .description_long =
                                         "Reconstruct a matrix from its SVD components. Computes M "
                                         "= U * S * V^T from pre-computed U, S, V matrices." };


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    *inmatU    = NULL;
static char    *invecS    = NULL;
static char    *inmatV    = NULL;
static char    *outmatM   = NULL;
static int32_t *GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".inU", &inmatU, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input matrix U")            \
    X(".inS", &invecS, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input singular values vec") \
    X(".inV", &inmatV, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input matrix V")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

errno_t SVDmkM(IMGID imgU, IMGID imgS, IMGID imgV, IMGID *imgM, int GPUdev)
{
    DEBUG_TRACE_FSTART();

    list_image_ID();

    resolveIMGID(&imgU, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(&imgS, ERRMODE_WARN, dcimg, dcnimg);
    if (imgU.ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (imgS.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imgV, ERRMODE_WARN, dcimg, dcnimg);

    // un-normalized modes
    //printf("Creating image from %s\n", imgU.md->name);
    if (imgV.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgunmodes         = imgid_make_from_name("XXSVDunmodes");
    imgunmodes.mdt->naxis    = imgU.md->naxis;
    imgunmodes.mdt->datatype = imgU.md->datatype;
    imgunmodes.mdt->size[0]  = imgU.md->size[0];
    imgunmodes.mdt->size[1]  = imgU.md->size[1];
    imgunmodes.mdt->size[2]  = imgU.md->size[2];

    printf("Creating temp img XXSVDunmodes  %d x %d x %d\n", imgunmodes.mdt->size[0],
           imgunmodes.mdt->size[1], imgunmodes.mdt->size[2]);
    createimagefromIMGID(&imgunmodes);

    list_image_ID();

    int  lastaxis  = imgunmodes.mdt->naxis - 1;
    long framesize = imgunmodes.mdt->size[0];
    if (lastaxis == 2)
    {
        framesize *= imgunmodes.mdt->size[1];
    }

    for (int kk = 0; kk < imgunmodes.mdt->size[lastaxis]; kk++)
    {
        float mfact = imgS.im->array.F[kk];
        for (long ii = 0; ii < framesize; ii++)
        {
            imgunmodes.im->array.F[kk * framesize + ii] =
                imgU.im->array.F[kk * framesize + ii] * mfact;
        }
    }

    list_image_ID();

    computeSGEMM(imgunmodes, imgV, imgM, 0, 1, GPUdev);
    delete_image_ID(imgunmodes.name, DELETE_IMAGE_ERRMODE_WARNING);
    imgid_free(&imgunmodes);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginU = imgid_make_from_name(inmatU);
    resolveIMGID(&imginU, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginS = imgid_make_from_name(invecS);
    if (imginU.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginS, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginV = imgid_make_from_name(inmatV);
    if (imginS.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginV, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgoutM = imgid_make_from_name(outmatM);
    if (imginV.ID == -1)
    {
        return RETURN_FAILURE;
    }


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        SVDmkM(imginU, imginS, imginV, &imgoutM, *GPUdevice);
        processinfo_update_output_stream(processinfo, imgoutM.im, NULL);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imginU);
    imgid_free(&imginS);
    imgid_free(&imginV);
    imgid_free(&imgoutM);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linalgebra__SVDmkM()
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
