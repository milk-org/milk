// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "ImageStreamIO/ImageStruct.h"
/**
 * @file SingularValueDecomp_mkU.c
 *
 * @brief make U from M, V and S
 *
 */

#include <math.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "linalgebra.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    *inmatM    = NULL;
static char    *inmatV    = NULL;
static char    *invecS    = NULL;
static char    *outmatU   = NULL;
static char    *outmatUS  = NULL;
static int32_t *GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                \
    X(".inM", &inmatM, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input matrix M") \
    X(".inV", &inmatV, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input matrix V") \
    X(".inS", &invecS, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input singular values vec")


/* ================================================================
 * 5.  PARAMETER BINDINGS
 * ============================================================= */

MILK_PARAMS_DEFINE(SVDmkU, FPS_PARAMS);

errno_t compute_SVDU(IMGID imgM, IMGID imgV, IMGID imgS, IMGID *imgU, IMGID *imgUS, int GPUdev)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&imgM, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(&imgV, ERRMODE_WARN, dcimg, dcnimg);
    if (imgM.ID == -1)
    {
        return RETURN_FAILURE;
    }
    if (imgV.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imgS, ERRMODE_WARN, dcimg, dcnimg);

    computeSGEMM(imgM, imgV, imgUS, 0, 0, GPUdev);
    if (imgS.ID == -1)
    {
        return RETURN_FAILURE;
    }

    printf("SGEMM DONE\n");
    fflush(stdout);
    list_image_ID();

    //uint32_t Ndim = imgV.md->size[imgV.md->naxis-1];

    uint64_t framesize;
    uint32_t nbframe;
    imgU->mdt->naxis    = imgUS->md->naxis;
    imgU->mdt->datatype = imgUS->md->datatype;
    switch (imgUS->md->naxis)
    {
    case 2:
        imgU->mdt->size[0] = imgUS->md->size[0];
        imgU->mdt->size[1] = imgUS->md->size[1];
        framesize          = imgUS->md->size[0];
        nbframe            = imgUS->md->size[1];
        break;

    case 3:
        imgU->mdt->size[0] = imgUS->md->size[0];
        imgU->mdt->size[1] = imgUS->md->size[1];
        imgU->mdt->size[2] = imgUS->md->size[2];
        framesize          = imgUS->md->size[0] * imgUS->md->size[1];
        nbframe            = imgUS->md->size[2];
        break;

    default:
        PRINT_ERROR("Invalid dimension");
        abort();
    }
    printf("CREATING imgU\n");
    fflush(stdout);
    createimagefromIMGID(imgU);

    list_image_ID();

    for (uint32_t frame = 0; frame < nbframe; frame++)
    {
        for (uint64_t ii = 0; ii < framesize; ii++)
        {
            imgU->im->array.F[frame * framesize + ii] =
                imgUS->im->array.F[frame * framesize + ii] / imgS.im->array.F[frame];
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t SVDmkU_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imginM = imgid_make_from_name(inmatM);
    resolveIMGID(&imginM, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginV = imgid_make_from_name(inmatV);
    if (imginM.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginV, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginS = imgid_make_from_name(invecS);
    if (imginV.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginS, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgoutU = imgid_make_from_name(outmatU);
    if (imginS.ID == -1)
    {
        return RETURN_FAILURE;
    }
    IMGID imgoutUS = imgid_make_from_name(outmatUS);


    MILK_PROCINFO_INIT(self)


    MILK_PROCINFO_LOOPSTART(self)
    {
        compute_SVDU(imginM, imginV, imginS, &imgoutU, &imgoutUS, *GPUdevice);
        processinfo_update_output_stream(processinfo, imgoutU.im, NULL);
        processinfo_update_output_stream(processinfo, imgoutUS.im, NULL);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&imginM);
    imgid_free(&imginV);
    imgid_free(&imginS);
    imgid_free(&imgoutU);
    imgid_free(&imgoutUS);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK_RUNNABLE DESCRIPTOR AND FPSEXEC EXPORT
 * ============================================================= */

const MILK_RUNNABLE runnable_linalgebra_SVDmkU = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "SVDmkU",
                 .cmdkey           = "SVDmkU",
                 .description      = "compute SVD U",
                 .description_long = "Compute only the U matrix of the SVD decomposition for large "
                                     "matrices where the full SVD is memory-prohibitive." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(SVDmkU),
    .compute = SVDmkU_compute,
};
EXPORT_FPSEXEC(milk, "linalg-SVDmkU", runnable_linalgebra_SVDmkU);
