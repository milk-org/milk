#include "ImageStreamIO/ImageStruct.h"
/**
 * @file PCAmatch.c
 *
 * @brief match two PCA decompositions
 *
 * Find corresponding linear combination across two basis
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
    .fps_name    = "PCAmatch",
    .cmdkey      = "PCAmatch",
    .description = "find matching linear combination across two modal bases",
    .description_long =
        "Find the best matching linear combination between two modal bases using Principal Component Analysis cross-matching."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * modesA = NULL;
static char * modesB = NULL;
static char * outcoeffA = NULL;
static char * outcoeffB = NULL;
static char * outimA = NULL;
static char * outimB = NULL;
static int32_t * GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".modesA", &modesA, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input modes A") \
    X(".modesB", &modesB, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input modes B") \
    X(".outcoeffA", &outcoeffA, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output coeffs A") \
    X(".outcoeffB", &outcoeffB, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output coeffs B") \
    X(".outimA", &outimA, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image A") \
    X(".outimB", &outimB, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image B")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

/**
 * @brief Find matching linear combinations across two bases
 *
 * @param imgmodesA  basis A
 * @param imgmodesB  basis B
 * @param imgoutcA   output coeffs A
 * @param imgoutcB   output coeffs B
 * @param imgoutimA  output image A
 * @param imgoutimB  output image B
 * @param GPUdev     GPU device
 * @return errno_t
 */
errno_t PCAmatch(
    IMGID    imgmodesA,
    IMGID    imgmodesB,
    IMGID    *imgoutcA,
    IMGID    *imgoutcB,
    IMGID    *imgoutimA,
    IMGID    *imgoutimB,
    int      GPUdev
)
{
    DEBUG_TRACE_FSTART();

    uint32_t NBmodesA = imgmodesA.md->size[2];
    uint32_t NBmodesB = imgmodesA.md->size[2];

    printf("NBmodesA = %u\n", NBmodesA);
    printf("NBmodesB = %u\n", NBmodesB);
    fflush(stdout);

    // output vectors

    imgoutcA->mdt->datatype = _DATATYPE_FLOAT;
    imgoutcA->mdt->naxis = 2;
    imgoutcA->mdt->size[0] = NBmodesA;
    imgoutcA->mdt->size[1] = 1;
    printf("CREATING %s\n", imgoutcA->name);
    fflush(stdout);
    createimagefromIMGID(imgoutcA);


    imgoutcB->mdt->datatype = _DATATYPE_FLOAT;
    imgoutcB->mdt->naxis = 2;
    imgoutcB->mdt->size[0] = NBmodesA;
    imgoutcB->mdt->size[1] = 1;
    printf("CREATING %s\n", imgoutcB->name);
    fflush(stdout);
    createimagefromIMGID(imgoutcB);


    // A->B coeff remapping matrix
    IMGID imgAtoB;
    snprintf(imgAtoB.name,
             sizeof(imgAtoB.name),
             "%s", "matAtoB");
    computeSGEMM(
        imgmodesB,
        imgmodesA,
        &imgAtoB,
        1, 0,
        GPUdev
    );

    // B->A coeff remapping matrix
    IMGID imgBtoA;
    snprintf(imgBtoA.name,
             sizeof(imgBtoA.name),
             "%s", "matBtoA");
    computeSGEMM(
        imgmodesA,
        imgmodesB,
        &imgBtoA,
        1, 0,
        GPUdev
    );


    // Initialization
    imgoutcA->im->array.F[0] = 1.0;
    for(uint32_t mode=1; mode < NBmodesA; mode++)
    {
        imgoutcA->im->array.F[mode] = 0.0;
    }

    imgoutcB->im->array.F[0] = 1.0;
    for(uint32_t mode=1; mode < NBmodesB; mode++)
    {
        imgoutcB->im->array.F[mode] = 0.0;
    }


    // residual0
    IMGID imgimres0  = imgid_make_from_name("imres0");
    imgimres0.mdt->naxis   = 2;
    imgimres0.mdt->size[0] = imgmodesA.md->size[0];
    imgimres0.mdt->size[1] = imgmodesA.md->size[1];
    createimagefromIMGID(&imgimres0);

    double resim0 = 0.0;
    for(uint64_t ii=0; ii< imgmodesA.md->size[0]*imgmodesA.md->size[1]; ii++)
    {
        double vA = imgmodesA.im->array.F[ii];
        double vB = imgmodesB.im->array.F[ii];
        double vdiff = vA-vB;
        resim0 += vdiff*vdiff;
        imgimres0.im->array.F[ii] =  vdiff;
    }


    // project to B
    computeSGEMM(
        imgAtoB,
        *imgoutcA,
        imgoutcB,
        0, 0,
        GPUdev
    );


    int NBiter = 1000;
    for(int iter=0; iter<NBiter; iter++)
    {
        // project to A
        computeSGEMM(
            imgBtoA,
            *imgoutcB,
            imgoutcA,
            0, 0,
            GPUdev
        );

        // attenuate non-average terms
        //imgoutcA->im->array.F[0] = 1.0;
        for(uint32_t mode=1; mode < NBmodesA; mode++)
        {
            imgoutcA->im->array.F[mode] *= 0.999;
        }

        // normalize vector A
        {
            double norm = 0.0;
            for(uint32_t mode=0; mode < NBmodesA; mode++)
            {
                double val = imgoutcA->im->array.F[mode];
                norm += val*val;
            }
            norm = sqrt(norm);
            printf("   A norm = %f\n", norm);

            for(uint32_t mode=0; mode < NBmodesA; mode++)
            {
                imgoutcA->im->array.F[mode] /= norm;
            }
        }


        // project to B
        computeSGEMM(
            imgAtoB,
            *imgoutcA,
            imgoutcB,
            0, 0,
            GPUdev
        );


        printf("[%5d] coeffs A :  ", iter);
        for(uint32_t mode=0; mode < NBmodesA; mode++)
        {
            if(mode < 16)
            {
                printf("%+8.6f  ", imgoutcA->im->array.F[mode]);
            }
        }
        printf("\n");

        printf("[%5d] coeffs B :  ", iter);
        for(uint32_t mode=0; mode < NBmodesB; mode++)
        {
            if(mode < 16)
            {
                printf("%+8.6f  ", imgoutcB->im->array.F[mode]);
            }
        }
        printf("\n");

        printf("\n");

    }


    // compute output images
    computeSGEMM(
        imgmodesA,
        *imgoutcA,
        imgoutimA,
        0, 0,
        GPUdev
    );

    computeSGEMM(
        imgmodesB,
        *imgoutcB,
        imgoutimB,
        0, 0,
        GPUdev
    );


    IMGID imgimres  = imgid_make_from_name("imres");
    imgimres.mdt->naxis   = 2;
    imgimres.mdt->size[0] = imgmodesA.md->size[0];
    imgimres.mdt->size[1] = imgmodesA.md->size[1];
    createimagefromIMGID(&imgimres);

    double resim = 0.0;
    for(uint64_t ii=0; ii< imgmodesA.md->size[0]*imgmodesA.md->size[1]; ii++)
    {
        double vA = imgoutimA->im->array.F[ii];
        double vB = imgoutimB->im->array.F[ii];
        double vdiff = vA-vB;
        resim += vdiff*vdiff;
        imgimres.im->array.F[ii] =  vdiff;
    }

    printf("RESIDUAL %g -> %g\n", resim0, resim);
    printf("GAIN = %f\n", resim0/resim);
    printf("\n");

    imgid_free(&imgAtoB);
    imgid_free(&imgBtoA);
    imgid_free(&imgimres0);
    imgid_free(&imgimres);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgmodesA = imgid_make_from_name(modesA);
    resolveIMGID(&imgmodesA, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imgmodesB = imgid_make_from_name(modesB);
    if (imgmodesA.ID == -1) {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imgmodesB, ERRMODE_WARN, dcimg, dcnimg);

    printf("Modes images IDs : %ld %ld\n", imgmodesA.ID, imgmodesB.ID);
    if (imgmodesB.ID == -1) {
        return RETURN_FAILURE;
    }
    fflush(stdout);


    printf("outcoeffA = %s\n", outcoeffA);
    fflush(stdout);
    IMGID imgoutcA  = imgid_make_from_name(outcoeffA);

    printf("outcoeffB = %s\n", outcoeffB);
    fflush(stdout);
    IMGID imgoutcB  = imgid_make_from_name(outcoeffB);


    printf("imgoutimA = %s\n", outimA);
    fflush(stdout);
    IMGID imgoutimA  = imgid_make_from_name(outimA);

    printf("imgoutimB = %s\n", outimB);
    fflush(stdout);
    IMGID imgoutimB  = imgid_make_from_name(outimB);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        PCAmatch(
            imgmodesA,
            imgmodesB,
            &imgoutcA,
            &imgoutcB,
            &imgoutimA,
            &imgoutimB,
            *GPUdevice
        );

        processinfo_update_output_stream(processinfo, imgoutcA.im, NULL);
        processinfo_update_output_stream(processinfo, imgoutcB.im, NULL);
        //processinfo_update_output_stream(processinfo, imgoutimA.im, NULL);
        //processinfo_update_output_stream(processinfo, imgoutimB.im, NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgmodesA);
    imgid_free(&imgmodesB);
    imgid_free(&imgoutcA);
    imgid_free(&imgoutcB);
    imgid_free(&imgoutimA);
    imgid_free(&imgoutimB);

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
CLIADDCMD_linalgebra__PCAmatch()
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

