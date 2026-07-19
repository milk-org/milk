/**
 * @file GramSchmidt.c
 * @brief Gramschmidt module
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_tools/COREMOD_tools.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "GramSchmidt",
    .cmdkey           = "GramSchmidt",
    .description      = "Gram-Schmidt process",
    .description_long = "Orthogonalize a set of vectors using the Gram-Schmidt process. Produces "
                        "an orthonormal basis from the input set."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    *inmodes   = NULL;
static char    *outmodes  = NULL;
static char    *auxmat    = NULL;
static int32_t *GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".inmodes", &inmodes, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input modes") \
    X(".outmodes", &outmodes, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output modes")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

errno_t GramSchmidt(IMGID imginm, IMGID *imgoutm, IMGID imgaux, int GPUdev __attribute__((unused)))
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&imginm, ERRMODE_WARN, dcimg, dcnimg);

    resolveIMGID(&imgaux, ERRMODE_WARN, dcimg, dcnimg);
    if (imginm.ID == -1)
    {
        return RETURN_FAILURE;
    }


    // Compute cross product on input
    //
    //IMGID imginxp  = imgid_make_from_name("_outxp");
    //computeSGEMM(imginm, imginm, &imginxp, 1, 0, GPUdev);


    // Create output
    //
    imcreatelikewiseIMGID(imgoutm, &imginm);

    uint32_t zsize;
    uint32_t xysize = imginm.md->size[0];
    if (imginm.md->naxis == 3)
    {
        zsize = imginm.md->size[2];
        xysize *= imginm.md->size[1];
    }
    else
    {
        zsize = imginm.md->size[1];
    }

    uint32_t xysizeaux = 0;
    if (imgaux.ID != -1)
    {
        xysizeaux = imgaux.md->size[0];
        if (imginm.md->naxis == 3)
        {
            xysizeaux *= imgaux.md->size[1];
        }
    }


    printf("xysize = %u, zsize = %u\n", xysize, zsize);

    printf("\n");
    for (uint32_t kk = 0; kk < zsize; kk++)
    {
        printf("\rGS mode %6u / %6u     ", kk, zsize);

        // initialization
        memcpy(&imgoutm->im->array.F[kk * xysize], &imginm.im->array.F[kk * xysize],
               sizeof(float) * xysize);

        for (uint32_t kk0 = 0; kk0 < kk; kk0++)
        {
            // cross-product
            double xpval = 0.0;

            // square sum v0
            double sqrsum0 = 0.0;

            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                float v0 = imgoutm->im->array.F[kk0 * xysize + ii];
                float v1 = imgoutm->im->array.F[kk * xysize + ii];

                xpval += v0 * v1;
                sqrsum0 += v0 * v0;
            }

            float vcoeff = xpval / sqrsum0;

            //printf("  %5u  %5u   %f\n", kk, kk0, vcoeff);

            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                imgoutm->im->array.F[kk * xysize + ii] -=
                    vcoeff * imgoutm->im->array.F[kk0 * xysize + ii];
            }

            if (imgaux.ID != -1)
            {
                for (uint32_t ii = 0; ii < xysizeaux; ii++)
                {
                    imgaux.im->array.F[kk * xysizeaux + ii] -=
                        vcoeff * imgaux.im->array.F[kk0 * xysizeaux + ii];
                }
            }
        }
    }
    printf("\n");


    list_image_ID();

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginm = imgid_make_from_name(inmodes);
    resolveIMGID(&imginm, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgoutm = imgid_make_from_name(outmodes);
    if (imginm.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgaux = imgid_make_from_name(auxmat);
    resolveIMGID(&imgaux, ERRMODE_WARN, dcimg, dcnimg);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        GramSchmidt(imginm, &imgoutm, imgaux, *GPUdevice);
    }
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
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linalgebra__GramSchmidt()
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
