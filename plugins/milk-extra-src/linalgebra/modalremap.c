#include "ImageStreamIO/ImageStruct.h"
/**
 * @file ModalRemap.c
 *
 * @brief Use mapping between two spaces to remap input
 *
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "SGEMM.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "Mremap",
    .cmdkey      = "Mremap",
    .description = "use modal mapping for linear transformation",
    .description_long =
        "Apply a linear transformation defined by a modal mapping matrix. Remaps coefficients from one modal basis to another."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * inM = NULL;
static char * inU0 = NULL;
static char * inU1 = NULL;
static char * outM = NULL;
static int32_t * GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".inM", &inM, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".inU0", &inU0, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input space mode") \
    X(".inU1", &inU1, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output space mode") \
    X(".outM", &outM, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output M")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

/**
 * @brief Remap input M0 in space U0 to output M1 in space U1
 *
 * U0 and U1 are each an orthonormal modal basis defining respectively input and output spaces
 * M0 is projected onto space U0
 * The coefficients of this decomposition are used to reconstruct M1 by expansion according to U1
 *
 * If image imsig exists, it is used to evaluate output reconstruction quality, by comparing M1 to imsig
 *
 * @param imgM0 input data
 * @param imgU0 input modal basis
 * @param imgU1 output modal basis
 * @param imgM1 output data
 * @param GPUdev GPU device
 * @return errno_t
 */
errno_t ModalRemap(
    IMGID    imgM0,
    IMGID    imgU0,
    IMGID    imgU1,
    IMGID    *imgM1,
    int      GPUdev
)
{
    DEBUG_TRACE_FSTART();

    list_image_ID();

    IMGID imgC0  = imgid_make_from_name("coeffM0");
    printf("Decompose %s %s -> %s\n", imgU0.name, imgM0.name, imgC0.name);
    fflush(stdout);
    // Decompose inM according to U0
    computeSGEMM(imgU0, imgM0, &imgC0, 1, 0, GPUdev);


    printf("Reconstruct %s %s -> %s\n", imgU1.name, imgC0.name, imgM1->name);
    fflush(stdout);
    // Project to output space
    computeSGEMM(imgU1, imgC0, imgM1, 0, 0, GPUdev);


    // evaluate fit quality
    {
        IMGID imgM1comp = imgid_make_from_name("imsig");
        resolveIMGID(&imgM1comp, ERRMODE_NULL, dcimg, dcnimg);

        FILE *fp = fopen("modalremap.log", "w");
        fprintf(fp, "# col1   frame index\n");
        fprintf(fp, "# col2   input space residual (part of input M0 that cannot be represented by U0)\n");
        fprintf(fp, "# col3   output space residual (part of ouput M1 that differs from imsig)\n");
        fprintf(fp, "# col4   decomposition vector norm 2\n");
        fprintf(fp, "# col5   decomposition vector norm 4\n");


        // Expand back to original space
        IMGID imgM0m  = imgid_make_from_name("imM0m");
        computeSGEMM(imgU0, imgC0, &imgM0m, 0, 0, GPUdev);

        // compute residual for each frame, and total
        double res0_total = 0.0;
        double res1_total = 0.0;

        uint64_t NBframe = imgM0.md->size[imgM0.md->naxis-1];
        uint64_t framesize0 = imgM0.md->nelement / NBframe;
        uint64_t framesize1 = imgM1->md->nelement / NBframe;

        double * __restrict res0array = (double*) malloc(sizeof(double)*NBframe);
        double * __restrict res1array = (double*) malloc(sizeof(double)*NBframe);

        for( uint_fast32_t frame = 0; frame < NBframe; frame ++ )
        {
            double flux0 = 0.0;
            double flux1 = 0.0;

            double res0_frame = 0.0;
            for( uint64_t ii = 0; ii < framesize0; ii++ )
            {
                float v0 = imgM0.im->array.F[frame*framesize0 + ii];
                float v1 = imgM0m.im->array.F[frame*framesize0 + ii];
                flux0 += v0;
                double vd = (v0-v1);
                res0_frame += vd*vd;
            }

            double res1_frame = 0.0;
            if(imgM1comp.ID != -1)
            {
                for( uint64_t ii = 0; ii < framesize1; ii++ )
                {
                    float v0 = imgM1->im->array.F[frame*framesize1 + ii];
                    float v1 = imgM1comp.im->array.F[frame*framesize1 + ii];
                    flux1 += v0;
                    double vd = (v0-v1);
                    res1_frame += vd*vd;
                }
            }
            else
            {
                flux1 = 1.0;
            }

            double vecC0n2 = 0.0;
            double vecC0n4 = 0.0;
            for( uint64_t ii = 0; ii < imgC0.md->size[0]; ii++ )
            {
                double vecval = imgC0.im->array.F[imgC0.md->size[0]*frame + ii];
                double vecval2 = vecval*vecval;
                double vecval4 = vecval2*vecval2;
                vecC0n2 += vecval2;
                vecC0n4 += vecval4;
            }

            // flux-normalize residuals
            //
            res0_frame /= (flux0*flux0);
            res1_frame /= (flux1*flux1);


            fprintf(fp, "%5ld %20g %20g  %20g %20g\n", frame, res0_frame, res1_frame, vecC0n2, vecC0n4);

            res0array[frame] = res0_frame;
            res1array[frame] = res1_frame;

            res0_total += res0_frame;
            res1_total += res1_frame;
        }
        double res0_average = res0_total / NBframe;
        double res1_average = res1_total / NBframe;

        quick_sort_double(res0array, NBframe);
        quick_sort_double(res1array, NBframe);

        fprintf(fp, "# AVERAGE  %20g  %20g   MEDIAN  %20g  %20g\n",
                res0_average, res1_average, res0array[NBframe/2], res1array[NBframe/2]);

        free(res0array);
        free(res1array);

        fclose(fp);
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM0 = imgid_make_from_name(inM);
    resolveIMGID(&imginM0, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginU0 = imgid_make_from_name(inU0);
    if (imginM0.ID == -1) {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginU0, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginU1 = imgid_make_from_name(inU1);
    if (imginU0.ID == -1) {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginU1, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgoutM1  = imgid_make_from_name(outM);
    if (imginU1.ID == -1) {
        return RETURN_FAILURE;
    }


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        ModalRemap(imginM0, imginU0, imginU1, &imgoutM1, *GPUdevice);
        processinfo_update_output_stream(processinfo, imgoutM1.im, NULL);

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
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_linalgebra__ModalRemap()
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

