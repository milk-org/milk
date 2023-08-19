/**
 * @file ModalRemap.c
 *
 * @brief Use mapping between two spaces to remap input
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "SGEMM.h"



static char *inM;
static long  fpi_inM;

static char *inU0;
static long  fpi_inU0;

static char *inU1;
static long  fpi_inU1;

static char *outM;
static long  fpi_outM;


static int32_t *GPUdevice;
static long     fpi_GPUdevice;








static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inM",
        "input image",
        "inM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inM,
        &fpi_inM
    },
    {
        CLIARG_IMG,
        ".inU0",
        "input space mode",
        "inU0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inU0,
        &fpi_inU0
    },
    {
        CLIARG_IMG,
        ".inU1",
        "output space mode",
        "inU1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inU1,
        &fpi_inU1
    },
    {
        CLIARG_STR,
        ".outM",
        "output M",
        "outM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outM,
        &fpi_outM
    },
    {
        // using GPU (99 : no GPU, otherwise GPU device)
        CLIARG_INT32,
        ".GPUdevice",
        "GPU device, 99 for CPU",
        "-1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUdevice,
        &fpi_GPUdevice
    }
};





static CLICMDDATA CLIcmddata =
{
    "Mremap", "use modal mapping for linear transformation", CLICMD_FIELDS_DEFAULTS
};



static errno_t help_function()
{
    printf("Use modal mapping for transformation\n");
    printf("Modal mapping is between input basis U0 and output basis U1\n");
    printf("First, decompose input M0 as coefficients of basis U0\n");
    printf("These coefficients are then re-expanded according to basis U1\n");

    return RETURN_SUCCESS;
}




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

    IMGID imgC0  = mkIMGID_from_name("coeffM0");
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
        IMGID imgM1comp = mkIMGID_from_name("imsig");
        resolveIMGID(&imgM1comp, ERRMODE_NULL);

        FILE *fp = fopen("modalremap.log", "w");
        fprintf(fp, "# col1   frame index\n");
        fprintf(fp, "# col2   input space residual (part of input M0 that cannot be represented by U0)\n");
        fprintf(fp, "# col3   output space residual (part of ouput M1 that differs from imsig)\n");
        fprintf(fp, "# col4   decomposition vector norm 2\n");
        fprintf(fp, "# col5   decomposition vector norm 4\n");


        // Expand back to original space
        IMGID imgM0m  = mkIMGID_from_name("imM0m");
        computeSGEMM(imgU0, imgC0, &imgM0m, 0, 0, GPUdev);

        // compute residual for each frame, and total
        double res0_total = 0.0;
        double res1_total = 0.0;

        uint64_t NBframe = imgM0.md->size[imgM0.md->naxis-1];
        uint64_t framesize0 = imgM0.md->nelement / NBframe;
        uint64_t framesize1 = imgM1->md->nelement / NBframe;



        for( uint_fast32_t frame = 0; frame < NBframe; frame ++ )
        {

            double res0_frame = 0.0;
            for( uint64_t ii = 0; ii < framesize0; ii++ )
            {
                float v0 = imgM0.im->array.F[frame*framesize0 + ii];
                float v1 = imgM0m.im->array.F[frame*framesize0 + ii];
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
                    double vd = (v0-v1);
                    res1_frame += vd*vd;
                }
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


            fprintf(fp, "%5ld %20g %20g  %20g %20g\n", frame, res0_frame, res1_frame, vecC0n2, vecC0n4);
            res0_total += res0_frame;
            res1_total += res1_frame;
        }
        double res0_average = res0_total / NBframe;
        double res1_average = res1_total / NBframe;
        fprintf(fp, "# AVERAGE %5d %20g %20g\n", -1, res0_average, res1_average);

        fclose(fp);
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM0 = mkIMGID_from_name(inM);
    resolveIMGID(&imginM0, ERRMODE_ABORT);

    IMGID imginU0 = mkIMGID_from_name(inU0);
    resolveIMGID(&imginU0, ERRMODE_ABORT);

    IMGID imginU1 = mkIMGID_from_name(inU1);
    resolveIMGID(&imginU1, ERRMODE_ABORT);


    IMGID imgoutM1  = mkIMGID_from_name(outM);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        ModalRemap(imginM0, imginU0, imginU1, &imgoutM1, *GPUdevice);
        processinfo_update_output_stream(processinfo, imgoutM1.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions


errno_t CLIADDCMD_linalgebra__ModalRemap()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}