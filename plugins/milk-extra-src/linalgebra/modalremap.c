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

    list_image_ID();

    printf("Reconstruct %s %s -> %s\n", imgU1.name, imgC0.name, imgM1->name);
    fflush(stdout);
    // Project to output space
    computeSGEMM(imgU1, imgC0, imgM1, 0, 0, GPUdev);

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