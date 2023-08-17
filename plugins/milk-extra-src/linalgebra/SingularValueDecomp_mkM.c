/**
 * @file SingularValueDecomp_mkU.c
 *
 * @brief make M from U, S, and V
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "SGEMM.h"





static char *inmatU;
static long  fpi_inmatU;

// singular values
static char *invecS;
static long  fpi_invecS;

static char *inmatV;
static long  fpi_inmatV;



static char *outmatM;
static long  fpi_outmatM;


static int32_t *GPUdevice;
static long     fpi_GPUdevice;





static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inU",
        "input matrix U",
        "inM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmatU,
        &fpi_inmatU
    },
    {
        CLIARG_IMG,
        ".inS",
        "input singular values vec",
        "inS",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &invecS,
        &fpi_invecS
    },
    {
        CLIARG_IMG,
        ".inV",
        "input matrix V",
        "inV",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmatV,
        &fpi_inmatV
    },
    {
        // output M
        CLIARG_STR,
        ".outM",
        "output M",
        "outM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outmatM,
        &fpi_outmatM
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
    "SVDmkM", "reconstruct SVD M", CLICMD_FIELDS_DEFAULTS
};



static errno_t help_function()
{
    printf("Compute M from SVD's U, S and V\n");

    return RETURN_SUCCESS;
}




errno_t SVDmkM(
    IMGID    imgU,
    IMGID    imgS,
    IMGID    imgV,
    IMGID    *imgM,
    int      GPUdev
)
{
    DEBUG_TRACE_FSTART();

    // un-normalized modes
    IMGID imgunmodes = mkIMGID_from_name("SVDunmodes");
    imgunmodes.naxis = imgU.md->naxis;
    imgunmodes.datatype = imgU.md->datatype;
    imgunmodes.size[0] = imgU.md->size[0];
    imgunmodes.size[1] = imgU.md->size[1];
    imgunmodes.size[2] = imgU.md->size[2];
    createimagefromIMGID(&imgunmodes);

    int lastaxis = imgunmodes.naxis-1;
    long framesize = imgunmodes.size[0];
    if(lastaxis==2)
    {
        framesize *= imgunmodes.size[1];
    }

    for(int kk=0; kk<imgunmodes.size[lastaxis]; kk++)
    {
        float mfact = imgS.im->array.F[kk];
        for(long ii=0; ii<framesize; ii++)
        {
            imgunmodes.im->array.F[kk*framesize+ii] = imgU.im->array.F[kk*framesize+ii] * mfact;
        }
    }

    computeSGEMM(imgunmodes, imgV, imgM, 0, 1, GPUdev);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginU = mkIMGID_from_name(inmatU);
    resolveIMGID(&imginU, ERRMODE_ABORT);

    IMGID imginS = mkIMGID_from_name(invecS);
    resolveIMGID(&imginS, ERRMODE_ABORT);

    IMGID imginV = mkIMGID_from_name(inmatV);
    resolveIMGID(&imginV, ERRMODE_ABORT);




    IMGID imgoutM  = mkIMGID_from_name(outmatM);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        SVDmkM(imginU, imginS, imginV, &imgoutM, *GPUdevice);
        processinfo_update_output_stream(processinfo, imgoutM.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_linalgebra__SVDmkM()
{

    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
