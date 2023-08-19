/**
 * @file SingularValueDecomp_mkU.c
 *
 * @brief make U from M, V and S
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "SGEMM.h"



static char *inmatM;
static long  fpi_inmatM;

static char *inmatV;
static long  fpi_inmatV;

// singular values
static char *invecS;
static long  fpi_invecS;

static char *outmatU;
static long  fpi_outmatU;

static char *outmatUS;
static long  fpi_outmatUS;

static int32_t *GPUdevice;
static long     fpi_GPUdevice;





static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inM",
        "input matrix M",
        "inM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmatM,
        &fpi_inmatM
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
        CLIARG_IMG,
        ".inS",
        "input singular values vec",
        "inS",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &invecS,
        &fpi_invecS
    },
    {
        // output U
        CLIARG_STR,
        ".outU",
        "output U",
        "outU",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outmatU,
        &fpi_outmatU
    },
    {
        // output US
        CLIARG_STR,
        ".outUS",
        "output US",
        "outU",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outmatUS,
        &fpi_outmatUS
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
    "SVDmkU", "compute SVD U", CLICMD_FIELDS_DEFAULTS
};



static errno_t help_function()
{
    printf("Compute SVD's U from M, V and S\n");

    return RETURN_SUCCESS;
}





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

    computeSGEMM(
        imgM,
        imgV,
        imgUS,
        0,
        0,
        GPUdev
    );

    printf("SGEMM DONE\n");
    fflush(stdout);
    list_image_ID();

    uint32_t Ndim = imgV.md->size[imgV.md->naxis-1];
    uint64_t framesize;
    uint32_t nbframe;
    imgU->naxis = imgUS->naxis;
    imgU->datatype = imgUS->md->datatype;
    switch ( imgUS->md->naxis )
    {
    case 2 :
        imgU->size[0] = imgUS->md->size[0];
        imgU->size[1] = imgUS->md->size[1];
        framesize = imgUS->md->size[0];
        nbframe = imgUS->md->size[1];
        break;

    case 3 :
        imgU->size[0] = imgUS->md->size[0];
        imgU->size[1] = imgUS->md->size[1];
        imgU->size[2] = imgUS->md->size[2];
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

    for(uint32_t frame=0; frame<nbframe; frame++)
    {
        for(uint64_t ii=0; ii<framesize; ii++)
        {
            imgU->im->array.F[frame*framesize + ii] =  imgUS->im->array.F[frame*framesize + ii] / imgS.im->array.F[frame];
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM = mkIMGID_from_name(inmatM);
    resolveIMGID(&imginM, ERRMODE_ABORT);

    IMGID imginV = mkIMGID_from_name(inmatV);
    resolveIMGID(&imginV, ERRMODE_ABORT);

    IMGID imginS = mkIMGID_from_name(invecS);
    resolveIMGID(&imginS, ERRMODE_ABORT);


    IMGID imgoutU  = mkIMGID_from_name(outmatU);
    IMGID imgoutUS  = mkIMGID_from_name(outmatUS);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        compute_SVDU(imginM, imginV, imginS, &imgoutU, &imgoutUS, *GPUdevice);
        processinfo_update_output_stream(processinfo, imgoutU.ID);
        processinfo_update_output_stream(processinfo, imgoutUS.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_linalgebra__compSVDU()
{

    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
