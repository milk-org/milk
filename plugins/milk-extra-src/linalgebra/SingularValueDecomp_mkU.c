#include "ImageStreamIO/ImageStruct.h"
/**
 * @file SingularValueDecomp_mkU.c
 *
 * @brief make U from M, V and S
 *
 */

#include <math.h>

#include "CLIcore.h"

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
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &inmatM,
        &fpi_inmatM
    },
    {
        CLIARG_IMG,
        ".inV",
        "input matrix V",
        "inV",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &inmatV,
        &fpi_inmatV
    },
    {
        CLIARG_IMG,
        ".inS",
        "input singular values vec",
        "inS",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &invecS,
        &fpi_invecS
    },
    {
        // output U
        CLIARG_STR,
        ".outU",
        "output U",
        "outU",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &outmatU,
        &fpi_outmatU
    },
    {
        // output US
        CLIARG_STR,
        ".outUS",
        "output US",
        "outU",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &outmatUS,
        &fpi_outmatUS
    },
    {
        // using GPU (99 : no GPU, otherwise GPU device)
        CLIARG_INT32,
        ".GPUdevice",
        "GPU device, 99 for CPU",
        "-1",
        FPFLAG_DEFAULT_INPUT,
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

    resolveIMGID(&imgM, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    resolveIMGID(&imgV, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);
    resolveIMGID(&imgS, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

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






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM = imgid_make_from_name(inmatM);
    resolveIMGID(&imginM, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    IMGID imginV = imgid_make_from_name(inmatV);
    resolveIMGID(&imginV, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    IMGID imginS = imgid_make_from_name(invecS);
    resolveIMGID(&imginS, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);


    IMGID imgoutU  = imgid_make_from_name(outmatU);
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
