#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_tools/COREMOD_tools.h"

#include "SGEMM.h"


static char *inmodes;
static long  fpi_inmodes;

static char *outmodes;
static long  fpi_outmodes;


static int32_t *GPUdevice;
static long     fpi_GPUdevice;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inmodes",
        "input modes",
        "inm",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmodes,
        &fpi_inmodes
    },
    {
        CLIARG_STR,
        ".outmodes",
        "output modes",
        "outm",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outmodes,
        &fpi_outmodes
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
    "GramSchmidt", "Gram-Schmidt process", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("Run Gram-Schmodt process\n");

    return RETURN_SUCCESS;
}


errno_t GramSchmidt(
    IMGID imginm,
    IMGID *imgoutm,
    int GPUdev
)
{
    DEBUG_TRACE_FSTART();


    // Compute cross product on input
    IMGID imginxp  = mkIMGID_from_name("_outxp");
    computeSGEMM(imginm, imginm, &imginxp, 1, 0, GPUdev);


    // Create output
    //
    imcreatelikewiseIMGID(
        imgoutm,
        &imginm
    );

    uint32_t zsize;
    uint32_t xysize = imginm.md->size[0];
    if(imginm.md->naxis == 3)
    {
        zsize = imginm.md->size[2];
        xysize *= imginm.md->size[1];
    }
    else
    {
        zsize = imginm.md->size[1];
    }

    printf("xysize = %u, zsize = %u\n", xysize, zsize);

    for ( uint32_t kk=0; kk<zsize; kk++ )
    {
        // initializatoin
        memcpy( &imgoutm->im->array.F[kk*xysize], &imginm.im->array.F[kk*xysize], sizeof(float)*xysize);

        for ( uint32_t kk0 = 0; kk0 < kk; kk0++ )
        {
            // cross-product
            double xpval = 0.0;

            // square sum v0
            double sqrsum0 = 0.0;

            // square sum v1
            double sqrsum1= 0.0;

            for( uint32_t ii=0; ii<xysize; ii++)
            {
                float v0 = imgoutm->im->array.F[ kk0*xysize + ii];
                float v1 = imgoutm->im->array.F[ kk*xysize + ii];

                xpval += v0*v1;
                sqrsum0 += v0*v0;
                sqrsum1 += v1*v1;
            }

            float vcoeff = xpval / sqrsum0;

            printf("  %5u  %5u   %f\n", kk, kk0, vcoeff);

            for( uint32_t ii=0; ii<xysize; ii++)
            {
                imgoutm->im->array.F[ kk*xysize + ii] -= vcoeff * imgoutm->im->array.F[ kk0*xysize + ii];
            }
        }

    }


    list_image_ID();

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginm = mkIMGID_from_name(inmodes);
    resolveIMGID(&imginm, ERRMODE_ABORT);


    IMGID imgoutm  = mkIMGID_from_name(outmodes);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {


        GramSchmidt(imginm, &imgoutm, *GPUdevice);


    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_linalgebra__GramSchmidt()
{

    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

