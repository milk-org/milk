/**
 * @file    image_vecmult.c
 * @brief   multiply image by vector
 *
 */

#include "CommandLineInterface/CLIcore.h"

static char *iminname;
static long fpi_iminname;

static char *vecname;
static long fpi_vecname;

static char *imoutname;
static long fpi_imoutname;

static uint32_t *multaxis;
static long fpi_multaxis;





static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".iminname",
        "input image name",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &iminname,
        &fpi_iminname
    },
    {
        CLIARG_IMG,
        ".vecname",
        "input vector name",
        "vec",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &vecname,
        &fpi_vecname
    },
    {
        CLIARG_STR,
        ".imoutname",
        "output image name",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imoutname,
        &fpi_imoutname
    },
    {
        CLIARG_UINT32,
        ".axis",
        "multiplication axis",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &multaxis,
        &fpi_multaxis
    }
};



// Optional custom configuration setup.
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        data.fpsptr->parray[fpi_iminname].fpflag |=
            FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
    }

    return RETURN_SUCCESS;
}

// Optional custom configuration checks.
// Runs at every configuration check loop iteration
//
static errno_t customCONFcheck()
{

    if(data.fpsptr != NULL)
    {
    }

    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata =
{
    "imvecmult", "multiply image by vector", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}









errno_t image_vect_multiply(
    IMGID    imgin,
    IMGID    imgvec,
    IMGID    *imgout,
    uint32_t multaxis
)
{
    DEBUG_TRACE_FSTART();

    // check if images already exist
    //
    resolveIMGID(&imgin, ERRMODE_ABORT);
    resolveIMGID(&imgvec, ERRMODE_ABORT);

    resolveIMGID(imgout, ERRMODE_NULL);



    // Create output
    //
    if( (*imgout).ID == -1 )
    {
        imcreatelikewiseIMGID(
            imgout,
            &imgin
        );
    }

    uint32_t size0 = imgin.md->size[0];
    if (size0 == 0)
    {
        size0 = 1;
    }

    uint32_t size1 = imgin.md->size[1];
    if (size1 == 0)
    {
        size1 = 1;
    }

    uint32_t size2 = imgin.md->size[2];
    if (size2 == 0)
    {
        size2 = 1;
    }

    printf("size : %u %u %u\n", size0, size1, size2);

    double multfact = 1.0;
    for(uint32_t ii=0; ii<size0; ii++)
    {
        if(multaxis==0)
        {
            multfact = imgvec.im->array.F[ii];
        }
        for(uint32_t jj=0; jj<size1; jj++)
        {
            if(multaxis==1)
            {
                multfact = imgvec.im->array.F[jj];
            }
            for(uint32_t kk=0; kk<size2; kk++)
            {
                if(multaxis==2)
                {
                    multfact = imgvec.im->array.F[kk];
                }
                uint64_t pixi = ii;
                pixi += jj*size0;
                pixi += kk*size0*size1;
                imgout->im->array.F[pixi] = multfact * imgin.im->array.F[pixi];
            }
        }
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // input

    IMGID imgimin = mkIMGID_from_name(iminname);
    resolveIMGID(&imgimin, ERRMODE_ABORT);

    IMGID imgvec = mkIMGID_from_name(vecname);
    resolveIMGID(&imgvec, ERRMODE_ABORT);


    // output

    IMGID imgout  = mkIMGID_from_name(imoutname);




    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        image_vect_multiply(imgimin, imgvec, &imgout, *multaxis);
        processinfo_update_output_stream(processinfo, imgout.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}











INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__image_vecmult()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
