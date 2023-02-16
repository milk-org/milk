#include "CommandLineInterface/CLIcore.h"


static char *insname;
long fpi_insname;

static char *masksname;
long fpi_masksname;

static char *outsname;
long fpi_outsname;


static uint32_t *cropxstart;
long fpi_cropxstart;

static uint32_t *cropxsize;
long fpi_cropxsize;


static uint32_t *cropystart;
long fpi_cropystart;

static uint32_t *cropysize;
long fpi_cropysize;







static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STREAM,
        ".insname",
        "input stream name",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &insname,
        &fpi_insname
    },
    {
        CLIARG_STREAM,
        ".masksname",
        "mask stream name",
        "maskim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &masksname,
        &fpi_masksname
    },
    {
        CLIARG_STREAM,
        ".outsname",
        "output stream name",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outsname,
        &fpi_outsname
    },
    {
        CLIARG_UINT32,
        ".cropxstart",
        "crop x coord start",
        "30",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &cropxstart,
        &fpi_cropxstart
    },
    {
        CLIARG_UINT32,
        ".cropxsize",
        "crop x coord size",
        "32",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &cropxsize,
        &fpi_cropxsize
    },
    {
        CLIARG_UINT32,
        ".cropystart",
        "crop y coord start",
        "20",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &cropystart,
        &fpi_cropystart
    },
    {
        CLIARG_UINT32,
        ".cropysize",
        "crop y coord size",
        "32",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &cropysize,
        &fpi_cropysize
    }
};



// Optional custom configuration setup.
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        data.fpsptr->parray[fpi_insname].fpflag |=
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
    "cropmask", "crop and mask image", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}







static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // CONNECT TO INPUT STREAM
    IMGID imgin = mkIMGID_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_ABORT);
    printf("Input stream size : %u %u\n", imgin.md->size[0], imgin.md->size[1]);
    long m = imgin.md->size[0] * imgin.md->size[1];


    // CONNNECT TO OR CREATE MASK STREAM
    IMGID imgmask = stream_connect_create_2Df32(masksname, *cropxsize, *cropysize);

    // CONNNECT TO OR CREATE OUTPUT STREAM
    IMGID imgout = stream_connect_create_2Df32(outsname, *cropxsize, *cropysize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;



    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        for(uint32_t jj = 0; jj < *cropysize; jj++)
        {
            uint64_t indjj = jj + (*cropystart);
            indjj *=  imgin.md->size[0];
            for(uint32_t ii = 0; ii < *cropxsize; ii++)
            {
                imgout.im->array.F[ jj * (*cropxsize) + ii ]
                = imgmask.im->array.F[ jj * (*cropxsize) + ii ]
                * imgin.im->array.F[ indjj + ii + (*cropxstart) ];
            }
        }
        processinfo_update_output_stream(processinfo, imgout.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__cropmask()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
