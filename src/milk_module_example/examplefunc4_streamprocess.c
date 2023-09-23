/**
 * @file    stream_process_loop_simple.c
 * @brief   template for simple stream processing loop
 *
 * Example 4
 * Function has input stream and output stream.
 */

#include "CommandLineInterface/CLIcore.h"

//static int cmdindex;

// required for create_2Dimage_ID()
//#include "COREMOD_memory/COREMOD_memory.h"

// required for timespec_diff()
//#include "COREMOD_tools/COREMOD_tools.h"

// required for timespec_diff
//#include "CommandLineInterface/timeutils.h"

// Local variables pointers

static char *inimname;

static LOCVAR_OUTIMG2D outim;


static uint32_t *cntindex;
static long      fpi_cntindex = -1;

static uint32_t *cntindexmax;
static long      fpi_cntindexmax = -1;

static int64_t *ex0mode;
static long     fpi_ex0mode = -1;

static int64_t *ex1mode;
static long     fpi_ex1mode = -1;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    FARG_OUTIM2D(outim),
    {
        CLIARG_UINT32,
        ".cntindex",
        "counter index",
        "5",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cntindex,
        &fpi_cntindex
    },
    {
        CLIARG_UINT32,
        ".cntindexmax",
        "counter index max value",
        "100",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cntindexmax,
        &fpi_cntindexmax
    },
    {
        CLIARG_ONOFF,
        ".option.ex0mode",
        "toggle0",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &ex0mode,
        &fpi_ex0mode
    },
    {
        CLIARG_ONOFF,
        ".option.ex1mode",
        "toggle1 conditional on toggle0",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &ex1mode,
        &fpi_ex1mode
    }
};

// Optional custom configuration setup
// Runs once at conf startup
//
// To use this function, set :
// CLIcmddata.FPS_customCONFsetup = customCONFsetup
// when registering function
// (see end of this file)
//
static errno_t customCONFsetup()
{
    // increment counter at every configuration check
    *cntindex = *cntindex + 1;

    if(*cntindex >= *cntindexmax)
    {
        *cntindex = 0;
    }

    return RETURN_SUCCESS;
}

// Optional custom configuration checks
// Runs at every configuration check loop iteration
//
// To use this function, set :
// CLIcmddata.FPS_customCONFcheck = customCONFcheck
// when registering function
// (see end of this file)
//
static errno_t customCONFcheck()
{
    if(data.fpsptr != NULL)
    {
        if(data.fpsptr->parray[fpi_ex0mode].fpflag & FPFLAG_ONOFF)  // ON state
        {
            data.fpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_USED;
            data.fpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_VISIBLE;
        }
        else // OFF state
        {
            data.fpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_USED;
            data.fpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_VISIBLE;
        }

        // increment counter at every configuration check
        *cntindex = *cntindex + 1;

        if(*cntindex >= *cntindexmax)
        {
            *cntindex = 0;
        }

    }

    return RETURN_SUCCESS;
}


static CLICMDDATA CLIcmddata =
{
    "streamprocess",
    "process input stream to output stream",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



static errno_t streamprocess(
    IMGID *inimg,
    IMGID *outimg
)
{
    DEBUG_TRACE_FSTART();
    // custom stream process function code


    // resolve imgpos
    resolveIMGID(inimg, ERRMODE_ABORT);

    // Create output image if needed
    imcreateIMGID(outimg);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);

    // link/create output image/stream
    FARG_OUTIM2DCREATE(outim, outimg, _DATATYPE_FLOAT);




    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    // If custom initialization with access to procinfo is not required
    // then replace
    // INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    // INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    // With :
    // INSERT_STD_PROCINFO_COMPUTEFUNC_START

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        streamprocess(&inimg, &outimg);

        // stream is updated here, and not in the function called above, so that multiple
        // the above function can be chained with others
        processinfo_update_output_stream(processinfo, outimg.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_milk_module_example__streamprocess()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
