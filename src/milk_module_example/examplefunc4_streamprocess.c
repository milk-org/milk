/**
 * @file    stream_process_loop_simple.c
 * @brief   template for simple stream processing loop
 *
 * Example 3
 * Function has input stream and output stream.
 */


#include "CommandLineInterface/CLIcore.h"

// required for create_2Dimage_ID()
//#include "COREMOD_memory/COREMOD_memory.h"

// required for timespec_diff()
//#include "COREMOD_tools/COREMOD_tools.h"

// required for timespec_diff
//#include "CommandLineInterface/timeutils.h"



// Local variables pointers
static char *inimname;
static char *outimname;
static uint32_t *cntindex;
static uint32_t *cntindexmax;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".in_name", "input image", "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname, NULL
    },
    {
        CLIARG_IMG, ".out_name", "output image", "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname, NULL
    },
    {
        CLIARG_UINT32, ".cntindex", "counter index", "5",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cntindex, NULL
    },
    {
        CLIARG_UINT32, ".cntindexmax", "counter index max value", "100",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cntindexmax, NULL
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
    // increment counter at every configuration check
    *cntindex = *cntindex + 1;

    if(*cntindex >= *cntindexmax)
    {
        *cntindex = 0;
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




static errno_t streamprocess(IMGID inimg, IMGID outimg)
{
    // custom stream process function code

    (void) inimg;
    (void) outimg;

    return RETURN_SUCCESS;
}



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = makeIMGID(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);

    IMGID outimg = makeIMGID(outimname);
    resolveIMGID(&outimg, ERRMODE_ABORT);


    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    streamprocess(inimg, outimg);
    processinfo_update_output_stream(processinfo, outimg.ID);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_milk_module_example__streamprocess()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}


