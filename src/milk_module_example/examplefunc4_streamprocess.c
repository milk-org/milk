// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_process_loop_simple.c
 * @brief   template for simple stream processing loop
 *
 * Example 4
 * Function has input stream and output stream.
 */

#include <math.h> // for sqrt()

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

static char *outimname;
// Alternative way: static LOCVAR_OUTIMG2D outim;


static uint32_t *cntindex;
static long      fpi_cntindex = -1;

static uint32_t *cntindexmax;
static long      fpi_cntindexmax = -1;

static int64_t *ex0mode;
static long     fpi_ex0mode = -1;

static int64_t *ex1mode;
static long     fpi_ex1mode = -1;


static CLICMDARGDEF farg[] = {
    { CLIARG_IMG, ".in_name", "input image", "im1", CLIARG_VISIBLE_DEFAULT, (void **) &inimname,
      NULL },
    { CLIARG_STR, ".out_name", "output image", "out1", CLIARG_VISIBLE_DEFAULT, (void **) &outimname,
      NULL },
    // Note: an alternative way to specify an output image is FARG_OUTIM2D(outim)
    { CLIARG_UINT32, ".cntindex", "counter index", "5", CLIARG_HIDDEN_DEFAULT, (void **) &cntindex,
      &fpi_cntindex },
    { CLIARG_UINT32, ".cntindexmax", "counter index max value", "100", CLIARG_HIDDEN_DEFAULT,
      (void **) &cntindexmax, &fpi_cntindexmax },
    { CLIARG_ONOFF, ".option.ex0mode", "toggle0", "0", CLIARG_HIDDEN_DEFAULT, (void **) &ex0mode,
      &fpi_ex0mode },
    { CLIARG_ONOFF, ".option.ex1mode", "toggle1 conditional on toggle0", "0", CLIARG_HIDDEN_DEFAULT,
      (void **) &ex1mode, &fpi_ex1mode }
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

    if (*cntindex >= *cntindexmax)
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
    if (data.fpsptr != NULL)
    {
        // Here we set the FPS entries properties

        if (data.fpsptr->parray[fpi_ex0mode].fpflag & FPFLAG_ONOFF) // if ex0mode is in ON state
        {
            // Then activate ex1mode argument
            data.fpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_USED;
            data.fpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_VISIBLE;

            // Commonly use flags include:
            // FPFLAG_WRITECONF : Allow parameter to be written/changed while conf process is running
            // FPFLAG_RUNCONF   : Allow parameter to be written/changed while run process is running
        }
        else // OFF state
        {
            data.fpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_USED;
            data.fpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_VISIBLE;
        }

        // increment counter at every configuration check
        *cntindex = *cntindex + 1;

        if (*cntindex >= *cntindexmax)
        {
            *cntindex = 0;
        }
    }

    return RETURN_SUCCESS;
}


static CLICMDDATA CLIcmddata = { "streamprocess", "process input stream to output stream",
                                 CLICMD_FIELDS_DEFAULTS };


// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}


/**
 * @brief process image to output image
 *
 * Function arguments:
 * Input and output are passed by reference as
 * they may be changed by resolveIMGID and imcreateIMGID.
 *
 * Make sure to pass by reference if the function may change
 * IMGID
 */
static errno_t streamprocess(IMGID *inimg, IMGID *outimg)
{
    DEBUG_TRACE_FSTART();
    // custom stream process function code

    // resolve image
    // This function call has low overhead, as it will acknowledge existing image
    resolveIMGID(inimg, ERRMODE_ABORT);

    uint32_t xsize  = inimg->size[0];
    uint32_t ysize  = inimg->size[1];
    uint64_t xysize = xsize * ysize;


    // Create output image if needed.
    // Delaying creation of the image until here is necessary if the image size or type needs
    // to be determined within this function.
    //
    // We have called copyIMGID in compute_function(), so outimg metadata is already filled up.
    // Otherwise, we would edit these lines:
    // outimg->naxis = 2;
    // outimg->size[0] = xsize;
    // outimg->size[1] = ysize;
    // outimg->datatype = _DATATYPE_FLOAT;
    // outimg->shared = inimg->shared;
    // outimg->NBkw = inimg->NBkw;
    // outimg->CBsize = 0;

    // Create image if not already done.
    // Otherwise, just proceed
    imcreateIMGID(outimg); // Image is created, memory allocated

    outimg->md->write = 1;

    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        outimg->im->array.F[ii] = sqrt(inimg->im->array.F[ii]);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


    // Check if image is in memory
    // First, create an IMGIG with the image name
    IMGID inimg = mkIMGID_from_name(inimname);
    // Then resolve it (connect it to an image in memory if possible)
    resolveIMGID(&inimg, ERRMODE_ABORT);

    // Create output image/stream.
    // Here we only fill in the name.
    // The image itself will be created in the compute function.
    IMGID outimg = mkIMGID_from_name(outimname);

    // If we are sure we want outimg to be the same format (size, type etc) as inimg, we can use:
    copyIMGID(&inimg, &outimg);

    // Alternate way: FARG_OUTIM2DCREATE(outim, outimg, _DATATYPE_FLOAT);


    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    if (CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
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

        // stream is updated here, and not in the function called above, so that
        // the above function can be chained with others
        processinfo_update_output_stream(processinfo, outimg.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


INSERT_STD_FPSCLIfunctions_DynamicSize


    // Register function in CLI
    errno_t CLIADDCMD_milk_module_example__streamprocess()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
