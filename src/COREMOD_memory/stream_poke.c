/**
 * @file    stream_poke.c
 * @brief   poke image stream
 */



#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"





// variables local to this translation unit
static char *inimname;




static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,  ".in_sname", "input stream", "ims1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname, NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "shmimpoke",
    "update stream without changing content",
    CLICMD_FIELDS_DEFAULTS
};


// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




// Wrapper function, used by all CLI calls
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = makeIMGID(inimname);
    resolveIMGID(&img, ERRMODE_ABORT);

    printf("running PCA on image %s\n", img.name);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    processinfo_update_output_stream(processinfo, img.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_COREMOD_memory__stream_poke()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
