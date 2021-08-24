/**
 * @file    updatestreamloop.c
 * @brief   simple procinfo+fps example - brief, no comments, uses macros
 *
 * Example 3
 */


#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"


// variables local to this translation unit
static char *inimname;


static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,  ".in_sname", "input stream", "ims1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname
    }
};

static CLICMDDATA CLIcmddata =
{
    "streamupdate",
    "update stream",
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

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    processinfo_update_output_stream(processinfo, img.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_milk_module_example__updatestreamloop()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
