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
        CLIARG_IMG,  ".in_name",   "input stream", "ims1",
        CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        (void **) &inimname
    }
};

static CLICMDDATA CLIcmddata =
{
    "streamupdate",
    "update stream",
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg,
    CLICMDFLAG_FPS,
    NULL
};



// Wrapper function, used by all CLI calls
static errno_t compute_function()
{
    IMGID img = makeIMGID(inimname);
    resolveIMGID(&img, ERRMODE_ABORT);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    processinfo_update_output_stream(processinfo, img.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_milk_module_example__updatestreamloop()
{
    INSERT_STD_FPSCLIREGISTERFUNC

    return RETURN_SUCCESS;
}
