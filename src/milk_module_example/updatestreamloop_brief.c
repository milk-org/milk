/**
 * @file    updatestreamloop_brief.c
 * @brief   simple procinfo+fps example - brief, no comments, uses macros
 *
 */


#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,  ".in_name",   "input stream", "ims1",
        CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        NULL
    },
    {
        CLIARG_LONG, ".delayus",   "delay [us]",   "2000",
        CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "streamupdatebrief",
    "update stream",
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg,
    0,
    NULL
};



/** @brief FPCONF function
 */
static errno_t FPSCONFfunction()
{
    FPS_SETUP_INIT(data.FPS_name, data.FPS_CMDCODE);
    fps_add_processinfo_entries(&fps);
    CMDargs_to_FPSparams_create(&fps);

    long fps_delayus = functionparameter_GetParamValue_INT64(&fps, ".delayus");

    FPS_CONFLOOP_START

    printf("delayus value inside conf loop: %ld\n", fps_delayus);

    FPS_CONFLOOP_END

    return RETURN_SUCCESS;
}



/** @brief Loop process code example
 */
static errno_t FPSRUNfunction()
{
    FPS_CONNECT(data.FPS_name, FPSCONNECT_RUN);

    char *fps_IDin_name = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
    long fps_delayus = functionparameter_GetParamValue_INT64(&fps, ".delayus");

    FPSPROCINFOLOOP_RUNINIT("streamupdate %.10s", fps_IDin_name);

    imageID IDin = image_ID(fps_IDin_name);

    PROCINFO_TRIGGER_DELAYUS(fps_delayus);



    PROCINFOLOOP_START

    processinfo_update_output_stream(processinfo, IDin);

    PROCINFOLOOP_END
    function_parameter_RUNexit(&fps);

    return RETURN_SUCCESS;
}


FPS_EXECFUNCTION_STD // macro defined in function_parameters.h
FPS_CLIFUNCTION_STD  // macro defined in function_parameters.h

errno_t FPSCLIADDCMD_milk_module_example__updatestreamloop_brief()
{
    RegisterCLIcmd(CLIcmddata, FPSCLIfunction);
    return RETURN_SUCCESS;
}





