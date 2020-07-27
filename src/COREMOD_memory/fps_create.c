/**
 * @file    fps_create.c
 * @brief   create function parameter structure
 */


#include "CommandLineInterface/CLIcore.h"





// ==========================================
// Forward declaration(s)
// ==========================================

errno_t fps_create();



// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t fps_create__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_LONG)
            + CLI_checkarg_noerrmsg(2, CLIARG_STR)
            == 0)
    {
        function_parameter_struct_create(
            data.cmdargtoken[1].val.numl,
            data.cmdargtoken[2].val.string
        );
		return CLICMD_SUCCESS;
    }
    else
    {
		return CLICMD_INVALID_ARG;
	}

}



// ==========================================
// Register CLI command(s)
// ==========================================

errno_t fps_create_addCLIcmd()
{

    RegisterCLIcommand(
        "fpscreate",
        __FILE__,
        fps_create__cli,
        "create function parameter structure (FPS)",
        "<NBparam> <name>",
        "fpscreate 100 newfps",
        "errno_t function_parameter_struct_create(int NBparamMAX, const char *name");

    return RETURN_SUCCESS;
}







