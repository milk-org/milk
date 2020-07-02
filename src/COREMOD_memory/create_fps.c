/**
 * @file    create_fps.c
 * @brief   create function parameter structure
 */


#include "CommandLineInterface/CLIcore.h"





// ==========================================
// Forward declaration(s)
// ==========================================

errno_t list_fps();



// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t create_fps__cli()
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

errno_t create_fps_addCLIcmd()
{

    RegisterCLIcommand(
        "createfps",
        __FILE__,
        create_fps__cli,
        "create function parameter structure (FPS)",
        "<NBparam> <name>",
        "createfps 100 newfps",
        "errno_t function_parameter_struct_create(int NBparamMAX, const char *name");

    return RETURN_SUCCESS;
}







