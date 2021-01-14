/**
 * @file mvprocCPUset.c
 */

#include "CommandLineInterface/CLIcore.h"



// ==========================================
// Forward declaration(s)
// ==========================================

int COREMOD_TOOLS_mvProcCPUset(const char *csetname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================


errno_t COREMOD_TOOLS_mvProcCPUset_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_STR_NOT_IMG)
            == 0)
    {
        COREMOD_TOOLS_mvProcCPUset(
            data.cmdargtoken[1].val.string);

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

errno_t mvprocCPUset_addCLIcmd()
{
    RegisterCLIcommand(
        "csetpmove",
        __FILE__,
        COREMOD_TOOLS_mvProcCPUset_cli,
        "move current process to CPU set",
        "<CPU set name>",
        "csetpmove realtime",
        "int COREMOD_TOOLS_mvProcCPUset(const char *csetname)");


    return RETURN_SUCCESS;
}







int COREMOD_TOOLS_mvProcCPUset(
    const char *csetname
)
{
    int pid;
    char command[200];

    pid = getpid();

#ifndef __MACH__

    if(seteuid(data.euid) != 0)     //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }

    sprintf(command, "sudo -n cset proc --threads -m -p %d -t %s\n", pid, csetname);
    printf("Executing command: %s\n", command);

    if(system(command) != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }

    if(seteuid(data.ruid) != 0)     //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }
#endif

    return(0);
}

