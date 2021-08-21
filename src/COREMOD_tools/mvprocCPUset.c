/**
 * @file mvprocCPUset.c
 */

#include "CommandLineInterface/CLIcore.h"

// ==========================================
// Forward declaration(s)
// ==========================================

int COREMOD_TOOLS_mvProcCPUset(const char *csetname);

int COREMOD_TOOLS_mvProcCPUsetExt(int pid, const char *csetname, int rtprio);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t COREMOD_TOOLS_mvProcCPUset_cli()
{
    if (0 + CLI_checkarg(1, CLIARG_STR_NOT_IMG) == 0)
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

errno_t COREMOD_TOOLS_mvProcCPUsetExt_cli()
{
    if (0 + CLI_checkarg(1, CLIARG_LONG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) + CLI_checkarg(3, CLIARG_LONG) == 0)
    {
        COREMOD_TOOLS_mvProcCPUsetExt(data.cmdargtoken[1].val.numl,
                                      data.cmdargtoken[2].val.string,
                                      data.cmdargtoken[3].val.numl);
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

errno_t mvprocCPUsetExt_addCLIcmd()
{
    RegisterCLIcommand(
        "csetandprioext",
        __FILE__,
        COREMOD_TOOLS_mvProcCPUsetExt_cli,
        "move any PID to CPU set and assign RT priority (SCHED_FIFO) - priority ignored if 0",
        "<PID> <CPU set name> <RT priority>",
        "csetandprioext 23445 ircam0_edt 80",
        "int COREMOD_TOOLS_mvProcCPUsetExt(long pid, const char *csetname, int rtprio)");

    return RETURN_SUCCESS;
}

int COREMOD_TOOLS_mvProcCPUset(
    const char *csetname)
{
    // Pass down to extended version and return retcode back up
    return COREMOD_TOOLS_mvProcCPUsetExt(getpid(), csetname, 0);
}

int COREMOD_TOOLS_mvProcCPUsetExt(int pid, const char *csetname, int rtprio)
{
    char command[200];

    // Must make TWO calls
    // First promote the EUID to root, then this will make setuid promote the RUID to root
    // Which is what we need for the cset call to pass without a sudo password prompt.
    if (seteuid(data.euid) != 0 || setuid(data.euid) != 0) //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid/setuid error");
    }

    sprintf(command, "sudo -n cset proc --threads --force -m -p %d -t %s\n", pid, csetname);
    printf("Executing command: %s\n", command);

    if (system(command) != 0)
    {
        PRINT_ERROR("system() [cset proc -m ...] returns non-zero value");
    }

    if (rtprio > 0)
    {
        sprintf(command, "sudo -n chrt -f -p %d %d\n", rtprio, pid);
        printf("Executing command: %s\n", command);

        if (system(command) != 0)
        {
            PRINT_ERROR("system() [chrt -f -p ...] returns non-zero value");
        }
    }

    if (seteuid(data.ruid) != 0) //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    return (0);
}