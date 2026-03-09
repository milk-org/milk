/**
 * @file mvprocCPUset.c
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

// ==========================================
// Forward declaration(s)
// ==========================================

int COREMOD_TOOLS_mvProcRTPrio(const int rtprio);

int COREMOD_TOOLS_mvProcTset(const char *tsetspec);

int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec);

int COREMOD_TOOLS_mvProcCPUset(const char *csetname);


int COREMOD_TOOLS_mvProcCPUsetExt(const int   pid,
                                  const char *csetname,
                                  const int   rtprio);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

#ifndef MILK_NO_CLI
errno_t COREMOD_TOOLS_mvProcRTPrio_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_LONG) == 0)
    {
        COREMOD_TOOLS_mvProcRTPrio(data.cmdargtoken[1].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t COREMOD_TOOLS_mvProcTset_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_STR_NOT_IMG) == 0)
    {
        COREMOD_TOOLS_mvProcTset(data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t COREMOD_TOOLS_mvProcTsetExt_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_INT64) +
            CLI_checkarg(2, CLIARG_STR_NOT_IMG) ==
            0)
    {
        COREMOD_TOOLS_mvProcTsetExt(data.cmdargtoken[1].val.numl,
                                    data.cmdargtoken[2].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t COREMOD_TOOLS_mvProcCPUset_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_STR_NOT_IMG) == 0)
    {
        COREMOD_TOOLS_mvProcCPUset(data.cmdargtoken[1].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t COREMOD_TOOLS_mvProcCPUsetExt_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_INT64) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_INT64) ==
            0)
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

errno_t cpuset_utils_addCLIcmd()
{
    RegisterCLIcommand("rtprio",
                       __FILE__,
                       COREMOD_TOOLS_mvProcRTPrio_cli,
                       "Set current process SCHED_FIFO priority",
                       "<prio>",
                       "rtprio <prio>",
                       "int COREMOD_TOOLS_mvProcRTPrio(const int rtprio)");
    RegisterCLIcommand("tsetpmove",
                       __FILE__,
                       COREMOD_TOOLS_mvProcTset_cli,
                       "Assign taskset to current process",
                       "<taskset spec list>",
                       "tsetpmove realtime",
                       "int COREMOD_TOOLS_mvProcTset(const char *tsetspec)");
    RegisterCLIcommand("tsetpmoveext",
                       __FILE__,
                       COREMOD_TOOLS_mvProcTsetExt_cli,
                       "Assign taskset for any process",
                       "<PID> <taskset spec list>",
                       "tsetpmoveext 33659 1-5",
                       "int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec)");
    RegisterCLIcommand("csetpmove",
                       __FILE__,
                       COREMOD_TOOLS_mvProcCPUset_cli,
                       "move current process to CPU set",
                       "<CPU set name>",
                       "csetpmove realtime",
                       "int COREMOD_TOOLS_mvProcCPUset(const char *csetname)");
    RegisterCLIcommand("csetandprioext",
                       __FILE__,
                       COREMOD_TOOLS_mvProcCPUsetExt_cli,
                       "move any PID to CPU set and assign RT priority "
                       "(SCHED_FIFO) - priority ignored if 0",
                       "<PID> <CPU set name> <RT priority>",
                       "csetandprioext 23445 ircam0_edt 80",
                       "int COREMOD_TOOLS_mvProcCPUsetExt(const int pid, const "
                       "char *csetname, const int rtprio)");

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */

int COREMOD_TOOLS_mvProcRTPrio(const int rtprio)
{
    if(rtprio <= 0)
    {
        PRINT_WARNING("Invoking RT prio with rtprio %d <= 0; skipping.", rtprio);
        return RETURN_SUCCESS;
    }

    char command[200];

    if(seteuid(dceuid) != 0 ||
            setuid(dceuid) != 0) // This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid/setuid error");
        return RETURN_FAILURE;
    }

    sprintf(command, "chrt -f -p %d %d\n", rtprio, getpid());
    printf("Executing command: %s\n", command);

    EXECUTE_SYSTEM_COMMAND_ERRCHECK("%s", command);

    if(setresuid(dcruid, dcruid, dceuid) !=
            0) // Go back to normal privileges
    {
        PRINT_ERROR("seteuid error after executing chrt");
        //TODO probably should force a quit here... since we're remaining
        //TODO at elevated privileges and we really shoudn't.
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
}

int COREMOD_TOOLS_mvProcTset(const char *tsetspec)
{
    // Pass down to extended version and return retcode back up
    return COREMOD_TOOLS_mvProcTsetExt(getpid(), tsetspec);
}

int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec)
{
    char command[200];

    // Must make TWO calls
    // First call: promote the EUID to root,
    // Second call: setuid promote the RUID to root
    // Which is what we need for the cset call to pass without a sudo password prompt.

    /* FOR DEBUG - WARNING dceuid and dcruid are NOT what they say
    PRINT_ERROR("(data) EUID %d - (data) RUID %d ", dceuid, dcruid);
    int euid, suid, ruid;
    getresuid(&ruid, &euid, &suid);
    PRINT_ERROR("AC EUID %d - SUID %d - RUID %d ", euid, suid, ruid);
    //*/

    if(seteuid(dceuid) != 0 ||
            setuid(dceuid) != 0) // This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid/setuid error");
    }

    sprintf(command, "taskset -pc %s %d\n", tsetspec, pid);
    printf("Executing command: %s\n", command);

    EXECUTE_SYSTEM_COMMAND_ERRCHECK("%s", command);

    if(setresuid(dcruid, dcruid, dceuid) !=
            0) // Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    return (0);
}


int COREMOD_TOOLS_mvProcCPUset(const char *csetname)
{
    // Pass down to extended version and return retcode back up
    return COREMOD_TOOLS_mvProcCPUsetExt(getpid(), csetname, -1);
}


int COREMOD_TOOLS_mvProcCPUsetExt(const int   pid,
                                  const char *csetname,
                                  const int   rtprio)
{
    char command[STRINGMAXLEN_COMMAND];

    /* FOR DEBUG - WARNING dceuid and dcruid are NOT what they say
    PRINT_ERROR("(data) EUID %d - (data) RUID %d ", dceuid, dcruid);
    int euid, suid, ruid;
    getresuid(&ruid, &euid, &suid);
    PRINT_ERROR("AC EUID %d - SUID %d - RUID %d ", euid, suid, ruid);
    //*/

    // Must make TWO calls - see COREMOD_TOOLS_mvProcTset
    if(seteuid(dceuid) != 0 ||
            setuid(dceuid) != 0) // This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid/setuid error");
    }

    sprintf(command,
            "cset proc --threads --force -m -p %d -t %s\n",
            pid,
            csetname);
    printf("Executing command: %s\n", command);

    if(system("which cset > /dev/null 2>&1"))
    {
        // Command doesn't exist...
        printf("Error: cset command is not installed\n");
    }
    else
    {
        // Command does exist
        EXECUTE_SYSTEM_COMMAND("%s", command);
        if(dcretval != 0)
        {
            if(dcretval == 512)
            {
                PRINT_ERROR("Error: cset-proc returns error 512 - cpuset %s does not exist.\n",
                            csetname);
            }
            else
            {
                // Re-raise as would EXECUTE_SYTEM_COMMAND_ERRCHECK
                PRINT_ERROR("Error: cset-proc returns error %d.", dcretval);
                abort();
            }
        }
    }

    if(rtprio > 0)
    {
        sprintf(command, "chrt -f -p %d %d\n", rtprio, pid);
        printf("Executing command: %s\n", command);

        EXECUTE_SYSTEM_COMMAND_ERRCHECK("%s", command);
    }

    if(setresuid(dcruid, dcruid, dceuid) !=
            0) // Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    return 0;
}
