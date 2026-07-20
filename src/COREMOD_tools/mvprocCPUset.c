// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mvprocCPUset.c
 */
#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include "CommandLineInterface/CLIcore.h"

// ==========================================
// Forward declaration(s)
// ==========================================

int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec);
int COREMOD_TOOLS_mvProcCPUsetExt(const int pid, const char *csetname, const int rtprio);
int COREMOD_TOOLS_mvProcTset(const char *tsetspec);
int COREMOD_TOOLS_mvProcRTPrio(const int rtprio);
int COREMOD_TOOLS_mvProcCPUset(const char *csetname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t COREMOD_TOOLS_mvProcRTPrio_cli()
{
    if (0 + CLI_checkarg(1, CLIARG_LONG) == 0)
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
    if (0 + CLI_checkarg(1, CLIARG_STR_NOT_IMG) == 0)
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
    if (0 + CLI_checkarg(1, CLIARG_INT64) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) == 0)
    {
        COREMOD_TOOLS_mvProcTsetExt(data.cmdargtoken[1].val.numl, data.cmdargtoken[2].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t COREMOD_TOOLS_mvProcCPUset_cli()
{
    if (0 + CLI_checkarg(1, CLIARG_STR_NOT_IMG) == 0)
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
    if (0 + CLI_checkarg(1, CLIARG_INT64) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_INT64) ==
        0)
    {
        COREMOD_TOOLS_mvProcCPUsetExt(data.cmdargtoken[1].val.numl, data.cmdargtoken[2].val.string,
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
    RegisterCLIcommand("rtprio", __FILE__, COREMOD_TOOLS_mvProcRTPrio_cli,
                       "Set current process SCHED_FIFO priority", "<prio>", "rtprio <prio>",
                       "int COREMOD_TOOLS_mvProcRTPrio(const int rtprio)");
    RegisterCLIcommand("tsetpmove", __FILE__, COREMOD_TOOLS_mvProcTset_cli,
                       "Assign taskset to current process", "<taskset spec list>",
                       "tsetpmove realtime", "int COREMOD_TOOLS_mvProcTset(const char *tsetspec)");
    RegisterCLIcommand("tsetpmoveext", __FILE__, COREMOD_TOOLS_mvProcTsetExt_cli,
                       "Assign taskset for any process", "<PID> <taskset spec list>",
                       "tsetpmoveext 33659 1-5",
                       "int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec)");
    RegisterCLIcommand("csetpmove", __FILE__, COREMOD_TOOLS_mvProcCPUset_cli,
                       "move current process to CPU set", "<CPU set name>", "csetpmove realtime",
                       "int COREMOD_TOOLS_mvProcCPUset(const char *csetname)");
    RegisterCLIcommand("csetandprioext", __FILE__, COREMOD_TOOLS_mvProcCPUsetExt_cli,
                       "move any PID to CPU set and assign RT priority "
                       "(SCHED_FIFO) - priority ignored if 0",
                       "<PID> <CPU set name> <RT priority>", "csetandprioext 23445 ircam0_edt 80",
                       "int COREMOD_TOOLS_mvProcCPUsetExt(const int pid, const "
                       "char *csetname, const int rtprio)");

    return RETURN_SUCCESS;
}

int COREMOD_TOOLS_mvProcRTPrio(const int rtprio)
{
    if (rtprio <= 0)
    {
        PRINT_WARNING("Invoking RT prio with rtprio %d <= 0; skipping.", rtprio);
        return RETURN_SUCCESS;
    }

    EXECUTE_SYSTEM_COMMAND_ERRCHECK("milk-rtsched -p %d %d", rtprio, getpid());

    return RETURN_SUCCESS;
}

int COREMOD_TOOLS_mvProcTset(const char *tsetspec)
{
    // Pass down to extended version and return retcode back up
    return COREMOD_TOOLS_mvProcTsetExt(getpid(), tsetspec);
}

int COREMOD_TOOLS_mvProcTsetExt(const int pid, const char *tsetspec)
{
    EXECUTE_SYSTEM_COMMAND("milk-rtsched -t %s %d", tsetspec, pid);
    return 0;
}


int COREMOD_TOOLS_mvProcCPUset(const char *csetname)
{
    // Pass down to extended version and return retcode back up
    return COREMOD_TOOLS_mvProcCPUsetExt(getpid(), csetname, -1);
}


int COREMOD_TOOLS_mvProcCPUsetExt(const int pid, const char *csetname, const int rtprio)
{
    if (rtprio > 0)
    {
        EXECUTE_SYSTEM_COMMAND("milk-rtsched -c %s -p %d %d", csetname, rtprio, pid);
    }
    else
    {
        EXECUTE_SYSTEM_COMMAND("milk-rtsched -c %s %d", csetname, pid);
    }

    return EXIT_SUCCESS;
}
