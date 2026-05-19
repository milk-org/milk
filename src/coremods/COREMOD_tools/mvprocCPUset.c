/**
 * @file mvprocCPUset.c
 * @brief CPU set and RT priority utilities
 *
 * Uses FPS V2 framework.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

/* forward decls */
int COREMOD_TOOLS_mvProcRTPrio(
    const int rtprio);
int COREMOD_TOOLS_mvProcTset(
    const char *tsetspec);
int COREMOD_TOOLS_mvProcTsetExt(
    const int  pid,
    const char *tsetspec);
int COREMOD_TOOLS_mvProcCPUset(
    const char *csetname);
int COREMOD_TOOLS_mvProcCPUsetExt(
    const int  pid,
    const char *csetname,
    const int  rtprio);


/* ================================================================
 *  COMMON PARAMS
 * ============================================================= */

static long long p_pid = 0;
static char p_name[
     FUNCTION_PARAMETER_STRMAXLEN]
    = "realtime";
static long long p_rtprio = 80;


/* ================================================================
 *  CMD 1: rtprio (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_rtp =
{
    .fps_name    = "rtprio",
    .cmdkey      = "rtprio",
    .description =
    "set SCHED_FIFO priority",
    .description_long =
    "Move processes to specific CPU sets for core pinning and isolation. Supports assigning PIDs to NUMA-aware CPU groups for real-time performance."
};

#define FPS_PARAMS_RTP(X) \
    X(".rtprio", &p_rtprio, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "RT priority")

static CLICMDDATA CLIcmddata_rtp =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(rtp, CLIcmddata_rtp, FPS_app_info_rtp)

static errno_t __attribute__((unused)) compute_rtp()
{
    COREMOD_TOOLS_mvProcRTPrio(p_rtprio);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: tsetpmove (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_tset =
{
    .fps_name    = "tsetpmove",
    .cmdkey      = "tsetpmove",
    .description =
    "assign taskset to current process",
    .description_long =
    "Move processes to specific CPU sets for core pinning and isolation. Supports assigning PIDs to NUMA-aware CPU groups for real-time performance."
};

#define FPS_PARAMS_TSET(X) \
    X(".name", p_name, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "taskset spec list")

static CLICMDDATA CLIcmddata_tset =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(tset, CLIcmddata_tset, FPS_app_info_tset)

static errno_t __attribute__((unused)) compute_tset()
{
    COREMOD_TOOLS_mvProcTset(p_name);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: tsetpmoveext (2 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_tsete =
{
    .fps_name    = "tsetpmoveext",
    .cmdkey      = "tsetpmoveext",
    .description =
    "assign taskset for any process",
    .description_long =
    "Move processes to specific CPU sets for core pinning and isolation. Supports assigning PIDs to NUMA-aware CPU groups for real-time performance."
};

#define FPS_PARAMS_TSETE(X) \
    X(".pid", &p_pid, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "PID") \
    X(".name", p_name, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "taskset spec list")

static CLICMDDATA CLIcmddata_tsete =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(tsete, CLIcmddata_tsete, FPS_app_info_tsete)

static errno_t __attribute__((unused)) compute_tsete()
{
    COREMOD_TOOLS_mvProcTsetExt(
        p_pid, p_name);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 4: csetpmove (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_cset =
{
    .fps_name    = "csetpmove",
    .cmdkey      = "csetpmove",
    .description =
    "move current process to CPU set",
    .description_long =
    "Move processes to specific CPU sets for core pinning and isolation. Supports assigning PIDs to NUMA-aware CPU groups for real-time performance."
};

static CLICMDDATA CLIcmddata_cset =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(cset, CLIcmddata_cset, FPS_app_info_cset)

static errno_t __attribute__((unused)) compute_cset()
{
    COREMOD_TOOLS_mvProcCPUset(p_name);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 5: csetandprioext (3 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "csetandprioext",
    .cmdkey      = "csetandprioext",
    .description =
    "move PID to CPU set and assign "
    "RT priority",
    .description_long =
    "Move processes to specific CPU sets for core pinning and isolation. Supports assigning PIDs to NUMA-aware CPU groups for real-time performance."
};

#define FPS_PARAMS(X) \
    X(".pid", &p_pid, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "PID") \
    X(".name", p_name, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "CPU set name") \
    X(".rtprio", &p_rtprio, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "RT priority (0=ignore)")

static FPS_CLI_BINDING my_bindings[] =
{
    FPS_PARAMS(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] =
{
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata =
{
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(main, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    COREMOD_TOOLS_mvProcCPUsetExt(
        p_pid, p_name, p_rtprio);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

/* bindings for single-param commands */
static FPS_CLI_BINDING bindings_rtp[] =
{
    FPS_PARAMS_RTP(FPS_X_BINDING)
};
static const int nb_bindings_rtp =
    sizeof(bindings_rtp) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_rtp[] =
{
    FPS_PARAMS_RTP(FPS_X_FARG)
};

static FPS_CLI_BINDING bindings_tset[] =
{
    FPS_PARAMS_TSET(FPS_X_BINDING)
};
static const int nb_bindings_tset =
    sizeof(bindings_tset) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_tset[] =
{
    FPS_PARAMS_TSET(FPS_X_FARG)
};

static FPS_CLI_BINDING bindings_tsete[] =
{
    FPS_PARAMS_TSETE(FPS_X_BINDING)
};
static const int nb_bindings_tsete =
    sizeof(bindings_tsete) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_tsete[] =
{
    FPS_PARAMS_TSETE(FPS_X_FARG)
};

/* csetpmove reuses TSET bindings (1 str) */

static errno_t CLIfunction_rtp(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_rtp,
               farg_rtp, &CLIcmddata_rtp,
               bindings_rtp, nb_bindings_rtp,
               compute_rtp);
}

static errno_t CLIfunction_tset(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_tset,
               farg_tset, &CLIcmddata_tset,
               bindings_tset, nb_bindings_tset,
               compute_tset);
}

static errno_t CLIfunction_tsete(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_tsete,
               farg_tsete, &CLIcmddata_tsete,
               bindings_tsete,
               nb_bindings_tsete,
               compute_tsete);
}

static errno_t CLIfunction_cset(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_cset,
               farg_tset, &CLIcmddata_cset,
               bindings_tset, nb_bindings_tset,
               compute_cset);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata,
               my_bindings, nb_bindings,
               compute_function);
}

errno_t
CLIADDCMD_COREMOD_tools__mvprocCPUset()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    safe_fps_fill_farg_examples(
        farg_rtp, bindings_rtp,
        nb_bindings_rtp);
    safe_fps_fill_farg_examples(
        farg_tset, bindings_tset,
        nb_bindings_tset);
    safe_fps_fill_farg_examples(
        farg_tsete, bindings_tsete,
        nb_bindings_tsete);

    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata_rtp,
                       CLIfunction_rtp);
        CLIcmddata_rtp.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata_tset,
                       CLIfunction_tset);
        CLIcmddata_tset.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata_tsete,
                       CLIfunction_tsete);
        CLIcmddata_tsete.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata_cset,
                       CLIfunction_cset);
        CLIcmddata_cset.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif

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

    snprintf(command,
             sizeof(command),
             "chrt -f -p %d %d\n",
             rtprio, getpid());
    printf("Executing command: %s\n", command);

    EXECUTE_SYSTEM_COMMAND("%s", command);

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

int COREMOD_TOOLS_mvProcTsetExt(
    const int  pid,
    const char *tsetspec)
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

    snprintf(command,
             sizeof(command),
             "taskset -pc %s %d\n",
             tsetspec, pid);
    printf("Executing command: %s\n", command);

    EXECUTE_SYSTEM_COMMAND("%s", command);

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


int COREMOD_TOOLS_mvProcCPUsetExt(
    const int  pid,
    const char *csetname,
    const int  rtprio)
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

    snprintf(command,
             sizeof(command),
             "cset proc --threads --force"
             " -m -p %d -t %s\n",
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
        EXECUTE_SYSTEM_COMMAND_NOCHECK("%s", command);
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
        snprintf(command,
                 sizeof(command),
                 "chrt -f -p %d %d\n",
                 rtprio, pid);
        printf("Executing command: %s\n", command);

        EXECUTE_SYSTEM_COMMAND("%s", command);
    }

    if(setresuid(dcruid, dcruid, dceuid) !=
            0) // Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    return 0;
}
