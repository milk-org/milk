/**
 * @file    fps_processcmdline_interactive.c
 * @brief   FPS process command line interactive execution
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <string.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"
#include "fps_scan.h"
#include "fps_CONFstart.h"
#include "fps_CONFstop.h"
#include "fps_RUNstart.h"
#include "fps_RUNstop.h"
#include "fps_tmux.h"
#include "fps_FPSremove.h"
#include "fps_outlog.h"
#include "fps_paramvalue.h"
#include "fps_save2disk.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_printparameter_valuestring.h"

/** @brief process command line
 *
 * ## Purpose
 *
 * Process command line.
 *
 * ## Commands
 *
 * - logsymlink  : create log sym link
 * - fpswfile    : write fps file to disk (writes to datadir)
 * - setval      : set parameter value
 * - getval      : get value, write to output log
 * - fwrval      : get value, write to file or fifo
 * - exec        : execute scripte (parameter must be FPTYPE_EXECFILENAME type)
 * - tmuxstart   : start tmux session
 * - tmuxstop    : stop tmux session
 * - confstart   : start RUN process associated with parameter
 * - confstop    : start RUN process associated with parameter
 * - confupdate  : update configuration
 * - confwupdate : update configuration, wait for completion to proceed
 * - runstart    : start RUN process associated with parameter
 * - runstop     : stop RUN process associated with parameter
 * - fpsrm       : remove fps
 * - cntinc      : counter test to check fifo connection
 * - rescan      : rescan fps tree
 * - exit        : exit fpsCTRL tool
 *
 * - queueprio   : change queue priority
 *
 *
 */

static void fps_cmd_handle_sys(
    const char                *FPScommand,
    int                        nbword,
    const char                *FPSarg0,
    const char                *FPSarg1,
    FPSCTRL_PROCESS_VARS      *fpsCTRLvar,
    FPS *fps,
    KEYWORD_TREE_NODE         *keywnode,
    FPSCTRL_TASK_QUEUE        *fpsctrlqueuelist,
    int                       *cmdFOUND,
    int                       *cmdOK,
    uint64_t                  *taskstatus,
    int                       *testcnt)
{
    if(*cmdFOUND)
        return;

    // exit
    if(strcmp(FPScommand, "exit") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 1)
        {
            functionparameter_outlog("ERROR", "COMMAND exit takes NBARGS = 0");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            fpsCTRLvar->exitloop = 1;
            functionparameter_outlog("DEBUG", "EXIT");
        }
        return;
    }

    // rescan
    if(strcmp(FPScommand, "rescan") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 1)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND rescan takes NBARGS = 0");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            functionparameter_scan_fps(fpsCTRLvar->mode,
                                       fpsCTRLvar->fpsnamemask,
                                       fps,
                                       keywnode,
                                       &fpsCTRLvar->NBkwn,
                                       &fpsCTRLvar->NBfps,
                                       &fpsCTRLvar->NBindex,
                                       0);
            functionparameter_outlog("DEBUG", "RESCAN");
        }
        return;
    }

    // cntinc
    if(strcmp(FPScommand, "cntinc") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND cntinc takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            (*testcnt)++;
            functionparameter_outlog("DEBUG",
                                     "TEST [%d] counter = %d",
                                     atoi(FPSarg0),
                                     *testcnt);
        }
        return;
    }

    // logsymlink
    if(strcmp(FPScommand, "logsymlink") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {

            functionparameter_outlog("ERROR",
                                     "COMMAND logsymlink takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            char logfname[STRINGMAXLEN_FULLFILENAME];
            getFPSlogfname(logfname);

            functionparameter_outlog("DEBUG",
                                     "CREATE SYM LINK %s <- %s",
                                     FPSarg0,
                                     logfname);

            if(symlink(logfname, FPSarg0) != 0)
            {
                PRINT_ERROR("symlink error %s %s", logfname, FPSarg0);
            }
        }
        return;
    }

    // queueprio
    if(strcmp(FPScommand, "queueprio") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 3)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND queueprio takes NBARGS = 2");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            int queue = atoi(FPSarg0);
            int prio  = atoi(FPSarg1);

            if((queue >= 0) && (queue < NB_FPSCTRL_TASKQUEUE_MAX))
            {
                fpsctrlqueuelist[queue].priority = prio;
                functionparameter_outlog("FPSCTRL",
                                         "%s",
                                         "QUEUE %d PRIO = %d",
                                         queue,
                                         prio);
            }
        }
        return;
    }
}

static void fps_cmd_handle_tmux(
    const char                *FPScommand,
    int                        nbword,
    FPS *fps,
    int                        fpsindex,
    int                       *cmdFOUND,
    int                       *cmdOK,
    uint64_t                  *taskstatus)
{
    if(*cmdFOUND)
        return;

    // tmuxstart
    if(strcmp(FPScommand, "tmuxstart") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "%s",
                                     "COMMAND tmuxstart takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            functionparameter_FPS_tmux_init(&fps[fpsindex]);

            functionparameter_outlog("FPSCTRL",
                                     "TMUXSTART %s",
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }

    // tmuxstop
    if(strcmp(FPScommand, "tmuxstop") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "%s",
                                     "COMMAND tmuxstop takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            functionparameter_FPS_tmux_kill(&fps[fpsindex]);

            functionparameter_outlog("FPSCTRL",
                                     "TMUXSTOP %s",
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }
}

static void fps_cmd_handle_conf(
    const char                *FPScommand,
    int                        nbword,
    FPS *fps,
    int                        fpsindex,
    int                       *cmdFOUND,
    int                       *cmdOK,
    uint64_t                  *taskstatus)
{
    if(*cmdFOUND)
        return;

    // confstart
    if(strcmp(FPScommand, "confstart") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "%s",
                                     "COMMAND confstart takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            functionparameter_CONFstart(&fps[fpsindex]);

            functionparameter_outlog("FPSCTRL",
                                     "CONFSTART %s",
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }

    // confstop
    if(strcmp(FPScommand, "confstop") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND confstop takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            functionparameter_CONFstop(&fps[fpsindex]);
            functionparameter_outlog("FPSCTRL",
                                     "CONFSTOP %s",
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }

    // confupdate
    if(strcmp(FPScommand, "confupdate") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND confupdate takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            fps[fpsindex].md->signal |=
                FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED;
            fps[fpsindex].md->signal |=
                FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

            functionparameter_outlog("FPSCTRL",
                                     "CONFUPDATE %s",
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }

    // confwupdate
    if(strcmp(FPScommand, "confwupdate") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog(
                "ERROR",
                "COMMAND confwupdate takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            int          looptry     = 1;
            int          looptrycnt  = 0;
            unsigned int timercnt    = 0;
            useconds_t   dt          = 100;
            unsigned int timercntmax = 10000;

            while(looptry == 1)
            {
                DEBUG_TRACEPOINT(" ");
                fps[fpsindex].md->signal |=
                    FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED;
                fps[fpsindex].md->signal |=
                    FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

                while(((fps[fpsindex].md->signal &
                        FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED)) &&
                        (timercnt < timercntmax))
                {
                    usleep(dt);
                    timercnt++;
                }
                usleep(dt);
                timercnt++;

                functionparameter_outlog("DEBUG",
                                         "CONFWUPDATE [%d] waited %d us on FPS %d %s. "
                                         "conferrcnt = %d",
                                         looptrycnt,
                                         dt * timercnt,
                                         fpsindex,
                                         fps[fpsindex].md->name,
                                         fps[fpsindex].md->conferrcnt);

                looptrycnt++;

                if(fps[fpsindex].md->conferrcnt == 0)
                {
                    looptry = 0;
                }

                if(timercnt > timercntmax)
                {
                    looptry = 0;
                }
            }

            *cmdOK = 1;
        }
        return;
    }
}

static void fps_cmd_handle_run(
    const char                *FPScommand,
    int                        nbword,
    FPS *fps,
    int                        fpsindex,
    int                       *cmdFOUND,
    int                       *cmdOK,
    uint64_t                  *taskstatus)
{
    if(*cmdFOUND)
        return;

    // runstart
    if(strcmp(FPScommand, "runstart") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND runstart takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            functionparameter_RUNstart(&fps[fpsindex]);

            functionparameter_outlog("FPSCTRL",
                                     "RUNSTART %s",
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }

    // runwait
    if(strcmp(FPScommand, "runwait") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND runwait takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");

            unsigned int timercnt    = 0;
            useconds_t   dt          = 10000;
            unsigned int timercntmax = 100000;

            while(((fps[fpsindex].md->status &
                    FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)) &&
                    (timercnt < timercntmax))
            {
                usleep(dt);
                timercnt++;
            }
            functionparameter_outlog("FPSCTRL",
                                     "RUNWAIT waited %d us on FPS %s",
                                     dt * timercnt,
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }

    // runstop
    if(strcmp(FPScommand, "runstop") == 0)
    {
        *cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND runstop takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            *cmdOK = 0;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            functionparameter_RUNstop(&fps[fpsindex]);
            functionparameter_outlog("FPSCTRL",
                                     "RUNSTOP %s",
                                     fps[fpsindex].md->name);
            *cmdOK = 1;
        }
        return;
    }
}

int functionparameter_FPSprocess_cmdline(
    char                 *FPScmdline,
    FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist,
    KEYWORD_TREE_NODE    *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar,
    FPS *fps,
    uint64_t                  *taskstatus
)
{
    int  fpsindex;
    long pindex;

    // break FPScmdline in words
    // [FPScommand] [FPSentryname]
    //
    char *pch;
    int   nbword = 0;
    int commandstringmaxlen = 200;
    char  FPScommand[commandstringmaxlen];

    int cmdOK    = 2; // 0 : failed, 1: OK
    int cmdFOUND = 0; // toggles to 1 when command has been found

    // first arg is always an FPS entry name
    char FPSentryname[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                           FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char FPScmdarg1[FUNCTION_PARAMETER_STRMAXLEN];

    char FPSarg0[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                      FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char FPSarg1[FUNCTION_PARAMETER_STRMAXLEN];
    char FPSarg2[FUNCTION_PARAMETER_STRMAXLEN];
    char FPSarg3[FUNCTION_PARAMETER_STRMAXLEN];

    char msgstring[STRINGMAXLEN_FPS_LOGMSG];
    char errmsgstring[STRINGMAXLEN_FPS_LOGMSG];
    char inputcmd[STRINGMAXLEN_FPS_CMDLINE];

    int inputcmdOK = 0; // 1 if command should be processed

    static int testcnt; // test counter to be incremented by cntinc command

    if(strlen(FPScmdline) > 0)  // only send command if non-empty
    {
        SNPRINTF_CHECK(inputcmd, STRINGMAXLEN_FPS_CMDLINE, "%s", FPScmdline);
        inputcmdOK = 1;
    }

    // don't process lines starting with # (comment)
    if(inputcmdOK == 1)
    {
        if(inputcmd[0] == '#')
        {
            inputcmdOK = 0;
        }
    }

    if(inputcmdOK == 0)
    {
        return (-1);
    }

    functionparameter_outlog("DEBUG", "CMDRCV [%s]", inputcmd);
    *taskstatus |= FPSTASK_STATUS_RECEIVED;

    DEBUG_TRACEPOINT(" ");

    if(strlen(inputcmd) > 1)
    {
        pch = strtok(inputcmd, " \t");
        snprintf(FPScommand, commandstringmaxlen, "%s", pch);
    }
    else
    {
        pch = NULL;
    }

    DEBUG_TRACEPOINT(" ");

    // Break command line into words
    //
    // output words are:
    //
    // FPScommand
    // FPSarg0
    // FPSarg1
    // FPSarg2
    // FPSarg3

    while(pch != NULL)
    {

        nbword++;
        pch = strtok(NULL, " \t");

        if(nbword == 1)  // first arg (0)
        {
            char *pos;
            snprintf(FPSarg0,
                     FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL,
                     "%s", pch);
            if((pos = strchr(FPSarg0, '\n')) != NULL)
            {
                *pos = '\0';
            }
        }

        if(nbword == 2)
        {
            char *pos;
            if(snprintf(FPSarg1, FUNCTION_PARAMETER_STRMAXLEN, "%s", pch) >=
                    FUNCTION_PARAMETER_STRMAXLEN)
            {
                printf("WARNING: string truncated\n");
                printf("STRING: %s\n", pch);
            }
            if((pos = strchr(FPSarg1, '\n')) != NULL)
            {
                *pos = '\0';
            }
        }

        if(nbword == 3)
        {
            char *pos;
            if(snprintf(FPSarg2, FUNCTION_PARAMETER_STRMAXLEN, "%s", pch) >=
                    FUNCTION_PARAMETER_STRMAXLEN)
            {
                printf("WARNING: string truncated\n");
                printf("STRING: %s\n", pch);
            }
            if((pos = strchr(FPSarg2, '\n')) != NULL)
            {
                *pos = '\0';
            }
        }

        if(nbword == 4)
        {
            char *pos;
            if(snprintf(FPSarg3, FUNCTION_PARAMETER_STRMAXLEN, "%s", pch) >=
                    FUNCTION_PARAMETER_STRMAXLEN)
            {
                printf("WARNING: string truncated\n");
                printf("STRING: %s\n", pch);
            }
            if((pos = strchr(FPSarg3, '\n')) != NULL)
            {
                *pos = '\0';
            }
        }
    }

    DEBUG_TRACEPOINT(" ");

    if(nbword == 0)
    {
        cmdFOUND = 1; // do nothing, proceed
        cmdOK    = 2;
    }

    fps_cmd_handle_sys(
        FPScommand, nbword, FPSarg0, FPSarg1,
        fpsCTRLvar, fps, keywnode, fpsctrlqueuelist,
        &cmdFOUND, &cmdOK, taskstatus, &testcnt);

    // From this point on, FPSarg0 is expected to be a FPS entry
    // so we resolve it and look for fps
    int kwnindex = -1;
    if(cmdFOUND == 0)
    {
        snprintf(FPSentryname,
                 sizeof(FPSentryname),
                 "%s", FPSarg0);
        snprintf(FPScmdarg1,
                 sizeof(FPScmdarg1),
                 "%s", FPSarg1);

        // look for entry, if found, kwnindex points to it
        if(nbword > 1)
        {
            //                printf("Looking for entry for %s\n", FPSentryname);

            int kwnindexscan = 0;
            while((kwnindex == -1) && (kwnindexscan < fpsCTRLvar->NBkwn))
            {
                if(strcmp(keywnode[kwnindexscan].keywordfull, FPSentryname) ==
                        0)
                {
                    kwnindex = kwnindexscan;
                }
                kwnindexscan++;
            }
        }

        if(kwnindex != -1)
        {
            fpsindex = keywnode[kwnindex].fpsindex;
            pindex   = keywnode[kwnindex].pindex;
            functionparameter_outlog("DEBUG",
                                     "FPS ENTRY FOUND : %-40s  %d %ld",
                                     FPSentryname,
                                     fpsindex,
                                     pindex);
        }
        else
        {
            functionparameter_outlog("ERROR",
                                     "FPS ENTRY NOT FOUND : %-40s",
                                     FPSentryname);
            *taskstatus |= FPSTASK_STATUS_ERR_NOFPS;
            cmdOK = 0;
        }
    }

    if(kwnindex != -1)  // if FPS has been found
    {

        fps_cmd_handle_tmux(FPScommand, nbword, fps, fpsindex, &cmdFOUND, &cmdOK, taskstatus);
        fps_cmd_handle_conf(FPScommand, nbword, fps, fpsindex, &cmdFOUND, &cmdOK, taskstatus);
        fps_cmd_handle_run(FPScommand, nbword, fps, fpsindex, &cmdFOUND, &cmdOK, taskstatus);


        // fpswfile : write FPS to file
        //
        if((cmdFOUND == 0) && (strcmp(FPScommand, "runstop") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR",
                                         "COMMAND fpswfile takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_SaveFPS2disk(&fps[fpsindex]);
                functionparameter_outlog("FPSCTRL",
                                         "FPSWFILE %s",
                                         fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }


        // fpsrm
        //
        if((cmdFOUND == 0) && (strcmp(FPScommand, "fpsrm") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR",
                                         "COMMAND fpsrm takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT("Removing fps number %d", fpsindex);
                functionparameter_FPSremove(&fps[fpsindex]);
                DEBUG_TRACEPOINT("Posting to fps log %s",
                                 fps[fpsindex].md->name);
                functionparameter_outlog("FPSCTRL",
                                         "FPSRM %s",
                                         fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }

        DEBUG_TRACEPOINT(" ");

        // exec
        //
        if((cmdFOUND == 0) && (strcmp(FPScommand, "exec") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR",
                                         "COMMAND exec takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                if(fps[fpsindex].parray[pindex].type == FPTYPE_EXECFILENAME)
                {
                    EXECUTE_SYSTEM_COMMAND_NOCHECK(
                        "tmux send-keys -t %s:run \"cd %s\" "
                        "C-m",
                        fps[fpsindex].md->name,
                        fps[fpsindex].md->workdir);
                    EXECUTE_SYSTEM_COMMAND_NOCHECK(
                        "tmux send-keys -t %s:run \"%s %s\" "
                        "C-m",
                        fps[fpsindex].md->name,
                        fps[fpsindex].parray[pindex].val.string[0],
                        fps[fpsindex].md->name);
                    cmdOK = 1;
                }
                else
                {
                    functionparameter_outlog(
                        "ERROR",
                        "COMMAND exec requires EXECFILENAME "
                        "type parameter");
                    *taskstatus |= FPSTASK_STATUS_ERR_ARGTYPE;
                    cmdOK = 0;
                }
            }
        }


        // setval
        //
        if((cmdFOUND == 0) && (strcmp(FPScommand, "setval") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 3)
            {
                SNPRINTF_CHECK(errmsgstring,
                               STRINGMAXLEN_FPS_LOGMSG,
                               "COMMAND setval takes NBARGS = 2");
                functionparameter_outlog("ERROR", "%s", errmsgstring);
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                int updated = 0;

                // Use the new consolidated API for parameter conversion, setting, and logging
                if (functionparameter_SetParamValue_fromString(&fps[fpsindex], pindex, FPScmdarg1) == 0)
                {
                    updated = 1;
                }
                else
                {
                    cmdOK = 0;
                    *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                    SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG, "argument conversion failed");
                    functionparameter_outlog("ERROR", "%s", errmsgstring);
                }

                // notify fpsCTRL that parameter has been updated
                if(updated == 1)
                {
                    cmdOK = 1;
                    functionparameter_WriteParameterToDisk(&fps[fpsindex],
                                                           pindex,
                                                           "setval",
                                                           "InputCommandFile");
                    fps[fpsindex].md->signal |=
                        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                }
                else
                {
                    cmdOK = 0;
                }
            }
        }


        // getval or fwrval
        //
        if((cmdFOUND == 0) && ((strcmp(FPScommand, "getval") == 0) ||
                               (strcmp(FPScommand, "fwrval") == 0)))
        {
            cmdFOUND = 1;
            cmdOK    = 0;

            if((strcmp(FPScommand, "getval") == 0) && (nbword != 2))
            {
                functionparameter_outlog("ERROR", "COMMAND getval NBARGS = 1");
            }
            else if((strcmp(FPScommand, "fwrval") == 0) && (nbword != 3))
            {
                functionparameter_outlog("ERROR", "COMMAND fwrval NBARGS = 2");
            }
            else
            {
                errno_t ret;
                ret = functionparameter_PrintParameter_ValueString(
                          &fps[fpsindex].parray[pindex],
                          msgstring,
                          STRINGMAXLEN_FPS_LOGMSG);

                if(ret == RETURN_SUCCESS)
                {
                    cmdOK = 1;
                }
                else
                {
                    cmdOK = 0;
                }


                if(cmdOK == 1)
                {
                    if(strcmp(FPScommand, "getval") == 0)
                    {
                        functionparameter_outlog("GETVAL", "%s", msgstring);
                    }
                    if(strcmp(FPScommand, "fwrval") == 0)
                    {

                        FILE *fpouttmp = fopen(FPScmdarg1, "a");
                        functionparameter_outlog_file("FWRVAL",
                                                      msgstring,
                                                      fpouttmp);
                        fclose(fpouttmp);

                        functionparameter_outlog("FWRVAL", "%s", msgstring);
                        char msgstring1[STRINGMAXLEN_FPS_LOGMSG];
                        SNPRINTF_CHECK(msgstring1,
                                       STRINGMAXLEN_FPS_LOGMSG,
                                       "WROTE to file %s",
                                       FPScmdarg1);
                        functionparameter_outlog("FWRVAL", "%s", msgstring1);
                    }
                }
            }
        }
    }

    if(cmdOK == 0)
    {
        SNPRINTF_CHECK(msgstring,
                       STRINGMAXLEN_FPS_LOGMSG,
                       "\"%s\" > %s",
                       FPScmdline,
                       errmsgstring);
        functionparameter_outlog("CMDFAIL", "%s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDFAIL;
    }

    if(cmdOK == 1)
    {
        SNPRINTF_CHECK(msgstring,
                       STRINGMAXLEN_FPS_LOGMSG,
                       "\"%s\"",
                       FPScmdline);
        functionparameter_outlog("DEBUG", "CMDOK %s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDOK;
    }

    if(cmdFOUND == 0)
    {
        SNPRINTF_CHECK(msgstring,
                       STRINGMAXLEN_FPS_LOGMSG,
                       "COMMAND NOT FOUND: %s",
                       FPScommand);
        functionparameter_outlog("ERROR", "%s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDNOTFOUND;
    }

    DEBUG_TRACEPOINT(" ");

    return fpsindex;
}


