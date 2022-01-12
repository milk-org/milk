/**
 * @file    fps_processcmdline.c
 * @brief   FPS process command line
 */


#include <stdlib.h>


#include "CommandLineInterface/CLIcore.h"


#include "fps_CONFstart.h"
#include "fps_CONFstop.h"

#include "fps_RUNstart.h"
#include "fps_RUNstop.h"

#include "fps_tmux.h"

#include "fps_FPSremove.h"

#include "fps_outlog.h"
#include "fps_paramvalue.h"
#include "fps_save2disk.h"

#include "fps_printparameter_valuestring.h"
#include "fps_WriteParameterToDisk.h"



/** @brief process command line
 *
 * ## Purpose
 *
 * Process command line.
 *
 * ## Commands
 *
 * - logsymlink  : create log sym link
 * - setval      : set parameter value
 * - getval      : get value, write to output log
 * - fwrval      : get value, write to file or fifo
 * - exec        : execute scripte (parameter must be FPTYPE_EXECFILENAME type)
 * - confupdate  : update configuration
 * - confwupdate : update configuration, wait for completion to proceed
 * - runstart    : start RUN process associated with parameter
 * - runstop     : stop RUN process associated with parameter
 * - fpsrm       : remove fps
 * - cntinc      : counter test to check fifo connection
 * - exit        : exit fpsCTRL tool
 *
 * - queueprio   : change queue priority
 *
 *
 */


int functionparameter_FPSprocess_cmdline(
    char *FPScmdline,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar,
    FUNCTION_PARAMETER_STRUCT *fps,
    uint64_t *taskstatus
)
{
    int  fpsindex;
    long pindex;

    // break FPScmdline in words
    // [FPScommand] [FPSentryname]
    //
    char *pch;
    int   nbword = 0;
    char  FPScommand[100];

    int   cmdOK = 2;    // 0 : failed, 1: OK
    int   cmdFOUND = 0; // toggles to 1 when command has been found

    // first arg is always an FPS entry name
    char  FPSentryname[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                            FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char  FPScmdarg1[FUNCTION_PARAMETER_STRMAXLEN];



    char  FPSarg0[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                       FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char  FPSarg1[FUNCTION_PARAMETER_STRMAXLEN];
    char  FPSarg2[FUNCTION_PARAMETER_STRMAXLEN];
    char  FPSarg3[FUNCTION_PARAMETER_STRMAXLEN];




    char msgstring[STRINGMAXLEN_FPS_LOGMSG];
    char errmsgstring[STRINGMAXLEN_FPS_LOGMSG];
    char inputcmd[STRINGMAXLEN_FPS_CMDLINE];


    int inputcmdOK = 0; // 1 if command should be processed


    static int testcnt; // test counter to be incremented by cntinc command


    if(strlen(FPScmdline) > 0)   // only send command if non-empty
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



    functionparameter_outlog("CMDRCV", "[%s]", inputcmd);
    *taskstatus |= FPSTASK_STATUS_RECEIVED;

    DEBUG_TRACEPOINT(" ");

    if(strlen(inputcmd) > 1)
    {
        pch = strtok(inputcmd, " \t");
        sprintf(FPScommand, "%s", pch);
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

        if(nbword == 1)   // first arg (0)
        {
            char *pos;
            sprintf(FPSarg0, "%s", pch);
            if((pos = strchr(FPSarg0, '\n')) != NULL)
            {
                *pos = '\0';
            }

        }

        if(nbword == 2)
        {
            char *pos;
            if(snprintf(FPSarg1, FUNCTION_PARAMETER_STRMAXLEN, "%s",
                        pch) >= FUNCTION_PARAMETER_STRMAXLEN)
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
            if(snprintf(FPSarg2, FUNCTION_PARAMETER_STRMAXLEN, "%s",
                        pch) >= FUNCTION_PARAMETER_STRMAXLEN)
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
            if(snprintf(FPSarg3, FUNCTION_PARAMETER_STRMAXLEN, "%s",
                        pch) >= FUNCTION_PARAMETER_STRMAXLEN)
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
        cmdFOUND = 1;   // do nothing, proceed
        cmdOK = 2;
    }






    // Handle commands for which FPSarg0 is NOT an FPS entry


    // cntinc
    if((cmdFOUND == 0)
            && (strcmp(FPScommand, "exit") == 0))
    {
        cmdFOUND = 1;
        if(nbword != 1)
        {
            functionparameter_outlog("ERROR", "COMMAND cntinc takes NBARGS = 0");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            cmdOK = 0;
        }
        else
        {
            fpsCTRLvar->exitloop = 1;
            functionparameter_outlog("INFO", "EXIT");
        }
    }




    // cntinc
    if((cmdFOUND == 0)
            && (strcmp(FPScommand, "cntinc") == 0))
    {
        cmdFOUND = 1;
        if(nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND cntinc takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            cmdOK = 0;
        }
        else
        {
            testcnt ++;
            functionparameter_outlog("INFO", "TEST [%d] counter = %d", atoi(FPSarg0),
                                     testcnt);
        }
    }







    // logsymlink
    if((cmdFOUND == 0)
            && (strcmp(FPScommand, "logsymlink") == 0))
    {
        cmdFOUND = 1;
        if(nbword != 2)
        {

            functionparameter_outlog("ERROR", "COMMAND logsymlink takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            cmdOK = 0;
        }
        else
        {
            char logfname[STRINGMAXLEN_FULLFILENAME];
            getFPSlogfname(logfname);

            functionparameter_outlog("INFO", "CREATE SYM LINK %s <- %s", FPSarg0, logfname);

            if(symlink(logfname, FPSarg0) != 0)
            {
                PRINT_ERROR("symlink error %s %s", logfname, FPSarg0);
            }

        }
    }




    // queueprio
    if((cmdFOUND == 0)
            && (strcmp(FPScommand, "queueprio") == 0))
    {
        cmdFOUND = 1;
        if(nbword != 3)
        {
            functionparameter_outlog("ERROR", "COMMAND queueprio takes NBARGS = 2");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            cmdOK = 0;
        }
        else
        {
            int queue = atoi(FPSarg0);
            int prio = atoi(FPSarg1);

            if((queue >= 0) && (queue < NB_FPSCTRL_TASKQUEUE_MAX))
            {
                fpsctrlqueuelist[queue].priority = prio;
                functionparameter_outlog("INFO", "%s", "QUEUE %d PRIO = %d", queue, prio);
            }
        }
    }









    // From this point on, FPSarg0 is expected to be a FPS entry
    // so we resolve it and look for fps
    int kwnindex = -1;
    if(cmdFOUND == 0)
    {
        strcpy(FPSentryname, FPSarg0);
        strcpy(FPScmdarg1, FPSarg1);


        // look for entry, if found, kwnindex points to it
        if(nbword > 1)
        {
            //                printf("Looking for entry for %s\n", FPSentryname);

            int kwnindexscan = 0;
            while((kwnindex == -1) && (kwnindexscan < fpsCTRLvar->NBkwn))
            {
                if(strcmp(keywnode[kwnindexscan].keywordfull, FPSentryname) == 0)
                {
                    kwnindex = kwnindexscan;
                }
                kwnindexscan ++;
            }
        }

        //            sprintf(msgstring, "nbword = %d  cmdOK = %d   kwnindex = %d",  nbword, cmdOK, kwnindex);
        //            functionparameter_outlog("INFO", "%s", msgstring);


        if(kwnindex != -1)
        {
            fpsindex = keywnode[kwnindex].fpsindex;
            pindex = keywnode[kwnindex].pindex;
            functionparameter_outlog("INFO", "FPS ENTRY FOUND : %-40s  %d %ld",
                                     FPSentryname, fpsindex, pindex);
        }
        else
        {
            functionparameter_outlog("ERROR", "FPS ENTRY NOT FOUND : %-40s", FPSentryname);
            *taskstatus |= FPSTASK_STATUS_ERR_NOFPS;
            cmdOK = 0;
        }
    }



    if(kwnindex != -1)   // if FPS has been found
    {

        // tmuxstart
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "tmuxstart") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "%s", "COMMAND tmuxstart takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_FPS_tmux_init(&fps[fpsindex]);

                functionparameter_outlog("TMUXSTART", "Init tmux session %d %s",
                                         fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }

        // tmuxstop
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "tmuxstop") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "%s", "COMMAND tmuxstop takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_FPS_tmux_kill(&fps[fpsindex]);

                functionparameter_outlog("TMUXSTOP", "Init tmux session %d %s",
                                         fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }

        // confstart
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "confstart") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "%s", "COMMAND confstart takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_CONFstart(&fps[fpsindex]);

                functionparameter_outlog("CONFSTART", "start CONF process %d %s",
                                         fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }


        // confstop
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "confstop") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND confstop takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_CONFstop(&fps[fpsindex]);
                functionparameter_outlog("CONFSTOP", "stop CONF process %d %s",
                                         fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }










        // confupdate

        DEBUG_TRACEPOINT(" ");
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "confupdate") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND confupdate takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                fps[fpsindex].md->signal |=
                    FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED; // update status: check waiting to be done
                fps[fpsindex].md->signal |=
                    FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // request an update

                functionparameter_outlog("CONFUPDATE", "update CONF process %d %s",
                                         fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }





        // confwupdate
        // Wait until update is cleared
        // if not successful, retry until time lapsed

        DEBUG_TRACEPOINT(" ");
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "confwupdate") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND confwupdate takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                int looptry = 1;
                int looptrycnt = 0;
                unsigned int timercnt = 0;
                useconds_t dt = 100;
                unsigned int timercntmax = 10000; // 1 sec max

                while(looptry == 1)
                {

                    DEBUG_TRACEPOINT(" ");
                    fps[fpsindex].md->signal |=
                        FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED; // update status: check waiting to be done
                    fps[fpsindex].md->signal |=
                        FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // request an update

                    while(((fps[fpsindex].md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED))
                            && (timercnt < timercntmax))
                    {
                        usleep(dt);
                        timercnt++;
                    }
                    usleep(dt);
                    timercnt++;

                    functionparameter_outlog("CONFWUPDATE",
                                             "[%d] waited %d us on FPS %d %s. conferrcnt = %d",
                                             looptrycnt,
                                             dt * timercnt,
                                             fpsindex,
                                             fps[fpsindex].md->name,
                                             fps[fpsindex].md->conferrcnt);

                    looptrycnt++;

                    if(fps[fpsindex].md->conferrcnt == 0)   // no error ! we can proceed
                    {
                        looptry = 0;
                    }

                    if(timercnt > timercntmax)    // ran out of time ... giving up
                    {
                        looptry = 0;
                    }


                }

                cmdOK = 1;
            }
        }




        // runstart
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "runstart") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND runstart takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_RUNstart(&fps[fpsindex]);

                functionparameter_outlog("RUNSTART", "start RUN process %d %s",
                                         fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;

            }
        }



        // runwait
        // wait until run process is completed

        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "runwait") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND runwait takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");

                unsigned int timercnt = 0;
                useconds_t dt = 10000;
                unsigned int timercntmax = 100000; // 10000 sec max

                while(((fps[fpsindex].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN))
                        && (timercnt < timercntmax))
                {
                    usleep(dt);
                    timercnt++;
                }
                functionparameter_outlog("RUNWAIT", "waited %d us on FPS %d %s",
                                         dt * timercnt, fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }



        // runstop

        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "runstop") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND runstop takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                functionparameter_RUNstop(&fps[fpsindex]);
                functionparameter_outlog("RUNSTOP", "stop RUN process %d %s",
                                         fpsindex, fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }




        // fpsrm

        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "fpsrm") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND fpsrm takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT("Removing fps number %d", fpsindex);
                functionparameter_FPSremove(&fps[fpsindex]);
                DEBUG_TRACEPOINT("Posting to fps log %s", fps[fpsindex].md->name);
                functionparameter_outlog("FPSRM", "FPS remove %d %s", fpsindex,
                                         fps[fpsindex].md->name);
                cmdOK = 1;
            }
        }







        DEBUG_TRACEPOINT(" ");




        // exec
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "exec") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 2)
            {
                functionparameter_outlog("ERROR", "COMMAND exec takes NBARGS = 1");
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                DEBUG_TRACEPOINT(" ");
                if(fps[fpsindex].parray[pindex].type == FPTYPE_EXECFILENAME)
                {
                    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"cd %s\" C-m",
                                           fps[fpsindex].md->name, fps[fpsindex].md->workdir);
                    EXECUTE_SYSTEM_COMMAND("tmux send-keys -t %s:run \"%s %s\" C-m",
                                           fps[fpsindex].md->name, fps[fpsindex].parray[pindex].val.string[0],
                                           fps[fpsindex].md->name);
                    cmdOK = 1;
                }
                else
                {
                    functionparameter_outlog("ERROR",
                                             "COMMAND exec requires EXECFILENAME type parameter");
                    *taskstatus |= FPSTASK_STATUS_ERR_ARGTYPE;
                    cmdOK = 0;
                }
            }
        }



        // setval
        if((cmdFOUND == 0)
                && (strcmp(FPScommand, "setval") == 0))
        {
            cmdFOUND = 1;
            if(nbword != 3)
            {
                SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG,
                               "COMMAND setval takes NBARGS = 2");
                functionparameter_outlog("ERROR", "%s", errmsgstring);
                *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
                cmdOK = 0;
            }
            else
            {
                int updated = 0;

                switch(fps[fpsindex].parray[pindex].type)
                {



                    case FPTYPE_INT32:
                    {
                        char *endptr;
                        long valn = strtol(FPScmdarg1, &endptr, 10);

                        if(*endptr == '\0')
                        {
                            // OK
                            if(functionparameter_SetParamValue_INT32(&fps[fpsindex], FPSentryname,
                                    valn) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s INT32      %ld",
                                                         FPSentryname, valn);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG, "argument is not INT32");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;



                    case FPTYPE_UINT32:
                    {
                        char *endptr;
                        long valn = strtol(FPScmdarg1, &endptr, 10);

                        if((*endptr == '\0') && (valn >= 0))
                        {
                            // OK
                            if(functionparameter_SetParamValue_UINT32(&fps[fpsindex], FPSentryname,
                                    valn) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s UINT32     %ld",
                                                         FPSentryname, valn);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG, "argument is not UINT32");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;



                    case FPTYPE_INT64:
                    {
                        char *endptr;
                        long valn = strtol(FPScmdarg1, &endptr, 10);

                        if(*endptr == '\0')
                        {
                            // OK
                            if(functionparameter_SetParamValue_INT64(&fps[fpsindex], FPSentryname,
                                    valn) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s INT64      %ld",
                                                         FPSentryname, valn);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG, "argument is not INT64");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;



                    case FPTYPE_UINT64:
                    {
                        char *endptr;
                        long valn = strtol(FPScmdarg1, &endptr, 10);

                        if((*endptr == '\0') && (valn >= 0))
                        {
                            // OK
                            if(functionparameter_SetParamValue_UINT64(&fps[fpsindex], FPSentryname,
                                    valn) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s UINT64     %ld",
                                                         FPSentryname, valn);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG, "argument is not UINT64");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;



                    case FPTYPE_FLOAT64:
                    {
                        char *endptr;
                        double valf64 = strtod(FPScmdarg1, &endptr);

                        if(*endptr == '\0')
                        {
                            // OK
                            if(functionparameter_SetParamValue_FLOAT64(&fps[fpsindex], FPSentryname,
                                    valf64) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s FLOAT64     %f",
                                                         FPSentryname, valf64);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG,
                                           "argument is not FLOAT64");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;



                    case FPTYPE_FLOAT32:
                    {
                        char *endptr;
                        double valf32 = strtof(FPScmdarg1, &endptr);

                        if(*endptr == '\0')
                        {
                            // OK
                            if(functionparameter_SetParamValue_FLOAT32(&fps[fpsindex], FPSentryname,
                                    valf32) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s FLOAT32     %f",
                                                         FPSentryname, valf32);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG,
                                           "argument is not FLOAT32");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;



                    case FPTYPE_PID:
                    {
                        char *endptr;
                        long valn = strtol(FPScmdarg1, &endptr, 10);

                        if(*endptr == '\0')
                        {
                            // OK
                            if(functionparameter_SetParamValue_INT64(&fps[fpsindex], FPSentryname,
                                    valn) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s PID      %ld",
                                                         FPSentryname, valn);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG, "argument is not INT64");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;

                    case FPTYPE_TIMESPEC:
                    {
                        char *endptr;
                        double valf32 = strtof(FPScmdarg1, &endptr);

                        if(*endptr == '\0')
                        {
                            // OK
                            if(functionparameter_SetParamValue_TIMESPEC(&fps[fpsindex], FPSentryname,
                                    valf32) == EXIT_SUCCESS)
                            {
                                updated = 1;

                                functionparameter_outlog("SETVAL", "%-40s TIMESPEC     %f",
                                                         FPSentryname, valf32);
                            }
                        }
                        else
                        {
                            *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                            cmdOK = 0;
                            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG,
                                           "argument is not FLOAT->TIMESPEC");
                            functionparameter_outlog("ERROR", "%s", errmsgstring);
                        }
                    }
                    break;

                    case FPTYPE_FILENAME:
                        if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname,
                                FPScmdarg1) == EXIT_SUCCESS)
                        {
                            updated = 1;
                        }
                        functionparameter_outlog("SETVAL", "%-40s FILENAME   %s",
                                                 FPSentryname, FPScmdarg1);
                        break;

                    case FPTYPE_FITSFILENAME:
                        if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname,
                                FPScmdarg1) == EXIT_SUCCESS)
                        {
                            updated = 1;
                        }
                        functionparameter_outlog("SETVAL", "%-40s FITSFILENAME   %s",
                                                 FPSentryname, FPScmdarg1);
                        break;

                    case FPTYPE_EXECFILENAME:
                        if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname,
                                FPScmdarg1) == EXIT_SUCCESS)
                        {
                            updated = 1;
                        }
                        functionparameter_outlog("SETVAL", "%-40s EXECFILENAME   %s",
                                                 FPSentryname, FPScmdarg1);
                        break;

                    case FPTYPE_DIRNAME:
                        if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname,
                                FPScmdarg1) == EXIT_SUCCESS)
                        {
                            updated = 1;
                        }
                        functionparameter_outlog("SETVAL", "%-40s DIRNAME    %s",
                                                 FPSentryname, FPScmdarg1);
                        break;

                    case FPTYPE_STREAMNAME:
                        if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname,
                                FPScmdarg1) == EXIT_SUCCESS)
                        {
                            updated = 1;
                        }
                        functionparameter_outlog("SETVAL", "%-40s STREAMNAME %s",
                                                 FPSentryname, FPScmdarg1);
                        break;

                    case FPTYPE_STRING:
                        if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname,
                                FPScmdarg1) == EXIT_SUCCESS)
                        {
                            updated = 1;
                        }
                        functionparameter_outlog("SETVAL", "%-40s STRING     %s",
                                                 FPSentryname, FPScmdarg1);
                        break;

                    case FPTYPE_ONOFF:
                        if(strncmp(FPScmdarg1, "ON", 2) == 0)
                        {
                            if(functionparameter_SetParamValue_ONOFF(&fps[fpsindex], FPSentryname,
                                    1) == EXIT_SUCCESS)
                            {
                                updated = 1;
                            }
                            functionparameter_outlog("SETVAL", "%-40s ONOFF      ON",
                                                     FPSentryname);
                        }
                        if(strncmp(FPScmdarg1, "OFF", 3) == 0)
                        {
                            if(functionparameter_SetParamValue_ONOFF(&fps[fpsindex], FPSentryname,
                                    0) == EXIT_SUCCESS)
                            {
                                updated = 1;
                            }
                            functionparameter_outlog("SETVAL", "%-40s ONOFF      OFF",
                                                     FPSentryname);
                        }
                        break;


                    case FPTYPE_FPSNAME:
                        if(functionparameter_SetParamValue_STRING(&fps[fpsindex], FPSentryname,
                                FPScmdarg1) == EXIT_SUCCESS)
                        {
                            updated = 1;
                        }
                        functionparameter_outlog("SETVAL", "%-40s FPSNAME   %s",
                                                 FPSentryname, FPScmdarg1);
                        break;

                    default:
                        SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG,
                                       "argument type not recognized");
                        functionparameter_outlog("ERROR", "%s", errmsgstring);
                        *taskstatus |= FPSTASK_STATUS_ERR_ARGTYPE;
                        break;

                }

                // notify fpsCTRL that parameter has been updated
                if(updated == 1)
                {
                    cmdOK = 1;
                    functionparameter_WriteParameterToDisk(&fps[fpsindex], pindex, "setval",
                                                           "InputCommandFile");
                    fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                }
                else
                {
                    cmdOK = 0;
                }

            }
        }





        // getval or fwrval
        if((cmdFOUND == 0)
                && ((strcmp(FPScommand, "getval") == 0) || (strcmp(FPScommand, "fwrval") == 0)))
        {
            cmdFOUND = 1;
            cmdOK = 0;

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
                          &fps[fpsindex].parray[pindex], msgstring, STRINGMAXLEN_FPS_LOGMSG);

                if(ret == RETURN_SUCCESS)
                {
                    cmdOK = 1;
                }
                else
                {
                    cmdOK = 0;
                }

                /*
                switch(fps[fpsindex].parray[pindex].type)
                {

                    case FPTYPE_INT64:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s INT64      %ld %ld %ld %ld",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.l[0],
                            fps[fpsindex].parray[pindex].val.l[1],
                            fps[fpsindex].parray[pindex].val.l[2],
                            fps[fpsindex].parray[pindex].val.l[3]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_FLOAT64:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s FLOAT64    %f %f %f %f",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.f[0],
                            fps[fpsindex].parray[pindex].val.f[1],
                            fps[fpsindex].parray[pindex].val.f[2],
                            fps[fpsindex].parray[pindex].val.f[3]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_FLOAT32:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s FLOAT32    %f %f %f %f",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.s[0],
                            fps[fpsindex].parray[pindex].val.s[1],
                            fps[fpsindex].parray[pindex].val.s[2],
                            fps[fpsindex].parray[pindex].val.s[3]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_PID:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s PID        %ld",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.l[0]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_TIMESPEC:
                        //
                        break;

                    case FPTYPE_FILENAME:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s FILENAME   %s",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.string[0]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_FITSFILENAME:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s FITSFILENAME   %s",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.string[0]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_EXECFILENAME:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s EXECFILENAME   %s",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.string[0]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_DIRNAME:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s DIRNAME    %s",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.string[0]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_STREAMNAME:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s STREAMNAME %s",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.string[0]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_STRING:
                        SNPRINTF_CHECK(
                            msgstring,
                            STRINGMAXLEN_FPS_LOGMSG,
                            "%-40s STRING     %s",
                            FPSentryname,
                            fps[fpsindex].parray[pindex].val.string[0]);
                        cmdOK = 1;
                        break;

                    case FPTYPE_ONOFF:
                        if(fps[fpsindex].parray[pindex].fpflag & FPFLAG_ONOFF)
                        {
                            SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s ONOFF      ON",
                                           FPSentryname);
                        }
                        else
                        {
                            SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s ONOFF      OFF",
                                           FPSentryname);
                        }
                        cmdOK = 1;
                        break;


                    case FPTYPE_FPSNAME:
                        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "%-40s FPSNAME   %s",
                                       FPSentryname, fps[fpsindex].parray[pindex].val.string[0]);
                        cmdOK = 1;
                        break;

                }

                */

                if(cmdOK == 1)
                {
                    if(strcmp(FPScommand, "getval") == 0)
                    {
                        functionparameter_outlog("GETVAL", "%s", msgstring);
                    }
                    if(strcmp(FPScommand, "fwrval") == 0)
                    {

                        FILE *fpouttmp = fopen(FPScmdarg1, "a");
                        functionparameter_outlog_file("FWRVAL", msgstring, fpouttmp);
                        fclose(fpouttmp);

                        functionparameter_outlog("FWRVAL", "%s", msgstring);
                        char msgstring1[STRINGMAXLEN_FPS_LOGMSG];
                        SNPRINTF_CHECK(msgstring1, STRINGMAXLEN_FPS_LOGMSG, "WROTE to file %s",
                                       FPScmdarg1);
                        functionparameter_outlog("FWRVAL", "%s", msgstring1);
                    }
                }

            }
        }


    }


    if(cmdOK == 0)
    {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "\"%s\" > %s", FPScmdline,
                       errmsgstring);
        functionparameter_outlog("CMDFAIL", "%s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDFAIL;
    }

    if(cmdOK == 1)
    {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "\"%s\"", FPScmdline);
        functionparameter_outlog("CMDOK", "%s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDOK;
    }

    if(cmdFOUND == 0)
    {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND NOT FOUND: %s",
                       FPScommand);
        functionparameter_outlog("ERROR", "%s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDNOTFOUND;
    }


    DEBUG_TRACEPOINT(" ");


    return fpsindex;
}





int functionparameter_FPSprocess_cmdfile(
    char *infname,
    FUNCTION_PARAMETER_STRUCT *fps,
    KEYWORD_TREE_NODE *keywnode,
    FPSCTRL_TASK_QUEUE *fpsctrlqueuelist,
    FPSCTRL_PROCESS_VARS *fpsCTRLvar
)
{
    FILE *fpinputcmd;
    fpinputcmd = fopen(infname, "r");

    if(fpinputcmd != NULL)
    {
        char *FPScmdline = NULL;
        size_t len = 0;
        ssize_t read;

        while((read = getline(&FPScmdline, &len, fpinputcmd)) != -1)
        {
            uint64_t taskstatus = 0;
            printf("Processing line : %s\n", FPScmdline);
            functionparameter_FPSprocess_cmdline(FPScmdline, fpsctrlqueuelist, keywnode,
                                                 fpsCTRLvar, fps, &taskstatus);
        }
        fclose(fpinputcmd);
    }

    return RETURN_SUCCESS;
}
