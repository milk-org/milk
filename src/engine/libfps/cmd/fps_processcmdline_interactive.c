/**
 * @file    fps_processcmdline_interactive.c
 * @brief   FPS process command line interactive execution
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif


#include "fps.h"
#include "fps_CONFstart.h"
#include "fps_RUNstart.h"
#include "fps_FPSremove.h"
#include "fps_cmd_handlers.h"

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


/**
 * @brief Process a single FPS control command line.
 *
 * Parses the command line into words, looks up the
 * target FPS, and dispatches to the appropriate
 * handler (system, tmux, conf, run, or parameter
 * set). Updates task status flags for error
 * reporting.
 *
 * @param FPScmdline        Raw command line text
 * @param fpsctrlqueuelist  Task queue array
 * @param keywnode          FPS keyword tree root
 * @param fpsCTRLvar        fpsCTRL process state
 * @param fps               Connected FPS array
 * @param taskstatus        Status bitmap (updated)
 * @return 1 on success, 0 on command error
 */
int functionparameter_FPSprocess_cmdline(char                 *FPScmdline,
                                         FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist,
                                         KEYWORD_TREE_NODE    *keywnode,
                                         FPSCTRL_PROCESS_VARS *fpsCTRLvar,
                                         FPS                  *fps,
                                         uint64_t             *taskstatus)
{
    int  fpsindex = -1;
    long pindex;

    char       inputcmd[STRINGMAXLEN_FPS_CMDLINE];
    int        inputcmdOK = 0;
    static int testcnt;

    if (strlen(FPScmdline) > 0)
    {
        SNPRINTF_CHECK(inputcmd, STRINGMAXLEN_FPS_CMDLINE, "%s", FPScmdline);
        inputcmdOK = 1;
    }

    if (inputcmdOK == 1 && inputcmd[0] == '#')
    {
        inputcmdOK = 0;
    }

    if (inputcmdOK == 0)
    {
        return (-1);
    }

    functionparameter_outlog("DEBUG", "CMDRCV [%s]", inputcmd);
    *taskstatus |= FPSTASK_STATUS_RECEIVED;

    DEBUG_TRACEPOINT(" ");

    int  nbword          = 0;
    char FPScommand[200] = { 0 };
    char FPSarg0[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL] = {
        0
    };
    char FPSarg1[FUNCTION_PARAMETER_STRMAXLEN] = { 0 };

    if (strlen(inputcmd) > 1)
    {
        char *pch = strtok(inputcmd, " \t");
        snprintf(FPScommand, sizeof(FPScommand), "%s", pch);

        DEBUG_TRACEPOINT(" ");

        while (pch != NULL)
        {
            nbword++;
            pch = strtok(NULL, " \t");

            if (nbword == 1 && pch != NULL)
            {
                char *pos;
                snprintf(FPSarg0, sizeof(FPSarg0), "%s", pch);
                if ((pos = strchr(FPSarg0, '\n')) != NULL)
                {
                    *pos = '\0';
                }
            }

            if (nbword == 2 && pch != NULL)
            {
                char *pos;
                if (snprintf(FPSarg1, sizeof(FPSarg1), "%s", pch) >= (int) sizeof(FPSarg1))
                {
                    printf("WARNING: string truncated\n");
                    printf("STRING: %s\n", pch);
                }
                if ((pos = strchr(FPSarg1, '\n')) != NULL)
                {
                    *pos = '\0';
                }
            }
        }
    }

    DEBUG_TRACEPOINT(" ");

    if (nbword == 0)
    {
        return -1;
    }

    FPS_CMD_RESULT result;
    if (FPS_CMD_NOT_FOUND !=
        (result = fps_cmd_handle_sys_common(FPScommand, nbword, FPSarg0, FPSarg1, fpsCTRLvar, fps,
                                            keywnode, fpsctrlqueuelist, taskstatus, &testcnt)))
    {
        goto out;
    }

    int  kwnindex = -1;
    char FPScmdarg1[FUNCTION_PARAMETER_STRMAXLEN];
    char msgstring[STRINGMAXLEN_FPS_LOGMSG];
    char errmsgstring[STRINGMAXLEN_FPS_LOGMSG] = "";


    {
        char FPSentryname[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                          FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];

        snprintf(FPSentryname, sizeof(FPSentryname), "%s", FPSarg0);
        snprintf(FPScmdarg1, sizeof(FPScmdarg1), "%s", FPSarg1);

        if (nbword > 1)
        {
            int kwnindexscan = 0;
            while ((kwnindex == -1) && (kwnindexscan < fpsCTRLvar->NBkwn))
            {
                if (strcmp(keywnode[kwnindexscan].keywordfull, FPSentryname) == 0)
                {
                    kwnindex = kwnindexscan;
                }
                kwnindexscan++;
            }
        }

        if (kwnindex != -1)
        {
            fpsindex = keywnode[kwnindex].fpsindex;
            pindex   = keywnode[kwnindex].pindex;
            functionparameter_outlog("DEBUG", "FPS ENTRY FOUND : %-40s  %d %ld", FPSentryname,
                                     fpsindex, pindex);
        }
        else
        {
            functionparameter_outlog("ERROR", "FPS ENTRY NOT FOUND : %-40s", FPSentryname);
            *taskstatus |= FPSTASK_STATUS_ERR_NOFPS;
            result = FPS_CMD_FAIL;
            goto out;
        }
    }


    if (FPS_CMD_NOT_FOUND !=
        (result = fps_cmd_handle_tmux(FPScommand, nbword, fps, fpsindex, taskstatus)))
    {
        goto out;
    };
    if (FPS_CMD_NOT_FOUND !=
        (result = fps_cmd_handle_conf(FPScommand, nbword, fps, fpsindex, taskstatus)))
    {
        goto out;
    };
    if (FPS_CMD_NOT_FOUND !=
        (result = fps_cmd_handle_run(FPScommand, nbword, fps, fpsindex, taskstatus)))
    {
        goto out;
    };

    // fpswfile : write FPS to file
    if (strcmp(FPScommand, "fpswfile") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND fpswfile takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            result = FPS_CMD_FAIL;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            functionparameter_SaveFPS2disk(&fps[fpsindex]);
            functionparameter_outlog("FPSCTRL", "FPSWFILE %s", fps[fpsindex].md->name);
            result = FPS_CMD_OK;
        }
        goto out;
    }

    // fpsrm
    if (strcmp(FPScommand, "fpsrm") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND fpsrm takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            result = FPS_CMD_FAIL;
        }
        else
        {
            DEBUG_TRACEPOINT("Removing fps number %d", fpsindex);
            functionparameter_FPSremove(&fps[fpsindex]);
            DEBUG_TRACEPOINT("Posting to fps log %s", fps[fpsindex].md->name);
            functionparameter_outlog("FPSCTRL", "FPSRM %s", fps[fpsindex].md->name);
            result = FPS_CMD_OK;
        }
        goto out;
    }

    DEBUG_TRACEPOINT(" ");

    // exec
    if (strcmp(FPScommand, "exec") == 0)
    {
        if (nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND exec takes NBARGS = 1");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            result = FPS_CMD_FAIL;
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            if (fps[fpsindex].parray[pindex].type == FPTYPE_EXECFILENAME)
            {
                EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run \"cd %s\" "
                                               "C-m",
                                               fps[fpsindex].md->name, fps[fpsindex].md->workdir);
                EXECUTE_SYSTEM_COMMAND_NOCHECK("tmux send-keys -t %s:run \"%s %s\" "
                                               "C-m",
                                               fps[fpsindex].md->name,
                                               fps[fpsindex].parray[pindex].val.string[0],
                                               fps[fpsindex].md->name);
                result = FPS_CMD_OK;
            }
            else
            {
                functionparameter_outlog("ERROR", "COMMAND exec requires EXECFILENAME "
                                                  "type parameter");
                *taskstatus |= FPSTASK_STATUS_ERR_ARGTYPE;
                result = FPS_CMD_FAIL;
            }
        }
        goto out;
    }

    // setval
    if (strcmp(FPScommand, "setval") == 0)
    {
        if (nbword != 3)
        {
            SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG,
                           "COMMAND setval takes NBARGS = 2");
            functionparameter_outlog("ERROR", "%s", errmsgstring);
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            result = FPS_CMD_FAIL;
        }
        else
        {
            int updated = 0;

            if (functionparameter_SetParamValue_fromString(&fps[fpsindex], pindex, FPScmdarg1) == 0)
            {
                updated = 1;
            }
            else
            {
                *taskstatus |= FPSTASK_STATUS_ERR_TYPECONV;
                SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG, "argument conversion failed");
                functionparameter_outlog("ERROR", "%s", errmsgstring);
            }

            if (updated == 1)
            {
                functionparameter_WriteParameterToDisk(&fps[fpsindex], pindex, "setval",
                                                       "InputCommandFile");
                fps[fpsindex].md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                result = FPS_CMD_OK;
            }
            else
            {
                result = FPS_CMD_FAIL;
            }
        }
        goto out;
    }

    // getval or fwrval
    if (strcmp(FPScommand, "getval") == 0 || strcmp(FPScommand, "fwrval") == 0)
    {
        if (strcmp(FPScommand, "getval") == 0 && nbword != 2)
        {
            functionparameter_outlog("ERROR", "COMMAND getval NBARGS = 1");
            result = FPS_CMD_FAIL;
        }
        else if (strcmp(FPScommand, "fwrval") == 0 && nbword != 3)
        {
            functionparameter_outlog("ERROR", "COMMAND fwrval NBARGS = 2");
            result = FPS_CMD_FAIL;
        }
        else
        {
            char    valstring[STRINGMAXLEN_FPS_LOGMSG];
            errno_t ret;
            ret = functionparameter_PrintParameter_ValueString(&fps[fpsindex].parray[pindex],
                                                               valstring, STRINGMAXLEN_FPS_LOGMSG);

            if (ret != RETURN_SUCCESS)
            {
                result = FPS_CMD_FAIL;
            }
            else
            {
                result = FPS_CMD_OK;

                if (strcmp(FPScommand, "getval") == 0)
                {
                    functionparameter_outlog("GETVAL", "%s", valstring);
                }
                else // fwrval
                {
                    FILE *fpouttmp = fopen(FPScmdarg1, "a");
                    if (fpouttmp == NULL)
                    {
                        PRINT_ERROR("cannot open file \"%s\"", FPScmdarg1);
                        SNPRINTF_CHECK(errmsgstring, STRINGMAXLEN_FPS_LOGMSG,
                                       "cannot open output file %s", FPScmdarg1);
                        result = FPS_CMD_FAIL;
                    }
                    else
                    {
                        functionparameter_outlog_file("FWRVAL", valstring, fpouttmp);
                        fclose(fpouttmp);
                        functionparameter_outlog("FWRVAL", "%s", valstring);
                        char msgstring1[STRINGMAXLEN_FPS_LOGMSG];
                        SNPRINTF_CHECK(msgstring1, STRINGMAXLEN_FPS_LOGMSG, "WROTE to file %s",
                                       FPScmdarg1);
                        functionparameter_outlog("FWRVAL", "%s", msgstring1);
                    }
                }
            }
        }
        goto out;
    }

out:
    if (result == FPS_CMD_FAIL)
    {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "\"%s\" > %s", FPScmdline, errmsgstring);
        functionparameter_outlog("CMDFAIL", "%s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDFAIL;
    }
    else if (result == FPS_CMD_OK)
    {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "\"%s\"", FPScmdline);
        functionparameter_outlog("DEBUG", "CMDOK %s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDOK;
    }
    else if (result == FPS_CMD_NOT_FOUND)
    {
        SNPRINTF_CHECK(msgstring, STRINGMAXLEN_FPS_LOGMSG, "COMMAND NOT FOUND: %s", FPScommand);
        functionparameter_outlog("ERROR", "%s", msgstring);
        *taskstatus |= FPSTASK_STATUS_CMDNOTFOUND;
    }
    // FPS_CMD_OK_QUIET: no logging needed

    DEBUG_TRACEPOINT(" ");

    return fpsindex;
}
