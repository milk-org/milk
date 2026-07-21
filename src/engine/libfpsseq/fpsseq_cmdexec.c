// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_processcmdline.c
 * @brief   FPS process command line
 */


#include "fpsseq.h"

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
 * milkseq_cmd_handle_sys - Handle system-level sequencer commands
 * @FPScommand:   Command verb (first word of the command line)
 * @FPScmdline:   Full command line string
 * @nbword:       Number of words in the command line
 * @FPSarg0:      First argument token
 * @FPSarg1:      Second argument token
 * @cmdindex:     Task index in the sequencer task list
 * @state:        Sequencer state
 * @fpsCTRLvar:   TUI-level process variables
 * @fps:          Array of all FPS entries
 * @keywnode:     Keyword tree root
 * @cmdFOUND:     Set to 1 if this handler matched the command
 * @cmdOK:        Set to 1 on success, 0 on failure
 * @taskstatus:   OR-ed with error/status flags
 * @testcnt:      Counter incremented by the "cntinc" command
 *
 * Dispatches: exit, rescan, cntinc, logsymlink, queueprio,
 * seq_send, wait_seq.
 */
static FPS_CMD_RESULT milkseq_cmd_handle_sys(const char           *FPScommand,
                                             const char           *FPScmdline,
                                             int                   nbword,
                                             const char           *FPSarg0,
                                             const char           *FPSarg1,
                                             uint32_t              cmdindex,
                                             MILKSEQ_STATE        *state,
                                             FPSCTRL_PROCESS_VARS *fpsCTRLvar,
                                             FPS                  *fps,
                                             KEYWORD_TREE_NODE    *keywnode,
                                             uint64_t             *taskstatus,
                                             int                  *testcnt)
{
    FPS_CMD_RESULT result;
    if (FPS_CMD_NOT_FOUND !=
        (result = fps_cmd_handle_sys_common(FPScommand, nbword, FPSarg0, FPSarg1, fpsCTRLvar, fps,
                                            keywnode, state->queuelist, taskstatus, testcnt)))
    {
        return result;
    }

    // seq_send
    if (strcmp(FPScommand, "seq_send") == 0)
    {
        if (nbword < 3)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND seq_send requires 2+ arguments (<seqname> <cmd...>)");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        {
            char fifo_path[4096];
            snprintf(fifo_path, sizeof(fifo_path), "/tmp/milkseq.%s.fifo", FPSarg0);
            FILE *fp = fopen(fifo_path, "w");
            if (fp == NULL)
            {
                functionparameter_outlog("ERROR", "seq_send: cannot open FIFO %s", fifo_path);
                return FPS_CMD_FAIL;
            }
            const char *cmd_start = strstr(FPScmdline, FPSarg1);
            if (cmd_start == NULL)
            {
                fclose(fp);
                return FPS_CMD_FAIL;
            }
            fprintf(fp, "%s\n", cmd_start);
            fflush(fp);
            fclose(fp);
        }
        return FPS_CMD_OK;
    }

    // wait_seq
    if (strcmp(FPScommand, "wait_seq") == 0)
    {
        if (nbword != 3 || strcmp(FPSarg1, "idle") != 0)
        {
            functionparameter_outlog("ERROR",
                                     "COMMAND wait_seq requires format: wait_seq <seqname> idle");
            *taskstatus |= FPSTASK_STATUS_ERR_NBARG;
            return FPS_CMD_FAIL;
        }
        state->tasklist[cmdindex].flag |= MILKSEQ_TASKFLAG_WAITSEQ_IDLE;
        *taskstatus |= FPSTASK_STATUS_RUNNING;
        return FPS_CMD_OK;
    }

    return FPS_CMD_NOT_FOUND;
}

/* fps_cmd_handle_tmux, fps_cmd_handle_conf, fps_cmd_handle_run
 * are defined in fps_cmd_handlers.h */

/**
 * milkseq_exec_cmd - Parse and execute one sequencer command
 * @cmdindex:    Task index in state->tasklist
 * @state:       Sequencer state mapped from SHM
 * @fps:         Array of all FPS entries
 * @keywnode:    Keyword tree root for FPS name resolution
 * @fpsCTRLvar:  TUI-level process variables (exitloop, scan state)
 * @taskstatus:  Output flags OR-ed with task status/error bits
 *
 * Tokenizes the command string into words, resolves the FPS
 * entry name (arg0) via the keyword tree, then dispatches to
 * one of the handler groups: system commands (exit, rescan,
 * queueprio), tmux lifecycle, configuration, run lifecycle,
 * or parameter access (setval, getval, fwrval, exec, fpswfile).
 *
 * Return: FPS index of the parameter accessed, or -1 if none
 */
int milkseq_exec_cmd(uint32_t              cmdindex,
                     MILKSEQ_STATE        *state,
                     FPS                  *fps,
                     KEYWORD_TREE_NODE    *keywnode,
                     FPSCTRL_PROCESS_VARS *fpsCTRLvar,
                     uint64_t             *taskstatus)
{
    const char *FPScmdline = state->tasklist[cmdindex].cmdstring;
    int         fpsindex   = -1;
    long        pindex;

    static int testcnt;

    // Validate and copy command line; reject empty or comment lines
    char inputcmd[STRINGMAXLEN_FPS_CMDLINE];
    if (strlen(FPScmdline) > 0 && inputcmd[0] != '#')
    {
        SNPRINTF_CHECK(inputcmd, STRINGMAXLEN_FPS_CMDLINE, "%s", FPScmdline);
    }
    else
    {
        return -1;
    }

    // Word-parse buffers: command, first arg, second arg
    char *pch;
    int   nbword              = 0;
    int   commandstringmaxlen = 200;
    char  FPScommand[commandstringmaxlen];
    char  FPSarg0[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char  FPSarg1[FUNCTION_PARAMETER_STRMAXLEN];

    functionparameter_outlog("DEBUG", "CMDRCV [%s]", inputcmd);
    *taskstatus |= FPSTASK_STATUS_RECEIVED;

    DEBUG_TRACEPOINT(" ");

    if (strlen(inputcmd) > 1)
    {
        pch = strtok(inputcmd, " \t");
        snprintf(FPScommand, commandstringmaxlen, "%s", pch);
    }
    else
    {
        pch = NULL;
    }

    DEBUG_TRACEPOINT(" ");

    while (pch != NULL)
    {
        nbword++;
        pch = strtok(NULL, " \t");

        if (nbword == 1)
        {
            char *pos;
            snprintf(FPSarg0,
                     FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL,
                     "%s", pch);
            if ((pos = strchr(FPSarg0, '\n')) != NULL)
            {
                *pos = '\0';
            }
        }

        if (nbword == 2)
        {
            char *pos;
            if (snprintf(FPSarg1, FUNCTION_PARAMETER_STRMAXLEN, "%s", pch) >=
                FUNCTION_PARAMETER_STRMAXLEN)
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

    DEBUG_TRACEPOINT(" ");

    if (nbword == 0)
    {
        return (-1);
    }

    FPS_CMD_RESULT result;

    if (FPS_CMD_NOT_FOUND !=
        (result = milkseq_cmd_handle_sys(FPScommand, FPScmdline, nbword, FPSarg0, FPSarg1, cmdindex,
                                         state, fpsCTRLvar, fps, keywnode, taskstatus, &testcnt)))
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
            errno_t ret;
            ret = functionparameter_PrintParameter_ValueString(&fps[fpsindex].parray[pindex],
                                                               msgstring, STRINGMAXLEN_FPS_LOGMSG);

            if (ret != RETURN_SUCCESS)
            {
                result = FPS_CMD_FAIL;
            }
            else
            {
                result = FPS_CMD_OK;

                if (strcmp(FPScommand, "getval") == 0)
                {
                    functionparameter_outlog("GETVAL", "%s", msgstring);
                }
                else // fwrval
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
        goto out;
    }

    // wait_fps
    if (strcmp(FPScommand, "wait_fps") == 0)
    {
        if (nbword != 3)
        {
            functionparameter_outlog(
                "ERROR", "COMMAND wait_fps requires 2 arguments (<fpsname> <running|norun>)");
            result = FPS_CMD_FAIL;
        }
        else
        {
            state->tasklist[cmdindex].fpsindex = fpsindex;

            if (strcmp(FPSarg1, "running") == 0)
            {
                state->tasklist[cmdindex].flag |= MILKSEQ_TASKFLAG_WAITFPS_RUNNING;
                *taskstatus |= FPSTASK_STATUS_RUNNING;
                result = FPS_CMD_OK;
            }
            else if (strcmp(FPSarg1, "norun") == 0)
            {
                state->tasklist[cmdindex].flag |= MILKSEQ_TASKFLAG_WAITFPS_NORUN;
                *taskstatus |= FPSTASK_STATUS_RUNNING;
                result = FPS_CMD_OK;
            }
            else
            {
                functionparameter_outlog("ERROR", "wait_fps: invalid condition %s", FPSarg1);
                result = FPS_CMD_FAIL;
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
